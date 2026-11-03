# -*- coding: utf-8 -*-
"""
2026 US Market Allocation Model
- Yahoo Finance market data via yfinance
- Optional FRED macro series via API key
- ECM-inspired relative-value feature + RandomForest direction model
- Walk-forward backtest, weekly rebalance, SPY / QQQ / cash allocation
Research/education tool. Not investment advice.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

st.set_page_config(page_title="US Market Model 2026", page_icon="📈", layout="wide")

TICKERS = {
    "SPY": "SPY", "QQQ": "QQQ", "VIX": "^VIX", "TNX": "^TNX",
    "USD": "DX-Y.NYB", "HYG": "HYG", "LQD": "LQD", "RSP": "RSP",
    "SOXX": "SOXX", "TLT": "TLT"
}
FEATURE_NAMES = [
    "spy_ret_1w","spy_ret_4w","spy_ret_13w","spy_ret_26w",
    "qqq_ret_1w","qqq_ret_4w","qqq_ret_13w","qqq_ret_26w",
    "spy_vol_13w","qqq_vol_13w","spy_ma20_gap","spy_ma40_gap",
    "qqq_ma20_gap","qqq_ma40_gap","spy_drawdown_52w","qqq_drawdown_52w",
    "qqq_spy_ratio_z","vix_level","vix_change_4w","tnx_change_4w",
    "usd_ret_4w","credit_ratio_ret_4w","breadth_ratio_ret_4w",
    "semi_ratio_ret_4w","tlt_ret_4w","fred_10y_change_4w",
    "fred_fedfunds_level","fred_unrate_change_13w","fred_cpi_yoy",
    "fred_hy_spread_change_4w"
]

def yf_download(period: str = "10y") -> tuple[pd.DataFrame, str]:
    raw = yf.download(
        list(TICKERS.values()), period=period, interval="1d",
        auto_adjust=True, progress=False, group_by="column",
        threads=True, timeout=25
    )
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance에서 가격 데이터를 받지 못했습니다.")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        else:
            close = raw.xs("Close", axis=1, level=1)
    else:
        close = raw[["Close"]].rename(columns={"Close": "SPY"})
    # Map ticker symbols to stable short names
    reverse = {v: k for k, v in TICKERS.items()}
    close = close.rename(columns=reverse)
    close = close.sort_index().ffill()
    required = ["SPY", "QQQ"]
    missing = [x for x in required if x not in close.columns or close[x].dropna().empty]
    if missing:
        raise RuntimeError("필수 가격 데이터 누락: " + ", ".join(missing))
    # Weekly Friday close; data are market-session closes, not guaranteed real-time quotes.
    weekly = close.resample("W-FRI").last().dropna(subset=["SPY", "QQQ"])
    last_daily = pd.Timestamp(close.dropna(how="all").index[-1])
    # Avoid labelling an unfinished current week with a future Friday date.
    if len(weekly) and last_daily.normalize() < pd.Timestamp(weekly.index[-1]).normalize():
        weekly.index = list(weekly.index[:-1]) + [last_daily]
    asof = last_daily.strftime("%Y-%m-%d")
    return weekly, asof

def build_features(pxw: pd.DataFrame) -> pd.DataFrame:
    p = pxw.copy()
    r = p.pct_change()
    x = pd.DataFrame(index=p.index)
    for ticker in ["SPY", "QQQ"]:
        for n in [1, 4, 13, 26]:
            x[f"{ticker.lower()}_ret_{n}w"] = p[ticker].pct_change(n)
        x[f"{ticker.lower()}_vol_13w"] = r[ticker].rolling(13).std() * np.sqrt(52)
        for n in [20, 40]:
            x[f"{ticker.lower()}_ma{n}_gap"] = p[ticker] / p[ticker].rolling(n).mean() - 1
        x[f"{ticker.lower()}_drawdown_52w"] = p[ticker] / p[ticker].rolling(52).max() - 1

    # ECM-inspired equilibrium deviation: QQQ/SPY relative-price ratio vs its 52-week equilibrium.
    # This is a rolling z-score proxy, not a formal cointegration/ECM estimation.
    ratio = np.log(p["QQQ"] / p["SPY"])
    ratio_mean = ratio.rolling(52).mean()
    ratio_std = ratio.rolling(52).std()
    x["qqq_spy_ratio_z"] = (ratio - ratio_mean) / ratio_std.replace(0, np.nan)

    if "VIX" in p:
        x["vix_level"] = p["VIX"]
        x["vix_change_4w"] = p["VIX"].pct_change(4)
    else:
        x["vix_level"] = np.nan
        x["vix_change_4w"] = np.nan
    x["tnx_change_4w"] = p["TNX"].diff(4) if "TNX" in p else np.nan
    x["usd_ret_4w"] = p["USD"].pct_change(4) if "USD" in p else np.nan
    x["credit_ratio_ret_4w"] = (p["HYG"] / p["LQD"]).pct_change(4) if {"HYG","LQD"}.issubset(p.columns) else np.nan
    x["breadth_ratio_ret_4w"] = (p["RSP"] / p["SPY"]).pct_change(4) if {"RSP","SPY"}.issubset(p.columns) else np.nan
    x["semi_ratio_ret_4w"] = (p["SOXX"] / p["SPY"]).pct_change(4) if {"SOXX","SPY"}.issubset(p.columns) else np.nan
    x["tlt_ret_4w"] = p["TLT"].pct_change(4) if "TLT" in p else np.nan
    # Optional FRED macro features. Monthly series are delayed by four weekly bars
    # to reduce release-date look-ahead; daily yields/spreads are used as observed.
    x["fred_10y_change_4w"] = p["FRED_DGS10"].diff(4) if "FRED_DGS10" in p else np.nan
    x["fred_fedfunds_level"] = p["FRED_FEDFUNDS"] if "FRED_FEDFUNDS" in p else np.nan
    x["fred_unrate_change_13w"] = p["FRED_UNRATE"].diff(13) if "FRED_UNRATE" in p else np.nan
    x["fred_cpi_yoy"] = p["FRED_CPIAUCSL"].pct_change(52) if "FRED_CPIAUCSL" in p else np.nan
    x["fred_hy_spread_change_4w"] = p["FRED_BAMLH0A0HYM2"].diff(4) if "FRED_BAMLH0A0HYM2" in p else np.nan
    return x.replace([np.inf, -np.inf], np.nan)

def make_model(seed: int, trees: int):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("rf", RandomForestClassifier(
            n_estimators=trees, max_depth=4, min_samples_leaf=5,
            max_features=0.8, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1
        ))
    ])

def target_frame(pxw: pd.DataFrame, x: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = x.copy()
    for asset in ["SPY", "QQQ"]:
        out[f"y_{asset}"] = (pxw[asset].shift(-horizon) / pxw[asset] - 1 > 0).astype(float)
        # Last horizon rows have no known outcome and must not enter training.
        out.loc[out.index[-horizon:], f"y_{asset}"] = np.nan
    return out

def probability(model, xrow):
    return float(model.predict_proba(xrow)[0, 1])

def allocation(p_spy: float, p_qqq: float, row: pd.Series,
               risk_mode: str = "Balanced") -> tuple[float,float,float]:
    # Direction probabilities are translated into a score, then risk controls cap exposure.
    s = max(0.0, (p_spy - 0.48) / 0.22)
    q = max(0.0, (p_qqq - 0.48) / 0.22)
    spy_trend = float(row.get("spy_ma40_gap", 0) > 0)
    qqq_trend = float(row.get("qqq_ma40_gap", 0) > 0)
    s *= 1.0 if spy_trend else 0.45
    q *= 1.0 if qqq_trend else 0.35

    total_cap = 1.0
    vix = row.get("vix_level", np.nan)
    spy_gap = row.get("spy_ma40_gap", 0)
    if pd.notna(vix):
        if vix >= 40: total_cap = 0.20
        elif vix >= 32: total_cap = 0.35
        elif vix >= 25: total_cap = 0.65
    if pd.notna(spy_gap) and spy_gap < -0.08:
        total_cap = min(total_cap, 0.25)
    elif pd.notna(spy_gap) and spy_gap < 0:
        total_cap = min(total_cap, 0.60)

    if risk_mode == "Conservative":
        total_cap *= 0.70
    elif risk_mode == "Aggressive":
        total_cap = min(1.0, total_cap * 1.15)

    score = s + q
    if score <= 1e-9:
        return 0.0, 0.0, 1.0
    # Cap single-asset concentration, redistribute remaining equity budget.
    w_spy = total_cap * s / score
    w_qqq = total_cap * q / score
    if w_spy > 0.70:
        excess = w_spy - 0.70
        w_spy = 0.70
        w_qqq = min(total_cap - w_spy, w_qqq + excess)
    if w_qqq > 0.60:
        excess = w_qqq - 0.60
        w_qqq = 0.60
        w_spy = min(total_cap - w_qqq, w_spy + excess)
    cash = max(0.0, 1.0 - w_spy - w_qqq)
    return w_spy, w_qqq, cash

def walk_forward(pxw: pd.DataFrame, data: pd.DataFrame, horizon: int,
                 train_weeks: int, trees: int, risk_mode: str):
    idx = data.index
    first = max(150, min(train_weeks, max(150, len(data) - horizon - 80)))
    rows, preds = [], []
    for i in range(first, len(data) - horizon):
        # At week i, only labels whose full forward horizon has elapsed are known.
        train_end = i - horizon
        start = max(0, train_end - train_weeks)
        train = data.iloc[start:train_end].dropna(subset=["y_SPY","y_QQQ"])
        if len(train) < 80 or train["y_SPY"].nunique() < 2 or train["y_QQQ"].nunique() < 2:
            continue
        x_train = train[FEATURE_NAMES]
        x_now = data.iloc[[i]][FEATURE_NAMES]
        m1 = make_model(31, trees).fit(x_train, train["y_SPY"].astype(int))
        m2 = make_model(47, trees).fit(x_train, train["y_QQQ"].astype(int))
        ps, pq = probability(m1, x_now), probability(m2, x_now)
        ws, wq, wc = allocation(ps, pq, data.iloc[i], risk_mode)
        next_ret_s = pxw["SPY"].iloc[i+1] / pxw["SPY"].iloc[i] - 1
        next_ret_q = pxw["QQQ"].iloc[i+1] / pxw["QQQ"].iloc[i] - 1
        strat = ws*next_ret_s + wq*next_ret_q
        rows.append({
            "date": idx[i], "p_spy": ps, "p_qqq": pq,
            "w_spy": ws, "w_qqq": wq, "w_cash": wc,
            "strategy_ret": strat, "spy_ret": next_ret_s, "qqq_ret": next_ret_q,
            "label_spy": int(data["y_SPY"].iloc[i]) if pd.notna(data["y_SPY"].iloc[i]) else np.nan,
            "label_qqq": int(data["y_QQQ"].iloc[i]) if pd.notna(data["y_QQQ"].iloc[i]) else np.nan
        })
    result = pd.DataFrame(rows).set_index("date")
    if not result.empty:
        result["strategy_equity"] = (1 + result["strategy_ret"]).cumprod()
        result["spy_equity"] = (1 + result["spy_ret"]).cumprod()
        result["qqq_equity"] = (1 + result["qqq_ret"]).cumprod()
        result["mix_equity"] = (1 + 0.5*result["spy_ret"] + 0.5*result["qqq_ret"]).cumprod()
        turnover = result[["w_spy","w_qqq","w_cash"]].diff().abs().sum(axis=1).fillna(0)
        result["turnover"] = turnover
        result["strategy_ret_net"] = result["strategy_ret"] - turnover*0.001
        result["strategy_equity_net"] = (1 + result["strategy_ret_net"]).cumprod()
    return result

def stats(series: pd.Series, turnover=None):
    s = series.dropna()
    if len(s) < 2: return {}
    years = len(s)/52
    total = float((1+s).prod()-1)
    cagr = float((1+s).prod()**(1/years)-1) if years > 0 else np.nan
    vol = float(s.std()*np.sqrt(52))
    sharpe = float((s.mean()*52)/(s.std()*np.sqrt(52))) if s.std() > 0 else np.nan
    wealth = (1+s).cumprod()
    dd = wealth/wealth.cummax()-1
    return {"Total return":total,"CAGR":cagr,"Volatility":vol,"Sharpe (rf=0)":sharpe,"Max drawdown":float(dd.min())}

def fred_fetch(api_key: str, series_ids: list[str]) -> pd.DataFrame:
    import requests
    all_series = {}
    for sid in series_ids:
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {"series_id":sid,"api_key":api_key,"file_type":"json","observation_start":"2010-01-01"}
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        vals = resp.json().get("observations", [])
        s = pd.Series(
            {pd.to_datetime(v["date"]): float(v["value"]) for v in vals if v["value"] not in (".", "")},
            name=sid
        )
        all_series[sid] = s
    out = pd.DataFrame(all_series).sort_index()
    # Convert observations to weekly Friday values. Delay monthly series by ~4 weeks
    # because FRED observation dates are not the public release timestamps.
    weekly = out.resample("W-FRI").last().ffill()
    for sid in ["FEDFUNDS", "UNRATE", "CPIAUCSL"]:
        if sid in weekly:
            weekly[sid] = weekly[sid].shift(4)
    return weekly.rename(columns={c: f"FRED_{c}" for c in weekly.columns})

st.title("미국 주식 주간 배분 모델")
st.caption("2026형 실시간 데이터 파이프라인 · ECM-inspired 상대가치 + Random Forest · SPY / QQQ / 현금")

with st.sidebar:
    st.header("모델 설정")
    capital = st.number_input("투자 원금 (원)", min_value=10000, max_value=1000000000, value=1000000, step=100000)
    period = st.selectbox("가격 데이터 기간", ["5y","10y"], index=1)
    horizon = st.selectbox("예측 기간", [1,2,4], index=0, format_func=lambda v: f"{v}주")
    train_weeks = st.selectbox("학습 구간", [156,208,260], index=2, format_func=lambda v: f"{v}주")
    trees = st.select_slider("랜덤포레스트 트리 수", options=[80,120,180,250], value=120)
    risk_mode = st.selectbox("위험 성향", ["Conservative","Balanced","Aggressive"], index=1,
                             format_func={"Conservative":"보수형","Balanced":"균형형","Aggressive":"공격형"}.get)
    run = st.button("최신 데이터 수집 및 분석", type="primary", use_container_width=True)

if "result" not in st.session_state:
    st.session_state.result = None
if run or st.session_state.result is None:
    with st.spinner("시장 데이터를 수집하고 워크포워드 모델을 계산하고 있습니다..."):
        try:
            prices, asof = yf_download(period)
            fred_status = "시장 데이터만 사용"
            if fred_key.strip():
                try:
                    macro = fred_fetch(fred_key.strip(), ["DGS10", "FEDFUNDS", "UNRATE", "CPIAUCSL", "BAMLH0A0HYM2"])
                    prices = prices.join(macro, how="left").ffill()
                    fred_status = "Yahoo Finance + FRED 거시경제 데이터"
                except Exception as fe:
                    st.warning(f"FRED 수집 실패 — 시장 데이터만으로 계속합니다: {fe}")
            features = build_features(prices)
            data = target_frame(prices, features, horizon)
            result = walk_forward(prices, data, horizon, train_weeks, trees, risk_mode)
            # Train final models using all fully matured labels for current signal.
            train = data.dropna(subset=["y_SPY","y_QQQ"]).tail(train_weeks)
            if len(train) < 80:
                raise RuntimeError("학습 가능한 데이터가 부족합니다.")
            current_x = features.iloc[[-1]][FEATURE_NAMES]
            model_spy = make_model(31, trees).fit(train[FEATURE_NAMES], train["y_SPY"].astype(int))
            model_qqq = make_model(47, trees).fit(train[FEATURE_NAMES], train["y_QQQ"].astype(int))
            p_spy = probability(model_spy, current_x)
            p_qqq = probability(model_qqq, current_x)
            ws,wq,wc = allocation(p_spy,p_qqq,features.iloc[-1],risk_mode)
            st.session_state.result = {
                "prices":prices,"features":features,"data":data,"backtest":result,
                "asof":asof,"p_spy":p_spy,"p_qqq":p_qqq,
                "weights":(ws,wq,wc),"risk_mode":risk_mode,"fred_status":fred_status,
                "horizon":horizon,"period":period
            }
        except Exception as e:
            st.error(f"데이터 수집 또는 계산 실패: {e}")
            st.info("인터넷 연결, Yahoo Finance 응답 또는 라이브러리 설치 상태를 확인한 뒤 다시 실행하세요.")
            st.stop()

d = st.session_state.result
prices, features, bt = d["prices"], d["features"], d["backtest"]
ws,wq,wc = d["weights"]
st.caption(f"최근 확보된 거래일: {d['asof']} · 주간 종가 기준 · 데이터: {d['fred_status']} · 화면 실행 시 재수집")
if d["asof"] < datetime.now().strftime("%Y-%m-%d"):
    st.warning("최신 확보 데이터가 오늘보다 이전입니다. 휴장일·시차·제공자 지연으로 당일 실시간 호가와 다를 수 있습니다.")

st.subheader("현재 모델 신호")
c1,c2,c3,c4 = st.columns(4)
c1.metric("SPY 상승확률", f"{d['p_spy']:.1%}")
c2.metric("QQQ 상승확률", f"{d['p_qqq']:.1%}")
c3.metric("주식 투자 비중", f"{ws+wq:.1%}")
c4.metric("현금 비중", f"{wc:.1%}")

st.markdown("#### 100만 원 기준 배분")
alloc_df = pd.DataFrame({
    "자산":["S&P500 (SPY)","나스닥100 (QQQ)","현금"],
    "비중":[ws,wq,wc],
    "금액(원)":[round(capital*ws/1000)*1000,round(capital*wq/1000)*1000,round(capital*wc/1000)*1000]
})
st.dataframe(alloc_df.style.format({"비중":"{:.1%}","금액(원)":"{:,.0f}"}), use_container_width=True, hide_index=True)
fig = px.pie(alloc_df, values="비중", names="자산", hole=0.45)
st.plotly_chart(fig, use_container_width=True)

st.subheader("시장 상태")
last = features.iloc[-1]
m1,m2,m3,m4 = st.columns(4)
m1.metric("SPY 40주선 괴리", f"{last['spy_ma40_gap']:.1%}" if pd.notna(last["spy_ma40_gap"]) else "N/A")
m2.metric("QQQ 40주선 괴리", f"{last['qqq_ma40_gap']:.1%}" if pd.notna(last["qqq_ma40_gap"]) else "N/A")
m3.metric("VIX", f"{last['vix_level']:.1f}" if pd.notna(last["vix_level"]) else "N/A")
m4.metric("QQQ/SPY 상대가치 Z", f"{last['qqq_spy_ratio_z']:.2f}" if pd.notna(last["qqq_spy_ratio_z"]) else "N/A")

st.subheader("워크포워드 백테스트")
if bt.empty:
    st.warning("백테스트 표본이 부족합니다. 데이터 기간을 10년으로 늘리거나 학습 구간을 줄이세요.")
else:
    metrics = {
        "모델 전략(거래비용 차감)": stats(bt["strategy_ret_net"]),
        "SPY 보유": stats(bt["spy_ret"]),
        "QQQ 보유": stats(bt["qqq_ret"]),
        "SPY/QQQ 50:50": stats(0.5*bt["spy_ret"]+0.5*bt["qqq_ret"])
    }
    metric_df = pd.DataFrame(metrics).T
    st.dataframe(metric_df.style.format({
        "Total return":"{:.2%}","CAGR":"{:.2%}","Volatility":"{:.2%}",
        "Sharpe (rf=0)":"{:.2f}","Max drawdown":"{:.2%}"
    }), use_container_width=True)
    score_cols = st.columns(2)
    for col, asset, prob_col, label_col in [
        (score_cols[0], "SPY", "p_spy", "label_spy"),
        (score_cols[1], "QQQ", "p_qqq", "label_qqq")
    ]:
        valid = bt[[prob_col,label_col]].dropna()
        if len(valid) and valid[label_col].nunique() > 1:
            score = balanced_accuracy_score(valid[label_col].astype(int), (valid[prob_col] >= 0.5).astype(int))
            col.metric(f"{asset} 방향성 Balanced Accuracy", f"{score:.1%}", help="워크포워드 예측과 이후 4주 실제 방향을 비교한 값")
    curve = bt[["strategy_equity_net","spy_equity","qqq_equity","mix_equity"]].rename(columns={
        "strategy_equity_net":"모델 전략(비용 차감)","spy_equity":"SPY 보유",
        "qqq_equity":"QQQ 보유","mix_equity":"50:50 보유"})
    st.plotly_chart(px.line(curve, labels={"value":"누적 자산(초기=1)","date":"주차","variable":"전략"},
                            title="주간 누적 성과"), use_container_width=True)
    st.caption("거래비용 가정: 포트폴리오 회전율 1회당 0.10% 차감. 세금·환전 스프레드·ETF 보수·실제 체결 슬리피지는 별도.")
    # Directional metrics only on matured labels in the simulated period.
    st.write(f"백테스트 구간: {bt.index.min().date()} ~ {bt.index.max().date()} · 관측 주: {len(bt)}")
    st.dataframe(bt.tail(12).style.format({
        "p_spy":"{:.1%}","p_qqq":"{:.1%}","w_spy":"{:.1%}","w_qqq":"{:.1%}",
        "w_cash":"{:.1%}","strategy_ret_net":"{:.2%}"}), use_container_width=True)

with st.expander("모델 설계 및 해석상 주의"):
    st.markdown("""
    - **예측 목표:** SPY와 QQQ 각각의 향후 설정 기간(기본 1주) 수익률이 양수인지 분류합니다.
    - **ECM 차용:** 로그(QQQ/SPY)의 52주 평균 대비 표준화 편차를 상대가치/균형 이탈 변수로 사용합니다. 정식 공적분 검정과 ECM 계수 추정은 하지 않으므로 논문 재현 모델이 아닙니다.
    - **머신러닝:** Random Forest 두 개를 별도로 학습하며, 매주 예측 시점 이전에 결과가 확정된 표본만 학습에 사용합니다.
    - **리스크 관리:** 40주 이동평균 추세, VIX 수준, 최대 비중 제한을 이용해 주식 총비중을 축소합니다.
    - **백테스트:** 주간 리밸런싱이며, 신호 생성 시점 이후의 다음 주 수익률로 평가합니다. 학습 레이블의 예측기간이 겹치지 않도록 학습 시점에 embargo를 둡니다.
    - **한계:** Yahoo Finance 데이터는 실시간 체결 호가가 아니며 제공자 정책·지연·누락의 영향을 받습니다. FRED 거시경제 원계열은 아직 모델 입력에 포함하지 않았습니다. 과거 데이터 수정, 생존편향, 모델 선택 편향이 남을 수 있습니다. 수익 보장은 없습니다.
    """)
    st.markdown("[Yahoo Finance 데이터 라이브러리 문서](https://ranaroussi.github.io/yfinance/)")
st.caption("이 도구는 연구·교육용 프로토타입입니다. 모델 신호만으로 매매하지 말고, 실제 주문 전 환율·수수료·세금·ETF 가격을 확인하세요.")
