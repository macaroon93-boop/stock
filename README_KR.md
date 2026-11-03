# 미국 주식 주간 배분 모델 — 웹 배포판

이 폴더는 **Streamlit Community Cloud**에 올릴 수 있도록 구성된 웹 앱입니다. PC에 Python을 설치해 실행할 필요 없이, GitHub에 파일을 올린 뒤 Streamlit에서 배포하면 스마트폰·PC 브라우저로 접속할 수 있습니다.

## 웹 주소 만들기 (최초 1회)

1. [GitHub](https://github.com/)에 로그인하고 `New repository`를 누릅니다.
2. 저장소 이름을 `us-market-model-2026`으로 정합니다. 공개 저장소로 만들면 앱 주소도 공개됩니다.
3. `Add file` → `Upload files`를 눌러 이 압축파일 안의 **app.py, requirements.txt, .streamlit 폴더**를 업로드하고 `Commit changes`를 누릅니다.
4. [Streamlit Community Cloud](https://share.streamlit.io/)에 접속해 GitHub 계정으로 로그인합니다.
5. `Create app` → 방금 만든 저장소 선택 → Main file path에 `app.py` 입력 → `Deploy`를 누릅니다.
6. 배포가 끝나면 `https://저장소이름-임의문자.streamlit.app` 형태의 접속 주소가 만들어집니다.

공식 배포 안내: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy

## 앱 기능

- Yahoo Finance에서 SPY, QQQ, VIX, 미국 10년물 금리 지표, 달러, HYG/LQD, RSP/SPY, SOXX/SPY, TLT 데이터를 수집
- 주간 종가 기준 1주/2주/4주 방향성 분류
- Random Forest 2개 모델로 SPY와 QQQ 상승확률 계산
- QQQ/SPY 상대가치 Z-score(ECM 아이디어 차용), 추세·변동성·시장 폭·신용 스프레드 대체지표 결합
- 매주 워크포워드 백테스트 및 SPY·QQQ·50:50 보유 전략 비교
- 투자금액 입력 후 SPY·QQQ·현금 배분액 계산

## 기본값

- 투자 원금: 1,000,000원
- 예측 기간: 1주
- 학습 구간: 최근 260주
- 리밸런싱: 주간
- 거래비용 가정: 포트폴리오 회전율 1회당 0.10%

## 데이터와 주의사항

- Yahoo Finance 데이터는 거래소의 실시간 체결 호가가 아니라 제공자가 반환하는 최신 거래 데이터입니다. 지연·누락·수정 가능성이 있습니다.
- 현재 버전은 FRED의 거시경제 원계열을 아직 모델 입력에 넣지 않았습니다. 금리·달러·VIX 등 시장에서 거래되는 대체 지표를 사용합니다.
- QQQ/SPY 상대가치 Z-score는 ECM에서 영감을 받은 대체변수이며, 정식 공적분 검정 및 ECM 계수 추정은 아닙니다.
- 과거 백테스트는 미래 성과를 보장하지 않습니다. 세금, 환전 스프레드, ETF 보수, 실제 슬리피지와 데이터 편향이 완전히 반영되지 않습니다.
- 이 앱은 연구·교육용 프로토타입이며 투자 권유가 아닙니다.
