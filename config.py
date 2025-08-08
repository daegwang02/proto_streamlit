# config.py

"""
프로젝트의 주요 설정 변수들을 저장합니다.
API 키, 파일 경로, 주요 파라미터 등을 이곳에서 관리합니다.
"""

# LLM 클라이언트 설정
# 중요: 실제 API 키는 환경 변수나 다른 안전한 방식으로 관리해야 합니다.
# 예시: os.environ.get("GOOGLE_API_KEY")
GOOGLE_API_KEY = "실제로 발급받은 API 키" 

# 데이터 경로 설정
# 사용자의 깃허브에 저장된 PARQUET 파일의 Raw URL을 입력해야 합니다.
KOR_STOCK_DATA_URL = "https://raw.githubusercontent.com/daegwang02/proto_streamlit/main/ohlcv_data.parquet"

# 백테스팅 설정
TRANSACTION_FEE_BUY = 0.0005   # 한국 주식 시장 매수 수수료
TRANSACTION_FEE_SELL = 0.0015  # 한국 주식 시장 매도 수수료 + 세금



