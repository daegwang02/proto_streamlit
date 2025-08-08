# main.py (Streamlit UI 통합 최종 버전)
import streamlit as st
import pandas as pd
import numpy as np

# --- 1. 최종 코드의 모든 모듈 및 클래스 import ---
from agents.idea_agent import IdeaAgent
from agents.factor_agent import FactorAgent
from agents.eval_agent import EvalAgent
from agents.advisory_agent import AdvisoryAgent
from clients.llm_client import LLMClient
from clients.database_client import DatabaseClient
from clients.backtester_client import BacktesterClient

# --- 2. 설정 파일 import ---
import config

# --- 3. Streamlit 페이지 설정 및 디자인 적용 ---
st.set_page_config(
    page_title="AlphaAgent: KB 금융 AI 투자 전략 탐색기",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사용자 정의 CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
    
    html, body, [class*="st-"] { font-family: 'Noto Sans KR', sans-serif; }
    .main-header h1 { color: #FFC107; text-align: center; font-size: 2.5rem; font-weight: 700; margin-bottom: 0; }
    .stButton>button { background-color: #FFC107; color: black; border-radius: 8px; font-weight: 700; border: none; }
    .stButton>button:hover { background-color: #E6B800; }
    .stMetric > div { background-color: #f7f7f7; padding: 1.5rem; border-radius: 8px; border: 1px solid #ddd; }
    .stMetric label { font-size: 1rem; color: #666; font-weight: normal; }
    .stMetric p { font-size: 1.5rem; font-weight: 700; margin-top: 0.5rem; }
    .report-header { color: #FFC107; font-weight: 700; border-bottom: 2px solid #FFC107; padding-bottom: 0.5rem; }
    .stCodeBlock pre { background-color: #f0f0f0; border-left: 5px solid #FFC107; }
    .streamlit-expander { border-left: 5px solid #FFC107; border-radius: 8px; }
    </style>
    <div class="main-header">
        <h1>🤖 AlphaAgent: KB 금융 AI 투자 전략 탐색기</h1>
    </div>
    <br>
    """, unsafe_allow_html=True)

# --- 4. 세션 상태 초기화 ---
if 'agents' not in st.session_state:
    st.session_state.agents = None
if 'db' not in st.session_state:
    st.session_state.db = None
if 'final_report' not in st.session_state:
    st.session_state.final_report = None
if 'best_factor_info' not in st.session_state:
    st.session_state.best_factor_info = None

# --- 5. 사이드바 (설정) ---
with st.sidebar:
    st.header("⚙️ 설정")
    st.info("API 키는 `config.py` 또는 `.streamlit/secrets.toml`에 설정되어야 합니다.")

    external_knowledge = st.text_area(
        "💡 AI에게 제공할 시장 분석 정보 (선택)",
        value=config.EXTERNAL_KNOWLEDGE,
        height=150
    )

    discovery_rounds = st.number_input(
        "🔄 알파 탐색 라운드 수",
        min_value=1,
        max_value=10,
        value=3
    )

    run_optimization = st.checkbox("🧠 하이퍼파라미터 최적화 실행 (추가 시간 소요)", value=False)

    start_button = st.button("✨ 분석 시작!")
    st.markdown("---")
    st.info("데이터 파일 URL: `config.py` 파일의 `KOR_STOCK_DATA_URL`")


# --- 6. 메인 화면 (분석 실행 및 결과 표시) ---
st.write("### AI 투자 아이디어 입력")
user_idea = st.text_area(
    "어떤 투자 아이디어를 탐색하고 싶으신가요?",
    "거래량이 급증하며 가격을 돌파하는 주식",
    height=80
)

# 분석 시작 버튼이 눌렸을 때 전체 워크플로우 실행
if start_button:
    # 세션 상태 초기화
    st.session_state.final_report = None
    st.session_state.best_factor_info = None

    try:
        # st.secrets에서 직접 GOOGLE_API_KEY를 가져옵니다.
        llm_client = LLMClient(api_key=st.secrets.GOOGLE_API_KEY)
        db_client = DatabaseClient()
        backtester_client = BacktesterClient(data_url=st.secrets.KOR_STOCK_DATA_URL) # 데이터 URL도 secrets에서 가져오도록 변경 가능

        # 에이전트 객체 생성
        st.session_state.agents = {
            'llm': llm_client,
            'db': db_client,
            'backtester': backtester_client,
            'idea': IdeaAgent(llm_client, db_client),
            'factor': FactorAgent(llm_client, db_client),
            'eval': EvalAgent(db_client, backtester_client),
            'advisory': AdvisoryAgent(llm_client, db_client)
        }
        st.session_state.db = db_client
    except Exception as e:
        st.error(f"초기화 오류: {e}. Secrets 설정이 올바른지 확인해주세요.")
        st.stop()


    # 2. 하이퍼파라미터 최적화 (선택 사항)
    if run_optimization:
        with st.spinner("🧠 하이퍼파라미터 최적화 중..."):
            optimizer = HyperparameterOptimizer(
                st.session_state.agents['idea'],
                st.session_state.agents['factor'],
                st.session_state.agents['eval'],
                external_knowledge
            )
            best_params = optimizer.optimize(init_points=3, n_iter=5)
            st.session_state.agents['factor'].max_complexity_sl = int(best_params['max_complexity_sl'])
            st.session_state.agents['factor'].max_complexity_pc = int(best_params['max_complexity_pc'])
            st.session_state.agents['factor'].max_similarity = best_params['max_similarity']
            st.session_state.agents['factor'].min_alignment = best_params['min_alignment']
            st.success("✅ 하이퍼파라미터 최적화 완료! 최적의 설정으로 분석을 시작합니다.")

    # 3. 알파 탐색 루프 (실시간 상태 표시)
    log_container = st.empty()
    all_logs = []
    
    with st.status("🚀 AlphaAgent 분석 시작...", expanded=True) as status:
        current_knowledge = user_idea + "\n\n" + external_knowledge
        
        for i in range(discovery_rounds):
            log_container.info(f"🔄 **라운드 {i+1}/{discovery_rounds} 시작**")
            status.update(label=f"🔄 라운드 {i+1}/{discovery_rounds} 진행 중...", state="running")
            
            try:
                with st.spinner("💡 가설 생성 중..."):
                    st.session_state.agents['idea'].run(current_knowledge)
                    all_logs.append("💡 가설 생성 완료.")

                with st.spinner("📝 팩터 생성 및 검증 중..."):
                    st.session_state.agents['factor'].run()
                    all_logs.append("📝 팩터 생성 및 검증 완료.")

                with st.spinner("📊 팩터 백테스팅 및 평가 중..."):
                    st.session_state.agents['eval'].run()
                    all_logs.append("📊 팩터 백테스팅 완료.")
                
                log_container.success(f"✅ **라운드 {i+1}/{discovery_rounds} 성공!**")
                
            except Exception as e:
                log_container.error(f"❌ **라운드 {i+1} 실패!** 오류: {e}")
                status.update(label=f"❌ 오류 발생! 라운드 {i+1} 중단", state="error")
                st.session_state.final_report = f"분석 중 오류 발생: {e}"
                st.session_state.best_factor_info = None
                break
        
        if st.session_state.final_report is None:
            status.update(label="📜 최종 투자 조언 리포트 생성 중...", state="running")
            best_factor_info = st.session_state.db.get_best_factor()
            if best_factor_info:
                st.session_state.best_factor_info = best_factor_info
                llm_client = st.session_state.agents['llm']
                st.session_state.final_report = llm_client.generate_investment_advice(best_factor_info)
                status.update(label="🎉 분석 완료! 최종 리포트가 생성되었습니다.", state="complete", expanded=False)
            else:
                st.error("분석을 통해 유의미한 팩터를 찾지 못했습니다.")
                status.update(label="분석 실패.", state="error")


# --- 7. 결과 화면 ---
if st.session_state.best_factor_info:
    best_factor = st.session_state.best_factor_info

    st.markdown("---")
    st.write("### 🥇 최고의 전략")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 연간 수익률 (AR)", f"{best_factor['ar']*100:.2f}%")
    with col2:
        st.metric("📉 최대 낙폭 (MDD)", f"{best_factor['mdd']*100:.2f}%")
    with col3:
        st.metric("📈 정보 비율 (IR)", f"{best_factor['ir']:.3f}")
    
    st.markdown("---")
    st.write("#### 📝 전략 상세 정보")
    with st.expander("AI가 찾은 최고의 팩터에 대한 설명 및 수식 보기"):
        st.write("##### 팩터 설명")
        st.info(f"{best_factor['description']}")
        st.write("##### 팩터 수식")
        st.code(best_factor['formula'], language='python')

    # 누적 수익률 그래프 시각화는 백테스터 코드를 수정해야 합니다.
    # 예시: st.line_chart(best_factor['cumulative_returns'])

    st.markdown("---")
    st.write("#### 🧠 AI 투자 조언")
    with st.expander("AlphaAgent가 제시하는 투자 조언 리포트 보기"):
        st.markdown(st.session_state.final_report)

# --- 8. 전체 로그 표시 (디버깅용) ---
with st.expander("🔍 전체 분석 과정 로그 보기"):
    st.write("이 섹션은 디버깅 및 전체 워크플로우를 확인하기 위해 사용됩니다.")
    st.dataframe(st.session_state.db.hypotheses if st.session_state.db else pd.DataFrame(), use_container_width=True)
    st.dataframe(st.session_state.db.factors if st.session_state.db else pd.DataFrame(), use_container_width=True)
    st.dataframe(st.session_state.db.evaluations if st.session_state.db else pd.DataFrame(), use_container_width=True)

