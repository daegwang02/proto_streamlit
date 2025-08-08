# app.py (최종 수정 완료 버전)

import streamlit as st
import os
import pandas as pd
import numpy as np
import json
import re
from openai import OpenAI
import ast
import lightgbm as lgb
import logging
import time

# --- 0. Streamlit 페이지 설정 및 로깅 ---
st.set_page_config(page_title="AlphaAgent", page_icon="🤖", layout="wide")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 1. OpenAI API 키 설정 ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    logging.info("OpenAI API 키를 Streamlit secrets에서 로드했습니다.")
except (KeyError, FileNotFoundError):
    logging.warning("Streamlit secrets에 OPENAI_API_KEY가 없습니다. 로컬 테스트를 위해 입력을 받습니다.")
    openai_api_key = st.text_input("OpenAI API Key를 입력하세요.", type="password", key="api_key_input")
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key)
    else:
        st.info("AI 기능을 사용하려면 좌측 사이드바에서 OpenAI API 키를 입력해주세요.")
        st.stop()

# --- 2. 핵심 로직 함수 및 클래스 (수정 없음) ---
# 이 부분은 사용자님의 원본 코드를 그대로 유지합니다.

# ===================================================================================
# [수정된 부분 1/2] 데이터 로딩 함수를 훨씬 더 간단하고 효율적으로 변경했습니다.
# ===================================================================================
@st.cache_data(ttl=3600) # 데이터 로딩 결과를 1시간 동안 캐싱
def load_pivoted_data(file_path: str):
    """
    프로젝트 폴더 내에 있는 단일 데이터 파일(CSV)을 읽어 피벗 데이터로 변환합니다.
    """
    logging.info(f"데이터 파일 로딩 시작: {file_path}")
    try:
        # GitHub에 함께 올린 CSV 파일을 직접 읽습니다.
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"데이터 파일 '{file_path}'을(를) 찾을 수 없습니다. app.py와 같은 폴더에 파일이 있는지, GitHub에 함께 업로드했는지 확인해주세요.")
        return None

    # 데이터 전처리 (파일에 '날짜' 컬럼이 없는 경우를 대비하여 이름 변경)
    if '날짜' in df.columns:
        df = df.rename(columns={'날짜': 'date'})
        
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date', 'symbol']).reset_index(drop=True)
    
    pivoted_data = {}
    # 'open', 'high', 'low', 'close', 'volume' 컬럼이 있는지 확인하고 피벗
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            pivoted = df.pivot(index='date', columns='symbol', values=col)
            pivoted_data[col] = pivoted.ffill().bfill() # 결측치 처리
    
    if 'close' not in pivoted_data:
        st.error("피벗 데이터 생성에 실패했습니다. 원본 데이터에 'close'와 'symbol' 컬럼이 있는지 확인해주세요.")
        return None
        
    logging.info(f"📊 총 {len(pivoted_data['close'].columns)}개 종목의 피벗 데이터 로딩 완료")
    return pivoted_data

# (OPERATORS, execute_expression, prepare_base_features, AlphaZoo, QualityGate 등 모든 핵심 함수/클래스 그대로 붙여넣기)
OPERATORS = {'ts_mean': lambda df, window: df.rolling(window, min_periods=max(1, window//2)).mean(),'ts_std': lambda df, window: df.rolling(window, min_periods=max(1, window//2)).std(),'ts_rank': lambda df, window: df.rolling(window, min_periods=max(1, window//2)).rank(pct=True),'delay': lambda df, period: df.shift(period),'delta': lambda df, period: df.diff(period),'rank': lambda df: df.rank(axis=1, pct=True),'scale': lambda df: df.div(df.abs().sum(axis=1), axis=0),'add': lambda a, b: a + b,'subtract': lambda a, b: a - b,'multiply': lambda a, b: a * b,'divide': lambda a, b: a / b.replace(0, np.nan),'negate': lambda a: -a,'abs': lambda a: a.abs()}
def execute_expression(expression: str, data: dict):
    local_data = {k: v.copy() for k, v in data.items()}
    while '(' in expression:
        match = re.search(r"(\w+)\(([^()]+)\)", expression)
        if not match:
            if expression in local_data: return local_data[expression]
            raise ValueError(f"잘못된 수식 형식: {expression}")
        op_name, args_str = match.groups()
        args = [arg.strip() for arg in args_str.split(',')]
        evaluated_args = []
        for arg in args:
            if arg.isdigit(): evaluated_args.append(int(arg))
            elif arg in local_data: evaluated_args.append(local_data[arg])
            else: raise ValueError(f"알 수 없는 인자 '{arg}' (수식: {expression})")
        if op_name in OPERATORS:
            temp_var_name = f"temp_{abs(hash(match.group(0)))}"
            local_data[temp_var_name] = OPERATORS[op_name](*evaluated_args)
            expression = expression.replace(match.group(0), temp_var_name, 1)
        else: raise ValueError(f"알 수 없는 연산자: {op_name}")
    if expression in local_data: return local_data[expression]
    else: raise ValueError("최종 결과를 찾을 수 없습니다.")

def prepare_base_features(pivoted_data: dict) -> dict:
    logging.info("... 기본 팩터(Base Features) 생성 중 ...")
    data_copy = {k: v.copy() for k, v in pivoted_data.items()}
    pivoted_data['base_1'] = execute_expression("divide(subtract(close, open), open)", data_copy)
    pivoted_data['base_2'] = execute_expression("subtract(divide(close, delay(close, 1)), 1)", data_copy)
    pivoted_data['base_3'] = execute_expression("divide(volume, ts_mean(volume, 20))", data_copy)
    pivoted_data['base_4'] = execute_expression("divide(subtract(high, low), close)", data_copy)
    logging.info("✅ 기본 팩터 4개 생성 완료.")
    return pivoted_data
    
class AlphaZoo:
    def __init__(self): self.known_factors = {"ts_mean(close, 20)", "ts_std(close, 20)", "rank(volume)"}
    def add_factor(self, expression: str): self.known_factors.add(expression)
    def get_all_factors(self) -> set: return self.known_factors

class QualityGate:
    def __init__(self, alpha_zoo: AlphaZoo, client: OpenAI):
        self.alpha_zoo, self.client = alpha_zoo, client
        self.COMPLEXITY_THRESHOLD, self.ORIGINALITY_THRESHOLD, self.ALIGNMENT_THRESHOLD = 15, 0.9, 0.6
    def _calculate_complexity(self, expression: str) -> int:
        try: return sum(1 for node in ast.walk(ast.parse(expression)) if isinstance(node, (ast.Call, ast.Num, ast.Constant)))
        except: return float('inf')
    def _calculate_originality(self, expression: str) -> float:
        new_ops, max_similarity = set(re.findall(r'(\w+)\(', expression)), 0
        for known_expr in self.alpha_zoo.get_all_factors():
            known_ops = set(re.findall(r'(\w+)\(', known_expr))
            if not new_ops and not known_ops: continue
            intersection, union = len(new_ops.intersection(known_ops)), len(new_ops.union(known_ops))
            similarity = intersection / union if union > 0 else 0
            if similarity > max_similarity: max_similarity = similarity
        return max_similarity
    def _check_alignment(self, hypothesis: str, factor_expression: str) -> float:
        prompt = f"다음 '가설'과 '팩터 수식'의 논리적 일치도를 0.0에서 1.0 사이의 점수로만 평가해줘.\n- 가설: \"{hypothesis}\"\n- 팩터 수식: \"{factor_expression}\""
        try:
            response = self.client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0.0)
            return float(response.choices[0].message.content.strip())
        except Exception as e:
            logging.error(f"정합성 평가 API 호출 오류: {e}")
            return 0.0
    def validate(self, hypothesis: str, factor: dict) -> (bool, str):
        expression = factor['expression']
        complexity = self._calculate_complexity(expression)
        if complexity > self.COMPLEXITY_THRESHOLD: return False, f"복잡도 초과 ({complexity}/{self.COMPLEXITY_THRESHOLD})"
        similarity = self._calculate_originality(expression)
        if similarity > self.ORIGINALITY_THRESHOLD: return False, f"독창성 부족 (유사도 {similarity:.2f})"
        alignment_score = self._check_alignment(hypothesis, expression)
        if alignment_score < self.ALIGNMENT_THRESHOLD: return False, f"정합성 부족 (점수 {alignment_score:.2f})"
        return True, "품질 검사 통과"

def generate_market_hypothesis(seed: str, feedback_history: list) -> str:
    system_prompt = "당신은 주식 시장을 분석하는 퀀트 분석가입니다. 주어진 테마나 이전의 성공/실패 경험을 바탕으로, 검증 가능한 새로운 정량적 투자 가설을 한 문장으로 생성해주세요."
    user_content = f"테마: {seed}"
    if feedback_history:
        recent_feedback = "\n".join(feedback_history[-3:])
        user_content += f"\n\n이전 시도에 대한 피드백입니다:\n{recent_feedback}\n\n이 피드백을 바탕으로 기존 아이디어를 개선하거나 완전히 새로운 방향의 가설을 제시해주세요."
    response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}], temperature=0.7)
    return response.choices[0].message.content.strip()

def generate_alpha_expression(hypothesis: str) -> dict:
    available_ops = ", ".join(OPERATORS.keys())
    system_prompt = f"너는 알파 팩터 생성 AI야. 반드시 아래 JSON 형식과 주어진 표준 연산자만을 사용해야 한다.\n{{ \"description\": \"...\", \"expression\": \"...\" }}\n---\n[사용 가능 연산자] {available_ops}\n[사용 가능 데이터] open, high, low, close, volume\n[규칙] 함수 형태로 작성. 예: rank(subtract(ts_mean(close, 20), ts_mean(close, 60)))\n---"
    response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"가설: {hypothesis}"}], temperature=0.2)
    raw_response = response.choices[0].message.content.strip()
    match = re.search(r'\{.*\}', raw_response, re.DOTALL)
    if match: return json.loads(match.group(0))
    else: raise ValueError(f"JSON 파싱 오류: {raw_response}")

def evaluate_factor_with_lgbm(new_factor_values: pd.DataFrame, data: dict):
    LOOKBACK_WINDOW = 252
    feature_names = ['base_1', 'base_2', 'base_3', 'base_4']
    features = {name: data[name] for name in feature_names}
    features['new_factor'] = new_factor_values
    feature_dfs = [df.stack().rename(name) for name, df in features.items()]
    master_df = pd.concat(feature_dfs, axis=1)
    target = data['close'].pct_change().shift(-1).stack().rename('target')
    master_df = master_df.join(target, how='inner').dropna()
    if master_df.empty: return {'success': False, 'error': '피처와 타겟 결합 후 데이터가 없습니다.'}
    all_predictions = []
    unique_dates = master_df.index.get_level_values('date').unique().sort_values()
    
    progress_bar = st.progress(0, text="모델 백테스팅 진행 중...")
    
    for i in range(LOOKBACK_WINDOW, len(unique_dates)):
        train_start_date, train_end_date, test_date = unique_dates[i - LOOKBACK_WINDOW], unique_dates[i - 1], unique_dates[i]
        train_slice, test_slice = master_df.loc[train_start_date:train_end_date], master_df.loc[test_date:test_date]
        if train_slice.empty or test_slice.empty: continue
        X_train, y_train = train_slice.drop('target', axis=1), train_slice['target']
        X_test, y_test = test_slice.drop('target', axis=1), test_slice['target']
        lgbm = lgb.LGBMRegressor(random_state=42, n_jobs=-1, n_estimators=100)
        lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(5, verbose=False)])
        predictions = pd.Series(lgbm.predict(X_test), index=X_test.index)
        all_predictions.append(predictions)
        
        progress_bar.progress((i - LOOKBACK_WINDOW + 1) / (len(unique_dates) - LOOKBACK_WINDOW), text=f"모델 백테스팅 진행 중... ({test_date.strftime('%Y-%m-%d')})")

    progress_bar.empty()
        
    if not all_predictions: return {'success': False, 'error': '롤링 예측 결과가 없습니다.'}
    alpha_scores = pd.concat(all_predictions).unstack()
    future_returns = data['close'].pct_change().shift(-1)
    aligned_scores, aligned_returns = alpha_scores.align(future_returns, join='inner')
    if aligned_scores.empty: return {'success': False, 'error': '모델 예측값과 수익률 정렬 실패.'}
    long_mask, short_mask = aligned_scores.rank(axis=1, pct=True) > 0.8, aligned_scores.rank(axis=1, pct=True) < 0.2
    long_returns, short_returns = aligned_returns[long_mask].mean(axis=1), aligned_returns[short_mask].mean(axis=1)
    strategy_returns = long_returns - short_returns
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
    annual_return = strategy_returns.mean() * 252
    cumulative_returns = (1 + strategy_returns).cumprod()
    mdd = (cumulative_returns.cummax() - cumulative_returns).max()
    return {'success': True, 'sharpe_ratio': sharpe_ratio, 'annual_return': annual_return, 'mdd': mdd, 'cumulative_returns': cumulative_returns}

def generate_seed_from_user_idea(user_input: str) -> str:
    system_prompt = """
    당신은 월스트리트의 최고 퀀트 분석가이자, 초보 투자자들이 아이디어를 구체화하도록 돕는 친절한 어드바이저입니다.
    사용자의 대화나 단편적인 아이디어에서 핵심을 파악하여, AlphaAgent가 사용할 수 있는 정량적 투자 '시드(seed) 가설'을 한 문장으로 정제해야 합니다.
    """
    response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}], temperature=0.1)
    return response.choices[0].message.content.strip()

# ===================================================================================
# [수정된 부분 2/2] 절대 경로 대신, 파일 이름을 직접 전달하도록 변경했습니다.
# ===================================================================================
# --- 3. 데이터 로딩 (앱 시작 시 한 번만 실행) ---
pivoted_data = load_pivoted_data("ohlcv_data.csv")
if pivoted_data:
    pivoted_data = prepare_base_features(pivoted_data)
else:
    # load_pivoted_data 함수 내부에서 이미 에러 메시지를 보여주므로 st.stop()만 호출
    st.stop()


# --- 4. Streamlit UI 구성 (수정 없음, 기존 코드와 동일) ---
st.title("🤖 AlphaAgent: 나만의 투자 전략 자동 탐색")
st.markdown("아이디어를 입력하면, AI가 자율적으로 분석하여 최적의 투자 팩터(Alpha Factor)를 찾아드립니다.")

if 'best_factor' not in st.session_state:
    st.session_state.best_factor = {'sharpe': -np.inf}
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

st.subheader("1. 아이디어 입력 및 설정")
with st.form("input_form"):
    user_idea = st.text_area("어떤 투자 전략을 찾아볼까요?", "조용하던 주식이 갑자기 확 튀는 현상", height=100)
    num_iterations = st.number_input("몇 번의 시도를 통해 전략을 개선할까요?", min_value=1, max_value=20, value=5)
    start_button = st.form_submit_button("✨ 자율 분석 시작!")

if start_button:
    st.session_state.analysis_done = True
    st.session_state.log_messages = []
    st.session_state.best_factor = {'sharpe': -np.inf}

    with st.status("1. AI가 아이디어를 전문가 수준의 가설로 다듬는 중...", expanded=True) as status:
        try:
            refined_seed = generate_seed_from_user_idea(user_idea)
            st.write(f"**정제된 가설:** *{refined_seed}*")
            status.update(label="✅ 아이디어 정제 완료!", state="complete")
        except Exception as e:
            st.error(f"아이디어 정제 실패: {e}")
            st.stop()
    
    log_container = st.container()
    feedback_history = []
    alpha_zoo = AlphaZoo()

    for i in range(1, num_iterations + 1):
        with st.status(f"2. 자율 분석 진행 중... [반복 {i}/{num_iterations}]", expanded=True) as status:
            try:
                hypothesis = generate_market_hypothesis(refined_seed, feedback_history)
                st.write(f"🧠 **생성된 가설:** {hypothesis}")
                
                factor = generate_alpha_expression(hypothesis)
                st.write(f"📝 **생성된 팩터:** `{factor['expression']}`")
                
                is_valid, reason = QualityGate(alpha_zoo, client).validate(hypothesis, factor)
                if not is_valid:
                    raise ValueError(f"품질 검사 실패: {reason}")
                
                factor_values = execute_expression(factor['expression'], pivoted_data)
                result = evaluate_factor_with_lgbm(factor_values, pivoted_data)
                
                if result['success']:
                    sharpe = result['sharpe_ratio']
                    feedback = f"반복 {i}: 팩터 '{factor['expression']}' -> Sharpe: {sharpe:.2f}."
                    feedback_history.append(feedback)
                    alpha_zoo.add_factor(factor['expression'])
                    
                    st.session_state.log_messages.append({
                        "iteration": i, "success": True, "hypothesis": hypothesis, 
                        "expression": factor['expression'], "sharpe": sharpe, "result": result
                    })

                    if sharpe > st.session_state.best_factor['sharpe']:
                        st.session_state.best_factor = {
                            'sharpe': sharpe, 'description': factor['description'], 
                            'expression': factor['expression'], 'result': result
                        }
                    
                    status.update(label=f"✅ 반복 {i} 성공! (Sharpe: {sharpe:.2f})", state="complete")
                else:
                    raise ValueError(f"모델 평가 실패: {result['error']}")

            except Exception as e:
                feedback = f"반복 {i}: 실패. ({str(e)})"
                feedback_history.append(feedback)
                st.session_state.log_messages.append({"iteration": i, "success": False, "error": str(e)})
                status.update(label=f"❌ 반복 {i} 실패", state="error")

    st.balloons()

if st.session_state.analysis_done:
    st.subheader("3. 분석 결과")

    best = st.session_state.best_factor
    if best['sharpe'] > -np.inf:
        st.success(f"🎉 **최고의 전략을 찾았습니다!** (Sharpe Ratio: {best['sharpe']:.3f})")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("연간 수익률", f"{best['result']['annual_return'] * 100:.2f} %")
            st.metric("최대 낙폭 (MDD)", f"{best['result']['mdd'] * 100:.2f} %")
        
        with col2:
            st.write("**📝 팩터 설명**")
            st.info(f"{best['description']}")
            st.write("**⚙️ 팩터 수식**")
            st.code(f"{best['expression']}", language="python")

        st.line_chart(best['result']['cumulative_returns'])

    else:
        st.error("유의미한 전략을 찾지 못했습니다. 아이디어를 바꿔 다시 시도해보세요.")

    with st.expander("🔍 전체 분석 과정 로그 보기"):
        for log in st.session_state.log_messages:
            if log['success']:
                st.markdown(f"--- \n**[성공] 반복 #{log['iteration']} | Sharpe: {log['sharpe']:.3f}**")
                st.text(f"가설: {log['hypothesis']}")
                st.code(f"수식: {log['expression']}", language="python")
            else:
                st.markdown(f"--- \n**[실패] 반복 #{log['iteration']}**")
                st.error(f"오류: {log['error']}")
