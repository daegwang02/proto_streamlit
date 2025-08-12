# clients/llm_client.py (수정된 코드)

import openai
import json
import time
import re
from typing import Dict, Any, List
import os
import httpx # 💡 httpx 라이브러리 추가

class LLMClient:
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다. 환경 변수 'OPENAI_API_KEY'를 설정하거나 인자로 전달해주세요.")
        
        # 💡 httpx.Client를 사용하여 프록시 설정을 처리
        #    Streamlit 환경 변수에서 프록시 URL을 가져와 설정
        http_client = None
        proxy_url = os.getenv("http_proxy") or os.getenv("https_proxy")
        if proxy_url:
            print(f"프록시 설정 감지: {proxy_url}")
            http_client = httpx.Client(proxies=proxy_url)
            
        # 💡 openai.OpenAI 클라이언트를 초기화할 때 http_client 인자를 전달
        self.client = openai.OpenAI(api_key=api_key, http_client=http_client)
        self.model = "gpt-4o-mini"
        self.temperature = 0.2
        self.top_p = 1.0
        self.max_tokens = 4096
    
    # ... 나머지 코드는 변경 없음
    def _send_request(self, prompt: str, retries=5, delay=10) -> str:
        """
        주어진 프롬프트를 API에 전송하고, 재시도 로직을 포함하여 응답을 받습니다.
        """
        for i in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"LLM API 호출 중 오류 발생: {e}. {delay}초 후 재시도합니다... ({i+1}/{retries})")
                time.sleep(delay)
        raise RuntimeError("LLM API 호출에 최종적으로 실패했습니다.")
    

    def _parse_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """LLM 응답에서 JSON 코드 블록을 추출하고 파싱합니다."""
        match = re.search(r'```json\s*([\s\S]+?)\s*```', response_text)
        if not match:
            raise ValueError("응답에서 JSON 객체를 찾을 수 없습니다.")
        
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 파싱 오류: {e}\n원본 JSON 문자열: {json_str}")

    def generate_hypothesis(self, external_knowledge: str, existing_hypotheses: List[str], feedback_summary: str) -> Dict[str, Any]:
        """
        시장 가설을 생성합니다. (IdeaAgent가 사용)
        """
        prompt = f"""
        당신은 월스트리트의 저명한 퀀트 분석가입니다. 당신의 임무는 새로운 알파 팩터를 발굴하기 위한 창의적이고 논리적인 시장 가설을 수립하는 것입니다.

        다음은 최근 시장 분석 리포트와 전문가 의견입니다:
        --- [외부 지식] ---
        {external_knowledge}
        --- [외부 지식 끝] ---

        다음은 이미 탐색했던 가설들이니, 이것들과는 다른 새로운 관점의 가설을 제시해야 합니다:
        --- [기존 가설 목록] ---
        {', '.join(existing_hypotheses) if existing_hypotheses else '없음'}
        --- [기존 가설 목록 끝] ---

        # --------------------------------------------------
        # <<< 피드백 루프를 위해 신규 추가된 부분 >>>
        # --------------------------------------------------
        다음은 이전 라운드에서 발굴했던 팩터들의 성공 및 실패 사례에 대한 요약입니다. 
        이 피드백을 반드시 참고하여, 성공적인 팩터의 아이디어는 발전시키고 실패한 팩터의 함정은 피하는 방향으로 새로운 가설을 만들어야 합니다.
        --- [과거 평가 피드백] ---
        {feedback_summary}
        --- [과거 평가 피드백 끝] ---
        # --------------------------------------------------

        위 모든 정보를 종합적으로 고려하여, 다음 5가지 구성요소를 포함하는 새로운 시장 가설을 JSON 형식으로 제안해주십시오.
        1.  knowledge: 가설의 기반이 되는 금융 이론 또는 시장 원리.
        2.  market_observation: 가설을 뒷받침하는 실제 시장 관찰 현상.
        3.  justification: 관찰 현상이 이론과 어떻게 연결되어 투자 기회로 이어지는지에 대한 논리적 설명.
        4.  hypothesis: 포착하고자 하는 구체적인 시장 비효율성 또는 패턴에 대한 명확한 가설 서술.
        5.  specification: 가설을 팩터로 구현할 때 고려해야 할 구체적인 조건이나 파라미터(e.g., "최근 20일간의 거래량 평균", "주가 돌파 기준 5%").

        출력 형식은 반드시 다음 JSON 구조를 따라야 합니다:
        ```json
        {{
          "knowledge": "...",
          "market_observation": "...",
          "justification": "...",
          "hypothesis": "...",
          "specification": "..."
        }}
        ```
        """
        response_text = self._send_request(prompt)
        return self._parse_json_from_response(response_text)

    # def generate_factor_from_hypothesis(self,
    #                                     hypothesis: Dict[str, Any],
    #                                     function_rules: str,
    #                                     syntax_rules: str,
    #                                     existing_factors: List[str] = []) -> Dict[str, str]:
    def generate_factor_from_hypothesis(self, hypothesis: Dict[str, Any]) -> Dict[str, str]:
        """
        주어진 가설로부터 팩터 설명과 공식을 생성합니다. (FactorAgent가 사용)
        """

        prompt = f"""
        당신은 퀀트 개발자입니다. 주어진 시장 가설을 실행 가능한 알파 팩터로 변환하는 것이 당신의 역할입니다.


        --- [시장 가설] ---
        - 지식: {hypothesis.get('knowledge')}
        - 관찰: {hypothesis.get('market_observation')}
        - 논리: {hypothesis.get('justification')}
        - 가설: {hypothesis.get('hypothesis')}
        - 조건: {hypothesis.get('specification')}
        --- [시장 가설 끝] ---

        [허용된 연산자 및 변수 목록]
        **오직 다음 목록에 있는 연산자와 변수만 사용해야 합니다. 다른 형태나 기호는 사용할 수 없습니다.**

        **1. 횡단면 연산자 (단일 인자):**
        - `rank(series)`: 시리즈 값의 순위를 백분율로 반환
        - `scale(series)`: 시리즈 값을 -0.5 ~ 0.5 범위로 스케일링

        **2. 시계열 연산자 (시리즈 및 윈도우/기간 인자):**
        - `ts_mean(series, window)`: 특정 기간의 이동 평균
        - `ts_std(series, window)`: 특정 기간의 이동 표준 편차
        - `ts_rank(series, window)`: 특정 기간 내 시리즈 값의 순위 (백분율)
        - `delay(series, period)`: 이전 기간의 값 반환
        - `delta(series, period)`: 현재 값과 이전 기간 값의 차이
        - `ts_min(series, window)`: 특정 기간의 최소값
        - `ts_max(series, window)`: 특정 기간의 최대값
        - `correlation(series1, series2, window)`: 두 시리즈 간의 상관관계
        - `covariance(series1, series2, window)`: 두 시리즈 간의 공분산
        - `count(series, window)`: 특정 기간 내 유효한(NaN이 아닌) 값의 개수
        - `sum(series, window)`: 특정 기간의 합계
        - `median(series, window)`: 특정 기간의 중앙값
        - `skew(series, window)`: 특정 기간의 왜도
        - `kurt(series, window)`: 특정 기간의 첨도
        - `wma(series, window)`: 특정 기간의 가중 이동 평균

        **3. 산술 연산자 (함수 형태):**
        - `add(value1, value2)`: 두 값의 합 (`+` 기호 대신 사용)
        - `subtract(value1, value2)`: 두 값의 차이 (`-` 기호 대신 사용)
        - `multiply(value1, value2)`: 두 값의 곱 (`*` 기호 대신 사용)
        - `divide(value1, value2)`: 두 값의 나눗셈 (`/` 기호 대신 사용)
        - `power(base, exponent)`: 거듭제곱 (`**` 기호 대신 사용)

        **4. 단항 연산자 (단일 인자):**
        - `negate(value)`: 음수 값으로 변환
        - `abs(value)`: 절대값
        - `log(value)`: 자연로그
        - `sign(value)`: 값의 부호 (-1, 0, 1)

        **5. 논리 연산자 (불리언 인자):**
        - `and(condition1, condition2, ...)`: 모든 조건이 True일 때 True
        - `or(condition1, condition2, ...)`: 하나라도 조건이 True일 때 True
        - `not(condition)`: 조건의 반대 (True -> False, False -> True)

        **6. 비교 연산자 (함수 형태):**
        - `gt(value1, value2)`: value1이 value2보다 클 때 True (`>` 기호 대신 사용)
        - `ge(value1, value2)`: value1이 value2보다 크거나 같을 때 True (`>=` 기호 대신 사용)
        - `lt(value1, value2)`: value1이 value2보다 작을 때 True (`<` 기호 대신 사용)
        - `le(value1, value2)`: value1이 value2보다 작거나 같을 때 True (`<=` 기호 대신 사용)
        - `eq(value1, value2)`: value1과 value2가 같을 때 True (`==` 기호 대신 사용)
        - `ne(value1, value2)`: value1과 value2가 다를 때 True (`!=` 기호 대신 사용)

        **7. 삼항 연산자:**
        - `if(condition, true_value, false_value)`: 조건이 True면 true_value, False면 false_value

        **8. 사용 가능한 데이터 변수:**
        - `volume`, `close`, `open`, `high`, `low`, `vwap`, `returns`, `adv(window)`:
            - `adv`는 반드시 윈도우 크기(예: 20)를 포함하여 `adv20`과 같은 형태로 사용해야 합니다.

        ---

        [팩터 공식 생성 규칙]
        1. **함수 형태 사용 엄수**: 위 목록에 제시된 **모든 연산자는 반드시 함수 형태로만 사용해야 합니다.** 특히 산술(`+`, `-`, `*`, `/`) 및 비교(`>`, `<`, `==`, `>=`, `<=`, `!=`) 기호는 절대 사용해서는 안 됩니다. 해당 함수명(예: `multiply`, `gt`)을 사용하세요.
        2. **단순성**: 팩터 공식은 가설을 명확히 반영하면서도 최대한 간결하게 작성해야 합니다. 불필요한 중첩이나 복잡한 구조는 피하세요.
        3. **함수 호출 구문**: 모든 함수 호출은 `함수명(인수1, 인수2, ...)` 형식을 엄격하게 준수해야 합니다.
        4. **논리 연산자 인자**: `and()`와 `or()` 함수는 **오직 불리언 값(예: `gt()` 함수의 결과)만을 인자로 받아야 합니다.** `if()` 함수를 `and()` 또는 `or()`의 직접적인 인자로 사용하지 마세요.
        5. **복합 조건 처리**: `if()` 함수 내에서 여러 조건을 결합할 때는 `and(조건1, 조건2)` 또는 `or(조건1, 조건2)`를 `if`의 첫 번째 인자로 사용하세요.
        6. **`if` 함수의 반환 값 제한**: `if(condition, true_value, false_value)`에서 `true_value`와 `false_value`는 **반드시 숫자(1, 0) 또는 단순한 변수(예: close, volume)여야 합니다.** 이 자리에 다른 함수 호출(예: `gt()`)이나 복잡한 수식을 사용하지 마세요.
        7. **인덱싱 금지**: `close[5]`와 같은 직접적인 배열 인덱싱은 사용하지 마세요. 대신 `delay(close, 5)`와 같은 적절한 시계열 함수를 사용하세요.
        8. **새로운 연산자 생성 금지**: 위에 정의된 연산자 목록 외의 새로운 연산자나 함수를 만들어내지 마세요.


        이 가설을 구현하기 위한 팩터의 '설명(description)'과 '공식(formula)'을 생성해주십시오.
        - '설명'은 이 팩터가 무엇을 측정하고 어떤 논리로 작동하는지 자연어로 명확하게 서술해야 합니다.
        - '공식'은 Operator Library(e.g., rank, correlation, ts_min, high, low, close, volume)의 함수들을 사용하여 작성해야 합니다.
        
        출력 형식은 반드시 다음 JSON 구조를 따라야 합니다:

        ```json

        {{
            "description": "...",
            "formula": "..."
        }}

        ```
        """
        response_text = self._send_request(prompt)
        return self._parse_json_from_response(response_text)
    
    def score_hypothesis_alignment(self, hypothesis: str, factor_description: str) -> Dict[str, Any]:
        """가설과 팩터 설명 간의 일치도를 평가합니다. (c1(h,d))"""
        prompt = f"""
        당신은 퀀트 리서치 팀장입니다. 가설과 이를 구현한 팩터 설명 간의 논리적 일관성을 평가해야 합니다.

        - [가설]: {hypothesis}
        - [팩터 설명]: {factor_description}
        
        '팩터 설명'이 '가설'을 얼마나 잘 구현하고 있는지 0.0에서 1.0 사이의 점수로 평가하고, 그 이유를 간략히 서술해주십시오.
        - 1.0: 가설의 핵심 아이디어를 완벽하게 반영함.
        - 0.5: 일부 관련성은 있으나, 가설의 핵심을 놓치거나 왜곡함.
        - 0.0: 가설과 전혀 관련 없음.
        
        출력은 반드시 다음 JSON 형식을 따라야 합니다:
        ```json
        {{
            "score": 0.8,
            "justification": "팩터 설명이 가설의 ... 측면은 잘 반영했지만, ... 부분은 고려하지 않아 감점됨."
        }}
        ```
        """
        response_text = self._send_request(prompt)
        return self._parse_json_from_response(response_text)

    def score_description_alignment(self, factor_description: str, factor_formula: str) -> Dict[str, Any]:
        """팩터 설명과 공식 간의 일치도를 평가합니다. (c2(d,f))"""
        prompt = f"""
        당신은 퀀트 코드 리뷰어입니다. 팩터 설명과 실제 구현 공식이 일치하는지 검토해야 합니다.

        - [팩터 설명]: {factor_description}
        - [팩터 공식]: {factor_formula}

        '팩터 공식'이 '팩터 설명'에 서술된 로직을 얼마나 정확하게 수학적으로 구현했는지 0.0에서 1.0 사이의 점수로 평가하고, 그 이유를 서술해주십시오.
        
        출력은 반드시 다음 JSON 형식을 따라야 합니다:
        ```json
        {{
            "score": 0.9,
            "justification": "설명의 대부분의 로직이 공식에 정확히 구현되었으나, '최근'이라는 표현이 상수로 고정된 점이 아쉬움."
        }}
        ```
        """
        response_text = self._send_request(prompt)
        return self._parse_json_from_response(response_text)

    def generate_investment_advice(self, factor_info: Dict[str, Any]) -> str:
        """최종 선정된 팩터를 기반으로 투자 조언 리포트를 생성합니다. (AdvisoryAgent가 사용)"""
        prompt = f"""
        당신은 개인 투자자를 위한 투자 자문가입니다. 복잡한 퀀트 모델의 결과를 이해하기 쉬운 투자 조언으로 변환하는 것이 당신의 임무입니다.
        최근 발굴된 우수한 알파 팩터 정보를 바탕으로, 아래 목차에 따라 '투자 조언 리포트'를 작성해주십시오.

        --- [최종 알파 팩터 정보] ---
        - 팩터 공식: {factor_info.get('formula')}
        - 팩터 설명: {factor_info.get('description')}
        - 연평균수익률(AR): {factor_info.get('ar'):.2%}
        - 정보비율(IR): {factor_info.get('ir'):.2f}
        - 최대낙폭(MDD): {factor_info.get('mdd'):.2%}
        --- [정보 끝] ---

        # 투자 조언 리포트

        <한 눈에 보는 투자 전략>
        
        ## 1. 신규 팩터 "{factor_info.get('name', 'X')}"의 정의
        (새롭게 발굴한 알파 팩터에 대한 간결한 정의와 이 팩터가 포착하는 시장 비효율성 또는 투자 기회에 대한 핵심 설명을 작성하세요.)

        ## 2. 투자 전략 개요
        ("{factor_info.get('name', 'X')}" 팩터를 활용한 투자 전략의 핵심 컨셉 및 목표 수익률, 위험 수준 요약을 작성하세요. 제공된 성과 지표를 참고하세요.)

        ## 3. 핵심 투자 제안
        (본 리포트가 투자자에게 제안하는 구체적인 행동 지침(Actionable Advice)을 요약하여 2-3가지 항목으로 작성하세요.)
        """
        return self._send_request(prompt)





































