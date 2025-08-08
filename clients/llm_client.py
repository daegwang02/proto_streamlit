# clients/llm_client.py
# ... (상단 import 및 __init__ 등은 기존과 동일) ...

class LLMClient:
    def __init__(self, api_key: str):
        # ... (기존 코드)
        pass

    # ... (_send_request, _parse_json_from_response 등 다른 메서드들은 기존과 동일) ...

    def generate_hypothesis(self, external_knowledge: str, existing_hypotheses: List[str], feedback_summary: str) -> Dict[str, Any]:
        """
        시장 가설을 생성합니다. (IdeaAgent가 사용)
        [수정] feedback_summary 파라미터 추가 및 프롬프트 수정
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

    # ... (generate_factor_from_hypothesis 등 나머지 메서드는 기존과 동일) ...

