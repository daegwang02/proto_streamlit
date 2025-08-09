# agents/idea_agent.py
from .base_agent import BaseAgent
from clients.llm_client import LLMClient
from clients.database_client import DatabaseClient

class IdeaAgent(BaseAgent):
    """
    LLM을 활용하여 새로운 시장 가설을 생성하는 에이전트입니다.
    """
    def __init__(self, llm_client: LLMClient, db_client: DatabaseClient):
        self.llm_client = llm_client
        self.db_client = db_client

    def run(self, external_knowledge: str):
        """
        새로운 가설을 하나 생성하고 데이터베이스에 저장합니다.
        과거 평가 결과를 조회하여 LLM에 전달하는 로직 추가
        Args:
            external_knowledge (str): 가설 생성의 기반이 될 시장 리포트, 뉴스 등 외부 지식.
        """
        print("\n--- IdeaAgent 실행: 새로운 가설 생성 시작 ---")
        
        # 1. 중복 생성을 피하기 위해 기존 가설들을 가져옴
        existing_hypotheses = self.db_client.get_all_hypothesis_texts()
        
        # 2. 피드백 루프: 과거 평가 결과를 요약하여 가져옴
        feedback_summary = self.db_client.get_evaluation_summary()
        print("  - 과거 평가 피드백 로드 완료.")
        
        # 3. LLM을 통해 새로운 가설 생성 (피드백 정보 포함)
        new_hypothesis_data = self.llm_client.generate_hypothesis(
            external_knowledge=external_knowledge,
            existing_hypotheses=existing_hypotheses,
            feedback_summary=feedback_summary  # 수정된 부분
        )
        
        # 4. 생성된 가설을 데이터베이스에 저장
        hypothesis_id = self.db_client.save_hypothesis(new_hypothesis_data)
        
        print(f"✅ IdeaAgent: 새로운 가설 #{hypothesis_id} 생성 완료.")
        print(f"   가설: {new_hypothesis_data.get('hypothesis')}")
        print("--- IdeaAgent 실행 종료 ---\n")

