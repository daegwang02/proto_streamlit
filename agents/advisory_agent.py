# agents/advisory_agent.py

from .base_agent import BaseAgent
from clients.llm_client import LLMClient
from clients.database_client import DatabaseClient

class AdvisoryAgent(BaseAgent):
    """
    최종적으로 발굴된 최고의 알파 팩터를 기반으로 투자 조언 리포트를 생성하는 에이전트입니다.
    """
    def __init__(self, llm_client: LLMClient, db_client: DatabaseClient):
        self.llm_client = llm_client
        self.db_client = db_client

    def run(self):
        """
        투자 조언 리포트를 생성하고 출력합니다.
        """
        print("\n--- AdvisoryAgent 실행: 투자 조언 리포트 생성 시작 ---")
        
        # 1. DB에서 가장 성과가 좋은 팩터 정보를 가져옴 (IR 기준)
        best_factor_info = self.db_client.get_best_factor()
        
        if best_factor_info is None:
            print("AdvisoryAgent: 평가된 팩터가 없어 리포트를 생성할 수 없습니다.")
            print("--- AdvisoryAgent 실행 종료 ---\n")
            return

        # 2. 리포트 생성을 위해 LLM에 전달할 정보 정리
        # 팩터 이름을 추가하여 가독성 향상
        best_factor_info['name'] = f"AlphaFactor-{best_factor_info['id']}"
        
        # 3. LLM을 통해 투자 조언 리포트 생성
        print(f"최고 성과 팩터 #{best_factor_info['id']} 기반으로 리포트 생성 중...")
        investment_report = self.llm_client.generate_investment_advice(best_factor_info)
        
        # 4. 최종 리포트 출력
        print("\n" + "="*80)
        print("                 ✨ 최종 투자 조언 리포트 ✨")
        print("="*80)
        print(investment_report)
        print("="*80)
        print("\n--- AdvisoryAgent 실행 종료 ---\n")
