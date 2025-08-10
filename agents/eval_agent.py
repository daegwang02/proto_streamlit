# agents/eval_agent.py

from .base_agent import BaseAgent
from clients.database_client import DatabaseClient
from clients.backtester_client import BacktesterClient

class EvalAgent(BaseAgent):
    """
    새롭게 생성된 팩터들의 성과를 백테스팅하여 평가하는 에이전트입니다.
    """
    def __init__(self, db_client: DatabaseClient, backtester_client: BacktesterClient):
        self.db_client = db_client
        self.backtester_client = backtester_client

    def run(self):
        """
        데이터베이스에 있는 새로운 팩터들을 평가합니다.
        """
        print("\n--- EvalAgent 실행: 신규 팩터 평가 시작 ---")
        new_factors = self.db_client.get_new_factors()
        
        if not new_factors:
            print("EvalAgent: 평가할 새로운 팩터가 없습니다.")
            print("--- EvalAgent 실행 종료 ---\n")
            return

        for factor_record in new_factors:
            factor_id = factor_record['id']
            formula = factor_record.get('formula', 'N/A')
            
            # ✅ 모든 단계를 하나의 try 블록으로 감싸서 안정성을 높입니다.
            try:
                # ast 키가 있는지 먼저 확인하여 KeyError를 방지합니다.
                if 'ast' not in factor_record:
                    raise KeyError("팩터 정보에 'ast' 키가 누락되었습니다.")
                
                ast = factor_record['ast']
                
                print(f"\n[팩터 #{factor_id} 평가 중]: {formula}")
                self.db_client.update_factor_status(factor_id, 'evaluating')

                # 백테스터를 실행하여 성과 지표를 얻음
                performance_metrics = self.backtester_client.run_full_backtest(formula, ast)
                
                # 평가 결과를 DB에 저장
                eval_data = {'factor_id': factor_id, **performance_metrics}
                self.db_client.save_evaluation(eval_data)

                print(f"  - ✅ 평가 완료: IR {performance_metrics.get('IR'):.3f}, MDD {performance_metrics.get('MDD'):.3f}")

            except Exception as e:
                print(f"  - ❌ 평가 실패: {e}")
                self.db_client.update_factor_status(factor_id, 'failed')
        
        print("\n--- EvalAgent 실행 종료 ---\n")
