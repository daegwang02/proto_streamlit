# agents/eval_agent.py (두 번째 버전)

from .base_agent import BaseAgent
from clients.database_client import DatabaseClient
from clients.backtester_client import BacktesterClient
from foundations.factor_structure import FactorParser

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
            formula = factor_record['formula']
            
            # 💡 formula를 재파싱하여 ast 객체 다시 생성
            try:
                parser = FactorParser()
                ast = parser.parse(formula)
            except Exception as e:
                print(f"  - ❌ AST 재파싱 실패: {e}")
                self.db_client.update_factor_status(factor_id, 'failed')
                continue # 다음 팩터로 넘어감
            
            print(f"\n[팩터 #{factor_id} 평가 중]: {formula}")
            self.db_client.update_factor_status(factor_id, 'evaluating')

            try:
                # 백테스터에 유효한 `ast` 객체 전달
                performance_metrics = self.backtester_client.run_full_backtest(formula, ast)
                
                # ✅ 평가 결과를 DB에 저장하는 핵심 로직
                eval_data = {'factor_id': factor_id, **performance_metrics}
                self.db_client.save_evaluation(eval_data)

                print(f"  - ✅ 평가 완료: IR {performance_metrics.get('IR'):.3f}, MDD {performance_metrics.get('MDD'):.3f}")

            except Exception as e:
                print(f"  - ❌ 평가 실패: {e}")
                self.db_client.update_factor_status(factor_id, 'failed')
        
        print("\n--- EvalAgent 실행 종료 ---\n")




