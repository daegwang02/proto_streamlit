# optimizer/hyperparameter_optimizer.py

from bayes_opt import BayesianOptimization
from agents.idea_agent import IdeaAgent
from agents.factor_agent import FactorAgent
from agents.eval_agent import EvalAgent

class HyperparameterOptimizer:
    """
    베이지안 최적화를 사용하여 FactorAgent의 핵심 하이퍼파라미터를 튜닝합니다.
    목표는 IR(정보비율)을 최대화하는 최적의 임계값을 찾는 것입니다.
    """
    def __init__(self, idea_agent: IdeaAgent, factor_agent: FactorAgent, eval_agent: EvalAgent, external_knowledge: str):
        self.idea_agent = idea_agent
        self.factor_agent = factor_agent
        self.eval_agent = eval_agent
        self.external_knowledge = external_knowledge

        # 탐색할 하이퍼파라미터의 범위(pbounds) 정의
        self.pbounds = {
            'max_complexity_sl': (15, 40),      # 상징적 길이 (int)
            'max_complexity_pc': (3, 8),        # 파라미터 개수 (int)
            'max_similarity': (0.7, 0.99),      # 최대 유사도
            'min_alignment': (0.5, 0.95)        # 최소 일치도
        }

    def _objective_function(self, max_complexity_sl, max_complexity_pc, max_similarity, min_alignment):
        """
        베이지안 최적화가 최대화하려는 목적 함수입니다.
        주어진 하이퍼파라미터로 한 사이클의 알파 탐색을 실행하고, 결과 팩터의 IR을 반환합니다.
        """
        # 정수형 파라미터 변환
        max_complexity_sl = int(max_complexity_sl)
        max_complexity_pc = int(max_complexity_pc)

        print(f"\n[최적화 시도] SL: {max_complexity_sl}, PC: {max_complexity_pc}, Sim: {max_similarity:.2f}, Align: {min_alignment:.2f}")

        # 1. FactorAgent에 현재 시도할 하이퍼파라미터 설정
        self.factor_agent.max_complexity_sl = max_complexity_sl
        self.factor_agent.max_complexity_pc = max_complexity_pc
        self.factor_agent.max_similarity = max_similarity
        self.factor_agent.min_alignment = min_alignment

        try:
            # 2. 한 사이클 실행 (가설 생성 -> 팩터 생성 -> 평가)
            # 새 가설이 필요하므로 IdeaAgent 실행
            self.idea_agent.run(self.external_knowledge)
            # 생성된 가설로 FactorAgent 실행
            self.factor_agent.run()
            
            # 새로 생성된 팩터가 있는지 확인
            new_factors = self.factor_agent.db_client.get_new_factors()
            if not new_factors: # 검증을 통과한 팩터가 없는 경우
                print("  -> 결과: 유효 팩터 생성 실패, 낮은 점수(-1.0) 반환")
                return -1.0
            
            # 생성된 팩터로 EvalAgent 실행
            self.eval_agent.run()
            
            # 3. 결과 확인 및 IR 반환
            last_eval = self.eval_agent.db_client.evaluations.iloc[-1]
            ir_score = last_eval['IR']
            
            # IR이 NaN이거나 비정상적인 경우 처리
            if pd.isna(ir_score) or not np.isfinite(ir_score):
                print(f"  -> 결과: IR이 유효하지 않음 ({ir_score}), 낮은 점수(-0.5) 반환")
                return -0.5
            
            print(f"  -> 결과: IR = {ir_score:.4f}")
            return ir_score

        except Exception as e:
            print(f"  -> 목적 함수 실행 중 오류: {e}, 낮은 점수(-2.0) 반환")
            return -2.0 # 예외 발생 시 매우 낮은 점수 반환
    
    def optimize(self, init_points=5, n_iter=10):
        """
        하이퍼파라미터 최적화를 시작합니다.

        Args:
            init_points (int): 무작위 탐색을 수행할 초기 스텝 수.
            n_iter (int): 베이지안 최적화를 수행할 반복 횟수.

        Returns:
            dict: 찾은 최적의 하이퍼파라미터 딕셔너리.
        """
        print("\n" + "="*80)
        print("          ✨ 하이퍼파라미터 최적화 시작 ✨")
        print("="*80)

        optimizer = BayesianOptimization(
            f=self._objective_function,
            pbounds=self.pbounds,
            random_state=42,
            verbose=2 # 0: صمت, 1: 진행상황, 2: 모든 정보
        )
        
        optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )
        
        print("\n" + "="*80)
        print("          ✨ 하이퍼파라미터 최적화 종료 ✨")
        print("="*80)
        print("최적 하이퍼파라미터:")
        print(optimizer.max['params'])
        
        # 정수형 파라미터 변환
        best_params = optimizer.max['params']
        best_params['max_complexity_sl'] = int(best_params['max_complexity_sl'])
        best_params['max_complexity_pc'] = int(best_params['max_complexity_pc'])

        return best_params

