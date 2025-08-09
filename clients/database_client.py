# clients/database_client.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class DatabaseClient:
    """
    생성된 가설, 팩터, 평가 결과를 인메모리에서 관리하는 간소화된 데이터베이스 클라이언트입니다.
    """
    def __init__(self, data_url: str, transaction_fee_buy: float = 0.0005, transaction_fee_sell: float = 0.0015):
        self.data_url = data_url
        self.transaction_fee_buy = transaction_fee_buy
        self.transaction_fee_sell = transaction_fee_sell
        self.data_cache = None
        self.factor_cache = {}

        # 아래 코드를 추가하여 데이터프레임을 미리 초기화해야 합니다.
        self._hypothesis_id_counter = 0
        self.hypotheses = pd.DataFrame(columns=['id', 'hypothesis_text', 'data', 'status'])
        
        self._factor_id_counter = 0
        self.factors = pd.DataFrame(columns=['id', 'formula', 'description', 'status', 'complexity_sl', 'originality_score', 'alignment_score'])
        
        self.evaluations = pd.DataFrame(columns=['factor_id', 'IC', 'RankIC', 'ICIR', 'AR', 'IR', 'MDD'])

    def save_hypothesis(self, data: Dict[str, Any]) -> int:
        """새로운 가설을 저장합니다."""
        self._hypothesis_id_counter += 1
        new_id = self._hypothesis_id_counter
        new_row = {
            'id': new_id,
            'hypothesis_text': data.get('hypothesis'),
            'data': data,
            'status': 'new' # 'new', 'processing', 'done'
        }
        self.hypotheses.loc[len(self.hypotheses)] = new_row
        return new_id

    def save_factor(self, data: Dict[str, Any]) -> int:
        """새로운 팩터를 저장합니다."""
        self._factor_id_counter += 1
        new_id = self._factor_id_counter
        data['id'] = new_id
        data['status'] = 'new' # 'new', 'evaluating', 'evaluated'
        self.factors.loc[len(self.factors)] = pd.Series(data)
        return new_id
    
    def save_evaluation(self, data: Dict[str, Any]):
        """팩터 평가 결과를 저장합니다."""
        self.evaluations.loc[len(self.evaluations)] = pd.Series(data)
        # 평가가 완료된 팩터의 상태를 업데이트
        self.update_factor_status(data['factor_id'], 'evaluated')
    
    def get_new_hypotheses(self) -> List[Dict[str, Any]]:
        """처리되지 않은 새로운 가설들을 가져옵니다."""
        new_hypotheses_df = self.hypotheses[self.hypotheses['status'] == 'new']
        return new_hypotheses_df.to_dict('records')

    def get_new_factors(self) -> List[Dict[str, Any]]:
        """평가되지 않은 새로운 팩터들을 가져옵니다."""
        new_factors_df = self.factors[self.factors['status'] == 'new']
        return new_factors_df.to_dict('records')

    def get_all_hypothesis_texts(self) -> List[str]:
        """모든 가설 텍스트를 리스트로 반환합니다."""
        return self.hypotheses['hypothesis_text'].tolist()

    def get_best_factor(self) -> Dict[str, Any]:
        """가장 성과가 좋은 팩터를 찾습니다. (IR 기준)"""
        if self.evaluations.empty:
            return None
        
        # factor 테이블과 evaluation 테이블을 factor_id 기준으로 join
        merged_df = pd.merge(self.factors, self.evaluations, left_on='id', right_on='factor_id')
        if merged_df.empty:
            return None
            
        # ir이 가장 높은 row를 선택
        best_factor_row = merged_df.loc[merged_df['ir'].idxmax()]
        return best_factor_row.to_dict()

    def update_hypothesis_status(self, hypothesis_id: int, status: str):
        self.hypotheses.loc[self.hypotheses['id'] == hypothesis_id, 'status'] = status

    def update_factor_status(self, factor_id: int, status: str):
        self.factors.loc[self.factors['id'] == factor_id, 'status'] = status

    def get_evaluation_summary(self) -> str:
        """
        과거의 성공 및 실패 사례를 요약하여 LLM 프롬프트에 제공할 문자열을 생성합니다.
        
        Returns:
            str: 성공/실패 사례 요약 정보.
        """
        if self.evaluations.empty:
            return "아직 평가된 팩터가 없습니다. 자유롭게 새로운 아이디어를 탐색하세요."

        merged_df = pd.merge(self.factors, self.evaluations, left_on='id', right_on='factor_id')
        sorted_df = merged_df.sort_values(by='ir', ascending=False)
        
        summary_parts = []
        
        summary_parts.append("### 성공적인 팩터 분석 (High IR)")
        top_factors = sorted_df.head(3)
        if not top_factors.empty:
            for _, row in top_factors.iterrows():
                summary_parts.append(
                    f"- 공식: {row['formula']}\n"
                    f"  - 설명: {row['description']}\n"
                    f"  - 성과: IR={row['ir']:.3f}, AR={row['ar']:.2%}, MDD={row['mdd']:.2%}\n"
                    f"  - 특징: 복잡도={row['complexity_sl']}, 유사도={row['originality_score']:.2f}"
                )
        else:
            summary_parts.append("- 아직 유의미한 성공 사례가 없습니다.")
            
        summary_parts.append("\n### 개선이 필요한 팩터 분석 (Low IR)")
        bottom_factors = sorted_df.tail(3)
        if not bottom_factors.empty:
            for _, row in bottom_factors.iterrows():
                 summary_parts.append(
                    f"- 공식: {row['formula']}\n"
                    f"  - 성과: IR={row['ir']:.3f}\n"
                    f"  - 교훈: 위와 같은 구조나 아이디어는 낮은 성과를 보였으므로, 다른 접근 방식이 필요합니다."
                )
        else:
            summary_parts.append("- 아직 유의미한 실패 사례가 없습니다.")

        return "\n".join(summary_parts)



