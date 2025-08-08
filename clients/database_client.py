# clients/database_client.py

import pandas as pd
from typing import Dict, Any, List

class DatabaseClient:
    """
    생성된 가설, 팩터, 평가 결과를 인메모리에서 관리하는 간소화된 데이터베이스 클라이언트입니다.
    """
    def __init__(self):
        self.hypotheses = pd.DataFrame(columns=['id', 'hypothesis_text', 'data', 'status'])
        self.factors = pd.DataFrame(columns=['id', 'hypothesis_id', 'description', 'formula', 'ast', 'complexity_sl', 'complexity_pc', 'originality_score', 'alignment_score', 'status'])
        self.evaluations = pd.DataFrame(columns=['factor_id', 'ic', 'rank_ic', 'icir', 'ar', 'ir', 'mdd'])

        self._hypothesis_id_counter = 0
        self._factor_id_counter = 0

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

