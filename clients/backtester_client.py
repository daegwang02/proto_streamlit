# clients/backtester_client.py

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any

# 이전 단계에서 구현한 클래스들을 import 합니다.
from foundations.factor_structure import ASTNode, OperatorNode, VariableNode, LiteralNode

# 팩터 연산에 사용할 연산자 라이브러리를 import 합니다.
# 이 라이브러리의 함수/클래스들은 _execute_ast에서 동적으로 호출됩니다.
from foundations import operator_library as op_lib

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

class BacktesterClient:
    """
    팩터 백테스팅을 총괄하는 클라이언트입니다.
    데이터 로딩, 팩터 값 계산, 모델 학습, 성능 평가를 수행합니다.
    """
    def __init__(self, data_url: str):
        """
        BacktesterClient를 초기화합니다.

        Args:
            data_url (str): 한국 주식 시장 데이터가 저장된 Parquet 파일의 URL.
        """
        self.data_url = data_url
        self.data_cache = None  # 데이터 로딩 캐시
        self.factor_cache = {}  # 팩터 계산 결과 캐시

    def _load_data(self) -> pd.DataFrame:
        """
        지정된 URL에서 주식 데이터를 로드하고 기본적인 전처리를 수행합니다.
        데이터는 한 번만 로드하여 캐시에 저장합니다.
        """
        if self.data_cache is not None:
            return self.data_cache

        print("데이터를 로드하는 중입니다...")
        try:
            # parquet 파일을 URL에서 직접 로드
            df = pd.read_parquet(self.data_url)
        except Exception as e:
            raise RuntimeError(f"데이터 로드 실패: {self.data_url} 경로를 확인해주세요. 오류: {e}")

        # 필수 컬럼 확인
        required_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"데이터에 필수 컬럼({required_cols})이 포함되어야 합니다.")

        # 데이터 타입 변환 및 멀티인덱스 설정
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index(['date', 'ticker']).sort_index()

        # 거래량 0인 데이터를 처리 (nan으로 변경 후 이전 값으로 채우기)
        df['volume'] = df['volume'].replace(0, np.nan)
        df[['open', 'high', 'low', 'close', 'volume']] = df.groupby('ticker')[['open', 'high', 'low', 'close', 'volume']].ffill()
        df.dropna(inplace=True)

        # 파생 변수(피처) 계산 (논문에서 언급된 adv{d} 등)
        # 예시: 20일 평균 거래대금(adv20)
        df['daily_turnover'] = df['close'] * df['volume']
        df['adv20'] = df.groupby('ticker')['daily_turnover'].rolling(window=20, min_periods=1).mean().reset_index(0, drop=True)
        # 그 외 필요한 피처들을 여기에 추가할 수 있습니다.
        df.rename(columns={'close': 'price'}, inplace=True) # close와 price를 동일하게 사용

        self.data_cache = df
        print("데이터 로드 및 기본 전처리 완료.")
        return self.data_cache

    def _execute_ast(self, node: ASTNode, market_data: pd.DataFrame) -> Any:
        """
        AST를 재귀적으로 실행하여 팩터 값을 계산합니다.

        Args:
            node (ASTNode): 실행할 AST의 현재 노드.
            market_data (pd.DataFrame): 전체 시장 데이터.

        Returns:
            pd.Series or float or int: 계산 결과.
        """
        # --- 베이스 케이스: 노드가 변수이거나 리터럴일 때 ---
        if isinstance(node, LiteralNode):
            return node.value
        if isinstance(node, VariableNode):
            # 'returns'와 같은 동적 변수는 여기서 계산
            if node.name == 'returns':
                return market_data.groupby('ticker')['price'].pct_change()
            if node.name in market_data.columns:
                return market_data[node.name]
            else:
                raise NameError(f"정의되지 않은 변수입니다: {node.name}")

        # --- 재귀 케이스: 노드가 연산자일 때 ---
        if isinstance(node, OperatorNode):
            # 자식 노드들의 값을 재귀적으로 먼저 계산
            children_values = [self._execute_ast(child, market_data) for child in node.children]

            # 연산자 이름에 따라 분기하여 실제 연산 수행
            op_name = node.op.lower() # 연산자 이름은 소문자로 통일하여 처리

            # 그룹화(Cross-sectional) 연산: rank, scale, indneutralize
            if op_name == 'rank':
                # 날짜별로 그룹화하여 순위(0~1)를 매김
                return children_values[0].groupby(level='date').rank(pct=True)
            if op_name == 'scale':
                # 날짜별로 그룹화하여 합이 1이 되도록 스케일링
                return children_values[0].groupby(level='date').transform(lambda x: x / x.abs().sum())

            # 시계열(Time-series) 연산: delay, delta, correlation, sum, stddev 등
            # 종목별로 그룹화하여 시계열 연산 적용
            if op_name in ['delay', 'delta', 'sum', 'stddev', 'ts_min', 'ts_max', 'ts_rank']:
                series = children_values[0]
                d = int(children_values[1])
                
                grouped_series = series.groupby(level='ticker')
                
                if op_name == 'delay': return grouped_series.shift(d)
                if op_name == 'delta': return series - grouped_series.shift(d)
                if op_name == 'sum': return grouped_series.rolling(window=d).sum()
                if op_name == 'stddev': return grouped_series.rolling(window=d).std()
                if op_name == 'ts_min': return grouped_series.rolling(window=d).min()
                if op_name == 'ts_max': return grouped_series.rolling(window=d).max()
                if op_name == 'ts_rank': return grouped_series.rolling(window=d).rank(pct=True)

            if op_name == 'correlation':
                series1, series2, d = children_values[0], children_values[1], int(children_values[2])
                return series1.groupby(level='ticker').rolling(window=d).corr(series2).unstack(level=1).iloc[:,0]

            if op_name == 'covariance':
                series1, series2, d = children_values[0], children_values[1], int(children_values[2])
                return series1.groupby(level='ticker').rolling(window=d).cov(series2).unstack(level=1).iloc[:,0]
            
            # 기본 산술/논리 연산자
            if op_name == '+': return children_values[0] + children_values[1]
            if op_name == '-': return children_values[0] - children_values[1]
            if op_name == '*': return children_values[0] * children_values[1]
            if op_name == '/': return children_values[0] / children_values[1].replace(0, 1e-6) # 0으로 나누기 방지
            if op_name == '^': return np.power(children_values[0], children_values[1])
            if op_name == 'neg': return -children_values[0]
            if op_name == 'abs': return children_values[0].abs()
            if op_name == 'sign': return np.sign(children_values[0])
            if op_name == 'log': return np.log(children_values[0].replace(0, 1e-6).abs())

            # 조건 연산자
            if op_name == 'if':
                condition, true_val, false_val = children_values
                return pd.Series(np.where(condition, true_val, false_val), index=condition.index)

            if op_name == '>': return children_values[0] > children_values[1]
            if op_name == '<': return children_values[0] < children_values[1]
            if op_name == '>=': return children_values[0] >= children_values[1]
            if op_name == '<=': return children_values[0] <= children_values[1]
            if op_name == '==': return children_values[0] == children_values[1]
            if op_name == '&&': return children_values[0] & children_values[1]
            if op_name == '||': return children_values[0] | children_values[1]

            raise NameError(f"정의되지 않은 연산자입니다: {node.op}")

        raise TypeError(f"처리할 수 없는 노드 타입입니다: {type(node)}")

    def calculate_factor_values(self, formula: str, ast: ASTNode) -> pd.Series:
        """
        주어진 공식과 AST를 사용하여 최종 팩터 값을 계산하고 캐시에 저장합니다.

        Args:
            formula (str): 팩터의 고유한 키로 사용될 공식 문자열.
            ast (ASTNode): 실행할 팩터의 AST.

        Returns:
            pd.Series: 날짜와 종목을 인덱스로 갖는 팩터 값 시계열.
        """
        if formula in self.factor_cache:
            return self.factor_cache[formula]

        print(f"팩터 값 계산 중: {formula}")
        market_data = self._load_data()
        
        # AST 실행
        factor_values = self._execute_ast(ast, market_data)
        
        # 결과가 Series 형태인지 확인하고 인덱스 정렬
        if not isinstance(factor_values, pd.Series):
             raise TypeError("팩터 계산 결과는 pandas Series여야 합니다.")

        # market_data와 인덱스를 맞추고, 무한대/결측값 처리
        factor_values = factor_values.reindex(market_data.index).sort_index()
        factor_values.replace([np.inf, -np.inf], np.nan, inplace=True)
        factor_values = factor_values.groupby(level='ticker').ffill().bfill() # 종목별로 결측치 채우기

        # 캐시에 저장
        self.factor_cache[formula] = factor_values
        print("팩터 값 계산 완료.")
        return factor_values


if __name__ == '__main__':
    # 테스트를 위해 임시 데이터 URL과 파서 인스턴스 생성
    # 중요: 아래 URL은 실제 유효한 Parquet 파일 URL로 대체해야 합니다.
    # 예: 'https://github.com/YourUsername/YourRepo/raw/main/korea_stock_data.parquet'
    TEST_DATA_URL = "YOUR_GITHUB_PARQUET_FILE_URL"
    
    from foundations.factor_structure import FactorParser

    try:
        backtester = BacktesterClient(TEST_DATA_URL)
        
        # 데이터 로딩 테스트
        # data = backtester._load_data()
        # print("로드된 데이터 샘플:")
        # print(data.head())
        # print(f"데이터 크기: {data.shape}")

        # 팩터 계산 테스트 (Alpha#5)
        parser = FactorParser()
        formula_to_test = "(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"
        ast_to_test = parser.parse(formula_to_test)
        
        # vwap이 없으므로 임시로 price로 대체하여 테스트
        formula_for_test = "(rank((open - (sum(price, 10) / 10))) * (-1 * abs(rank((price - price)))))"
        ast_for_test = parser.parse(formula_for_test)
        
        # 실제 계산을 위해서는 유효한 데이터 URL이 필요하므로, 여기서는 실행 흐름만 확인
        print("\nBacktesterClient가 정상적으로 초기화되었습니다.")
        print("실제 팩터 계산을 위해서는 config.py의 KOR_STOCK_DATA_URL을 유효한 경로로 설정해야 합니다.")

    except (RuntimeError, ValueError) as e:
        print(f"테스트 중 오류 발생: {e}")

