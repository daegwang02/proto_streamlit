# clients/backtester_client.py (수정 완료)

# import pandas as pd
# import numpy as np
# import warnings
# import lightgbm as lgb
# from typing import Dict, Any, Tuple

# from foundations.factor_structure import ASTNode, OperatorNode, VariableNode, LiteralNode
# from foundations import operator_library as op_lib

# warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# class BacktesterClient:
#     """
#     팩터 백테스팅을 총괄하는 클라이언트입니다.
#     데이터 로딩, 팩터 값 계산, 모델 학습, 성능 평가를 수행합니다.
#     """
#     def __init__(self, data_url: str, transaction_fee_buy: float = 0.0005, transaction_fee_sell: float = 0.0015):
#         """
#         BacktesterClient를 초기화합니다.
        
#         Args:
#             data_url (str): 한국 주식 시장 데이터가 저장된 Parquet 파일의 URL.
#             transaction_fee_sell (float): 매도 시 발생하는 총 거래비용 (수수료+세금)
#         """
#         #transaction_fee_buy (float): 매수 시 발생하는 거래비용 (수수료)
#         self.data_url = data_url
#         #self.transaction_fee_buy = transaction_fee_buy
#         self.transaction_fee_sell = transaction_fee_sell
#         self.data_cache = None
#         self.factor_cache = {}

#     def _load_data(self) -> pd.DataFrame:
#         """
#         지정된 URL에서 주식 데이터를 로드하고 기본적인 전처리를 수행합니다.
#         데이터는 한 번만 로드하여 캐시에 저장합니다.
#         """
#         if self.data_cache is not None:
#             return self.data_cache

#         print("데이터를 로드하는 중입니다...")
#         try:
#             df = pd.read_parquet(self.data_url)
#         except Exception as e:
#             raise RuntimeError(f"데이터 로드 실패: {self.data_url} 경로를 확인해주세요. 오류: {e}")

#         required_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
#         if not all(col in df.columns for col in required_cols):
#             raise ValueError(f"데이터에 필수 컬럼({required_cols})이 포함되어야 합니다.")

#         df['date'] = pd.to_datetime(df['date'])
#         df = df.set_index(['date', 'ticker']).sort_index()

#         df['volume'] = df['volume'].replace(0, np.nan)
#         df[['open', 'high', 'low', 'close', 'volume']] = df.groupby('ticker')[['open', 'high', 'low', 'close', 'volume']].ffill()
#         df.dropna(inplace=True)

#         df.rename(columns={'close': 'price'}, inplace=True)
        
#         # 파생 변수(피처) 계산
#         df['daily_turnover'] = df['price'] * df['volume']
#         df['adv20'] = df.groupby(level='ticker')['daily_turnover'].rolling(window=20, min_periods=1).mean().reset_index(level=0, drop=True)
#         # 필요한 다른 adv 파생 변수도 여기에 추가할 수 있습니다.
        
#         self.data_cache = df
#         print("데이터 로드 및 기본 전처리 완료.")
#         return self.data_cache

#     def _execute_ast(self, node: ASTNode, market_data: pd.DataFrame) -> Any:
#         # 이 부분의 로직은 이전과 동일하지만, 누락된 연산자들을 추가했습니다.
#         if isinstance(node, LiteralNode): return node.value
#         if isinstance(node, VariableNode):
#             if node.name == 'returns': return market_data.groupby('ticker')['price'].pct_change()
#             if node.name in market_data.columns: return market_data[node.name]
#             if node.name.startswith('adv'):
#                 try:
#                     days = int(node.name[3:])
#                     turnover_col = market_data['price'] * market_data['volume']
#                     return turnover_col.groupby(level='ticker').rolling(window=days, min_periods=1).mean().reset_index(level=0, drop=True)
#                 except: raise NameError(f"adv 파생 변수 파싱 오류: {node.name}")
#             raise NameError(f"정의되지 않은 변수입니다: {node.name}")
#         if isinstance(node, OperatorNode):
#             children_values = [self._execute_ast(child, market_data) for child in node.children]
#             op_name = node.op.lower()
            
#             # 여기서 모든 연산자를 명시적으로 구현해야 합니다.
#             if op_name == 'rank': return children_values[0].groupby(level='date').rank(pct=True)
#             if op_name == 'delay': return children_values[0].groupby(level='ticker').shift(int(children_values[1]))
#             if op_name == '+': return children_values[0] + children_values[1]
#             if op_name == '-': return children_values[0] - children_values[1]
#             if op_name == '*': return children_values[0] * children_values[1]
#             if op_name == '/': return children_values[0] / children_values[1].replace(0, 1e-6)
#             if op_name == 'if':
#                 condition, true_val, false_val = children_values
#                 return pd.Series(np.where(condition, true_val, false_val), index=condition.index)
            
#             # --- 누락된 연산자 추가 ---
#             if op_name == 'sum': return children_values[0].groupby(level='ticker').rolling(int(children_values[1])).sum().reset_index(level=0, drop=True)
#             if op_name == 'abs': return np.abs(children_values[0])
#             if op_name == 'log': return np.log(children_values[0].replace(0, np.nan)) # log(0) 방지
#             # ... 그 외 누락된 연산자들을 여기에 추가 ...

#             raise NameError(f"정의되지 않은 연산자입니다: {node.op}")
#         raise TypeError(f"처리할 수 없는 노드 타입입니다: {type(node)}")

#     def calculate_factor_values(self, formula: str, ast: ASTNode) -> pd.Series:
#         if formula in self.factor_cache: return self.factor_cache[formula]
#         market_data = self._load_data()
#         factor_values = self._execute_ast(ast, market_data)
#         factor_values = factor_values.reindex(market_data.index).sort_index()
#         factor_values.replace([np.inf, -np.inf], np.nan, inplace=True)
#         factor_values = factor_values.groupby(level='ticker').ffill().bfill()
#         self.factor_cache[formula] = factor_values
#         return factor_values

#     def _prepare_data_for_model(self, new_factor: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
#         # 이 메서드는 변경 없이 그대로 사용합니다.
#         market_data = self._load_data().copy()
#         base_features = pd.DataFrame(index=market_data.index)
#         base_features['intra_ret'] = (market_data['price'] - market_data['open']) / market_data['open']
#         base_features['daily_ret'] = market_data.groupby('ticker')['price'].pct_change()
#         vol_mean_20 = market_data.groupby('ticker')['volume'].rolling(20).mean().reset_index(level=0, drop=True)
#         base_features['vol_ratio_20'] = market_data['volume'] / vol_mean_20
#         base_features['range_norm'] = (market_data['high'] - market_data['low']) / market_data['price']
#         X = pd.concat([base_features, new_factor.rename('new_factor')], axis=1)
#         y = market_data.groupby('ticker')['price'].pct_change().shift(-1)
#         y.name = 'target'
#         data = pd.concat([X, y], axis=1).dropna()
#         return data.drop(columns='target'), data['target']

#     def _calculate_ic(self, predictions: pd.Series, actuals: pd.Series, rank=False) -> pd.Series:
#         # 이 메서드는 변경 없이 그대로 사용합니다.
#         df = pd.DataFrame({'pred': predictions, 'actual': actuals})
#         if rank:
#             df = df.groupby(level='date').rank(pct=True)
#         daily_ic = df.groupby(level='date').apply(lambda x: x['pred'].corr(x['actual']))
#         return daily_ic

#     def run_full_backtest(self, formula: str, ast: ASTNode) -> Dict[str, float]:
#         """
#         전체 백테스팅 파이프라인을 실행합니다.
#         (데이터 준비 -> 모델 학습 및 예측 -> 포트폴리오 수익률 계산 -> 성과 지표 산출)
#         """
#         print("전체 백테스팅을 시작합니다...")
        
#         # 팩터 값을 먼저 계산합니다.
#         factor_values = self.calculate_factor_values(formula, ast)
        
#         X, y = self._prepare_data_for_model(factor_values)
        
#         train_end = '2019-12-31'
#         test_start = '2021-01-01'

#         X_train, y_train = X.loc[:train_end], y.loc[:train_end]
#         X_test, y_test = X.loc[test_start:], y.loc[test_start:]

#         print("LightGBM 모델을 학습합니다...")
#         model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=200, learning_rate=0.05, num_leaves=31, n_jobs=-1)
#         model.fit(X_train, y_train)
        
#         predictions = pd.Series(model.predict(X_test), index=X_test.index)
        
#         print("포트폴리오 수익률을 계산합니다...")
#         daily_returns = []
#         for date in y_test.index.get_level_values('date').unique():
#             daily_pred = predictions.loc[date]
#             daily_actual_ret = y_test.loc[date]
            
#             if len(daily_pred) < 55: continue

#             long_stocks = daily_pred.nlargest(50).index
#             short_stocks = daily_pred.nsmallest(5).index
            
#             # 매수/매도 수수료를 분리하여 적용합니다.
#             long_return = daily_actual_ret.loc[long_stocks].mean() - self.transaction_fee_buy
#             short_return = -daily_actual_ret.loc[short_stocks].mean() - self.transaction_fee_sell
            
#             daily_portfolio_return = 0.5 * long_return + 0.5 * short_return
#             daily_returns.append({'date': date, 'return': daily_portfolio_return})

#         portfolio_returns = pd.DataFrame(daily_returns).set_index('date')['return']

#         print("최종 성과 지표를 산출합니다...")
#         daily_ic = self._calculate_ic(predictions, y_test)
#         ic_mean = daily_ic.mean()
#         icir = daily_ic.mean() / daily_ic.std()

#         daily_rank_ic = self._calculate_ic(predictions, y_test, rank=True)
#         rank_ic_mean = daily_rank_ic.mean()

#         cumulative_returns = (1 + portfolio_returns).cumprod()
#         total_days = len(portfolio_returns)
#         annualized_return = (cumulative_returns.iloc[-1])**(252 / total_days) - 1 if total_days > 0 else 0.0
#         annualized_vol = portfolio_returns.std() * np.sqrt(252)
#         information_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0.0

#         peak = cumulative_returns.expanding(min_periods=1).max()
#         drawdown = (cumulative_returns - peak) / peak
#         mdd = drawdown.min()

#         results = {
#             'IC': ic_mean,
#             'RankIC': rank_ic_mean,
#             'ICIR': icir,
#             'AR': annualized_return,
#             'IR': information_ratio,
#             'MDD': mdd,
#         }
        
#         print("백테스팅 완료.")
#         return results

import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from typing import Dict, Any, Tuple

# 이전 단계에서 구현한 클래스들을 import 합니다.
from foundations.factor_structure import ASTNode, OperatorNode, VariableNode, LiteralNode

# 팩터 연산에 사용할 연산자 라이브러리를 import 합니다.
from foundations import operator_library as op_lib

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

class BacktesterClient:
    """
    팩터 백테스팅을 총괄하는 클라이언트입니다.
    데이터 로딩, 팩터 값 계산, 모델 학습, 성능 평가를 수행합니다.
    """
    def __init__(self, data_url: str, transaction_fee: float = 0.0020):
        """
        BacktesterClient를 초기화합니다.

        Args:
            data_url (str): 한국 주식 시장 데이터가 저장된 Parquet 파일의 URL.
            transaction_fee (float): 매수/매도 시 발생하는 총 거래비용 (수수료+세금).
        """
        self.data_url = data_url
        self.transaction_fee = transaction_fee
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

        # 종가(close)를 price로 통일
        df.rename(columns={'close': 'price'}, inplace=True) 

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
            if node.name == 'returns':
                return market_data.groupby('ticker')['price'].pct_change()
            if node.name in market_data.columns:
                return market_data[node.name]
            # adv20과 같은 파생변수 처리 추가
            if node.name.startswith('adv'):
                try:
                    days = int(node.name[3:])
                    turnover_col = market_data['price'] * market_data['volume']
                    return turnover_col.groupby(level='ticker').rolling(window=days, min_periods=1).mean().reset_index(0, drop=True)
                except:
                    raise NameError(f"adv 파생 변수 파싱 오류: {node.name}")
            raise NameError(f"정의되지 않은 변수입니다: {node.name}")

        # --- 재귀 케이스: 노드가 연산자일 때 ---
        if isinstance(node, OperatorNode):
            # 자식 노드들의 값을 재귀적으로 먼저 계산
            children_values = [self._execute_ast(child, market_data) for child in node.children]

            # 연산자 이름에 따라 분기하여 실제 연산 수행
            op_name = node.op.lower() # 연산자 이름은 소문자로 통일하여 처리

            # 그룹화(Cross-sectional) 연산: rank, scale
            if op_name == 'rank':
                return children_values[0].groupby(level='date').rank(pct=True)
            if op_name == 'scale':
                return children_values[0].groupby(level='date').transform(lambda x: x / x.abs().sum())

            # 시계열(Time-series) 연산: delay, delta, correlation, sum, stddev 등
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

    # --- 백테스팅 실행 및 성과 측정 메서드 (신규 추가) ---

    def _prepare_data_for_model(self, new_factor: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """모델 학습을 위한 피처(X)와 타겟(y) 데이터를 준비합니다."""
        market_data = self._load_data().copy()
        
        # 1. 기본 피처(Base Alphas) 생성
        # AlphaAgent 논문에서 언급된 4가지 기본 알파와 유사하게 구성
        base_features = pd.DataFrame(index=market_data.index)
        base_features['intra_ret'] = (market_data['price'] - market_data['open']) / market_data['open']
        base_features['daily_ret'] = market_data.groupby('ticker')['price'].pct_change()
        vol_mean_20 = market_data.groupby('ticker')['volume'].rolling(20).mean().reset_index(0,drop=True)
        base_features['vol_ratio_20'] = market_data['volume'] / vol_mean_20
        base_features['range_norm'] = (market_data['high'] - market_data['low']) / market_data['price']
        
        # 2. 새로운 팩터를 피처셋에 추가
        X = pd.concat([base_features, new_factor.rename('new_factor')], axis=1)
        
        # 3. 타겟 변수(y) 생성: 다음 날의 수익률
        y = market_data.groupby('ticker')['price'].pct_change().shift(-1)
        y.name = 'target'
        
        # 4. 데이터 정렬 및 결측치 제거
        data = pd.concat([X, y], axis=1).dropna()
        
        return data.drop(columns='target'), data['target']

    def _calculate_ic(self, predictions: pd.Series, actuals: pd.Series, rank=False) -> pd.Series:
        """일별 IC 또는 Rank IC를 계산합니다."""
        df = pd.DataFrame({'pred': predictions, 'actual': actuals})
        if rank:
            df = df.groupby(level='date').rank(pct=True)
        
        # 날짜별로 상관계수 계산
        daily_ic = df.groupby(level='date').apply(lambda x: x['pred'].corr(x['actual']))
        return daily_ic

    def run_full_backtest(self, factor_values: pd.Series) -> Dict[str, float]:
        """
        전체 백테스팅 파이프라인을 실행합니다.
        (데이터 준비 -> 모델 학습 및 예측 -> 포트폴리오 수익률 계산 -> 성과 지표 산출)
        """
        print("전체 백테스팅을 시작합니다...")
        X, y = self._prepare_data_for_model(factor_values)
        
        # 논문의 기간 설정에 따라 학습/검증/테스트 기간 정의
        train_end = '2019-12-31'
        valid_end = '2020-12-31'
        test_start = '2021-01-01'

        X_train, y_train = X.loc[:train_end], y.loc[:train_end]
        X_test, y_test = X.loc[test_start:], y.loc[test_start:]

        # 모델 학습
        print("LightGBM 모델을 학습합니다...")
        model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=200, learning_rate=0.05, num_leaves=31, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # 테스트 기간 예측
        predictions = pd.Series(model.predict(X_test), index=X_test.index)
        
        # 포트폴리오 수익률 계산 (Long Top 50, Short Bottom 5)
        print("포트폴리오 수익률을 계산합니다...")
        daily_returns = []
        # 테스트 기간의 날짜들을 순회
        for date in y_test.index.get_level_values('date').unique():
            daily_pred = predictions.loc[date]
            daily_actual_ret = y_test.loc[date]
            
            if len(daily_pred) < 55: continue # 하루에 최소 55개 종목이 있어야 함

            # 롱/숏 포지션 결정
            long_stocks = daily_pred.nlargest(50).index
            short_stocks = daily_pred.nsmallest(5).index
            
            # 롱/숏 수익률 계산 (거래비용 반영)
            long_return = daily_actual_ret.loc[long_stocks].mean() - self.transaction_fee
            short_return = -daily_actual_ret.loc[short_stocks].mean() - self.transaction_fee # 숏 포지션은 수익률에 음수
            
            # 일일 포트폴리오 수익률 (롱/숏 비중 50:50 가정)
            daily_portfolio_return = 0.5 * long_return + 0.5 * short_return
            daily_returns.append({'date': date, 'return': daily_portfolio_return})

        portfolio_returns = pd.DataFrame(daily_returns).set_index('date')['return']

        # 성과 지표 계산
        print("최종 성과 지표를 산출합니다...")
        # 1. IC 기반 지표
        daily_ic = self._calculate_ic(predictions, y_test)
        ic_mean = daily_ic.mean()
        icir = daily_ic.mean() / daily_ic.std()

        daily_rank_ic = self._calculate_ic(predictions, y_test, rank=True)
        rank_ic_mean = daily_rank_ic.mean()

        # 2. 수익률 기반 지표
        cumulative_returns = (1 + portfolio_returns).cumprod()
        total_days = len(portfolio_returns)
        annualized_return = (cumulative_returns.iloc[-1])**(252 / total_days) - 1 if total_days > 0 else 0.0
        annualized_vol = portfolio_returns.std() * np.sqrt(252)
        information_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0.0

        # 3. 최대 낙폭 (MDD)
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        mdd = drawdown.min()

        results = {
            'IC': ic_mean,
            'RankIC': rank_ic_mean,
            'ICIR': icir,
            'AR': annualized_return,
            'IR': information_ratio,
            'MDD': mdd,
        }
        
        print("백테스팅 완료.")
        return results

# if __name__ == '__main__':
#     # 테스트를 위해 임시 데이터 URL과 파서 인스턴스 생성
#     # 중요: 아래 URL은 실제 유효한 Parquet 파일 URL로 대체해야 합니다.
#     TEST_DATA_URL = "YOUR_GITHUB_PARQUET_FILE_URL"
    
#     from foundations.factor_structure import FactorParser
    
#     try:
#         backtester = BacktesterClient(TEST_DATA_URL)
#         parser = FactorParser()
        
#         # Alpha#2를 테스트
#         # formula = "(-1 * correlation(rank(delta(log(volume), 2)), rank(((price - open) / open)), 6))"
#         # ast = parser.parse(formula)
        
#         # Alpha#5를 테스트
#         formula = "(rank((open - (sum(adv20, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"
#         # vwap이 없으므로 adv20, close로 대체
#         formula = "(rank((open - (sum(adv20, 10) / 10))) * (-1 * abs(rank((close - adv20)))))"
#         ast = parser.parse(formula)
        
#         # 1. 팩터 값 계산
#         factor_values = backtester.calculate_factor_values(formula, ast)
#         print("계산된 팩터 값 샘플:")
#         print(factor_values.head())
        
#         # 2. 전체 백테스트 실행
#         performance_metrics = backtester.run_full_backtest(factor_values)
        
#         print("\n--- 최종 백테스팅 결과 ---")
#         for key, value in performance_metrics.items():
#             print(f"{key:>8}: {value:.4f}")

#     except (RuntimeError, ValueError, NameError) as e:
#         print(f"\n테스트 중 오류 발생: {e}")
#         print("config.py의 KOR_STOCK_DATA_URL을 유효한 경로로 설정했는지 확인해주세요.")

