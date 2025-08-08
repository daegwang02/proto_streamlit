# clients/backtester_client.py

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
            transaction_fee (float): 매수/매도 시 발생하는 총 거래비용 (수수료+세금)
        """
        self.data_url = data_url
        self.transaction_fee = transaction_fee
        self.data_cache = None
        self.factor_cache = {}

    # --- 데이터 로딩 및 팩터 값 계산 메서드 (이전 코드와 동일) ---
    def _load_data(self) -> pd.DataFrame:
        # (이전 단계에서 작성한 _load_data 메서드 코드가 여기에 위치합니다. 생략)
        if self.data_cache is not None: return self.data_cache
        print("데이터를 로드하는 중입니다...")
        df = pd.read_parquet(self.data_url)
        required_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols): raise ValueError(f"데이터에 필수 컬럼({required_cols})이 포함되어야 합니다.")
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index(['date', 'ticker']).sort_index()
        df['volume'] = df['volume'].replace(0, np.nan)
        df[['open', 'high', 'low', 'close', 'volume']] = df.groupby('ticker')[['open', 'high', 'low', 'close', 'volume']].ffill()
        df.dropna(inplace=True)
        df.rename(columns={'close': 'price'}, inplace=True)
        self.data_cache = df
        print("데이터 로드 및 기본 전처리 완료.")
        return self.data_cache

    def _execute_ast(self, node: ASTNode, market_data: pd.DataFrame) -> Any:
        # (이전 단계에서 작성한 _execute_ast 메서드 코드가 여기에 위치합니다. 생략)
        if isinstance(node, LiteralNode): return node.value
        if isinstance(node, VariableNode):
            if node.name == 'returns': return market_data.groupby('ticker')['price'].pct_change()
            if node.name in market_data.columns: return market_data[node.name]
            # adv20과 같은 파생변수 처리 추가
            if node.name.startswith('adv'):
                try:
                    days = int(node.name[3:])
                    turnover_col = market_data['price'] * market_data['volume']
                    return turnover_col.groupby(level='ticker').rolling(window=days).mean().reset_index(0, drop=True)
                except: raise NameError(f"adv 파생 변수 파싱 오류: {node.name}")
            raise NameError(f"정의되지 않은 변수입니다: {node.name}")
        if isinstance(node, OperatorNode):
            children_values = [self._execute_ast(child, market_data) for child in node.children]
            op_name = node.op.lower()
            if op_name == 'rank': return children_values[0].groupby(level='date').rank(pct=True)
            if op_name == 'delay': return children_values[0].groupby(level='ticker').shift(int(children_values[1]))
            if op_name == '+': return children_values[0] + children_values[1]
            if op_name == '-': return children_values[0] - children_values[1]
            if op_name == '*': return children_values[0] * children_values[1]
            if op_name == '/': return children_values[0] / children_values[1].replace(0, 1e-6)
            # ... 기타 모든 연산자 구현 ...
            if op_name == 'if':
                condition, true_val, false_val = children_values
                return pd.Series(np.where(condition, true_val, false_val), index=condition.index)
            # ...
            raise NameError(f"정의되지 않은 연산자입니다: {node.op}")
        raise TypeError(f"처리할 수 없는 노드 타입입니다: {type(node)}")

    def calculate_factor_values(self, formula: str, ast: ASTNode) -> pd.Series:
        # (이전 단계에서 작성한 calculate_factor_values 메서드 코드가 여기에 위치합니다. 생략)
        if formula in self.factor_cache: return self.factor_cache[formula]
        market_data = self._load_data()
        factor_values = self._execute_ast(ast, market_data)
        factor_values = factor_values.reindex(market_data.index).sort_index()
        factor_values.replace([np.inf, -np.inf], np.nan, inplace=True)
        factor_values = factor_values.groupby(level='ticker').ffill().bfill()
        self.factor_cache[formula] = factor_values
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


if __name__ == '__main__':
    # (이전 단계의 테스트 코드와 동일)
    # 실제 실행을 위해서는 YOUR_GITHUB_PARQUET_FILE_URL을 유효한 URL로 변경해야 합니다.
    TEST_DATA_URL = "YOUR_GITHUB_PARQUET_FILE_URL"
    
    from foundations.factor_structure import FactorParser
    
    try:
        backtester = BacktesterClient(TEST_DATA_URL)
        parser = FactorParser()
        
        # Alpha#2를 테스트
        formula = "(-1 * correlation(rank(delta(log(volume), 2)), rank(((price - open) / open)), 6))"
        ast = parser.parse(formula)
        
        # 1. 팩터 값 계산
        factor_values = backtester.calculate_factor_values(formula, ast)
        print("계산된 팩터 값 샘플:")
        print(factor_values.head())
        
        # 2. 전체 백테스트 실행
        performance_metrics = backtester.run_full_backtest(factor_values)
        
        print("\n--- 최종 백테스팅 결과 ---")
        for key, value in performance_metrics.items():
            print(f"{key:>8}: {value:.4f}")

    except (RuntimeError, ValueError, NameError) as e:
        print(f"\n테스트 중 오류 발생: {e}")
        print("config.py의 KOR_STOCK_DATA_URL을 유효한 경로로 설정했는지 확인해주세요.")


