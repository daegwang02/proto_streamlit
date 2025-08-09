# foundations/operator_library.py (수정 완료)

"""
AlphaAgent의 팩터 연산에 사용될 함수들을 정의합니다.
이 파일은 clients/backtester_client.py의 _execute_ast 메서드에서 사용됩니다.
"""
import numpy as np
import pandas as pd

# 기본적인 연산자들을 딕셔너리 형태로 정의합니다.
# 이 딕셔너리의 키(예: 'ts_mean', 'add')는 FactorParser가 생성한 AST의 op_name과 일치해야 합니다.

OPERATORS = {
    # 횡단면(Cross-sectional) 연산자
    'rank': lambda df: df.rank(axis=1, pct=True),
    'scale': lambda df: df.div(df.abs().sum(axis=1), axis=0),

    # 시계열(Time-series) 연산자
    'ts_mean': lambda df, window: df.rolling(window, min_periods=1).mean(),
    'ts_std': lambda df, window: df.rolling(window, min_periods=1).std(),
    'ts_rank': lambda df, window: df.rolling(window, min_periods=1).rank(pct=True),
    'delay': lambda df, period: df.shift(period),
    'delta': lambda df, period: df.diff(period),
    'ts_min': lambda df, window: df.rolling(window, min_periods=1).min(),
    'ts_max': lambda df, window: df.rolling(window, min_periods=1).max(),
    'correlation': lambda df1, df2, window: df1.rolling(window, min_periods=1).corr(df2),
    'covariance': lambda df1, df2, window: df1.rolling(window, min_periods=1).cov(df2),
    
    # 산술 연산자
    'add': lambda a, b: a + b,
    'subtract': lambda a, b: a - b,
    'multiply': lambda a, b: a * b,
    'divide': lambda a, b: a / b.replace(0, np.nan), # 0으로 나누기 방지
    
    # 단항 연산자
    'negate': lambda a: -a,
    'abs': lambda a: np.abs(a),
    'log': lambda a: np.log(a.replace(0, np.nan)), # log(0) 방지

    # 논리 연산자
    'and': lambda a, b: a & b,
    'or': lambda a, b: a | b,
    'not': lambda a: ~a,
    
    # 비교 연산자
    'gt': lambda a, b: a > b,
    'ge': lambda a, b: a >= b,
    'lt': lambda a, b: a < b,
    'le': lambda a, b: a <= b,
    'eq': lambda a, b: a == b,
    'ne': lambda a, b: a != b,
    
    # 삼항 연산자
    'if': lambda cond, t_val, f_val: np.where(cond, t_val, f_val)
}
