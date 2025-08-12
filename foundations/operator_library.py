# foundations/operator_library.py (수정 완료)

# """
# AlphaAgent의 팩터 연산에 사용될 함수들을 정의합니다.
# 이 파일은 clients/backtester_client.py의 _execute_ast 메서드에서 사용됩니다.
# """
# import numpy as np
# import pandas as pd

# # 기본적인 연산자들을 딕셔너리 형태로 정의합니다.
# # 이 딕셔너리의 키(예: 'ts_mean', 'add')는 FactorParser가 생성한 AST의 op_name과 일치해야 합니다.

# OPERATORS = {
#     # 횡단면(Cross-sectional) 연산자
#     # 횡단면(Cross-sectional) 연산자
#     'rank': lambda df: df.rank(axis=1, pct=True),
#     'scale': lambda df: df.div(df.abs().sum(axis=1), axis=0),

#     # 시계열(Time-series) 연산자
#     'ts_mean': lambda df, window: df.rolling(window, min_periods=1).mean(),
#     'ts_std': lambda df, window: df.rolling(window, min_periods=1).std(),
#     'ts_rank': lambda df, window: df.rolling(window, min_periods=1).rank(pct=True),
#     'delay': lambda df, period: df.shift(period),
#     'delta': lambda df, period: df.diff(period),
#     'ts_min': lambda df, window: df.rolling(window, min_periods=1).min(),
#     'ts_max': lambda df, window: df.rolling(window, min_periods=1).max(),
#     'correlation': lambda df1, df2, window: df1.rolling(window, min_periods=1).corr(df2),
#     'covariance': lambda df1, df2, window: df1.rolling(window, min_periods=1).cov(df2),
    
#     # QLib에서 누락된 연산자 추가
#     'count': lambda df, window: df.rolling(window, min_periods=1).count(),
#     'sum': lambda df, window: df.rolling(window, min_periods=1).sum(),
#     'median': lambda df, window: df.rolling(window, min_periods=1).median(),
#     'skew': lambda df, window: df.rolling(window, min_periods=1).skew(),
#     'kurt': lambda df, window: df.rolling(window, min_periods=1).kurt(),
#     'wma': lambda df, window: df.ewm(span=window, adjust=False).mean(), # WMA는 EMA와 유사한 방식으로 구현

#     # 산술 연산자
#     'add': lambda a, b: a + b,
#     'subtract': lambda a, b: a - b,
#     'multiply': lambda a, b: a * b,
#     'divide': lambda a, b: a / b.replace(0, np.nan),
#     'power': lambda a, b: a ** b,

#     # 단항 연산자
#     'negate': lambda a: -a,
#     'abs': lambda a: np.abs(a),
#     'log': lambda a: np.log(a.replace(0, np.nan)),
#     'sign': lambda a: np.sign(a),

#     # 논리 연산자
#     'and': lambda a, b: a & b,
#     'or': lambda a, b: a | b,
#     'not': lambda a: ~a,
    
#     # 비교 연산자
#     'gt': lambda a, b: a > b,
#     'ge': lambda a, b: a >= b,
#     'lt': lambda a, b: a < b,
#     'le': lambda a, b: a <= b,
#     'eq': lambda a, b: a == b,
#     'ne': lambda a, b: a != b,
    
#     # 삼항 연산자
#     'if': lambda cond, t_val, f_val: np.where(cond, t_val, f_val)
# }


# foundations/operator_library.py

import numpy as np
import pandas as pd

OPERATORS = {
    'rank': lambda series: series.groupby(level='date').rank(pct=True),
    'scale': lambda series: series.groupby(level='date').transform(lambda x: x / (x.abs().sum() if x.abs().sum() != 0 else 1)),

    'ts_mean': lambda series, window: series.groupby(level='ticker').rolling(int(window)).mean().reset_index(level=0, drop=True),
    'ts_std': lambda series, window: series.groupby(level='ticker').rolling(int(window)).std().reset_index(level=0, drop=True),
    'ts_rank': lambda series, window: series.groupby(level='ticker').rolling(int(window)).rank(pct=True).reset_index(level=0, drop=True),
    'delay': lambda series, period: series.groupby(level='ticker').shift(int(period)),
    'delta': lambda series, period: series.groupby(level='ticker').diff(int(period)),
    'ts_min': lambda series, window: series.groupby(level='ticker').rolling(int(window)).min().reset_index(level=0, drop=True),
    'ts_max': lambda series, window: series.groupby(level='ticker').rolling(int(window)).max().reset_index(level=0, drop=True),
    
    # 💡 correlation, covariance 함수를 apply를 사용해 안정적으로 변경
    'correlation': lambda series1, series2, window: series1.groupby(level='ticker').rolling(int(window)).apply(lambda x: x.corr(series2.loc[x.index]), raw=False).reset_index(level=0, drop=True),
    'covariance': lambda series1, series2, window: series1.groupby(level='ticker').rolling(int(window)).apply(lambda x: x.cov(series2.loc[x.index]), raw=False).reset_index(level=0, drop=True),
    
    'count': lambda series, window: series.groupby(level='ticker').rolling(int(window)).count().reset_index(level=0, drop=True),
    'sum': lambda series, window: series.groupby(level='ticker').rolling(int(window)).sum().reset_index(level=0, drop=True),
    'median': lambda series, window: series.groupby(level='ticker').rolling(int(window)).median().reset_index(level=0, drop=True),
    'skew': lambda series, window: series.groupby(level='ticker').rolling(int(window)).skew().reset_index(level=0, drop=True),
    'kurt': lambda series, window: series.groupby(level='ticker').rolling(int(window)).kurt().reset_index(level=0, drop=True),
    'wma': lambda series, window: series.groupby(level='ticker').ewm(span=int(window), adjust=False).mean().reset_index(level=0, drop=True),

    'add': lambda a, b: a + b,
    'subtract': lambda a, b: a - b,
    'multiply': lambda a, b: a * b,
    'divide': lambda a, b: a / b.replace(0, 1e-6),
    'power': lambda a, b: np.power(a, b),

    'negate': lambda a: -a,
    'abs': lambda a: np.abs(a),
    'log': lambda a: np.log(a.replace(0, 1e-6).abs()),
    'sign': lambda a: np.sign(a),

    'and': lambda a, b: a & b,
    'or': lambda a, b: a | b,
    'not': lambda a: ~a,
    
    'gt': lambda a, b: a > b,
    'ge': lambda a, b: a >= b,
    'lt': lambda a, b: a < b,
    'le': lambda a, b: a <= b,
    'eq': lambda a, b: a == b,
    'ne': lambda a, b: a != b,
    
    'if': lambda cond, t_val, f_val: np.where(cond, t_val, f_val)
}
