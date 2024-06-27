import jqdata
from jqlib.technical_analysis import *

# 选择沪深300指数成分为股票池
stocks = get_index_stocks("000300.XSHG")
q = query(
    valuation.code,
    valuation.market_cap,
    balance.total_current_assets - balance.total_current_liability,
    balance.total_liability,
    balance.total_liability / balance.equities_parent_company_owners,
    (balance.total_assets - balance.total_current_assets) / balance.total_assets,
    balance.equities_parent_company_owners / balance.total_assets,
    indicator.inc_total_revenue_year_on_year,
    valuation.turnover_ratio,
    valuation.pe_ratio,
    valuation.pb_ratio,
    valuation.ps_ratio,
    indicator.roa,
).filter(valuation.code.in_(stocks))
df = get_fundamentals(q, date=None)
df.columns = [
    "code",
    "市值",
    "净运营资本",
    "净债务",
    "产权比率",
    "非流动资产比率",
    "股东权益比率",
    "营收增长率",
    "换手率",
    "PE",
    "PB",
    "PS",
    "总资产收益率",
]
df.head()


import datetime

df.index = df.code.values
del df["code"]
today = datetime.datetime.today()
delta50 = datetime.timedelta(days=50)
delta1 = datetime.timedelta(days=1)
delta2 = datetime.timedelta(days=2)

history = today - delta50
yesterday = today - delta1
two_days_ago = today - delta2

# 获取股票的动量线、成交量、累计能量线、平均差、指数移动平均、移动平均、乖离率等因子
# 时间范围都设置为10天
df["动量线"] = list(
    MTM(
        df.index,
        two_days_ago,
        timeperiod=10,
        unit="1d",
        include_now=True,
        fq_ref_date=None,
    ).values()
)
df["成交量"] = list(
    VOL(df.index, two_days_ago, M1=10, unit="1d", include_now=True, fq_ref_date=None)[
        0
    ].values()
)
df["累计能量线"] = list(OBV(df.index, check_date=two_days_ago, timeperiod=10).values())
df["平均差"] = list(
    DMA(df.index, two_days_ago, N1=10, unit="1d", include_now=True, fq_ref_date=None)[
        0
    ].values()
)
df["指数移动平均"] = list(
    EMA(
        df.index,
        two_days_ago,
        timeperiod=10,
        unit="1d",
        include_now=True,
        fq_ref_date=None,
    ).values()
)
df["移动平均"] = list(
    MA(
        df.index,
        two_days_ago,
        timeperiod=10,
        unit="1d",
        include_now=True,
        fq_ref_date=None,
    ).values()
)
df["乖离率"] = list(
    BIAS(df.index, two_days_ago, N1=10, unit="1d", include_now=True, fq_ref_date=None)[
        0
    ].values()
)
df.fillna(0, inplace=True)
df.head()

import numpy as np

df["close"] = list(
    get_price(stocks, end_date=yesterday, count=1, fq="pre", panel=False)["close"]
)
df["close2"] = list(
    get_price(stocks, end_date=hisotry, count=1, fq="pre", panel=False)["close"]
)
# 获取每支股票近50天的收益率
df["return"] = df["close1"] / df["close2"] - 1
df["signal"] = np.where(df["return"] < df["return"].mean(), 0, 1)
df.head()


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X = df.drop(["close1", "close2", "return", "signal"], axis=1)
y = df["signal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier(random_state=1000)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

factor_weight = pd.DataFrame({'features':list(X.columns), 'importance':clf.feature_importances_}).sort_values(by='importance', ascending=False)
factor_weight
