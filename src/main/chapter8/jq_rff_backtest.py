import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import jqdata
from jqlib.technical_analysis import *
import datetime

# 在JoinQuant平台进行回测


def initialize(context):
    set_params()
    set_backtest()
    set_variables()


def set_params():
    # 调仓频率
    g.tc = 10
    # 投资组合最大持股数
    g.stocknum = 6
    # 设置初始的收益为
    g.ret = -0.05


def set_backtest():
    set_benchmark("000001.XSHG")
    set_option("use_real_price", True)
    log.set_level("order", "error")


def set_variables():
    # 交易天数
    g.days = 0
    g.if_trade = False


def before_trading_start(context):
    """盘前的准备工作"""
    # 每10天调仓一次
    if g.days % g.tc == 0:
        g.if_trade = True
        set_slip_fee(context)
        g.stocks = get_index_stocks("000300.XSHG")
        g.feasible_stocks = set_feasible_stocks(g.stocks, context)
        g.days += 1


def set_feasible_stocks(initial_stocks, context):
    paused_info = []
    current_data = get_current_data()
    for i in initial_stocks:
        paused_info.append(current_data[i].paused)
    df_paused_info = pd.DataFrame({"paused_info": paused_info}, index=initial_stocks)
    stock_list = list(df_paused_info.index[df_paused_info.paused_info == False])
    return stock_list


def set_slip_fee(context):
    set_slippage(FixedSlippage(0.02))
    set_commission(PerTrade(buy_cost=0.0002, sell_cost=0.0002, min_cost=5))


def handle_data(context, data):
    if g.if_trade == True:
        list_to_buy = stocks_to_buy(context)
        list_to_sell = stocks_to_sell(context, list_to_buy)
        sell_operation(list_to_sell)
        buy_operation(context, list_to_buy)
    g.if_trade = False


def get_rff(context, stock_list):
    """最重要的前7个因子去预测股票市值，找出被低估的股票

    Args:
        context (_type_): 交易的上下文
        stock_list (_type_): 需要使用随机森林处理的股票列表

    Returns:
        factor: 股票的预测的市值和实际市值的差值
    """
    today = context.current_dt
    delta = datetime.timedelta(days=1)
    yesterday = today - delta
    q = query(
        valuation.code,
        valuation.market_cap,
        # balance.total_liability,
        valuation.pe_ratio,
        indicator.inc_total_revenue_year_on_year,
        balance.total_current_assets - balance.total_current_liability,
    ).filter(valuation.code.in_(stock_list))
    dataset = get_fundamentals(q)
    dataset["平均差"] = list(DMA(dataset.code, yesterday)[0].values())
    dataset["乖离率"] = list(BIAS(dataset.code, yesterday)[0].values())
    dataset["动量线"] = list(MTM(dataset.code, yesterday).values())
    dataset.index = dataset.code
    dataset.drop("code", axis=1, inplace=True)
    dataset.fillna(0, inplace=True)
    X = dataset.drop("market_cap", axis=1)
    y = dataset["market_cap"]
    reg = RandomForestRegressor(random_state=20)
    reg.fit(X, y)
    factor = y - pd.DataFrame(reg.predict(X), index=y.index, columns=["market_cap"])
    factor = factor.sort_index(by="market_cap", ascending=True)
    return factor


def stocks_to_buy(context):
    """使用随机森林在沪深300中筛选出被低估的股票

    Args:
        context (_type_): _description_

    Returns:
        _type_: _description_
    """
    list_to_buy = []
    day1 = context.current_dt
    day2 = day1 - datetime.timedelta(days=5)
    hs300_close = get_price('000300.XSHG', day2, day1, fq='pre')['close']
    hs300_ret = hs300_close[-1]/hs300_close[0] - 1
    if hs300_ret > g.ret:
        factor = get_rff(context, g.feasible_stocks)
        list_to_buy = list(factor.index[:g.stocknum])
    else:
        pass
    return list_to_buy


def stocks_to_sell(context, list_to_buy):
    """获取需要卖出的股票

    Args:
        context (_type_): _description_
        list_to_buy (_type_): _description_

    Returns:
        _type_: _description_
    """
    list_to_sell = []
    day1 = context.current_dt
    day2 = day1 - datetime.timedelta(days=5)
    hs300_close = get_price('000300.XSHG', day2, day1, fq='pre')['close']
    hs300_ret = hs300_close[-1]/hs300_close[0] - 1
    for stock_sell in context.portfolio.positions:
        if hs300_ret <= g.ret:
            list_to_sell.append(stock_sell)
        else:
            if context.portfolio.positions[stock_sell].price^context.portfolio.positions[stock_sell].avg_cost<0.95 \
                or stock_sell not in list_to_buy:
                    list_to_sell.append(stock_sell)
    return list_to_sell
     

def sell_operation(list_to_sell):
    for stock_sell in list_to_sell:
        # order_target_value(stock_sell, 0)
        order_target_value(stock_sell, context.portfolio.positions[stock_sell].num*0.75)

def buy_operation(context, list_to_buy):
    if len(context.portfolio.positions) < g.stocknum:
        num = g.stocknum - len(context.portfolio.positions)
        cash = context.portfolio.cash / num
    else:
        cash = 0
        num = 0
    for stock_buy in list_to_buy[:num+1]:
        order_target_value(stock_buy, cash)
        num -= 1
        if num == 0:
            break
        else:
            pass
