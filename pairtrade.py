"""
Ports Thomas Wiecki's pairtrade algorithm from PyData 2012.

Original source:
https://github.com/quantopian/zipline/blob/master/zipline/examples/pairtrade.py
"""

import numpy as np
from statsmodels import api as sm

@batch_transform(refresh_period=100, days=100)
def ols_transform(datapanel, sid1, sid2):
    """
    Computes regression coefficient (slope and intercept)
    via Ordinary Least Squares between two SIDs.
    """
    p0 = datapanel.price[sid1]

    p1 = sm.add_constant(datapanel.price[sid2])
    slope, intercept = sm.OLS(p0, p1).fit().params

    return slope, intercept

    """Pairtrading relies on cointegration of two stocks.

    The expectation is that once the two stocks drifted apart
    (i.e. there is spread), they will eventually revert again. Thus,
    if we short the upward drifting stock and long the downward
    drifting stock (in short, we buy the spread) once the spread
    widened we can sell the spread with profit once they converged
    again. A nice property of this algorithm is that we enter the
    market in a neutral position.

    This specific algorithm tries to exploit the cointegration of
    Pepsi and Coca Cola by estimating the correlation between the
    two. Divergence of the spread is evaluated by z-scoring.
    """

def initialize(context):
    context.spreads = []
    context.zscores = []
    context.invested = 0
    context.window_length = 100


def handle_data(context, data):
    ######################################################
    # 1. Compute regression coefficients between PEP and KO
    params = ols_transform(data, sid(5885), sid(4283))
    if params is None:
        return
    slope, intercept = params

    ######################################################
    # 2. Compute spread and zscore
    zscore = compute_zscore(context, data, slope, intercept)
    context.zscores.append(zscore)

    ######################################################
    # 3. Place orders
    place_orders(context, data, zscore)


def compute_zscore(context, data, slope, intercept):
    """1. Compute the spread given slope and intercept.
       2. zscore the spread.
    """
    spread = (data[sid(5885)].price - \
              (slope * data[sid(4283)].price + intercept))
    context.spreads.append(spread)
    spread_wind = context.spreads[-context.window_length:]
    zscore = (spread - np.mean(spread_wind)) / np.std(spread_wind)
    return zscore


def place_orders(context, data, zscore):
    """Buy spread if zscore is > 2, sell if zscore < .5.
    """

    if zscore >= 2.0 and not context.invested:
        log.info("buying over zscore")
        order(sid(5885), int(100000 / data[sid(5885)].price))
        order(sid(4283), -int(100000 / data[sid(4283)].price))
        context.invested = True
    elif zscore <= -2.0 and not context.invested:
        log.info("buying with under zscore")
        order(sid(4283), -int(100000 / data[sid(4283)].price))
        order(sid(5885), int(100000 / data[sid(5885)].price))
        context.invested = True
    elif abs(zscore) < .5 and context.invested:
        sell_spread(context)
        context.invested = False

def sell_spread(context):
    """
    decrease exposure, regardless of position long/short.
    buy for a short position, sell for a long.
    """
    ko_amount = context.portfolio.positions[sid(4283)].amount
    order(sid(4283), -1 * ko_amount)
    pep_amount = context.portfolio.positions[sid(5885)].amount
    order(sid(4283), -1 * pep_amount)
