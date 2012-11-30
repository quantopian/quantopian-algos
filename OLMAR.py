import numpy as np

def initialize(context):

    context.stocks = [sid(700),sid(8229),sid(4283),sid(1267),sid(698),sid(3951),sid(5923),sid(3496),sid(7792),sid(7883)]
    context.m = len(context.stocks)

    context.total_investment = 100000.0
    context.price = {}
    context.eps = 10
    context.b_t = np.ones(context.m) / context.m
    log.debug(context.b_t)
    context.init = False

def handle_data(context, data):
    if not context.init:
        context.init = True
        rebalance_portfolio(context, data, context.b_t)
        return

    m = context.m

    x_bar = 0.0
    x_tilde = np.zeros(m)

    b = np.zeros(m)

    log.debug(context.init)

    # find relative moving average price for each security
    for i, stock in enumerate(context.stocks):
        price = data[stock].price
        x_tilde[i] = float(data[stock].mavg(3))/price
        context.b_t[i] = context.portfolio.positions[stock].amount * price
        context.b_t[i] = context.b_t[i]/context.portfolio.positions_value
        log.debug(context.b_t[i])

        x_bar = x_bar + x_tilde[i]


    x_bar = x_bar/m #average predicted relative price

    log.debug(x_tilde)
    log.debug(x_bar)
    log.debug(x_tilde-x_bar)

    ###########################
    # Inside of OLMAR (algo 2)

    # Calculate terms for lambda (lam)
    sq_norm = (np.linalg.norm((x_tilde-x_bar)))**2
    dot_prod = np.dot(context.b_t, x_tilde)

    lam = max(0,(context.eps-dot_prod)/sq_norm)


    db = lam*(x_tilde-x_bar)
    log.debug(db)
    log.debug(np.dot(np.ones(m),db))

    b = context.b_t + lam*(x_tilde-x_bar)
    log.debug(b)

    b_norm = b #simplex_projection(b)

    log.debug(b_norm)

    rebalance_portfolio(context, data, b_norm)

def rebalance_portfolio(context, data, desired_port):
    #rebalance portfolio
    cur_port = np.zeros_like(desired_port)
    prices = np.zeros_like(desired_port)
    for i, stock in enumerate(context.stocks):
        cur_port[i] = context.portfolio.positions[stock].amount / context.portfolio.starting_cash
        prices[i] = data[stock].price

    diff_port = desired_port - cur_port
    diff_cash = diff_port * context.portfolio.starting_cash

    order_size = diff_cash / prices

    for stock, order_stock in zip(context.stocks, order_size):
        order(stock, order_stock)#order_stock)


def simplex_projection(v, b=1):
    """Projection vectors to the simplex domain

    Implemented according to the paper: Efficient projections onto the
    l1-ball for learning in high dimensions, John Duchi, et al. ICML 2008.
    Implementation Time: 2011 June 17 by Bin@libin AT pmail.ntu.edu.sg
    Optimization Problem: min_{w}\| w - v \|_{2}^{2}
                          s.t. sum_{i=1}^{m}=z, w_{i}\geq 0

    Input: A vector v \in R^{m}, and a scalar z > 0 (default=1)
    Output: Projection vector w

    :Example:
    >>> proj = simplex_projection([.4 ,.3, -.4, .5])
    >>> print proj
    array([ 0.33333333,  0.23333333,  0.        ,  0.43333333])
    >>> print proj.sum()
    1.0

    Original matlab implementation: John Duchi (jduchi@cs.berkeley.edu)
    Python-port: Copyright 2012 by Thomas Wiecki (thomas.wiecki@gmail.com).
    """

    v = np.asarray(v)
    p = len(v)

    # Sort v into u in descending order
    v = (v > 0) * v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)

    rho = np.where(u > (sv - b) / np.arange(1, p+1))[0][-1]
    theta = np.max([0, (sv[rho] - b) / (rho+1)])
    w = (v - theta)
    w[w<0] = 0
    return w