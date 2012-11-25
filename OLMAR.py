#
# Algorithm based on this publication:
# http://alphapowertrading.com/papers/OnLinePortfolioSelectionMovingAverageReversion.pdf
#

import numpy as np

def initialize(context):

    context.stocks = [sid(700),sid(8229),sid(4283),sid(1267),sid(698),sid(3951),sid(5923),sid(3496),sid(7792),sid(7883)]
    context.total_investment = 100000.0
    context.price = {}
    context.eps = 10
    
    context.init = 0

def handle_data(context,data):

    x_bar = 0.0
    m = len(context.stocks)
    x_tilde = np.zeros(m)
    b_t = np.zeros(m)
    b = np.zeros(m)
    
    #if context.portfolio.positions_value > 0.0:
        #init = 1https://www.quantopian.com/algorithms
    #else:
        #init = 0
    
    log.debug(context.init)
    
    i = 0
    # find relative moving average price for each security
    for stock in context.stocks:
        price = data[stock].price
        x_tilde[i] = float(data[stock].mavg(3))/price
        if context.init > 0:
            b_t[i] = context.portfolio.positions[stock].amount * price
            b_t[i] = b_t[i]/context.portfolio.positions_value
            log.debug(b_t[i])
        else:
            b_t[i] = 1.0/m
        x_bar = x_bar + x_tilde[i]
        i = i + 1

    x_bar = x_bar/m #average predicted relative price
    
    log.debug(x_tilde)
    log.debug(x_bar)
    log.debug(x_tilde-x_bar)
    
    sq_norm = (np.linalg.norm((x_tilde-x_bar)))**2
    dot_prod = np.dot(b_t,x_tilde)

    lam = max(0,(context.eps-dot_prod)/sq_norm)
    
    db = lam*(x_tilde-x_bar)
    log.debug(db)
    log.debug(np.dot(np.ones(m),db))
    
    b = b_t + lam*(x_tilde-x_bar)
    log.debug(b)
    
    if context.init > 0:
        b_norm = simplex_projection(b)
    else:
        b_norm = b_t
    
    if context.init > 0:
        positions_value = context.portfolio.positions_value
    else:
        positions_value = context.total_investment
    
    log.debug(b_norm)
    
    i = 0
    #rebalance portfolio
    for stock in context.stocks:
         n = b_norm[i]*positions_value/data[stock].price
         dn = n - context.portfolio.positions[stock].amount
         #order(stock,dn)
         order(stock,100)
         log.debug(dn)
         i = i + 1

    context.init = 1

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
