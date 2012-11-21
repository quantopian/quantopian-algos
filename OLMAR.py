#
# Algorithm based on this publication:
# http://alphapowertrading.com/papers/OnLinePortfolioSelectionMovingAverageReversion.pdf
#

import numpy as np

def initialize(context):

    context.stocks = [sid(8554),sid(19920),sid(22739)]
    context.price = {}
    context.x_tilde = {}
    context.b_t = {}
    #context.b = {}
    #context.b_norm = {}

    context.eps = 1.1

def handle_data(context,data):

    x_bar = 0.0
    m = len(context.stocks)
    sq_norm = 0.0
    dot_prod = 0.0
    b = []
    #b_tot = 0.0

    # find relative moving average price for each security
    for stock in context.stocks:
        price = data[stock].price
        x_tilde = float(data[stock].mavg(3))/price
        if context.portfolio.positions_value > 0.0:
            b_t = context.portfolio.positions[stock].amount * price
            b_t = b_t/context.portfolio.positions_value
        else:
            b_t = 1.0/m
        x_bar = x_bar + x_tilde

    x_bar = x_bar/m  # average predicted relative price

    for stock in context.stocks:
        sq_norm = sq_norm + (x_tilde-x_bar)**2
        dot_prod = dot_prod + b_t*x_tilde

    lam = max(0,(context.eps-dot_prod)/sq_norm)

    for stock in context.stocks:
        b.append(b_t + lam*(x_tilde-x_bar))
        #b_tot = b_tot + b

    #for stock in context.stocks:
        #b_norm = b/b_tot  # new portfolio

    #log.debug(len(b))

    #for stock in context.stocks:
        #log.debug(b)

    log.debug(b)
    b_norm = simplex_projection(b)
    log.debug(b_norm)

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
