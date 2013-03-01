import numpy as np
import datetime

def initialize(context):
    context.eps = 2  #change epsilon here
    context.init = False
    context.counter = 0
    context.stocks = []
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.25, price_impact=0, delay=datetime.timedelta(minutes=0)))
    set_commission(commission.PerShare(cost=0))
    set_universe(universe.DollarVolumeUniverse(floor_percentile=98.0, ceiling_percentile=100.0))
    
def handle_data(context, data):    
    context.counter += 1
    if context.counter <= 5:
        return
    
    context.stocks = [sid for sid in data]
    m = len(context.stocks)
    
    if not context.init:
        context.b_t = np.ones(m) / m
        rebalance_portfolio(context, data, context.b_t)
        context.init = True
        return

    if len(context.b_t) > m:
        # need to decrease portfolio vector
        context.b_t = context.b_t[:m]
    elif len(context.b_t) < m:
        # need to grow portfolio vector
        len_bt = len(context.b_t)
        context.b_t = np.concatenate([context.b_t, np.ones(m-len_bt) / m])
    
    assert len(context.b_t) == m
    
    x_tilde = np.zeros(m)

    b = np.zeros(m)

    # find relative moving average price for each security
    for i, stock in enumerate(context.stocks):
        price = data[stock].price
        x_tilde[i] = data[stock].mavg(5) / price
        
    ###########################
    # Inside of OLMAR (algo 2)
    x_bar = x_tilde.mean()
        
    # market relative deviation
    mark_rel_dev = x_tilde - x_bar

    # Expected return with current portfolio
    exp_return = np.dot(context.b_t, x_tilde)
    log.debug("Expected Return: {exp_return}".format(exp_return=exp_return))
    weight = context.eps - exp_return
    log.debug("Weight: {weight}".format(weight=weight))
    variability = (np.linalg.norm(mark_rel_dev))**2
    log.debug("Variability: {norm}".format(norm=variability))
    # test for divide-by-zero case
    if variability == 0.0:
        step_size = 0 # no portolio update
    else:
        step_size = max(0, weight/variability)
    log.debug("Step-size: {size}".format(size=step_size))
    log.debug("Market relative deviation:")
    log.debug(mark_rel_dev)
    log.debug("Weighted market relative deviation:")
    log.debug(step_size*mark_rel_dev)
    b = context.b_t + step_size*mark_rel_dev
    b_norm = simplex_projection(b)
    #np.testing.assert_almost_equal(b_norm.sum(), 1)
        
    rebalance_portfolio(context, data, b_norm)
        
    # Predicted return with new portfolio
    pred_return = np.dot(b_norm, x_tilde)
    log.debug("Predicted return: {pred_return}".format(pred_return=pred_return))
    
    # Make sure that we actually optimized our objective
    #assert exp_return-.001 <= pred_return, "{new} <= {old}".format(new=exp_return, old=pred_return)
    # update portfolio
    context.b_t = b_norm
    
def rebalance_portfolio(context, data, desired_port):
    print 'desired'
    print desired_port
    desired_amount = np.zeros_like(desired_port)
    current_amount = np.zeros_like(desired_port)
    prices = np.zeros_like(desired_port)
        
    if context.init:
        positions_value = context.portfolio.starting_cash
    else:
        positions_value = context.portfolio.positions_value + context.portfolio.cash
        
    
    for i, stock in enumerate(context.stocks):
        current_amount[i] = context.portfolio.positions[stock].amount
        prices[i] = data[stock].price
    
    desired_amount = np.round(desired_port * positions_value / prices)
    diff_amount = desired_amount - current_amount
    for i, stock in enumerate(context.stocks):
        order(stock, diff_amount[i]) #order_stock

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
array([ 0.33333333, 0.23333333, 0. , 0.43333333])
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
