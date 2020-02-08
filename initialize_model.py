from objects.trader import *
from objects.orderbook import *
import random
import numpy as np
from functions.helpers import calculate_covariance_matrix, div0


def init_objects(parameters, seed):
    """
    Init object for the distribution version of the model
    :param parameters:
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    traders = []
    n_traders = parameters["n_traders"]

    weight_f = (1 - parameters['strat_share_chartists']) * (1 - parameters['w_random'])
    weight_c = parameters['strat_share_chartists'] * (1 - parameters['w_random'])

    f_points = int(weight_f * 100 * n_traders)
    c_points = int(weight_c * 100 * n_traders)
    r_points = int(parameters['w_random'] * 100 * n_traders)

    # create list of strategy points, shuffle it and divide in equal parts
    strat_points = ['f' for f in range(f_points)] + ['c' for c in range(c_points)] + ['r' for r in range(r_points)]
    random.shuffle(strat_points)
    agent_points = np.array_split(strat_points, n_traders)

    max_horizon = parameters['horizon'] * 2  # this is the max horizon of an agent if 100% fundamentalist
    historical_stock_returns = np.random.normal(0, parameters["std_fundamental"], max_horizon)

    for idx in range(n_traders):
        weight_fundamentalist = list(agent_points[idx]).count('f') / float(len(agent_points[idx]))
        weight_chartist = list(agent_points[idx]).count('c') / float(len(agent_points[idx]))
        weight_random = list(agent_points[idx]).count('r') / float(len(agent_points[idx]))

        init_stocks = int(np.random.uniform(0, parameters["init_stocks"]))
        init_money = np.random.uniform(0, (parameters["init_stocks"] * parameters['fundamental_value']))

        if weight_random < 1.0:
            c_share_strat = div0(weight_chartist, (weight_fundamentalist + weight_chartist))
        else:
            c_share_strat = 0.0

        # initialize co_variance_matrix
        init_covariance_matrix = calculate_covariance_matrix(historical_stock_returns, parameters["std_fundamental"])

        lft_vars = TraderVariables(weight_fundamentalist, weight_chartist, weight_random, c_share_strat,
                                   init_money, init_stocks, init_covariance_matrix,
                                   parameters['fundamental_value'])

        # determine heterogeneous horizon and risk aversion based on
        individual_horizon = np.random.randint(10, parameters['horizon'])

        individual_risk_aversion = abs(np.random.normal(parameters["base_risk_aversion"], parameters["base_risk_aversion"] / 5.0))#parameters["base_risk_aversion"] * relative_fundamentalism
        individual_learning_ability = min(abs(np.random.normal(parameters['average_learning_ability'], 0.1)), 1.0) #TODO what to do with std_dev

        lft_params = TraderParameters(individual_horizon, individual_risk_aversion,
                                      individual_learning_ability, parameters['spread_max'])
        lft_expectations = TraderExpectations(parameters['fundamental_value'])
        traders.append(Trader(idx, lft_vars, lft_params, lft_expectations))

    # Add market maker with 100k money and 1k stocks
    mm_tradervariables = TraderVariables(weight_fundamentalist=0, weight_chartist=0, weight_random=0,
                                         c_share_strat=0, money=10000000, stocks=100000, covariance_matrix=0,
                                         init_price=parameters['fundamental_value'])
    market_maker = MarketMaker(0, mm_tradervariables)

    orderbook = LimitOrderBook(parameters['fundamental_value'], parameters["std_fundamental"],
                               max_horizon,
                               parameters['ticks'])

    # initialize order-book returns for initial variance calculations
    orderbook.returns = list(historical_stock_returns)

    return traders, orderbook, market_maker
