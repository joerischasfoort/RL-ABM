import random
import numpy as np
from functions.portfolio_optimization import *
from functions.helpers import calculate_covariance_matrix, div0, ornstein_uhlenbeck_evolve


def ABM_model(traders, orderbook, market_maker, parameters, seed=1):
    """
    The main model function of distribution model where trader stocks are tracked.
    :param traders: list of Agent objects
    :param orderbook: object Order book
    :param parameters: dictionary of parameters
    :param seed: integer seed to initialise the random number generators
    :return: list of simulated Agent objects, object simulated Order book
    """
    random.seed(seed)
    np.random.seed(seed)
    fundamental = [parameters["fundamental_value"]]
    orderbook.tick_close_price.append(fundamental[-1])

    traders_by_wealth = [t for t in traders]
    initial_mm_wealth = market_maker.var.wealth[0]

    for tick in range(parameters['horizon'] + 1, parameters["ticks"] + parameters['horizon'] + 1): # for init history
        if tick == parameters['horizon'] + 1:
            print('Start of simulation ', seed)

        # update money and stocks history for agents
        for trader in traders:
            trader.var.money.append(trader.var.money[-1])
            trader.var.stocks.append(trader.var.stocks[-1])
            trader.var.wealth.append(trader.var.money[-1] + trader.var.stocks[-1] * orderbook.tick_close_price[-1])

        # Keep track of market maker metrics
        market_maker.var.money.append(market_maker.var.money[-1])
        market_maker.var.stocks.append(market_maker.var.stocks[-1])
        market_maker.var.wealth.append(market_maker.var.money[-1] + market_maker.var.stocks[-1] * orderbook.tick_close_price[-1])


        # sort the traders by wealth to
        traders_by_wealth.sort(key=lambda x: x.var.wealth[-1], reverse=True)

        # evolve the fundamental value via random walk process
        fundamental.append(max(
            ornstein_uhlenbeck_evolve(parameters["fundamental_value"], fundamental[-1], parameters["std_fundamental"],
                                      parameters['mean_reversion'], seed), 0.1))

        # allow for multiple trades in one day
        for turn in range(parameters["trades_per_tick"]):
            # select random sample of active traders
            active_traders = random.sample(traders, int((parameters['trader_sample_size'])))

            mid_price = np.mean([orderbook.highest_bid_price, orderbook.lowest_ask_price])
            fundamental_component = np.log(fundamental[-1] / mid_price)

            orderbook.returns[-1] = (mid_price - orderbook.tick_close_price[-2]) / orderbook.tick_close_price[-2]
            chartist_component = np.cumsum(orderbook.returns[:-len(orderbook.returns) - 1:-1]
                                           ) / np.arange(1., float(len(orderbook.returns) + 1))


            # Market maker posts its asks and/or bids given current orderbook and a "reference price"
            market_maker.post_orders(orderbook, fundamental[0])

            for trader in active_traders:
                # Cancel any active orders
                if trader.var.active_orders:
                    for order in trader.var.active_orders:
                        orderbook.cancel_order(order)
                    trader.var.active_orders = []

                # Update trader specific expectations
                noise_component = parameters['std_noise'] * np.random.randn()

                # Expectation formation
                trader.exp.returns['stocks'] = (
                        trader.var.weight_fundamentalist[-1] * np.divide(1, float(trader.par.horizon) * parameters["fundamentalist_horizon_multiplier"]) * fundamental_component +
                        trader.var.weight_chartist[-1] * chartist_component[trader.par.horizon - 1] +
                        trader.var.weight_random[-1] * noise_component)
                fcast_price = mid_price * np.exp(trader.exp.returns['stocks'])
                trader.var.covariance_matrix = calculate_covariance_matrix(orderbook.returns[-trader.par.horizon:],
                                                                           parameters["std_fundamental"])

                # employ portfolio optimization algo
                ideal_trader_weights = portfolio_optimization(trader, tick)

                # Determine price and volume
                trader_price = np.random.normal(fcast_price, trader.par.spread)
                position_change = (ideal_trader_weights['stocks'] * (trader.var.stocks[-1] * trader_price + trader.var.money[-1])
                          ) - (trader.var.stocks[-1] * trader_price)
                volume = int(div0(position_change, trader_price))

                # Trade:
                if volume > 0:
                    bid = orderbook.add_bid(trader_price, volume, trader)
                    trader.var.active_orders.append(bid)
                elif volume < 0:
                    ask = orderbook.add_ask(trader_price, -volume, trader)
                    trader.var.active_orders.append(ask)

            # Match orders in the order-book
            while True:
                matched_orders = orderbook.match_orders()
                if matched_orders is None:
                    break
                # execute trade
                matched_orders[3].owner.sell(matched_orders[1], matched_orders[0] * matched_orders[1])
                matched_orders[2].owner.buy(matched_orders[1], matched_orders[0] * matched_orders[1])

        # Clear and update order-book history
        orderbook.cleanse_book()
        orderbook.fundamental = fundamental

    return traders, orderbook, market_maker
