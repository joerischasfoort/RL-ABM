import numpy as np
from objects.learningagent import LearningAgent


class Trader:
    """Class holding low frequency trader properties"""
    def __init__(self, name, variables, parameters, expectations):
        """
        Initialize trader class
        :param name: integer number which will be the name of the trader
        :param variables: object Tradervariables
        :param parameters: object TraderParameters
        :param expectations: object TraderExpectations
        """
        self.name = name
        self.var = variables
        self.par = parameters
        self.exp = expectations
        self.metrics = dict.fromkeys(['money', 'inventory', 'inventory_value', 'wealth'], [])

    def __repr__(self):
        """
        :return: String representation of the trader
        """
        return 'Trader' + str(self.name)

    def sell(self, amount, price):
        """
        Sells `amount` of stocks for a total of `price`
        :param amount: int Number of stocks sold.
        :param price: float Total price for stocks.
        :return: -
        """
        if self.var.stocks[-1] < amount:
            raise ValueError("not enough stocks to sell this amount")
        self.var.stocks[-1] -= amount
        self.var.money[-1] += price

    def buy(self, amount, price):
        """
        Buys `amount` of stocks for a total of `price`
        :param amount: int number of stocks bought.
        :param price: float total price for stocks.
        :return: -
        """
        if self.var.money[-1] < price:
            raise ValueError("not enough money to buy this amount of stocks")

        self.var.stocks[-1] += amount
        self.var.money[-1] -= price


class TraderVariables:
    """
    Holds the initial variables for the traders
    """
    def __init__(self, weight_fundamentalist, weight_chartist, weight_random, c_share_strat,
                 money, stocks, covariance_matrix, init_price):
        """
        Initializes variables for the trader
        :param weight_fundamentalist: float fundamentalist expectation component
        :param weight_chartist: float trend-following chartism expectation component
        :param weight_random: float random or heterogeneous expectation component
        :param weight_mean_reversion: float mean-reversion chartism expectation component
        """
        self.weight_fundamentalist = [weight_fundamentalist]
        self.weight_chartist = [weight_chartist]
        self.weight_random = [weight_random]
        self.c_share_strat = c_share_strat
        self.money = [money]
        self.stocks = [stocks]
        self.wealth = [money + stocks * init_price]
        self.covariance_matrix = covariance_matrix
        self.active_orders = []
        self.hypothetical_money = [money]
        self.hypothetical_stocks = [stocks]
        self.hypothetical_wealth = [money + stocks * init_price]


class TraderParameters:
    """
    Holds the the trader parameters for the distribution model
    """

    def __init__(self, ref_horizon, risk_aversion, learning_ability, max_spread):
        """
        Initializes trader parameters
        :param ref_horizon: integer horizon over which the trader can observe the past
        :param max_spread: Maximum spread at which the trader will submit orders to the book
        :param risk_aversion: float aversion to price volatility
        """
        self.horizon = ref_horizon
        self.risk_aversion = risk_aversion
        self.learning_ability = learning_ability
        self.spread = max_spread * np.random.rand()


class TraderExpectations:
    """
    Holds the agent expectations for several variables
    """
    def __init__(self, price):
        """
        Initializes trader expectations
        :param price: float
        """
        self.price = price
        self.returns = {'stocks': 0.0, 'money': 0.0}


class MarketMaker:
    """Class holding mm properties"""
    def __init__(self, name, variables):
        """
        Initialize trader class
        :param name: integer number which will be the name of the trader
        :param variables: object Tradervariables
        """
        self.name = name
        self.var = variables
        self.metrics = dict.fromkeys(['money', 'inventory', 'inventory_value', 'wealth'], [])

    def __repr__(self):
        """
        :return: String representation of the trader
        """
        return 'Trader' + str(self.name)

    def sell(self, amount, price):
        """
        Sells `amount` of stocks for a total of `price`
        :param amount: int Number of stocks sold.
        :param price: float Total price for stocks.
        :return: -
        """
        if self.var.stocks[-1] < amount:
            raise ValueError("not enough stocks to sell this amount")
        self.var.stocks[-1] -= amount
        self.var.money[-1] += price

    def buy(self, amount, price):
        """
        Buys `amount` of stocks for a total of `price`
        :param amount: int number of stocks bought.
        :param price: float total price for stocks.
        :return: -
        """
        if self.var.money[-1] < price:
            raise ValueError("not enough money to buy this amount of stocks")

        self.var.stocks[-1] += amount
        self.var.money[-1] -= price

    def post_orders(self, orderbook, reference_price):
        # First cancel active orders
        for order in self.var.active_orders:
            orderbook.cancel_order(order)
        self.var.active_orders = []

        # Then decide on new orders and add these to the orderbook
        bid_price = orderbook.highest_bid_price
        buy_bool = bid_price < reference_price
        if buy_bool:
            if self.var.money[-1] > 0:
                bid = orderbook.add_bid(bid_price, 1, self)
                self.var.active_orders.append(bid)
        else:
            if self.var.stocks[-1] > 0:
                ask = orderbook.add_ask(orderbook.lowest_ask_price, 1, self)
                self.var.active_orders.append(ask)


class RLMarketMaker(LearningAgent):
    def __init__(self, name, variables):
        self.name = name
        self.var = variables
        self.metrics = dict.fromkeys(['money', 'inventory', 'inventory_value', 'wealth'], [])
        self.previous_state = None
        self.previous_action = None
        super().__init__()

    def __repr__(self):
        return 'Trader' + str(self.name)

    def sell(self, amount, price, respect_stocks=True):
        if self.var.stocks[-1] < amount and respect_stocks:
            raise ValueError("not enough stocks to sell this amount")
        self.var.stocks[-1] -= amount
        self.var.money[-1] += price

    def buy(self, amount, price, respect_stocks=True):
        if self.var.money[-1] < price and respect_stocks:
            raise ValueError("not enough money to buy this amount of stocks")

        self.var.stocks[-1] += amount
        self.var.money[-1] -= price

    def post_orders(self, orderbook, reference_price):
        # First cancel active orders
        for order in self.var.active_orders:
            orderbook.cancel_order(order)
        self.var.active_orders = []

        # Then decide on new orders and add these to the orderbook
        #todo: implement these in a sensible way
        state = orderbook
        reward = 0
        done = False

        if not self.previous_state:
            self.previous_state = state
            self.previous_action = self.select_action(state)
        else:
            self.step(self.previous_state, self.previous_action, reward, state, done)
            self.previous_action = self.select_action(state)