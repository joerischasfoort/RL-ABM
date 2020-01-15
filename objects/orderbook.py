"""Limit orderbook updated from Schasfoort & Stockermans 2017"""

import bisect
import operator
import numpy as np


class LimitOrderBook:
    """
    Class which represents the limit-orderbook used by modern Stock markets

    The class contains two separate books represented as lists:

    1. A bids book which contains orders of type 'bid'
    2. An asks book which contains orders of type 'ask'
    """
    def __init__(self, last_price, spread_max, max_return_interval, order_expiration):
        """
        Initialize order-book class
        :param last_price: float initial price
        :param spread_max: float initial spread used to initialize highest bid and ask
        :param max_return_interval: integer length of initial returns series
        :param order_expiration: integer amount of periods after which orders are deleted from the book
        """
        self.bids = []
        self.asks = []
        self.order_expiration = order_expiration
        self.highest_bid_price = last_price - (spread_max / 2)
        self.lowest_ask_price = last_price + (spread_max / 2)
        self.tick_close_price = [np.mean([self.highest_bid_price, self.lowest_ask_price])]

        # historical prices, volumes, and returns for the tick
        self.transaction_prices = []
        self.transaction_volumes = []
        self.returns = [0 for i in range(max_return_interval)]

        # historical prices, volumes, for the total simulation
        self.transaction_prices_history = []
        self.transaction_volumes_history = []
        self.highest_bid_price_history = []
        self.lowest_ask_price_history = []

        # historical sentiment in market data storage
        self.sentiment = []
        self.sentiment_history = []

    def add_bid(self, price, volume, agent):
        """
        Add a bid to the (price low-high, age young-old) sorted bids book
        :param price: float price of the bid
        :param volume: integer volume of the bid
        :param agent: object agent which issues the bid
        :return: object bid
        """
        bid = Order(order_type='b', owner=agent, price=price, volume=volume)
        bisect.insort_left(self.bids, bid)
        self.update_bid_ask_spread('bid')
        return bid

    def add_ask(self, price, volume, agent):
        """
        Add an ask to the (price low-high, age old-young) sorted asks book
        :param price: float price of the ask
        :param volume: integer volume of the ask
        :param agent: object agent which issues the ask
        :return: object ask
        """
        ask = Order(order_type='a', owner=agent, price=price, volume=volume)
        bisect.insort_right(self.asks, ask)
        self.update_bid_ask_spread('ask')
        return ask

    def cancel_order(self, order):
        """
        Removes a particular order from the order book
        :param order: class Order
        :return: None
        """
        for book in [self.bids, self.asks]:
            if order in book:
                book.remove(order)

    def cleanse_book(self):
        """
        Can be invoked at the end of a period to clean all orders from the book and update historical
        variables.
        :return: None
        """
        # store and clean recorded transaction prices
        if len(self.transaction_prices):
            self.transaction_prices_history.append(self.transaction_prices)
        self.transaction_prices = []

        # store and clean recorded transaction volumes
        self.transaction_volumes_history.append(self.transaction_volumes)
        self.transaction_volumes = []

        # story and clean sentiment data
        self.sentiment_history.append(self.sentiment)
        self.sentiment = []

        # increase the age of all orders by 1
        for book in [self.bids, self.asks]:
            for order in book:
                order.age += 1
                if order.age > self.order_expiration:
                    book.remove(order)

        # update current highest bid and lowest ask
        for order_type in ['bid', 'ask']:
            self.update_bid_ask_spread(order_type)

        # update the tick close price for the next tick
        self.tick_close_price.append(np.mean([self.highest_bid_price, self.lowest_ask_price]))

        # update returns
        self.returns.append((self.tick_close_price[-1] - self.tick_close_price[-2]) / self.tick_close_price[-2])

    def match_orders(self):
        """
        Return a price, volume, bid and ask and delete them from the order book if volume of either reaches zero
        :return: None
        """
        # First, make sure that neither the bids or asks books are empty
        if not (self.bids and self.asks):
            return None

        # Then, match highest bid with lowest ask
        if self.bids[-1].price >= self.asks[0].price:
            winning_bid = self.bids[-1]
            winning_ask = self.asks[0]
            price = winning_ask.price
            # The volume is the minimum of the bid and ask
            min_index, volume = min(enumerate([winning_bid.volume, winning_ask.volume]), key=operator.itemgetter(1))
            # both bid and ask are then reduced by that volume, if 0, then removed
            if winning_bid.volume == winning_ask.volume:
                # notify owner it no longer has an order in the market
                for order in [winning_bid, winning_ask]:
                    order.owner.var.active_orders = []
                # remove these elements from list
                del self.bids[-1]
                del self.asks[0]
                # update current highest bid and lowest ask
                for order_type in ['bid', 'ask']:
                    self.update_bid_ask_spread(order_type)
            else:
                # decrease volume for both bid and ask
                self.asks[0].volume -= volume
                self.bids[-1].volume -= volume
                # delete the empty bid or ask
                if min_index == 0:
                    self.bids[-1].owner.var.active_orders = []
                    del self.bids[-1]
                    # update current highest bid
                    self.update_bid_ask_spread('bid')
                else:
                    self.asks[0].owner.var.active_orders = []
                    del self.asks[0]
                    # update current lowest ask
                    self.update_bid_ask_spread('ask')
            self.transaction_prices.append(price)
            self.transaction_volumes.append(volume)

            return price, volume, winning_bid, winning_ask

    def update_bid_ask_spread(self, order_type):
        """
        Update the current highest bid or lowest ask and store previous values
        :param order_type: string 'bid' or 'ask'
        :return:
        """
        if ('ask' not in order_type) and ('bid' not in order_type):
            raise ValueError("unknown order_type")

        if order_type == 'ask' and self.asks:
            self.lowest_ask_price_history.append(self.lowest_ask_price)
            self.highest_bid_price_history.append(self.highest_bid_price)
            self.lowest_ask_price = self.asks[0].price
        if order_type == 'bid' and self.bids:
            self.highest_bid_price_history.append(self.highest_bid_price)
            self.lowest_ask_price_history.append(self.lowest_ask_price)
            self.highest_bid_price = self.bids[-1].price

    def __repr__(self):
        """
        :return: String representation of the order book object
        """
        return "order_book"


class Order:
    """The order class can represent both bid or ask type orders"""
    def __init__(self, order_type, owner, price, volume):
        self.order_type = order_type
        self.owner = owner
        self.price = price
        self.volume = volume
        self.age = 0

    def __lt__(self, other):
        """Allows comparison to other orders based on price"""
        return self.price < other.price

    def __repr__(self):
        """
        :return: String representation of the order
        """
        return 'Order_p={}_t={}_o={}_a={}'.format(self.price, self.order_type, self.owner, self.age)
