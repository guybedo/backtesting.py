'''
Created on Sep 27, 2021

@author: guybedo
'''
import logging
from math import copysign
from time import sleep
from typing import _alias, CT_co, List
import warnings

from backtesting._util import _Data, _Indicator
from backtesting.backtesting import Strategy, _Broker, Order, Trade, Position,\
    _OutOfMoneyError
import ccxt

import numpy as np
import pandas as pd
from datetime import datetime


Type = _alias(type, CT_co, inst=False)


class Ohlc:
    open = None
    low = None
    high = None
    close = None

    closed = False

    def __init__(self):
        pass


class DataFetcher:

    def __init__(self):
        pass

    def get_data(self) -> pd.DataFrame:
        pass


class DbPosition(object):

    symbol = None
    side = None
    size = None
    entry_date = None
    entry_price = None
    exit_date = None
    exit_price = None
    pnl = None

    def __init__(self):
        pass


class TradingRepository(object):

    def __init__(self):
        pass

    def find_most_recent_position(self):
        pass

    def find_open_positions(self, symbol, size=None):
        pass

    def save_position(self, symbol, size, side, entry_price, entry_date):
        pass

    def close_position(self, symbol, size, side, exit_price, exit_date):
        pass


class LiveBroker(_Broker):

    def __init__(self, symbol, margin, exclusive_orders, repository=None):
        assert 0 < margin <= 1, f"margin should be between 0 and 1, is {margin}"
        self._symbol = symbol
        self._leverage = 1 / margin
        self._exclusive_orders = exclusive_orders
        self._repository = repository

        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.position = Position(self)
        self.closed_trades: List[Trade] = []

    def _get_current_price(self):
        pass

    def load_state(self):
        pass

    def _close_order(self, order):
        pass

    def _reduce_trade(self, trade: Trade, price: float, size: float):
        pass

    def _close_trade(self, trade: Trade, price: float):
        pass

    def _open_trade(self, price: float, size: int, sl: float, tp: float):
        pass

    def _close_orders(self):
        for order in self.orders:
            self._close_order(order)

    def _close_trades(self):
        for trade in self.trades:
            self._close_trade(trade)

    def new_order(self,
                  size: float,
                  limit: float = None,
                  stop: float = None,
                  sl: float = None,
                  tp: float = None,
                  *,
                  trade: Trade = None):
        """
        Argument size indicates whether the order is long or short
        """
        size = float(size)
        stop = stop and float(stop)
        limit = limit and float(limit)
        sl = sl and float(sl)
        tp = tp and float(tp)

        order = Order(self, size, limit, stop, sl, tp, trade)
        self.orders.append(order)

        return order

    @property
    def last_price(self) -> float:
        """ Price at the last (current) close. """
        return self._data.Close[-1]

    @property
    def equity(self) -> float:
        return self._cash + sum(trade.pl for trade in self.trades)

    @property
    def margin_available(self) -> float:
        # From https://github.com/QuantConnect/Lean/pull/3768
        margin_used = sum(
            trade.value / self._leverage for trade in self.trades)
        return max(0, self.equity - margin_used)

    def next(self):
        i = self._i = len(self._data) - 1
        self._process_orders()

    def _process_orders(self):
        price = self._get_current_price()
        for order in list(self.orders):  # type: Order
            is_long = order.size > 0
            if is_long:
                if not (order.sl or -np.inf) < (order.limit or order.stop or price) < (order.tp or np.inf):
                    raise ValueError(
                        "Long orders require: "
                        f"SL ({order.sl}) < LIMIT ({order.limit or order.stop or price}) < TP ({order.tp})")
            else:
                if not (order.tp or -np.inf) < (order.limit or order.stop or price) < (order.sl or np.inf):
                    raise ValueError(
                        "Short orders require: "
                        f"TP ({order.tp}) < LIMIT ({order.limit or order.stop or price}) < SL ({order.sl})")

            is_market_order = not order.limit and not order.stop

            if is_market_order:
                if self._exclusive_orders:
                    for o in self.orders:
                        if not o.is_contingent:
                            o.cancel()
                    for t in self.trades:
                        t.close()
                pass
            elif order.limit or order.stop:
                self._create_order(
                    order.size,
                    order.limit,
                    order.stop,
                    order.sl,
                    order.tp)

            if order.size <= 0 or abs(order.size) * price > self.margin_available * self._leverage:
                self.orders.remove(order)
                continue

            self._open_trade(
                price,
                order.size,
                order.sl,
                order.tp)
            self.orders.remove(order)


class CcxtBroker(LiveBroker):

    def __init__(self, symbol, margin, exclusive_orders,
                 exchange_id, api_key, secret, exchange_headers=None, repository=None):
        super().__init__(symbol, margin, exclusive_orders, repository=repository)
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': secret,
        })
        if exchange_headers:
            self.exchange.headers = exchange_headers

    def _get_current_price(self):
        self._ticker = self._ticker or self.exchange.fetch_ticker(self._symbol)
        return self._ticker['ask']

    def load_state(self):
        self._load_balance()
        self._load_orders()
        self._load_trades()
        self._load_history()
        self.position = Position(self)

    def _load_balance(self):
        self._balance = self._balance or self.exchange.fetch_balance()
        self._free_usd_balance = self._balance['free']['USD']

    def _load_orders(self):
        self.orders = self.exchange.fetch_open_orders(
            self._symbol,
            since=None,
            limit=None,
            params=dict())

    def _load_trades(self):
        self.trades = [
            Trade(
                broker=self,
                size=p['size'] * (1 if p['side'] == 'buy' else -1))
            for p in self.exchange.fetch_positions(self._symbol, params=dict())
            if p['size'] > 0]

    def _load_history(self):
        self.closed_trades = None

    def _open_trade(self, size: int, sl: float, tp: float):
        if self.exclusive_orders:
            self._close_trades()
            self._close_orders()
        side = 'buy' if size > 0 else 'sell'
        self.exchange.create_order(
            symbol=self._symbol,
            type="market",
            side=side,
            amount=abs(size))
        self.trades.append(Trade(broker=self, size=size))
        if self._repository:
            self._repository.save_position(
                symbol=self._symbol,
                size=abs(size),
                side=side,
                entry_price=self._get_current_price(),
                entry_date=datetime.utcnow())

    def _close_trade(self, trade: Trade):
        original_side = 'buy' if trade.size > 0 else 'sell'
        side = 'sell' if trade.size > 0 else 'buy'
        self.exchange.create_order(
            symbol=self._symbol,
            type="market",
            side=side,
            amount=abs(trade.size),
            params={"reduceOnly": True})
        self.trades.remove(trade)
        if self._repository:
            self._repository.close_position(
                symbol=self._symbol,
                side=original_side,
                size=abs(trade.size),
                exit_price=self._get_current_price(),
                exit_date=datetime.utcnow())

    def _reduce_trade(self, trade: Trade, price: float, size: float):
        pass

    def _create_order(self,
                      size: float,
                      limit: float = None,
                      stop: float = None,
                      sl: float = None,
                      tp: float = None):
        pass

    def _close_order(self, order):
        self.exchange.cancel_order(order['id'])
        self.orders.remove(order)


class Livetrading:

    def __init__(self,
                 strategy: Type[Strategy],
                 data_fetcher: DataFetcher,
                 live_broker: LiveBroker,
                 hearbeat=1 * 60):
        self._strategy = strategy
        self.data_fetcher = data_fetcher
        self.broker = live_broker
        self.hearbeat = hearbeat

    def run(self, **kwargs):
        while True:
            data = self.data_fetcher.get_data()
            if data.iloc[-1]['Closed']:
                self._next(data, **kwargs)
            sleep(self.hearbeat)

    def _next(self, data, **kwargs):
        try:
            data = data.copy(deep=False)
            sanitize_data(data)
            self._data: pd.DataFrame = data

            strategy: Strategy = self._strategy(self.live_broker, data, kwargs)
            strategy.init()
            data._update()  # Strategy.init might have changed/added to data.df

            # Indicators used in Strategy.next()
            indicator_attrs = {
                attr: indicator
                for attr, indicator in strategy.__dict__.items()
                if isinstance(indicator, _Indicator)}.items()

            i = len(self._data)
            data._set_length(i + 1)
            for attr, indicator in indicator_attrs:
                setattr(strategy, attr, indicator[..., :i + 1])

            self.broker.load_state()
            strategy.next()
            self.broker.next()
        except:
            logging.error('Error processing data')


def sanitize_data(data):
        # Convert index to datetime index
    if (not isinstance(data.index, pd.DatetimeIndex) and
        not isinstance(data.index, pd.RangeIndex) and
        # Numeric index with most large numbers
        (data.index.is_numeric() and
         (data.index > pd.Timestamp('1975').timestamp()).mean() > .8)):
        try:
            data.index = pd.to_datetime(
                data.index, infer_datetime_format=True)
        except ValueError:
            pass

    if 'Volume' not in data:
        data['Volume'] = np.nan

    if len(data) == 0:
        raise ValueError('OHLC `data` is empty')
    if len(data.columns.intersection({'Open', 'High', 'Low', 'Close', 'Volume'})) != 5:
        raise ValueError("`data` must be a pandas.DataFrame with columns "
                         "'Open', 'High', 'Low', 'Close', and (optionally) 'Volume'")
    if data[['Open', 'High', 'Low', 'Close']].isnull().values.any():
        raise ValueError('Some OHLC values are missing (NaN). '
                         'Please strip those lines with `df.dropna()` or '
                         'fill them in with `df.interpolate()` or whatever.')
    if not data.index.is_monotonic_increasing:
        warnings.warn('Data index is not sorted in ascending order. Sorting.',
                      stacklevel=2)
        data = data.sort_index()
    if not isinstance(data.index, pd.DatetimeIndex):
        warnings.warn('Data index is not datetime. Assuming simple periods, '
                      'but `pd.DateTimeIndex` is advised.',
                      stacklevel=2)
