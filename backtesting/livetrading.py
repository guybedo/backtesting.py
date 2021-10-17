'''
Created on Sep 27, 2021

@author: guybedo
'''
from abc import abstractproperty, abstractmethod
from datetime import datetime
import logging
from math import copysign
from time import sleep
import traceback
from typing import _alias, CT_co, List
import warnings

from backtesting._util import _Data, _Indicator
from backtesting.backtesting import Strategy, _Broker, Order, Trade, Position,\
    _OutOfMoneyError
import ccxt

import numpy as np
import pandas as pd


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


class NotificationService(object):

    def __init__(self):
        pass

    def notify_open_trade(self, symbol, size, side, entry_price):
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
    _data = None
    _ticker = None
    _balance = None
    orders = list()
    open_orders = list()
    trades = list()
    closed_trades = list()

    def __init__(self, symbol, margin, exclusive_orders,
                 repository: TradingRepository=None, notification_service: NotificationService=None):
        assert 0 < margin <= 1, f"margin should be between 0 and 1, is {margin}"
        self._symbol = symbol
        self._leverage = 1 / margin
        self._exclusive_orders = exclusive_orders
        self._repository = repository
        self._notification_service = notification_service

        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.position = Position(self)
        self.closed_trades: List[Trade] = []

    @abstractmethod
    def _get_current_price(self):
        pass

    @abstractmethod
    def load_state(self):
        pass

    @abstractmethod
    def _close_order(self, order):
        pass

    @abstractmethod
    def _reduce_trade(self, trade: Trade, price: float, size: float):
        pass

    @abstractmethod
    def _close_trade(self, trade: Trade, price: float):
        pass

    @abstractmethod
    def _open_trade(self, price: float, size: int, sl: float, tp: float):
        pass

    @property
    @abstractmethod
    def equity_total(self) -> float:
        return 0

    @property
    @abstractmethod
    def equity_free(self) -> float:
        return 0

    @property
    @abstractmethod
    def equity_used(self) -> float:
        return 0

    def _close_orders(self):
        for order in self.open_orders:
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
                size = order.size
                if -1 < size < 1:
                    available = self.equity_total if self._exclusive_orders else self.equity_free
                    size = copysign(available / price, size)
                self._open_trade(
                    price,
                    size,
                    order.sl,
                    order.tp)
            else:
                self._create_order(
                    order.size,
                    order.limit,
                    order.stop,
                    order.sl,
                    order.tp)
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
        current_candle_idx = None
        while True:
            try:
                data = self.data_fetcher.get_data()
                data = data[data['Closed'] == True]
                is_new_candle = current_candle_idx is None or data.shape[0] > current_candle_idx
                if is_new_candle:
                    current_candle_idx = data.shape[0]
                    logging.info(
                        'Processing new candle {idx}'.format(
                            idx=data.shape[0]))
                    self._next(data, **kwargs)
                else:
                    logging.info(
                        'No new candle to process, data size {data_size}, candle idx {idx}'.format(
                            data_size=data.shape[0],
                            idx=current_candle_idx))
            except Exception as e:
                logging.error('Error running live trading', e)
            sleep(self.hearbeat)

    def _next(self, data, **kwargs):
        try:
            self._data: pd.DataFrame = data.copy(deep=False)
            sanitize_data(self._data)
            data = self.broker._data = _Data(self._data.copy(deep=False))

            strategy: Strategy = self._strategy(self.broker, data, kwargs)
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
        except Exception as ex:
            traceback.format_exc()
            logging.error('Error processing data', ex)


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
