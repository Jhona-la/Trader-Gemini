import pandas as pd
import time
from ib_insync import *
from .data_provider import DataProvider
from core.events import MarketEvent
from config import Config

class IBKRData(DataProvider):
    def __init__(self, events_queue, symbol_list):
        self.events_queue = events_queue
        self.symbol_list = symbol_list
        self.latest_data = {s: [] for s in symbol_list}
        
        # Initialize IB connection
        self.ib = IB()
        self.connected = False
        self.connect()

    def connect(self):
        try:
            print(f"Connecting to IBKR at {Config.IBKR_HOST}:{Config.IBKR_PORT}...")
            self.ib.connect(Config.IBKR_HOST, Config.IBKR_PORT, clientId=Config.IBKR_CLIENT_ID)
            self.connected = True
            print("Connected to IBKR!")
        except Exception as e:
            print(f"Failed to connect to IBKR: {e}")
            print("Make sure TWS or IB Gateway is running and API connections are enabled.")
            self.connected = False

    def _get_contract(self, symbol):
        """
        Create an IB contract object from a string symbol.
        Format expected: 'AAPL' (Stock) or 'EURUSD' (Forex)
        """
        if len(symbol) == 6 and symbol.isalpha():
            # Assume Forex (e.g., EURUSD)
            return Forex(symbol[:3] + symbol[3:])
        else:
            # Assume Stock (Smart Routing)
            return Stock(symbol, 'SMART', 'USD')

    def get_latest_bars(self, symbol, n=1):
        try:
            bars_list = self.latest_data[symbol]
        except KeyError:
            print("That symbol is not available in the historical data set.")
            raise
        return bars_list[-n:]

    def update_bars(self):
        if not self.connected:
            return

        for s in self.symbol_list:
            try:
                contract = self._get_contract(s)
                
                # Determine data type based on asset class
                # Forex uses MIDPOINT, Stocks usually require TRADES (unless you have pro data)
                data_type = 'MIDPOINT'
                if isinstance(contract, Stock):
                    data_type = 'TRADES'
                
                # Request 1 min bars, just the last one
                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr='120 S',
                    barSizeSetting='1 min',
                    whatToShow=data_type,
                    useRTH=True
                )

                if bars:
                    latest_bar = bars[-1]
                    timestamp = latest_bar.date
                    
                    # Convert to standard format
                    bar_data = {
                        'symbol': s,
                        'datetime': timestamp,
                        'open': latest_bar.open,
                        'high': latest_bar.high,
                        'low': latest_bar.low,
                        'close': latest_bar.close,
                        'volume': latest_bar.volume if hasattr(latest_bar, 'volume') else 0
                    }

                    # Avoid duplicates
                    if not self.latest_data[s] or self.latest_data[s][-1]['datetime'] != timestamp:
                        self.latest_data[s].append(bar_data)
                        print(f"New Bar for {s}: {timestamp} - Close: {bar_data['close']}")
                        self.events_queue.put(MarketEvent())

            except Exception as e:
                print(f"Error fetching IBKR data for {s}: {e}")
