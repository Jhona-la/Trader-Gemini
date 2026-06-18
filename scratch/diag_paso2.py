import os, sys
sys.path.append(os.path.abspath('.'))
from config import Config
from core.backtest_engine import BacktestDataProvider
from strategies.ml_strategy import UniversalEnsembleStrategy as MLStrategy
from queue import Queue

events_queue = Queue()

sym = Config.TRADING_PAIRS[0]
dp = BacktestDataProvider(events_queue, [sym], {sym: {}})
# Wait, backtest_data_provider constructor requires all_data.
# In run_god_mode_backtest.py it loads data and then passes to BacktestDataProvider.

