import asyncio
import argparse
import time
import sys
import os
from datetime import datetime, timezone, timedelta

# Aseguramos que Python detecte la raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import Config
from core.events import MarketEvent, FillEvent, OrderEvent
from core.enums import OrderSide, OrderType
from core.engine import Engine
from data.data_provider import DataProvider
from data.binance_loader import BinanceData
from core.portfolio import Portfolio
from core.market_regime import MarketRegimeDetector
from risk.risk_manager import RiskManager
from utils.logger import logger
from utils.time_helpers import ensure_utc_aware

# Forzamos la carga del bot profundo
from strategies.ml_strategy import UniversalEnsembleStrategy

import numpy as np

class MockDataProvider(DataProvider):
    """
    Subclase de DataProvider para Sandbox que almacena datos in-memory
    y retorna NumPy Structured Arrays (compatible con BacktestDataProvider).
    
    QUÉ: Simula el DataProvider para backtesting sandbox
    POR QUÉ: La ML strategy accede a datos via bars[-1][4] (numpy indexing)
    CÓMO: Mantiene DataFrame interno y lo convierte a structured array en get_latest_bars
    """
    STRUCT_DTYPE = [
        ('timestamp', 'i8'), ('open', 'f4'), ('high', 'f4'), 
        ('low', 'f4'), ('close', 'f4'), ('volume', 'f4')
    ]
    
    def __init__(self):
        super().__init__()
        self.db = {}           # symbol -> pd.DataFrame (index=datetime)
        self.symbol_list = []  # For strategy compatibility
    
    def get_latest_bars(self, symbol: str, n: int = 1, timeframe='1m'):
        """Retorna NumPy Structured Array compatible con BacktestDataProvider."""
        if symbol not in self.db or self.db[symbol].empty:
            return None
        
        df = self.db[symbol]
        df_slice = df.iloc[-n:] if len(df) >= n else df
        
        if df_slice.empty:
            return None
        
        # Convertir a Structured Array (Zero-Copy parity con BacktestDataProvider)
        result = np.empty(len(df_slice), dtype=self.STRUCT_DTYPE)
        result['timestamp'] = df_slice.index.values.astype('datetime64[ms]').astype('int64')
        result['open'] = df_slice['open'].values.astype('float32')
        result['high'] = df_slice['high'].values.astype('float32')
        result['low'] = df_slice['low'].values.astype('float32')
        result['close'] = df_slice['close'].values.astype('float32')
        result['volume'] = df_slice['volume'].values.astype('float32')
        return result
        
    def update_bars(self, event):
        pass # Data is updated directly by stream_historical_data
    
    def get_active_positions(self):
        """Mock for strategy compatibility."""
        return {}

class MockExecutionHandler:
    """
    Simula interceptar el OrderEvent de la IA y rellenar un FillEvent asincrónicamente
    para que la estrategia de ML reciba feedback e inicie su ciclo PPO.
    """
    def __init__(self, events_queue, data_provider):
        self.events = events_queue
        self.data_provider = data_provider
        self.portfolio = None
        logger.info("🛠️ [SANDBOX] Mock Execution Handler initialized.")

    def sync_portfolio_state(self, portfolio):
        self.portfolio = portfolio

    async def execute_order(self, event: OrderEvent):
        logger.info(f"⚡ [SANDBOX] Executing mock order for {event.symbol} / {event.direction}")
        
        dh = self.data_provider
        bars = dh.get_latest_bars(event.symbol, n=1)
        if not bars:
             logger.warning(f"❌ [SANDBOX] Cannot fill {event.symbol}, no local data.")
             return
             
        fill_price = bars[-1]['close']
        
        # Simula latencia de la red de Binance (Slip & Latency)
        await asyncio.sleep(0.005) # 5ms
        
        commission = (fill_price * event.quantity) * 0.0004 # 0.04% maker fee as default
        
        fill_event = FillEvent(
            timeindex=datetime.now(timezone.utc),
            symbol=event.symbol,
            exchange="BINANCE_MOCK",
            quantity=event.quantity,
            direction=event.direction,
            fill_cost=fill_price,
            commission=commission,
            strategy_id=event.strategy_id,
            fill_price=fill_price,
            order_id="mock_ws_001",
            is_closed=False # Not tracking partials in simple sandbox
        )
        
        logger.info(f"✅ [SANDBOX] Filled: {fill_event}")
        self.events.put(fill_event)

    def remove_order(self, order_id, event=None):
        pass # Not tracked deeply in God-Mode sandbox unless necessary

async def stream_historical_data(symbol: str, df: pd.DataFrame, engine: Engine):
    """
    Lee un DataFrame hacia adelante y expide Eventos de Mercado artificiales,
    inyectando `asyncio.sleep` para alimentar el motor AsyncBoundedQueue en tiempo 'falso'
    que despierte la rutina PPO y simule Websockets.
    """
    logger.info(f"🌊 [SANDBOX] Empezando WS Emulator Data Stream para: {symbol} ({len(df)} ticks)")
    
    # Pre-cargar DataProvider para que MarketRegime funcione
    dh = engine.data_handlers[0]
    
    # In order to supply data naturally, we insert data bar by bar to the provider
    # and then emit the MarketEvent for the Engine to pick up.
    
    # Buffer para emular `get_latest_bars` (requerimos min 200 para entrenamiento)
    warmup_period = 250
    if len(df) < warmup_period + 10:
        logger.warning("Dataframe demasiado corto para God-Mode.")
        return
        
    # Inicialización de DF frío
    hist_raw = df.iloc[:warmup_period].copy()
    hist_raw.set_index('datetime', inplace=True)
    dh.db[symbol] = hist_raw
    logger.info(f"🥶 [SANDBOX] Warmup inyectado: {warmup_period} velas frías.")
    
    # Comienza la simulación en tiempo Real simulado
    total_iters = len(df) - warmup_period
    
    for i in range(warmup_period, len(df)):
        row = df.iloc[i]
        dt = ensure_utc_aware(row['datetime'].to_pydatetime())
        
        # 1. Update In-Memory BD "Silenciosa"
        new_row = pd.DataFrame([{
            'open': row['open'], 
            'high': row['high'], 
            'low': row['low'], 
            'close': row['close'], 
            'volume': row['volume']
        }], index=[dt])
        
        dh.db[symbol] = pd.concat([dh.db[symbol], new_row])
        # Keep buffer large enough for ML training (min_bars_to_train=300)
        if len(dh.db[symbol]) > 2000:
             dh.db[symbol] = dh.db[symbol].iloc[-2000:]

        # 2. Fire Async MarketEvent
        event = MarketEvent(symbol=symbol, close_price=row['close'], timestamp=dt)
        engine.events.put(event)
        
        # 3. Simulate Async I/O (The magic that unstucks the PPO WaitLocks)
        await asyncio.sleep(0.001) # 1ms pseudo-WS tick
        
        if i % 500 == 0:
            logger.info(f"🌊 [SANDBOX] Progress: {((i-warmup_period)/total_iters)*100:.1f}% ({symbol})")
            
    # Cuando termine el ciclo, mandamos a frenar el motor.
    logger.info(f"🏁 [SANDBOX] Stream de Cintas Finalizado: {symbol}")
    engine.running = False # Detener el loop del engine asíncronamente
    await asyncio.sleep(2) # Give it 2 secs to clear last orders
    engine.stop()

async def run_sandbox(symbol: str, days: int):
    print("=====================================================")
    print(f"🤖 TRADER GEMINI: GOD-MODE SANDBOX ENGINE")
    print(f"Simulando Entorno WebSockets para: {symbol}")
    print("=====================================================")
    
    # 1. Build infrastructure
    engine = Engine()
    data_provider = MockDataProvider()
    engine.register_data_handler(data_provider)
    
    # Load Real historical data to mock
    client = Client()
    raw_df = client.get_historical_klines(symbol.replace('/', ''), Client.KLINE_INTERVAL_1MINUTE, f"{days} days ago UTC")
    # Clean data (Similar to data_provider formatting)
    df = pd.DataFrame(raw_df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    portfolio = Portfolio(initial_capital=100.0, auto_save=False) # Starting capital $100
    engine.register_portfolio(portfolio)
    
    # 2. Register Mock Sandbox execution
    mock_exec = MockExecutionHandler(engine.events, data_provider)
    mock_exec.sync_portfolio_state(portfolio)
    engine.register_execution_handler(mock_exec)
    
    # 3. Register standard Risk and Regime
    market_regime = MarketRegimeDetector()
    risk_manager = RiskManager(portfolio=portfolio)
    engine.register_risk_manager(risk_manager)
    
    # 4. SET CONFIG OVERRIDES BEFORE strategy instantiation
    Config.Strategies.ML_LOOKBACK_BARS = 60  # Reducir para sandbox (2 días = 2880 barras)
    
    # 5. Inject ML God-Mode Strategy with Sandbox Flag
    logger.info("🧠 Injecting UniversalEnsembleStrategy in SANDBOX mode.")
    strategy = UniversalEnsembleStrategy(data_provider, engine.events)
    # Enable Sandbox mode
    if hasattr(strategy, 'is_sandbox'):
         strategy.is_sandbox = True
    
    # ⚠️ FORCE INITIAL TRAINING TO FIT SCALER
    strategy.is_trained = False
    strategy.min_bars_to_train = 400
    strategy.MIN_MODEL_ACCURACY = 0.0 # Don't reject for low score in mock data
    strategy.LOOKAHEAD_BARS = 5 # Reduce lookahead to find more signals in short data
    Config.Strategies.ML_LOOKBACK_BARS = 600 # Satisfy len(df) < 200 check
    
    # Pre-initialize ML components...
    from strategies.components.feature_engineering import FeatureEngineering
    from strategies.components.signal_generator import SignalGenerator
    from strategies.phalanx import OrderFlowAnalyzer, OnlineGARCH
    from core.xai_engine import XAIEngine
    
    strategy.feature_engineer = FeatureEngineering()
    strategy.signal_generator = SignalGenerator(strategy.strategy_id)
    strategy.phalanx = OrderFlowAnalyzer()
    strategy.garch = OnlineGARCH(1e-6, 0.1, 0.85, 1e-4)
    strategy.xai_engine = XAIEngine()
    
    engine.register_strategy(strategy)
    
    try:
        # Run Engine and Data Streamer concurrently
        logger.info("🚀 Launching Engine and Data Streamer Tasks...")
        await asyncio.gather(
            engine.start(),
            stream_historical_data(symbol, df, engine)
        )
    except KeyboardInterrupt:
        logger.info("⏹️ Sandbox halted by user.")
        engine.stop()
    finally:
        logger.info("=====================================================")
        logger.info("📊 SANDBOX POST-MORTEM (GOD-MODE PPO BEHAVIOR)")
        logger.info(f"🏁 Final Capital: ${portfolio.current_cash:.2f}")
        logger.info(f"📈 Total Realized PnL: ${portfolio.realized_pnl:.2f}")
        
        # Print internal learning evolution summary
        if hasattr(strategy, 'ai') and hasattr(strategy.ai, 'rf_weight'):
             logger.info(f"🧠 ENSEMBLE END WEIGHTS: RF=({strategy.ai.rf_weight:.2f}) XGB=({strategy.ai.xgb_weight:.2f}) GB=({strategy.ai.gb_weight:.2f})")
             

if __name__ == "__main__":
    from binance.client import Client
    
    parser = argparse.ArgumentParser(description="Trader Gemini - Sandbox ML Simulator")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--days", type=int, default=3)
    args = parser.parse_args()
    
    asyncio.run(run_sandbox(args.symbol, args.days))
