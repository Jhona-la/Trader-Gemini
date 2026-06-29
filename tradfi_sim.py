import asyncio
import time
import logging
from utils.logger import setup_logger

# Import the new Multi-Broker Data layers
from data.tradfi_loader import TradFiLoader
from core.data_aggregator import DataAggregator

# Setup logger
logger = setup_logger("TradFi-Sim")

async def run_simulation():
    print("\n" + "="*80)
    print("📈 INICIANDO SIMULACION TRADFI (S&P 500) - MULTI BROKER")
    print("="*80 + "\n")
    
    # 1. Initialize loaders
    logger.info("⚡ Conectando al Gateway de Datos TradFi (yfinance)...")
    tradfi_loader = TradFiLoader()
    
    # We mock BinanceData to keep it simple, since we only query SPY
    class MockCrypto:
        def __getattr__(self, name): return lambda *a, **k: None
        
    crypto_loader = MockCrypto()
    
    # 2. Setup Aggregator
    aggregator = DataAggregator(crypto_loader=crypto_loader, tradfi_loader=tradfi_loader)
    
    # 3. Request Data for SPY
    symbol = "SPY"
    logger.info(f"📊 Extrayendo velas historicas y tick actual para {symbol}...")
    start_t = time.perf_counter()
    bars = aggregator.get_latest_bars(symbol, n=150, timeframe="1m")
    bid, ask = aggregator.get_fast_bid_ask(symbol)
    latency = (time.perf_counter() - start_t) * 1000
    
    if not bars:
        logger.error(f"❌ Fallo al extraer datos para {symbol}.")
        return

    # 4. Display Results
    logger.info(f"✅ {len(bars)} barras procesadas con exito.")
    logger.info(f"💵 Fast Bid: ${bid:.2f} | Ask: ${ask:.2f} | Last Close: ${bars[-1]['close']:.2f}")
    logger.info(f"⚡ Latencia del DataAggregator: {latency:.2f}ms")
    
    print("\n" + "="*80)
    print("🧠 TENSOR DE EVALUACION OMNI - MULTI BROKER")
    print("="*80 + "\n")
    
    logger.info("🧪 Inyectando SPY en el Feature Store Cuantico...")
    from strategies.components.feature_engineering import FeatureEngineering
    fe = FeatureEngineering()
    import pandas as pd
    df = pd.DataFrame(bars)
    
    fe_start = time.perf_counter()
    features = fe.prepare_features(df)
    fe_latency = (time.perf_counter() - fe_start) * 1000
    
    if features.is_empty():
        logger.error("❌ Fallo de extraccion de features (Data Starvation).")
    else:
        logger.info(f"🧬 Extraccion HFT de Features completada: {fe_latency:.2f}ms")
        logger.info(f"   Shape: {features.shape}. Último RSI: {features['rsi_14'][-1]:.2f}")
        
    print("\n" + "="*80)
    print("🏁 SIMULACION TRADFI COMPLETADA EXITOSAMENTE")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(run_simulation())
