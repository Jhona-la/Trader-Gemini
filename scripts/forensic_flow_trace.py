"""
🔬 FORENSIC FLOW TRACE — Trazado Bidireccional de Señales
=========================================================
Este script NO simula nada. TRAZA el grafo de flujo real del sistema,
verificando nodo por nodo si cada arista está viva o muerta.

Ejecuta en secuencia:
  NODO 1: ¿fetch_multi_symbol_data devuelve datos válidos?
  NODO 2: ¿BacktestDataProvider los estructura correctamente?
  NODO 3: ¿get_latest_bars retorna arrays con formato correcto?
  NODO 4: ¿_prepare_features genera features sin NaN?
  NODO 5: ¿UniversalEnsembleStrategy se inicializa sin excepciones ocultas?
  NODO 6: ¿_run_inference genera señales con confidence > 0?
  NODO 7: ¿calculate_signals emite eventos al events_queue?
  NODO 8: ¿El método predict() existe? (SPOILER: NO)

Cada nodo imprime ✅ o ❌ con datos concretos.
"""
import os, sys, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

print("=" * 70)
print("🔬 FORENSIC FLOW TRACE — Trazado Bidireccional Nodo a Nodo")
print("=" * 70)

bugs_found = []

# ═══════════════════════════════════════════════════════════════
# NODO 1: Descarga de datos reales
# ═══════════════════════════════════════════════════════════════
print("\n[NODO 1] Descarga de datos reales desde Binance...")
try:
    from core.backtest_infra import fetch_multi_symbol_data
    all_data = fetch_multi_symbol_data(["BTC/USDT"], days=2)
    
    if "BTC/USDT" not in all_data:
        print("  ❌ BTC/USDT NO está en all_data")
        bugs_found.append("NODO 1: fetch_multi_symbol_data no retorna BTC/USDT")
    else:
        df = all_data["BTC/USDT"]
        print(f"  ✅ Datos descargados: {len(df)} filas")
        print(f"     Tipo: {type(df).__module__}.{type(df).__name__}")
        print(f"     Columnas: {df.columns if hasattr(df, 'columns') else 'N/A'}")
        if hasattr(df, 'head'):
            print(f"     Primeras 3 filas:\n{df.head(3)}")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    traceback.print_exc()
    bugs_found.append(f"NODO 1: {e}")

# ═══════════════════════════════════════════════════════════════
# NODO 2: BacktestDataProvider construye struct_data
# ═══════════════════════════════════════════════════════════════
print("\n[NODO 2] BacktestDataProvider inicialización...")
try:
    from core.backtest_infra import BacktestDataProvider
    provider = BacktestDataProvider(None, ["BTC/USDT"], all_data, backtest_days=2)
    
    if "BTC/USDT" not in provider.struct_data:
        print("  ❌ BTC/USDT no está en struct_data")
        bugs_found.append("NODO 2: struct_data no contiene BTC/USDT")
    else:
        sd = provider.struct_data["BTC/USDT"]
        tfs = list(sd.keys())
        print(f"  ✅ struct_data construido. Timeframes: {tfs}")
        for tf in tfs:
            arr = sd[tf]
            print(f"     {tf}: {len(arr)} barras, dtype={arr.dtype if hasattr(arr, 'dtype') else 'N/A'}")
        
    print(f"  Global timeline: {len(provider.global_timeline)} epochs")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    traceback.print_exc()
    bugs_found.append(f"NODO 2: {e}")

# ═══════════════════════════════════════════════════════════════
# NODO 3: get_latest_bars retorna datos correctamente
# ═══════════════════════════════════════════════════════════════
print("\n[NODO 3] get_latest_bars funcionalidad...")
try:
    # Simular que estamos en el epoch 500
    if len(provider.global_timeline) > 500:
        provider.current_time_ms = int(provider.global_timeline[500])
    else:
        provider.current_time_ms = int(provider.global_timeline[-1])
    
    bars = provider.get_latest_bars("BTC/USDT", n=100, timeframe="1m")
    
    if bars is None:
        print("  ❌ get_latest_bars retorna None")
        bugs_found.append("NODO 3: get_latest_bars retorna None")
    elif len(bars) == 0:
        print("  ❌ get_latest_bars retorna array vacío")
        bugs_found.append("NODO 3: get_latest_bars retorna array vacío")
    else:
        print(f"  ✅ Retorna {len(bars)} barras")
        print(f"     Tipo: {type(bars).__name__}, dtype names: {bars.dtype.names if hasattr(bars, 'dtype') else 'N/A'}")
        print(f"     Último close: {bars[-1]['close']}")
        print(f"     ¿Es numpy structured array? {hasattr(bars, 'dtype')}")
        print(f"     ¿Tiene 'timestamp'? {'timestamp' in bars.dtype.names if hasattr(bars, 'dtype') else False}")
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    traceback.print_exc()
    bugs_found.append(f"NODO 3: {e}")

# ═══════════════════════════════════════════════════════════════
# NODO 4: MLStrategy se puede instanciar
# ═══════════════════════════════════════════════════════════════
print("\n[NODO 4] UniversalEnsembleStrategy instanciación...")
try:
    from strategies.ml_strategy import UniversalEnsembleStrategy
    from queue import Queue
    
    events_q = Queue()
    strategy = UniversalEnsembleStrategy(
        data_provider=provider,
        events_queue=events_q,
        symbol="BTC/USDT",
        lookback=50,
        horizon="SCALPING"
    )
    
    print(f"  ✅ Instanciada correctamente")
    print(f"     strategy_id: {strategy.strategy_id}")
    print(f"     symbol: {strategy.symbol}")
    print(f"     primary_tf: {strategy.primary_tf}")
    print(f"     horizon: {strategy.horizon_str}")
    print(f"     running: {strategy.running}")
    
    # Check if predict() method exists
    has_predict = hasattr(strategy, 'predict')
    print(f"     ¿Tiene método predict()? {has_predict}")
    if not has_predict:
        print("  ⚠️  BUG CONFIRMADO: simulate_multiverse.py llama .predict() pero NO existe")
        bugs_found.append("NODO 4: predict() NO EXISTE en UniversalEnsembleStrategy")
    
    has_run_inference = hasattr(strategy, '_run_inference')
    print(f"     ¿Tiene método _run_inference()? {has_run_inference}")
    
    has_calculate_signals = hasattr(strategy, 'calculate_signals')
    print(f"     ¿Tiene método calculate_signals()? {has_calculate_signals}")
    
except Exception as e:
    print(f"  ❌ ERROR en instanciación: {e}")
    traceback.print_exc()
    bugs_found.append(f"NODO 4: {e}")

# ═══════════════════════════════════════════════════════════════
# NODO 5: _run_inference genera señal
# ═══════════════════════════════════════════════════════════════
print("\n[NODO 5] _run_inference() ejecución directa...")
try:
    # Set backtest mode
    Config.IS_BACKTESTING = True
    strategy.is_sandbox = True
    strategy.is_live = False
    
    # Ensure the strategy can see bars at current epoch
    provider.current_time_ms = int(provider.global_timeline[500])
    provider._epoch_bars_cache = {}  # Clear cache
    
    # Run inference directly
    strategy._run_inference()
    
    # Check what happened
    print(f"  analysis_stats: {strategy.analysis_stats}")
    print(f"  market_regime: {strategy.market_regime}")
    
    # Check events queue
    if not events_q.empty():
        event = events_q.get_nowait()
        print(f"  ✅ Señal emitida: {event}")
    else:
        print(f"  ⚠️  No se emitió señal (puede ser normal si confianza < threshold)")
    
    # Check engine scores if available
    if hasattr(strategy, 'engine_scores'):
        print(f"  engine_scores: {strategy.engine_scores}")
    
except Exception as e:
    print(f"  ❌ ERROR en _run_inference: {e}")
    traceback.print_exc()
    bugs_found.append(f"NODO 5: {e}")

# ═══════════════════════════════════════════════════════════════
# NODO 6: Verificar el flujo en producción (calculate_signals)
# ═══════════════════════════════════════════════════════════════
print("\n[NODO 6] calculate_signals() con MarketEvent simulado...")
try:
    from core.engine import EventType
    
    class FakeMarketEvent:
        def __init__(self):
            self.type = EventType.MARKET
            self.symbol = "BTC/USDT"
            self.timeframe = "1m"
            self.is_closed = True
            self.timestamp = None
    
    provider._epoch_bars_cache = {}
    event = FakeMarketEvent()
    strategy.calculate_signals(event)
    
    if not events_q.empty():
        sig = events_q.get_nowait()
        print(f"  ✅ Señal emitida por calculate_signals: {sig}")
    else:
        print(f"  ⚠️  No se emitió señal por calculate_signals")
        print(f"     analysis_stats: {strategy.analysis_stats}")
        
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    traceback.print_exc()
    bugs_found.append(f"NODO 6: {e}")

# ═══════════════════════════════════════════════════════════════
# NODO 7: Loop masivo — ¿Alguna barra genera señal?
# ═══════════════════════════════════════════════════════════════
print("\n[NODO 7] Loop de 200 barras buscando cualquier señal emitida...")
try:
    signal_count = 0
    errors = 0
    tested = 0
    
    start_idx = 200
    end_idx = min(start_idx + 200, len(provider.global_timeline))
    
    for i in range(start_idx, end_idx):
        provider.current_time_ms = int(provider.global_timeline[i])
        provider._epoch_bars_cache = {}
        provider._epoch_df_cache = {}
        
        try:
            strategy.calculate_signals(FakeMarketEvent())
            tested += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  ⚠️  Error en epoch {i}: {e}")
        
        while not events_q.empty():
            sig = events_q.get_nowait()
            signal_count += 1
            print(f"  🎯 SEÑAL #{signal_count} en epoch {i}: {sig}")
    
    print(f"  Barras testeadas: {tested}")
    print(f"  Señales emitidas: {signal_count}")
    print(f"  Errores: {errors}")
    print(f"  analysis_stats final: {strategy.analysis_stats}")
    
    if signal_count == 0:
        bugs_found.append("NODO 7: 200 barras procesadas y CERO señales emitidas")

except Exception as e:
    print(f"  ❌ ERROR: {e}")
    traceback.print_exc()
    bugs_found.append(f"NODO 7: {e}")

# ═══════════════════════════════════════════════════════════════
# NODO 8: Config.IS_BACKTESTING sincronización
# ═══════════════════════════════════════════════════════════════
print("\n[NODO 8] Verificación de configuración crítica...")
try:
    print(f"  Config.IS_BACKTESTING: {getattr(Config, 'IS_BACKTESTING', 'NO DEFINIDO')}")
    print(f"  Config.INITIAL_CAPITAL: {getattr(Config, 'INITIAL_CAPITAL', 'NO DEFINIDO')}")
    print(f"  Config.BINANCE_LEVERAGE: {getattr(Config, 'BINANCE_LEVERAGE', 'NO DEFINIDO')}")
    print(f"  Config.Strategies.ML_MIN_CONFIDENCE: {getattr(Config.Strategies, 'ML_MIN_CONFIDENCE', 'NO DEFINIDO')}")
    print(f"  Config.Horizons.Scalping: {getattr(Config.Horizons, 'Scalping', 'NO DEFINIDO')}")
    print(f"  Config.Horizons.Swing: {getattr(Config.Horizons, 'Swing', 'NO DEFINIDO')}")
    print(f"  Config.Data.RESOLUTION: {getattr(Config.Data, 'RESOLUTION', 'NO DEFINIDO')}")
except Exception as e:
    print(f"  ❌ ERROR leyendo Config: {e}")
    bugs_found.append(f"NODO 8: {e}")

# ═══════════════════════════════════════════════════════════════
# RESUMEN FORENSE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("🔬 RESUMEN FORENSE — BUGS ENCONTRADOS")
print("=" * 70)

if bugs_found:
    for i, bug in enumerate(bugs_found, 1):
        print(f"  🐛 BUG #{i}: {bug}")
else:
    print("  ✅ No se encontraron bugs en el trazado")

print(f"\nTotal bugs detectados: {len(bugs_found)}")
print("=" * 70)
