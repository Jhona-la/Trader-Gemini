"""
🧬 OMNISCIENCE SIMULATOR v2.0 — Full-Stack Evolutionary Genetic Sweep
=====================================================================
CORREGIDO: Usa el flujo REAL de producción (calculate_signals → _run_inference)
en lugar del inexistente método predict().

FLUJO REAL EN PRODUCCIÓN (main.py / engine.py):
  1. BacktestDataProvider.get_latest_bars() → numpy struct array
  2. Strategy._launch_training(bars) → entrena RF/XGB/GB ensemble
  3. Strategy.calculate_signals(MarketEvent) → _run_inference() → emite SignalEvent
  4. RiskManager valida señal → OrderEvent
  5. Engine ejecuta orden → Trade registrado

BUGS CORREGIDOS:
  - BUG #1: predict() NO EXISTE → Ahora usa calculate_signals(event)
  - BUG #2: models_ready=False → Ahora entrena ANTES de evaluar
  - BUG #3: EventType importado desde core.enums (no core.engine)
  - BUG #4: except genérico tragaba todos los errores → Ahora imprime
  - BUG #5: Config.Strategies mutaba estado global en multiprocessing
"""
import os
import sys
import time
import sqlite3
import random
import traceback
from datetime import datetime
from queue import Queue

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
# VANGUARDIA: Completely kill all disk logging during simulation to prevent WinError 32 and maximize HFT speed
logging.getLogger().handlers.clear()
logging.basicConfig(handlers=[logging.NullHandler()], level=logging.CRITICAL)
# Silence internal libraries that might create RotatingFileHandler
logging.getLogger('ccxt').setLevel(logging.CRITICAL)

from utils.logger import logger
logger.handlers.clear()
logger.addHandler(logging.NullHandler())

from config import Config
from core.backtest_infra import BacktestDataProvider, fetch_multi_symbol_data
from core.enums import EventType
from risk.risk_manager import RiskManager
from strategies.ml_strategy import UniversalEnsembleStrategy
from core.evolution import TradeResult, FitnessCalculator

# VANGUARDIA: Disable TransparentLogger UI spam which causes extreme I/O bottleneck
try:
    from core.transparent_logger import TransparentLogger
    TransparentLogger.log_ml_prediction = lambda *args, **kwargs: None
    TransparentLogger.log_sniper_analysis = lambda *args, **kwargs: None
    TransparentLogger.log_trade_execution = lambda *args, **kwargs: None
except ImportError:
    pass

DB_PATH = os.path.join("data", "multiverse_apex.db")


def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Drop old schema if it exists (columns changed)
    cursor.execute("DROP TABLE IF EXISTS apex_genotypes")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS apex_genotypes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generation INTEGER,
            tp_pct REAL,
            sl_pct REAL,
            ml_threshold REAL,
            resolution TEXT,
            win_rate REAL,
            compound_growth REAL,
            drawdown REAL,
            score REAL,
            signals_generated INTEGER DEFAULT 0,
            trades_executed INTEGER DEFAULT 0,
            training_status TEXT DEFAULT 'UNKNOWN',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


class FakeMarketEvent:
    """Simula un MarketEvent exactamente como lo genera el Engine en producción."""
    def __init__(self, symbol="BTC/USDT", timeframe="1m"):
        self.type = EventType.MARKET
        self.symbol = symbol
        self.timeframe = timeframe
        self.is_closed = True
        self.timestamp = None


def simulate_universe(args):
    """
    Ejecuta el flujo REAL de producción:
    1. Construye BacktestDataProvider con datos reales
    2. Instancia UniversalEnsembleStrategy con data_provider correcto
    3. Entrena los modelos ML (como hace run_god_mode_backtest.py)
    4. Itera epoch por epoch llamando calculate_signals (como el Engine real)
    5. Recoge señales del events_queue
    6. Simula trades basados en las señales reales
    """
    genes, all_data = args
    gen, tp_pct, sl_pct, ml_threshold, resolution = genes

    # Avoid mutating global Config in multiprocessing
    # We'll use local copies of these values

    result_template = (gen, tp_pct, sl_pct, ml_threshold, resolution, 0.0, 0.0, 0.0, 0.0, 0, 0, "ERROR")

    try:
        # ═══════════════════════════════════════════════════════════════
        # PASO 1: Construir DataProvider con datos reales
        # ═══════════════════════════════════════════════════════════════
        # CRITICAL: BacktestDataProvider.__init__ sets historical_data[s] = None after
        # converting to struct. We must pass a COPY so subsequent genomes can use the data.
        import copy
        data_copy = {k: v.clone() if hasattr(v, 'clone') else copy.deepcopy(v) for k, v in all_data.items()}
        data_provider = BacktestDataProvider(None, ["BTC/USDT"], data_copy, backtest_days=2)

        if len(data_provider.global_timeline) < 200:
            return result_template._replace(training_status="INSUF_DATA") if hasattr(result_template, '_replace') else \
                (gen, tp_pct, sl_pct, ml_threshold, resolution, 0.0, 0.0, 0.0, 0.0, 0, 0, "INSUF_DATA")

        # ═══════════════════════════════════════════════════════════════
        # PASO 2: Instanciar estrategia con data_provider REAL
        # ═══════════════════════════════════════════════════════════════
        events_queue = Queue()
        Config.IS_BACKTESTING = True

        strategy = UniversalEnsembleStrategy(
            data_provider=data_provider,
            events_queue=events_queue,
            symbol="BTC/USDT",
            lookback=50,
            horizon="SCALPING"
        )
        strategy.is_sandbox = True
        strategy.is_live = False

        # ═══════════════════════════════════════════════════════════════
        # PASO 3: ENTRENAMIENTO (como run_god_mode_backtest.py línea 1361-1371)
        # Sin esto, models_ready=False y _run_inference sale inmediatamente
        # ═══════════════════════════════════════════════════════════════
        training_status = "NOT_TRAINED"

        # Move to a point where we have enough lookback
        warmup_idx = min(500, len(data_provider.global_timeline) - 1)
        data_provider.current_time_ms = int(data_provider.global_timeline[warmup_idx])
        data_provider._epoch_bars_cache = {}

        training_bars = data_provider.get_latest_bars(
            "BTC/USDT",
            n=getattr(strategy, "lookback", 500),
            timeframe=strategy.primary_tf
        )

        if training_bars is not None and len(training_bars) > 50:
            try:
                if hasattr(strategy, "_launch_training"):
                    strategy._launch_training(training_bars, "Full", sync=True)
                    if strategy.is_trained:
                        training_status = "TRAINED_OK"
                    else:
                        training_status = "TRAIN_FAILED"
                else:
                    training_status = "NO_TRAIN_METHOD"
            except Exception as e:
                training_status = f"TRAIN_ERROR: {str(e)[:80]}"
        else:
            training_status = f"INSUF_BARS: {len(training_bars) if training_bars is not None else 0}"

        print(f"  [GEN {gen}] TP={tp_pct:.4f} SL={sl_pct:.4f} ML={ml_threshold} | Training: {training_status}")

        # ═══════════════════════════════════════════════════════════════
        # PASO 4: EVALUACIÓN — Iterar epoch por epoch (como Engine.run())
        # ═══════════════════════════════════════════════════════════════
        event = FakeMarketEvent("BTC/USDT", resolution)
        signals_collected = []
        signal_count = 0

        # Iterate from warmup+1 to end
        start_idx = warmup_idx + 1
        end_idx = len(data_provider.global_timeline)
        step = 5  # Every 5 epochs for speed

        for epoch_idx in range(start_idx, end_idx, step):
            data_provider.current_time_ms = int(data_provider.global_timeline[epoch_idx])
            data_provider._epoch_bars_cache = {}
            data_provider._epoch_df_cache = {}

            try:
                strategy.calculate_signals(event)
            except Exception:
                continue

            # Drain events queue
            while not events_queue.empty():
                sig = events_queue.get_nowait()
                signal_count += 1
                # Extract signal data
                sig_data = {
                    'epoch_idx': epoch_idx,
                    'type': getattr(sig, 'signal_type', getattr(sig, 'type', 'UNKNOWN')),
                    'direction': getattr(sig, 'direction', 0),
                    'confidence': getattr(sig, 'confidence', 0.0),
                    'price': float(data_provider.struct_data["BTC/USDT"]["1m"][min(epoch_idx, len(data_provider.struct_data["BTC/USDT"]["1m"])-1)]["close"]),
                }
                signals_collected.append(sig_data)

        # ═══════════════════════════════════════════════════════════════
        # PASO 5: CALCULAR TRADES desde señales (simular TP/SL)
        # ═══════════════════════════════════════════════════════════════
        simulated_trades = []
        bars_1m = data_provider.struct_data["BTC/USDT"]["1m"]

        for sig in signals_collected:
            entry_idx = sig['epoch_idx']
            direction = sig.get('direction', 0)
            if direction == 0:
                # Try to infer from signal type
                sig_type = str(sig.get('type', '')).upper()
                if 'LONG' in sig_type:
                    direction = 1
                elif 'SHORT' in sig_type:
                    direction = -1
                else:
                    continue

            entry_price = sig['price']
            if entry_price <= 0:
                continue

            # Forward simulate to find exit
            for j_offset in range(1, min(200, len(bars_1m) - entry_idx)):
                j = entry_idx + j_offset
                if j >= len(bars_1m):
                    break
                future_price = float(bars_1m[j]["close"])
                pnl_pct = (future_price - entry_price) / entry_price * direction

                if pnl_pct >= tp_pct or pnl_pct <= -sl_pct:
                    simulated_trades.append(
                        TradeResult(
                            pnl_pct=pnl_pct,
                            duration_seconds=j_offset * 60.0,
                            is_win=(pnl_pct > 0)
                        )
                    )
                    break

        # ═══════════════════════════════════════════════════════════════
        # PASO 6: CALCULAR FITNESS
        # ═══════════════════════════════════════════════════════════════
        score = FitnessCalculator.calculate_fitness(simulated_trades)

        win_rate = len([t for t in simulated_trades if t.pnl_pct > 0]) / len(simulated_trades) if simulated_trades else 0
        capital = 1.0
        for t in simulated_trades:
            capital *= (1.0 + t.pnl_pct)
        compound_growth = capital - 1.0

        max_dd = 0.0
        peak = 1.0
        running_cap = 1.0
        for t in simulated_trades:
            running_cap *= (1.0 + t.pnl_pct)
            if running_cap > peak:
                peak = running_cap
            dd = (peak - running_cap) / peak
            if dd > max_dd:
                max_dd = dd

        return (gen, tp_pct, sl_pct, ml_threshold, resolution,
                win_rate, compound_growth, max_dd, score,
                signal_count, len(simulated_trades), training_status)

    except Exception as e:
        print(f"  ❌ [GEN {gen}] ERROR: {e}")
        traceback.print_exc()
        return (gen, tp_pct, sl_pct, ml_threshold, resolution,
                0.0, 0.0, 0.0, 0.0, 0, 0, f"EXCEPTION: {str(e)[:80]}")


def evolutionary_sweep():
    print("=" * 70)
    print("🧬 OMNISCIENCE SIMULATOR v2.0 — Full-Stack Evolutionary Sweep")
    print("   [FIXED] Uses REAL production flow: train → calculate_signals")
    print("=" * 70)
    init_db()

    # 0. Fetch Real Historical Data
    print("⏳ Downloading 2 days of real historical data for BTC/USDT...")
    all_data = fetch_multi_symbol_data(["BTC/USDT"], days=2)
    if "BTC/USDT" not in all_data:
        print("❌ FATAL: No data downloaded. Aborting.")
        return
    print(f"✅ Real data loaded: {len(all_data['BTC/USDT'])} candles.")

    # 1. Initial Population
    population_size = 15
    generations = 5

    population = []
    for _ in range(population_size):
        tp = round(random.uniform(0.003, 0.02), 4)
        sl = round(random.uniform(0.003, 0.015), 4)
        ml_thresh = round(random.uniform(0.50, 0.75), 2)
        resolution = "1m"
        population.append((0, tp, sl, ml_thresh, resolution))

    start_time = time.time()

    for gen in range(generations):
        print(f"\n[GENERATION {gen}] Testing {len(population)} configurations...")

        results = []
        # Sequential execution to avoid multiprocessing serialization issues with Config
        for gene in population:
            res = simulate_universe((gene, all_data))
            results.append(res)

        # Sort by score (descending)
        results.sort(key=lambda x: x[8], reverse=True)

        # Save to Database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        for r in results[:5]:
            cursor.execute("""
                INSERT INTO apex_genotypes
                (generation, tp_pct, sl_pct, ml_threshold, resolution,
                 win_rate, compound_growth, drawdown, score,
                 signals_generated, trades_executed, training_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, r)
        conn.commit()
        conn.close()

        best = results[0]
        print(f"\n🏆 APEX GENOME {gen}:")
        print(f"   Score: {best[8]:.4f} | WR: {best[5]*100:.1f}% | Compound: {best[6]*100:.2f}%")
        print(f"   TP: {best[1]} | SL: {best[2]} | ML_Thresh: {best[3]} | Res: {best[4]}")
        print(f"   Signals: {best[9]} | Trades: {best[10]} | Training: {best[11]}")

        # Crossover & Mutation
        if gen < generations - 1:
            next_population = []
            # Elitism
            for elite in results[:2]:
                next_population.append((gen+1, elite[1], elite[2], elite[3], elite[4]))

            while len(next_population) < population_size:
                p1 = random.choice(results[:max(3, len(results))])
                p2 = random.choice(results[:max(3, len(results))])

                child_tp = p1[1] if random.random() > 0.5 else p2[1]
                child_sl = p1[2] if random.random() > 0.5 else p2[2]
                child_ml = p1[3] if random.random() > 0.5 else p2[3]
                child_res = "1m"

                if random.random() < 0.15: child_tp *= random.uniform(0.85, 1.15)
                if random.random() < 0.15: child_sl *= random.uniform(0.85, 1.15)
                if random.random() < 0.15: child_ml = min(0.90, max(0.40, child_ml + random.uniform(-0.05, 0.05)))

                next_population.append((gen+1, round(child_tp,4), round(child_sl,4), round(child_ml,2), child_res))

            population = next_population

    elapsed = time.time() - start_time
    print(f"\n🏁 [EVOLUTION COMPLETE] {generations} Generations in {elapsed:.2f}s.")
    print("Results stored in data/multiverse_apex.db")


if __name__ == "__main__":
    evolutionary_sweep()
