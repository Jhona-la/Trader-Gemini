import os
import sys
import math
import time
import json
import uuid
import random
import numpy as np
from datetime import datetime, timezone
import argparse

# Añadir raíz al path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Importamos infraestructura original
from core.backtest_infra import BacktestDataProvider, fetch_binance_data
import core.backtest_infra
from scripts.run_god_mode_backtest import run_global_backtest

# Use the exact MarketEvent class from backtest_infra
MarketEvent = core.backtest_infra.MarketEvent

# ════════════════════════════════════════════════════════════════
# MONKEY-PATCH: Inyección de Estrés de Red y Dark Alpha Sintético
# QUÉ: Intercepta el flujo de MarketEvents para simular condiciones
#   adversas de producción (micro-gaps, slippage, liquidation squeezes).
# POR QUÉ: Backtest sin estrés sobreestima PnL. En producción hay
#   latencia de red, gaps de datos, y squeezes de liquidación.
# CÓMO: Monkey-patch sobre update_bars() interceptando events_queue.put().
#   Usa object.__setattr__() para mutar frozen dataclasses.
# ════════════════════════════════════════════════════════════════
_original_update_bars = BacktestDataProvider.update_bars

def patched_update_bars(self):
    """
    Fase IV: Simulación de Flujo y Calidad de Datos
    Llama al update_bars original pero intercepta events.put para inyectar anomalías.
    """
    if getattr(self, '_stress_random', None) is None:
        self._stress_random = random.Random(42)  # Determinismo
        
    original_put = self.events_queue.put
    
    def intercepted_put(item):
        # Probabilidad de Micro-Gaps (Faltan ticks) — 1% chance
        if self._stress_random.random() < 0.01:
            return  # Drop event
            
        if hasattr(item, "high_price") and hasattr(item, "low_price"):
            # Simulation of slippage due to latency / liquidity — 5% chance
            if self._stress_random.random() < 0.05:
                if hasattr(item, "close_price"):
                    object.__setattr__(item, "close_price", item.close_price * 0.9995)
            
            # Liquidation squeeze detection & slippage
            if hasattr(item, "low_price") and hasattr(item, "high_price"):
                if item.low_price and item.low_price > 0:
                    if (item.high_price - item.low_price) / item.low_price > 0.01:
                        object.__setattr__(item, "close_price", item.close_price * 0.998)
                    
        original_put(item)
        
    self.events_queue.put = intercepted_put
    try:
        _original_update_bars(self)
    finally:
        self.events_queue.put = original_put

# Aplicar el parche
BacktestDataProvider.update_bars = patched_update_bars


def calculate_duplication_time(final_capital, initial_capital, days):
    """
    Fase II: El Tiempo de Duplicación
    T_duplicacion = log(2) / log(1 + retorno_fraccional_por_tick_promedio)
    """
    if final_capital <= initial_capital:
        return 999.0  # Infinito / No duplicó
        
    # Retorno diario promedio
    roi_total = final_capital / initial_capital
    retorno_diario = (roi_total ** (1/days)) - 1
    
    if retorno_diario <= 0:
        return 999.0
        
    t_duplicacion = math.log(2) / math.log(1 + retorno_diario)
    return t_duplicacion

def run_mirror(symbols, days=7, initial_capital=13.0, scenario="A", isolated_strategy="omni"):
    """
    Ejecuta un Espejo Absoluto del motor de producción.
    
    QUÉ: Wrapper sobre run_global_backtest con inyección de estrés.
    POR QUÉ: Para medir el comportamiento real bajo condiciones adversas.
    PARA QUÉ: Alimentar el Oráculo Surrogado con datos representativos.
    
    Args:
        symbols: Lista de símbolos a operar
        days: Días de datos históricos
        initial_capital: Capital inicial
        scenario: Escenario forense (A=normal, B=sin exits reactivos, etc.)
        isolated_strategy: "omni" usa solo OmniStrategy (sin ML training),
                           None usa el pipeline completo (requiere más datos)
    """
    # 1. Preparar data
    all_data = {}
    for sym in symbols:
        df = fetch_binance_data(sym, days=days)
        if df is not None and not df.empty:
            all_data[sym] = df
            
    if not all_data:
        return {"TD": 999.0, "Sharpe": 0.0, "MaxDD": 1.0, "PnL": -13.0, "TotalTrades": 0}

    # 2. Correr Espejo
    try:
        results = run_global_backtest(
            all_data=all_data,
            symbols=symbols,
            days=days,
            initial_capital=initial_capital,
            verbose=False,
            seed=random.randint(1, 10000),
            scenario=scenario,
            isolated_strategy=isolated_strategy
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error en espejo: {e}")
        return {"TD": 999.0, "Sharpe": 0.0, "MaxDD": 1.0, "PnL": -13.0, "TotalTrades": 0}
        
    if not results or "metrics" not in results:
        return {"TD": 999.0, "Sharpe": 0.0, "MaxDD": 1.0, "PnL": -13.0, "TotalTrades": 0}
        
    # ════════════════════════════════════════════════════════════════
    # FORENSIC FIX: CORRECT METRICS KEY MAPPING
    # QUÉ: Las claves del dict de métricas son snake_case, no Title Case.
    # POR QUÉ: run_global_backtest retorna "final_capital" no "Final Equity".
    #   Esta divergencia causaba que run_mirror() siempre leyera defaults (0).
    # PARA QUÉ: El Oráculo Surrogado recibe datos REALES, no zeros.
    # ════════════════════════════════════════════════════════════════
    metrics = results["metrics"]
    final_cap = metrics.get("final_capital", initial_capital)
    max_dd = metrics.get("max_drawdown_pct", 100.0) / 100.0
    sharpe = metrics.get("sharpe_ratio", 0.0)
    trades = metrics.get("total_trades", 0)
    
    td = calculate_duplication_time(final_cap, initial_capital, days)
    
    # Clean memory globally
    import gc
    try:
        from core.omniscient_registry import registry
        from core.consensus_filter import _consensus_filter as consensus_filter
        if hasattr(registry, '_metrics'):
            registry._metrics.clear()
        if hasattr(registry, 'active_positions'):
            registry.active_positions.clear()
        if hasattr(consensus_filter, 'last_n_trades'):
            consensus_filter.last_n_trades.clear()
    except Exception:
        pass
    gc.collect()
    
    return {
        "TD": td,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "PnL": final_cap - initial_capital,
        "TotalTrades": trades
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--strategy", type=str, default="omni", help="omni=OmniOnly, None=Full ML")
    args = parser.parse_args()
    
    iso = args.strategy if args.strategy != "None" else None
    print(f"🪞 Iniciando ESPEJO ABSOLUTO ({args.days} days, strategy={iso})...")
    res = run_mirror(["BTC/USDT"], days=args.days, isolated_strategy=iso)
    print("\n" + "="*40)
    print("📈 RESULTADOS DEL ESPEJO ABSOLUTO")
    print("="*40)
    print(f"Tiempo Duplicación: {res['TD']:.2f} días")
    print(f"Sharpe Ratio:       {res['Sharpe']:.2f}")
    print(f"Max Drawdown:       {res['MaxDD']*100:.2f}%")
    print(f"PnL Bruto:          ${res['PnL']:.2f}")
    print(f"Trades Totales:     {res['TotalTrades']}")
    print("="*40)
