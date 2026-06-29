"""
🔬 OPTIMIZADOR MASIVO DE FLOTA (MASSIVE FLEET OPTIMIZER) — Trader Gemini
========================================================================
QUÉ: Ejecuta miles de backtests en paralelo sobre meses de datos históricos
     para calibrar perfiles específicos por símbolo (TP/SL, confianza, veto).
POR QUÉ: Para maximizar la esperanza matemática del capital de $13 USD,
         encontrando perfiles específicos de volatilidad y detectando fallas.
PARA QUÉ: Lograr un 100% de consistencia, evitar el decaimiento por comisiones,
          y diagnosticar sesgos de dirección (short-bias) y trampas de tendencia.
CÓMO: Paralelización con multiprocessing.Pool, descarga determinista de Binance,
      barrido de rejilla multidimensional y generación de reportes de fallas.
"""

import sys
import os
import time
import json
import io
import contextlib
import argparse
import multiprocessing
from queue import Queue
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Asegurar que se detecte la raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backtest_infra import fetch_binance_data, calculate_metrics, BacktestPortfolio, INITIAL_CAPITAL, LEVERAGE
from strategies.technical import HybridScalpingStrategy
from core.events import MarketEvent, SignalEvent
from core.enums import SignalType
from config import Config
from core.backtest_infra import BacktestDataProvider

# Configuración de los rangos de barrido
SL_ATR_MULTS = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]      # Multiplicadores de ATR para Stop Loss
TP_RR_RATIOS = [1.5, 1.8, 2.0, 2.5, 3.0]          # Ratio R:R (TP / SL)
MIN_CONFIDENCES = [0.50, 0.55, 0.60, 0.65, 0.70]   # Umbrales mínimos de confianza de señal
VETO_THRESHOLDS = [0.55, 0.60, 0.65]              # Umbrales de veto de Exit Oracle

def run_single_backtest_task(args):
    """
    Worker function que ejecuta un único backtest para una combinación de parámetros.
    """
    # Mock structure analyzer to bypass expensive SMC context calculations in backtest workers
    import sys
    try:
        from data.market_structure import structure_analyzer
        structure_analyzer.evaluate_market_context = lambda data: {}
    except Exception:
        pass

    symbol, data, sl_atr_mult, tp_rr_ratio, min_confidence, veto_threshold, days, horizon = args
    
    import logging
    logging.getLogger('trader_gemini').setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)
    
    try:
        events_queue = Queue()
        historical_data = {symbol: data}
        data_provider = BacktestDataProvider(events_queue, [symbol], historical_data)
        
        # Simulación de comisiones realistas de Binance Futures (0.02% Maker)
        commission_rate = 0.0002
        portfolio = BacktestPortfolio(INITIAL_CAPITAL, LEVERAGE)
        
        # Instanciar estrategia real con el horizonte especificado
        strategy = HybridScalpingStrategy(data_provider, events_queue, horizon=horizon)
        
        warmup_bars = 100
        bar_count = 0
        trades_executed = 0
        signals_filtered = 0
        
        # Métricas de control de fallas
        commission_paid = 0.0
        short_trades = 0
        short_wins = 0
        long_trades = 0
        long_wins = 0
        
        while data_provider.continue_backtest:
            data_provider.update_bars()
            bar_count += 1
            if bar_count < warmup_bars:
                continue
                
            bars = data_provider.get_latest_bars(symbol, 1)
            if not bars:
                continue
                
            current_bar = bars[-1]
            price = float(current_bar['close'])
            ts = datetime.fromtimestamp(current_bar['timestamp'] / 1000.0, tz=timezone.utc)
            high = float(current_bar['high'])
            low = float(current_bar['low'])
            
            # 1. Gestión de Salidas (Evaluación de SL/TP e indicación del Oracle)
            if symbol in portfolio.positions:
                pos = portfolio.positions[symbol]
                entry = pos['entry']
                side = pos['side']
                sl_price = pos.get('sl_price')
                tp_price = pos.get('tp_price')
                
                # Check hits
                exit_price = None
                exit_reason = None
                
                if side == 'LONG':
                    if low <= sl_price:
                        exit_price = sl_price
                        exit_reason = 'STOP_LOSS'
                    elif high >= tp_price:
                        exit_price = tp_price
                        exit_reason = 'TAKE_PROFIT'
                else:  # SHORT
                    if high >= sl_price:
                        exit_price = sl_price
                        exit_reason = 'STOP_LOSS'
                    elif low <= tp_price:
                        exit_price = tp_price
                        exit_reason = 'TAKE_PROFIT'
                        
                if exit_price:
                    pos_to_close = portfolio.positions[symbol]
                    size_usd_val = pos_to_close['size_usd']
                    side = pos_to_close['side']
                    trade = portfolio.close_position(symbol, exit_price, ts)
                    if trade:
                        trades_executed += 1
                        commission = size_usd_val * commission_rate
                        commission_paid += commission
                        
                        # Tracking de dirección
                        if side == 'LONG':
                            long_trades += 1
                            if trade['pnl_usd'] > 0: long_wins += 1
                        else:
                            short_trades += 1
                            if trade['pnl_usd'] > 0: short_wins += 1
            
            # 2. Calcular Señales de la Estrategia
            strategy.bought[symbol] = symbol in portfolio.positions
            market_event = MarketEvent(symbol=symbol, close_price=price, timestamp=ts)
            strategy.calculate_signals(market_event)
            
            # 3. Procesar Señales de la Cola
            while not events_queue.empty():
                event = events_queue.get()
                if not isinstance(event, SignalEvent):
                    continue
                    
                # Señal de salida explícita (Exit Oracle con veto_threshold)
                if event.signal_type == SignalType.EXIT:
                    if symbol in portfolio.positions:
                        # Simulamos veredicto del Exit Oracle usando la fuerza del consenso
                        strength = getattr(event, 'strength', 0.5)
                        if strength >= veto_threshold:
                            pos_to_close = portfolio.positions[symbol]
                            size_usd_val = pos_to_close['size_usd']
                            side = pos_to_close['side']
                            trade = portfolio.close_position(symbol, price, ts)
                            if trade:
                                trades_executed += 1
                                commission = size_usd_val * commission_rate
                                commission_paid += commission
                                
                                if side == 'LONG':
                                    long_trades += 1
                                    if trade['pnl_usd'] > 0: long_wins += 1
                                else:
                                    short_trades += 1
                                    if trade['pnl_usd'] > 0: short_wins += 1
                    continue
                
                # Señal de entrada
                if symbol not in portfolio.positions:
                    # Filtro de confianza de señal
                    signal_strength = getattr(event, 'strength', 0.5)
                    if signal_strength < min_confidence:
                        signals_filtered += 1
                        continue
                        
                    # ATR local para stop asimétrico
                    meta = event.metadata if event.metadata else {}
                    atr = meta.get('atr', price * 0.005) # fallback 0.5%
                    if atr <= 0:
                        atr = price * 0.005
                        
                    # Fórmulas de TP/SL Dinámicas por ATR
                    sl_pct = (atr / price) * sl_atr_mult
                    sl_pct = np.clip(sl_pct, 0.0015, 0.05) # Clamped bounds
                    tp_pct = sl_pct * tp_rr_ratio
                    tp_pct = np.clip(tp_pct, 0.003, 0.10)
                    
                    # Tamaño de posición basado en capital de $13 USD
                    cap = portfolio.current_capital
                    risk_pct = 0.05 # 5% de riesgo máximo
                    risk_usd = cap * risk_pct
                    size_usd = risk_usd / sl_pct
                    
                    # Margin limit checks
                    max_size = cap * LEVERAGE
                    size_usd = min(size_usd, max_size)
                    if size_usd < 5.0:
                        size_usd = 5.0 # Mínimo notional
                    
                    if size_usd > max_size:
                        continue
                        
                    side = 'LONG' if event.signal_type == SignalType.LONG else 'SHORT'
                    if side == 'LONG':
                        sl_price = price * (1 - sl_pct)
                        tp_price = price * (1 + tp_pct)
                    else:
                        sl_price = price * (1 + sl_pct)
                        tp_price = price * (1 - tp_pct)
                        
                    opened = portfolio.open_position_with_metadata(
                        symbol, side, price, size_usd, ts, meta, sl_price, tp_price
                    )
                    if opened:
                        trades_executed += 1
                        commission = size_usd * commission_rate
                        commission_paid += commission
            
            if bar_count % 60 == 0:
                portfolio.update_equity(ts)
                
        # Forzar cierre de posiciones restantes al final del backtest
        for sym_k in list(portfolio.positions.keys()):
            bars = data_provider.get_latest_bars(sym_k, 1)
            if bars:
                dt_close = datetime.fromtimestamp(bars[-1]['timestamp'] / 1000.0, tz=timezone.utc)
                pos_to_close = portfolio.positions[sym_k]
                size_usd_val = pos_to_close['size_usd']
                side = pos_to_close['side']
                trade = portfolio.close_position(sym_k, float(bars[-1]['close']), dt_close)
                if trade:
                    trades_executed += 1
                    commission = size_usd_val * commission_rate
                    commission_paid += commission
                    
                    if side == 'LONG':
                        long_trades += 1
                        if trade['pnl_usd'] > 0: long_wins += 1
                    else:
                        short_trades += 1
                        if trade['pnl_usd'] > 0: short_wins += 1
                    
        # Calcular métricas finales
        metrics = calculate_metrics(portfolio)
        net_pnl = portfolio.current_capital - portfolio.initial_capital
        
        # Calcular tasa de éxito por dirección
        long_wr = (long_wins / long_trades * 100) if long_trades > 0 else 0.0
        short_wr = (short_wins / short_trades * 100) if short_trades > 0 else 0.0
        
        # Puntuación compuesta (Score): PnL + Sharpe bonus - Penalización de Drawdown
        score = net_pnl + (metrics['sharpe_ratio'] * 0.5) - (metrics['max_drawdown_pct'] * 0.2)
        
        return {
            'symbol': symbol,
            'sl_atr_mult': sl_atr_mult,
            'tp_rr_ratio': tp_rr_ratio,
            'min_confidence': min_confidence,
            'veto_threshold': veto_threshold,
            'pnl': net_pnl,
            'return_pct': metrics['total_return'],
            'sharpe': metrics['sharpe_ratio'],
            'win_rate': metrics['win_rate'],
            'max_dd': metrics['max_drawdown_pct'],
            'trades': metrics['total_trades'],
            'commission_paid': commission_paid,
            'signals_filtered': signals_filtered,
            'long_trades': long_trades,
            'long_wr': long_wr,
            'short_trades': short_trades,
            'short_wr': short_wr,
            'score': score,
            'status': 'OK'
        }
    except Exception as e:
        return {
            'symbol': symbol,
            'sl_atr_mult': sl_atr_mult,
            'tp_rr_ratio': tp_rr_ratio,
            'min_confidence': min_confidence,
            'veto_threshold': veto_threshold,
            'status': 'ERROR',
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Massive Fleet Optimizer - Trader Gemini")
    parser.add_argument("--days", type=int, default=30, help="Number of historical days to backtest")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Number of parallel workers")
    parser.add_argument("--horizon", type=str, default="SCALPING", choices=["SCALPING", "SWING"], help="Horizon to optimize")
    args = parser.parse_args()
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']
    days = args.days
    workers = args.workers
    horizon = args.horizon.upper()
    
    global SL_ATR_MULTS, TP_RR_RATIOS, MIN_CONFIDENCES, VETO_THRESHOLDS
    if horizon == "SWING":
        # Rango enfocado para Swing (menor densidad, mayor paso temporal)
        SL_ATR_MULTS = [0.4, 0.6, 0.8]        # Multiplicadores de ATR Swing típicos
        TP_RR_RATIOS = [2.0, 2.5, 3.0]        # Ratios de Payout Swing típicos
        MIN_CONFIDENCES = [0.45, 0.50]        # Swing requiere menor fuerza de confluencia por la baja frecuencia
        VETO_THRESHOLDS = [0.55, 0.60]        # Umbrales del oráculo
    else:
        # Scalping (Rejilla estándar amplia)
        SL_ATR_MULTS = [0.4, 0.6, 0.8, 1.0]
        TP_RR_RATIOS = [1.5, 2.0, 2.5, 3.0]
        MIN_CONFIDENCES = [0.55, 0.60, 0.65, 0.70]
        VETO_THRESHOLDS = [0.55, 0.60, 0.65]
    
    print("=" * 80)
    print(f"🔬 SWEEP INICIADO: {len(symbols)} monedas | {days} días de historia | {workers} workers")
    print("=" * 80)
    
    # 1. Descargar datos históricos para cada símbolo con caché local
    all_data = {}
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    for sym in symbols:
        clean_sym = sym.replace('/', '_')
        cache_file = os.path.join(cache_dir, f"{clean_sym}_{days}d.csv")
        
        if os.path.exists(cache_file):
            print(f"📂 Cargando {sym} desde caché local ({cache_file})...")
            try:
                df = pd.read_csv(cache_file, parse_dates=['datetime'])
                df.set_index('datetime', inplace=True)
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)
                if len(df) > 1000:
                    all_data[sym] = df
                    print(f"   ✅ {len(df):,} velas cargadas desde caché.")
                    continue
            except Exception as e:
                print(f"   ⚠️ Error leyendo caché para {sym}: {e}, descargando de nuevo...")
                
        print(f"📡 Descargando datos kline para {sym} ({days} días)...")
        try:
            df = fetch_binance_data(sym, days=days)
            if df is not None and len(df) > 1000:
                all_data[sym] = df
                df.to_csv(cache_file)
                print(f"   ✅ {len(df):,} velas descargadas y guardadas en caché.")
            else:
                print(f"   ⚠️ Datos insuficientes para {sym}, omitiendo.")
        except Exception as e:
            print(f"   ❌ Error descargando {sym}: {e}")
            
    if not all_data:
        print("🚨 Error fatal: No se pudo descargar información para ningún activo.")
        sys.exit(1)
        
    # 2. Pre-remuestrear datos en formato estructurado una sola vez para evitar overhead en workers
    print("\n⚙️ Pre-remuestreando y preparando structured arrays para evitar overhead repetitivo...")
    pre_resampled_data = {}
    for sym, df in all_data.items():
        try:
            # Creamos un BacktestDataProvider temporal para inicializar los struct data de la moneda
            temp_provider = BacktestDataProvider(Queue(), [sym], {sym: df.copy()})
            pre_resampled_data[sym] = temp_provider.struct_data[sym]
            print(f"   ✅ {sym} pre-remuestreado exitosamente para todos los timeframes.")
        except Exception as e:
            print(f"   ❌ Error pre-remuestreando {sym}: {e}")
            
    if not pre_resampled_data:
        print("🚨 Error fatal: No se pudo pre-remuestrear ningún activo.")
        sys.exit(1)
        
    # 3. Generar lista de tareas para el barrido usando datos pre-procesados
    tasks = []
    for sym, struct in pre_resampled_data.items():
        for sl in SL_ATR_MULTS:
            for tp in TP_RR_RATIOS:
                for conf in MIN_CONFIDENCES:
                    for veto in VETO_THRESHOLDS:
                        tasks.append((sym, struct, sl, tp, conf, veto, days, horizon))
                        
    print(f"🎯 Total de simulaciones a ejecutar: {len(tasks):,} backtests...")
    
    start_time = time.time()
    
    # 3. Ejecutar simulaciones en paralelo con reporte de progreso en tiempo real
    results = []
    completed = 0
    total_tasks = len(tasks)
    
    with multiprocessing.Pool(processes=workers) as pool:
        for res in pool.imap_unordered(run_single_backtest_task, tasks):
            results.append(res)
            completed += 1
            if completed % 100 == 0 or completed == total_tasks:
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                remaining = avg_time * (total_tasks - completed)
                print(f"▓ PROGRESO: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%) | "
                      f"Transcurrido: {elapsed:.1f}s | Restante Est: {remaining:.1f}s | Tasa: {1/avg_time:.1f} backtests/s")
        
    duration = time.time() - start_time
    print(f"⏱️ Barrido masivo finalizado en {duration:.1f}s ({duration/60:.2f}m)")
    
    # 4. Procesar resultados
    successful_results = [r for r in results if r['status'] == 'OK']
    error_results = [r for r in results if r['status'] == 'ERROR']
    
    if error_results:
        print(f"⚠️ {len(error_results)} simulaciones fallaron con error.")
        for err in error_results[:5]:
            print(f"   - Error en {err['symbol']}: {err['error']}")
            
    if not successful_results:
        print("🚨 Error fatal: Todas las simulaciones fallaron.")
        sys.exit(1)
        
    df_res = pd.DataFrame(successful_results)
    
    # 5. Hallar la mejor configuración para cada moneda
    optimal_profiles = {}
    print("\n" + "="*80)
    print("🏆 RESULTADOS ÓPTIMOS ENCONTRADOS POR ACTIVO")
    print("="*80)
    
    for sym in all_data.keys():
        sym_df = df_res[df_res['symbol'] == sym]
        if sym_df.empty:
            continue
            
        # El ganador se define por el Score compuesto
        winner = sym_df.loc[sym_df['score'].idxmax()]
        
        # Identificar la configuración con mayor drawdown para auditar fallas
        worst_dd = sym_df.loc[sym_df['max_dd'].idxmax()]
        
        optimal_profiles[sym] = {
            'sl_atr_mult': float(winner['sl_atr_mult']),
            'tp_rr_ratio': float(winner['tp_rr_ratio']),
            'min_confidence': float(winner['min_confidence']),
            'veto_threshold': float(winner['veto_threshold']),
            'pnl': float(winner['pnl']),
            'return_pct': float(winner['return_pct']),
            'sharpe': float(winner['sharpe']),
            'win_rate': float(winner['win_rate']),
            'max_dd': float(winner['max_dd']),
            'trades': int(winner['trades']),
            'commission_paid': float(winner['commission_paid']),
            'long_trades': int(winner['long_trades']),
            'long_wr': float(winner['long_wr']),
            'short_trades': int(winner['short_trades']),
            'short_wr': float(winner['short_wr']),
            'score': float(winner['score'])
        }
        
        print(f"🥇 {sym}: PnL: ${winner['pnl']:+.3f} | Sharpe: {winner['sharpe']:.2f} | WinRate: {winner['win_rate']:.1f}% | DD: {winner['max_dd']:.2f}% | Trades: {winner['trades']}")
        print(f"   Config óptima: SL Mult={winner['sl_atr_mult']}x ATR | TP Ratio={winner['tp_rr_ratio']}:1 | Confianza ≥ {winner['min_confidence']} | Veto ≥ {winner['veto_threshold']}")
        print(f"   Fallas detectadas en este activo:")
        
        # Diagnóstico de fallas:
        # A. Comisión excesiva (fee erosion)
        comm_pct_of_pnl = (winner['commission_paid'] / winner['pnl'] * 100) if winner['pnl'] > 0 else 999.0
        if comm_pct_of_pnl > 30.0:
            print(f"      ☢️ FEE EROSION: Las comisiones representan el {comm_pct_of_pnl:.1f}% del PnL neto. Se sugiere subir el umbral de confianza.")
            
        # B. Sesgo de dirección (short-bias o long-bias)
        if winner['long_trades'] > 5 and winner['short_trades'] > 5:
            wr_diff = abs(winner['long_wr'] - winner['short_wr'])
            if wr_diff > 20.0:
                print(f"      ☢️ DIRECTIONAL BIAS: WinRate Long ({winner['long_wr']:.1f}%) y Short ({winner['short_wr']:.1f}%) difieren por {wr_diff:.1f}%. Ajustar asimetría de stops.")
                
        # C. Drawdown extremo observado en configuraciones incorrectas
        print(f"      ☢️ PEOR DRAWDOWN: Una mala configuración llevó a un DD del {worst_dd['max_dd']:.1f}% (SL={worst_dd['sl_atr_mult']}x, TP={worst_dd['tp_rr_ratio']}:1).")
        print("-" * 80)
        
    # 6. Guardar perfiles óptimos como JSON por horizonte y actualizando el central
    file_name = f'optimal_profiles_{horizon.lower()}.json'
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), file_name)
    with open(output_path, 'w') as f:
        json.dump(optimal_profiles, f, indent=2)
    print(f"\n💾 Perfiles óptimos guardados en: {output_path}")
    
    # También actualizamos el central optimal_profiles.json estructurado por horizontes
    central_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'optimal_profiles.json')
    central_data = {}
    if os.path.exists(central_path):
        try:
            with open(central_path, 'r') as f:
                central_data = json.load(f)
        except Exception:
            pass
            
    central_data[horizon] = optimal_profiles
    with open(central_path, 'w') as f:
        json.dump(central_data, f, indent=2)
    print(f"💾 Central optimal_profiles.json actualizado para el horizonte {horizon}")
    
    # 7. Imprimir diccionario para pegar en core/asset_parameter_engine.py si es necesario
    print("\n📋 CÓDIGO GENERADO PARA ESTRUCTURA DYNAMIC_PROFILES (COPIAR Y PEGAR):")
    print("=" * 80)
    print("DYNAMIC_PROFILES = {")
    for sym, prof in optimal_profiles.items():
        print(f"    '{sym}': {{'sl_atr_mult': {prof['sl_atr_mult']:.2f}, 'tp_rr_ratio': {prof['tp_rr_ratio']:.2f}, 'min_confidence': {prof['min_confidence']:.2f}, 'veto_threshold': {prof['veto_threshold']:.2f}}},")
    print("}")
    print("=" * 80)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
