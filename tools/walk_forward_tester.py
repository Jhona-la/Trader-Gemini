"""
🧪 [PHASE 10] WALK-FORWARD TESTER - Trader Gemini
Validates strategy robustness by simulating periodic retraining cycles.

QUÉ: Motor de validación Walk-Forward (Entrenamiento → Validación → Desplazamiento).
POR QUÉ: Evita el overfitting y mide la capacidad de adaptación real del bot.
PARA QUÉ: Confirmar que el Sharpe Ratio > 2.0 se mantiene "extra-muestra".
CÓMO: Segmenta datos históricos en 'folds' de entrenamiento y testeo secuenciales.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from queue import Queue
import time
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from strategies.ml_strategy import MLStrategyHybridUltimate
from core.events import MarketEvent, SignalEvent, OrderEvent
from core.enums import SignalType
from utils.logger import logger
from tests.run_backtest import BacktestDataProvider, BacktestPortfolio, fetch_binance_data

class WalkForwardTester:
    """
    🧪 [PHASE 10.2] MULTI-SYMBOL WALK-FORWARD TESTER
    Validates robustness across the entire Elite Basket.
    
    👨‍🏫 MODO PROFESOR:
    - QUÉ: Evaluación recursiva de la estrategia en múltiples activos.
    - POR QUÉ: Un modelo que solo funciona en BTC es frágil. Buscamos universalidad.
    - PARA QUÉ: Identificar el 'Core Basket' de alta confianza para producción.
    """
    def __init__(self, symbols=None, train_days=7, test_days=2, total_days=14):
        self.symbols = symbols if symbols else ["BTC/USDT"]
        self.train_days = train_days
        self.test_days = test_days
        self.total_days = total_days
        self.all_results = {} # {symbol: [folds]}
        
    def run(self):
        print(f"🚀 Iniciando Multi-Symbol Walk-Forward Validation")
        print(f"📊 Canasta: {len(self.symbols)} activos")
        print(f"📅 Ventana: {self.train_days}d Train | {self.test_days}d Test | Total: {self.total_days}d")
        
        for symbol in self.symbols:
            print(f"\n{'='*60}")
            print(f"🔍 PROCESANDO: {symbol}")
            print(f"{'='*60}")
            
            # 1. Fetch data
            df = fetch_binance_data(symbol, self.total_days)
            if df is None or len(df) < 1440 * (self.train_days + self.test_days):
                print(f"❌ [{symbol}] No hay suficientes datos para el test.")
                continue

            # 2. Divide into folds
            total_bars = len(df)
            train_bars = 1440 * self.train_days
            test_bars = 1440 * self.test_days
            step_bars = test_bars
            
            start_idx = 0
            fold_idx = 1
            symbol_folds = []
            
            while start_idx + train_bars + test_bars <= total_bars:
                print(f"\n--- 📂 {symbol} | FOLD {fold_idx} ---")
                
                train_df = df.iloc[start_idx : start_idx + train_bars]
                test_df = df.iloc[start_idx + train_bars : start_idx + train_bars + test_bars]
                
                fold_metrics = self.execute_fold(symbol, train_df, test_df)
                symbol_folds.append({
                    'fold': fold_idx,
                    'start_date': test_df.index[0].isoformat(),
                    'metrics': fold_metrics
                })
                
                start_idx += step_bars
                fold_idx += 1
            
            self.all_results[symbol] = symbol_folds
            
        self.print_final_report()

    def execute_fold(self, symbol, train_df, test_df):
        """Processes a single walk-forward fold for a specific symbol."""
        events_queue = Queue()
        
        # Initialize components with Test Data + Warmup
        warmup_df = train_df.iloc[-1000:]
        test_with_warmup = pd.concat([warmup_df, test_df])
        
        historical_data = {symbol: test_with_warmup}
        data_provider = BacktestDataProvider(events_queue, [symbol], historical_data)
        for _ in range(len(warmup_df)):
            data_provider.update_bars()
            
        portfolio = BacktestPortfolio(Config.INITIAL_CAPITAL, Config.BINANCE_LEVERAGE)
        
        # Initialize Strategy
        strategy = MLStrategyHybridUltimate(data_provider, events_queue, symbol=symbol)
        strategy.models_dir = f".models_wfv_{symbol.replace('/', '_')}"
        os.makedirs(strategy.models_dir, exist_ok=True)
        strategy.is_trained = False
        
        # --- PHASE 10: TRAINING ---
        bars_for_prep = train_df.reset_index().rename(columns={'datetime': 'timestamp'}).to_dict('records')
        for b in bars_for_prep: b['datetime'] = b['timestamp']
        
        df_ready = strategy._prepare_features(bars_for_prep)
        if df_ready.empty or df_ready['close'].nunique() <= 1:
            print(f"⚠️ [{symbol}] Datos insuficientes o planos para entrenamiento.")
            return {'sharpe': 0, 'pnl': 0, 'win_rate': 0}

        print(f"🧠 Entrenando {symbol}...")
        result, score = strategy._train_with_cross_validation(df_ready)
        if result:
            models, scaler, feature_cols = result
            strategy.rf_model, strategy.xgb_model, strategy.gb_model = models['rf'], models['xgb'], models['gb']
            strategy.scaler, strategy._feature_cols = scaler, feature_cols
            strategy.is_trained = True
            print(f"✨ Score: {score:.3f}")
        else:
            print(f"⚠️ Score bajo: {score:.3f}")
            return {'sharpe': 0, 'pnl': 0, 'win_rate': 0}
            
        # Freeze and Validate
        strategy.retrain_interval = 999999
        strategy.min_bars_to_train = 100
        
        bar_count = 0
        while data_provider.continue_backtest:
            data_provider.update_bars()
            event = MarketEvent(symbol=symbol, close_price=test_df.iloc[bar_count]['close'])
            strategy.calculate_signals(event)
            
            while not events_queue.empty():
                signal = events_queue.get()
                if isinstance(signal, SignalEvent):
                    side = 'LONG' if signal.signal_type == SignalType.LONG else 'SHORT'
                    portfolio.open_position(
                        symbol, side, signal.current_price, 
                        portfolio.current_capital * 0.1, # 10% risk
                        signal.datetime,
                        sl_price=signal.current_price * (1 - strategy.BASE_SL_TARGET) if side == 'LONG' else signal.current_price * (1 + strategy.BASE_SL_TARGET),
                        tp_price=signal.current_price * (1 + strategy.BASE_TP_TARGET) if side == 'LONG' else signal.current_price * (1 - strategy.BASE_TP_TARGET)
                    )
            
            if symbol in portfolio.positions:
                pos = portfolio.positions[symbol]
                price = test_df.iloc[bar_count]['close']
                if (pos['side'] == 'LONG' and price <= pos['sl_price']) or (pos['side'] == 'SHORT' and price >= pos['sl_price']):
                    portfolio.close_position(symbol, pos['sl_price'], test_df.index[bar_count])
                elif (pos['side'] == 'LONG' and price >= pos['tp_price']) or (pos['side'] == 'SHORT' and price <= pos['tp_price']):
                    portfolio.close_position(symbol, pos['tp_price'], test_df.index[bar_count])

            bar_count += 1
            if bar_count >= len(test_df): break

        metrics = self.calculate_metrics(portfolio)
        print(f"✅ Sharpe: {metrics['sharpe']:.2f} | PnL: ${metrics['pnl']:.2f}")
        return metrics

    def calculate_metrics(self, portfolio):
        trades = portfolio.trades
        if not trades: return {'sharpe': 0, 'pnl': 0, 'win_rate': 0}
        pnls = [t['pnl_pct'] for t in trades]
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if len(pnls) > 1 and np.std(pnls) > 0 else 0
        return {
            'sharpe': float(sharpe),
            'pnl': float(portfolio.current_capital - portfolio.initial_capital),
            'win_rate': float(len([p for p in pnls if p > 0]) / len(pnls) * 100)
        }

    def print_final_report(self):
        print("\n" + "="*60)
        print("🏆 REPORTA FINAL: ROBUSTEZ MULTI-SYMBOL")
        print("="*60)
        
        summary = []
        for symbol, folds in self.all_results.items():
            sharpes = [f['metrics']['sharpe'] for f in folds]
            pnls = [f['metrics']['pnl'] for f in folds]
            avg_sharpe = np.mean(sharpes) if sharpes else 0
            total_pnl = sum(pnls)
            
            status = "✅ MASTER" if avg_sharpe > 1.5 else "⚠️ WEAK"
            print(f"• {symbol:10} | Sharpe: {avg_sharpe:5.2f} | PnL: ${total_pnl:8.2f} | {status}")
            summary.append({'symbol': symbol, 'avg_sharpe': avg_sharpe, 'total_pnl': total_pnl})
            
        with open("logs/multi_symbol_wfv.json", "w") as f:
            json.dump(self.all_results, f, indent=4)
        print(f"\n📂 Resultados completos en logs/multi_symbol_wfv.json")

if __name__ == "__main__":
    # Test top 5 liquidity from Elite Basket
    top_elite = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
    tester = WalkForwardTester(symbols=top_elite, total_days=10)
    tester.run()
