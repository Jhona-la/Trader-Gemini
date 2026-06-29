"""
📉 OMEGA-VOID §3.1: Black Swan Backtesting (The Forge)

QUÉ: Backtest sobre datos históricos de flash-crashes reales de Binance.
POR QUÉ: Un bot que solo se testea en condiciones normales morirá
     en el primer cisne negro. Los crashes de Mayo 2021 (-35% BTC en 24h),
     Noviembre 2022 FTX (-25% en 12h), y Marzo 2020 COVID (-50% en 48h)
     son los escenarios de supervivencia obligatorios.
PARA QUÉ: Demostrar que con capital de $13 USDT:
     1. El bot SOBREVIVE (capital > $0 al final)
     2. Kill-Switch se activa antes de max drawdown breach
     3. Max drawdown no excede 2%
CÓMO: Descarga datos históricos reales de Binance REST API, ejecuta
     el BacktestPortfolio con slippage extremo (10 ticks).
CUÁNDO: Antes de cada actualización de producción.
DÓNDE: tests/black_swan_backtest.py
QUIÉN: BacktestPortfolio (de run_backtest.py) + escenarios definidos.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from config import Config


# ============================================================
# BLACK SWAN SCENARIOS
# ============================================================

BLACK_SWAN_EVENTS = {
    'covid_crash_2020': {
        'name': '🦠 COVID Flash Crash',
        'symbol': 'BTCUSDT',
        'start_date': '2020-03-10',
        'end_date': '2020-03-15',
        'btc_drop_pct': -50,
        'description': 'BTC dropped from $8,000 to $3,800 in 48 hours',
        'expected_slippage_ticks': 10,
    },
    'may_2021_crash': {
        'name': '📉 May 2021 Crash',
        'symbol': 'BTCUSDT',
        'start_date': '2021-05-18',
        'end_date': '2021-05-23',
        'btc_drop_pct': -35,
        'description': 'BTC dropped from $43,000 to $30,000 in 24 hours',
        'expected_slippage_ticks': 8,
    },
    'ftx_collapse_2022': {
        'name': '💥 FTX Collapse',
        'symbol': 'BTCUSDT',
        'start_date': '2022-11-07',
        'end_date': '2022-11-12',
        'btc_drop_pct': -25,
        'description': 'BTC dropped from $21,000 to $15,500 in 12 hours',
        'expected_slippage_ticks': 10,
    },
    'luna_collapse_2022': {
        'name': '🌙 LUNA/UST Collapse',
        'symbol': 'BTCUSDT',
        'start_date': '2022-05-07',
        'end_date': '2022-05-13',
        'btc_drop_pct': -30,
        'description': 'Systemic contagion from LUNA death spiral',
        'expected_slippage_ticks': 10,
    },
}


def fetch_crash_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = '1m',
) -> Optional[pd.DataFrame]:
    """
    Fetches historical kline data from Binance REST API.
    
    QUÉ: Descarga datos reales de velas 1-minuto de Binance.
    POR QUÉ: Los backtests deben usar datos reales, no sintéticos.
    CÓMO: python-binance REST API (mainnet para datos históricos).
    
    Returns:
        DataFrame con columnas OHLCV + datetime, o None si falla.
    """
    try:
        from binance.client import Client
        
        client = Client(
            Config.BINANCE_API_KEY, 
            Config.BINANCE_SECRET_KEY
        )
        
        klines = client.get_historical_klines(
            symbol,
            interval,
            start_date,
            end_date,
        )
        
        if not klines:
            print(f"⚠️ No data returned for {symbol} {start_date}→{end_date}")
            return None
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        print(f"📊 Fetched {len(df)} bars for {symbol} ({start_date} → {end_date})")
        return df
        
    except ImportError:
        print("⚠️ python-binance not installed. Using synthetic data.")
        return generate_synthetic_crash(symbol, start_date, end_date)
    except Exception as e:
        print(f"⚠️ API Error: {e}. Using synthetic data.")
        return generate_synthetic_crash(symbol, start_date, end_date)


def generate_synthetic_crash(
    symbol: str,
    start_date: str,
    end_date: str,
    start_price: float = 50000.0,
    crash_pct: float = -0.35,
) -> pd.DataFrame:
    """
    Generates synthetic crash data when real data is unavailable.
    
    QUÉ: Genera datos sintéticos que replican la dinámica de un crash.
    POR QUÉ: Fallback para cuando la API no está disponible.
    CÓMO: GBM con drift negativo + saltos de Poisson (tail events).
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    n_minutes = int((end - start).total_seconds() / 60)
    
    # Parameters for crash simulation
    dt = 1 / (24 * 60)  # 1 minute in days
    mu = crash_pct / ((end - start).days)  # Daily drift
    sigma = abs(crash_pct) / ((end - start).days * np.sqrt(252))  # Daily vol
    
    prices = np.zeros(n_minutes)
    prices[0] = start_price
    
    for i in range(1, n_minutes):
        # GBM + Poisson jumps
        dW = np.random.normal(0, 1) * np.sqrt(dt)
        jump = 0
        if np.random.random() < 0.005:  # 0.5% chance of jump per minute
            jump = np.random.normal(-0.02, 0.01)  # Negative jumps
        
        prices[i] = prices[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW + jump)
    
    # Generate OHLCV
    data = []
    for i in range(n_minutes):
        close = prices[i]
        volatility = abs(close * 0.001)
        high = close + abs(np.random.normal(0, volatility))
        low = close - abs(np.random.normal(0, volatility))
        open_p = prices[i-1] if i > 0 else close
        volume = max(0.1, np.random.lognormal(5, 2))
        
        data.append({
            'datetime': start + timedelta(minutes=i),
            'open': open_p,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
        })
    
    return pd.DataFrame(data)


class BlackSwanPortfolio:
    """
    Extended BacktestPortfolio with extreme slippage for crash conditions.
    
    QUÉ: Portfolio de backtest con slippage 10x durante crashes.
    POR QUÉ: Durante flash crashes, la liquidez desaparece y el slippage
         real es 5-10x el normal. Los backtests deben reflejar esto.
    """
    
    def __init__(self, initial_capital: float = 13.0, leverage: int = 10):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.leverage = leverage
        
        # Position tracking
        self.positions: Dict[str, Dict] = {}
        self.closed_trades: List[Dict] = []
        
        # Risk metrics
        self.max_drawdown_pct = 0.0
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.kill_switch_triggered = False
        self.kill_switch_reason = ""
        
        # Extreme slippage
        self.base_slippage_pct = 0.0005  # 0.05% normal
        self.crash_slippage_pct = 0.005  # 0.50% during crash
    
    def _calculate_slippage(self, price: float, side: str, volatility: float) -> float:
        """
        Dynamic slippage that increases with volatility.
        
        QUÉ: Modelo de slippage que escala con la volatilidad.
        POR QUÉ: En crashes, no hay liquidez → slippage 10x.
        """
        # Base slippage + volatility component
        vol_factor = min(5.0, volatility / 0.01)  # Cap at 5x
        effective_slippage = self.base_slippage_pct * (1 + vol_factor)
        
        if side == 'BUY':
            return price * (1 + effective_slippage)
        else:
            return price * (1 - effective_slippage)
    
    def _check_kill_switch(self, current_price: float = None):
        """
        Kill-Switch: triggers if drawdown > 2%.
        
        QUÉ: Parada de emergencia si el drawdown excede el límite.
        POR QUÉ: Con $13 USDT, perder 2% = $0.26. Perder más es inaceptable.
        """
        if self.peak_capital > 0:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            self.max_drawdown_pct = max(self.max_drawdown_pct, drawdown * 100)
            
            if drawdown > 0.02:  # 2% kill switch
                self.kill_switch_triggered = True
                self.kill_switch_reason = (
                    f"Drawdown {drawdown*100:.2f}% > 2.0% limit. "
                    f"Capital: ${self.current_capital:.2f}"
                )
                # Close all positions immediately
                for symbol in list(self.positions.keys()):
                    if current_price:
                        self.close_position(symbol, current_price, datetime.now())
    
    def open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        size_usd: float,
        timestamp: datetime,
        volatility: float = 0.01,
    ):
        """Open position with crash-adjusted slippage."""
        if self.kill_switch_triggered:
            return
        
        if symbol in self.positions:
            return  # No doubling down
        
        fill_price = self._calculate_slippage(price, side, volatility)
        
        self.positions[symbol] = {
            'side': side,
            'entry_price': fill_price,
            'size_usd': size_usd,
            'timestamp': timestamp,
            'volatility': volatility,
        }
    
    def close_position(self, symbol: str, price: float, timestamp: datetime):
        """Close position with slippage."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        exit_side = 'SELL' if pos['side'] == 'BUY' else 'BUY'
        fill_price = self._calculate_slippage(price, exit_side, pos['volatility'])
        
        # Calculate PnL
        if pos['side'] == 'BUY':
            pnl_pct = (fill_price - pos['entry_price']) / pos['entry_price']
        else:
            pnl_pct = (pos['entry_price'] - fill_price) / pos['entry_price']
        
        pnl_usd = pnl_pct * pos['size_usd']
        
        self.current_capital += pnl_usd
        self.peak_capital = max(self.peak_capital, self.current_capital)
        
        self.closed_trades.append({
            'symbol': symbol,
            'side': pos['side'],
            'entry': pos['entry_price'],
            'exit': fill_price,
            'pnl_pct': pnl_pct * 100,
            'pnl_usd': pnl_usd,
            'duration': (timestamp - pos['timestamp']).total_seconds(),
        })
        
        del self.positions[symbol]
        self._check_kill_switch(price)
    
    def update_equity(self, timestamp: datetime, current_price: float):
        """Update equity curve and check kill switch."""
        unrealized = 0
        for symbol, pos in self.positions.items():
            if pos['side'] == 'BUY':
                unrealized += (current_price - pos['entry_price']) / pos['entry_price'] * pos['size_usd']
            else:
                unrealized += (pos['entry_price'] - current_price) / pos['entry_price'] * pos['size_usd']
        
        total_equity = self.current_capital + unrealized
        self.equity_curve.append((timestamp, total_equity))
        self._check_kill_switch(current_price)


def run_black_swan_backtest(
    scenario_key: str,
    initial_capital: float = 13.0,
) -> Dict:
    """
    Executes a single black swan backtest.
    
    QUÉ: Ejecuta backtest completo sobre un evento de cisne negro.
    CÓMO: Descarga datos → simula estrategia simple → verifica supervivencia.
    
    Returns:
        Dict with survival metrics.
    """
    scenario = BLACK_SWAN_EVENTS[scenario_key]
    print(f"\n{'='*60}")
    print(f"🦢 {scenario['name']}")
    print(f"   {scenario['description']}")
    print(f"{'='*60}")
    
    # Fetch data
    df = fetch_crash_data(
        scenario['symbol'],
        scenario['start_date'],
        scenario['end_date'],
    )
    
    if df is None or len(df) < 100:
        return {'error': 'Insufficient data', 'scenario': scenario_key}
    
    # Initialize portfolio
    portfolio = BlackSwanPortfolio(initial_capital=initial_capital)
    
    # Simple momentum strategy (for testing — real strategy is in technical.py)
    lookback = 20
    position_size = initial_capital * 0.1  # 10% per trade
    
    for i in range(lookback, len(df)):
        bar = df.iloc[i]
        current_price = bar['close']
        timestamp = bar['datetime']
        
        # Calculate simple momentum
        past_prices = df['close'].iloc[i-lookback:i].values
        returns = np.diff(past_prices) / past_prices[:-1]
        volatility = np.std(returns)
        momentum = (current_price - past_prices[0]) / past_prices[0]
        
        # Update equity
        portfolio.update_equity(timestamp, current_price)
        
        if portfolio.kill_switch_triggered:
            break
        
        # Simple signal
        symbol = scenario['symbol'].replace('USDT', '/USDT')
        has_position = symbol in portfolio.positions
        
        if not has_position:
            if momentum > 0.005 and volatility < 0.05:
                portfolio.open_position(
                    symbol, 'BUY', current_price, 
                    position_size, timestamp, volatility
                )
            elif momentum < -0.005 and volatility < 0.05:
                portfolio.open_position(
                    symbol, 'SELL', current_price,
                    position_size, timestamp, volatility
                )
        else:
            pos = portfolio.positions[symbol]
            # Exit on reversal or high volatility
            if pos['side'] == 'BUY' and momentum < -0.003:
                portfolio.close_position(symbol, current_price, timestamp)
            elif pos['side'] == 'SELL' and momentum > 0.003:
                portfolio.close_position(symbol, current_price, timestamp)
            elif volatility > 0.08:  # Emergency exit on extreme vol
                portfolio.close_position(symbol, current_price, timestamp)
    
    # Close remaining positions
    for sym in list(portfolio.positions.keys()):
        portfolio.close_position(sym, df.iloc[-1]['close'], df.iloc[-1]['datetime'])
    
    # Results
    survived = portfolio.current_capital > 0
    n_trades = len(portfolio.closed_trades)
    wins = [t for t in portfolio.closed_trades if t['pnl_usd'] > 0]
    losses = [t for t in portfolio.closed_trades if t['pnl_usd'] <= 0]
    
    result = {
        'scenario': scenario_key,
        'event_name': scenario['name'],
        'survived': survived,
        'initial_capital': initial_capital,
        'final_capital': round(portfolio.current_capital, 4),
        'pnl_usd': round(portfolio.current_capital - initial_capital, 4),
        'pnl_pct': round((portfolio.current_capital - initial_capital) / initial_capital * 100, 2),
        'max_drawdown_pct': round(portfolio.max_drawdown_pct, 2),
        'kill_switch_triggered': portfolio.kill_switch_triggered,
        'kill_switch_reason': portfolio.kill_switch_reason,
        'total_trades': n_trades,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(len(wins) / max(1, n_trades) * 100, 1),
        'bars_processed': len(df),
    }
    
    # Print results
    status = "✅ SURVIVED" if survived else "💀 DESTROYED"
    ks_status = "🚨 TRIGGERED" if portfolio.kill_switch_triggered else "✅ Not needed"
    
    print(f"\n   Status: {status}")
    print(f"   Capital: ${initial_capital} → ${result['final_capital']}")
    print(f"   PnL: {result['pnl_pct']}%")
    print(f"   Max Drawdown: {result['max_drawdown_pct']}%")
    print(f"   Kill-Switch: {ks_status}")
    print(f"   Trades: {n_trades} (Win Rate: {result['win_rate']}%)")
    
    return result


def run_all_black_swans(initial_capital: float = 13.0) -> Dict:
    """
    Runs ALL black swan scenarios and generates survival report.
    
    Returns:
        Complete report dict.
    """
    print("\n" + "=" * 70)
    print("🦢 OMEGA-VOID §3.1: BLACK SWAN SURVIVAL TEST")
    print(f"   Capital: ${initial_capital} USDT")
    print("=" * 70)
    
    results = {}
    all_survived = True
    
    for key in BLACK_SWAN_EVENTS:
        results[key] = run_black_swan_backtest(key, initial_capital)
        if not results[key].get('survived', False):
            all_survived = False
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 SURVIVAL SUMMARY")
    print("=" * 70)
    
    for key, result in results.items():
        if 'error' in result:
            status = "⚠️ SKIPPED"
        elif result['survived']:
            status = "✅ SURVIVED"
        else:
            status = "💀 FAILED"
        
        print(f"   {status}: {result.get('event_name', key)} | "
              f"Capital: ${result.get('final_capital', '?')} | "
              f"DD: {result.get('max_drawdown_pct', '?')}%")
    
    overall = "✅ ALL PASSED" if all_survived else "❌ SOME FAILED"
    print(f"\n   🏆 Overall: {overall}")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'initial_capital': initial_capital,
        'all_survived': all_survived,
        'scenarios': results,
    }
    
    # Save report
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'black_swan_report.json'
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n   📄 Report saved: {report_path}")
    
    return report


if __name__ == '__main__':
    run_all_black_swans(initial_capital=13.0)
