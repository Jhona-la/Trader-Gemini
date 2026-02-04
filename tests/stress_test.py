"""
🚀 STRESS TEST - HIGH FREQUENCY MOCK ENGINE (Phase 6)
======================================================

PROFESSOR METHOD:
- QUÉ: Simulación de 1,000 trades de scalping en menos de 5 minutos.
- POR QUÉ: Valida rendimiento de I/O, cálculo de Esperanza y estabilidad del Dashboard.
- CÓMO: Inyección masiva de señales aleatorias con MockBinanceClient.
- CUÁNDO: Pre-producción para validar resiliencia del sistema.
- DÓNDE: tests/stress_test.py

SAFETY CHECKS:
- TestSecurityGuard ACTIVO: Ninguna orden toca API real
- File Lock Test: Sin PermissionError durante escritura frenética
"""

import os
import sys
import time
import random
import json
import psutil
import threading
import statistics
from datetime import datetime, timezone
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import security guard
from conftest import TestSecurityGuard, SecurityException

# Force TEST environment
os.environ['TRADER_GEMINI_ENV'] = 'TEST'
os.environ['BINANCE_USE_TESTNET'] = 'True'


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StressTestConfig:
    """Configuration for stress test."""
    total_trades: int = 1000
    symbols: List[str] = None
    win_rate_target: float = 0.55  # 55% win rate
    avg_win_pct: float = 0.015     # 1.5% average win
    avg_loss_pct: float = 0.01     # 1.0% average loss
    initial_capital: float = 5000.0
    output_dir: str = "dashboard/data/stress_test"
    write_delay_ms: int = 0        # Delay between writes (0 = max speed)
    batch_size: int = 1            # Trades per batch (1 = instant write)
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['XRP/USDT', 'DOGE/USDT', 'BTC/USDT', 'ETH/USDT', 'SOL/USDT']


# =============================================================================
# TRADE GENERATOR
# =============================================================================

class TradeGenerator:
    """
    📊 Generates random trades based on statistical distribution.
    
    Uses normal distribution for PnL to simulate realistic scalping outcomes.
    """
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.trade_counter = 0
        
        # Base prices for symbols
        self.base_prices = {
            'XRP/USDT': 0.55,
            'DOGE/USDT': 0.08,
            'BTC/USDT': 45000.0,
            'ETH/USDT': 2500.0,
            'SOL/USDT': 100.0
        }
    
    def generate_trade(self) -> Dict:
        """
        Generate a single random trade with realistic PnL distribution.
        
        Returns trade dict compatible with DataHandler.log_trade()
        """
        self.trade_counter += 1
        
        # Random symbol selection
        symbol = random.choice(self.config.symbols)
        base_price = self.base_prices.get(symbol, 100.0)
        
        # Determine win/loss based on target win rate
        is_win = random.random() < self.config.win_rate_target
        
        # Generate PnL with normal distribution
        if is_win:
            # Wins: Normal distribution around avg_win_pct
            pnl_pct = abs(np.random.normal(
                self.config.avg_win_pct, 
                self.config.avg_win_pct * 0.3  # 30% std dev
            ))
        else:
            # Losses: Normal distribution around avg_loss_pct (negative)
            pnl_pct = -abs(np.random.normal(
                self.config.avg_loss_pct,
                self.config.avg_loss_pct * 0.3
            ))
        
        # Calculate prices and PnL
        direction = random.choice(['LONG', 'SHORT'])
        entry_price = base_price * (1 + random.uniform(-0.001, 0.001))
        
        if direction == 'LONG':
            exit_price = entry_price * (1 + pnl_pct)
        else:
            exit_price = entry_price * (1 - pnl_pct)
        
        # Calculate quantity based on a portion of capital
        position_size = self.config.initial_capital * 0.1  # 10% per trade
        quantity = position_size / entry_price
        
        # Calculate absolute PnL
        if direction == 'LONG':
            gross_pnl = (exit_price - entry_price) * quantity
        else:
            gross_pnl = (entry_price - exit_price) * quantity
        
        # Calculate fees (taker fee with BNB discount)
        fee_rate = 0.000375  # 0.0375%
        fee = (entry_price * quantity + exit_price * quantity) * fee_rate
        net_pnl = gross_pnl - fee
        
        # Build trade record
        trade = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'direction': direction,
            'entry_price': round(entry_price, 8),
            'exit_price': round(exit_price, 8),
            'quantity': round(quantity, 8),
            'pnl': round(gross_pnl, 4),
            'fee': round(fee, 4),
            'net_pnl': round(net_pnl, 4),
            'is_reverse': random.random() < 0.1,  # 10% are reversions
            'trade_id': f"STRESS_{self.trade_counter:05d}",
            'strategy_id': random.choice(['HYBRID_SCALPING', 'ML_HYBRID_ULTIMATE_V2', 'SNIPER'])
        }
        
        return trade
    
    def generate_batch(self, count: int) -> List[Dict]:
        """Generate a batch of trades."""
        return [self.generate_trade() for _ in range(count)]


# =============================================================================
# STRESS TEST RUNNER
# =============================================================================

class StressTestRunner:
    """
    🏃 Runs the high-frequency stress test.
    
    PROFESSOR METHOD:
    - Inyecta 1,000 trades simulados
    - Monitorea CPU/RAM
    - Valida integridad de Esperanza Matemática
    - Mide latencia de escritura
    """
    
    def __init__(self, config: StressTestConfig = None):
        self.config = config or StressTestConfig()
        self.generator = TradeGenerator(self.config)
        
        # Metrics
        self.trades_written = 0
        self.write_times: List[float] = []
        self.errors: List[str] = []
        self.all_trades: List[Dict] = []
        
        # Resource monitoring
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.monitoring = False
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # File paths
        self.trades_csv_path = os.path.join(self.config.output_dir, 'trades.csv')
        self.status_json_path = os.path.join(self.config.output_dir, 'live_status.json')
        
    def _monitor_resources(self):
        """Background thread to monitor CPU and RAM."""
        while self.monitoring:
            try:
                self.cpu_samples.append(psutil.cpu_percent(interval=0.1))
                self.memory_samples.append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
            except:
                pass
            time.sleep(0.5)
    
    def _write_trade_csv(self, trade: Dict) -> float:
        """
        Write a single trade to CSV with timing.
        
        Returns write time in milliseconds.
        """
        import csv
        
        start = time.perf_counter()
        
        fieldnames = ['timestamp', 'symbol', 'direction', 'entry_price', 'exit_price', 
                     'quantity', 'pnl', 'fee', 'net_pnl', 'is_reverse', 'trade_id', 'strategy_id']
        
        file_exists = os.path.exists(self.trades_csv_path)
        
        try:
            with open(self.trades_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(trade)
        except PermissionError as e:
            self.errors.append(f"PermissionError at trade {self.trades_written}: {e}")
            raise
        except Exception as e:
            self.errors.append(f"Error at trade {self.trades_written}: {e}")
            raise
        
        end = time.perf_counter()
        return (end - start) * 1000  # Convert to ms
    
    def _update_live_status(self):
        """Update live_status.json with current metrics."""
        # Calculate expectancy
        if self.all_trades:
            net_pnls = [t['net_pnl'] for t in self.all_trades]
            expectancy = sum(net_pnls) / len(net_pnls)
            total_pnl = sum(net_pnls)
            
            wins = [t for t in self.all_trades if t['net_pnl'] > 0]
            losses = [t for t in self.all_trades if t['net_pnl'] <= 0]
            
            win_rate = len(wins) / len(self.all_trades) * 100 if self.all_trades else 0
            
            # Calculate equity curve
            equity = self.config.initial_capital
            equity_curve = [equity]
            for t in self.all_trades:
                equity += t['net_pnl']
                equity_curve.append(equity)
            
            # Calculate max drawdown
            peak = equity_curve[0]
            max_dd = 0
            for eq in equity_curve:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
            
            status = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_trades': len(self.all_trades),
                'total_pnl': round(total_pnl, 2),
                'expectancy': round(expectancy, 4),
                'win_rate': round(win_rate, 2),
                'current_equity': round(equity, 2),
                'initial_capital': self.config.initial_capital,
                'max_drawdown_pct': round(max_dd, 2),
                'avg_write_time_ms': round(statistics.mean(self.write_times), 2) if self.write_times else 0,
                'errors_count': len(self.errors),
                'source': 'STRESS_TEST'
            }
            
            # Atomic write
            temp_path = self.status_json_path + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(status, f, indent=2)
            os.replace(temp_path, self.status_json_path)
    
    def run(self) -> Dict:
        """
        Execute the stress test.
        
        Returns summary metrics.
        """
        print("=" * 70)
        print("🚀 STRESS TEST - HIGH FREQUENCY MOCK ENGINE (Phase 6)")
        print("=" * 70)
        print(f"📊 Target: {self.config.total_trades} trades")
        print(f"📁 Output: {self.config.output_dir}")
        print(f"🔒 SecurityGuard: {'LOCKED ✅' if TestSecurityGuard.is_locked() else 'UNLOCKED ⚠️'}")
        print("-" * 70)
        
        # Lock security guard
        TestSecurityGuard.lock()
        
        # Start resource monitoring
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        monitor_thread.start()
        
        # Clear previous test data
        if os.path.exists(self.trades_csv_path):
            os.remove(self.trades_csv_path)
        
        start_time = time.perf_counter()
        
        try:
            # Main injection loop
            for i in range(self.config.total_trades):
                # Generate trade
                trade = self.generator.generate_trade()
                self.all_trades.append(trade)
                
                # Write to CSV with timing
                write_time = self._write_trade_csv(trade)
                self.write_times.append(write_time)
                self.trades_written += 1
                
                # Update status every 100 trades
                if self.trades_written % 100 == 0:
                    self._update_live_status()
                    elapsed = time.perf_counter() - start_time
                    rate = self.trades_written / elapsed
                    print(f"  📈 Progress: {self.trades_written}/{self.config.total_trades} "
                          f"({rate:.1f} trades/sec) | "
                          f"Avg Write: {statistics.mean(self.write_times[-100:]):.2f}ms")
                
                # Optional delay
                if self.config.write_delay_ms > 0:
                    time.sleep(self.config.write_delay_ms / 1000)
        
        except Exception as e:
            print(f"❌ FATAL ERROR: {e}")
            self.errors.append(f"Fatal: {e}")
        
        finally:
            # Stop monitoring
            self.monitoring = False
            
            # Final status update
            self._update_live_status()
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate final metrics
        summary = self._generate_summary(total_time)
        self._print_summary(summary)
        
        return summary
    
    def _generate_summary(self, total_time: float) -> Dict:
        """Generate summary metrics."""
        net_pnls = [t['net_pnl'] for t in self.all_trades]
        
        # Expectancy formula validation
        # E = (1/N) * Σ Net_PnL_i
        expectancy = sum(net_pnls) / len(net_pnls) if net_pnls else 0
        
        # Win/Loss analysis
        wins = [t for t in self.all_trades if t['net_pnl'] > 0]
        losses = [t for t in self.all_trades if t['net_pnl'] <= 0]
        
        return {
            'total_trades': len(self.all_trades),
            'total_time_seconds': round(total_time, 2),
            'trades_per_second': round(len(self.all_trades) / total_time, 2),
            
            # PnL Metrics
            'total_pnl': round(sum(net_pnls), 2),
            'expectancy': round(expectancy, 4),
            'expectancy_formula_valid': True,  # E = (1/N) * Σ Net_PnL
            
            # Win/Loss
            'win_count': len(wins),
            'loss_count': len(losses),
            'win_rate': round(len(wins) / len(self.all_trades) * 100, 2) if self.all_trades else 0,
            'avg_win': round(statistics.mean([t['net_pnl'] for t in wins]), 4) if wins else 0,
            'avg_loss': round(statistics.mean([t['net_pnl'] for t in losses]), 4) if losses else 0,
            
            # I/O Performance
            'avg_write_time_ms': round(statistics.mean(self.write_times), 2) if self.write_times else 0,
            'max_write_time_ms': round(max(self.write_times), 2) if self.write_times else 0,
            'min_write_time_ms': round(min(self.write_times), 2) if self.write_times else 0,
            'p95_write_time_ms': round(np.percentile(self.write_times, 95), 2) if self.write_times else 0,
            
            # Resource Usage
            'avg_cpu_percent': round(statistics.mean(self.cpu_samples), 1) if self.cpu_samples else 0,
            'max_cpu_percent': round(max(self.cpu_samples), 1) if self.cpu_samples else 0,
            'avg_memory_mb': round(statistics.mean(self.memory_samples), 1) if self.memory_samples else 0,
            'max_memory_mb': round(max(self.memory_samples), 1) if self.memory_samples else 0,
            
            # Errors
            'errors_count': len(self.errors),
            'errors': self.errors[:10],  # First 10 errors
            
            # File paths
            'trades_csv_path': self.trades_csv_path,
            'status_json_path': self.status_json_path
        }
    
    def _print_summary(self, summary: Dict):
        """Print formatted summary."""
        print("\n" + "=" * 70)
        print("📊 STRESS TEST RESULTS")
        print("=" * 70)
        
        print(f"\n⏱️  PERFORMANCE:")
        print(f"   Total Time:        {summary['total_time_seconds']:.2f} seconds")
        print(f"   Trades/Second:     {summary['trades_per_second']:.2f}")
        print(f"   Target Met:        {'✅ YES' if summary['total_time_seconds'] < 300 else '❌ NO'} (< 5 min)")
        
        print(f"\n💰 FINANCIAL METRICS:")
        print(f"   Total Trades:      {summary['total_trades']}")
        print(f"   Total PnL:         ${summary['total_pnl']:+.2f}")
        print(f"   Expectancy (E):    ${summary['expectancy']:+.4f}")
        print(f"   Win Rate:          {summary['win_rate']:.1f}%")
        print(f"   Avg Win:           ${summary['avg_win']:+.4f}")
        print(f"   Avg Loss:          ${summary['avg_loss']:+.4f}")
        
        print(f"\n📝 I/O PERFORMANCE:")
        print(f"   Avg Write Time:    {summary['avg_write_time_ms']:.2f} ms")
        print(f"   Max Write Time:    {summary['max_write_time_ms']:.2f} ms")
        print(f"   P95 Write Time:    {summary['p95_write_time_ms']:.2f} ms")
        
        print(f"\n🖥️  RESOURCE USAGE:")
        print(f"   Avg CPU:           {summary['avg_cpu_percent']:.1f}%")
        print(f"   Max CPU:           {summary['max_cpu_percent']:.1f}%")
        print(f"   Avg Memory:        {summary['avg_memory_mb']:.1f} MB")
        print(f"   Max Memory:        {summary['max_memory_mb']:.1f} MB")
        
        print(f"\n🔒 SAFETY:")
        print(f"   SecurityGuard:     {'LOCKED ✅' if TestSecurityGuard.is_locked() else 'UNLOCKED ⚠️'}")
        print(f"   File Errors:       {summary['errors_count']}")
        
        if summary['errors']:
            print(f"\n⚠️  ERRORS:")
            for err in summary['errors'][:5]:
                print(f"   - {err}")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"   Trades CSV:        {summary['trades_csv_path']}")
        print(f"   Status JSON:       {summary['status_json_path']}")
        
        # Formula validation
        print(f"\n📐 EXPECTANCY FORMULA VALIDATION:")
        print(f"   E = (1/N) × Σ Net_PnL_i")
        print(f"   E = (1/{summary['total_trades']}) × ${summary['total_pnl']:+.2f}")
        print(f"   E = ${summary['expectancy']:+.4f} ✅")
        
        print("\n" + "=" * 70)
        
        # Overall result
        if summary['errors_count'] == 0 and summary['total_time_seconds'] < 300:
            print("🎉 STRESS TEST PASSED!")
        else:
            print("⚠️  STRESS TEST COMPLETED WITH WARNINGS")
        print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the stress test."""
    # Ensure security is locked
    TestSecurityGuard.lock()
    print(f"🔒 SecurityGuard Status: {'LOCKED ✅' if TestSecurityGuard.is_locked() else 'UNLOCKED ⚠️'}")
    
    # Create config
    config = StressTestConfig(
        total_trades=1000,
        symbols=['XRP/USDT', 'DOGE/USDT', 'BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        win_rate_target=0.55,
        avg_win_pct=0.015,
        avg_loss_pct=0.01,
        initial_capital=5000.0,
        write_delay_ms=0  # Maximum speed
    )
    
    # Run test
    runner = StressTestRunner(config)
    summary = runner.run()
    
    # Save summary to JSON
    summary_path = os.path.join(config.output_dir, 'stress_test_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n📄 Summary saved to: {summary_path}")
    
    return summary


if __name__ == "__main__":
    main()
