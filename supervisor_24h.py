"""
🛡️ SUPERVISOR-24H: CONTINUOUS MISSION AUDIT & SUPERVISION
===========================================================

PROFESSOR METHOD:
- QUÉ: Script supervisor que corre 24 horas vigilando el bot.
- POR QUÉ: Garantiza que el test de 24h sea fuente de verdad para producción.
- CÓMO: Loop continuo que lee logs, audita señales, y genera reportes.
- CUÁNDO: Ejecutar en terminal separada durante pruebas pre-producción.

USAGE:
    python supervisor_24h.py

MONITORING:
- Ghost Signals: Señales perdidas por umbral
- Latency: Pipeline < 500ms
- Margin: Desviaciones sin posiciones
- Equity Drift: Inconsistencias cada 15min
- Strategy Leaderboard: Win rate y momentum

REPORTS:
- Every 4 hours: Quantitative summary
- On exception: STALE_DATA, DISCONNECTED alerts
"""

import os
import sys
try:
    import ujson as json
except ImportError:
    import json
import time
import csv
import signal
import subprocess
import threading
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force non-test environment
os.environ['TRADER_GEMINI_ENV'] = os.environ.get('TRADER_GEMINI_ENV', 'PRODUCTION')

from utils.sentinel import (
    Sentinel24, 
    HealthStatus, 
    AlertSeverity,
    get_sentinel,
    reset_sentinel
)


# =============================================================================
# CONFIGURATION
# =============================================================================

class SupervisorConfig:
    """Supervisor configuration."""
    
    # Paths
    DATA_DIR = Path("dashboard/data")
    LOGS_DIR = Path("logs")
    TRADES_CSV = DATA_DIR / "futures" / "trades.csv"
    LIVE_STATUS = DATA_DIR / "futures" / "live_status.json"
    HEALTH_LOG = LOGS_DIR / "health_log.json"
    STRATEGY_LOG = LOGS_DIR / "strategy.log"
    
    # Intervals (seconds)
    POLL_INTERVAL = 5  # Check every 5 seconds
    REPORT_INTERVAL = 4 * 60 * 60  # 4 hours
    STALE_THRESHOLD = 60  # 60 seconds without update = stale
    
    # Thresholds
    SIGNAL_THRESHOLD = 0.60
    LATENCY_TARGET_MS = 500.0
    BASE_MARGIN = 21.52
    MARGIN_DEVIATION = 2.0
    EQUITY_DRIFT_PCT = 0.5


# =============================================================================
# FILE WATCHERS
# =============================================================================

class FileWatcher:
    """Watches a file for changes."""
    
    def __init__(self, path: Path):
        self.path = path
        self.last_modified = 0
        self.last_size = 0
    
    def has_changed(self) -> bool:
        """Check if file has changed."""
        if not self.path.exists():
            return False
        
        stat = self.path.stat()
        current_mtime = stat.st_mtime
        current_size = stat.st_size
        
        if current_mtime > self.last_modified or current_size != self.last_size:
            self.last_modified = current_mtime
            self.last_size = current_size
            return True
        
        return False
    
    def get_content(self) -> Optional[str]:
        """Read file content."""
        try:
            if self.path.exists():
                return self.path.read_text(encoding='utf-8')
        except Exception:
            pass
        return None

class LogTailer:
    """Tails a log file and yields new lines."""
    def __init__(self, path: Path):
        self.path = path
        self._last_size = 0
        if path.exists():
            self._last_size = path.stat().st_size
    
    def get_new_lines(self) -> list:
        if not self.path.exists(): return []
        current_size = self.path.stat().st_size
        if current_size < self._last_size: # Rotated
            self._last_size = 0
        
        lines = []
        if current_size > self._last_size:
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    f.seek(self._last_size)
                    lines = f.readlines()
                    self._last_size = f.tell()
            except Exception:
                pass
        return lines


# =============================================================================
# SUPERVISOR ENGINE
# =============================================================================

class Supervisor24H:
    """
    🛡️ 24-Hour Continuous Supervisor.
    
    Runs continuously, monitoring all data streams and generating reports.
    """
    
    def __init__(self, config: SupervisorConfig = None):
        self.config = config or SupervisorConfig()
        self.sentinel = Sentinel24(
            signal_threshold=self.config.SIGNAL_THRESHOLD,
            latency_target_ms=self.config.LATENCY_TARGET_MS,
            base_margin=self.config.BASE_MARGIN
        )
        
        # File watchers
        self.trades_watcher = FileWatcher(self.config.TRADES_CSV)
        self.status_watcher = FileWatcher(self.config.LIVE_STATUS)
        self.health_watcher = FileWatcher(self.config.HEALTH_LOG)
        self.log_tailer = LogTailer(self.config.STRATEGY_LOG)
        
        # State
        self.running = False
        self.start_time: Optional[datetime] = None
        self.last_report_time: Optional[datetime] = None
        self.cycles_completed = 0
        self.last_status_update: Optional[datetime] = None
        
        # Stats
        self.total_trades_seen = 0
        self.total_ghost_signals = 0
        self.latency_breaches = 0
        
        # Pro Monitoring State
        self.last_restart_time = 0
        self.restart_cooldown = 60  # seconds
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    self.config.LOGS_DIR / 'supervisor_24h.log',
                    encoding='utf-8'
                )
            ]
        )
        self.logger = logging.getLogger('supervisor')
    
    def _print_banner(self):
        """Print startup banner."""
        print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║     🛡️  SENTINEL-24 SUPERVISOR                                          ║
║     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                          ║
║                                                                          ║
║     24-Hour Continuous Audit & Supervision System                        ║
║                                                                          ║
║     MONITORING:                                                          ║
║     • Ghost Signals (confidence > 0 but < threshold)                     ║
║     • Pipeline Latency (target < 500ms)                                  ║
║     • Margin Ratio Deviations                                            ║
║     • Equity Drift Detection                                             ║
║     • Strategy Leaderboard & Momentum                                    ║
║                                                                          ║
║     REPORTS:                                                             ║
║     • Every 4 hours: Quantitative summary                                ║
║     • On exception: STALE_DATA, DISCONNECTED alerts                      ║
║                                                                          ║
║     Press Ctrl+C to stop gracefully                                      ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
        """)
    
    def _read_live_status(self) -> Optional[Dict]:
        """Read live_status.json."""
        try:
            if self.config.LIVE_STATUS.exists():
                content = self.config.LIVE_STATUS.read_text(encoding='utf-8')
                return json.loads(content)
        except Exception as e:
            self.logger.error(f"Error reading live_status: {e}")
        return None
    
    def _read_trades(self) -> list:
        """Read trades from CSV."""
        trades = []
        try:
            if self.config.TRADES_CSV.exists():
                with open(self.config.TRADES_CSV, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    trades = list(reader)
        except Exception as e:
            self.logger.error(f"Error reading trades: {e}")
        return trades
    
    def _check_staleness(self, status: Dict) -> bool:
        """Check if data is stale."""
        if not status:
            return True
        
        try:
            last_update_str = status.get('timestamp') or status.get('last_update')
            if last_update_str:
                # Parse timestamp
                if 'T' in last_update_str:
                    last_update = datetime.fromisoformat(last_update_str.replace('Z', '+00:00'))
                else:
                    last_update = datetime.now(timezone.utc)
                
                self.last_status_update = last_update
                age_seconds = (datetime.now(timezone.utc) - last_update).total_seconds()
                
                if age_seconds > self.config.STALE_THRESHOLD:
                    self.sentinel.handle_stale_data(
                        'live_status.json', 
                        age_seconds
                    )
                    return True
        except Exception as e:
            self.logger.debug(f"Staleness check error: {e}")
        
        return False
    
    def _check_process_health(self):
        """Check if bot process is alive, if not, try to restart."""
        # Check if already alive (ProcessMonitor.is_alive() will try to find it if None)
        if self.sentinel.process_monitor.is_alive():
            return

        # Cooldown check
        now = time.time()
        if now - self.last_restart_time < self.restart_cooldown:
            return

        self.logger.error("🚨 BOT PROCESS NOT DETECTED (Terminated or Manual Stop). Attempting auto-recovery...")
        self.sentinel.raise_alert(
            AlertSeverity.CRITICAL, 
            "ProcessMonitor", 
            "Bot process not found in current directory"
        )
        
        # Start the bot with correct arguments for the current session
        try:
            self.last_restart_time = now
            
            # Using current executable (likely .venv) and specifying mode
            cmd = [sys.executable, "main.py", "--mode", "futures"] 
            
            self.logger.info(f"🚀 RESTARTING BOT: {' '.join(cmd)}")
            
            # Redirect stdout/stderr to debug file to catch startup errors
            # Redirect stdout/stderr to debug file to catch startup errors
            # Open without 'with' to keep handle alive (file leak acceptable for debug)
            out = open("logs/startup_debug.log", "a")
            
            # Prepare environment with BOT_MODE for logger.py
            env = os.environ.copy()
            env['BOT_MODE'] = 'futures'
            
            # On Windows, Popen might fail if handle closes too fast.
            subprocess.Popen(cmd, stdout=out, stderr=out, env=env)
            
            self.logger.info("✅ Restart command issued. Monitoring PID...")
        except Exception as e:
            self.logger.error(f"Restart failed: {e}")
            self.last_restart_time = 0 
    
    def _check_resources(self):
        """Check system resources and print deep breakdown."""
        res = self.sentinel.resource_vigilance.check_resources()
        details = res.get('details', {})
        
        # Log deep breakdown periodically (every 10 cycles to avoid spam)
        if self.cycles_completed % 10 == 0:
            print("\n" + "═"*50)
            print("📊 DEEP RESOURCE AUDIT")
            print("═"*50)
            print(f"🖥️  System RAM: {res['ram_pct']}% | CPU: {details.get('system_cpu_pct')}%")
            print(f"📦 PROJECT TOTAL RAM: {res['project_mb']:.2f} MB")
            
            breakdown = details.get('process_breakdown', {})
            for role, data in breakdown.items():
                if data['mb'] > 0:
                    pids = ",".join(map(str, data['pids']))
                    print(f"   • {role:10}: {data['mb']:7.2f} MB (CPU: {data['cpu']:4.1f}%) [PIDs: {pids}]")
            
            print("-" * 30)
            disk = details.get('disk_breakdown', {})
            print(f"💾 DISK USAGE (Project):")
            for cat, size in disk.items():
                print(f"   • {cat:10}: {size:7.2f} MB")
            print(f"   TOTAL DISK: {res['disk_pct']}%")
            print("═"*50 + "\n")
    
    def _process_status(self, status: Dict):
        """Process live status update."""
        if not status:
            return
        
        # Extract margin ratio
        margin_ratio = float(status.get('margin_ratio', 0)) if status.get('margin_ratio') else 0
        positions = status.get('positions', {})
        has_positions = bool(positions) and any(
            float(p.get('quantity', 0)) != 0 for p in positions.values()
        ) if isinstance(positions, dict) else False
        
        # Record margin
        alert = self.sentinel.margin_vigilance.record_margin(margin_ratio, has_positions)
        if alert:
            self._print_alert(alert)
        
        # Check equity drift
        wallet_balance = float(status.get('wallet_balance', 0)) if status.get('wallet_balance') else 0
        unrealized_pnl = float(status.get('unrealized_pnl', 0)) if status.get('unrealized_pnl') else 0
        total_equity = float(status.get('total_equity', 0)) if status.get('total_equity') else 0
        
        expected_equity = wallet_balance + unrealized_pnl
        if expected_equity > 0 and total_equity > 0:
            drift = self.sentinel.equity_detector.record_equity(total_equity, expected_equity)
            if drift:
                self.logger.warning(f"⚠️ Equity drift: {drift['drift_pct']:.2f}%")
    
    def _process_trades(self, trades: list):
        """Process trades for leaderboard."""
        if len(trades) > self.total_trades_seen:
            new_trades = trades[self.total_trades_seen:]
            
            for trade in new_trades:
                # MODO PROFESOR: En trades.csv la columna es 'strategy', no 'strategy_id'
                strategy_id = trade.get('strategy') or trade.get('strategy_id', 'UNKNOWN')
                pnl_str = trade.get('net_pnl') or trade.get('pnl') or '0'
                try:
                    pnl = float(pnl_str)
                except ValueError:
                    pnl = 0.0
                
                is_win = pnl > 0
                self.sentinel.leaderboard.record_trade(strategy_id, is_win, pnl)
                
                self.logger.info(
                    f"📊 Trade recorded: {strategy_id} "
                    f"{'🟢 Win' if is_win else '🔴 Loss'} ${pnl:+.4f}"
                )
            
            self.total_trades_seen = len(trades)
    
    def _print_alert(self, alert):
        """Print formatted alert."""
        severity_icons = {
            'INFO': 'ℹ️',
            'WARNING': '⚠️',
            'CRITICAL': '🚨',
            'EMERGENCY': '🆘'
        }
        icon = severity_icons.get(alert.severity.value, '❓')
        self.logger.warning(f"{icon} [{alert.source}] {alert.message}")
    
    def _print_status(self):
        """Print current status line."""
        health = self.sentinel.get_health_status()
        ghost_count = len(self.sentinel.ghost_auditor.ghost_signals)
        
        runtime = datetime.now(timezone.utc) - self.start_time if self.start_time else timedelta(0)
        hours = runtime.seconds // 3600
        minutes = (runtime.seconds % 3600) // 60
        
        # Get leaderboard top
        lb = self.sentinel.leaderboard.get_leaderboard()
        top_strategy = lb[0]['strategy_id'] if lb else 'N/A'
        total_pnl = sum(s['total_pnl'] for s in lb) if lb else 0
        
        status_line = (
            f"{health.value} "
            f"Runtime: {hours:02d}h{minutes:02d}m | "
            f"Trades: {self.total_trades_seen} | "
            f"Ghost: {ghost_count} | "
            f"PnL: ${total_pnl:+.4f} | "
            f"Top: {top_strategy}"
        )
        print(f"\r{status_line}", end='', flush=True)
    
    def _maybe_generate_report(self):
        """Generate report if 4 hours elapsed."""
        now = datetime.now(timezone.utc)
        
        if self.last_report_time is None:
            self.last_report_time = now
            return
        
        elapsed = (now - self.last_report_time).total_seconds()
        
        if elapsed >= self.config.REPORT_INTERVAL:
            print("\n")  # Clear status line
            report = self.sentinel.generate_4h_report()
            self.last_report_time = now
            
            # Also log ghost signal summary
            ghost_summary = self.sentinel.ghost_auditor.get_summary()
            if ghost_summary['total'] > 0:
                self.logger.info(f"👻 {ghost_summary['message']}")
    
    def run(self):
        """Main supervisor loop."""
        self._print_banner()
        
        self.running = True
        self.start_time = datetime.now(timezone.utc)
        self.last_report_time = self.start_time
        
        self.logger.info("🚀 Supervisor started")
        self.logger.info(f"   Watching: {self.config.LIVE_STATUS}")
        self.logger.info(f"   Trades:   {self.config.TRADES_CSV}")
        self.logger.info(f"   Reports:  Every 4 hours")
        print()
        
        try:
            while self.running:
                self.cycles_completed += 1
                
                # Read and process live status
                if self.status_watcher.has_changed():
                    status = self._read_live_status()
                    if status:
                        is_stale = self._check_staleness(status)
                        if not is_stale:
                            self._process_status(status)
                
                # Read and process trades
                if self.trades_watcher.has_changed():
                    trades = self._read_trades()
                    self._process_trades(trades)
                
                # Real-time ghost signal auditing
                for line in self.log_tailer.get_new_lines():
                    self.sentinel.ghost_auditor.audit_from_log(line)
                    self.sentinel.zombie_auditor.audit_from_log(line)
                
                # Pro Monitoring: Process & Resources
                self._check_process_health()
                self._check_resources()
                
                # Check for report generation
                self._maybe_generate_report()
                
                # Print status
                self._print_status()
                
                # Wait before next cycle (Responsive Poll)
                for _ in range(int(self.config.POLL_INTERVAL)):
                    if not self.running: break
                    time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n")
            self.logger.info("🛑 Supervisor stopped by user")
            self._print_final_summary()
    
    def _print_final_summary(self):
        """Print final summary on exit."""
        runtime = datetime.now(timezone.utc) - self.start_time if self.start_time else timedelta(0)
        
        print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                        FINAL SUPERVISOR SUMMARY                          ║
╠══════════════════════════════════════════════════════════════════════════╣""")
        print(f"║  Runtime:          {runtime}")
        print(f"║  Cycles:           {self.cycles_completed}")
        print(f"║  Trades Seen:      {self.total_trades_seen}")
        print(f"║  Ghost Signals:    {len(self.sentinel.ghost_auditor.ghost_signals)}")
        print(f"║  Latency Breaches: {len(self.sentinel.latency_monitor.breaches)}")
        print(f"║  Margin Alerts:    {len(self.sentinel.margin_vigilance.alerts)}")
        print(f"║  Equity Drifts:    {len(self.sentinel.equity_detector.drifts_detected)}")
        print(f"║  Total Alerts:     {len(self.sentinel.alerts)}")
        print("╚══════════════════════════════════════════════════════════════════════════╝")
        
        # Leaderboard
        lb = self.sentinel.leaderboard.get_leaderboard()
        if lb:
            print("\n📊 STRATEGY LEADERBOARD:")
            for i, s in enumerate(lb[:5], 1):
                print(f"   {i}. {s['strategy_id']}: {s['win_rate']:.1f}% WR, ${s['total_pnl']:.4f} PnL")
        
        # Ghost signals summary
        ghost = self.sentinel.ghost_auditor.get_summary()
        if ghost['total'] > 0:
            print(f"\n👻 {ghost['message']}")
    
    def stop(self):
        """Stop the supervisor."""
        self.running = False


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    supervisor = Supervisor24H()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        supervisor.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    supervisor.run()


if __name__ == "__main__":
    main()
