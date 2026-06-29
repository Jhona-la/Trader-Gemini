"""
Unit Tests - Omega Observability Stack (Phase 99)
==================================================
MetricsExporter (per-symbol), WandBTracker (offline), ShadowDarwin (Optuna)
"""
import pytest
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════
#  TEST 1: MetricsExporter Per-Symbol
# ═══════════════════════════════════════════════════════

class MockPortfolio:
    """Mock portfolio with active positions."""
    def get_total_equity(self):
        return 16.50
    
    positions = {
        'BTC/USDT': {
            'quantity': 0.001,
            'avg_price': 100000.0,
            'current_price': 100500.0,
            'tp_price': 101000.0,
            'sl_price': 99500.0,
        },
        'ETH/USDT': {
            'quantity': -0.05,
            'avg_price': 3000.0,
            'current_price': 2950.0,
            'tp_price': 2900.0,
            'sl_price': 3100.0,
        },
        'SOL/USDT': {
            'quantity': 0,
            'avg_price': 0,
        },
    }
    realized_pnl = 5.25
    used_margin = 8.0
    peak_equity = 17.0


class MockEngine:
    strategies = ['tech', 'ml']
    
    class events:
        @staticmethod
        def qsize():
            return 3


class MockKillSwitch:
    active = False
    api_errors = 2
    daily_losses = 1


class MockRiskManager:
    def __init__(self):
        self.kill_switch = MockKillSwitch()


class TestMetricsExporterPerSymbol:
    """Tests for per-symbol Prometheus metrics."""
    
    def test_update_sets_equity(self):
        from utils.metrics_exporter import MetricsExporter
        m = MetricsExporter()
        m.update(portfolio=MockPortfolio(), engine=MockEngine(), queue_size=5)
        assert m.g_equity._value._value == 16.50
        print("✅ Equity gauge set")
    
    def test_update_per_symbol_pnl(self):
        from utils.metrics_exporter import MetricsExporter
        m = MetricsExporter()
        m.update(portfolio=MockPortfolio())
        
        # BTC LONG: (100500-100000)/100000 = 0.5%
        btc_pnl = m.g_pnl_pct.labels(symbol="BTC_USDT")._value._value
        assert abs(btc_pnl - 0.5) < 0.01
        print(f"✅ BTC PnL% = {btc_pnl}")
    
    def test_update_per_symbol_gap(self):
        from utils.metrics_exporter import MetricsExporter
        m = MetricsExporter()
        m.update(portfolio=MockPortfolio())
        
        # BTC LONG: (101000-100500)/100500 ≈ 0.4975%
        btc_gap = m.g_gap_tp.labels(symbol="BTC_USDT")._value._value
        assert abs(btc_gap - 0.4975) < 0.01
        print(f"✅ BTC Gap TP% = {btc_gap}")
    
    def test_no_zero_quantity(self):
        """SOL (qty=0) should NOT appear in per-symbol metrics."""
        from utils.metrics_exporter import MetricsExporter
        m = MetricsExporter()
        m.update(portfolio=MockPortfolio())
        
        # SOL should not have a label set
        # We check by not finding SOL_USDT in the samples
        samples = list(m.g_pnl_pct.collect())
        sol_found = any(
            s.labels.get('symbol') == 'SOL_USDT' 
            for metric in samples 
            for s in metric.samples
        )
        assert not sol_found
        print("✅ SOL (idle) excluded from per-symbol metrics")
    
    def test_update_health(self):
        from utils.metrics_exporter import MetricsExporter
        m = MetricsExporter()
        rm = MockRiskManager()
        m.update_health(risk_manager=rm)
        
        assert m.g_kill_switch._value._value == 0  # Not active
        assert m.g_api_error_rate._value._value == 2
        assert m.g_daily_losses._value._value == 1
        print("✅ Health metrics set correctly")
    
    def test_kill_switch_active(self):
        from utils.metrics_exporter import MetricsExporter
        m = MetricsExporter()
        rm = MockRiskManager()
        rm.kill_switch.active = True
        m.update_health(risk_manager=rm)
        
        assert m.g_kill_switch._value._value == 1
        print("✅ Kill Switch active = 1")
    
    def test_record_tick_latency(self):
        from utils.metrics_exporter import MetricsExporter
        m = MetricsExporter()
        m.record_tick_latency("BTC/USDT", 150.5)
        
        assert m._tick_latencies["BTC/USDT"] == 150.5
        print("✅ Tick latency recorded")
    
    def test_drawdown_calculation(self):
        from utils.metrics_exporter import MetricsExporter
        m = MetricsExporter()
        m.update(portfolio=MockPortfolio())
        
        # Peak=17, Equity=16.5 → DD = (17-16.5)/17*100 ≈ 2.94%
        dd = m.g_drawdown._value._value
        assert abs(dd - 2.94) < 0.1
        print(f"✅ Drawdown = {dd:.2f}%")


# ═══════════════════════════════════════════════════════
#  TEST 2: WandBTracker (Offline Mode)
# ═══════════════════════════════════════════════════════

class TestWandBTracker:
    """Tests for WandBTracker in offline/local mode."""
    
    def test_init_without_api_key(self):
        from utils.wandb_tracker import WandBTracker
        tracker = WandBTracker()
        assert not tracker.is_active
        print("✅ WandB tracker starts inactive")
    
    def test_log_generation_local(self):
        from utils.wandb_tracker import WandBTracker
        tracker = WandBTracker()
        tracker.log_generation(
            gen_id=1, fitness=0.85, diversity=0.3,
            params={"rsi_period": 14}, symbol="BTC/USDT"
        )
        
        log = tracker.get_local_log()
        assert len(log) == 1
        assert log[0]["BTC/USDT/fitness"] == 0.85
        print(f"✅ Local log: {log[0]}")
    
    def test_log_rl_epoch_local(self):
        from utils.wandb_tracker import WandBTracker
        tracker = WandBTracker()
        tracker.log_rl_epoch(epoch=5, reward=0.7, loss=0.02, symbol="ETH/USDT")
        
        log = tracker.get_local_log()
        assert len(log) == 1
        assert log[0]["rl/ETH/USDT/reward"] == 0.7
        print("✅ RL epoch logged locally")
    
    def test_compare_epochs(self):
        from utils.wandb_tracker import WandBTracker
        tracker = WandBTracker()
        
        current = {"fitness": 0.9, "sharpe": 2.5, "drawdown": 0.8}
        previous = {"fitness": 0.7, "sharpe": 2.0, "drawdown": 1.2}
        
        deltas = tracker.compare_epochs(current, previous, symbol="fleet")
        assert abs(deltas["delta/fleet/fitness"] - 0.2) < 1e-9
        assert abs(deltas["delta/fleet/sharpe"] - 0.5) < 1e-9
        assert abs(deltas["delta/fleet/drawdown"] - (-0.4)) < 1e-9
        print(f"✅ Epoch comparison: {deltas}")
    
    def test_get_summary(self):
        from utils.wandb_tracker import WandBTracker
        tracker = WandBTracker()
        tracker.log_generation(1, 0.5, symbol="A")
        tracker.log_generation(2, 0.6, symbol="A")
        
        summary = tracker.get_local_summary()
        assert summary["total_entries"] == 2
        print(f"✅ Summary: {summary}")
    
    def test_log_efficacy(self):
        from utils.wandb_tracker import WandBTracker
        tracker = WandBTracker()
        tracker.log_efficacy("BTC/USDT", efficacy_ratio=0.8, rl_outcome=0.9, pnl_pct=0.45)
        
        log = tracker.get_local_log()
        assert log[0]["efficacy/BTC/USDT/ratio"] == 0.8
        print("✅ Efficacy logged locally")


# ═══════════════════════════════════════════════════════
#  TEST 3: ShadowDarwin Optuna Integration
# ═══════════════════════════════════════════════════════

class TestShadowDarwinOptuna:
    """Tests for Optuna integration (compile-level, not simulation)."""
    
    def test_optuna_imports(self):
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
        assert True
        print(f"✅ Optuna {optuna.__version__} imported OK")
    
    def test_study_creation(self):
        import optuna
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        assert study is not None
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE
        print("✅ Optuna study created")
    
    def test_genotype_from_params(self):
        """Verify Genotype can be constructed from Optuna params."""
        from core.genotype import Genotype
        
        params = {
            "bollinger_period": 25,
            "rsi_period": 10,
            "tp_pct": 0.02,
            "weight_trend": 0.5,
            "weight_momentum": 0.3,
            "weight_volatility": 0.2,
        }
        
        genes = dict(Genotype(symbol="_default_").genes)
        for k, v in params.items():
            if k in genes:
                genes[k] = v
        
        g = Genotype(symbol="TEST/USDT", genes=genes)
        assert g.genes["bollinger_period"] == 25
        assert g.genes["rsi_period"] == 10
        print(f"✅ Genotype from Optuna params: {g.genes['bollinger_period']}, {g.genes['rsi_period']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
