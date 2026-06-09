"""
DEEP ROOTS AUDIT v3 — Trade Lifecycle & Signal Chain (CORRECTED API)
Tests the actual flow: Signal → RiskManager → Order → Portfolio → Exit
"""
import sys, os, time, traceback, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TRADER_GEMINI_BACKTEST"] = "true"  # Prevent KillSwitch lock file

results = []

def test(name, fn):
    try:
        result = fn()
        if result:
            results.append(('PASS', name, ''))
            print(f"  ✅ {name}")
        else:
            results.append(('FAIL', name, 'returned False'))
            print(f"  ❌ {name}: returned False/None")
    except Exception as e:
        tb_lines = traceback.format_exc().split('\n')
        loc = [l for l in tb_lines if 'File' in l and 'scratch' not in l and '_audit' not in l]
        loc_str = loc[-1].strip() if loc else ''
        results.append(('FAIL', name, f"{type(e).__name__}: {str(e)[:150]}"))
        print(f"  ❌ {name}: {type(e).__name__}: {str(e)[:180]}")
        if loc_str:
            print(f"      → {loc_str}")

print(f"\n{'='*70}")
print(f"  DEEP ROOTS AUDIT v3 — Trade Lifecycle Chain")
print(f"{'='*70}\n")

print("── A: Signal → RiskManager Gate ──")

def t_signal_event_correct_fields():
    from core.events import SignalEvent
    from core.enums import TradeDirection, SignalType
    se = SignalEvent(
        symbol='BTC/USDT',
        signal_type=SignalType.LONG,
        strategy_id='test_scalp',
        horizon='SCALPING',
        datetime=datetime.datetime.now(datetime.timezone.utc),
        current_price=100000.0,
        strength=0.85,
        metadata={'tp_pct': 0.008, 'sl_pct': 0.004}
    )
    assert hasattr(se, 'symbol'), "Missing symbol"
    assert hasattr(se, 'signal_type'), "Missing signal_type"
    assert hasattr(se, 'strength'), "Missing strength"
    assert hasattr(se, 'strategy_id'), "Missing strategy_id"
    assert hasattr(se, 'metadata'), "Missing metadata"
    return True
test("SignalEvent has all required fields", t_signal_event_correct_fields)

def t_risk_generate_order_accepts_signal():
    from core.events import SignalEvent
    from core.enums import TradeDirection, SignalType
    from risk.risk_manager import RiskManager
    from core.portfolio import Portfolio
    
    p = Portfolio(initial_capital=13.0, auto_save=False)
    rm = RiskManager(p)
    
    se = SignalEvent(
        symbol='BTC/USDT',
        signal_type=SignalType.LONG,
        strategy_id='test_scalp',
        horizon='SCALPING',
        datetime=datetime.datetime.now(datetime.timezone.utc),
        current_price=100000.0,
        strength=0.85,
        metadata={
            'horizon': 'SCALPING',
            'tp_pct': 0.008,
            'sl_pct': 0.004,
            'dollar_size': 6.0
        }
    )
    
    order = rm.generate_order(se)
    return True
test("RiskManager.generate_order doesn't crash", t_risk_generate_order_accepts_signal)

print("\n── B: Portfolio Position Management ──")

def t_portfolio_open_position():
    from core.portfolio import Portfolio
    from core.events import FillEvent
    from core.enums import TradeDirection, OrderSide
    
    p = Portfolio(initial_capital=13.0, auto_save=False)
    
    fill = FillEvent(
        symbol='BTC/USDT',
        exchange='BINANCE',
        quantity=0.001,
        direction=OrderSide.BUY,
        fill_cost=100.0,
        commission=0.04,
        order_id='TEST_FILL_001',
        timeindex=datetime.datetime.now(datetime.timezone.utc),
        strategy_id='test_scalp',
        horizon='SCALPING',
        metadata={
            'horizon': 'SCALPING',
            'tp_pct': 0.008,
            'sl_pct': 0.004,
            'position_side': 'LONG'
        }
    )
    
    p.update_fill(fill)
    
    vk = 'BTC/USDT_SCALPING_LONG'
    pos = p.virtual_ledger.get(vk)
    if pos is None:
        vk_legacy = 'BTC/USDT_SCALPING'
        pos = p.virtual_ledger.get(vk_legacy)
    
    assert pos is not None, f"Position not found in virtual_ledger. Keys: {list(p.virtual_ledger.keys())}"
    assert abs(pos['quantity']) > 0, f"Quantity is 0 in ledger"
    assert pos.get('horizon') == 'SCALPING', f"Horizon mismatch: {pos.get('horizon')}"
    return True
test("Portfolio.update_fill creates ledger entry", t_portfolio_open_position)

def t_portfolio_margin_accounting():
    from core.portfolio import Portfolio
    from core.events import FillEvent
    from core.enums import TradeDirection, OrderSide
    
    p = Portfolio(initial_capital=13.0, auto_save=False)
    initial_cash = p.current_cash
    
    fill = FillEvent(
        symbol='BTC/USDT',
        exchange='BINANCE',
        quantity=0.001,
        direction=OrderSide.BUY,
        fill_cost=100.0,
        commission=0.04,
        order_id='TEST_MARGIN_001',
        timeindex=datetime.datetime.now(datetime.timezone.utc),
        strategy_id='test_scalp',
        horizon='SCALPING',
        metadata={
            'horizon': 'SCALPING',
            'tp_pct': 0.008,
            'sl_pct': 0.004,
            'position_side': 'LONG'
        }
    )
    
    p.update_fill(fill)
    
    cash_diff = initial_cash - p.current_cash
    assert cash_diff >= 0, f"Cash increased after buying?! diff={cash_diff}"
    assert p.total_fees_paid >= 0.04, f"Fees not tracked: {p.total_fees_paid}"
    return True
test("Portfolio margin accounting after fill", t_portfolio_margin_accounting)

print("\n── C: Exit Chain ──")

def t_check_stops_no_crash():
    from risk.risk_manager import RiskManager
    from core.portfolio import Portfolio
    
    p = Portfolio(initial_capital=13.0, auto_save=False)
    rm = RiskManager(p)
    
    class MockDH:
        def get_latest_bars(self, symbol, n=1, timeframe='1m'): return None
        symbol_list = ['BTC/USDT']
        buffers_1m = {}
    
    signals = rm.check_stops(p, MockDH(), symbol_filter='BTC/USDT')
    assert signals is not None, "check_stops returned None (should be empty list)"
    assert isinstance(signals, list), f"check_stops returned {type(signals)}"
    return True
test("RiskManager.check_stops (empty portfolio)", t_check_stops_no_crash)

print("\n── D: Prediction Tracker ──")

def t_prediction_tracker_register():
    from core.prediction_tracker import PredictionTracker
    pt = PredictionTracker()
    
    pt.record_signal(
        symbol='BTC/USDT',
        strategy_id='test_ml',
        direction='long',
        entry_price=100000.0,
        predicted_magnitude=0.005,
        confidence=0.85,
        horizon='SCALPING',
        sl_pct=0.004,
        tp_pct=0.008
    )
    
    active = pt._active_by_symbol.get('BTC/USDT', [])
    assert len(active) > 0, f"No active predictions after register: {active}"
    return True
test("PredictionTracker record_signal", t_prediction_tracker_register)

def t_prediction_tracker_update():
    from core.prediction_tracker import PredictionTracker
    pt = PredictionTracker()
    
    pt.record_signal(
        symbol='BTC/USDT',
        strategy_id='test_ml',
        direction='long',
        entry_price=100000.0,
        predicted_magnitude=0.005,
        confidence=0.85,
        horizon='SCALPING',
        sl_pct=0.004,
        tp_pct=0.008
    )
    
    pt.update_forward_returns(
        symbol='BTC/USDT',
        current_price=100500.0
    )
    
    active = pt._active_by_symbol.get('BTC/USDT', [])
    if active:
        pred = active[0]
        assert pred.mfe > 0 or pred.mae > 0, "Forward tracking failed"
    return True
test("PredictionTracker update_forward_returns", t_prediction_tracker_update)

print("\n── E: Feature Engineering ──")

def t_feature_engineering_import():
    try:
        from strategies.components.feature_engineering import FeatureEngineering
        fe = FeatureEngineering()
        assert hasattr(fe, 'prepare_features'), \
            f"FeatureEngineering has no prepare_features method. Available: {[m for m in dir(fe) if not m.startswith('_')]}"
        return True
    except ImportError:
        try:
            from models.feature_engineering import FeatureEngineering
            return True
        except ImportError:
            raise ImportError("FeatureEngineering not found")
test("FeatureEngineering importable", t_feature_engineering_import)

print("\n── F: Dead Code Detection ──")

def t_portfolio_check_exits_exists():
    from core.portfolio import Portfolio
    p = Portfolio(initial_capital=13.0, auto_save=False)
    assert hasattr(p, 'check_exits'), "Portfolio.check_exits MISSING"
    return True
test("Portfolio.check_exits exists", t_portfolio_check_exits_exists)

def t_risk_manager_has_exit_oracle_methods():
    from risk.risk_manager import RiskManager
    from core.portfolio import Portfolio
    p = Portfolio(initial_capital=13.0, auto_save=False)
    rm = RiskManager(p)
    oracle = rm.exit_oracle
    
    assert hasattr(oracle, 'evaluate_open_positions'), "Missing evaluate_open_positions"
    assert hasattr(oracle, 'register_strategy'), "Missing register_strategy"
    return True
test("ExitOracle has all required methods", t_risk_manager_has_exit_oracle_methods)

print(f"\n{'='*70}")
passed = sum(1 for r in results if r[0] == 'PASS')
failed = sum(1 for r in results if r[0] == 'FAIL')
total = len(results)
print(f"  RESULTS: {passed} PASSED, {failed} FAILED out of {total}")
print(f"{'='*70}")

if failed:
    sys.exit(1)
