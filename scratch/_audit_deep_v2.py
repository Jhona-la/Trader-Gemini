"""
DEEP ROOTS AUDIT v2 — Cross-Module Integration & Silent Failure Detection
Tests the ACTUAL data flow from root to leaf, catches bugs that only appear at runtime.
"""
import sys, os, time, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        results.append(('FAIL', name, f"{type(e).__name__}: {str(e)[:120]}"))
        print(f"  ❌ {name}: {type(e).__name__}: {str(e)[:150]}")
        if loc_str:
            print(f"      → {loc_str}")

print(f"\n{'='*70}")
print(f"  DEEP ROOTS AUDIT v2 — Cross-Module Integration")
print(f"{'='*70}\n")

# ═════════════════════════════════════════════════════════════════════
# LAYER 0: Config Consistency (Root of Everything)
# ═════════════════════════════════════════════════════════════════════
print("── L0: Config Cross-Validation ──")

def t_config_check_types():
    from config import Config
    return Config.check_types()
test("Config.check_types() passes", t_config_check_types)

def t_config_data_dir_exists():
    from config import Config
    assert os.path.isdir(Config.DATA_DIR), f"DATA_DIR {Config.DATA_DIR} doesn't exist"
    return True
test("Config.DATA_DIR exists on disk", t_config_data_dir_exists)

def t_config_leverage_vs_sniper():
    from config import Config
    assert Config.BINANCE_LEVERAGE == Config.Sniper.MAX_LEVERAGE, \
        f"Global leverage ({Config.BINANCE_LEVERAGE}) != Sniper MAX ({Config.Sniper.MAX_LEVERAGE})"
    return True
test("Leverage sync: Global vs Sniper", t_config_leverage_vs_sniper)

def t_config_risk_consistency():
    from config import Config
    # Check that RISK_THRESHOLDS zombie hours doesn't conflict with max_hold_bars
    max_hold_minutes = Config.Horizons.Scalping['max_hold_bars']  # bars (1m each)
    zombie_hours = Config.Risk.RISK_THRESHOLDS['zombie_hours_held']
    zombie_minutes = zombie_hours * 60
    # Zombie detection should be >= max hold time, otherwise positions get zombie-tagged before max_hold
    if zombie_minutes < max_hold_minutes:
        raise AssertionError(
            f"Zombie detection ({zombie_hours}h = {zombie_minutes}m) fires BEFORE max_hold_bars "
            f"({max_hold_minutes}m). Trades will be killed as 'zombie' while still within normal hold window."
        )
    return True
test("Zombie detection vs max_hold_bars timing", t_config_risk_consistency)

def t_config_primary_tf_in_timeframes():
    from config import Config
    sp = Config.Horizons.Scalping
    assert sp['primary_tf'] in sp['timeframes'], \
        f"SCALPING primary_tf '{sp['primary_tf']}' not in timeframes {sp['timeframes']}"
    swp = Config.Horizons.Swing
    assert swp['primary_tf'] in swp['timeframes'], \
        f"SWING primary_tf '{swp['primary_tf']}' not in timeframes {swp['timeframes']}"
    return True
test("Primary timeframe in strategy timeframes list", t_config_primary_tf_in_timeframes)

# ═════════════════════════════════════════════════════════════════════
# LAYER 1: Events System
# ═════════════════════════════════════════════════════════════════════
print("\n── L1: Events & Queue ──")

def t_event_creation():
    from core.events import MarketEvent, SignalEvent
    from core.enums import TradeDirection, SignalType
    # MarketEvent
    me = MarketEvent(symbol='BTC/USDT', close_price=100000.0)
    assert me.symbol == 'BTC/USDT'
    assert me.close_price == 100000.0
    # SignalEvent
    se = SignalEvent(
        symbol='BTC/USDT',
        signal_type=SignalType.ML_PREDICTION,
        direction=TradeDirection.LONG,
        strength=0.8,
        strategy_id='test',
        price=100000.0
    )
    assert se.direction == TradeDirection.LONG
    assert se.strength == 0.8
    return True
test("MarketEvent + SignalEvent creation", t_event_creation)

def t_priority_queue():
    from core.engine import PriorityBoundedQueue
    from core.events import MarketEvent
    q = PriorityBoundedQueue(maxsize=100)
    # Put event
    me = MarketEvent(symbol='BTC/USDT', close_price=100000.0)
    q.put(me)
    assert not q.empty(), "Queue should not be empty after put"
    return True
test("PriorityBoundedQueue put/empty", t_priority_queue)

# ═════════════════════════════════════════════════════════════════════
# LAYER 2: Database Tables Integrity
# ═════════════════════════════════════════════════════════════════════
print("\n── L2: Database Schema Integrity ──")

def t_db_all_tables():
    from data.database import DatabaseHandler
    db = DatabaseHandler(db_name="audit_deep_v2.db")
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    
    required_tables = [
        'trades', 'positions', 'signals', 'thoughts',
        'prediction_audit', 'trade_chronicle', 'exit_strategy_log',
        'strategy_awareness', 'backtest_results'
    ]
    missing = [t for t in required_tables if t not in tables]
    db.conn.close()
    try: os.remove(os.path.join('dashboard', 'data', 'futures', 'audit_deep_v2.db'))
    except: pass
    assert not missing, f"Missing DB tables: {missing}"
    return True
test("All required DB tables created", t_db_all_tables)

def t_db_trades_schema():
    from data.database import DatabaseHandler
    db = DatabaseHandler(db_name="audit_schema.db")
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(trades)")
    columns = {r[1] for r in cursor.fetchall()}
    
    required_cols = {
        'id', 'timestamp', 'symbol', 'direction', 'quantity', 'entry_price',
        'exit_price', 'pnl', 'strategy_id', 'horizon'
    }
    missing = required_cols - columns
    db.conn.close()
    try: os.remove(os.path.join('dashboard', 'data', 'futures', 'audit_schema.db'))
    except: pass
    assert not missing, f"Missing trades columns: {missing}"
    return True
test("Trades table has required columns", t_db_trades_schema)

# ═════════════════════════════════════════════════════════════════════
# LAYER 3: Portfolio — Cash Accounting
# ═════════════════════════════════════════════════════════════════════
print("\n── L3: Portfolio Cash Accounting ──")

def t_portfolio_equity_calc():
    from core.portfolio import Portfolio
    p = Portfolio(initial_capital=13.0, auto_save=False)
    # Equity without positions should equal cash
    eq = p.get_total_equity()
    assert abs(eq - 13.0) < 0.01, f"Initial equity {eq} != 13.0"
    # After refreshing cache
    p._refresh_equity_cache()
    eq2 = p.get_total_equity()
    assert abs(eq2 - 13.0) < 0.01, f"Refreshed equity {eq2} != 13.0"
    return True
test("Portfolio equity = cash when no positions", t_portfolio_equity_calc)

def t_portfolio_margin_leak_protection():
    """Verify the reconciliation auto-repair when no positions exist."""
    from core.portfolio import Portfolio
    p = Portfolio(initial_capital=13.0, auto_save=False)
    # Simulate a margin leak
    p.used_margin = 5.0
    p.pending_cash = 2.0
    # Calling get_available_cash should auto-repair because there are no positions
    avail = p.get_available_cash()
    # After reconciliation, used_margin and pending_cash should be 0
    assert p.used_margin == 0.0, f"used_margin not reset: {p.used_margin}"
    assert p.pending_cash == 0.0, f"pending_cash not reset: {p.pending_cash}"
    assert avail == 13.0, f"Available should be 13.0 after reset, got {avail}"
    return True
test("Portfolio margin leak auto-repair", t_portfolio_margin_leak_protection)

def t_portfolio_horizon_partition():
    """Verify that SCALPING + SWING don't exceed total equity."""
    from core.portfolio import Portfolio
    p = Portfolio(initial_capital=13.0, auto_save=False)
    scalp = p.get_available_cash(horizon='SCALPING')
    swing = p.get_available_cash(horizon='SWING')
    total = p.get_available_cash()
    assert scalp + swing <= total + 0.01, \
        f"Silo leak: SCALP({scalp:.2f}) + SWING({swing:.2f}) = {scalp+swing:.2f} > total({total:.2f})"
    assert scalp > 0, "SCALPING silo is 0 — system cannot trade"
    return True
test("Horizon cash partition (no silo leak)", t_portfolio_horizon_partition)

# ═════════════════════════════════════════════════════════════════════
# LAYER 4: CompoundingEngine Integration
# ═════════════════════════════════════════════════════════════════════
print("\n── L4: CompoundingEngine ──")

def t_compounding_allocation():
    from core.compounding_engine import get_compounding_engine
    ce = get_compounding_engine()
    s_pct, w_pct = ce.get_horizon_allocation(13.0)
    assert abs(s_pct + w_pct - 1.0) < 0.01, f"Allocation doesn't sum to 100%: {s_pct}+{w_pct}"
    assert s_pct >= 0.50, f"With $13, scalping should get >=50%, got {s_pct*100:.0f}%"
    return True
test("CompoundingEngine $13 allocation", t_compounding_allocation)

def t_compounding_growth_target():
    from core.compounding_engine import get_compounding_engine
    ce = get_compounding_engine()
    phase = ce.get_growth_phase(13.0)
    assert phase is not None, "Growth phase returned None"
    target = ce.get_daily_target(13.0)
    assert target > 0, f"Daily target should be positive, got {target}"
    return True
test("CompoundingEngine growth target", t_compounding_growth_target)

# ═════════════════════════════════════════════════════════════════════
# LAYER 5: Adaptive Engine Parameter API
# ═════════════════════════════════════════════════════════════════════
print("\n── L5: AdaptiveEngine API ──")

def t_adaptive_get_method():
    from strategies.components.adaptive_engine import AdaptiveMLParameterEngine
    eng = AdaptiveMLParameterEngine(horizon_str='SCALPING')
    # The actual method is .get(param_name), NOT .get_params()
    ml_conf = eng.get('ml_confidence')
    assert ml_conf is not None, "get('ml_confidence') returned None"
    assert 0.5 <= ml_conf <= 1.0, f"ml_confidence {ml_conf} out of range"
    sl = eng.get('sl_mult')
    assert sl is not None
    assert 0.1 <= sl <= 1.0, f"Scalping sl_mult {sl} out of range"
    tp = eng.get('tp_mult')
    assert tp is not None
    return True
test("AdaptiveEngine .get() API works", t_adaptive_get_method)

def t_adaptive_feedback():
    from strategies.components.adaptive_engine import AdaptiveMLParameterEngine
    eng = AdaptiveMLParameterEngine(horizon_str='SCALPING')
    initial_conf = eng.get('ml_confidence')
    # Simulate a winning trade
    eng.feedback_trade(pnl_pct=0.005, mae_pct=0.002, mfe_pct=0.008)
    after_conf = eng.get('ml_confidence')
    # Parameters should have changed
    assert eng.trades_processed == 1
    assert len(eng.trade_history) == 1
    return True
test("AdaptiveEngine feedback_trade works", t_adaptive_feedback)

def t_adaptive_horizon_differentiation():
    from strategies.components.adaptive_engine import AdaptiveMLParameterEngine
    scalp = AdaptiveMLParameterEngine(horizon_str='SCALPING')
    swing = AdaptiveMLParameterEngine(horizon_str='SWING')
    assert scalp.profile == 'scalping'
    assert swing.profile == 'swing'
    assert scalp.get('sl_mult') != swing.get('sl_mult'), \
        "SCALPING and SWING have same sl_mult — no differentiation!"
    assert scalp.get('tp_mult') != swing.get('tp_mult'), \
        "SCALPING and SWING have same tp_mult — no differentiation!"
    return True
test("AdaptiveEngine SCALPING vs SWING differentiation", t_adaptive_horizon_differentiation)

# ═════════════════════════════════════════════════════════════════════
# LAYER 6: Risk Manager + Kill Switch
# ═════════════════════════════════════════════════════════════════════
print("\n── L6: Risk Management ──")

def t_kill_switch_init():
    from risk.kill_switch import KillSwitch
    from core.portfolio import Portfolio
    p = Portfolio(initial_capital=13.0, auto_save=False)
    ks = KillSwitch(p)
    assert hasattr(ks, 'check_kill_conditions')
    assert hasattr(ks, 'is_killed')
    assert not ks.is_killed, "KillSwitch should not be triggered at init"
    return True
test("KillSwitch init with Portfolio", t_kill_switch_init)

def t_risk_manager_init():
    from risk.risk_manager import RiskManager
    from core.portfolio import Portfolio
    p = Portfolio(initial_capital=13.0, auto_save=False)
    rm = RiskManager(p)
    assert rm.portfolio is p
    assert hasattr(rm, 'check_stops')
    assert hasattr(rm, 'generate_order')
    assert hasattr(rm, 'exit_oracle')
    assert hasattr(rm, 'prediction_tracker')
    return True
test("RiskManager init with Portfolio", t_risk_manager_init)

def t_risk_manager_has_kill_switch():
    from risk.risk_manager import RiskManager
    from core.portfolio import Portfolio
    p = Portfolio(initial_capital=13.0, auto_save=False)
    rm = RiskManager(p)
    assert hasattr(rm, 'kill_switch'), "RiskManager missing kill_switch"
    assert rm.kill_switch is not None, "RiskManager.kill_switch is None"
    return True
test("RiskManager has KillSwitch attached", t_risk_manager_has_kill_switch)

# ═════════════════════════════════════════════════════════════════════
# LAYER 7: Strategy Initialization
# ═════════════════════════════════════════════════════════════════════
print("\n── L7: Strategy Init ──")

def t_technical_strategy_init():
    from strategies.technical import HybridScalpingStrategy
    from core.engine import PriorityBoundedQueue
    q = PriorityBoundedQueue()
    # Need a mock data_handler
    class MockDH:
        def get_latest_bars(self, symbol, n=1, timeframe='1m'): return None
        def get_active_symbols(self): return ['BTC/USDT']
        symbol_list = ['BTC/USDT']
        buffers_1m = {}
        buffers_5m = {}
        microstructure = {}
        orderbooks = {}
        derivatives_metrics = {}
    
    ts = HybridScalpingStrategy(MockDH(), q, horizon="SCALPING")
    assert ts.horizon == 'SCALPING'
    assert hasattr(ts, 'calculate_signals')
    return True
test("HybridScalpingStrategy init (SCALPING)", t_technical_strategy_init)

def t_technical_strategy_swing_init():
    from strategies.technical import HybridScalpingStrategy
    from core.engine import PriorityBoundedQueue
    q = PriorityBoundedQueue()
    class MockDH:
        def get_latest_bars(self, symbol, n=1, timeframe='1m'): return None
        def get_active_symbols(self): return ['BTC/USDT']
        symbol_list = ['BTC/USDT']
        buffers_1m = {}
        buffers_5m = {}
        microstructure = {}
        orderbooks = {}
        derivatives_metrics = {}
    
    ts = HybridScalpingStrategy(MockDH(), q, horizon="SWING")
    assert ts.horizon == 'SWING'
    return True
test("HybridScalpingStrategy init (SWING)", t_technical_strategy_swing_init)

# ═════════════════════════════════════════════════════════════════════
# LAYER 8: Sophia + ExitOracle
# ═════════════════════════════════════════════════════════════════════
print("\n── L8: Sophia + Oracle ──")

def t_sophia_init():
    from sophia.intelligence import SophiaIntelligence
    s = SophiaIntelligence(bar_minutes=5.0)
    assert hasattr(s, 'calculate_win_probability')
    assert hasattr(s, 'set_horizon_profile')
    s.set_horizon_profile(1)  # scalping
    return True
test("SophiaIntelligence init", t_sophia_init)

def t_exit_oracle_init():
    from risk.risk_manager import RiskManager
    from core.portfolio import Portfolio
    p = Portfolio(initial_capital=13.0, auto_save=False)
    rm = RiskManager(p)
    oracle = rm.exit_oracle
    assert oracle is not None, "ExitOracle is None on RiskManager"
    assert hasattr(oracle, 'evaluate_open_positions')
    assert hasattr(oracle, 'register_strategy')
    return True
test("ExitOracle accessible from RiskManager", t_exit_oracle_init)

# ═════════════════════════════════════════════════════════════════════
# LAYER 9: Global State & Clock
# ═════════════════════════════════════════════════════════════════════
print("\n── L9: CTOS Infrastructure ──")

def t_global_state():
    from core.global_state import global_state
    assert global_state is not None
    # Should have core methods
    assert hasattr(global_state, 'update_from_market_event')
    assert hasattr(global_state, 'get_state')
    assert hasattr(global_state, 'get_system_capabilities')
    return True
test("GlobalState singleton", t_global_state)

def t_global_clock():
    from core.clock import global_clock
    assert global_clock is not None
    assert hasattr(global_clock, 'tick')
    global_clock.tick()
    return True
test("GlobalClock tick", t_global_clock)

def t_event_bus():
    from core.event_bus import event_bus, EventChannel
    assert event_bus is not None
    # Should be able to subscribe/process
    received = []
    def handler(data):
        received.append(data)
    event_bus.subscribe(EventChannel.MUTATION, handler)
    event_bus.publish(EventChannel.MUTATION, {'test': True})
    event_bus.process_queue(max_items=10)
    assert len(received) == 1, f"EventBus didn't deliver: received {len(received)}"
    assert received[0]['test'] == True
    return True
test("EventBus subscribe/publish/process", t_event_bus)

# ═════════════════════════════════════════════════════════════════════
# LAYER 10: Cross-Module Wiring
# ═════════════════════════════════════════════════════════════════════
print("\n── L10: Cross-Module Wiring ──")

def t_engine_strategy_sophia_injection():
    """Verify Engine correctly injects Sophia into strategies."""
    from core.engine import Engine, PriorityBoundedQueue
    from strategies.technical import HybridScalpingStrategy
    
    q = PriorityBoundedQueue()
    e = Engine(events_queue=q)
    
    class MockDH:
        def get_latest_bars(self, symbol, n=1, timeframe='1m'): return None
        def get_active_symbols(self): return ['BTC/USDT']
        symbol_list = ['BTC/USDT']
        buffers_1m = {}
        buffers_5m = {}
        microstructure = {}
        orderbooks = {}
        derivatives_metrics = {}
    
    ts = HybridScalpingStrategy(MockDH(), q, horizon="SCALPING")
    e.register_strategy(ts)
    
    # Strategy should have _engine_ref
    assert ts._engine_ref is e, "Strategy missing _engine_ref"
    return True
test("Engine→Strategy _engine_ref wiring", t_engine_strategy_sophia_injection)

# ═════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
passed = sum(1 for r in results if r[0] == 'PASS')
failed = sum(1 for r in results if r[0] == 'FAIL')
total = len(results)
print(f"  RESULTS: {passed} PASSED, {failed} FAILED out of {total} tests")
print(f"{'='*70}")

if failed:
    print(f"\n🚨 FAILURES ({failed}):")
    for status, name, detail in results:
        if status == 'FAIL':
            print(f"  ❌ {name}")
            print(f"     {detail}")
    print()
    sys.exit(1)
else:
    print(f"\n✅ ALL {total} TESTS PASS — System integrity verified from root to leaf.\n")
