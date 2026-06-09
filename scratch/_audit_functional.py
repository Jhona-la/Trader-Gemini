"""
DEEP FORENSIC AUDIT — Functional Integrity Tests
Tests that core subsystems actually WORK, not just import.
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
        tb = traceback.format_exc().split('\n')
        last_relevant = [l for l in tb if 'File' in l and 'scratch' not in l]
        loc = last_relevant[-1].strip() if last_relevant else ''
        results.append(('FAIL', name, f"{type(e).__name__}: {str(e)[:100]}"))
        print(f"  ❌ {name}: {type(e).__name__}: {str(e)[:120]}")
        if loc:
            print(f"      → {loc}")

print(f"\n{'='*70}")
print(f"  DEEP FUNCTIONAL AUDIT")
print(f"{'='*70}\n")

# ═════ TEST 1: Config Integrity ═════
print("── Config Layer ──")
def t_config_types():
    from config import Config
    assert isinstance(Config.INITIAL_CAPITAL, (int, float))
    assert Config.INITIAL_CAPITAL > 0
    assert isinstance(Config.BINANCE_LEVERAGE, int)
    assert Config.MAX_RISK_PER_TRADE == Config.Risk.MAX_RISK_PER_TRADE
    assert Config.STOP_LOSS_PCT == Config.Risk.STOP_LOSS_PCT
    return True
test("Config type integrity", t_config_types)

def t_config_sync():
    from config import Config
    # Verify SCALPING_PARAMS are structured correctly
    assert 'tp_pct' in Config.Horizons.Scalping, "Missing tp_pct in SCALPING"
    assert 'sl_pct' in Config.Horizons.Scalping, "Missing sl_pct in SCALPING"
    # Verify Mutations synced
    assert Config.Horizons.Mutations['max_tp_cap'] == Config.Horizons.Scalping['tp_pct'], \
        f"Mutations max_tp_cap ({Config.Horizons.Mutations['max_tp_cap']}) != SCALPING tp_pct ({Config.Horizons.Scalping['tp_pct']})"
    return True
test("Config parameter sync (TP/SL/Mutations)", t_config_sync)

def t_config_fee_sync():
    from config import Config
    # Sniper fees must match global
    assert Config.Sniper.TAKER_FEE == Config.BINANCE_TAKER_FEE_BNB, \
        f"Sniper TAKER_FEE ({Config.Sniper.TAKER_FEE}) != global ({Config.BINANCE_TAKER_FEE_BNB})"
    assert Config.Sniper.MAKER_FEE == Config.BINANCE_MAKER_FEE_BNB, \
        f"Sniper MAKER_FEE ({Config.Sniper.MAKER_FEE}) != global ({Config.BINANCE_MAKER_FEE_BNB})"
    return True
test("Fee sync (Sniper vs Global)", t_config_fee_sync)

# ═════ TEST 2: Database ═════
print("\n── Database Layer ──")
def t_db_init():
    from data.database import DatabaseHandler
    db = DatabaseHandler(db_name="test_audit.db")
    conn = db.get_connection()
    assert conn is not None, "DB connection failed"
    # Verify critical tables exist
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    required = ['trades', 'positions', 'signals', 'thoughts', 'prediction_audit',
                'trade_chronicle', 'exit_strategy_log', 'strategy_awareness']
    missing = [t for t in required if t not in tables]
    assert not missing, f"Missing tables: {missing}"
    # Cleanup
    db.conn.close()
    try: os.remove(os.path.join('dashboard', 'data', 'futures', 'test_audit.db'))
    except: pass
    return True
test("Database table creation", t_db_init)

# ═════ TEST 3: Portfolio ═════
print("\n── Portfolio Layer ──")
def t_portfolio_init():
    from core.portfolio import Portfolio
    p = Portfolio(initial_capital=13.0, auto_save=False)
    assert p.current_cash == 13.0
    assert p._equity_cache == 13.0
    assert p.realized_pnl == 0.0
    assert p.used_margin == 0.0
    assert p.pending_cash == 0.0
    assert len(p.positions) == 0
    assert len(p.virtual_ledger) == 0
    return True
test("Portfolio initialization", t_portfolio_init)

def t_portfolio_cash_horizon():
    from core.portfolio import Portfolio
    p = Portfolio(initial_capital=13.0, auto_save=False)
    scalp_cash = p.get_available_cash(horizon='SCALPING')
    swing_cash = p.get_available_cash(horizon='SWING')
    total_cash = p.get_available_cash()
    assert scalp_cash > 0, f"Scalping cash = {scalp_cash}"
    assert swing_cash >= 0, f"Swing cash = {swing_cash}"
    assert total_cash == 13.0, f"Total cash = {total_cash}"
    # Scalp + Swing should not exceed total
    assert scalp_cash + swing_cash <= total_cash + 0.01, \
        f"Horizon leak: scalp({scalp_cash}) + swing({swing_cash}) > total({total_cash})"
    return True
test("Portfolio horizon cash partitioning", t_portfolio_cash_horizon)

def t_portfolio_reserve_release():
    from core.portfolio import Portfolio
    p = Portfolio(initial_capital=13.0, auto_save=False)
    ok = p.reserve_cash(5.0, horizon='SCALPING', order_id='TEST_001')
    assert ok, "Reserve failed"
    assert p.pending_cash == 5.0
    assert 'TEST_001' in p._pending_reservations
    p.release_order_margin(order_id='TEST_001')
    assert p.pending_cash == 0.0
    assert 'TEST_001' not in p._pending_reservations
    return True
test("Portfolio reserve/release atomic", t_portfolio_reserve_release)

# ═════ TEST 4: State Manager ═════
print("\n── State Manager Layer ──")
def t_state_checkpoint_recover():
    from core.state_manager import AtomicStateManager
    AtomicStateManager._last_checkpoint = 0  # Force checkpoint
    test_state = {'cash': 13.0, 'test': True, 'positions': {}}
    AtomicStateManager.checkpoint(test_state, key='test_audit')
    recovered = AtomicStateManager.recover(key='test_audit')
    assert recovered is not None, "Recovery returned None"
    assert recovered['cash'] == 13.0
    assert recovered['test'] == True
    return True
test("StateManager checkpoint + recover", t_state_checkpoint_recover)

# ═════ TEST 5: Adaptive Engine ═════
print("\n── Adaptive Engine Layer ──")
def t_adaptive_scalping():
    from strategies.components.adaptive_engine import AdaptiveMLParameterEngine
    eng = AdaptiveMLParameterEngine(horizon_str='SCALPING')
    p = eng.params
    assert 'ml_confidence' in p
    assert 'sl_mult' in p
    assert 'tp_mult' in p
    # Verify refined ranges
    assert p['ml_confidence'] >= 0.70, f"ml_confidence {p['ml_confidence']} < 0.70 (not refined)"
    assert p['sl_mult'] <= 0.40, f"sl_mult {p['sl_mult']} > 0.40 (not refined)"
    return True
test("AdaptiveEngine SCALPING params", t_adaptive_scalping)

def t_adaptive_swing():
    from strategies.components.adaptive_engine import AdaptiveMLParameterEngine
    eng = AdaptiveMLParameterEngine(horizon_str='SWING')
    p = eng.params
    assert p['sl_mult'] >= 1.0, f"Swing sl_mult {p['sl_mult']} < 1.0 (wrong horizon?)"
    assert p['tp_mult'] >= 2.0, f"Swing tp_mult {p['tp_mult']} < 2.0"
    return True
test("AdaptiveEngine SWING params", t_adaptive_swing)

# ═════ TEST 6: Micro-Optimized Strategy (FIXED: was broken import) ═════
print("\n── Broken Module Detection ──")
def t_micro_optimized_import():
    from strategies.micro_optimized import MicroOptimizedStrategy
    from core.micro_awareness import MicroAccountAwareness
    # Verify instantiation works (the old bug was super().__init__() with no args)
    micro = MicroAccountAwareness()
    # Need a mock data_provider and events_queue
    class _MockDP:
        symbol_list = ['BTC/USDT']
    import queue
    strat = MicroOptimizedStrategy(_MockDP(), queue.Queue(), micro)
    assert strat.horizon == "SCALPING", f"Expected SCALPING, got {strat.horizon}"
    assert strat.micro is not None
    return True
test("micro_optimized.py import + init (fixed)", t_micro_optimized_import)

# ═════ TEST 7: Kill Switch ═════
print("\n── Risk Layer ──")
def t_kill_switch():
    from risk.kill_switch import KillSwitch
    from core.portfolio import Portfolio
    p = Portfolio(initial_capital=13.0, auto_save=False)
    ks = KillSwitch(portfolio=p)
    assert hasattr(ks, 'check_triggers'), "Missing check_triggers"
    assert hasattr(ks, 'active'), "Missing active attribute"
    return True
test("KillSwitch initialization", t_kill_switch)

# ═════ SUMMARY ═════
print(f"\n{'='*70}")
passed = sum(1 for r in results if r[0] == 'PASS')
failed = sum(1 for r in results if r[0] == 'FAIL')
print(f"  RESULTS: {passed} PASSED, {failed} FAILED out of {len(results)} tests")
print(f"{'='*70}")

if failed:
    print(f"\n🚨 FAILURES:")
    for status, name, detail in results:
        if status == 'FAIL':
            print(f"  ❌ {name}")
            print(f"     {detail}")
    sys.exit(1)
