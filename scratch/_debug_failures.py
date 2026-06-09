import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test 1: KillSwitch
print("=== KillSwitch ===")
try:
    from risk.kill_switch import KillSwitch
    from core.portfolio import Portfolio
    p = Portfolio(initial_capital=13.0, auto_save=False)
    ks = KillSwitch(p)
    print(f"  active: {ks.active}")
    print(f"  has check_status: {hasattr(ks, 'check_status')}")
    print(f"  check_status(): {ks.check_status()}")
    print("  ✅ KillSwitch OK")
except Exception as e:
    traceback.print_exc()

# Test 2: RiskManager
print("\n=== RiskManager ===")
try:
    from risk.risk_manager import RiskManager
    from core.portfolio import Portfolio
    p2 = Portfolio(initial_capital=13.0, auto_save=False)
    rm = RiskManager(p2)
    print(f"  has check_stops: {hasattr(rm, 'check_stops')}")
    print(f"  has generate_order: {hasattr(rm, 'generate_order')}")
    print(f"  has exit_oracle: {hasattr(rm, 'exit_oracle')}")
    print(f"  has prediction_tracker: {hasattr(rm, 'prediction_tracker')}")
    print(f"  kill_switch.active: {rm.kill_switch.active}")
    print("  ✅ RiskManager OK")
except Exception as e:
    traceback.print_exc()

# Test 3: SophiaIntelligence
print("\n=== SophiaIntelligence ===")
try:
    from sophia.intelligence import SophiaIntelligence
    s = SophiaIntelligence(bar_minutes=5.0)
    print(f"  has calculate_win_probability: {hasattr(s, 'calculate_win_probability')}")
    s.set_horizon_profile(1)
    print("  ✅ SophiaIntelligence OK")
except Exception as e:
    traceback.print_exc()

# Test 4: DB Schema
print("\n=== DB Schema ===")
try:
    from data.database import DatabaseHandler
    db = DatabaseHandler(db_name="test_schema_debug.db")
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(trades)")
    cols = [r[1] for r in cursor.fetchall()]
    print(f"  trades columns: {cols}")
    db.conn.close()
    try: os.remove(os.path.join('dashboard', 'data', 'futures', 'test_schema_debug.db'))
    except: pass
except Exception as e:
    traceback.print_exc()
