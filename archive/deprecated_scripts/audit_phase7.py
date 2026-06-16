"""
PHASE 7: ENGINE FLOW GRAPH AUDIT
Traces the COMPLETE execution path from market data to order placement.
Identifies every disconnection in the signal lifecycle.
"""
import re

def trace_engine_flow():
    """Trace the complete signal lifecycle through the engine."""
    print("=" * 70)
    print("PHASE 7: COMPLETE ENGINE FLOW GRAPH AUDIT")
    print("=" * 70)
    
    eng = open('core/engine.py', 'r', encoding='utf-8').read()
    eng_lines = eng.split('\n')
    
    # ═══════════════════════════════════════════════════
    # 1. Engine Initialization — what gets connected?
    # ═══════════════════════════════════════════════════
    print("\n🔗 1. ENGINE INITIALIZATION CHAIN:")
    init_checks = {
        'risk_manager': r'self\.risk_manager\s*=',
        'portfolio': r'self\.portfolio\s*=',
        'data_provider': r'self\.data_provider\s*=',
        'strategies': r'self\.strategies\s*[=\[]',
        'meta_coordinator/arbitrator': r'self\.(meta_coordinator|meta_arbitrator)\s*=',
        'exit_oracle': r'self\.exit_oracle\s*=',
        'sophia': r'self\.sophia',
        'graph_intelligence': r'self\.graph',
        'kill_switch': r'self\.kill_switch',
        'prediction_tracker': r'self\.prediction_tracker',
        'notifier': r'self\.notifier',
        'sentinel': r'self\.sentinel',
        'session_manager': r'self\.session',
        'cooldown_manager': r'self\.cooldown',
    }
    
    for name, pattern in init_checks.items():
        found = False
        for i, line in enumerate(eng_lines):
            if re.search(pattern, line) and not line.strip().startswith('#'):
                found = True
                break
        status = "✅ CONNECTED" if found else "❌ DISCONNECTED"
        print(f"  {status} — {name}")
    
    # ═══════════════════════════════════════════════════
    # 2. Signal Processing — does _process_signal_event wire to executor?
    # ═══════════════════════════════════════════════════
    print("\n🔗 2. SIGNAL → EXECUTION CHAIN:")
    
    # Check: Does engine submit to meta_coordinator?
    has_submit = 'submit_intent' in eng or 'meta_arbitrator' in eng or 'meta_coordinator' in eng
    print(f"  {'✅' if has_submit else '❌'} Engine submits to MetaCoordinator")
    
    # Check: Does engine read approved intents?
    has_approved = 'get_approved_intent' in eng or 'approved_queue' in eng
    print(f"  {'✅' if has_approved else '❌'} Engine reads approved intents")
    
    # Check: Does engine call executor.place_order?
    has_place_order = 'place_order' in eng or 'executor' in eng
    print(f"  {'✅' if has_place_order else '❌'} Engine calls executor.place_order")
    
    # Check: Does executor feed back to portfolio?
    exec_content = open('execution/binance_executor.py', 'r', encoding='utf-8').read()
    has_fill_callback = 'fill_callback' in exec_content or 'on_fill' in exec_content or 'portfolio' in exec_content
    print(f"  {'✅' if has_fill_callback else '❌'} Executor feeds back to Portfolio")
    
    # ═══════════════════════════════════════════════════
    # 3. EXIT FLOW — single authority check
    # ═══════════════════════════════════════════════════
    print("\n🔗 3. EXIT SIGNAL LIFECYCLE:")
    
    # Count ExitOracle evaluations
    oracle_eval_engine = eng.count('evaluate_open_positions')
    rm = open('risk/risk_manager.py', 'r', encoding='utf-8').read()
    oracle_eval_rm = rm.count('exit_oracle')
    
    # Count check_stops calls
    check_stops_calls = eng.count('check_stops')
    print(f"  ExitOracle evaluations in engine.py: {oracle_eval_engine}")
    print(f"  ExitOracle references in risk_manager: {oracle_eval_rm}")
    print(f"  check_stops() calls from engine: {check_stops_calls}")
    
    # Check if engine generates its own EXIT signals (BAD!)
    exit_signal_gen = 0
    for i, line in enumerate(eng_lines):
        if 'signal_type="EXIT"' in line or "signal_type='EXIT'" in line:
            if not line.strip().startswith('#'):
                exit_signal_gen += 1
                print(f"  ⚠️ engine.py:{i+1} generates EXIT signal with STRING type!")
    
    # Check for SignalType.EXIT in engine
    for i, line in enumerate(eng_lines):
        if 'SignalType.EXIT' in line and 'events.put' in eng_lines[max(0,i-3):i+1]:
            exit_signal_gen += 1
            print(f"  ⚠️ engine.py:{i+1} generates EXIT signal")
    
    if exit_signal_gen == 0:
        print(f"  ✅ Engine does NOT generate EXIT signals (correct — risk_manager only)")
    
    # ═══════════════════════════════════════════════════
    # 4. POSITION LIFECYCLE — open/close/zombie cleanup
    # ═══════════════════════════════════════════════════
    print("\n🔗 4. POSITION LIFECYCLE:")
    
    pf = open('core/portfolio.py', 'r', encoding='utf-8').read()
    
    has_update_fill = 'def update_fill' in pf
    has_record_closed = '_record_closed_trade' in pf
    has_exit_pending = 'exit_pending_time' in pf
    has_get_stats = 'def get_statistics' in pf
    has_virtual_ledger = 'virtual_ledger' in pf
    has_get_horizon_pos = 'def get_horizon_position' in pf
    
    print(f"  {'✅' if has_update_fill else '❌'} Portfolio.update_fill()")
    print(f"  {'✅' if has_record_closed else '❌'} Portfolio._record_closed_trade()")
    print(f"  {'✅' if has_exit_pending else '❌'} exit_pending_time lock exists")
    print(f"  {'✅' if has_get_stats else '❌'} Portfolio.get_statistics() for Kelly")
    print(f"  {'✅' if has_virtual_ledger else '❌'} Virtual ledger (horizon tracking)")
    print(f"  {'✅' if has_get_horizon_pos else '❌'} get_horizon_position()")
    
    # Check exit_pending_time is cleared on close
    exit_clear_on_close = pf.count("pos['exit_pending_time'] = 0")
    print(f"  {'✅' if exit_clear_on_close >= 2 else '❌'} exit_pending_time cleared on close ({exit_clear_on_close} paths)")
    
    # ═══════════════════════════════════════════════════
    # 5. STRATEGY REGISTRATION — are all strategies wired?
    # ═══════════════════════════════════════════════════
    print("\n🔗 5. STRATEGY REGISTRATION:")
    
    # What strategies exist?
    import os
    strategy_files = [f for f in os.listdir('strategies') if f.endswith('.py') and f != '__init__.py']
    print(f"  Strategy files: {strategy_files}")
    
    # Which ones does engine register?
    for sf in strategy_files:
        name = sf.replace('.py', '')
        if name in eng or name.replace('_', '') in eng.lower():
            print(f"  ✅ {sf} — referenced in engine")
        else:
            print(f"  ❓ {sf} — NOT referenced in engine (check if registered dynamically)")
    
    # ═══════════════════════════════════════════════════
    # 6. DATA FLOW — are strategies getting data?
    # ═══════════════════════════════════════════════════
    print("\n🔗 6. DATA FLOW (Market Data → Strategies):")
    
    has_process_market = 'def _process_market_event' in eng or 'def process_market_event' in eng
    has_calculate_signals = 'calculate_signals' in eng
    has_on_bar = 'on_bar' in eng
    
    print(f"  {'✅' if has_process_market else '❌'} Engine._process_market_event()")
    print(f"  {'✅' if has_calculate_signals else '❌'} strategy.calculate_signals() called")
    print(f"  {'✅' if has_on_bar else '❌'} strategy.on_bar() called")
    
    # ═══════════════════════════════════════════════════
    # 7. META COORDINATOR FLOW — is it actually running?
    # ═══════════════════════════════════════════════════
    print("\n🔗 7. META COORDINATOR:")
    
    mc = open('core/meta_coordinator.py', 'r', encoding='utf-8').read()
    has_start = '.start()' in eng and 'meta' in eng
    has_loop = '_arbitration_loop' in mc
    has_dedup = 'EXIT_DEDUPLICATION' in mc or 'dedup' in mc.lower()
    has_graph_veto = '_apply_graph_vetoes' in mc
    has_invariant = '_check_invariants' in mc
    
    print(f"  {'✅' if has_start else '❌'} MetaCoordinator.start() called in engine")
    print(f"  {'✅' if has_loop else '❌'} _arbitration_loop exists")
    print(f"  {'✅' if has_dedup else '❌'} EXIT deduplication")
    print(f"  {'✅' if has_graph_veto else '❌'} Graph vetoes")
    print(f"  {'✅' if has_invariant else '❌'} Invariant checking")
    
    # ═══════════════════════════════════════════════════
    # 8. SILENT EXCEPTIONS in engine (the worst)
    # ═══════════════════════════════════════════════════
    print("\n🔗 8. SILENT EXCEPTIONS IN ENGINE:")
    for i, line in enumerate(eng_lines):
        stripped = line.strip()
        if stripped == 'pass' and i > 0:
            prev = eng_lines[i-1].strip()
            if 'except' in prev:
                print(f"  ⚠️ engine.py:{i+1} — {prev} → pass (SILENT!)")

trace_engine_flow()
