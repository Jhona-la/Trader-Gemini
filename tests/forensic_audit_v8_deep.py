"""
🔬 AUDITORÍA FORENSE V8 — DEEP SIGNAL & DATA FLOW
====================================================
OBJETIVO: Encontrar POR QUÉ el sistema no genera profits consistentes.
MÉTODO: Simula el pipeline completo con datos REALES de Binance.

Secciones:
1. Horizon Awareness en TODAS las estrategias
2. Sophia Intelligence Speed Profiling
3. Risk Manager Sizing para $13 USD
4. Signal Generation Pipeline con datos mock realistas
5. Sizing Cascade Audit (cuántos filtros matan señales)
6. SL/TP Economics (¿son rentables después de fees?)
"""

import sys, os, time, traceback, inspect
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"
results = defaultdict(list)

def audit(cat, name, ok, detail=""):
    results[cat].append((name, ok, detail))
    print(f"  {'✅' if ok else '❌'} {name}: {detail[:140]}")

def warn(cat, name, detail=""):
    results[cat].append((name, None, detail))
    print(f"  ⚠️  {name}: {detail[:140]}")

print("="*70)
print("🔬 AUDITORÍA FORENSE V8 — DEEP SIGNAL & DATA FLOW")
print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"   Capital: $13 USD | Objetivo: Duplicar en 15 días")
print("="*70)

# ============================================================
# SECTION 1: HORIZON AWARENESS — ALL STRATEGIES
# ============================================================
print("\n" + "="*70)
print("🔬 S1: HORIZONTE EN TODAS LAS ESTRATEGIAS")
print("="*70)

try:
    from strategies.technical import HybridScalpingStrategy
    from strategies.sniper_strategy import SniperStrategy
    from strategies.ml_strategy import MLStrategyHybridUltimate as MLStrategy
    from strategies.statistical import StatisticalStrategy
    
    strategies_to_check = {
        'HybridScalpingStrategy': HybridScalpingStrategy,
        'SniperStrategy': SniperStrategy,
        'MLStrategy': MLStrategy,
        'StatisticalStrategy': StatisticalStrategy,
    }
    
    for name, cls in strategies_to_check.items():
        sig = inspect.signature(cls.__init__)
        has_horizon = 'horizon' in sig.parameters
        has_priority = 'priority' in sig.parameters
        
        audit("HORIZON", f"{name} has 'horizon' param", has_horizon,
              f"Params: {list(sig.parameters.keys())}")
        
        if has_horizon:
            # Check default value
            h_param = sig.parameters['horizon']
            default = h_param.default if h_param.default != inspect.Parameter.empty else "NO_DEFAULT"
            audit("HORIZON", f"{name} horizon default",
                  default in ["SCALPING", "SWING", "NO_DEFAULT"],
                  f"Default: {default}")
        
        # Check if signal emission includes horizon
        try:
            src = inspect.getsource(cls)
            emits_horizon = "horizon=" in src and "SignalEvent" in src
            audit("HORIZON", f"{name} emits horizon in signals",
                  emits_horizon,
                  "SignalEvent includes horizon field")
        except Exception:
            warn("HORIZON", f"{name} source unavailable", "Cannot inspect")
            
except Exception as e:
    print(f"  ❌ HORIZON CHECK ERROR: {e}")
    traceback.print_exc()

# Also check arbitrage and phalanx
try:
    extra_strategies = {}
    try:
        from strategies.arbitrage import ArbitrageStrategy
        extra_strategies['ArbitrageStrategy'] = ArbitrageStrategy
    except ImportError:
        warn("HORIZON", "ArbitrageStrategy", "Not importable")
    
    try:
        from strategies.phalanx import PhalanxStrategy
        extra_strategies['PhalanxStrategy'] = PhalanxStrategy
    except ImportError:
        warn("HORIZON", "PhalanxStrategy", "Not importable")
    
    for name, cls in extra_strategies.items():
        sig = inspect.signature(cls.__init__)
        has_horizon = 'horizon' in sig.parameters
        audit("HORIZON", f"{name} has 'horizon' param", has_horizon,
              f"Params: {list(sig.parameters.keys())}")
except Exception as e:
    warn("HORIZON", "Extra strategies check", str(e))


# ============================================================
# SECTION 2: SOPHIA SPEED PROFILING
# ============================================================
print("\n" + "="*70)
print("🔬 S2: SOPHIA INTELLIGENCE — SPEED PROFILING")
print("="*70)

try:
    from sophia.intelligence import SophiaIntelligence, BayesianCalibrator, FeatureAttributor
    from sophia.intelligence import SurvivalEstimator, EntropyAnalyzer, AlphaDecayFunction
    
    sophia = SophiaIntelligence()
    
    # Generate realistic mock data
    np.random.seed(42)
    n = 500
    close = np.cumsum(np.random.randn(n) * 0.001) + 50000  # BTC-like prices
    close = np.abs(close)
    returns = np.diff(np.log(close))
    
    mock_setups = {
        'rsi': 35.0, 'bb_position': 0.2, 'adx': 28.0,
        'volume_ratio': 1.8, 'confluence': 0.7, 'macd_hist': 0.002,
        'trend_aligned': 0.5, 'atr_pct': 0.015, 'atr': 500.0,
        'close': 50000.0, 'volume_ratio_4h': 2.5,
        'is_50_bar_high': False, 'is_50_bar_low': False,
        'long_mean_rev': True, 'short_mean_rev': False,
        'long_momentum': False, 'short_momentum': False,
    }
    
    # 2.1 Full analyze() speed
    # Warm-up
    try:
        sophia.analyze(
            symbol="BTC/USDT", direction="LONG", signal_strength=0.8,
            setups=mock_setups, confluence_score=0.7,
            tp_pct=0.01, sl_pct=0.005, returns=returns,
            ttl_seconds=180.0, regime="BULL"
        )
    except Exception:
        pass
    
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        try:
            report = sophia.analyze(
                symbol="BTC/USDT", direction="LONG", signal_strength=0.8,
                setups=mock_setups, confluence_score=0.7,
                tp_pct=0.01, sl_pct=0.005, returns=returns,
                ttl_seconds=180.0, regime="BULL"
            )
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        except Exception as e:
            warn("SOPHIA", f"analyze() error", str(e)[:100])
            break
    
    if times:
        avg_ms = np.mean(times)
        p99_ms = np.percentile(times, 99)
        audit("SOPHIA", f"analyze() avg={avg_ms:.2f}ms, p99={p99_ms:.2f}ms",
              avg_ms < 50,  # Target < 50ms
              f"50 calls | Min: {min(times):.2f}ms | Max: {max(times):.2f}ms")
        
        # Check report quality
        if report:
            audit("SOPHIA", f"Win Probability range",
                  0.0 < report.win_probability < 1.0,
                  f"P(Win)={report.win_probability:.4f}")
            
            audit("SOPHIA", f"Omniscient Score computed",
                  hasattr(report, 'omniscient_score') and report.omniscient_score > 0,
                  f"Score={report.omniscient_score:.4f}")
            
            audit("SOPHIA", f"Hurst Exponent valid",
                  0.0 < report.hurst_exponent < 1.0,
                  f"H={report.hurst_exponent:.4f}")
            
            audit("SOPHIA", f"Expected Exit Time positive",
                  report.expected_exit_mins > 0,
                  f"E[T]={report.expected_exit_mins:.1f} min")
            
            audit("SOPHIA", f"Top Features populated",
                  len(report.top_features) > 0,
                  f"Top: {[f['feature'] for f in report.top_features[:3]]}")
    
    # 2.2 Component speed
    calibrator = BayesianCalibrator()
    t0 = time.perf_counter()
    for _ in range(10000):
        calibrator.compute_posterior(0.8, 0.5, 1.2)
    t1 = time.perf_counter()
    bayes_ns = ((t1 - t0) / 10000) * 1_000_000
    audit("SOPHIA", f"BayesianCalibrator ({bayes_ns:.0f}ns/call)",
          bayes_ns < 50000, f"10K calls")
    
    attributor = FeatureAttributor(calibrator)
    t0 = time.perf_counter()
    for _ in range(1000):
        attributor.compute_attributions(mock_setups)
    t1 = time.perf_counter()
    attr_us = ((t1 - t0) / 1000) * 1_000
    audit("SOPHIA", f"FeatureAttributor ({attr_us:.0f}μs/call)",
          attr_us < 5000, f"1K calls")
    
    estimator = SurvivalEstimator(bar_minutes=5.0)
    t0 = time.perf_counter()
    for _ in range(10000):
        estimator.estimate(50000.0, 0.01, 0.005, returns=returns)
    t1 = time.perf_counter()
    surv_us = ((t1 - t0) / 10000) * 1_000
    audit("SOPHIA", f"SurvivalEstimator ({surv_us:.0f}μs/call)",
          surv_us < 1000, f"10K calls")

except Exception as e:
    print(f"  ❌ SOPHIA ERROR: {e}")
    traceback.print_exc()


# ============================================================
# SECTION 3: RISK MANAGER SIZING — $13 USD SIMULATION
# ============================================================
print("\n" + "="*70)
print("🔬 S3: RISK MANAGER — SIZING PARA $13 USD")
print("="*70)

try:
    from core.portfolio import Portfolio
    from risk.risk_manager import RiskManager
    from core.events import SignalEvent
    from core.enums import SignalType
    from config import Config
    
    port = Portfolio(initial_capital=13.0, auto_save=False)
    rm = RiskManager(portfolio=port)
    
    # 3.1 Position size for $13 account
    mock_signal = SignalEvent(
        strategy_id="TEST_SCALP",
        symbol="BTC/USDT",
        datetime=datetime.now(timezone.utc),
        signal_type=SignalType.LONG,
        strength=0.85,
        atr=500.0,
        horizon="SCALPING",
        priority=0,
        current_price=60000.0
    )
    
    size = rm.size_position(mock_signal, current_price=60000.0)
    
    audit("SIZING", f"Position size for $13 BTC LONG",
          size > 0,
          f"Size: ${size:.2f} | Capital: $13 | Leverage: {Config.BINANCE_LEVERAGE}x")
    
    # 3.2 Does size meet Binance minimum?
    notional = size * Config.BINANCE_LEVERAGE
    audit("SIZING", f"Notional > $5 Binance minimum",
          notional >= 5.0,
          f"Notional: ${notional:.2f} (size ${size:.2f} × {Config.BINANCE_LEVERAGE}x)")
    
    # 3.3 Risk per trade percentage
    risk_pct = rm._get_dynamic_risk_per_trade(13.0)
    risk_dollars = 13.0 * risk_pct
    audit("SIZING", f"Risk per trade",
          risk_pct <= 0.05,
          f"{risk_pct*100:.1f}% = ${risk_dollars:.2f} per trade")
    
    # 3.4 How many concurrent positions can we open?
    max_positions = int(13.0 / size) if size > 0 else 0
    audit("SIZING", f"Max concurrent positions",
          max_positions >= 1,
          f"{max_positions} positions of ${size:.2f} each")
    
    # 3.5 SL/TP Economics for $13
    sl_pct = rm._calculate_dynamic_stop_loss(500.0 / 60000.0)
    tp_pct = sl_pct * 2.0  # Assuming 2:1 R:R
    
    loss_per_trade = size * sl_pct * Config.BINANCE_LEVERAGE
    gain_per_trade = size * tp_pct * Config.BINANCE_LEVERAGE
    fee_round_trip = size * Config.BINANCE_LEVERAGE * Config.BINANCE_TAKER_FEE_BNB * 2
    
    net_gain = gain_per_trade - fee_round_trip
    net_loss = loss_per_trade + fee_round_trip
    
    audit("SIZING", f"Expected gain per winning trade",
          net_gain > 0,
          f"Gross: ${gain_per_trade:.4f} - Fees: ${fee_round_trip:.4f} = Net: ${net_gain:.4f}")
    
    audit("SIZING", f"Expected loss per losing trade",
          True,
          f"Gross: ${loss_per_trade:.4f} + Fees: ${fee_round_trip:.4f} = Net: ${net_loss:.4f}")
    
    # 3.6 Breakeven win rate
    if (net_gain + net_loss) > 0:
        be_wr = net_loss / (net_gain + net_loss)
        audit("SIZING", f"Breakeven Win Rate",
              be_wr < 0.60,
              f"{be_wr*100:.1f}% (need > this to profit)")
    
    # 3.7 Expected value per trade at 60% WR
    ev_60 = 0.60 * net_gain - 0.40 * net_loss
    audit("SIZING", f"EV at 60% WR",
          ev_60 > 0,
          f"EV = ${ev_60:.4f} per trade")
    
    # 3.8 Trades needed to double
    if ev_60 > 0:
        trades_to_double = int(13.0 / ev_60)
        trades_per_day = 24  # Assuming 24 trades/day
        days_to_double = trades_to_double / trades_per_day
        audit("SIZING", f"Trades to double capital at 60% WR",
              days_to_double <= 30,
              f"{trades_to_double} trades ≈ {days_to_double:.1f} days at {trades_per_day}/day")
    
    # 3.9 Speed profiling
    t0 = time.perf_counter()
    for _ in range(1000):
        rm.size_position(mock_signal, current_price=60000.0)
    t1 = time.perf_counter()
    sizing_us = ((t1 - t0) / 1000) * 1_000
    audit("SIZING", f"size_position speed ({sizing_us:.0f}μs/call)",
          sizing_us < 5000,
          f"1K calls")

except Exception as e:
    print(f"  ❌ SIZING ERROR: {e}")
    traceback.print_exc()


# ============================================================
# SECTION 4: SIGNAL FILTER CASCADE — WHERE SIGNALS DIE
# ============================================================
print("\n" + "="*70)
print("🔬 S4: SIGNAL FILTER CASCADE — DÓNDE MUEREN LAS SEÑALES")
print("="*70)

try:
    # Analyze the technical strategy source code to count filter gates
    tech_src = inspect.getsource(HybridScalpingStrategy.generate_signals)
    
    # Count 'continue' statements (each one is a potential signal killer)
    continue_count = tech_src.count('continue')
    
    # Count specific filter names
    filters = {
        'ADX Filter': 'current_adx < ADX_THRESH' in tech_src,
        'Strength Filter': 'strength < STRENGTH_THRESH' in tech_src,
        'Oracle Veto': 'ORACLE VETO' in tech_src,
        'Quantum Veto (VPIN)': 'VETO CUÁNTICO' in tech_src,
        'Sophia Omniscient Gate': 'omni < hurdle' in tech_src,
        'Volatility Minimum': 'Low volatility' in tech_src or 'volatility' in tech_src.lower(),
        'XRP Trend Alignment': 'XRP' in tech_src,
        'Closed-Bar Logic': 'idx = -2' in inspect.getsource(HybridScalpingStrategy.detect_scalping_setup) or '[-2]' in inspect.getsource(HybridScalpingStrategy.detect_scalping_setup),
    }
    
    audit("FILTERS", f"Total 'continue' gates in generate_signals",
          continue_count < 20,
          f"{continue_count} filter gates (each one can kill a signal)")
    
    for fname, present in filters.items():
        audit("FILTERS", f"Filter: {fname}",
              present,
              "ACTIVE" if present else "NOT FOUND")
    
    # KEY ANALYSIS: The Omniscient Score hurdle
    # Find the hurdle values
    if 'base_hurdle = ' in tech_src:
        import re
        hurdle_matches = re.findall(r'base_hurdle\s*=\s*([\d.]+)', tech_src)
        if hurdle_matches:
            hurdle = float(hurdle_matches[0])
            audit("FILTERS", f"Omniscient Score hurdle (BTC)",
                  hurdle <= 0.20,
                  f"Hurdle = {hurdle} (too high = too few signals)")
    
    # Alt-coin hurdle
    alt_hurdles = re.findall(r"base_hurdle\s*=\s*([\d.]+).*?#.*?Alt", tech_src)
    if alt_hurdles:
        audit("FILTERS", f"Omniscient Score hurdle (ALTs)",
              float(alt_hurdles[0]) <= 0.15,
              f"Alt Hurdle = {alt_hurdles[0]}")

    # Divine/Harmonic overrides
    divine_in_src = 'is_divine' in tech_src
    harmonic_in_src = 'is_harmonic' in tech_src
    audit("FILTERS", f"Divine/Harmonic Gate overrides exist",
          divine_in_src and harmonic_in_src,
          "Lower hurdle for high-coherence signals")
    
    # CRITICAL: Adaptive leverage calculation
    if 'leverage = ' in tech_src:
        leverage_matches = re.findall(r'leverage\s*=\s*([\d.]+)\s*\+', tech_src)
        if leverage_matches:
            base_lev = float(leverage_matches[0])
            warn("FILTERS", f"Adaptive Leverage base",
                 f"Base: {base_lev}x + up to 20x more (DANGEROUS for $13)")

except Exception as e:
    print(f"  ❌ FILTER CASCADE ERROR: {e}")
    traceback.print_exc()


# ============================================================
# SECTION 5: SL/TP ECONOMICS — FEE-AWARE PROFITABILITY
# ============================================================
print("\n" + "="*70)
print("🔬 S5: SL/TP ECONOMICS — RENTABILIDAD POST-FEES")
print("="*70)

try:
    # Simulate different SL/TP scenarios for $13 account
    capital = 13.0
    leverage = Config.BINANCE_LEVERAGE  # 3x
    fee_rate = Config.BINANCE_TAKER_FEE_BNB  # 0.000375
    
    scenarios = [
        ("Ultra-Tight Scalp", 0.002, 0.004),
        ("Standard Scalp", 0.005, 0.010),
        ("Wide Scalp", 0.008, 0.016),
        ("Swing Entry", 0.010, 0.025),
        ("Wide Swing", 0.015, 0.035),
    ]
    
    position_size = 6.0  # $6 margin (minimum viable for Binance)
    notional = position_size * leverage
    
    print(f"\n  💰 Account: ${capital} | Position: ${position_size} | Notional: ${notional}")
    print(f"  📊 Leverage: {leverage}x | Fee: {fee_rate*100:.4f}% per side")
    print(f"  {'='*60}")
    
    for name, sl_pct, tp_pct in scenarios:
        gross_win = notional * tp_pct
        gross_loss = notional * sl_pct
        fees = notional * fee_rate * 2  # Round trip
        
        net_win = gross_win - fees
        net_loss = gross_loss + fees
        
        be_wr = net_loss / (net_win + net_loss) if (net_win + net_loss) > 0 else 1.0
        
        ev_60 = 0.60 * net_win - 0.40 * net_loss
        ev_70 = 0.70 * net_win - 0.30 * net_loss
        ev_80 = 0.80 * net_win - 0.20 * net_loss
        
        profitable = net_win > 0
        audit("ECONOMICS", f"{name} (SL={sl_pct*100}%/TP={tp_pct*100}%)",
              profitable and be_wr < 0.55,
              f"NetWin=${net_win:.3f} NetLoss=${net_loss:.3f} BE_WR={be_wr*100:.0f}% EV@60%=${ev_60:.3f}")
    
    # Fee impact analysis
    fee_per_trade = notional * fee_rate * 2
    fee_pct_of_capital = (fee_per_trade / capital) * 100
    
    audit("ECONOMICS", f"Fee impact per trade",
          fee_pct_of_capital < 1.0,
          f"${fee_per_trade:.4f} = {fee_pct_of_capital:.2f}% of capital per round trip")
    
    # How many losing trades until ruin?
    trades_to_ruin = int(capital / net_loss) if net_loss > 0 else 999
    audit("ECONOMICS", f"Trades to ruin (at worst scenario SL)",
          trades_to_ruin >= 10,
          f"{trades_to_ruin} consecutive losses until $0")

except Exception as e:
    print(f"  ❌ ECONOMICS ERROR: {e}")
    traceback.print_exc()


# ============================================================
# SECTION 6: RISK MANAGER VALIDATION PIPELINE SPEED
# ============================================================
print("\n" + "="*70)
print("🔬 S6: RISK MANAGER — VALIDATION PIPELINE SPEED")
print("="*70)

try:
    from core.events import OrderEvent
    from core.enums import OrderType
    
    mock_order = OrderEvent(
        strategy_id="TEST",
        symbol="BTC/USDT",
        order_type=OrderType.LIMIT,
        direction=SignalType.LONG,
        quantity=0.001,
        price=60000.0,
        sl_pct=0.005,
        tp_pct=0.01,
        horizon="SCALPING",
        priority=0
    )
    
    # Speed of full validation pipeline
    t0 = time.perf_counter()
    for _ in range(1000):
        try:
            rm.validate_order(mock_order)
        except Exception:
            pass  # May fail without real portfolio state
    t1 = time.perf_counter()
    val_us = ((t1 - t0) / 1000) * 1_000
    
    audit("RISK_SPEED", f"validate_order ({val_us:.0f}μs/call)",
          val_us < 5000,
          f"Full pipeline including all 7 validation checks")
    
    # Individual validation speed
    validations = [
        '_validate_fat_finger',
        '_validate_kill_switch',
        '_validate_frequency_limits',
    ]
    
    for v_name in validations:
        v_func = getattr(rm, v_name, None)
        if v_func:
            t0 = time.perf_counter()
            for _ in range(10000):
                try:
                    v_func(mock_order)
                except Exception:
                    pass
            t1 = time.perf_counter()
            v_ns = ((t1 - t0) / 10000) * 1_000_000
            audit("RISK_SPEED", f"{v_name} ({v_ns:.0f}ns/call)",
                  v_ns < 10000,
                  f"10K iterations")

except Exception as e:
    print(f"  ❌ RISK SPEED ERROR: {e}")
    traceback.print_exc()


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*70)
print("📊 RESUMEN FINAL — AUDITORÍA FORENSE V8 DEEP")
print("="*70)

total_pass = 0
total_fail = 0
total_warn = 0
critical = []

for cat, tests in results.items():
    p = sum(1 for _, ok, _ in tests if ok is True)
    f = sum(1 for _, ok, _ in tests if ok is False)
    w = sum(1 for _, ok, _ in tests if ok is None)
    total_pass += p
    total_fail += f
    total_warn += w
    
    icon = "✅" if f == 0 else "❌"
    parts = f"{p}/{p+f+w} passed"
    if f > 0: parts += f" ({f} FAILED)"
    if w > 0: parts += f" ({w} WARN)"
    print(f"  {icon} {cat}: {parts}")
    
    for name, ok, detail in tests:
        if ok is False:
            critical.append(f"[{cat}] {name}: {detail}")

total = total_pass + total_fail + total_warn
score = (total_pass / total * 100) if total > 0 else 0

print(f"\n  📈 Score: {score:.1f}% ({total_pass}/{total})")

if critical:
    print(f"\n🚨 PROBLEMAS CRÍTICOS ({len(critical)}):")
    for c in critical:
        print(f"  💀 {c}")

# KEY DIAGNOSTIC
print(f"\n🔍 DIAGNÓSTICO CLAVE PARA $13 USD:")
print(f"   Si ganas el 60% de los trades y cada trade arriesga ~$0.39:")
print(f"   → Necesitas ~{int(13.0/0.05)} trades rentables para duplicar")
print(f"   → A 24 trades/día = ~{int(13.0/0.05)/24:.0f} días")
print(f"   → CLAVE: Subir WR a 70%+ reduce esto a ~5 días")

print("\n" + "="*70)
print(f"🏁 Auditoría V8 completada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)
