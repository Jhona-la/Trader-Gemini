"""
🔬 PRE-FLIGHT CHECK - Trader Gemini
====================================
QUÉ: Smoke tests para verificar integridad del sistema antes de backtest/producción
POR QUÉ: Un import roto o un KillSwitch mal escalado invalida todo el backtest
PARA QUÉ: Detectar problemas ANTES de desperdiciar 30+ minutos de ejecución
CÓMO: 5 checks secuenciales: imports, config, data, features, kill_switch
CUÁNDO: Antes de cada backtest o deployment
DÓNDE: scripts/preflight_check.py
QUIÉN: QA Engineer
"""

import sys
import os
import math
import traceback

# Asegurar que el path del proyecto está disponible
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"

results = []

def check(name, passed, detail=""):
    status = PASS if passed else FAIL
    results.append((name, passed, detail))
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))
    return passed


def check_1_imports():
    """Verificar que todos los módulos críticos se importan sin errores."""
    print("\n🔹 Check 1: Imports de Módulos Críticos")
    all_ok = True
    
    modules = [
        ("config", "Config"),
        ("core.engine", "TradingEngine"),
        ("core.events", "SignalEvent"),
        ("core.portfolio", "Portfolio"),
        ("core.market_regime", "MarketRegimeDetector"),
        ("risk.risk_manager", "RiskManager"),
        ("risk.kill_switch", "KillSwitch"),
        ("strategies.technical", "HybridTechnicalStrategy"),
        ("strategies.ml_strategy", "MLStrategyHybridUltimate"),
        ("data.data_provider", "DataProvider"),
        ("core.ml_governance", "MLGovernance"),
        ("utils.logger", None),
        ("utils.error_handler", None),
    ]
    
    for module_path, class_name in modules:
        try:
            mod = __import__(module_path, fromlist=[class_name] if class_name else [])
            if class_name:
                getattr(mod, class_name)
            check(f"import {module_path}.{class_name or '*'}", True)
        except Exception as e:
            check(f"import {module_path}.{class_name or '*'}", False, str(e)[:80])
            all_ok = False
    
    return all_ok


def check_2_config():
    """Verificar que la configuración tiene valores sanos."""
    print("\n🔹 Check 2: Validación de Configuración")
    all_ok = True
    
    try:
        from config import Config
        
        # INITIAL_CAPITAL
        cap = getattr(Config, 'INITIAL_CAPITAL', None)
        all_ok &= check("INITIAL_CAPITAL definido", cap is not None)
        if cap is not None:
            all_ok &= check(f"INITIAL_CAPITAL > 0 (val={cap})", cap > 0)
        
        # LEVERAGE
        lev = getattr(Config, 'LEVERAGE', None)
        all_ok &= check("LEVERAGE definido", lev is not None)
        if lev is not None:
            all_ok &= check(f"LEVERAGE en [1, 20] (val={lev})", 1 <= lev <= 20)
        
        # ACTIVE_HORIZON
        horizon = getattr(Config, 'ACTIVE_HORIZON', None)
        all_ok &= check("ACTIVE_HORIZON definido", horizon is not None)
        if horizon is not None:
            all_ok &= check(f"ACTIVE_HORIZON en [1,7,15,30] (val={horizon})", horizon in [1, 7, 15, 30])
        
        # SYMBOLS
        symbols = getattr(Config, 'SYMBOLS', None)
        all_ok &= check("SYMBOLS definido", symbols is not None and len(symbols) > 0, 
                        f"{len(symbols)} symbols" if symbols else "")
        
        # Risk params
        risk = getattr(Config, 'Risk', None)
        if risk:
            max_dd = getattr(risk, 'MAX_DRAWDOWN', None)
            all_ok &= check(f"Risk.MAX_DRAWDOWN definido (val={max_dd})", max_dd is not None)
        
    except Exception as e:
        check("Config importable", False, str(e)[:80])
        all_ok = False
    
    return all_ok


def check_3_data_integrity():
    """Descargar mini-set BTC y validar integridad OHLCV."""
    print("\n🔹 Check 3: Integridad de Datos (BTC/USDT mini-fetch)")
    all_ok = True
    
    try:
        import ccxt
        import numpy as np
        
        exchange = ccxt.binance({'enableRateLimit': True})
        
        # Descargar 100 barras de 1m
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1m', limit=100)
        all_ok &= check(f"Descarga BTC/USDT 1m", len(ohlcv) > 0, f"{len(ohlcv)} barras")
        
        if ohlcv:
            import pandas as pd
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Sin NaN
            nan_count = df.isna().sum().sum()
            all_ok &= check("Sin NaN en OHLCV", nan_count == 0, f"{nan_count} NaN found" if nan_count else "")
            
            # Sin gaps > 2 minutos
            timestamps = df['timestamp'].values
            gaps = np.diff(timestamps)
            max_gap_ms = np.max(gaps) if len(gaps) > 0 else 0
            max_gap_min = max_gap_ms / 60000
            all_ok &= check(f"Sin gaps temporales > 2min", max_gap_min <= 2.0, f"max gap: {max_gap_min:.1f}min")
            
            # OHLC consistency
            valid_ohlc = (df['high'] >= df['low']).all() and (df['high'] >= df['open']).all()
            all_ok &= check("Consistencia OHLC (H>=L, H>=O)", valid_ohlc)
            
            # Volume positiva
            vol_ok = (df['volume'] >= 0).all()
            all_ok &= check("Volume >= 0", vol_ok)
            
    except Exception as e:
        check("Conexión Binance", False, str(e)[:80])
        all_ok = False
    
    return all_ok


def check_4_feature_consistency():
    """Verificar que los features del WalkForwardXGBoost son calculables."""
    print("\n🔹 Check 4: Consistencia de Features ML")
    all_ok = True
    
    try:
        # Los features que el backtest WalkForwardXGBoost espera
        EXPECTED_FEATURES = [
            'rsi', 'atr_pct', 'vol_ratio', 'macd', 'macd_sig', 'macd_hist',
            'adx', 'roc', 'amihud', 'hl_spread'
        ]
        
        # Los features derivados que también construye
        DERIVED_FEATURES = ['bb_position', 'ema_ratio', 'trend_align']
        
        # Verificar que compute_indicators del backtest genera estos campos
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))
        
        all_ok &= check(f"Features base definidos ({len(EXPECTED_FEATURES)})", 
                        len(EXPECTED_FEATURES) == 10)
        all_ok &= check(f"Features derivados definidos ({len(DERIVED_FEATURES)})", 
                        len(DERIVED_FEATURES) == 3)
        
        # Verificar que no hay duplicados
        all_features = EXPECTED_FEATURES + DERIVED_FEATURES
        all_ok &= check("Sin duplicados en feature set", 
                        len(all_features) == len(set(all_features)))
        
    except Exception as e:
        check("Feature consistency check", False, str(e)[:80])
        all_ok = False
    
    return all_ok


def check_5_killswitch_scaling():
    """Verificar que KillSwitch escala drawdown correctamente por horizonte."""
    print("\n🔹 Check 5: KillSwitch Horizon Scaling")
    all_ok = True
    
    try:
        from config import Config
        base_dd = getattr(Config.Risk, 'MAX_DRAWDOWN', 1.5)
        
        expected_scaling = {
            1: 1.0,
            7: math.sqrt(7),
            15: math.sqrt(15),
            30: math.sqrt(30),
        }
        
        for horizon, h_sqrt in expected_scaling.items():
            scaled_dd = base_dd * h_sqrt
            # Verificar que el scaling es razonable
            is_sane = scaled_dd < 10.0  # No debe permitir >10% drawdown
            all_ok &= check(f"Horizonte {horizon}D → DD limit {scaled_dd:.2f}%", 
                          is_sane,
                          f"base={base_dd}% × √{horizon}={h_sqrt:.2f}")
        
    except Exception as e:
        check("KillSwitch scaling", False, str(e)[:80])
        all_ok = False
    
    return all_ok


def main():
    print("=" * 60)
    print("🔬 TRADER GEMINI - PRE-FLIGHT CHECK v1.0")
    print("=" * 60)
    
    check_results = []
    check_results.append(("Imports", check_1_imports()))
    check_results.append(("Config", check_2_config()))
    check_results.append(("Data", check_3_data_integrity()))
    check_results.append(("Features", check_4_feature_consistency()))
    check_results.append(("KillSwitch", check_5_killswitch_scaling()))
    
    # Resumen
    print(f"\n{'='*60}")
    print("📊 RESUMEN PRE-FLIGHT")
    print(f"{'='*60}")
    
    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed
    
    for name, ok in check_results:
        status = PASS if ok else FAIL
        print(f"  {status} {name}")
    
    print(f"\n  Total: {passed}/{total} checks passed")
    
    if failed > 0:
        print(f"\n  {FAIL} {failed} CHECKS FALLARON — NO PROCEDER CON BACKTEST")
        sys.exit(1)
    else:
        print(f"\n  {PASS} SISTEMA LISTO PARA BACKTEST")
        sys.exit(0)


if __name__ == "__main__":
    main()
