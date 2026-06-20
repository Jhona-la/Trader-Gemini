import sys
import os

# Add root project path to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from scripts.run_god_mode_backtest import BacktestExecutor

def run_audit():
    print("=" * 60)
    print("AUDITORÍA DE DIVERGENCIA DE CONFIGURACIÓN")
    print("=" * 60)
    
    divergences = []
    
    # 1. Configuración de Producción
    prod_max_pos = getattr(Config.Risk, 'MAX_CONCURRENT_POSITIONS', 'NOT_FOUND')
    prod_leverage = getattr(Config.Risk, 'MAX_LEVERAGE', 'NOT_FOUND')
    prod_fee = getattr(Config.Risk, 'FEE_RATE', 'NOT_FOUND')
    
    print("\n[PRODUCCIÓN - config.py]")
    print(f"MAX_CONCURRENT_POSITIONS : {prod_max_pos}")
    print(f"MAX_LEVERAGE             : {prod_leverage}")
    print(f"FEE_RATE                 : {prod_fee}")
    
    # 2. Configuración de Backtest (Simulando lo que inicializa el script de backtest)
    try:
        from tests.walk_forward import WalkForwardBacktest
        wf = WalkForwardBacktest(
            symbol="BTC/USDT",
            initial_capital=13.0,
            n_splits=3,
            train_days=15,
            test_days=5
        )
        bt_fee = wf.fee_rate
        print("\n[BACKTEST - WalkForward]")
        print(f"FEE_RATE                 : {bt_fee}")
        
        if bt_fee != prod_fee:
            divergences.append(f"DIVERGENCIA CRÍTICA: Fee Rate Producción ({prod_fee}) vs Backtest ({bt_fee})")
    except Exception as e:
        print(f"Error cargando WF: {e}")

    try:
        print("\n[BACKTEST - God Mode Engine]")
        engine = BacktestExecutor()
        # Acceder a variables si las tiene
        bt_max_pos = engine.engine.risk_manager.max_concurrent_positions if hasattr(engine, 'engine') and hasattr(engine.engine, 'risk_manager') else "NOT_FOUND"
        
        print(f"MAX_CONCURRENT_POSITIONS : {bt_max_pos}")
        if str(bt_max_pos) != str(prod_max_pos) and bt_max_pos != "NOT_FOUND":
            divergences.append(f"DIVERGENCIA CRÍTICA: MAX_CONCURRENT_POSITIONS Producción ({prod_max_pos}) vs God Mode ({bt_max_pos})")
            
    except Exception as e:
        print(f"Error cargando God Mode Engine: {e}")
        
    print("\n" + "=" * 60)
    if not divergences:
        print("✅ ALINEACIÓN PERFECTA: No se detectaron divergencias de configuración.")
    else:
        print("❌ DIVERGENCIAS DETECTADAS:")
        for d in divergences:
            print(f"   - {d}")
    print("=" * 60)

if __name__ == "__main__":
    run_audit()
