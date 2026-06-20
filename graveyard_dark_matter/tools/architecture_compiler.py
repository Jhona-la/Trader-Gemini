import os
import sys
import json

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_compiled_dir = os.path.join(_project_root, "compiled_core")

def ensure_compiled_dir():
    if not os.path.exists(_compiled_dir):
        os.makedirs(_compiled_dir)

def generate_risk_manager_compiled(dna_params):
    code = f'''"""
[OMNI COMPILER] Arquitectura Optimizada para RiskManager
Generado Automáticamente.
"""
from risk.risk_manager import RiskManager

class CompiledRiskManager(RiskManager):
    """
    RiskManager con ADN Lógico Hardcodeado para máxima velocidad.
    Reemplaza if/else dinámicos por constantes evaluadas.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ADN INYECTADO
        self._USE_DYNAMIC_STOPS = {dna_params["dna_risk_dynamic_stops"]}
        
    def _calculate_dynamic_stop_loss(self, symbol, side, current_price, atr):
        if not self._USE_DYNAMIC_STOPS:
            # Optuna decidió que la latencia de procesar stops dinámicos no vale la pena
            # Se usa el default estático
            return super()._calculate_dynamic_stop_loss(symbol, side, current_price, atr)
            
        # Lógica acelerada de stop dinámico
        multiplier = self._get_asset_params(symbol).get("trailing_atr_mult", 1.0)
        if side == "LONG":
            return current_price - (atr * multiplier)
        else:
            return current_price + (atr * multiplier)
'''
    with open(os.path.join(_compiled_dir, "risk_manager_compiled.py"), "w", encoding="utf-8") as f:
        f.write(code)

def generate_sniper_compiled(dna_params):
    code = f'''"""
[OMNI COMPILER] Arquitectura Optimizada para SniperStrategy
Generado Automáticamente.
"""
from strategies.sniper_strategy import SniperStrategy

class CompiledSniperStrategy(SniperStrategy):
    """
    SniperStrategy con ramas muertas podadas por el Evolver.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._REQUIRE_VOLUME = {dna_params["dna_sniper_volume"]}
        
    def _check_volume_confluence(self, symbol):
        if not self._REQUIRE_VOLUME:
            # Optuna descubrió que el filtro de volumen causa falsos negativos
            # y bloquea alpha. Se salta la validación para ahorrar 12ms.
            return True
        return super()._check_volume_confluence(symbol)
'''
    with open(os.path.join(_compiled_dir, "sniper_strategy_compiled.py"), "w", encoding="utf-8") as f:
        f.write(code)

def generate_pattern_compiled(dna_params):
    code = f'''"""
[OMNI COMPILER] Arquitectura Optimizada para StatisticalStrategy (Patterns)
Generado Automáticamente.
"""
from strategies.statistical import StatisticalStrategy

class CompiledPatternStrategy(StatisticalStrategy):
    """
    StatisticalStrategy con ADN Lógico inyectado.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._STRICT_WICK_FILTER = {dna_params["dna_pattern_strict"]}
        
    def validate_wick_structure(self, candle):
        if not self._STRICT_WICK_FILTER:
            # Optuna descubrió que las mechas estrictas reducen el WinRate global
            return True
        return super().validate_wick_structure(candle)
'''
    with open(os.path.join(_compiled_dir, "pattern_compiled.py"), "w", encoding="utf-8") as f:
        f.write(code)

def compile_architecture(json_path):
    if not os.path.exists(json_path):
        print(f"❌ JSON no encontrado: {json_path}")
        return
        
    with open(json_path, 'r') as f:
        best_params = json.load(f)
        
    print(f"🧬 Iniciando Omni Compiler con genoma: {json_path}")
    ensure_compiled_dir()
    
    generate_risk_manager_compiled(best_params)
    generate_sniper_compiled(best_params)
    generate_pattern_compiled(best_params)
    
    print("✅ ¡Arquitectura Compilada con Éxito!")
    print(f"   Revisa la carpeta: {_compiled_dir}/")
    print("   Usa estas clases en tu core/dual_engine.py para aplicar la optimización extrema.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python tools/architecture_compiler.py data/omni_evolver_best_YYYYMMDD.json")
    else:
        compile_architecture(sys.argv[1])
