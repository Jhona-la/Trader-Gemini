import os
import sys
import json

# Add root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from core.forensic_auditor import ForensicAuditor

def extract_config_state(config_class, prefix="Config"):
    state = {}
    for key in dir(config_class):
        if not key.startswith("__"):
            val = getattr(config_class, key)
            if isinstance(val, (int, float, str, bool, list, dict, tuple)):
                state[f"{prefix}.{key}"] = val
            elif type(val) == type:
                state.update(extract_config_state(val, f"{prefix}.{key}"))
    return state

def dump_config_diff():
    print("="*60)
    print("🔍 INICIANDO VOLCADO DE MEMORIA DE CONFIGURACIÓN")
    print("="*60)
    
    # Dump active config
    config_state = extract_config_state(Config)
    
    print(f"✅ Variables de configuración extraídas: {len(config_state)}")
    
    # Save the dump
    dump_path = os.path.join(os.path.dirname(__file__), "config_production_dump.json")
    with open(dump_path, 'w') as f:
        json.dump(config_state, f, indent=4)
        
    print(f"✅ Volcado guardado en: {dump_path}")
    
    # We will instantiate the backtest infrastructure and see if it mutates anything
    # We need to snapshot before and after to see if there are divergences
    
    print("\n🔍 Simulando inicialización de God Mode Backtest...")
    from scripts.run_god_mode_backtest import run_global_backtest
    
    # We can't easily run the full backtest without triggering the 28-hour run, 
    # but we can check if ForensicAuditor.verify_parity detects anything natively.
    
    is_valid = ForensicAuditor.verify_parity(Config)
    print(f"\n✅ ForensicAuditor.verify_parity(Config) devolvió: {is_valid}")
    
    # If there are mutations injected by the DNA mutator, they would be applied here.
    from core.dna_mutator import DNAMutator
    
    print("\n🔍 Verificando Mutaciones Genéticas (DNA_Loader)...")
    mutator = DNAMutator()
    print(f"Mutator Status: Enabled={mutator.is_enabled}")
    
    mutated_config_state = extract_config_state(Config)
    
    diffs = []
    for k, v in config_state.items():
        if k in mutated_config_state and mutated_config_state[k] != v:
            diffs.append((k, v, mutated_config_state[k]))
            
    if not diffs:
        print("\n✅ NO SE DETECTARON DIVERGENCIAS TRAS INICIALIZACIÓN. PARIDAD 100%.")
    else:
        print("\n❌ DIVERGENCIAS DETECTADAS:")
        for k, v1, v2 in diffs:
            print(f"  - {k}: {v1} -> {v2}")

if __name__ == "__main__":
    dump_config_diff()
