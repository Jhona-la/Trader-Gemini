
import os
import json
import numpy as np
from datetime import datetime
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

def audit_fabric():
    print("🧬 [FABRIC AUDIT] Analyzing Neuro-Evolutionary Convergence...")
    genotype_dir = "data/genotypes"
    
    if not os.path.exists(genotype_dir):
        print("❌ Error: Genotype directory not found.")
        return

    files = [f for f in os.listdir(genotype_dir) if f.endswith("_gene.json")]
    print(f"   Found {len(files)} universes (symbols) in the fabric.")
    print("-" * 60)
    print(f"{'SYMBOL':<15} | {'GEN':<5} | {'FITNESS':<8} | {'STATUS':<10}")
    print("-" * 60)

    for f in files:
        path = os.path.join(genotype_dir, f)
        try:
            with open(path, 'r') as j:
                data = json.load(j)
            
            symbol = data.get('symbol', f.split('_')[0])
            gen = data.get('generation', 0)
            fitness = data.get('fitness_score', 0.0)
            genes = data.get('genes', {})
            
            # Convergence Check: Are parameters at the edge of clamping?
            is_clamped = False
            # Bounds: TP [0.005, 0.10], SL [0.003, 0.05], Strength [0.1, 0.95]
            if genes.get('tp_pct') in [0.005, 0.10]: is_clamped = True
            if genes.get('sl_pct') in [0.003, 0.05]: is_clamped = True
            
            status = "HEALTHY"
            if is_clamped: status = "CLAMPED"
            if fitness < 0.2 and gen > 10: status = "STRUGGLING"
            
            print(f"{symbol:<15} | {gen:<5} | {fitness:<8.4f} | {status:<10}")
            
        except Exception as e:
            print(f"❌ Error auditing {f}: {e}")

    print("-" * 60)
    print("📝 AUDIT COMPLETE.")

if __name__ == "__main__":
    audit_fabric()
