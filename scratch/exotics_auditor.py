import os
from pathlib import Path
import json

def audit_exotics(root_dir):
    exotics = {
        "DEX Whispering": False,
        "Hyperliquid Cascades": False,
        "RBF Urgency": False,
        "MEV Bundle Sniffing": False,
        "Kill Switch Synchronous": False,
        "Heartbeat ZMQ": False,
        "Quantum Ingester": False,
        "Surrogate Oracle": False,
        "Auto-Scaling Micro-Capital": False,
        "Shadow Mode": False,
        "Network Physics Wrapper": False,
        "COWConfigManager": False,
        "Espejo Cuántico": False,
        "ScalpState": False,
        "SwingState": False
    }

    keywords = {
        "DEX Whispering": ["dex", "jupiter", "whispering", "outamount"],
        "Hyperliquid Cascades": ["hyperliquid", "cascade", "hyper_kernel"],
        "RBF Urgency": ["rbf", "replace-by-fee", "urgency", "mev_rbf_engine"],
        "MEV Bundle Sniffing": ["mev", "bundle", "sniffing", "relay"],
        "Kill Switch Synchronous": ["_synchronous_panic_exit", "kill_switch"],
        "Heartbeat ZMQ": ["zmq", "heartbeat", "_heartbeat_loop"],
        "Quantum Ingester": ["quantum_ingester", "ingest_raw_ws_frame"],
        "Surrogate Oracle": ["surrogate", "optuna", "gaussian_process", "random_forest"],
        "Auto-Scaling Micro-Capital": ["5.05", "auto-scaling", "micro-capital", "strict_microcapital_veto"],
        "Shadow Mode": ["shadow_mode", "shadow"],
        "Network Physics Wrapper": ["networkphysicswrapper", "network_physics"],
        "COWConfigManager": ["cowconfigmanager", "copy-on-write", "copy_on_write"],
        "Espejo Cuántico": ["espejo", "quantum_mirror", "run_god_mode_backtest", "simulation"],
        "ScalpState": ["scalpstate", "mode='scalping'"],
        "SwingState": ["swingstate", "mode='swing'"]
    }

    exclude_dirs = {'.venv', 'venv', 'node_modules', '.git', '__pycache__', 'build', 'dist'}

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in ['.py', '.pyx', '.c', '.cpp', '.rs']:
                filepath = Path(root) / file
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().lower()
                        for key, words in keywords.items():
                            if not exotics[key]:
                                for word in words:
                                    if word.lower() in content:
                                        exotics[key] = True
                                        break
                except Exception as e:
                    pass
                    
    return exotics

if __name__ == "__main__":
    project_root = Path("C:/Users/jhona/Documents/Proyectos/Trader Gemini")
    print("Scanning for Exotics and Axiom modules...")
    results = audit_exotics(project_root)
    
    output_file = project_root / "scratch" / "exotics_findings.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    for k, v in results.items():
        print(f"{k}: {'FOUND' if v else 'MISSING'}")
