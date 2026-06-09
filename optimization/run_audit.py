import sys
import os
import logging

sys.path.append(r"c:\Users\jhona\Documents\Proyectos\Trader Gemini")

from optimization.optimizer_core import OptimizerCore
from optimization.strategy_integrator import StrategyIntegrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def audit_strategy(strategy_name: str):
    logging.info(f"🚀 INICIANDO AUDITORÍA MASIVA PARA: {strategy_name}")
    integrator = StrategyIntegrator(strategy_name)
    opt_core = OptimizerCore()
    
    yaml_result = opt_core.execute_perpetual_cycle(evaluation_func=integrator.evaluate_config, strategy_name=strategy_name)
    
    os.makedirs("scratch/logs", exist_ok=True)
    with open(f"scratch/logs/opt_protocol_{strategy_name}.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_result)
    logging.info(f"✅ PROTOCOLO PARA {strategy_name} GUARDADO.\n")

if __name__ == "__main__":
    strategies = ["MICRO", "SCALPING", "SWING"]
    for strat in strategies:
        audit_strategy(strat)
    
    logging.info("🎯 AUDITORÍA DE TODAS LAS ESTRATEGIAS FINALIZADA.")
