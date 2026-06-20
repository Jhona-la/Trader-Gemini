"""
😈 EVOLUTION DAEMON (Shadow Darwin Background Process)
======================================================
QUÉ: Un demonio independiente que ejecuta la Optimización Bayesiana de Optuna (Shadow Darwin).
POR QUÉ: La evolución bayesiana toma minutos/horas y consume CPU. Si lo conectamos directamente 
         a `engine.py`, congelaría el bot en vivo y causaríamos liquidaciones por latencia.
PARA QUÉ: Permitir que el sistema evolucione, encuentre el mejor ADN y escriba los parámetros dorados 
          en disco (hot-reload) MIENTRAS el bot principal sigue haciendo scalping a nano-latencia.
CÓMO USAR: Ejecutar este script en una terminal separada:
           > python scripts/evolution_daemon.py
"""

import time
import os
import sys
import logging

# Ensure root path is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from core.shadow_darwin import ShadowDarwin
from core.simulation import SimDataProvider
from utils.logger import setup_logger

logger = setup_logger("EvolutionDaemon", level=logging.INFO)

def run_evolution_daemon(cycle_delay_minutes=60, trials_per_epoch=20):
    """
    Despierta cada X minutos, descarga los últimos datos, corre simulaciones
    cruzadas (walk-forward) y guarda el mejor genoma.
    """
    logger.info("😈 [EVOLUTION DAEMON] Starting Background Evolution...")
    
    # We only need the historical fetcher from data provider
    from data.data_provider import DataProvider
    live_dp = DataProvider()
    
    symbols = getattr(Config.Strategies, 'WHITELISTED_SYMBOLS', ['BTC/USDT', 'ETH/USDT'])
    
    while True:
        try:
            logger.info("==================================================")
            logger.info("🧬 [EVOLUTION DAEMON] Waking up for new Epoch...")
            
            # Fetch fresh historical data to train on the immediate past
            for symbol in symbols:
                logger.info(f"📊 Fetching latest history for {symbol}...")
                live_dp.get_latest_bars(symbol, n=1500, timeframe=Config.Data.RESOLUTION)
            
            # Use SimDataProvider for backtesting speed inside ShadowDarwin
            sim_dp = SimDataProvider(live_dp)
            darwin = ShadowDarwin(data_provider=sim_dp)
            
            for symbol in symbols:
                logger.info(f"🧪 [EVOLUTION DAEMON] Optimizing {symbol}...")
                
                # Run Optuna Bayesian TPE Optimizer
                results = darwin.run_epoch_optuna(symbol, n_trials=trials_per_epoch)
                
                best_fitness = results['best_fitness']
                logger.info(f"🏆 [EVOLUTION DAEMON] {symbol} Epoch Complete! Best Expected Fitness: {best_fitness:.4f}")
                
            logger.info(f"💤 [EVOLUTION DAEMON] Epoch finished. Sleeping for {cycle_delay_minutes} minutes...")
            time.sleep(cycle_delay_minutes * 60)
            
        except KeyboardInterrupt:
            logger.info("🛑 [EVOLUTION DAEMON] Terminated by User.")
            break
        except Exception as e:
            logger.error(f"❌ [EVOLUTION DAEMON] Fatal Error: {e}")
            time.sleep(60) # Wait 1 min before retrying if network failed

if __name__ == "__main__":
    run_evolution_daemon()
