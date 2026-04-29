import os
import sys
import time
import subprocess
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import logger
from utils.notifier import Notifier
from config import Config

class AutoCICDDaemon:
    """
    🤖 AUTO-CICD: Continuous Evolution Engine
    ===================================================
    QUÉ: Demonio que ejecuta backtesting y mutación autónoma.
    POR QUÉ: Para escalar el capital de $13 USD iterativamente sin intervención.
    PARA QUÉ: Detectar degradación de estrategias y evolucionar parámetros.
    CÓMO: Ejecuta run_god_mode_backtest.py y evalúa PnL/WR periódicamente.
    """
    
    def __init__(self):
        self.interval_seconds = 4 * 3600  # 4 hours
        self.notifier = Notifier()
        self.running = False
        self.project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.backtest_script = self.project_root / "scripts" / "run_god_mode_backtest.py"
        self.tuner_script = self.project_root / "scripts" / "run_optuna_oracle.py"
        
        self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [CICD] %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.project_root / "logs" / "auto_cicd.log")
            ]
        )
        self.log = logging.getLogger("CICD")

    async def alert(self, message: str):
        self.log.info(message)
        if Config.TELEGRAM_ENABLED:
            await self.notifier.send_telegram(f"🤖 *AUTO-CICD EVOLUTION*\n\n{message}")

    def run_backtest(self) -> bool:
        """Executes God Mode Backtest and checks for zero errors."""
        self.log.info("🚀 Initiating God Mode Backtest...")
        
        env = os.environ.copy()
        env['TRADER_GEMINI_ENV'] = 'TESTING'
        
        try:
            # We run it as a subprocess to prevent memory leaks in the main daemon
            result = subprocess.run(
                [sys.executable, str(self.backtest_script)],
                capture_output=True,
                text=True,
                env=env,
                timeout=3600 # 1 hour max
            )
            
            if result.returncode == 0:
                self.log.info("✅ Backtest completed successfully.")
                return True
            else:
                self.log.error(f"❌ Backtest failed with code {result.returncode}")
                self.log.error(f"Stderr: {result.stderr[-1000:]}") # Show last 1000 chars
                return False
                
        except subprocess.TimeoutExpired:
            self.log.error("⏳ Backtest timed out after 1 hour.")
            return False
        except Exception as e:
            self.log.error(f"🔥 Critical error running backtest: {e}")
            return False

    def run_mutation(self) -> bool:
        """Executes the Meta-Optimizer (Optuna) to evolve strategies."""
        self.log.info("🧬 Initiating Strategy Mutation (Optuna)...")
        
        env = os.environ.copy()
        try:
            result = subprocess.run(
                [sys.executable, str(self.tuner_script)],
                capture_output=True,
                text=True,
                env=env,
                timeout=7200 # 2 hours max
            )
            
            if result.returncode == 0:
                self.log.info("✅ Mutation completed successfully.")
                return True
            else:
                self.log.error(f"❌ Mutation failed: {result.returncode}")
                return False
        except Exception as e:
            self.log.error(f"🔥 Critical error during mutation: {e}")
            return False

    async def execute_cycle(self):
        """Executes one full CICD cycle."""
        await self.alert("🔄 Iniciando Ciclo Auto-CICD (Backtest + Validación)")
        
        success = self.run_backtest()
        
        if success:
            await self.alert("✅ God Mode Backtest Exitoso.\n\n📊 *Estado*: Sin degradación detectada.\n⏳ Esperando próximo ciclo.")
        else:
            await self.alert("⚠️ *DEGRADACIÓN DETECTADA*\nEl backtest falló o el WR es bajo.\n🧬 Iniciando Mutación Genética...")
            mutated = self.run_mutation()
            if mutated:
                await self.alert("🧬 *MUTACIÓN COMPLETADA*\nNuevos parámetros inyectados en producción. Reiniciando validación...")
                self.run_backtest() # Verify again
            else:
                await self.alert("🚨 *FALLA CRÍTICA EN MUTACIÓN*\nRequiere intervención humana inmediata.")

    async def start(self):
        self.running = True
        await self.alert("🚀 *AUTO-CICD ENGINE INICIADO*\nVigilancia de estrategias activa 24/7.")
        
        while self.running:
            try:
                await self.execute_cycle()
            except Exception as e:
                self.log.error(f"Error in CICD loop: {e}")
            
            # Sleep for interval
            self.log.info(f"💤 Sleeping for {self.interval_seconds} seconds...")
            await asyncio.sleep(self.interval_seconds)

def main():
    daemon = AutoCICDDaemon()
    try:
        asyncio.run(daemon.start())
    except KeyboardInterrupt:
        print("\n🛑 Auto-CICD Daemon stopped.")

if __name__ == "__main__":
    main()
