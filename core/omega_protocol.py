"""
Protocolo Omega (Fase 9): Capa de Volatilidad Secundaria
========================================================
El Protocolo Omega es la autoridad suprema (God Mode) del sistema durante eventos
de cisne negro (Black Swan) o pánico generalizado.

Monitorea el Order Flow Imbalance (OFI) global y las métricas de Swarm Intelligence.
Si detecta una anomalía macro-sistémica (ej. Cascada de liquidaciones cruzadas),
el Protocolo Omega:
1. Revoca el control a las estrategias ML / Swing convencionales.
2. Cancela órdenes límite expuestas.
3. Activa el "Modo Phalanx" (Escudo defensivo) o "Modo Sniper" (Ataque HFT de oportunidad).
"""

import logging
import time
from enum import Enum
from typing import Dict, Any

from aits_research.swarm_intelligence import SwarmIntelligence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class OmegaState(Enum):
    DORMANT = "DORMANT"       # Normal market conditions. Engine runs ML/Technical.
    PHALANX = "PHALANX"       # Defensive posture: cancel open orders, pause new entries.
    SNIPER = "SNIPER"         # Offensive HFT posture: look for extreme mispricings.

class OmegaProtocolManager:
    """
    Director Ejecutivo Final. Inyectado en el Core Engine.
    """
    def __init__(self):
        self.state = OmegaState.DORMANT
        self.swarm_intel = SwarmIntelligence()
        self.last_state_change = 0.0
        self.cooldown_period = 300 # 5 minutes minimum in Omega state
        
        # Simulated global metrics for POC
        self.global_ofi = 0.0
        self.liquidation_cascade_detected = False

    def assess_ecosystem_health(self, active_symbol: str, ofi_metric: float, liquidations_usd: float) -> OmegaState:
        """
        Evaluates real-time order flow and liquidations to determine if Omega Protocol should activate.
        Returns the new or current OmegaState.
        """
        current_time = time.time()
        
        # Maintain state if within cooldown
        if self.state != OmegaState.DORMANT and (current_time - self.last_state_change) < self.cooldown_period:
            return self.state

        # Update global OFI moving average (simplistic for POC)
        self.global_ofi = (self.global_ofi * 0.8) + (ofi_metric * 0.2)
        
        # Check for extreme panic (OFI deeply negative + massive liquidations)
        if self.global_ofi < -10.0 or liquidations_usd > 5000000.0:
            if not self.liquidation_cascade_detected:
                logging.critical(f"🌌 [PROTOCOL OMEGA] BLACK SWAN DETECTED. Liquidation Cascade ({liquidations_usd} USD) + OFI ({self.global_ofi:.1f}).")
                self.liquidation_cascade_detected = True
                self._engage_phalanx(active_symbol)
        else:
            # Check for hyper-oversold rebound opportunity (Sniper)
            if self.liquidation_cascade_detected and self.global_ofi > -2.0:
                 logging.critical(f"🌌 [PROTOCOL OMEGA] PANIC SUBSIDING. Engaging SNIPER MODE to exploit mispricing.")
                 self._engage_sniper()
            elif self.liquidation_cascade_detected and (current_time - self.last_state_change) >= self.cooldown_period:
                 logging.critical(f"🌌 [PROTOCOL OMEGA] Volatility normalizing. Returning to DORMANT.")
                 self._stand_down()

        return self.state

    def _engage_phalanx(self, trigger_symbol: str):
        """Activates defensive mode."""
        self.state = OmegaState.PHALANX
        self.last_state_change = time.time()
        logging.error("🛡️ [OMEGA PHALANX] Revoking ML control. Canceling all exposed LIMIT orders.")
        # Broadcast shock to Swarm
        self.swarm_intel.register_ecosystem_shock(trigger_symbol, -15.0)

    def _engage_sniper(self):
        """Activates offensive mode."""
        self.state = OmegaState.SNIPER
        self.last_state_change = time.time()
        logging.warning("🎯 [OMEGA SNIPER] Authorizing HFT Scalping sub-systems. Max hold time: 3 seconds.")

    def _stand_down(self):
        """Returns to normal operations."""
        self.state = OmegaState.DORMANT
        self.liquidation_cascade_detected = False
        self.last_state_change = time.time()
        logging.info("🟩 [OMEGA] Protocol deactivated. Returning control to Core Engine.")

    def should_veto_signal(self, signal: dict) -> bool:
        """
        Called by the Engine before executing any ML/Technical signal.
        """
        if self.state == OmegaState.PHALANX:
            logging.warning(f"🚫 [OMEGA VETO] Signal {signal.get('symbol')} blocked by PHALANX mode.")
            return True
        elif self.state == OmegaState.SNIPER and signal.get('horizon') != 'SCALPING':
            logging.warning(f"🚫 [OMEGA VETO] Swing signal {signal.get('symbol')} blocked. Only SCALPING allowed in SNIPER mode.")
            return True
        return False

if __name__ == "__main__":
    omega = OmegaProtocolManager()
    
    # 1. Normal market
    logging.info("--- Tick 1: Normal Market ---")
    omega.assess_ecosystem_health("BTC", 1.0, 50000.0)
    assert not omega.should_veto_signal({"symbol": "BTC", "horizon": "SWING"})
    
    # 2. Crash starts
    logging.info("\n--- Tick 2: Black Swan Starts ---")
    omega.assess_ecosystem_health("ETH", -12.0, 10000000.0)
    assert omega.state == OmegaState.PHALANX
    assert omega.should_veto_signal({"symbol": "SOL", "horizon": "SCALPING"})
    
    # 3. Fast forward past cooldown, panic subsiding
    logging.info("\n--- Tick 3: Panic Subsiding (Sniper Mode) ---")
    omega.last_state_change = time.time() - 400 # Simulate time passing
    omega.assess_ecosystem_health("ETH", 5.0, 10000.0) # Send positive OFI to pull global OFI > -2.0
    assert omega.state == OmegaState.SNIPER
    assert omega.should_veto_signal({"symbol": "SOL", "horizon": "SWING"}) # Swing vetoed
    assert not omega.should_veto_signal({"symbol": "SOL", "horizon": "SCALPING"}) # Scalping allowed
