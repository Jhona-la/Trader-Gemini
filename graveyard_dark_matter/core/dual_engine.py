import logging
import asyncio
from typing import Dict, Any

from core.event_bus import event_bus
from core.events import MarketEvent, SignalEvent, OrderEvent
from config.adaptive_config import adaptive_config
from core.portfolio import portfolio
from core.strategy_registry import UniversalStrategyRegistry

logger = logging.getLogger("DualEngine")

class MotorHorizonte:
    """
    AXIOMA: MÓDULO HORIZON
    Motor de ejecución dedicado a un horizonte específico (SCALP o SWING).
    """
    def __init__(self, name: str, capital_pct: float, max_concurrent: int, sip_interval: int):
        self.name = name
        self.capital_pct = capital_pct
        self.max_concurrent = max_concurrent
        self.sip_interval = sip_interval
        self.active_positions = 0
        
        # Suscribirse a las señales de su propio horizonte
        event_bus.subscribe(SignalEvent, self.on_signal)
        
    async def run_sip_loop(self):
        """Systematic Internalizer Protocol loop"""
        while True:
            await asyncio.sleep(self.sip_interval)
            logger.debug(f"[{self.name}] Ejecutando SIP (Evaluación de posiciones...)")
            # Logic para evaluar trailing stops, funding, auto-upgrades
            
    def on_signal(self, event: SignalEvent):
        # Ignorar señales que no son de este horizonte
        if getattr(event, 'horizon', 'SCALP') != self.name:
            return
            
        logger.info(f"[{self.name}] Signal recibida: {event.strategy_id} {event.symbol} {event.direction}")
        
        # Regla SM-1: Los motores no comparten capital
        allocated_capital = portfolio.get_available_capital() * self.capital_pct
        
        # Validar max concurrent
        if self.active_positions >= self.max_concurrent:
            logger.warning(f"[{self.name}] Ignorando señal: Máximo de posiciones concurrentes alcanzado.")
            return
            
        # Simular ruteo al OrderManager
        # El Non-Collision Engine procesaría antes de colocar la orden
        self.active_positions += 1
        logger.info(f"[{self.name}] Orden colocada exitosamente. Activas: {self.active_positions}")
        
class DualEngineOrchestrator:
    """
    Orquestador maestro que corre SCALP y SWING en paralelo sin interferencia.
    """
    def __init__(self):
        # Extraer configuración del AdaptiveConfig
        scalp_conf = adaptive_config.matrix['SCALP']['global_horizon']
        swing_conf = adaptive_config.matrix['SWING']['global_horizon']
        
        self.motor_scalp = MotorHorizonte(
            name="SCALP",
            capital_pct=scalp_conf['capital_allocation_base_pct'],
            max_concurrent=scalp_conf['max_concurrent_positions'],
            sip_interval=scalp_conf['evaluation_frequency_seconds']
        )
        
        self.motor_swing = MotorHorizonte(
            name="SWING",
            capital_pct=swing_conf['capital_allocation_base_pct'],
            max_concurrent=swing_conf['max_concurrent_positions'],
            sip_interval=swing_conf['evaluation_frequency_seconds']
        )
        
    async def start_all(self):
        logger.info("[DualEngine] Iniciando Motores Paralelos (SCALP & SWING)...")
        await asyncio.gather(
            self.motor_scalp.run_sip_loop(),
            self.motor_swing.run_sip_loop()
        )
