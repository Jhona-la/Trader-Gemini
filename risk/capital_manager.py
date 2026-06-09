import logging
import asyncio
from enum import Enum
from typing import Dict, Any

logger = logging.getLogger(__name__)

class InjectionType(Enum):
    MICRO = "MICRO"            # < 25%
    STANDARD = "STANDARD"      # 25% - 100%
    MAJOR = "MAJOR"            # 100% - 500%
    MASSIVE = "MASSIVE"        # > 500%
    EMERGENCY = "EMERGENCY"    # Drawdown rescue

class CapitalPhase(Enum):
    INIT = "INIT"
    FREEZE = "FREEZE"          # Congelamiento primeros 60s
    RECALIBRATION = "RECALIBRATION" # Ajuste de parámetros
    DEPLOYMENT = "DEPLOYMENT"  # Despliegue escalonado

class CapitalManager:
    """
    Bloque VIII del Prompt Supremo.
    Manejo de Inyecciones de Capital y Despliegue Escalonado.
    """
    def __init__(self, risk_manager):
        self.risk_manager = risk_manager
        
        self.base_capital = 0.0
        self.total_capital = 0.0
        self.deployed_capital = 0.0
        self.reserved_capital = 0.0
        
        self.current_phase = CapitalPhase.INIT
        self.weeks_since_injection = 0
        self.active_injection_type = None

    def initialize_capital(self, initial_capital: float):
        self.base_capital = initial_capital
        self.total_capital = initial_capital
        self.deployed_capital = initial_capital
        self.reserved_capital = 0.0
        logger.info(f"💰 [CAPITAL] Capital Inicial fijado en ${initial_capital:.2f}")

    async def detect_injection(self, current_exchange_balance: float):
        """Llamado periódicamente para revisar si hay aportes externos."""
        # Se ignora el crecimiento orgánico (P&L).
        # Esto asume que el sistema sabe cuál debería ser el capital según P&L.
        # Si current_exchange_balance brinca repentinamente fuera de P&L, es inyección.
        pass
        
    async def process_manual_injection(self, new_total: float, is_emergency: bool = False):
        """Invocado por el usuario o interfaz cuando aporta capital."""
        if new_total <= self.total_capital: return
        
        injection_amount = new_total - self.total_capital
        ratio = injection_amount / self.total_capital
        
        # 1. Determinar Tipo
        if is_emergency:
            self.active_injection_type = InjectionType.EMERGENCY
        elif ratio < 0.25:
            self.active_injection_type = InjectionType.MICRO
        elif ratio <= 1.0:
            self.active_injection_type = InjectionType.STANDARD
        elif ratio <= 5.0:
            self.active_injection_type = InjectionType.MAJOR
        else:
            self.active_injection_type = InjectionType.MASSIVE
            
        logger.warning(f"🚨 [CAPITAL] INYECCIÓN DETECTADA: +${injection_amount:.2f} (+{ratio*100:.1f}%). Tipo: {self.active_injection_type.value}")
        
        self.total_capital = new_total
        self.weeks_since_injection = 0
        
        # 2. Fase de Congelamiento (Fase 1 y 2)
        self.current_phase = CapitalPhase.FREEZE
        logger.warning("❄️ [CAPITAL] FASE FREEZE: Suspendiendo nuevas entradas por 60s para recalibrar...")
        self.risk_manager.suspend_new_entries = True
        
        # Simular freeze asíncrono sin bloquear el loop de websockets
        asyncio.create_task(self._recalibration_workflow())

    async def _recalibration_workflow(self):
        await asyncio.sleep(60) # Wait 60s
        
        self.current_phase = CapitalPhase.RECALIBRATION
        logger.info("🔧 [CAPITAL] RECALIBRANDO PARÁMETROS...")
        
        # 3. Fase de Plan de Despliegue
        self._apply_deployment_plan()
        
        self.current_phase = CapitalPhase.DEPLOYMENT
        self.risk_manager.suspend_new_entries = False
        logger.info("✅ [CAPITAL] RECALIBRACIÓN FINALIZADA. Operaciones reanudadas con nuevo capital desplegable.")

    def _apply_deployment_plan(self):
        """Asigna qué % del nuevo capital se puede usar (deployed) y qué % queda en reserva."""
        
        if self.active_injection_type == InjectionType.MICRO:
            # Micro: 75% Semana 1
            deployment_pct = 0.75 if self.weeks_since_injection == 0 else 1.0
        elif self.active_injection_type == InjectionType.STANDARD:
            # Estándar: 30%, 55%, 80%, 100%
            schedule = [0.30, 0.55, 0.80, 1.0]
            idx = min(self.weeks_since_injection, 3)
            deployment_pct = schedule[idx]
        elif self.active_injection_type == InjectionType.MAJOR:
            # Mayor: 20%, 40%, 65%, 85%, 100%
            schedule = [0.20, 0.40, 0.65, 0.85, 1.0]
            idx = min(self.weeks_since_injection, 4)
            deployment_pct = schedule[idx]
        elif self.active_injection_type == InjectionType.MASSIVE:
            # Masivo: 10%, 25%, 45%, 70%, 100%
            schedule = [0.10, 0.25, 0.45, 0.70, 1.0]
            idx = min(self.weeks_since_injection, 4)
            deployment_pct = schedule[idx]
        elif self.active_injection_type == InjectionType.EMERGENCY:
            # Emergencia: 50% inmediato, 100% después (simplificado a weeks)
            deployment_pct = 0.50 if self.weeks_since_injection == 0 else 1.0
        else:
            deployment_pct = 1.0
            
        self.deployed_capital = self.base_capital + ((self.total_capital - self.base_capital) * deployment_pct)
        self.reserved_capital = self.total_capital - self.deployed_capital
        
        logger.info(f"📊 [CAPITAL DESPLIEGUE] Semana {self.weeks_since_injection+1}:")
        logger.info(f"   -> Desplegado (Operativo): ${self.deployed_capital:.2f} ({deployment_pct*100:.1f}% de la inyección)")
        logger.info(f"   -> Reservado Temporal: ${self.reserved_capital:.2f}")

    def advance_week(self):
        """Invocado por el TemporalSupervisor cada 7 ciclos (1 semana)."""
        if self.current_phase == CapitalPhase.DEPLOYMENT:
            self.weeks_since_injection += 1
            logger.info(f"📅 [CAPITAL] Avanzando despliegue de capital a Semana {self.weeks_since_injection+1}")
            self._apply_deployment_plan()
            
            if self.deployed_capital >= self.total_capital:
                logger.info("🎉 [CAPITAL] INYECCIÓN 100% INTEGRADA. Regresando a estado INIT.")
                self.base_capital = self.total_capital
                self.current_phase = CapitalPhase.INIT
                self.active_injection_type = None

    def get_operative_capital(self) -> float:
        """Retorna el capital que el RiskManager y Kelly pueden usar matemáticamente."""
        return self.deployed_capital
