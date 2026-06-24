# core/omniscient_registry.py
"""
Capa 2 — Registro Omnisciente y Sistema de No-Colisión

Núcleo de integridad absoluta (Mandato Supremo):
- Ninguna configuración, feature o parámetro puede solaparse, contradecirse ni sobrescribir a otro.
- Registro centralizado inmutable con dos categorías:
  - Valores fijos: límites inamovibles (solo por intervención humana). Tienen jerarquía absoluta.
  - Valores adaptativos: ajustables dinámicamente dentro de rangos autorizados.
- Contribuye al objetivo de +100% neto cada 3 días.
"""

import time
import logging
from config import Config
from typing import Dict, Any, Optional

logger = logging.getLogger("OmniscientRegistry")

class OmniscientRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OmniscientRegistry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.version = "1.0.0"
        # 1. Valores fijos (Inmutables)
        # Modificar estos valores arroja una excepción.
        self._fixed_values: Dict[str, Any] = {
            "GLOBAL_MAX_RISK_PER_TRADE": Config.MAX_RISK_PER_TRADE,
            "MICRO_MAX_HEAT": Config.MICROSCALPING_MARGIN_CAP,
            "SCALP_MAX_HEAT": Config.SCALPING_MARGIN_CAP,
            "SWING_MAX_HEAT": Config.SWING_MARGIN_CAP,
            "SYSTEM_CAPITAL_BASE": 13.0,
            "HARD_SL_REQUIRED": True,
            "PROHIBIT_AVERAGING_DOWN": True,
            "LIQUIDATION_BUFFER_PCT": 2.0,
            "EMERGENCY_DRAWDOWN_LIMIT_PCT": Config.Risk.MAX_DRAWDOWN * -1
        }

        # 2. Valores Adaptativos (Mutables dentro de rangos)
        # Se guardan con su historial de cambios para trazabilidad total.
        self._adaptive_values: Dict[str, Dict[str, Any]] = {
            "CURRENT_PHASE_LEVERAGE": {
                "value": 10,
                "min": 1,
                "max": 20,
                "history": []
            },
            "SCALPING_ALLOCATION_PCT": {
                "value": 0.50,
                "min": 0.20,
                "max": 0.80,
                "history": []
            },
            "MIN_SCORE_THRESHOLD": {
                "value": 60,
                "min": 40,
                "max": 90,
                "history": []
            }
        }

        # Registro de resoluciones de conflictos
        self._conflict_log = []

    # --- VALORES FIJOS ---
    def get_fixed(self, key: str) -> Any:
        if key not in self._fixed_values:
            raise KeyError(f"❌ Valor Fijo no encontrado: {key}")
        return self._fixed_values[key]

    def set_fixed(self, key: str, value: Any, override_token: str = ""):
        """
        Solo puede ser llamado mediante intervención humana directa con token de override.
        """
        if override_token != "MANUAL_OVERRIDE_V6":
            logger.error(f"❌ Intento no autorizado de sobrescribir Valor Fijo: {key}")
            raise PermissionError(f"Prohibición Absoluta: No se puede alterar el valor fijo '{key}' sin override humano.")
        
        logger.warning(f"⚠️ INTERVENCIÓN HUMANA: Valor Fijo '{key}' modificado de {self._fixed_values[key]} a {value}")
        self._fixed_values[key] = value

    # --- VALORES ADAPTATIVOS ---
    def get_adaptive(self, key: str) -> Any:
        if key not in self._adaptive_values:
            raise KeyError(f"❌ Valor Adaptativo no encontrado: {key}")
        return self._adaptive_values[key]["value"]

    def update_adaptive(self, key: str, new_value: float, reason: str, agent_id: str = "Meta-Aprendizaje"):
        """
        Actualiza un valor adaptativo, respetando los límites fijados y registrando la causa.
        """
        if key not in self._adaptive_values:
            raise KeyError(f"Valor adaptativo no existe: {key}")
            
        record = self._adaptive_values[key]
        min_val = record["min"]
        max_val = record["max"]
        
        if not (min_val <= new_value <= max_val):
            error_msg = f"Rechazo: {new_value} fuera de rango [{min_val}, {max_val}] para {key}"
            logger.warning(error_msg)
            # Log collision attempt
            self.log_conflict(
                conflict_type="ADAPTIVE_OUT_OF_BOUNDS",
                description=error_msg,
                resolution="Rechazado, se mantiene el valor actual."
            )
            return False

        old_value = record["value"]
        record["value"] = new_value
        record["history"].append({
            "timestamp": time.time(),
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason,
            "agent_id": agent_id
        })
        logger.info(f"🔄 Adaptativo '{key}' ajustado: {old_value} -> {new_value} | Razón: {reason}")
        return True

    # --- SISTEMA DE RESOLUCIÓN DE CONFLICTOS ---
    def log_conflict(self, conflict_type: str, description: str, resolution: str):
        """
        Registra un conflicto evitado por la Capa 7.
        """
        entry = {
            "timestamp": time.time(),
            "type": conflict_type,
            "description": description,
            "resolution": resolution
        }
        self._conflict_log.append(entry)
        logger.warning(f"⚔️ CONFLICTO EVITADO [{conflict_type}]: {description} -> {resolution}")

    def check_trade_validity(self, trade_risk_pct: float, current_portfolio_heat: float, has_sl: bool, horizon: str = "1m") -> bool:
        """
        Pre-Flight Checklist final: Verifica que la operación cumpla con todos los Axiomas Fijos.
        Ahora con validación vectorial por Horizonte.
        """
        # AXIOMA: Riesgo máximo por operación
        if trade_risk_pct > self.get_fixed("GLOBAL_MAX_RISK_PER_TRADE"):
            self.log_conflict(
                "RISK_VIOLATION",
                f"Trade risk {trade_risk_pct*100:.2f}% excede límite fijo {self.get_fixed('GLOBAL_MAX_RISK_PER_TRADE')*100:.2f}%",
                "Trade Bloqueado"
            )
            return False

        # AXIOMA: Portfolio heat Vectorizado
        if horizon == "SWING":
            max_heat_key = "SWING_MAX_HEAT"
        elif horizon == "MICROSCALPING":
            max_heat_key = "MICRO_MAX_HEAT"
        else:
            max_heat_key = "SCALP_MAX_HEAT"
            
        max_heat_limit = self.get_fixed(max_heat_key)
        
        # Adding a small epsilon 1e-6 to avoid floating point strictness blocks
        if current_portfolio_heat + trade_risk_pct > max_heat_limit + 1e-6:
            self.log_conflict(
                "HEAT_VIOLATION",
                f"Añadir trade excedería el Heat máximo para {horizon} ({(max_heat_limit)*100:.2f}%)",
                "Trade Bloqueado"
            )
            return False

        # AXIOMA: Stop Loss Obligatorio
        if self.get_fixed("HARD_SL_REQUIRED") and not has_sl:
            self.log_conflict(
                "NO_STOP_LOSS",
                "Señal de trade sin Hard Stop Loss detectada.",
                "Trade Bloqueado"
            )
            return False

        return True

# Singleton access
registry = OmniscientRegistry()
