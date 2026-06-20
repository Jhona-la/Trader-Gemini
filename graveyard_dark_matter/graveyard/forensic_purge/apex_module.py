import logging
import numpy as np
from datetime import datetime

try:
    from numba import jit
except ImportError:
    # Fallback to no-op decorator if Numba is missing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger("ApexEngine")

@jit(nopython=True, cache=True)
def fast_ths(pnl: float) -> float:
    base_health = 50.0
    health = base_health + (pnl * 10.0)
    return max(0.0, min(100.0, health))

@jit(nopython=True, cache=True)
def fast_vs(pnl: float, duration_minutes: float) -> float:
    if duration_minutes < 1.0:
        return 50.0
    pnl_per_minute = pnl / duration_minutes
    vs = pnl_per_minute * 1000.0
    return max(-100.0, min(100.0, vs))

@jit(nopython=True, cache=True)
def fast_ces(pnl: float, duration_minutes: float) -> float:
    if pnl < 0.0 and duration_minutes > 15.0:
        return 10.0
    elif pnl > 0.5:
        return 90.0
    return max(0.0, 100.0 - (duration_minutes * 0.5))

@jit(nopython=True, cache=True)
def fast_zs(ths: float, vs: float, ces: float, duration_minutes: float) -> float:
    z1 = 1.0 if (duration_minutes > 45.0 and vs < 0.0) else 0.0
    z2 = 1.0 if ths < 30.0 else 0.0
    z3 = 1.0 if ces < 20.0 else 0.0
    return (z1 + z2 + z3) / 3.0


class ApexEngine:
    """
    MÓDULO APEX (Motor Dinámico de Salidas y Eficiencia de Capital)
    Evalúa la viabilidad de una posición activa en tiempo real.
    Reemplaza cierres ingenuos basados en tiempo con cierres basados en mérito.
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.evaluation_history = {} # symbol -> list of THS evaluations

    def _calculate_ths(self, position, current_market_data):
        return fast_ths(position['unrealized_pnl'])

    def _calculate_vs(self, position, current_market_data, now=None):
        current_time = now if now else datetime.utcnow()
        if current_time.tzinfo is None and position['entry_time'].tzinfo is not None:
            current_time = current_time.replace(tzinfo=position['entry_time'].tzinfo)
        duration_minutes = (current_time - position['entry_time']).total_seconds() / 60.0
        return fast_vs(position['unrealized_pnl'], duration_minutes)

    def _calculate_ces(self, position, now=None):
        current_time = now if now else datetime.utcnow()
        if current_time.tzinfo is None and position['entry_time'].tzinfo is not None:
            current_time = current_time.replace(tzinfo=position['entry_time'].tzinfo)
        duration_minutes = (current_time - position['entry_time']).total_seconds() / 60.0
        return fast_ces(position['unrealized_pnl'], duration_minutes)

    def _calculate_zs(self, ths, vs, ces, duration_minutes):
        return fast_zs(float(ths), float(vs), float(ces), float(duration_minutes))

    def evaluate_position(self, position, current_market_data, now=None):
        """
        Evalúa una posición activa y determina la acción de salida.
        Retorna un dict con la acción ('HOLD', 'CLOSE_ZOMBIE', 'HERO_UPGRADE', 'SC2_REDUCTION')
        """
        sym = position['symbol']
        ths = self._calculate_ths(position, current_market_data)
        vs = self._calculate_vs(position, current_market_data, now)
        ces = self._calculate_ces(position, now)
        
        # PVC = (THS × 0.50) + (max(VS,0) × 0.30) + (CES × 0.20)
        pvc = (ths * 0.50) + (max(vs, 0) * 0.30) + (ces * 0.20)
        
        current_time = now if now else datetime.utcnow()
        if current_time.tzinfo is None and position['entry_time'].tzinfo is not None:
            current_time = current_time.replace(tzinfo=position['entry_time'].tzinfo)
        duration_minutes = (current_time - position['entry_time']).total_seconds() / 60.0
        zs = self._calculate_zs(ths, vs, ces, duration_minutes)
        
        # Track THS for SC2 Condition
        if sym not in self.evaluation_history:
            self.evaluation_history[sym] = []
        self.evaluation_history[sym].append(ths)
        if len(self.evaluation_history[sym]) > 3:
            self.evaluation_history[sym].pop(0)

        action = 'HOLD'
        
        # Decision Logic (APEX Protocol)
        if zs > 0.70:
            logger.info(f"🧟 [APEX ZOMBIE] {sym} is a Zombie (ZS: {zs:.2f}, PVC: {pvc:.2f}). KILLING.")
            action = 'CLOSE_ZOMBIE'
        elif len(self.evaluation_history[sym]) == 3 and all(t < 35 for t in self.evaluation_history[sym]):
            logger.info(f"⚠️ [APEX SC2] {sym} THS < 35 for 3 consecutive ticks. Engaging SC2 Reduction.")
            action = 'SC2_REDUCTION'
        elif pvc > 55 and position.get('horizon') == 'SCALPING' and duration_minutes > 15:
            logger.info(f"🦸‍♂️ [APEX HERO] {sym} PVC is {pvc:.2f}. Upgrading to HERO PROTOCOL (Swing Transformation).")
            action = 'HERO_UPGRADE'
            
        return {
            'action': action,
            'pvc': pvc,
            'ths': ths,
            'vs': vs,
            'ces': ces,
            'zs': zs
        }
