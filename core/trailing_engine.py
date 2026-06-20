import logging
import time
from typing import Dict, Any, List, Optional
import numpy as np

class TrailingEngine:
    """
    MOTOR DE PERSECUCIÓN DINÁMICA DE GANANCIAS (V7)
    Diseñado para gestionar el ciclo de vida de posiciones basado puramente en ATRs y Fases.
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("TrailingEngine")
        
        # Fases
        self.FASE_0 = 'FASE_0_RIESGO_INICIAL'
        self.FASE_1 = 'FASE_1_CONFIRMACION'
        self.FASE_2 = 'FASE_2_R1'
        self.FASE_3 = 'FASE_3_EXTENSION'
        self.FASE_4 = 'FASE_4_RUNNER'

    def _get_asset_profile(self, symbol: str) -> dict:
        """Fetch asset-specific trailing parameters."""
        if hasattr(self.config, 'Trailing'):
            return self.config.Trailing.get_asset_profile(symbol)
        # Fallback si no está configurado
        return {"pullback_tol": 0.40, "trail_f1": 2.0, "trail_f2": 1.5, "trail_f3": 1.2, "trail_runner": 1.0}

    def _get_family_profile(self, family: str) -> dict:
        """Fetch strategy family parameters."""
        if hasattr(self.config, 'Trailing'):
            return self.config.Trailing.get_family_profile(family)
        return {"r1_pct": 0.50, "r2_pct": 0.30, "runner_pct": 0.20}

    def calculate_atr_movement(self, pos: dict, current_price: float, current_atr: float) -> float:
        """Calcula el movimiento actual de la posición en múltiplos de ATR."""
        if current_atr <= 0:
            # Fallback a un default si no hay ATR (ej. startup)
            profile = self._get_asset_profile(pos['symbol'])
            current_atr = current_price * profile['default_atr_pct']
            
        entry_price = pos['avg_price']
        if entry_price <= 0:
            return 0.0
            
        pos_side = pos['pos_side']
        
        if pos_side == 'LONG':
            pnl_atr = (current_price - entry_price) / current_atr
        else:
            pnl_atr = (entry_price - current_price) / current_atr
            
        return pnl_atr

    def update_position_state(self, pos: dict, pnl_atr: float):
        """Actualiza MFE y FASE de la posición en base al PnL en ATRs."""
        # 1. Update MFE
        current_mfe = pos['mfe_atr']
        if pnl_atr > current_mfe:
            pos['mfe_atr'] = pnl_atr
            
        # 2. Capture Ratio
        if pos['mfe_atr'] > 0:
            pos['ratio_captura'] = max(0.0, pnl_atr / pos['mfe_atr'])
        else:
            pos['ratio_captura'] = 0.0

        # 3. Phase Transitions
        fase = pos['fase_actual']
        
        # Fase 0 -> Fase 1: +0.5 ATR
        if fase == self.FASE_0 and pnl_atr >= 0.5:
            pos['fase_actual'] = self.FASE_1
            self.logger.info(f"🚀 [TRAILING] {pos['symbol']} entró a FASE 1 (Confirmación). PnL: {pnl_atr:.2f} ATR")
            
        # Fase 1 -> Fase 2 (R1): asumiendo R1 = 1.5 ATR (o dictado por Familia)
        elif fase == self.FASE_1 and pnl_atr >= 1.5:
            pos['fase_actual'] = self.FASE_2
            self.logger.info(f"🎯 [TRAILING] {pos['symbol']} alcanzó R1. Entró a FASE 2. PnL: {pnl_atr:.2f} ATR")
            
        # Fase 2 -> Fase 3 (Extensión): asumiendo R2 = 3.0 ATR
        elif fase == self.FASE_2 and pnl_atr >= 3.0:
            pos['fase_actual'] = self.FASE_3
            self.logger.info(f"🔥 [TRAILING] {pos['symbol']} alcanzó R2. Entró a FASE 3 (Extensión). PnL: {pnl_atr:.2f} ATR")
            
        # Fase 3 -> Fase 4 (Runner): MFE > 4.0 ATR
        elif fase == self.FASE_3 and pos['mfe_atr'] >= 4.0:
            pos['fase_actual'] = self.FASE_4
            self.logger.info(f"🏃 [TRAILING] {pos['symbol']} es un RUNNER. FASE 4 Activa. MFE: {pos['mfe_atr']:.2f} ATR")

    def evaluate_trailing_mechanisms(self, pos: dict, current_price: float, current_atr: float, data_pkg: dict = None) -> dict:
        """
        Evalúa los 8 mecanismos de Trailing en paralelo.
        Devuelve el precio del Trailing Stop sugerido y un flag si se debe cerrar YA.
        """
        if current_atr <= 0:
            return {'stop_price': None, 'force_close': False, 'reason': None}

        symbol = pos['symbol']
        pos_side = pos['pos_side']
        fase = pos['fase_actual']
        pnl_atr = self.calculate_atr_movement(pos, current_price, current_atr)
        
        self.update_position_state(pos, pnl_atr)
        fase = pos['fase_actual'] # Re-leer por si cambió
        mfe_atr = pos['mfe_atr']
        
        profile = self._get_asset_profile(symbol)
        
        # Stop Proposals de cada mecanismo
        proposals = []
        
        # T1: ATR Step Trailing
        if fase != self.FASE_0:
            # Obtener distancia según la fase
            if fase == self.FASE_1:
                dist_atr = profile['trail_f1']
            elif fase == self.FASE_2:
                dist_atr = profile['trail_f2']
            elif fase == self.FASE_3:
                dist_atr = profile['trail_f3']
            elif fase == self.FASE_4:
                dist_atr = profile['trail_runner']
            else:
                dist_atr = 2.0
                
            # Calcular stop en T1
            if pos_side == 'LONG':
                t1_stop = current_price - (dist_atr * current_atr)
            else:
                t1_stop = current_price + (dist_atr * current_atr)
            
            # 🛡️ ESCUDO CUÁNTICO: Breakeven Inmediato a +1.0% de PnL Absoluto
            entry = pos['avg_price']
            fee_rate = getattr(self.config, 'BINANCE_TAKER_FEE_BNB', 0.000375) * 2
            
            pnl_pct = 0.0
            if entry > 0:
                pnl_pct = (current_price - entry) / entry if pos_side == 'LONG' else (entry - current_price) / entry

            # Guardamos el Max PnL Pct (Absoluto)
            max_pnl_pct = pos['max_pnl_pct']
            if pnl_pct > max_pnl_pct:
                pos['max_pnl_pct'] = pnl_pct
                max_pnl_pct = pnl_pct

            # Si alguna vez tocamos +1.0% de ganancia, el stop NUNCA puede ser peor que Break-Even + Comisión
            if max_pnl_pct >= 0.01:
                if pos_side == 'LONG':
                    breakeven_price = entry * (1 + fee_rate)
                    t1_stop = max(t1_stop, breakeven_price)
                    # Si superamos el 1.5%, bloqueamos al menos un 0.5% de profit puro
                    if max_pnl_pct >= 0.015:
                        profit_lock = entry * (1 + 0.005)
                        t1_stop = max(t1_stop, profit_lock)
                else:
                    breakeven_price = entry * (1 - fee_rate)
                    t1_stop = min(t1_stop, breakeven_price) if t1_stop > 0 else breakeven_price
                    if max_pnl_pct >= 0.015:
                        profit_lock = entry * (1 - 0.005)
                        t1_stop = min(t1_stop, profit_lock) if t1_stop > 0 else profit_lock
            
            proposals.append({'name': 'T1_ATR', 'price': t1_stop})
            
        # T3: Parabolic Trailing (Aceleración)
        if mfe_atr >= 3.0:
            # Factor crece con el MFE
            factor = min(0.20, 0.02 + (mfe_atr - 3.0) * 0.05)
            dist_parabolic = max(0.5, profile['trail_f3'] - (mfe_atr * factor))
            
            if pos_side == 'LONG':
                t3_stop = current_price - (dist_parabolic * current_atr)
            else:
                t3_stop = current_price + (dist_parabolic * current_atr)
                
            proposals.append({'name': 'T3_Parabolic', 'price': t3_stop})
            
        # T5: Volatility Contraction Trailing
        if fase != self.FASE_0:
            # Si el mercado se comprime, apretar el stop
            k_factor = 1.5
            dist_vol = k_factor * current_atr
            if pos_side == 'LONG':
                t5_stop = current_price - dist_vol
            else:
                t5_stop = current_price + dist_vol
            proposals.append({'name': 'T5_Volatility', 'price': t5_stop})

        # Seleccionar el Trailing Stop más apretado (El que esté más cerca del precio)
        best_stop = None
        best_name = None
        
        if pos_side == 'LONG':
            for p in proposals:
                if best_stop is None or p['price'] > best_stop:
                    best_stop = p['price']
                    best_name = p['name']
        else:
            for p in proposals:
                if best_stop is None or p['price'] < best_stop:
                    best_stop = p['price']
                    best_name = p['name']
                    
        # Check T6 Exhaustion si data_pkg fue provisto (RSI, Vol, CVD)
        force_close = False
        reason = None
        
        if data_pkg:
            rsi = data_pkg['rsi']
            # Detección simple de extremo
            if pos_side == 'LONG' and rsi > 75 and mfe_atr > 1.5:
                # Apretar brutalmente (T6)
                t6_dist = 0.5 * current_atr
                t6_stop = current_price - t6_dist
                if best_stop is None or t6_stop > best_stop:
                    best_stop = t6_stop
                    best_name = 'T6_Exhaustion(RSI)'
                    
            elif pos_side == 'SHORT' and rsi < 25 and mfe_atr > 1.5:
                t6_dist = 0.5 * current_atr
                t6_stop = current_price + t6_dist
                if best_stop is None or t6_stop < best_stop:
                    best_stop = t6_stop
                    best_name = 'T6_Exhaustion(RSI)'

        # Guardar el stop propuesto en la posición (Solo sube en Longs, solo baja en Shorts)
        current_trail = pos['trail_stop_price']
        
        if best_stop is not None:
            if pos_side == 'LONG':
                if current_trail <= 1e-9 or best_stop > current_trail:
                    pos['trail_stop_price'] = best_stop
            else:
                if current_trail <= 1e-9 or best_stop < current_trail:
                    pos['trail_stop_price'] = best_stop
                    
        # Force Close Check (Drawdown desde MFE_ATR supera tolerancia)
        dd_atr = mfe_atr - pnl_atr
        tol = profile['pullback_tol']
        
        # En Fases altas, somos más estrictos
        if fase in [self.FASE_3, self.FASE_4]:
            tol = tol * 0.8 # 20% menos de tolerancia
            
        if mfe_atr > 1.0 and dd_atr > tol:
            force_close = True
            reason = f'Pullback_Tolerancia_Excedida (MFE:{mfe_atr:.1f} ATR, DD:{dd_atr:.1f} ATR)'

        return {
            'stop_price': pos['trail_stop_price'],
            'force_close': force_close,
            'reason': reason,
            'active_mechanism': best_name
        }
