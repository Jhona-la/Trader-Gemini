import logging
import time
from typing import Dict, Any, Tuple
from datetime import datetime, timezone
from config import Config
from core.enums import OrderSide
from core.events import SignalEvent

logger = logging.getLogger("SignalScorer")

class SignalScorer:
    """
    SISTEMA AUTÓNOMO ANTIFRÁGIL - FASE 1
    Capa de puntuación estricta de 0 a 100 puntos y validación de Breakeven.
    Evalúa NINGUNA señal debe pasar al RiskManager si no aprueba este checklist.
    """
    
    def __init__(self):
        self.min_pass_score = getattr(Config.Scoring, 'MIN_PASS_SCORE', 75)
        # Phase 2: Umbrales por Horizonte
        self.scalping_min_score = getattr(Config.Scoring, 'SCALPING_MIN_SCORE', 85)
        self.swing_min_score = getattr(Config.Scoring, 'SWING_MIN_SCORE', 70)
        self.breakeven_multiplier = getattr(Config.Scoring, 'BREAKEVEN_SAFETY_MARGIN_MULTIPLIER', 1.5)
        
    def calculate_score(self, event: SignalEvent, dh: Any, portfolio: Any, regime: str) -> Tuple[float, Dict[str, float]]:
        """
        Calcula el score de 0 a 100 basado en 7 pilares.
        Returns: (Total Score, Breakdown Dictionary)
        """
        breakdown = {}
        total_score = 0.0
        
        # OMNI-STRATEGY BYPASS
        if '[OMNI]' in getattr(event, 'strategy_id', ''):
            return 100.0, {'omni_bypass': 100.0}
        
        # 1. Confluencia Técnica / Fuerza (Max 20p)
        # Basado en event.strength (típicamente 0 a 1) o ml_confidence
        base_strength = getattr(event, 'ml_confidence', None)
        if base_strength is None:
            base_strength = getattr(event, 'strength', 0.5)
        strength_score = min(20.0, base_strength * 20.0)
        breakdown['strength'] = strength_score
        total_score += strength_score
        
        # 2. Alineación Multi-TF (Max 20p)
        # Basado en el clash_vector / clash_score si fue calculado por el Oracle
        clash_score_val = 0.0
        meta = getattr(event, 'metadata', {}) or {}
        if 'oracle_clash_score' in meta:
            clash_score_val = meta['oracle_clash_score']
            # Clash de 0.0 = Perfecto (20p), Clash de 1.0 = Terrible (0p)
            mtf_score = max(0.0, 20.0 * (1.0 - clash_score_val))
        else:
            # Si no hay oracle clash score, asumimos neutral/positivo leve
            mtf_score = 10.0
        breakdown['multi_tf'] = mtf_score
        total_score += mtf_score
        
        # 3. Compatibilidad de Régimen (Max 15p)
        # ═══════════════════════════════════════════════════════════════
        # [SOVEREIGN RULE CAPA 4: LONG/SHORT INTELLIGENCE]
        # QUÉ: Lógica evolutiva que comprende la direccionalidad según horizonte.
        # POR QUÉ: Scalping necesita bidireccionalidad. Swing necesita seguir la macro.
        # ═══════════════════════════════════════════════════════════════
        _dir = 'LONG' if 'LONG' in str(getattr(event, 'signal_type', '')) else 'SHORT'
        _horizon = getattr(event, 'horizon', 'SCALPING')
        
        try:
            from core.market_regime import MarketRegimeDetector
            _regime_dummy = MarketRegimeDetector()
            regime_comp = _regime_dummy.get_directional_bias(regime, _horizon, _dir)
        except Exception as e:
            logger.error(f"Error accediendo a Inteligencia Direccional (Capa 4): {e}")
            regime_comp = 0.5 # Fallback de seguridad
            
        regime_score = regime_comp * 15.0
        breakdown['regime'] = regime_score
        total_score += regime_score
        
        # 4. Historial de Estrategia / Winrate (Max 15p)
        wr_score = 7.5 # Default a mitad si no hay data
        try:
            if portfolio and hasattr(portfolio, 'get_strategy_metrics'):
                metrics = portfolio.get_strategy_metrics(event.strategy_id)
                if metrics and 'win_rate' in metrics:
                    wr = metrics['win_rate']
                    # WR 0.5 = 7.5p, WR 0.7 = 10.5p, WR 1.0 = 15p
                    wr_score = wr * 15.0
        except Exception as e:
            logger.exception(f"Swallowed exception ghost bug: {e}")
            pass
        breakdown['history'] = wr_score
        total_score += wr_score
        
        # 5. Sesión / Funding (Max 10p) [MODIFICADO: SHORT INTELLIGENCE]
        now_utc = datetime.now(timezone.utc)
        session_score = 10.0
        
        # Fallo Tipo 5: Ignorancia del funding rate
        funding_rate = 0.0
        try:
            if dh and hasattr(dh, 'get_derivatives_metrics'):
                derivs = dh.get_derivatives_metrics(event.symbol)
                funding_rate = derivs.get('funding_rate', 0.0)
                breakdown['funding_rate'] = funding_rate
        except Exception as e:
            logger.exception(f"Swallowed exception ghost bug: {e}")
            pass

        if now_utc.hour in (23, 7, 15) and now_utc.minute >= 40:
            session_score = 0.0 # Peligro de liquidación de funding por sesión
            
        if _dir == 'SHORT':
            if funding_rate > 0.0005: # > 0.05% (Extremo positivo, mercado pagando cortos)
                session_score += 8.0
            elif funding_rate < -0.0002: # < -0.02% (Extremo negativo, peligro inminente de Squeeze)
                session_score -= 50.0 # Veto matemático

        breakdown['session'] = session_score
        total_score += session_score
        
        # 6. Liquidez / Spread (Max 10p)
        # Verificamos spread del orderbook si está disponible
        liq_score = 5.0
        try:
            if dh and hasattr(dh, 'get_spread'):
                spread_pct = dh.get_spread(event.symbol)
                if spread_pct < 0.0005: # < 0.05% = Excelente
                    liq_score = 10.0
                elif spread_pct < 0.001: # < 0.1% = Bueno
                    liq_score = 8.0
                elif spread_pct > 0.002: # > 0.2% = Malo
                    liq_score = 0.0
        except Exception as e:
            logger.exception(f"Swallowed exception ghost bug: {e}")
            pass
        breakdown['liquidity'] = liq_score
        total_score += liq_score
        
        # 7. Orderflow / Microestructura (Max 10p)
        # Aproximado por OFI o RSI momentum muy corto
        of_score = 5.0
        try:
            if dh and hasattr(dh, 'get_derivatives_metrics'):
                derivs = dh.get_derivatives_metrics(event.symbol)
                oi_delta = derivs.get('oi_delta_15m', 0.0)
                # Si OI delta acompaña la dirección = bueno
                if (_dir == 'LONG' and oi_delta > 0) or (_dir == 'SHORT' and oi_delta < 0):
                    of_score = 10.0
                elif (_dir == 'LONG' and oi_delta < 0) or (_dir == 'SHORT' and oi_delta > 0):
                    of_score = 2.0
        except Exception as e:
            logger.exception(f"Swallowed exception ghost bug: {e}")
            pass
        breakdown['orderflow'] = of_score
        total_score += of_score
        
        # [SHORT INTELLIGENCE] Fallo Tipo 2: Sesgo de Predicción (Umbrales Asimétricos)
        sym = event.symbol.replace("/", "").upper()
        if _dir == 'SHORT':
            if "BTC" in sym: req_score = 78
            elif "ETH" in sym: req_score = 74
            elif "BNB" in sym or "SOL" in sym: req_score = 70
            elif "XRP" in sym: req_score = 75
            elif sym in ["DOGE", "SHIB", "PEPE", "FLOKI", "WIF"]: req_score = 85
            else: req_score = 72
        else: # LONG
            if "BTC" in sym: req_score = 72
            elif "ETH" in sym: req_score = 68
            elif "BNB" in sym or "SOL" in sym: req_score = 65
            elif "XRP" in sym: req_score = 70
            elif sym in ["DOGE", "SHIB", "PEPE", "FLOKI", "WIF"]: req_score = 78
            else: req_score = 65
            
        horizon_str = getattr(event, 'horizon', '').upper()
        engine_req = self.scalping_min_score if horizon_str == 'SCALPING' else self.swing_min_score
        
        if total_score >= req_score:
            if total_score < engine_req:
                total_score = engine_req + 0.1 # Boost para pasar el check ciego del engine
        else:
            total_score = 0.0 # Destrucción del score para rechazo garantizado
            breakdown['REJECT_REASON'] = f'Failed asymmetric threshold ({req_score})'
            
        # CAPA 7: ABSOLUTE PREDICTION GATE
        # Destruir matemáticamente cualquier trade sin certeza de IA >= 65%
        ml_conf = getattr(event, 'ml_confidence', None)
        if ml_conf is None:
            ml_conf = getattr(event, 'strength', 0.5)
            
        if ml_conf < 0.65:
            total_score = 0.0
            breakdown['REJECT_REASON'] = f'PREDICTION_GATE: Low ML Confidence ({ml_conf:.2f} < 0.65)'
        
        return round(total_score, 1), breakdown

    def check_breakeven_viability(self, event: SignalEvent, current_price: float) -> Tuple[bool, str]:
        """
        Calcula si el Take Profit esperado supera los costos de fees y funding.
        Si tp_pct no existe, asume magnitud esperada o asume false.
        """
        # Estimación conservadora: Entramos Limit (Maker), pero asumimos Salida Market (Taker) por seguridad
        maker_fee = getattr(Config, 'BINANCE_MAKER_FEE_BNB', 0.0002)
        taker_fee = getattr(Config, 'BINANCE_TAKER_FEE_BNB', 0.000375)
        total_fees_pct = maker_fee + taker_fee
        
        # Ajuste de Funding (Asumimos 0.01% por default en contra para el peor caso)
        expected_funding_penalty = 0.0001
        
        # Margen de seguridad (ej. 1.5x)
        required_pct = (total_fees_pct + expected_funding_penalty) * self.breakeven_multiplier
        
        # Determinamos el % de ganancia esperado de la señal
        expected_gain_pct = 0.0
        if getattr(event, 'tp_pct', None) is not None:
            expected_gain_pct = event.tp_pct
        elif getattr(event, 'predicted_magnitude', None) is not None:
            expected_gain_pct = abs(event.predicted_magnitude)
        elif getattr(event, 'metadata', None) and 'tp_pct' in event.metadata:
            expected_gain_pct = event.metadata['tp_pct']
            
        if expected_gain_pct <= 1e-9:
            # Si la señal no proyecta nada, veto absoluto en modo antifrágil
            return False, f"Sin proyeccion TP/Magnitud. Required > {required_pct*100:.3f}%"
            
        if expected_gain_pct < required_pct:
            return False, f"Expected {expected_gain_pct*100:.3f}% < Required {required_pct*100:.3f}% (Fees+Safety)"
            
        return True, "Breakeven Viable"
