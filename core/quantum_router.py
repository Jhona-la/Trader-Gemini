import numpy as np
from typing import List, Dict, Any, Optional
from utils.logger import logger
from core.events import SignalEvent
from core.enums import SignalType
import dataclasses
from datetime import datetime, timezone

class QuantumRouter:
    """
    🌌 QUANTUM ROUTER - Meta-Orquestador Adaptativo
    
    QUÉ: Un "Ensemble" router que recibe datos del mercado, pide la opinión de TODAS las estrategias
         simultáneamente y genera una Probabilidad Compuesta (Quantum Signal).
    POR QUÉ: Para evitar Whipsaw y dejar que las estrategias compitan ciegamente.
    PARA QUÉ: Para escalar el Position Size basado en la Certeza Matemática (Criterio de Kelly).
    CÓMO: Usa el Softmax del StrategySelector, calcula Kelly (f* = p - q/b) penalizando
          por volatilidad del Market Regime.
    """

    def __init__(self, portfolio, strategy_selector, market_regime):
        self.portfolio = portfolio
        self.strategy_selector = strategy_selector
        self.market_regime = market_regime
        # Mínimo de probabilidad compuesta para ejecutar un trade
        self.confidence_threshold = 0.60 

    def evaluate_tick(self, event: Any, strategies: List[Any], current_price: float) -> List[SignalEvent]:
        """
        Consulta a todas las estrategias por su evaluación del tick actual.
        """
        if getattr(event, 'type', None) != 'MARKET':
            return []

        # 1. Obtener la opinión de todas las estrategias
        signals = []
        for strategy in strategies:
            try:
                # Modificaremos las estrategias para que NO hagan events.put(), sino que retornen la señal.
                signal = strategy.calculate_signals(event)
                if signal:
                    if isinstance(signal, list):
                        signals.extend(signal)
                    else:
                        signals.append(signal)
            except Exception as e:
                logger.error(f"Error evaluating strategy {getattr(strategy, 'strategy_id', 'Unknown')}: {e}")

        if not signals:
            return []

        # 2. Separar Exits y Entradas. 
        # Los EXITS pasan de inmediato al RiskManager sin Kelly.
        exits = [s for s in signals if getattr(s, 'signal_type', None) == SignalType.EXIT or str(getattr(s, 'signal_type', None)) == 'SignalType.EXIT']
        entries = [s for s in signals if s not in exits]
        
        unified_signals = []
        
        # Procesar Exits de inmediato (enviarlos tal cual o agruparlos)
        if exits:
            # Retornamos la señal de Exit más urgente/fuerte (o podríamos retornar todas)
            best_exit = max(exits, key=lambda s: getattr(s, 'confidence', 0.5))
            unified_signals.append(best_exit)

        if not entries:
            return unified_signals

        # 3. Agrupación por Horizonte y Dirección (Ensemble)
        symbol = event.symbol
        
        # Sincronizamos los pesos del Meta-Brain (Softmax)
        weights = self.strategy_selector.get_anti_whipsaw_weights()
        
        horizon_groups = {}
        for sig in entries:
            hz = getattr(sig, 'horizon', 'SCALPING')
            if hz not in horizon_groups:
                horizon_groups[hz] = []
            horizon_groups[hz].append(sig)

        for hz, hz_entries in horizon_groups.items():
            long_conf = 0.0
            short_conf = 0.0
            total_weight = 0.0

            for sig in hz_entries:
                strat_id = getattr(sig, 'strategy_id', 'UNKNOWN')
                weight = weights.get(strat_id, 1.0)
                conf = getattr(sig, 'confidence', 0.5)
                
                # Penalización suave de confidence baseada en weights (el líder opina más fuerte)
                weighted_conf = conf * weight
                
                direction = str(getattr(sig, 'signal_type', 'UNKNOWN'))
                if 'LONG' in direction:
                    long_conf += weighted_conf
                    total_weight += weight
                elif 'SHORT' in direction:
                    short_conf += weighted_conf
                    total_weight += weight

            if total_weight == 0:
                continue

            # 4. Probabilidad Compuesta
            prob_long = long_conf / total_weight if total_weight > 0 else 0
            prob_short = short_conf / total_weight if total_weight > 0 else 0

            # Si chocan fuerzas fuertemente, se neutralizan
            net_prob = abs(prob_long - prob_short)
            winning_dir = SignalType.LONG if prob_long > prob_short else SignalType.SHORT
            winning_prob = max(prob_long, prob_short)

            logger.debug(f"🌌 [QUANTUM][{hz}] {symbol} | LONG: {prob_long:.2%} | SHORT: {prob_short:.2%} | Net: {net_prob:.2%}")

            # Si la probabilidad ganadora neta no supera el umbral
            if winning_prob < self.confidence_threshold:
                continue

            # 5. Criterio de Kelly Fractal
            # f* = p - (1-p)/b
            p = winning_prob
            b = 2.0 
            kelly_fraction = p - ((1.0 - p) / b)

            if kelly_fraction <= 0:
                logger.debug(f"🌌 [QUANTUM][{hz}] Kelly fraction <= 0 para {symbol}. Abortando.")
                continue

            # Aplicar FULL Kelly para crecimiento exponencial agresivo en micro-cuentas
            applied_kelly = kelly_fraction
            
            # Penalización por Market Regime (Volatilidad)
            dh = self.strategy_selector.data_provider
            bars = dh.get_latest_bars(symbol, n=50) if dh else None
            
            if bars is not None and self.market_regime.is_volatility_shock(bars):
                # En shock, castigamos Kelly un 50%
                applied_kelly *= 0.5
                logger.info(f"🌌 [QUANTUM][{hz}] Shock detectado en {symbol}, reduciendo Kelly al {applied_kelly:.2%}")

            # Maximum position fraction (Lifted cap to 80% for micro accounts)
            final_size_pct = np.clip(applied_kelly, 0.01, 0.80)

            # 6. Crear la Señal Unificada para este horizonte
            best_sig = max(hz_entries, key=lambda x: getattr(x, 'confidence', 0.0))
            
            unified_signal = SignalEvent(
                strategy_id="QUANTUM_ROUTER", # Override del ID
                symbol=symbol,
                datetime=datetime.now(timezone.utc).isoformat(),
                signal_type=winning_dir,
                strength=winning_prob,
                confidence=winning_prob,
                # Pass through the TP/SL from the strongest signal
                tp_pct=getattr(best_sig, 'tp_pct', 0.0),
                sl_pct=getattr(best_sig, 'sl_pct', 0.0),
                horizon=hz,
                metadata={
                    "quantum_ensemble": True,
                    "prob_long": prob_long,
                    "prob_short": prob_short,
                    "kelly_fraction": applied_kelly,
                    "contributing_strategies": [getattr(s, 'strategy_id', 'UNKNOWN') for s in hz_entries]
                }
            )
            
            # Truco sucio para propagar quantity_pct
            unified_signal.quantity_pct = final_size_pct
            unified_signals.append(unified_signal)
            
            import logging
            from utils.logger import log_supreme_event
            
            logger.info(f"🚀 [QUANTUM] Unified Signal Fired: [{hz}] {symbol} {winning_dir} (Conf: {winning_prob:.1%}, Kelly Size: {final_size_pct:.1%})")
            
            log_supreme_event(
                logger_instance=logger,
                level=logging.INFO,
                event_id=f"QUANTUM_SIGNAL_{symbol}_{hz}_{int(datetime.now().timestamp())}",
                que_ocurrio={
                    "tipo_evento": "UNIFIED_SIGNAL_GENERATED",
                    "descripcion": f"Señal unificada {winning_dir} creada para {symbol} [{hz}]",
                    "resultado": "SIGNAL_FIRED"
                },
                por_que_ocurrio={
                    "winning_prob": winning_prob,
                    "net_prob": net_prob,
                    "prob_long": prob_long,
                    "prob_short": prob_short
                },
                como_ocurrio={
                    "kelly_fraction_raw": kelly_fraction,
                    "final_size_pct": final_size_pct,
                    "contributing_strategies": [getattr(s, 'strategy_id', 'UNKNOWN') for s in hz_entries]
                },
                donde_ocurrio={
                    "modulo": "QuantumRouter",
                    "funcion": "evaluate_tick"
                },
                quien_lo_provoco={
                    "componente": "QuantumRouter",
                    "weights_utilizados": weights
                }
            )
            
        return unified_signals
