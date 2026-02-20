import numpy as np
from utils.logger import logger
from utils.debug_tracer import trace_execution

class SignalGenerator:
    """
    🧠 COMPONENT: Signal Generator
    Handles the logic for generating trading signals based on ML predictions and technical confluence.
    Extracted from MLStrategy (Excelsior Phase I).
    """
    def __init__(self, strategy_id):
        self.strategy_id = strategy_id

    @trace_execution
    def generate_signal(self, df, prediction, probability, threshold=0.65, regime="UNKNOWN", threshold_mod=0.0):
        """
        Genera una señal de trading basada en la predicción del modelo y la confluencia.
        """
        # Apply Dynamic Threshold Modifier from Regime
        threshold += threshold_mod
        if df.empty:
            return None
            
        current_bar = df.iloc[-1]
        timestamp = current_bar['datetime']
        close_price = current_bar['close']
        
        # Lógica de Señal Básica
        signal_type = "NEUTRAL"
        confidence = probability
        
        # Filtro de Régimen (Solo operar a favor de tendencia si es TRENDING)
        if regime == "TRENDING":
            if prediction == 1 and current_bar['trend_alignment'] > 0:
                signal_type = "BUY"
            elif prediction == 0 and current_bar['trend_alignment'] < 0:
                signal_type = "SELL"
            else:
                confidence *= 0.8 # Penalizar contra-tendencia
                if prediction == 1: signal_type = "BUY"
                elif prediction == 0: signal_type = "SELL"
                
        elif regime == "RANGING":
             # En rango, favorecer reversión a la media
             if prediction == 1 and current_bar['rsi_14'] < 40:
                 signal_type = "BUY"
             elif prediction == 0 and current_bar['rsi_14'] > 60:
                 signal_type = "SELL"
             else:
                 confidence *= 0.7 # Penalizar señales de ruptura en rango
                 if prediction == 1: signal_type = "BUY"
                 elif prediction == 0: signal_type = "SELL"
        
        else: # VOLATILE/UNKNOWN
             if prediction == 1: signal_type = "BUY"
             elif prediction == 0: signal_type = "SELL"
             confidence *= 0.6 # Penalizar alta volatilidad
             
        # Filtro de Confluencia
        confluence = current_bar.get('confluence_score', 0)
        
        # Boost de confianza si hay alta confluencia
        if (signal_type == "BUY" and confluence > 0.3) or \
           (signal_type == "SELL" and confluence < -0.3):
            confidence = min(confidence * 1.2, 0.99)
            
        # Veto si la confluencia es opuesta fuerte
        if (signal_type == "BUY" and confluence < -0.2) or \
           (signal_type == "SELL" and confluence > 0.2):
             logger.info(f"⛔ Link/Confluence Veto: {signal_type} blocked by confluence {confluence:.2f}")
             return None

        # Umbral Final
        if confidence < threshold:
            return None
            
        return {
            'type': signal_type,
            'confidence': confidence,
            'price': close_price,
            'timestamp': timestamp,
            'confluence': confluence,
            'regime': regime
        }
