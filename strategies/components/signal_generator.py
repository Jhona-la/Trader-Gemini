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
        [PHASE 3: Diaria] Meta-Ensembling usa el `market_cluster` para alterar este `threshold`!
        """
        if df.empty:
            return None
            
        current_bar = df.iloc[-1]
        timestamp = current_bar['datetime']
        close_price = current_bar['close']
        
        # ---------------------------------------------------------
        # PHASE 3: SOPHIA KMEANS META-ENSEMBLE THRESHOLD ADJUSTMENT
        # ---------------------------------------------------------
        cluster = current_bar.get('market_cluster', -1)
        # Cluster Anchors: 0=Ranging, 1=Bull, 2=Bear, 3=Choppy
        # Prediction: 1(Buy), 0/-1(Sell)
        is_buy_pred = (prediction == 1)
        is_sell_pred = (prediction == 0 or prediction == -1)
        
        if cluster == 0:
            threshold_mod -= 0.02 # Rango: Ligeramente más permisivos en reversiones
        elif cluster == 1:
            if is_buy_pred: threshold_mod -= 0.05   # Bull: Promover long
            if is_sell_pred: threshold_mod += 0.08  # Bull: Restringir short
        elif cluster == 2:
            if is_sell_pred: threshold_mod -= 0.05  # Bear: Promover short
            if is_buy_pred: threshold_mod += 0.08   # Bear: Restringir long
        elif cluster == 3:
            threshold_mod += 0.05 # Choppy: Exigir alta certeza (+5%) para minimizar whipsaws
            
        # Apply combined modifiers
        threshold += threshold_mod
        
        # Lógica de Señal Básica
        signal_type = "NEUTRAL"
        confidence = probability
        
        # Filtro de Régimen Clásico (Mantiene retro-compatibilidad)
        if regime == "TRENDING":
            if is_buy_pred and current_bar['trend_alignment'] > 0:
                signal_type = "BUY"
            elif is_sell_pred and current_bar['trend_alignment'] < 0:
                signal_type = "SELL"
            else:
                confidence *= 0.8 # Penalizar contra-tendencia
                if is_buy_pred: signal_type = "BUY"
                elif is_sell_pred: signal_type = "SELL"
                
        elif regime == "RANGING":
             # En rango, favorecer reversión a la media
             if is_buy_pred and current_bar['rsi_14'] < 40:
                 signal_type = "BUY"
             elif is_sell_pred and current_bar['rsi_14'] > 60:
                 signal_type = "SELL"
             else:
                 confidence *= 0.7 # Penalizar señales de ruptura en rango
                 if is_buy_pred: signal_type = "BUY"
                 elif is_sell_pred: signal_type = "SELL"
        
        else: # VOLATILE/UNKNOWN
             if is_buy_pred: signal_type = "BUY"
             elif is_sell_pred: signal_type = "SELL"
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
