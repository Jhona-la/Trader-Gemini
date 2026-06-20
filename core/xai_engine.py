import numpy as np
from utils.logger import logger
import os

class XAIEngine:
    """
    Explainable AI (XAI) Engine for Trader Gemini.
    QUÉ: Utiliza SHAP para explicar por qué un modelo tomó una decisión específica.
    POR QUÉ: La transparencia es vital para auditar sesgos y mejorar la lógica del bot.
    PARA QUÉ: Proporcionar una capa de interpretabilidad a los modelos de "caja negra" (XGBoost/RF).
    """
    
    def __init__(self, model_dir=".xai"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.explainers = {}

    def explain_global_model(self, model, X_background, model_name: str, symbol: str):
        """
        Generates global SHAP values and saves a summary plot. (Batch/Offline).
        """
        import shap
        import pandas as pd
        import matplotlib.pyplot as plt
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_background)
            
            plt.figure()
            shap.summary_plot(shap_values, X_background, show=False)
            plt.savefig(os.path.join(self.model_dir, f"global_shap_{symbol.replace('/','_')}_{model_name}.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Global XAI Error: {e}")

    def explain_local_prediction(self, model, X_row, model_name: str) -> str:
        """
        Explains a SINGLE prediction (Real-time).
        Returns a string summary of the top 3 features driving this specific decision.
        """
        import shap
        import pandas as pd
        
        try:
            if model_name not in self.explainers:
                # Inicializar explainer (TreeExplainer es óptimo para XGB/RF)
                # feature_perturbation='interventional' is safer for real-time single row
                try:
                    self.explainers[model_name] = shap.TreeExplainer(model)
                except Exception as e:
                    logger.warning(f"⚠️ Could not init SHAP for {model_name}: {e}")
                    return "XAI_UnAvailable"
            
            explainer = self.explainers[model_name]
            
            # Calculate SHAP values for this single row
            shap_values = explainer.shap_values(X_row)
            
            # Handling Binary Classification (taking positive class contribution)
            if isinstance(shap_values, list):
                # shap_values is list of arrays [class0_shap, class1_shap]
                vals = shap_values[1][0] 
            else:
                # shap_values is array (if regression or flattened)
                vals = shap_values[0]

            # Map values to feature names
            feature_names = X_row.columns
            feature_importance = dict(zip(feature_names, vals))
            
            # Sort by MAGNITUDE of impact (absolute value)
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Get Top 3 Drivers
            top_drivers = []
            for k, v in sorted_features[:3]:
                direction = "⬆️" if v > 0 else "⬇️"
                top_drivers.append(f"{k} {direction} ({v:.2f})")
                
            return " | ".join(top_drivers)

        except Exception as e:
            # SHAP errors should never crash the trading bot
            # moderate logging to avoid spam
            return f"XAI_Error: {str(e)[:20]}"

    def log_trade_explanation(self, symbol: str, signal_type: str, explanation: str):
        """
        Registra la explicación de un trade en un archivo dedicado.
        """
        try:
            log_path = os.path.join(self.model_dir, f"xai_audit_{symbol.replace('/','_')}.log")
            
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"[{pd.Timestamp.now()}] Signal: {signal_type} | Drivers: {explanation}\n")
                
        except Exception as e:
            logger.error(f"XAI Log Error: {e}")

    def log_ppo_weight_evolution(self, symbol: str, old_weights: dict, new_weights: dict, reward: float, reason: str):
        """
        Registra la evolución evolutiva de los pesos del ensemble (Neural Fortress - Phase 9).
        QUÉ: Traza cómo los pesos de RF, XGB y GB cambian después de un batch de PPO.
        POR QUÉ: Permite auditar que el sistema se está adaptando correctamente a los Black Swans (PER).
        """
        try:
            log_path = os.path.join(self.model_dir, f"ppo_evolution_{symbol.replace('/','_')}.log")
            
            # Formatear el cambio de pesos para fácil lectura
            weight_diffs = []
            for k in old_weights.keys():
                diff = new_weights.get(k, 0) - old_weights.get(k, 0)
                direction = "📈" if diff > 0 else "📉" if diff < 0 else "➖"
                weight_diffs.append(f"{k}: {old_weights[k]:.2f} -> {new_weights[k]:.2f} ({direction} {diff:+.2f})")
                
            explanation = (
                f"[{pd.Timestamp.now()}] PPO UPDATE | "
                f"Reward: {reward:+.4f} | "
                f"Trigger: {reason} | "
                f"Weights: {' | '.join(weight_diffs)}\n"
            )
            
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(explanation)
                
        except Exception as e:
            logger.error(f"PPO Evolution Log Error: {e}")
