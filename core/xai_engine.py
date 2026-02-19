import shap
import pandas as pd
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

    def get_feature_importance(self, model, X: pd.DataFrame, model_name: str) -> Dict[str, float]:
        """
        Calcula la importancia de las características para una predicción específica.
        """
        try:
            if model_name not in self.explainers:
                # Inicializar explainer (TreeExplainer es óptimo para XGB/RF)
                self.explainers[model_name] = shap.TreeExplainer(model)
            
            explainer = self.explainers[model_name]
            shap_values = explainer.shap_values(X)
            
            # Si es clasificación binaria, tomamos la clase 1
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
                
            # Promedio de valores SHAP absolutos por feature
            importance = np.abs(shap_values).mean(axis=0)
            feature_importance = dict(zip(X.columns, importance))
            
            # Ordenar por importancia
            sorted_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))
            
            return sorted_importance

        except Exception as e:
            logger.error(f"XAI Error for {model_name}: {e}")
            return {}

    def log_trade_explanation(self, symbol: str, signal_type: str, importance: Dict[str, float]):
        """
        Registra la explicación de un trade en un archivo dedicado.
        """
        try:
            log_path = os.path.join(self.model_dir, f"xai_audit_{symbol.replace('/','_')}.log")
            top_3 = list(importance.items())[:3]
            explanation = ", ".join([f"{k}: {v:.4f}" for k, v in top_3])
            
            with open(log_path, "a") as f:
                f.write(f"[{pd.Timestamp.now()}] Signal: {signal_type} | Top Features: {explanation}\n")
                
        except Exception as e:
            logger.error(f"XAI Log Error: {e}")
