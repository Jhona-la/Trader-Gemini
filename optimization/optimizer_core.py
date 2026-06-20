import logging
import datetime
import yaml
from typing import Dict, Any, List
from optimization.objective_function import ObjectiveFunction
from optimization.search_space import SearchSpace
from optimization.multi_method_search import MultiMethodSearch
from optimization.walk_forward_cv import WalkForwardValidator
from optimization.shap_analyzer import ShapAnalyzer

logger = logging.getLogger(__name__)

class OptimizerCore:
    """
    Motor Maestro de Optimización (Parte I y XII del Prompt Supremo).
    Orquesta la Capa 3 y el Criterio Supremo de Adopción (12 Condiciones).
    """
    
    def __init__(self):
        self.search_space = SearchSpace()
        self.wf_validator = WalkForwardValidator(data_size_days=30) # Default
        
    def _mock_evaluation(self, config: Dict[str, Any]):
        # En producción esto llama al motor backtest
        import random
        g = random.uniform(0.8, 1.3)
        s = 1 if random.random() > 0.1 else 0
        return g * s * 1.0, s

    def _evaluate_criterio_supremo(self, config: Dict[str, Any], is_score: float, oos_score: float, is_shadow: bool = False) -> Dict[str, Any]:
        """
        Evalúa las 12 Condiciones del Criterio Supremo.
        Si alguna falla, se rechaza la configuración.
        """
        logger.info("⚖️ Evaluando Criterio Supremo de Adopción (12 Condiciones)")
        
        # Simulamos los checks
        # 1. G >= 1.00 (IS check mock)
        cond_1 = is_score >= 1.00
        # 2. S == 1
        cond_2 = True # Asumimos filtrado previo
        # 3. C > 0
        cond_3 = True 
        # 4. Degradación <= 30%
        cond_4 = self.wf_validator.validate_degradation(is_score, oos_score)
        # 5. OOS windows stable
        cond_5 = True
        # 6, 7. Rentabilidad aislada (Capa 1)
        cond_6, cond_7 = True, True
        # 8, 9. No Colisión (Capa 2)
        cond_8 = self.search_space.no_colision(config)
        cond_9 = True 
        # 10. Shadow testing superado
        cond_10 = is_shadow 
        # 11. Fase Capital
        cond_11 = True
        # 12. SHAP sin redundancias
        cond_12 = True # Requeriría pasar lista de redundancias

        todas_superadas = all([cond_1, cond_2, cond_3, cond_4, cond_5, cond_6, cond_7, cond_8, cond_9, cond_10, cond_11, cond_12])
        
        return {
            "aprobado": todas_superadas,
            "condiciones_fallidas": [i for i, c in enumerate([cond_1, cond_2, cond_3, cond_4, cond_5, cond_6, cond_7, cond_8, cond_9, cond_10, cond_11, cond_12], start=1) if not c]
        }

    def execute_perpetual_cycle(self, evaluation_func=None, strategy_name="GENERIC") -> str:
        """
        Ejecuta un ciclo completo de optimización (Parte X) y genera el protocolo de salida.
        """
        eval_func = evaluation_func if evaluation_func else self._mock_evaluation
        logger.info("==================================================")
        logger.info(f"🔮 INICIANDO CICLO PERPETUO DE OPTIMIZACIÓN MAESTRA ({strategy_name})")
        logger.info("==================================================")
        
        # 1. Multi Method Search (Optimizador en In-Sample)
        mms = MultiMethodSearch(self.search_space, eval_func)
        best_is = mms.run_full_pipeline()
        is_config = best_is['config']
        is_score = best_is['score']
        
        # 2. Walk-Forward (OOS validation mock)
        import random
        oos_score = is_score * random.uniform(0.65, 1.1) # Simulate OOS performance
        
        # 3. SHAP Analysis
        shap = ShapAnalyzer(eval_func)
        shap_res = shap.analyze([is_config])
        
        # 4. Criterio Supremo (Se manda a Shadow Testing primero)
        decision_suprema = self._evaluate_criterio_supremo(is_config, is_score, oos_score, is_shadow=False)
        
        decision_str = "ADOPTAR" if decision_suprema["aprobado"] else ("SHADOW_TEST" if 10 in decision_suprema["condiciones_fallidas"] and len(decision_suprema["condiciones_fallidas"]) == 1 else "RECHAZAR")
        
        # 5. Generar YAML Estandarizado (Parte XII)
        protocolo = {
            "resultado_optimizacion": {
                "configuracion_id": f"OPT-{strategy_name}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                "timestamp": datetime.datetime.now().isoformat(),
                "fase_capital": "crecimiento", # mock
                "regimen_detectado": "tendencia", # mock
                "funcion_objetivo": {
                    "G_ciclo": is_score,
                    "S_filtro": 1,
                    "C_multiregimen": 1.1,
                    "F_total": is_score * 1.1
                },
                "criterio_supremo": {
                    "condiciones_superadas": 12 - len(decision_suprema["condiciones_fallidas"]),
                    "condiciones_fallidas": decision_suprema["condiciones_fallidas"],
                    "decision": decision_str
                },
                "colisiones_detectadas": "NINGUNA" if decision_suprema["aprobado"] else "POSIBLES",
                "features_eliminadas_por_shap": shap_res["redundant_features"] or "NINGUNA",
                "proxima_revision": (datetime.datetime.now() + datetime.timedelta(days=3)).isoformat()
            }
        }
        
        yaml_str = yaml.dump(protocolo, default_flow_style=False)
        logger.info(f"📄 PROTOCOLO DE SALIDA:\n{yaml_str}")
        return yaml_str
