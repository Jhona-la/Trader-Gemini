import os
import sys
import json
import ast

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_adaptive_config(json_path):
    """
    Toma los mejores parámetros del Omni Evolver y genera el código 
    final de adaptive_config.py
    """
    if not os.path.exists(json_path):
        print(f"❌ Error: Archivo JSON no encontrado: {json_path}")
        return
        
    with open(json_path, 'r') as f:
        best_params = json.load(f)
        
    print(f"🧬 Generando arquitectura desde genoma: {json_path}...")
    
    # Plantilla base (Copia de config/adaptive_config.py original con variables inyectadas)
    template = f'''import logging
from typing import Dict, Any

logger = logging.getLogger("AdaptiveConfig")

class AdaptiveConfigIntegral:
    """
    AXIOMA 3: MÓDULO HORIZON
    Generado dinámicamente por OMNI-EVOLVER.
    """
    def __init__(self):
        self.global_config = {{
            'exchange': 'binance',
            'base_currency': 'USDT',
            'kill_switch_drawdown_pct': 0.15,
            'min_trade_size_usdt': 10.0,
            'event_bus_heartbeat_seconds': 30,
            'fee_taker_pct': 0.0004,
            'fee_maker_pct': 0.0002
        }}

        self.matrix = {{
            'MICRO': {{
                'global_horizon': {{
                    'max_hold_seconds': {best_params['micro_max_hold_sec']},
                    'signal_expiry_seconds': 60,
                    'evaluation_frequency_seconds': 30,
                    'max_concurrent_positions': {best_params['micro_max_conc']},
                    'capital_allocation_base_pct': {best_params['micro_alloc_base']},
                    'capital_allocation_min_pct': 0.10,
                    'capital_allocation_max_pct': 0.40,
                    'order_type_default': 'MARKET',
                    'max_latency_ms': 200,
                    'warmup_candles_minimum': 500
                }},
                'por_activo': {{
                    'BTC': {{
                        'tp_pct_default': {best_params['micro_btc_tp']},
                        'sl_pct_default': {best_params['micro_btc_sl']},
                        'tp_sl_ratio_min': 1.5,
                        'leverage': {best_params['micro_btc_lev']},
                        'signal_score_min': {best_params['micro_btc_score']},
                        'trailing_atr_mult': {best_params['micro_btc_trail_atr']},
                        'zombie_n_velas_inactividad': {best_params['micro_btc_zombie']}
                    }},
                    'SOL': {{
                        'tp_pct_default': 0.27,
                        'sl_pct_default': 0.16,
                        'leverage': 10,
                        'signal_score_min': 70
                    }},
                    'ALL': {{
                        'tp_pct_default': 0.2,
                        'sl_pct_default': 0.12,
                        'leverage': 10,
                        'signal_score_min': 70
                    }}
                }},
                'por_estrategia': {{
                    'TFTF': {{'consensus_gate': 1.4, 'zombie_n_velas': 15, 'enabled': True}},
                    'LCA': {{'consensus_gate': 1.3, 'zombie_n_velas': 3, 'enabled': True}}
                }},
                'por_regimen': {{
                    'TENDENCIAL': {{'capital_allocation_modifier': 1.0, 'signal_score_modifier': 0.0}},
                    'LATERAL': {{'capital_allocation_modifier': 0.8, 'signal_score_modifier': -5.0}},
                    'ALTA_VOLATILIDAD': {{'capital_allocation_modifier': 0.6, 'signal_score_modifier': -10.0}}
                }}
            }},
            'SCALP': {{
                'global_horizon': {{
                    'max_hold_seconds': {best_params['scalp_max_hold_sec']},
                    'signal_expiry_seconds': 480,
                    'evaluation_frequency_seconds': 180,
                    'max_concurrent_positions': {best_params['scalp_max_conc']},
                    'capital_allocation_base_pct': {best_params['scalp_alloc_base']},
                    'order_type_default': 'LIMIT'
                }},
                'por_activo': {{
                    'BTC': {{
                        'tp_pct_default': {best_params['scalp_btc_tp']},
                        'sl_pct_default': {best_params['scalp_btc_sl']},
                        'leverage': {best_params['scalp_btc_lev']},
                        'signal_score_min': {best_params['scalp_btc_score']},
                        'trailing_atr_mult': {best_params['scalp_btc_trail_atr']},
                        'zombie_n_velas_inactividad': {best_params['scalp_btc_zombie']}
                    }},
                    'ALL': {{'tp_pct_default': 0.60, 'sl_pct_default': 0.35, 'leverage': 5, 'signal_score_min': 65}}
                }},
                'por_estrategia': {{
                    'TFTF': {{'consensus_gate': 2.0, 'zombie_n_velas': 20, 'enabled': True}}
                }},
                'por_regimen': {{
                    'TENDENCIAL': {{'capital_allocation_modifier': 1.0, 'signal_score_modifier': 0.0}},
                    'LATERAL': {{'capital_allocation_modifier': 0.5, 'signal_score_modifier': -10.0}},
                    'ALTA_VOLATILIDAD': {{'capital_allocation_modifier': 0.8, 'signal_score_modifier': -5.0}}
                }}
            }},
            'SWING': {{
                'global_horizon': {{
                    'max_hold_seconds': 1209600,
                    'max_concurrent_positions': 3,
                    'capital_allocation_base_pct': 0.30
                }},
                'por_activo': {{
                    'BTC': {{'tp_pct_default': 2.50, 'sl_pct_default': 1.30, 'leverage': 5, 'signal_score_min': 68}},
                    'ALL': {{'tp_pct_default': 3.0, 'sl_pct_default': 1.50, 'leverage': 3, 'signal_score_min': 68}}
                }},
                'por_estrategia': {{
                    'WYCKOFF': {{'consensus_gate': 3.2, 'enabled': True}}
                }},
                'por_regimen': {{
                    'TENDENCIAL': {{'capital_allocation_modifier': 1.0, 'signal_score_modifier': 0.0}},
                    'LATERAL': {{'capital_allocation_modifier': 0.0, 'signal_score_modifier': -100.0}},
                    'ALTA_VOLATILIDAD': {{'capital_allocation_modifier': 0.5, 'signal_score_modifier': -15.0}}
                }}
            }}
        }}

        self.capital_phases = {{
            'SEED': {{'leverage_multiplier': 0.5, 'max_concurrent_all': 2}},
            'GROWTH': {{'leverage_multiplier': 0.75, 'max_concurrent_all': 5}},
            'SCALE': {{'leverage_multiplier': 1.0, 'max_concurrent_all': 10}},
            'INSTITUTIONAL': {{'leverage_multiplier': 0.8, 'max_concurrent_all': 15}}
        }}

    def get(self, horizon: str, asset: str, strategy: str, regime: str, param: str) -> Any:
        if horizon in self.matrix:
            h_matrix = self.matrix[horizon]
            if strategy in h_matrix['por_estrategia'] and param in h_matrix['por_estrategia'][strategy]:
                return h_matrix['por_estrategia'][strategy][param]
            if asset in h_matrix['por_activo'] and param in h_matrix['por_activo'][asset]:
                return h_matrix['por_activo'][asset][param]
            if 'ALL' in h_matrix['por_activo'] and param in h_matrix['por_activo']['ALL']:
                return h_matrix['por_activo']['ALL'][param]
            if param in h_matrix['global_horizon']:
                return h_matrix['global_horizon'][param]
        if param in self.global_config:
            return self.global_config[param]
        raise ValueError(f"Configuracion '{{param}}' no encontrada.")

adaptive_config = AdaptiveConfigIntegral()
'''
    
    out_path = os.path.join(_project_root, "config", "best_adaptive_config.py")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(template)
        
    print(f"✅ ¡Éxito! Nueva arquitectura maestra generada en:")
    print(f"   -> {out_path}")
    print(f"   Para usarla, renombra 'best_adaptive_config.py' a 'adaptive_config.py'.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python tools/config_generator.py data/omni_evolver_best_YYYYMMDD.json")
    else:
        generate_adaptive_config(sys.argv[1])
