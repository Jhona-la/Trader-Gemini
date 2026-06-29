import logging
from typing import Dict, Any

logger = logging.getLogger("AdaptiveConfig")

class AdaptiveConfigIntegral:
    """
    AXIOMA 3: MÓDULO HORIZON
    Matriz 5D: CONFIG[horizonte][activo][estrategia][régimen][parámetro]
    Esta clase reemplaza el Config global rígido. Todo el sistema debe consultar
    esta matriz para tomar decisiones.
    """
    def __init__(self):
        # Nivel 0: Global (Inamovible)
        self.global_config = {
            'exchange': 'binance',
            'base_currency': 'USDT',
            'kill_switch_drawdown_pct': 0.15,
            'min_trade_size_usdt': 10.0,
            'event_bus_heartbeat_seconds': 30,
            'fee_taker_pct': 0.0004,
            'fee_maker_pct': 0.0002
        }

        # Nivel 1: Por Horizonte
        self.matrix = {
            'MICRO': {
                'global_horizon': {
                    'max_hold_seconds': 600,
                    'signal_expiry_seconds': 60,
                    'evaluation_frequency_seconds': 30,
                    'max_concurrent_positions': 3,
                    'capital_allocation_base_pct': 0.25,
                    'capital_allocation_min_pct': 0.10,
                    'capital_allocation_max_pct': 0.40,
                    'order_type_default': 'MARKET',
                    'max_latency_ms': 200,
                    'warmup_candles_minimum': 500
                },
                'por_activo': {
                    'BTC': {
                        'tp_pct_default': 0.14,
                        'sl_pct_default': 0.09,
                        'tp_sl_ratio_min': 1.5,
                        'leverage': 20,
                        'signal_score_min': 72,
                        'trailing_atr_mult': 0.6,
                        'ths_weights': {'cvd_weight': 0.35, 'ema_weight': 0.15, 'prediction_weight': 0.25, 'volume_weight': 0.25},
                        'zombie_n_velas_inactividad': 15,
                        'zombie_z2_vs_threshold': 5
                    },
                    'SOL': {
                        'tp_pct_default': 0.27,
                        'sl_pct_default': 0.16,
                        'tp_sl_ratio_min': 1.6,
                        'leverage': 10,
                        'signal_score_min': 70,
                        'trailing_atr_mult': 0.8,
                        'zombie_n_velas_inactividad': 12
                    },
                    'ALL': { # Fallback
                        'tp_pct_default': 0.2,
                        'sl_pct_default': 0.12,
                        'leverage': 10,
                        'signal_score_min': 70
                    }
                },
                'por_estrategia': {
                    'TFTF': {'consensus_gate': 1.4, 'zombie_n_velas': 15, 'enabled': True},
                    'LCA': {'consensus_gate': 1.3, 'zombie_n_velas': 3, 'enabled': True, 'order_type_override': 'MARKET'},
                    'WYCKOFF': {'enabled': False, 'razon': 'Require D1/H4 structure'}
                },
                'por_regimen': {
                    'TENDENCIAL': {'capital_allocation_modifier': 1.0, 'signal_score_modifier': 0.0},
                    'LATERAL': {'capital_allocation_modifier': 0.8, 'signal_score_modifier': -5.0},
                    'ALTA_VOLATILIDAD': {'capital_allocation_modifier': 0.6, 'signal_score_modifier': -10.0}
                }
            },
            'SCALP': {
                'global_horizon': {
                    'max_hold_seconds': 28800,
                    'signal_expiry_seconds': 480,
                    'evaluation_frequency_seconds': 180,
                    'max_concurrent_positions': 5,
                    'capital_allocation_base_pct': 0.45,
                    'order_type_default': 'LIMIT'
                },
                'por_activo': {
                    'BTC': {'tp_pct_default': 0.55, 'sl_pct_default': 0.32, 'leverage': 10, 'signal_score_min': 65, 'trailing_atr_mult': 1.2, 'zombie_n_velas_inactividad': 20},
                    'SOL': {'tp_pct_default': 0.90, 'sl_pct_default': 0.55, 'leverage': 5, 'signal_score_min': 65, 'zombie_n_velas_inactividad': 15},
                    'ALL': {'tp_pct_default': 0.60, 'sl_pct_default': 0.35, 'leverage': 5, 'signal_score_min': 65}
                },
                'por_estrategia': {
                    'TFTF': {'consensus_gate': 2.0, 'zombie_n_velas': 20, 'enabled': True},
                    'WYCKOFF': {'enabled': False}
                },
                'por_regimen': {
                    'TENDENCIAL': {'capital_allocation_modifier': 1.0, 'signal_score_modifier': 0.0},
                    'LATERAL': {'capital_allocation_modifier': 0.5, 'signal_score_modifier': -10.0},
                    'ALTA_VOLATILIDAD': {'capital_allocation_modifier': 0.8, 'signal_score_modifier': -5.0}
                }
            },
            'SWING': {
                'global_horizon': {
                    'max_hold_seconds': 1209600,
                    'signal_expiry_seconds': 14400,
                    'evaluation_frequency_seconds': 900,
                    'max_concurrent_positions': 3,
                    'capital_allocation_base_pct': 0.30,
                    'order_type_default': 'LIMIT'
                },
                'por_activo': {
                    'BTC': {'tp_pct_default': 2.50, 'sl_pct_default': 1.30, 'leverage': 5, 'signal_score_min': 68, 'trailing_atr_mult': 2.0, 'zombie_n_velas_inactividad': 30},
                    'ALL': {'tp_pct_default': 3.0, 'sl_pct_default': 1.50, 'leverage': 3, 'signal_score_min': 68}
                },
                'por_estrategia': {
                    'WYCKOFF': {'consensus_gate': 3.2, 'zombie_n_velas': 40, 'enabled': True},
                    'LCA': {'enabled': False}
                },
                'por_regimen': {
                    'TENDENCIAL': {'capital_allocation_modifier': 1.0, 'signal_score_modifier': 0.0},
                    'LATERAL': {'capital_allocation_modifier': 0.0, 'signal_score_modifier': -100.0}, # Swing in lateral is bad
                    'ALTA_VOLATILIDAD': {'capital_allocation_modifier': 0.5, 'signal_score_modifier': -15.0}
                }
            }
        }

        # Nivel 2: Fases de Capital
        self.capital_phases = {
            'SEED': {'leverage_multiplier': 0.5, 'max_concurrent_all': 2},
            'GROWTH': {'leverage_multiplier': 0.75, 'max_concurrent_all': 5},
            'SCALE': {'leverage_multiplier': 1.0, 'max_concurrent_all': 10},
            'INSTITUTIONAL': {'leverage_multiplier': 0.8, 'max_concurrent_all': 15, 'require_limit_orders': True}
        }

    def get(self, horizon: str, asset: str, strategy: str, regime: str, param: str) -> Any:
        """
        Retorna O(1) puro el parámetro buscando en la jerarquía:
        1. por_estrategia
        2. por_activo
        3. global_horizon
        4. global_config
        """
        # Tratar de encontrar en estrategia
        if horizon in self.matrix:
            h_matrix = self.matrix[horizon]
            
            # Buscar en estratergia específica
            if strategy in h_matrix['por_estrategia'] and param in h_matrix['por_estrategia'][strategy]:
                return h_matrix['por_estrategia'][strategy][param]
                
            # Buscar en activo específico
            if asset in h_matrix['por_activo'] and param in h_matrix['por_activo'][asset]:
                return h_matrix['por_activo'][asset][param]
                
            # Buscar en activo ALL (Fallback)
            if 'ALL' in h_matrix['por_activo'] and param in h_matrix['por_activo']['ALL']:
                return h_matrix['por_activo']['ALL'][param]
                
            # Buscar en global horizon
            if param in h_matrix['global_horizon']:
                return h_matrix['global_horizon'][param]
                
        # Buscar en global config absoluto
        if param in self.global_config:
            return self.global_config[param]
            
        raise ValueError(f"Configuracion '{param}' no encontrada en matriz 5D para H:{horizon} A:{asset} S:{strategy} R:{regime}")

adaptive_config = AdaptiveConfigIntegral()
