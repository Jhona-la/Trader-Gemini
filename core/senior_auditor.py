# core/senior_auditor.py
"""
🕵️ MÓDULO AUDIT SENIOR: AUDITORÍA INTEGRAL DE APERTURA, SEGUIMIENTO Y CIERRE
=============================================================================
PROFESSOR METHOD:
- QUÉ: Motor de auditoría omnisciente para supervisar el ciclo de vida completo
       de cada posición activa basado en el ADN específico de cada estrategia.
- POR QUÉ: Prevenir la ceguera operativa post-apertura y asegurar que el cierre
           razone íntimamente sobre el contexto de la apertura y el activo.
- PARA QUÉ: Maximizar la captura de rentabilidad, evitar salidas injustificadas
            y garantizar un registro forense íntegro para auto-aprendizaje.
- CÓMO: 
    - Formaliza el ADN de 11 estrategias.
    - Ejecuta roles senior: ACS (Coherencia), ACI (Continuidad), AEA (Activo),
      ACR (Rentabilidad) y ATA (Trazabilidad).
    - Implementa el protocolo de degradación por ceguera (Niveles 1 a 3).
- CUÁNDO: Evaluado en la admisión de señales (apertura), en cada kline (seguimiento)
          y en la revisión de stop loss (cierre).
- DÓNDE: core/senior_auditor.py (este archivo).
- QUIÉN: Diseñado por el Equipo de Auditores Senior de Trader Gemini.
"""

import os
import json
import time
import hashlib
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Tuple, Optional, List

from utils.logger import logger
from core.events import SignalEvent, SignalType, SignalState
from config import Config

# =========================================================================
# 🧬 STRATEGY DNA REGISTRY (11 Estructuras de ADN Formales)
# =========================================================================

STRATEGY_DNA: Dict[str, Dict[str, Any]] = {
    "TFTF": {
        "nombre": "TFTF — Trend Following Multi-Timeframe",
        "TESIS_DE_APERTURA": {
            "descripcion": "Los mercados con tendencia en timeframe mayor producen pullbacks en el menor que ofrecen entradas en la dirección de la tendencia mayor.",
            "condicion_necesaria": "Tendencia confirmada en TF mayor (ADX > 25, EMA en dirección)",
            "condicion_suficiente": "Pullback en zona Fibonacci 0.382-0.618 del último impulso",
            "señal_generadora": "Llegada al pullback con volumen decreciente y CVD favorable",
            "regime_requerido": "TENDENCIAL"
        },
        "INDICADORES_CRITICOS_DE_LA_TESIS": {
            "indicador_primario": "ADX",
            "indicadores_de_confirmacion": ["CVD_trend", "Volume_pullback_decreciente", "Hurst > 0.55"],
            "indicadores_de_invalidacion": ["ADX < 20", "CVD_streak_reversa >= 3", "Cierre_contra_EMA_HTF"]
        },
        "NIVELES_CLAVE_AL_MOMENTO_DE_APERTURA": {
            "nivel_de_entrada": "Zona 0.382-0.618 Fib",
            "nivel_de_invalidacion_estructural": "Swing Low/High del impulso HTF",
            "nivel_objetivo_R1": "1.5x riesgo",
            "nivel_objetivo_R2": "2.5x riesgo"
        },
        "VENTANA_TEMPORAL_DE_LA_TESIS": {
            "tiempo_minimo_para_validar": 60,
            "tiempo_maximo_de_validez": 7200,
            "velocidad_esperada_del_movimiento": "MODERADA"
        },
        "CONDICIONES_DE_CONTINUACION": ["ADX > 20 en HTF", "CVD sin reversa sostenida"],
        "CONDICIONES_DE_INVALIDACION": ["ADX < 20", "Cierre contra EMA HTF", "CVD_streak_reversa >= 3", "CHoCH en contra"],
        "SEÑAL_DE_CIERRE_NATURAL": {
            "descripcion": "El impulso tendencial llega a zona de soporte/resistencia HTF con CVD girando en contra.",
            "señal_técnica": "RSI extremo + CVD reversa + volumen descendente"
        },
        "ASIMETRIAS_DEL_ACTIVO": {
            "BTC/USDT": {"leverage_max": 20, "atr_multiplier": 1.5},
            "DOGE/USDT": {"leverage_max": 10, "atr_multiplier": 2.5}
        }
    },
    "OB_RETEST": {
        "nombre": "OB_RETEST — Order Block Retest (Smart Money Concepts)",
        "TESIS_DE_APERTURA": {
            "descripcion": "Las instituciones dejan huellas de volumen (Order Blocks). El precio al regresar a esa zona reactiva la oferta/demanda institucional.",
            "condicion_necesaria": "Existencia de OB fresh no mitigado con fuerte impulso previo",
            "condicion_suficiente": "Llegada al OB con momentum decreciente y CVD con absorción",
            "señal_generadora": "Vela de rechazo en OB + CVD girando",
            "regime_requerido": "TENDENCIAL"
        },
        "INDICADORES_CRITICOS_DE_LA_TESIS": {
            "indicador_primario": "OB_frescura_score",
            "indicadores_de_confirmacion": ["CVD_absorcion", "Spread_normalizado", "Alineacion_HTF"],
            "indicadores_de_invalidacion": ["Cierre_fuera_extremo_OB", "Volume_perforacion > 2x_promedio", "CVD_perforacion_contra"]
        },
        "NIVELES_CLAVE_AL_MOMENTO_DE_APERTURA": {
            "nivel_de_entrada": "Extremo del OB",
            "nivel_de_invalidacion_estructural": "1 tick más allá del OB",
            "nivel_objetivo_R1": "FVG opuesto",
            "nivel_objetivo_R2": "OB opuesto"
        },
        "VENTANA_TEMPORAL_DE_LA_TESIS": {
            "tiempo_minimo_para_validar": 30,
            "tiempo_maximo_de_validez": 5400,
            "velocidad_esperada_del_movimiento": "RÁPIDA"
        },
        "CONDICIONES_DE_CONTINUACION": ["Precio no cierra fuera de OB", "CVD neutral/favorable"],
        "CONDICIONES_DE_INVALIDACION": ["Cierre de vela fuera del OB", "Volumen de perforación > 2x", "Catalizador en contra"],
        "SEÑAL_DE_CIERRE_NATURAL": {
            "descripcion": "El precio alcanza el FVG u OB opuesto con rechazo direccional.",
            "señal_técnica": "Rechazo en target + CVD reversa"
        },
        "ASIMETRIAS_DEL_ACTIVO": {
            "BTC/USDT": {"atr_multiplier": 1.0},
            "DOGE/USDT": {"atr_multiplier": 2.0}
        }
    },
    "LCA": {
        "nombre": "LCA — Liquidation Cascade Anticipation",
        "TESIS_DE_APERTURA": {
            "descripcion": "El momentum empuja el precio hacia un clúster pesado de liquidaciones. Su activación genera un spike en cascada que el bot captura.",
            "condicion_necesaria": "Clúster denso de liquidaciones cerca del precio actual",
            "condicion_suficiente": "Momentum activo + sin soportes/resistencias entre el precio y el clúster",
            "señal_generadora": "CVD acelerando hacia el clúster + órdenes grandes en order flow",
            "regime_requerido": "CUALQUIERA"
        },
        "INDICADORES_CRITICOS_DE_LA_TESIS": {
            "indicador_primario": "Cluster_distancia",
            "indicadores_de_confirmacion": ["CVD_aceleracion", "Orderflow_size_creciente", "Cluster_size_ratio > 1.5"],
            "indicadores_de_invalidacion": ["CVD_reversa_inmediata", "Precio_plano_3_velas"]
        },
        "NIVELES_CLAVE_AL_MOMENTO_DE_APERTURA": {
            "nivel_de_entrada": "Mercado",
            "nivel_de_invalidacion_estructural": "ATR * 0.8 en contra",
            "nivel_objetivo_R1": "Nivel justo post-clúster"
        },
        "VENTANA_TEMPORAL_DE_LA_TESIS": {
            "tiempo_minimo_para_validar": 10,
            "tiempo_maximo_de_validez": 90,  # 90 segundos máximo para HFT
            "velocidad_esperada_del_movimiento": "EXTREMA"
        },
        "CONDICIONES_DE_CONTINUACION": ["Precio avanzando hacia el clúster", "CVD en dirección"],
        "CONDICIONES_DE_INVALIDACION": ["CVD reversa en primera vela", "Estancamiento > 3 velas", "Cascada en contra"],
        "SEÑAL_DE_CIERRE_NATURAL": {
            "descripcion": "Barrido del clúster completado, CVD empieza a revertir por absorción.",
            "señal_técnica": "Precio en post-clúster + volumen spike + CVD reversa"
        },
        "ASIMETRIAS_DEL_ACTIVO": {
            "BTC/USDT": {"stale_threshold": 15},
            "SOL/USDT": {"stale_threshold": 10}
        }
    },
    "MRBB": {
        "nombre": "MRBB — Mean Reversion Bollinger Bands",
        "TESIS_DE_APERTURA": {
            "descripcion": "Divergencias extremas del precio respecto a la media tienden a revertir en mercados laterales.",
            "condicion_necesaria": "Precio en banda exterior de Bollinger",
            "condicion_suficiente": "Señal de agotamiento + CVD girando + ADX < 25",
            "señal_generadora": "Vela de rechazo en banda + CVD reversa",
            "regime_requerido": "LATERAL"
        },
        "INDICADORES_CRITICOS_DE_LA_TESIS": {
            "indicador_primario": "Bollinger_%B",
            "indicadores_de_confirmacion": ["RSI_divergencia", "Bollinger_Bandwidth_estable", "Hurst < 0.50"],
            "indicadores_de_invalidacion": ["ADX > 25", "Cierre_sostenido_fuera_banda", "Bandwidth_expansion_brusca"]
        },
        "NIVELES_CLAVE_AL_MOMENTO_DE_APERTURA": {
            "nivel_de_entrada": "Extremo de banda",
            "nivel_de_invalidacion_estructural": "Cierre fuera de banda + ATR * 1.0",
            "nivel_objetivo_R1": "Media móvil (EMA-20)"
        },
        "VENTANA_TEMPORAL_DE_LA_TESIS": {
            "tiempo_minimo_para_validar": 60,
            "tiempo_maximo_de_validez": 3600,
            "velocidad_esperada_del_movimiento": "MODERADA"
        },
        "CONDICIONES_DE_CONTINUACION": ["ADX < 25", "Bandwidth plano", "CVD moviéndose a media"],
        "CONDICIONES_DE_INVALIDACION": ["ADX > 25", "Precio perforando con volumen", "Explosión de Bandwidth"],
        "SEÑAL_DE_CIERRE_NATURAL": {
            "descripcion": "El precio toca la EMA-20 media con normalización del %B.",
            "señal_técnica": "Precio en media + %B cercano a 0.5 + CVD plano"
        },
        "ASIMETRIAS_DEL_ACTIVO": {
            "BTC/USDT": {"std_dev": 2.0},
            "SOL/USDT": {"std_dev": 2.2}
        }
    },
    "WYCKOFF": {
        "nombre": "WYCKOFF — Wyckoff Spring y UTAD",
        "TESIS_DE_APERTURA": {
            "descripcion": "Falsas rupturas en los límites del rango institucional (Spring / UTAD) limpian stop losses antes del markup/markdown real.",
            "condicion_necesaria": "Estructura de acumulación/distribución identificada",
            "condicion_suficiente": "Spring/UTAD que rompe soporte/resistencia con volumen bajo y recupera rápidamente",
            "señal_generadora": "Test del Spring/UTAD con volumen bajo + CVD favorable",
            "regime_requerido": "POST-RANGO"
        },
        "INDICADORES_CRITICOS_DE_LA_TESIS": {
            "indicador_primario": "Fase_Wyckoff",
            "indicadores_de_confirmacion": ["Volumen_spring < 60%_BC", "OBV_acumulacion", "CVD_positivo_test"],
            "indicadores_de_invalidacion": ["Spring_perfora_con_volumen", "OBV_distribucion", "Falta_recuperacion_soporte"]
        },
        "NIVELES_CLAVE_AL_MOMENTO_DE_APERTURA": {
            "nivel_de_entrada": "Test del Spring",
            "nivel_de_invalidacion_estructural": "Mínimo del Spring",
            "nivel_objetivo_R1": "Creek (resistencia de rango)",
            "nivel_objetivo_R2": "Target de Markup"
        },
        "VENTANA_TEMPORAL_DE_LA_TESIS": {
            "tiempo_minimo_para_validar": 600,
            "tiempo_maximo_de_validez": 172800,  # 48 horas
            "velocidad_esperada_del_movimiento": "LENTA"
        },
        "CONDICIONES_DE_CONTINUACION": ["Estructura intacta", "OBV alcista", "CVD neto a favor"],
        "CONDICIONES_DE_INVALIDACION": ["Perforación del mínimo del Spring", "OBV en contra", "Ruptura del Creek en contra"],
        "SEÑAL_DE_CIERRE_NATURAL": {
            "descripcion": "El precio alcanza el Creek o la proyección de markup y muestra distribución secundaria.",
            "señal_técnica": "Precio en markup target + OBV plano + CVD plano"
        },
        "ASIMETRIAS_DEL_ACTIVO": {
            "BTC/USDT": {"min_hold": 1200},
            "ETH/USDT": {"min_hold": 1800}
        }
    },
    "VBA": {
        "nombre": "VBA — Volatility Breakout ATR",
        "TESIS_DE_APERTURA": {
            "descripcion": "Períodos de compresión de volatilidad extrema acumulan fuerza direccional que al expandirse genera un movimiento con momentum.",
            "condicion_necesaria": "ATR en percentil bajo histórico + Bandwidth en mínimos",
            "condicion_suficiente": "Vela expansiva > 2x ATR con volumen y CVD coordinado",
            "señal_generadora": "Cierre de vela de expansión + CVD a favor",
            "regime_requerido": "POST-COMPRESIÓN"
        },
        "INDICADORES_CRITICOS_DE_LA_TESIS": {
            "indicador_primario": "ATR_ratio (ATR-14/ATR-50)",
            "indicadores_de_confirmacion": ["Bollinger_Bandwidth_squeeze", "Spike_volumen > 1.5x", "CVD_momentum"],
            "indicadores_de_invalidacion": ["Regreso_al_rango_compresion", "Vela_siguiente_volumen_nulo"]
        },
        "NIVELES_CLAVE_AL_MOMENTO_DE_APERTURA": {
            "nivel_de_entrada": "Cierre de vela breakout",
            "nivel_de_invalidacion_estructural": "Punto medio del rango de compresión",
            "nivel_objetivo_R1": "Breakout + ATR * 2.0",
            "nivel_objetivo_R2": "Breakout + ATR * 4.0"
        },
        "VENTANA_TEMPORAL_DE_LA_TESIS": {
            "tiempo_minimo_para_validar": 60,
            "tiempo_maximo_de_validez": 3600,
            "velocidad_esperada_del_movimiento": "RÁPIDA"
        },
        "CONDICIONES_DE_CONTINUACION": ["ATR no se contrae de inmediato", "CVD a favor", "Precio sobre 50% de vela breakout"],
        "CONDICIONES_DE_INVALIDACION": ["Regreso al rango", "CVD reversa antes de R1"],
        "SEÑAL_DE_CIERRE_NATURAL": {
            "descripcion": "El ATR empieza a contraerse indicando agotamiento de la volatilidad.",
            "señal_técnica": "ATR decrece en máximos + CVD plano"
        },
        "ASIMETRIAS_DEL_ACTIVO": {
            "BTC/USDT": {"atr_mult": 2.0},
            "DOGE/USDT": {"atr_mult": 3.0}
        }
    },
    "MBV": {
        "nombre": "MBV — Momentum Breakout con Volumen",
        "TESIS_DE_APERTURA": {
            "descripcion": "Ruptura de niveles de resistencia/soporte críticos validados por volumen excepcional e intención de compra/venta real.",
            "condicion_necesaria": "Ruptura física de nivel relevante (3+ toques previos)",
            "condicion_suficiente": "Volumen > percentil 80 histórico + CVD masivo en dirección",
            "señal_generadora": "Cierre de vela breakout + volumen + CVD",
            "regime_requerido": "TENDENCIAL"
        },
        "INDICADORES_CRITICOS_DE_LA_TESIS": {
            "indicador_primario": "Volume_breakout_percentile",
            "indicadores_de_confirmacion": ["CVD_masivo", "RSI_expansion", "Falta_soportes_resistencias_cercanos"],
            "indicadores_de_invalidacion": ["Regreso_bajo_nivel_roto", "CVD_reversa_inmediata", "Falta_continuidad_volumen"]
        },
        "NIVELES_CLAVE_AL_MOMENTO_DE_APERTURA": {
            "nivel_de_entrada": "Cierre de vela breakout",
            "nivel_de_invalidacion_estructural": "Nivel roto + ATR * 0.5",
            "nivel_objetivo_R1": "Próxima resistencia/soporte HTF"
        },
        "VENTANA_TEMPORAL_DE_LA_TESIS": {
            "tiempo_minimo_para_validar": 30,
            "tiempo_maximo_de_validez": 3600,
            "velocidad_esperada_del_movimiento": "RÁPIDA"
        },
        "CONDICIONES_DE_CONTINUACION": ["Precio mantiene el nivel", "CVD a favor"],
        "CONDICIONES_DE_INVALIDACION": ["Cierre por debajo del nivel roto", "CVD reversa", "Falta de volumen"],
        "SEÑAL_DE_CIERRE_NATURAL": {
            "descripcion": "El precio alcanza el nivel técnico HTF con volumen descendente.",
            "señal_técnica": "Precio en nivel HTF + volumen decreciente + CVD plano"
        },
        "ASIMETRIAS_DEL_ACTIVO": {
            "BTC/USDT": {"atr_mult": 1.0},
            "DOGE/USDT": {"atr_mult": 2.0}
        }
    },
    "FRA": {
        "nombre": "FRA — Funding Rate Arbitrage",
        "TESIS_DE_APERTURA": {
            "descripcion": "Funding rates extremos representan desequilibrio excesivo. El costo de carry insostenible fuerza liquidaciones que el bot aprovecha en contra.",
            "condicion_necesaria": "Funding rate en percentil > 90 o < 10",
            "condicion_suficiente": "Estructura técnica de debilidad del lado mayoritario (reversa)",
            "señal_generadora": "Funding rate extremo + vela de debilidad + CVD reversa",
            "regime_requerido": "CUALQUIERA"
        },
        "INDICADORES_CRITICOS_DE_LA_TESIS": {
            "indicador_primario": "Funding_rate_percentile",
            "indicadores_de_confirmacion": ["Long/Short_ratio_extremo", "Open_Interest_elevado", "CVD_debilidad"],
            "indicadores_de_invalidacion": ["El funding extremo persiste sin reversión", "Catalizador fundamental justifica el desequilibrio"]
        },
        "NIVELES_CLAVE_AL_MOMENTO_DE_APERTURA": {
            "nivel_de_entrada": "Señal",
            "nivel_de_invalidacion_estructural": "Swing High/Low + ATR * 1.0",
            "nivel_objetivo_R1": "Reversión a media histórica"
        },
        "VENTANA_TEMPORAL_DE_LA_TESIS": {
            "tiempo_minimo_para_validar": 300,
            "tiempo_maximo_de_validez": 28800,  # 8 horas
            "velocidad_esperada_del_movimiento": "MODERADA"
        },
        "CONDICIONES_DE_CONTINUACION": ["CVD moviéndose a favor", "OI cayendo (desapalancamiento)"],
        "CONDICIONES_DE_INVALIDACION": ["Funding sigue expandiéndose", "Precio perfora en contra de reversa"],
        "SEÑAL_DE_CIERRE_NATURAL": {
            "descripcion": "El funding se normaliza hacia niveles sanos.",
            "señal_técnica": "Funding en niveles neutros + precio en target"
        },
        "ASIMETRIAS_DEL_ACTIVO": {
            "BNB/USDT": {"funding_threshold": 0.001},
            "BTC/USDT": {"funding_threshold": 0.0005}
        }
    },
    "SC": {
        "nombre": "SC — Sentiment Contrarian",
        "TESIS_DE_APERTURA": {
            "descripcion": "El pánico o euforia extrema del mercado (sentimiento) deja sin contrapartes el movimiento, provocando reversiones bruscas.",
            "condicion_necesaria": "Fear & Greed Index compuesto en extremos (< 15 o > 85)",
            "condicion_suficiente": "Vela de reversión técnica en zonas S/R clave + CVD girando",
            "señal_generadora": "F&G extremo + vela reversión + CVD",
            "regime_requerido": "CUALQUIERA"
        },
        "INDICADORES_CRITICOS_DE_LA_TESIS": {
            "indicador_primario": "Fear_&_Greed_Index",
            "indicadores_de_confirmacion": ["Funding_rate_coherente", "Long/Short_ratio_extremo", "Social_volume_percentile > 90"],
            "indicadores_de_invalidacion": ["El sentimiento extremo persiste sin reacción por 3 días", "Noticia fundamental de alto impacto"]
        },
        "NIVELES_CLAVE_AL_MOMENTO_DE_APERTURA": {
            "nivel_de_entrada": "Reversal vela",
            "nivel_de_invalidacion_estructural": "Extremo del spike de sentimiento",
            "nivel_objetivo_R1": "Zona de valor / Media móvil"
        },
        "VENTANA_TEMPORAL_DE_LA_TESIS": {
            "tiempo_minimo_para_validar": 600,
            "tiempo_maximo_de_validez": 86400,  # 24 horas
            "velocidad_esperada_del_movimiento": "MODERADA"
        },
        "CONDICIONES_DE_CONTINUACION": ["Sentimiento relajándose", "CVD a favor de reversión"],
        "CONDICIONES_DE_INVALIDACION": ["Sentimiento empeora", "Noticia relevante que valida el pánico/euforia"],
        "SEÑAL_DE_CIERRE_NATURAL": {
            "descripcion": "El sentimiento vuelve a zonas neutras (40-60).",
            "señal_técnica": "F&G entre 40 y 60 + precio en zona de valor"
        },
        "ASIMETRIAS_DEL_ACTIVO": {
            "BTC/USDT": {"sentiment_source": "composite"},
            "DOGE/USDT": {"sentiment_source": "social"}
        }
    },
    "STATARB": {
        "nombre": "STATARB — Statistical Arbitrage de Pares",
        "TESIS_DE_APERTURA": {
            "descripcion": "Activos con alta correlación histórica que divergen temporalmente. El bot compra el rezagado y vende el adelantado esperando reversión del spread.",
            "condicion_necesaria": "Correlación rolling de 30 días > 0.80 entre el par",
            "condicion_suficiente": "Divergencia del spread > 2 desviaciones estándar (Z-score > 2.0)",
            "señal_generadora": "Z-score > 2.0 + inicio de convergencia del spread",
            "regime_requerido": "CUALQUIERA"
        },
        "INDICADORES_CRITICOS_DE_LA_TESIS": {
            "indicador_primario": "Spread_Z-score",
            "indicadores_de_confirmacion": ["Correlacion_rolling_4h > 0.80", "Divergencia_sin_catalizadores_fundamentales"],
            "indicadores_de_invalidacion": ["Correlación < 0.70 (desacople estructural)", "Z-score > 3.5"]
        },
        "NIVELES_CLAVE_AL_MOMENTO_DE_APERTURA": {
            "nivel_de_entrada": "Divergencia extrema",
            "nivel_de_invalidacion_estructural": "Z-score > 3.5",
            "nivel_objetivo_R1": "Z-score = 0.0"
        },
        "VENTANA_TEMPORAL_DE_LA_TESIS": {
            "tiempo_minimo_para_validar": 300,
            "tiempo_maximo_de_validez": 43200,  # 12 horas
            "velocidad_esperada_del_movimiento": "MODERADA"
        },
        "CONDICIONES_DE_CONTINUACION": ["Correlación alta", "Z-score moviéndose a 0"],
        "CONDICIONES_DE_INVALIDACION": ["Desacople permanente", "Spread sigue divergiendo (Z > 3.5)"],
        "SEÑAL_DE_CIERRE_NATURAL": {
            "descripcion": "El spread converge a su media (Z-score = 0.0).",
            "señal_técnica": "Z-score entre -0.5 y 0.5"
        },
        "ASIMETRIAS_DEL_ACTIVO": {
            "BTC/ETH": {"correlation_threshold": 0.85},
            "ETH/BNB": {"correlation_threshold": 0.80}
        }
    },
    "OCS": {
        "nombre": "OCS — On-Chain Signals",
        "TESIS_DE_APERTURA": {
            "descripcion": "Métricas on-chain muestran acumulación/distribución institucional (SOPR, MVRV, Exchange Netflow) antes de reflejarse en precio.",
            "condicion_necesaria": "Métrica on-chain en zona de desviación histórica extrema",
            "condicion_suficiente": "SOPR < 1 + MVRV en infravaloración + Exchange Netflow cayendo",
            "señal_generadora": "Alineación de 3+ métricas on-chain en zona de compra/venta",
            "regime_requerido": "POST-CAPITULACIÓN"
        },
        "INDICADORES_CRITICOS_DE_LA_TESIS": {
            "indicador_primario": "MVRV_Ratio",
            "indicadores_de_confirmacion": ["SOPR", "NUPL", "Exchange_Netflow_negativo", "Realized_Price"],
            "indicadores_de_invalidacion": ["Deterioro_onchain (Exchange inflows masivos)", "MVRV perfora soporte histórico"]
        },
        "NIVELES_CLAVE_AL_MOMENTO_DE_APERTURA": {
            "nivel_de_entrada": "Zona de valor",
            "nivel_de_invalidacion_estructural": "Perforación del Realized Price * 0.90",
            "nivel_objetivo_R1": "Revalorización MVRV > 2.0"
        },
        "VENTANA_TEMPORAL_DE_LA_TESIS": {
            "tiempo_minimo_para_validar": 7200,
            "tiempo_maximo_de_validez": 604800,  # 7 días para swing
            "velocidad_esperada_del_movimiento": "LENTA"
        },
        "CONDICIONES_DE_CONTINUACION": ["Netflow negativo", "SOPR > 1 en subida"],
        "CONDICIONES_DE_INVALIDACION": ["Netflow masivo a exchanges", "MVRV cayendo fuera del rango"],
        "SEÑAL_DE_CIERRE_NATURAL": {
            "descripcion": "Las métricas alcanzan zonas de sobrevaloración/distribución.",
            "señal_técnica": "MVRV > 2.5 + NUPL > 0.5 + SOPR > 1.1"
        },
        "ASIMETRIAS_DEL_ACTIVO": {
            "BTC/USDT": {"mvrv_target": 2.5},
            "ETH/USDT": {"mvrv_target": 2.2}
        }
    }
}

# =========================================================================
# 🕵️ SENIOR AUDITOR ENGINE (ACS, ACI, AEA, ACR, ATA)
# =========================================================================

class SeniorAuditor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SeniorAuditor, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        self.db_path = os.path.join(Config.DATA_DIR, "audit_chronicle.json")
        self.stale_limits = {
            "MICROSCALPING": 15,    # 15s max lag
            "SCALPING": 45,         # 45s max lag
            "SWING": 600            # 10m max lag
        }
        # Iniciar base de datos de auditoría vacía si no existe
        if not os.path.exists(self.db_path):
            try:
                os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
                with open(self.db_path, "w", encoding="utf-8") as f:
                    json.dump([], f)
            except Exception as e:
                logger.error(f"Error initializing audit chronicle DB: {e}")

    def _map_strategy_name(self, strategy_id: str) -> str:
        """Map strategy_id to standard DNA keys."""
        sid = strategy_id.upper()
        if "TFTF" in sid: return "TFTF"
        elif "OB" in sid or "SMC" in sid: return "OB_RETEST"
        elif "CASCADE" in sid or "LIQ" in sid or "LCA" in sid: return "LCA"
        elif "MEAN" in sid or "REVERSION" in sid or "MRBB" in sid: return "MRBB"
        elif "WYCKOFF" in sid: return "WYCKOFF"
        elif "VBA" in sid or "VOLATILITY" in sid or "BREAKOUT" in sid: return "VBA"
        elif "MBV" in sid or "MOMENTUM" in sid: return "MBV"
        elif "FRA" in sid or "FUNDING" in sid: return "FRA"
        elif "SC" in sid or "SENTIMENT" in sid: return "SC"
        elif "STAT" in sid or "PAIR" in sid: return "STATARB"
        elif "OCS" in sid or "CHAIN" in sid: return "OCS"
        return "TFTF"

    # ---------------------------------------------------------------------
    # 🚥 ROL 1 & 3: APERTURA (verify_opening_audit)
    # ---------------------------------------------------------------------
    def verify_opening_audit(self, intent: SignalEvent, portfolio) -> Tuple[bool, str]:
        """
        Gobernanza ACS + AEA.
        Valida que la señal de entrada coincida al 100% con su ADN y con el activo.
        """
        symbol = intent.symbol
        horizon = getattr(intent, 'horizon', 'SCALPING')
        strat_key = self._map_strategy_name(intent.strategy_id)
        
        if strat_key not in STRATEGY_DNA:
            return False, "FAIL_ACS: Strategy DNA not declared"
            
        dna = STRATEGY_DNA[strat_key]
        
        # ACS — Coherencia de Régimen
        regime = getattr(intent, 'regime', None)
        if not regime and hasattr(intent, 'metadata') and intent.metadata:
            regime = intent.metadata.get('regime')
        if not regime:
            from core.global_state import global_state
            regime = getattr(global_state, 'market_regime', 'UNKNOWN')
        regime = str(regime).upper()
        req_regime = dna["TESIS_DE_APERTURA"]["regime_requerido"]
        
        # Support both English (TREND) and Spanish (TENDENCIAL) regime names
        is_trending = "TREND" in regime or "TENDENCIAL" in regime
        is_lateral = "LATERAL" in regime or "RANGE" in regime or "CHOPPY" in regime
        is_mixed = "MIXED" in regime or "UNKNOWN" in regime
        
        # OMEGA FIX: MIXED/UNKNOWN regimes are permissive — they allow both
        # trend-following AND mean-reversion strategies because the market
        # hasn't committed to a clear direction. Only block when there's a
        # clear mismatch (e.g., mean reversion in confirmed TRENDING).
        if req_regime == "TENDENCIAL" and not is_trending and not is_mixed:
            return False, f"FAIL_ACS: Tesis requires TRENDING regime, active is {regime}"
        elif req_regime == "LATERAL" and is_trending and not is_mixed:
            return False, f"FAIL_ACS: Tesis requires RANGE/LATERAL regime, active is {regime}"
            
        # AEA — Criterio de Confianza y Sizing por Activo
        from core.asset_intelligence import get_asset_intelligence
        profile = get_asset_intelligence().get_profile(symbol)
        
        # OMEGA FIX: When Sophia is not loaded, strategies hardcode
        # ml_confidence=0.5 as fallback. This made the old chain read 0.5
        # instead of the real strength. Now we use max(strength, ml_confidence)
        # to always pick the real strategy-calculated confidence.
        _strength = getattr(intent, 'strength', 0.5)
        _ml_conf = getattr(intent, 'ml_confidence', None)
        _meta_conf = None
        if hasattr(intent, 'metadata') and intent.metadata:
            _meta_conf = intent.metadata.get('confidence')
        
        candidates = [v for v in [_strength, _ml_conf, _meta_conf] if v is not None]
        confidence = max(candidates) if candidates else 0.5
            
        if confidence < profile.min_signal_threshold:
            return False, f"FAIL_AEA: Signal confidence {confidence:.2f} below target {profile.min_signal_threshold} for asset {symbol}"
            
        # AEA — BTC Trend filter check for Tiers 2-4
        if profile.tier.value >= 2 and symbol != "BTC/USDT":
            # Obtener dirección de BTC
            btc_pos = portfolio.get_horizon_position("BTC/USDT", horizon) if portfolio else None
            if btc_pos and abs(btc_pos.get("quantity", 0.0)) > 1e-8:
                btc_dir = "LONG" if btc_pos.get("quantity", 0.0) > 0 else "SHORT"
                intent_dir = "LONG" if intent.signal_type == SignalType.LONG else "SHORT"
                if btc_dir != intent_dir:
                    return False, f"FAIL_AEA: Altcoin entry direction {intent_dir} contradicts active BTC trend {btc_dir}"
                    
        # ACS — Confirmación de indicadores críticos del ADN
        meta = getattr(intent, 'metadata', None) or {}
        if strat_key == "TFTF":
            # Pullback volume check
            vol_ratio = meta.get("pullback_volume_ratio", 0.5)
            if vol_ratio > 0.60:
                return False, f"FAIL_ACS: Pullback volume ratio {vol_ratio:.2f} exceeds 0.60 limit (reversion danger)"
        elif strat_key == "OB_RETEST":
            # OB strength check
            ob_strength = meta.get("ob_strength_atr", 1.6)
            if ob_strength < 1.5:
                return False, f"FAIL_ACS: OB strength {ob_strength:.2f} is below 1.5x ATR"
        elif strat_key == "LCA":
            # Distance to cluster check
            dist = meta.get("distance_to_cluster", 0.01)
            max_dist = 0.015 if symbol in ["BTC/USDT", "ETH/USDT"] else 0.03
            if dist > max_dist:
                return False, f"FAIL_ACS: Distance to cluster {dist:.3f} exceeds max {max_dist}"
        elif strat_key == "MRBB":
            # ADX must be < 25
            adx = meta.get("adx", 20)
            if adx >= 25:
                return False, f"FAIL_ACS: Mean reversion blocked because ADX {adx} >= 25 (trending market)"
                
        return True, "APPROVED"

    # ---------------------------------------------------------------------
    # 🚥 ROL 2 & 3: SEGUIMIENTO (verify_tracking_audit)
    # ---------------------------------------------------------------------
    def verify_tracking_audit(self, position: Dict[str, Any], data_provider, current_price: float, now: datetime) -> Tuple[int, str]:
        """
        Gobernanza ACI + AEA.
        Monitorea la posición en tiempo real para detectar ceguera de datos, features y predicciones.
        Retorna: (degradation_level, alert_reason)
        """
        symbol = position.get("symbol") or "BTC/USDT"
        horizon = position.get("horizon", "SCALPING")
        strat_key = self._map_strategy_name(position.get("opener_strategy_id", "TFTF"))
        
        # 1. Chequear ceguera de datos (Staleness check)
        stale_limit = self.stale_limits.get(horizon, 45)
        last_feed_time = position.get("last_feed_time", 0.0)
        
        if last_feed_time == 0.0 and data_provider:
            # Fallback a kline de data provider
            try:
                bars = data_provider.get_latest_bars(symbol, n=1)
                if bars is not None and len(bars) > 0:
                    last_feed_time = bars[-1].get("timestamp", 0.0)
            except:
                pass
                
        now_ts = now.timestamp()
        lag = now_ts - last_feed_time if last_feed_time > 0 else 0
        
        # Protocolo de Degradación por Ceguera
        deg_level = 0
        reason = "OK"
        
        # Level 1 (Stale): lag > limit
        if lag > stale_limit:
            deg_level = 1
            reason = f"CEGUERA_PARCIAL: Data feed lag is {lag:.1f}s (> {stale_limit}s)"
            
        # Level 2 (Disconnected): lag > 3x limit
        if lag > 3 * stale_limit:
            deg_level = 2
            reason = f"CEGUERA_CRÍTICA: Data feed disconnected! Lag is {lag:.1f}s"
            
        # Level 3 (Panic): lag > 10x limit
        if lag > 10 * stale_limit:
            deg_level = 3
            reason = f"CEGUERA_TOTAL_EMERGENCY: Closed due to extreme data feed outage! Lag: {lag:.1f}s"
            
        # 2. Chequear predicción expirada
        exp_ts = position.get("expiration_timestamp")
        if exp_ts:
            if isinstance(exp_ts, (int, float)):
                exp_ts_val = exp_ts
            else:
                exp_ts_val = exp_ts.timestamp()
                
            time_left = exp_ts_val - now_ts
            initial_validity = position.get("predicted_duration", 60)
            
            if time_left < 0:
                # La predicción expiró totalmente y no hay nueva
                if deg_level < 2:
                    deg_level = 2
                    reason = "CEGUERA_CRÍTICA: Active prediction expired and no replacement found"
            elif initial_validity > 0 and (time_left / initial_validity) < 0.20:
                # Le queda < 20% de vida, alertar alerta temprana
                if deg_level < 1:
                    deg_level = 1
                    reason = "PREDICTION_DECAY_WARNING: Prediction time left is less than 20%"
                    
        # Publicar Heartbeat de Auditoría
        self._publish_heartbeat(position, lag, deg_level, reason, now_ts)
        
        return deg_level, reason

    def _publish_heartbeat(self, pos: Dict[str, Any], lag: float, deg_level: int, reason: str, now_ts: float):
        """Registra el pulso de seguimiento en la posición para visibilidad en el event loop."""
        heartbeat = {
            "timestamp": now_ts,
            "lag_seconds": lag,
            "degradation_level": deg_level,
            "status": "VALID" if deg_level == 0 else ("WARNING" if deg_level == 1 else "STALE"),
            "reason": reason
        }
        if "tracking_heartbeats" not in pos:
            pos["tracking_heartbeats"] = []
        pos["tracking_heartbeats"].append(heartbeat)
        # Limitar historial en memoria para evitar leaks
        if len(pos["tracking_heartbeats"]) > 100:
            pos["tracking_heartbeats"] = pos["tracking_heartbeats"][-100:]

    # ---------------------------------------------------------------------
    # 🚥 ROL 1 & 4 & 5: CIERRE (verify_closing_audit)
    # ---------------------------------------------------------------------
    def verify_closing_audit(self, position: Dict[str, Any], current_price: float, data_provider, now: datetime) -> Tuple[bool, str]:
        """
        Gobernanza ACS + ACR + ATA.
        Evalúa si la tesis de apertura se ha invalidado según las reglas de su ADN.
        """
        symbol = position.get("symbol") or "BTC/USDT"
        strat_key = self._map_strategy_name(position.get("opener_strategy_id", "TFTF"))
        qty = position.get("quantity", 0.0)
        entry_price = position.get("avg_price", 0.0)
        
        if strat_key not in STRATEGY_DNA:
            return False, ""
            
        dna = STRATEGY_DNA[strat_key]
        unrealized_pnl_pct = (current_price - entry_price) / entry_price if qty > 0 else (entry_price - current_price) / entry_price
        
        # ACS — Cierre por Invalidación Estructural de Tesis
        if strat_key == "TFTF":
            # 1. ADX cae de 20
            adx_val = position.get("last_adx_value", 25)
            if adx_val < 20:
                return True, "INVALIDATION_TFTF_ADX_DROPPED_BELOW_20"
            # 2. Reversión de CVD sostenido
            cvd_streak = position.get("cvd_divergence_streak", 0)
            if cvd_streak >= 3:
                return True, "INVALIDATION_TFTF_SUSTAINED_CVD_REVERSAL"
                
        elif strat_key == "OB_RETEST":
            # 1. Cierre de vela fuera del OB
            ob_broken = position.get("ob_extremum_violated", False)
            if ob_broken:
                return True, "INVALIDATION_OB_RETEST_OB_EXTREME_VIOLATED"
                
        elif strat_key == "MRBB":
            # 1. ADX sube de 25 (reversión invalidada por tendencia fuerte)
            adx_val = position.get("last_adx_value", 20)
            if adx_val >= 25:
                return True, "INVALIDATION_MRBB_MARKET_TRENDED_ADX_ABOVE_25"
                
        elif strat_key == "LCA":
            # 1. Spike decay: LCA espera un movimiento ultra rápido. Si el precio no avanza rápido (exhaustion), sale
            entry_time = position.get("entry_time", 0.0)
            if entry_time:
                if hasattr(entry_time, "timestamp"):
                    entry_time = entry_time.timestamp()
                held = now.timestamp() - entry_time
                if held > 90 and unrealized_pnl_pct < 0.001: # 90s time stop para spikes
                    return True, "INVALIDATION_LCA_SPIKE_DECAY_EXHAUSTION"

        return False, ""

    # ---------------------------------------------------------------------
    # 🚥 ROL 5: REGISTRO Y APRENDIZAJE (log_trade_lifecycle)
    # ---------------------------------------------------------------------
    def log_trade_lifecycle(self, trade_id: str, action: str, details: Dict[str, Any]):
        """
        Gobernanza ATA.
        Escribe en logs/audit_chronicle.json un registro completo de los eventos
        de entrada (ENTRY), seguimiento (HEARTBEAT) y salida (EXIT) de cada posición.
        """
        if not trade_id:
            return
            
        import shutil
        with self._lock:
            try:
                chronicle = []
                if os.path.exists(self.db_path):
                    try:
                        with open(self.db_path, "r", encoding="utf-8") as f:
                            chronicle = json.load(f)
                    except Exception as load_err:
                        logger.warning(f"⚠️ [SeniorAuditor] Database corrupted, resetting: {load_err}")
                        # Backup corrupted file
                        corrupted_backup = self.db_path + f".corrupted.{int(time.time())}"
                        try:
                            shutil.copyfile(self.db_path, corrupted_backup)
                            logger.info(f"💾 Corrupted chronicle backed up to: {corrupted_backup}")
                        except Exception as backup_err:
                            logger.error(f"❌ Failed to backup corrupted chronicle: {backup_err}")
                        chronicle = []
                
                # Crear o buscar la bitácora del trade
                trade_log = None
                for entry in chronicle:
                    if entry.get("trade_id") == trade_id:
                        trade_log = entry
                        break
                        
                if not trade_log:
                    trade_log = {
                        "trade_id": trade_id,
                        "events": []
                    }
                    chronicle.append(trade_log)
                    
                event_entry = {
                    "timestamp": time.time(),
                    "action": action,
                    "details": self._sanitize_for_json(details)
                }
                trade_log["events"].append(event_entry)
                
                # Guardar base de datos de manera atómica (temp file + rename)
                temp_path = self.db_path + ".tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(chronicle, f, indent=4, default=str)
                os.replace(temp_path, self.db_path)
                    
                logger.info(f"💾 [AUDIT CHRONICLE] Recorded action '{action}' for trade {trade_id}")
            except Exception as e:
                logger.error(f"Error saving to audit chronicle: {e}")

    def _sanitize_for_json(self, obj):
        """Recursively sanitize objects to make them JSON serializable."""
        import numpy as np
        if isinstance(obj, dict):
            # Ensure all keys are strings
            return {str(k): self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, set):
            return [self._sanitize_for_json(i) for i in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            # Convert numpy scalars to native python types
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return self._sanitize_for_json(obj.tolist())
        elif hasattr(obj, "to_dict"):
            return self._sanitize_for_json(obj.to_dict())
        elif hasattr(obj, "__dict__"):
            return self._sanitize_for_json(obj.__dict__)
        else:
            return obj
