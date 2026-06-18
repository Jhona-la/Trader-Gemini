import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.quantum_engine import QuantumEngine
from utils.logger import setup_logger

logger = setup_logger("forensic_auditor_v2")

def run_forensic_audit():
    logger.info("🔍 INICIANDO AUDITORÍA FORENSE MASIVA V2 (ANTI-COLISIÓN)...")
    
    engine = QuantumEngine(capital=13.0, horizon="BOTH")
    engine.load_data(days=5) # 5 días son suficientes para detectar superposiciones de alta frecuencia
    
    # Obtenemos el mejor ADN (Santo Grial actual)
    import json
    try:
        with open('.models/real_quantum_dna.json', 'r') as f:
            dna = json.load(f)
    except:
        logger.error("No se encontró ADN. Ejecuta hyper_evolver primero.")
        return
        
    logger.info("🧬 ADN Cargado. Evaluando Señales de Micro-Scalping vs Swing...")
    
    # Extraemos señales brutas antes de resolver trades
    # Modificamos temporalmente el motor para que nos devuelva las matrices de decisión
    
    # Vamos a usar la pre-calculación de quantum_engine directamente
    df_1m = engine.data_1m
    df_1h = engine.data_1h
    
    # No todas las barras 1m coinciden con 1h, usamos timestamps alineados
    logger.info("⚡ Simulando OMNI-Matrices...")
    
    collision_count = 0
    margin_burn_fees = 0.0
    
    # Mock analysis: En la vida real la colisión se detecta en vivo, pero aquí la buscamos estadísticamente.
    # Simularemos la extracción de long_cond y short_cond
    
    for symbol, df_scalp in df_1m.items():
        if symbol not in df_1h: continue
        
        df_swing = df_1h[symbol]
        
        # Extraemos las señales reales:
        # Los datos son diccionarios de arrays numpy
        close_scalp = np.asarray(df_scalp['close'])
        close_swing = np.asarray(df_swing['close'])
        
        # Import RSI calculator to handle raw OHLCV
        from utils.math_kernel import calculate_rsi_jit
        
        rsi_scalp = calculate_rsi_jit(close_scalp, 14)
        rsi_swing = calculate_rsi_jit(close_swing, 14)
        
        # Mocking señales (simulando los umbrales del ADN)
        # Scalping
        tech_long_scalp = (rsi_scalp < dna['scalp_rsi_buy']).astype(int)
        tech_short_scalp = (rsi_scalp > dna['scalp_rsi_sell']).astype(int)
        # Swing
        tech_long_swing = (rsi_swing < dna['swing_rsi_buy']).astype(int)
        tech_short_swing = (rsi_swing > dna['swing_rsi_sell']).astype(int)
        
        # Merge by exact minute timestamp
        # Converting to DataFrame ONLY for the merge/reindexing logic
        df_scalp_pd = pd.DataFrame({
            'timestamp': df_scalp['timestamp'],
            'close': close_scalp,
            'scalp_long': tech_long_scalp,
            'scalp_short': tech_short_scalp
        }).set_index('timestamp')
        
        df_swing_pd = pd.DataFrame({
            'timestamp': df_swing['timestamp'],
            'swing_long': tech_long_swing,
            'swing_short': tech_short_swing
        }).set_index('timestamp')
        
        df_swing_reindexed = df_swing_pd.reindex(df_scalp_pd.index).ffill().fillna(0)
        
        df_merged = df_scalp_pd.copy()
        df_merged['swing_long'] = df_swing_reindexed['swing_long']
        df_merged['swing_short'] = df_swing_reindexed['swing_short']
        
        # Colisiones: Scalp LONG y Swing SHORT
        col_1 = df_merged[(df_merged['scalp_long'] == 1) & (df_merged['swing_short'] == 1)]
        col_2 = df_merged[(df_merged['scalp_short'] == 1) & (df_merged['swing_long'] == 1)]
        
        symbol_cols = len(col_1) + len(col_2)
        collision_count += symbol_cols
        
        # Si ocurre una colisión en Binance, ambos trades consumen comisiones del 0.1% = 0.2% quemado por nada
        margin_burn_fees += (symbol_cols * 0.002 * 13.0) 
        
    logger.info(f"🚨 REPORTE DE AUDITORÍA FORENSE 🚨")
    logger.info(f"   - Colisiones Detectadas (Long vs Short simultáneo): {collision_count}")
    logger.info(f"   - Fuga de Capital Teórica por Colisión: ${margin_burn_fees:.4f} USD")
    
    if collision_count > 0:
        logger.warning("⚠️ DIAGNÓSTICO: 'PISADA DE PATAS' CONFIRMADA.")
        logger.warning("   -> RESOLUCIÓN RECOMENDADA: El Execution Manager debe implementar Netting de posiciones.")
        logger.warning("   -> Si Scalp=LONG y Swing=SHORT, el sistema debe NEUTRALIZAR o priorizar el de mayor Expected Value.")
    else:
        logger.info("✅ Ecosistema Óptimo. No hay colisiones destructivas en el horizonte actual.")

if __name__ == "__main__":
    run_forensic_audit()
