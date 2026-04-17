"""
🧪 VALIDACIÓN DE EVOLUCIÓN GENÉTICA (G-001)
Este script verifica independientemente la persistencia y la mutación neuronal
del Genotype y "Cognitive Memory" en HybridScalpingStrategy, sin necesidad de conectarse
a Binance o consumir datos de mercado pesados.
"""

import os
import sys
import numpy as np
from datetime import datetime, timezone
import queue
import time
import json

# Setup system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.technical import HybridScalpingStrategy
from core.genotype import Genotype
from utils.logger import logger

def run_genotype_evolution_validation():
    print("\n" + "="*50)
    print("🧬 INICIANDO VALIDACIÓN DE EVOLUCIÓN GENÉTICA")
    print("="*50)
    
    # Directorio de genotipos
    genotype_dir = "data/genotypes"
    os.makedirs(genotype_dir, exist_ok=True)
    symbol = "BTC/USDT"
    setup = "MOMENTUM"
    
    # 1. Borrar genotipo anterior si existe para tener estado limpio
    fp = f"{genotype_dir}/{symbol.replace('/','')}_gene.json"
    if os.path.exists(fp):
        os.remove(fp)
        
    print(f"[1] Entorno limpiado. Inicializando Estrategia para {symbol}...")
    
    # 2. Inicializar Motor Tecnico
    events_queue = queue.Queue()
    strategy = HybridScalpingStrategy(data_provider=None, events_queue=events_queue, horizon="SCALPING", priority=0)
    
    # 3. Forzar Carga de Parámetros (Automáticamente instancia Genotype y Brain Weights)
    params = strategy.get_symbol_params(symbol)
    if symbol not in strategy.genotypes:
        print("❌ CRITICAL: No se generó Genotype para el símbolo.")
        return False
        
    initial_genotype = strategy.genotypes[symbol]
    
    # Cargar matriz cerebral simulada (25 Entradas, 4 Acciones)
    brain_weights = initial_genotype.genes.get('brain_weights')
    initial_tp = initial_genotype.genes.get('tp_pct')
    print(f"[2] Genotype Base generado. TP Inicial: {initial_tp:.4f}.")
    print(f"    Brain Weights Count: {len(brain_weights) if brain_weights else 0}")
    
    if not brain_weights or len(brain_weights) != 100:
        print("❌ CRITICAL: Brain weights no inicializados en (25x4)=100.")
        return False
        
    print("\n[3] 💥 INFUNDIENDO RACHA PERDEDORA (Dolor Asimétrico)...")
    # Para poder aplicar Online Learning necesitamos injectar memoria primero simulando una predicción
    # 25 entradas random
    mock_state = np.random.uniform(0, 1, 25)
    mock_prediction = 0.8
    action_idx = 1 # Supongamos Long Action
    
    strategy.brain_memory[symbol] = {
        'state': mock_state,
        'prediction': mock_prediction,
        'weights': np.array(brain_weights),
        'action_idx': action_idx
    }
    
    # Simulamos 5 pérdidas
    for i in range(5):
        trade = {
            'symbol': symbol,
            'pnl_usd': -5.0,
            'metadata': {'setup_type': setup}
        }
        strategy.process_reward(trade)
        
    # Verificar Cognitive State
    cog_state = strategy.cognitive_memory[symbol][setup]['state']
    print(f"    Estado Cognitivo Actual: {cog_state} (Debe ser INJURED)")
    if cog_state != 'INJURED':
         print(f"❌ CRITICAL: Memoria cognitiva falló al registrar INJURED state.")
         return False

    # Guardar estado actual
    strategy.stop() # Llama a la persistencia en technical.py (guarda a JSON)
    time.sleep(0.5)
    
    print("\n[4] 🔄 RECARGANDO MOTOR Y MEMORIA (Prueba de Amnesia)...")
    strategy_2 = HybridScalpingStrategy(data_provider=None, events_queue=events_queue, horizon="SCALPING", priority=0)
    params_2 = strategy_2.get_symbol_params(symbol)
    
    second_genotype = strategy_2.genotypes.get(symbol)
    reloaded_tp = second_genotype.genes.get('tp_pct')
    reloaded_weights = second_genotype.genes.get('brain_weights')
    
    # Extraer pesos actualizados (deberían ser diferentes al initial_genotype)
    w_sum_initial = sum(brain_weights)
    w_sum_reloaded = sum(reloaded_weights)
    
    print(f"    Suma de Pesos Inicial: {w_sum_initial:.4f}")
    print(f"    Suma de Pesos Aprendida: {w_sum_reloaded:.4f}")
    print(f"    TP Recargado vs TP Inicial: {reloaded_tp:.4f} == {initial_tp:.4f}")
    
    if abs(w_sum_initial - w_sum_reloaded) < 0.0001:
        print("❌ CRITICAL: Los pesos de la red neuronal no sufrieron mutación (Online Learning fallback).")
        return False
        
    if not os.path.exists(fp):
        print(f"❌ CRITICAL: El archivo genético no fue guardado en: {fp}")
        return False

    print("\n[5] 🏆 INFUNDIENDO RACHA GANADORA (Estado ALPHA)...")
    strategy_2.brain_memory[symbol] = {
        'state': mock_state,
        'prediction': mock_prediction,
        'weights': np.array(reloaded_weights),
        'action_idx': action_idx
    }
    
    # Simulamos 10 ganancias para purgar el memory ring de 8
    for i in range(10):
         trade = {
             'symbol': symbol,
             'pnl_usd': 10.0,
             'metadata': {'setup_type': setup}
         }
         strategy_2.process_reward(trade)
         
    cog_state_2 = strategy_2.cognitive_memory[symbol][setup]['state']
    print(f"    Estado Cognitivo Evolucionado: {cog_state_2} (Debe ser ALPHA)")
    if cog_state_2 != 'ALPHA':
         print(f"❌ CRITICAL: Memoria cognitiva no alcanzó estado ALPHA tras rachas positivas.")
         return False

    print("\n==================================================")
    print("✅ VALIDACIÓN O.K. - EVOLUCIÓN (METAMORFOSIS) ACTIVA")
    print("==================================================\n")
    return True

if __name__ == "__main__":
    success = run_genotype_evolution_validation()
    sys.exit(0 if success else 1)
