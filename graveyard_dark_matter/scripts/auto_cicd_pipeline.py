#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 AUTO-CICD PIPELINE ENGINE (Phase 20)
=======================================

PROFESSOR METHOD:
- QUÉ: Un daemon automatizado de Integración y Mejora Continua (Auto-CICD).
- POR QUÉ: Para escalar una cuenta de $13, el sistema debe mutar y testearse 
  autónomamente mientras dormimos. La optimización manual es lenta e imprecisa.
- PARA QUÉ: Ejecutar un bucle infinito: [Backtest -> Diagnóstico Forense -> Ajuste -> Validación].
- CÓMO: 
  1. Ejecuta `run_god_mode_backtest.py`.
  2. Parsea `dashboard/data/backtest_temp/bt_trades.csv` o su log.
  3. Evalúa si supera los umbrales (Sharpe > 2.5, DD < 1.2%, WR > 60%).
  4. Si falla, invoca a `nemesis.py` / Evolución y repite.
- CUÁNDO: Ejecutado como un proceso en segundo plano (daemon).
- DÓNDE: scripts/auto_cicd_pipeline.py
- QUIÉN: Orquestador Maestro de Mejora Continua.
"""

import sys
import os
import time
import json
import subprocess
import pandas as pd
from datetime import datetime, timezone
import argparse

# Configuración del Entorno Virtual (asegura correr con las dependencias correctas)
VENV_PYTHON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.venv', 'Scripts', 'python.exe')
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = "python"  # Fallback to system python

BACKTEST_SCRIPT = "scripts/run_god_mode_backtest.py"
TRADES_FILE = "dashboard/data/backtest_temp/bt_trades.csv"

# --- UMERALES DE ÉXITO ESTRICTOS (GOD MODE) ---
TARGET_SHARPE = 2.5
MAX_DRAWDOWN = 1.2  # %
MIN_WIN_RATE = 60.0 # %

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] 🔄 [AUTO-CICD] {msg}")

def run_backtest() -> bool:
    """Ejecuta el backtest masivo de forma silenciosa, capturando resultados."""
    log(f"Iniciando ciclo de backtest con {BACKTEST_SCRIPT}...")
    try:
        # Run backtest synchronously
        result = subprocess.run(
            [VENV_PYTHON, BACKTEST_SCRIPT, "--quiet"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            log(f"❌ Error crítico en ejecución del backtest:\n{result.stderr[-500:]}")
            return False
        log("✅ Backtest completado con éxito.")
        return True
    except Exception as e:
        log(f"❌ Excepción al lanzar backtest: {e}")
        return False

def analyze_results() -> dict:
    """Auditoría forense sobre los resultados del backtest."""
    log("Iniciando análisis forense de métricas...")
    
    if not os.path.exists(TRADES_FILE):
        log(f"⚠️ Archivo de trades no encontrado: {TRADES_FILE}")
        return {}

    try:
        total_trades = 0
        wins = 0
        pnls = []
        
        with open(TRADES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                # Try to find PnL by parsing numeric values after column 5
                pnl_val = 0.0
                has_pnl = False
                for p in parts[6:]:
                    try:
                        v = float(p)
                        # Identify likely PnL (not price or quantity)
                        if abs(v) < 1000 and abs(v) > 0.0001: 
                            pnl_val = v
                            has_pnl = True
                    except:
                        from utils.error_handler import SystemIntegrityError
                        raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')
                
                # Check if it was an exit line (which usually has PnL)
                if 'EXIT' in line or 'CLOSE' in line or has_pnl:
                    # Very rough heuristic, just for the test
                    if has_pnl:
                        pnls.append(pnl_val)
                        total_trades += 1
                        if pnl_val > 0:
                            wins += 1

        if total_trades == 0:
            log("⚠️ El CSV de trades no contiene operaciones cerradas parseables.")
            return {"total_trades": 0}

        win_rate = (wins / total_trades) * 100
        
        pnls_arr = np.array(pnls)
        total_pnl = pnls_arr.sum()
        
        # Sharpe estandar simplificado (diario/barras)
        mean_pnl = pnls_arr.mean()
        std_pnl = pnls_arr.std()
        sharpe = (mean_pnl / std_pnl * np.sqrt(len(pnls_arr))) if std_pnl > 0 else 0.0
        
        # Drawdown simplificado
        cum_pnl = pnls_arr.cumsum()
        running_max = np.maximum.accumulate(cum_pnl)
        drawdown = running_max - cum_pnl
        max_dd_usd = drawdown.max() if len(drawdown) > 0 else 0.0
        # Aproximado % relativo al capital inicial de 13 USD
        max_dd_pct = (max_dd_usd / 13.0) * 100

        metrics = {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd_pct,
            "total_pnl": total_pnl
        }
        
        log(f"📊 Métricas extraídas: WR={win_rate:.1f}% | Sharpe={sharpe:.2f} | DD={max_dd_pct:.2f}% | PnL=${total_pnl:.2f}")
        return metrics

    except Exception as e:
        log(f"❌ Error en análisis de resultados: {e}")
        return {}

def check_success_criteria(metrics: dict) -> bool:
    """Verifica si las métricas cumplen los estrictos umbrales de God Mode."""
    if not metrics or metrics["total_trades"] < 5:
        log("⛔ Falla: Muestra estadística insuficiente (menos de 5 trades).")
        return False
        
    passed = True
    
    if metrics["sharpe_ratio"] < TARGET_SHARPE:
        log(f"⛔ Falla: Sharpe Ratio ({metrics['sharpe_ratio']:.2f}) < Umbral ({TARGET_SHARPE}).")
        passed = False
        
    if metrics["max_drawdown"] > MAX_DRAWDOWN:
        log(f"⛔ Falla: Max Drawdown ({metrics['max_drawdown']:.2f}%) > Umbral ({MAX_DRAWDOWN}%).")
        passed = False
        
    if metrics["win_rate"] < MIN_WIN_RATE:
        log(f"⛔ Falla: Win Rate ({metrics['win_rate']:.1f}%) < Umbral ({MIN_WIN_RATE}%).")
        passed = False
        
    return passed

def evolve_parameters(metrics: dict):
    """
    Motor Auto-CICD Avanzado: Muta los genotipos cuando las métricas fallan.
    Invoca a EvolutionEngine y aplica gaussian mutations a los parámetros.
    """
    log("🧠 [NEMESIS] Generando hipótesis evolutiva para ajustar parámetros...")
    
    from core.evolution import EvolutionEngine
    from core.genotype import Genotype
    from core.gene_bank import gene_bank
    import numpy as np

    # Instanciar motor de evolución
    engine = EvolutionEngine(mutation_rate=0.5, mutation_strength=0.3)
    
    # Extraer todos los símbolos configurados
    from config import Config
    symbols = Config.Trading.SYMBOLS
    
    for symbol in symbols:
        # Cargar el genotipo "élito" actual o crear uno por defecto
        current_genes = gene_bank.get_best_gene(symbol, "NORMAL")
        
        genotype = Genotype(symbol=symbol)
        if current_genes:
            genotype.genes = current_genes.copy()
            
        # Determinar qué gen mutar en base a la falla específica
        if metrics["max_drawdown"] > MAX_DRAWDOWN:
            log(f"   👉 [MUTACIÓN] Reduciendo SL_MULTIPLIER para {symbol} por Drawdown Alto.")
            if "sl_pct" in genotype.genes:
                genotype.genes["sl_pct"] *= 0.8  # Hacer SL más estricto
                
        if metrics["win_rate"] < MIN_WIN_RATE:
            log(f"   👉 [MUTACIÓN] Ajustando Confirmaciones (RSI/Bollinger) para {symbol} por WR Bajo.")
            if "rsi_overbought" in genotype.genes:
                genotype.genes["rsi_overbought"] = min(95, genotype.genes["rsi_overbought"] + 2)
            if "rsi_oversold" in genotype.genes:
                genotype.genes["rsi_oversold"] = max(5, genotype.genes["rsi_oversold"] - 2)
                
        # Mutación General Aleatoria para no estancarse en mínimos locales
        mutated_genotype = engine.mutate(genotype)
        mutated_genotype.fitness_score = -1.0 # Reset fitness para el próximo backtest
        
        # Guardar en el banco de genes (Archetype Memory)
        gene_bank.save_elite_gene(mutated_genotype, "NORMAL")
        
    log("🧬 Mutación Genética Global completada. Listos para el próximo ciclo de Backtest.")

def main():
    parser = argparse.ArgumentParser(description="Auto-CICD Pipeline Engine")
    parser.add_argument("--test-run", action="store_true", help="Ejecuta solo 1 ciclo de prueba.")
    args = parser.parse_args()

    import numpy as np # Ensure numpy is available for stats
    
    log("🚀 INICIANDO MOTOR DE MEJORA CONTINUA (AUTO-CICD)")
    cycle = 1
    
    while True:
        log(f"\n{'='*50}\n🌟 INICIANDO CICLO {cycle}\n{'='*50}")
        
        success = run_backtest()
        if success:
            metrics = analyze_results()
            if check_success_criteria(metrics):
                log("🏆 ¡ÉXITO! Las métricas superan los umbrales de Producción (God Mode).")
                log("🚀 Despliegue recomendado a Paper Trading (Sandbox).")
                break
            else:
                log("⚠️ Los umbrales no se superaron. Se requiere evolución genética.")
                evolve_parameters(metrics)
        else:
            log("❌ Ciclo abortado por error de ejecución.")
            
        if args.test_run:
            log("🛑 Test-run completado. Saliendo del Auto-CICD.")
            break
            
        log("💤 Esperando 10 segundos antes del próximo ciclo mutacional...")
        time.sleep(10)
        cycle += 1

if __name__ == "__main__":
    main()
