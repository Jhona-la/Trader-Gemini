import numpy as np
import logging
import json
import os
from typing import Dict, Any

# QUÉ: NanoRLAgent - Agente de Reinforcement Learning ultraligero y continuo.
# POR QUÉ: Permite al sistema aprender en tiempo real del PnL en Paper Trading/Live, sin GPUs ni backtests pesados.
# PARA QUÉ: Adaptar dinámicamente los genotipos (Umbrales técnicos, SL, TP) según el dolor o la euforia del mercado.
# CÓMO: Policy Gradient Heurístico. Transforma el Reward (PnL) en Gradientes de Mutación.
# CUÁNDO: Se invoca en cada actualización de mercado (Tick) y en cada cierre de posición (Fill).
# DÓNDE: core/evolution/rl_agent.py
# QUIÉN: QA Engineer & Quant Developer

logger = logging.getLogger("NanoRLAgent")

class NanoRLAgent:
    """
    Continuous CPU-Optimized Reinforcement Learning Agent.
    Aplica ascensión de gradiente estocástico sobre los Genotipos en memoria.
    """
    def __init__(self, genome_registry, memory_path=".models/rl_memory.json"):
        self.registry = genome_registry
        self.memory_path = memory_path
        self.learning_rate = 0.05  # Tasa de aprendizaje hiper conservadora
        
        # Meta-Memoria: Guarda la tendencia de recompensas recientes por símbolo
        self.reward_history: Dict[str, list] = {}
        
        # Diccionario para persistir el "Dolor acumulado" o "Euforia"
        self.rl_state: Dict[str, float] = {}
        self._load_memory()
        
        logger.info("🧠 [NanoRL] Reinforcement Learning Agent Inicializado (CPU-Optimized).")

    def _load_memory(self):
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r') as f:
                    self.rl_state = json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ [NanoRL] Failed to load memory: {e}")

    def _save_memory(self):
        try:
            with open(self.memory_path, 'w') as f:
                json.dump(self.rl_state, f)
        except Exception:
            from utils.error_handler import SystemIntegrityError
            raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')

    def apply_tick_penalty(self, symbol: str, horizon: str, unrealized_pnl: float, duration_seconds: int):
        """
        [Recompensa Continua / Tick-a-Tick]
        Aplica estrés evolutivo si el trade lleva mucho tiempo en negativo (Zombie Trade).
        """
        if duration_seconds < 60:
            return # No castigamos el ruido inicial (1 minuto de gracia)
            
        # Si estamos sangrando dinero pasivamente (Unrealized PnL < 0)
        if unrealized_pnl < -0.01: # -1% flotante
            punishment = unrealized_pnl * (duration_seconds / 3600.0) # Escala por el tiempo atrapado
            self._apply_gradient(symbol, horizon, reward=punishment, is_tick=True)

    def process_trade_closure(self, symbol: str, horizon: str, net_pnl: float, trade_metadata: Dict[str, Any]):
        """
        [Recompensa Discreta / Cierre de Trade]
        Se dispara cuando una posición se cierra oficialmente.
        """
        if symbol not in self.reward_history:
            self.reward_history[symbol] = []
            
        self.reward_history[symbol].append(net_pnl)
        if len(self.reward_history[symbol]) > 20:
            self.reward_history[symbol].pop(0)
            
        # El PnL neto es directamente el Reward base
        reward = net_pnl
        
        # Modificadores cognitivos
        exit_reason = trade_metadata['exit_reason']
        if exit_reason == "STOP_LOSS":
            reward -= abs(reward) * 0.5 # Castigo extra por golpear el SL duro
        elif exit_reason == "TAKE_PROFIT":
            reward += abs(reward) * 0.2 # Bono por golpear el objetivo perfecto
            
        self._apply_gradient(symbol, horizon, reward=reward, is_tick=False)
        self._save_memory()

    def _apply_gradient(self, symbol: str, horizon: str, reward: float, is_tick: bool):
        """
        Calcula las derivadas parciales (Heurística de Gradiente) y aplica mutaciones al Genome.
        """
        if abs(reward) < 0.0001:
            return
            
        # Recuperamos el Genotipo actual
        genome = self.registry.get_genes(symbol, horizon)
        if not genome:
            return
            
        gradients = {}
        
        # STATE ENCODING: Determinamos si fue una Victoria (Euforia) o Pérdida (Dolor)
        if reward > 0:
            # 🟢 REWARD POSITIVO: El trade funcionó.
            # Queremos aflojar sutilmente los filtros para que tome más oportunidades como esta.
            # Y acercar el TP/SL a los valores óptimos.
            
            # Si el RSI de compra estaba en 30, subirlo a 30.1 nos dará entradas más fáciles
            gradients['rsi_buy'] = 0.1 * self.learning_rate
            gradients['rsi_sell'] = -0.1 * self.learning_rate
            
            # Aumentar muy ligeramente el tamaño del SL/TP (Confianza)
            gradients['atr_tp_mult'] = 0.05 * self.learning_rate
            
        else:
            # 🔴 REWARD NEGATIVO (PUNISHMENT): El trade fracasó o está sangrando.
            # Queremos APRETAR los filtros (volverse más estricto/Conservador)
            
            # Reducir el RSI de compra (hacerlo más sobrevendido, ej 30 -> 29.5)
            gradients['rsi_buy'] = -0.5 * self.learning_rate # Mayor peso al dolor
            gradients['rsi_sell'] = 0.5 * self.learning_rate
            
            # Reducir el SL (Tighten stop) si fue un cierre de pérdida
            if not is_tick:
                gradients['atr_sl_mult'] = -0.1 * self.learning_rate
                
            # Si el mercado está súper tóxico, subimos los multiplicadores de BB
            gradients['bb_std'] = 0.05 * self.learning_rate
            
        # Log del aprendizaje
        action_str = "RELAXING" if reward > 0 else "TIGHTENING"
        if not is_tick:
            logger.info(f"🧠 [NanoRL] {symbol} {horizon} | Reward: {reward:.4f} | Action: {action_str} DNA thresholds.")

        # Aplicar los gradientes al registro centralizado
        self.registry.apply_rl_gradient(symbol, horizon, gradients)
