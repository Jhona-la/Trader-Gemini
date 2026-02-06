"""
🧠 THE SOVEREIGN META-BRAIN - Strategy Selector
QUÉ: Módulo de meta-cognición que decide qué estrategia priorizar.
POR QUÉ: Los regímenes de mercado cambian; una estrategia que ganó ayer puede perder hoy.
PARA QUÉ: Maximizar el Expected Value (EV) seleccionando la herramienta adecuada.
CÓMO: Ejecuta simulaciones rápidas (mocking) y pondera con resultados reales del Portfolio.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from utils.logger import logger
from config import Config
from core.enums import SignalType

class StrategySelector:
    def __init__(self, portfolio=None, data_provider=None):
        self.portfolio = portfolio
        self.data_provider = data_provider
        self.strategy_health = {} # {strategy_id: {'score': 0.0, 'rank': 1}}
        self.last_update = None
        self.update_interval_hours = 2
        
        # Strategies to monitor
        self.strategies_pool = [
            'TECHNICAL', 'ML_XGBOOST', 'PATTERN_RECOGNITION', 
            'SNIPER_MOMENTUM', 'STATISTICAL_PAIRS'
        ]

    def update_strategy_rankings(self):
        """
        Main loop for the Meta-Brain.
        Combines Sim Results + Real Portfolio Results.
        """
        logger.info("🧠 [Meta-Brain] Starting real-time strategy re-evaluation...")
        
        rankings = {}
        for strat in self.strategies_pool:
            # 1. Get Real Performance from Portfolio
            real_perf = self._get_real_performance(strat)
            
            # 2. Run Mini-Simulation (Mocking)
            sim_perf = self._run_mini_sim(strat)
            
            # 3. Blended Score (70% Real / 30% Sim)
            # Sim helps predict future, Real confirms past reliability.
            blended_score = (real_perf * 0.7) + (sim_perf * 0.3)
            rankings[strat] = blended_score
            
        # Normalize and Rank
        sorted_ranks = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        
        self.strategy_health = {
            strat: {'score': score, 'rank': i+1} 
            for i, (strat, score) in enumerate(sorted_ranks)
        }
        
        self.last_update = datetime.now(timezone.utc)
        logger.info(f"🏆 [Meta-Brain] New Strategy Ranking: {self.strategy_health}")
        
        # Sync with Portfolio for Dashboard/Oracle visibility
        if self.portfolio:
            self.portfolio.strategy_rankings = self.strategy_health

    def _get_real_performance(self, strategy_id) -> float:
        """Fetch win rate and profit factor from Portfolio."""
        if not self.portfolio: return 0.5 # Neutral
        
        perf = self.portfolio.strategy_performance.get(strategy_id)
        if not perf or perf['trades'] < 5:
            return 0.5 # Neutral for new strategies
            
        wr = perf['wins'] / perf['trades']
        # Profit factor or expectancy would be better, but WR is a good proxy for scalping
        return wr

    def _run_mini_sim(self, strategy_id) -> float:
        """
        Perform a high-speed 'mock' backtest on the last 100 bars.
        Simulates how the strategy would have performed 'right now'.
        """
        try:
            # For brevity in this implementation, we use a simplified proxy.
            # In production, this would call strategy.calculate_signals() over a loop of bars.
            # Here we return a confidence score based on recent indicator alignment.
            
            # TODO: Integrate full event-loop simulation for each strategy
            # For now: placeholder logic that favors strategies based on volatility
            # (Statistical likes Range, ML likes Trend, etc.)
            
            # Simplified Logic:
            # We fetch BTC data as a proxy for the 'current vibe'
            bars = self.data_provider.get_latest_bars('BTC/USDT', n=100)
            if not bars: return 0.5
            
            closes = np.array([b['close'] for b in bars])
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns)
            
            # Simple heuristic mapping for the "Brain"
            if volatility > 0.005: # High Vol
                if strategy_id == 'ML_XGBOOST': return 0.8
                if strategy_id == 'SNIPER_MOMENTUM': return 0.7
                return 0.4
            else: # Low Vol / Range
                if strategy_id == 'STATISTICAL_PAIRS': return 0.8
                if strategy_id == 'PATTERN_RECOGNITION': return 0.6
                return 0.5
                
        except Exception as e:
            logger.error(f"Sim Error for {strategy_id}: {e}")
            return 0.5

    def get_strategy_multiplier(self, strategy_id) -> float:
        """
        Returns a weight multiplier for the RiskManager.
        Top Rank = 1.2x sizing/priority
        Bottom Rank = 0.5x sizing/priority
        """
        health = self.strategy_health.get(strategy_id, {'rank': 3})
        rank = health['rank']
        
        if rank == 1: return 1.2   # Boost winner
        if rank == 2: return 1.0   # Standard
        if rank == 3: return 0.8   # Cautious
        return 0.5                 # Drastic reduction for losers

    def get_governance_advice(self) -> dict:
        """
        PROFESSOR METHOD:
        QUÉ: Opinión experta del Meta-Brain sobre los límites operativos actuales.
        POR QUÉ: Para que el usuario entienda por qué el bot se auto-limita.
        """
        total_health = sum(d['score'] for d in self.strategy_health.values()) / max(1, len(self.strategy_health))
        
        advice = {
            "status": "NORMAL",
            "message": "Límites estándar operacionales (15/símbolo, 100/total).",
            "concurrency_target": Config.MAX_CONCURRENT_POSITIONS
        }
        
        if total_health < 0.35:
            advice["status"] = "DEFENSIVE"
            advice["message"] = "Salud estratégica baja. Se recomienda reducir límites a 5/símbolo."
            advice["concurrency_target"] = 1
        elif total_health > 0.75:
            advice["status"] = "AGGRESSIVE"
            advice["message"] = "Alta confianza estratégica. El sistema puede manejar mayor carga."
            advice["concurrency_target"] = min(5, Config.MAX_CONCURRENT_POSITIONS + 1)
            
        return advice

    def should_allow_trade(self, strategy_id) -> bool:
        """Global veto power for the Meta-Brain."""
        health = self.strategy_health.get(strategy_id, {'score': 0.5})
        # If a strategy is performing horribly (< 0.3 blended score), block it
        if health['score'] < 0.3:
            logger.warning(f"🧠 [Meta-Brain] VETO: Strategy {strategy_id} performing poorly (Score: {health['score']:.2f})")
            return False
        return True
