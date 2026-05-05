"""
AITS Phase 3: Swarm Intelligence
Evaluates signals from predictive ML models (LSTM/XGBoost) and scales their confidence 
up or down based on systemic shocks propagating through the Liquidity Hypergraph.

Example: If LSTM says "Buy ETH" but Neo4j/NetworkX shows a massive -5.0 shock radiating 
from BTC, Swarm Intelligence will veto or scale down the ETH buy signal.
"""

import logging
from typing import Dict, Any

from aits_research.neo4j_graph_builder import AITSGraphBuilder, URI, AUTH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class SwarmIntelligence:
    def __init__(self):
        self.graph = AITSGraphBuilder(URI, AUTH)
        self.graph.initialize_schema()
        self.graph.seed_initial_ontology()
        
        # In a real scenario, this would be continuously updated by the Event/Data layer
        self.active_shocks: Dict[str, float] = {}

    def register_ecosystem_shock(self, source_symbol: str, magnitude: float):
        """
        Registers an active shock (e.g., massive liquidation event) and computes its
        propagation across the entire ecosystem.
        Magnitude: -10 to 10 (Negative = sell pressure, Positive = buy pressure)
        """
        logging.warning(f"🐝 SWARM ALERT: Detecting ecosystem shock originating from {source_symbol} (Mag: {magnitude})")
        propagated_impacts = self.graph.propagate_shock(source_symbol, magnitude)
        
        # Merge with active shocks
        for sym, impact in propagated_impacts.items():
            self.active_shocks[sym] = self.active_shocks.get(sym, 0.0) + impact
            
        self.active_shocks[source_symbol] = magnitude
        return propagated_impacts

    def evaluate_signal_context(self, symbol: str, predicted_direction: str, raw_confidence: float) -> float:
        """
        Adjusts raw ML confidence based on the current Swarm topology state.
        Returns the adjusted confidence.
        """
        current_shock = self.active_shocks.get(symbol, 0.0)
        
        if current_shock == 0.0:
            return raw_confidence

        adjusted_confidence = raw_confidence
        
        if predicted_direction == "UP":
            if current_shock < -1.0:
                # Disagreement: Model says UP, but Ecosystem is pulling DOWN hard
                penalty = abs(current_shock) * 0.1
                adjusted_confidence -= penalty
                logging.info(f"🐝 SWARM VETO: {symbol} UP signal penalized by {penalty:.2f} due to negative systemic shock.")
            elif current_shock > 1.0:
                # Agreement: Model says UP, Ecosystem is pushing UP
                bonus = current_shock * 0.05
                adjusted_confidence += bonus
                logging.info(f"🐝 SWARM BOOST: {symbol} UP signal boosted by {bonus:.2f} due to positive systemic shock.")
        else: # DOWN
            if current_shock > 1.0:
                # Disagreement
                penalty = current_shock * 0.1
                adjusted_confidence -= penalty
                logging.info(f"🐝 SWARM VETO: {symbol} DOWN signal penalized by {penalty:.2f} due to positive systemic shock.")
            elif current_shock < -1.0:
                # Agreement
                bonus = abs(current_shock) * 0.05
                adjusted_confidence += bonus
                logging.info(f"🐝 SWARM BOOST: {symbol} DOWN signal boosted by {bonus:.2f} due to negative systemic shock.")
                
        return min(max(adjusted_confidence, 0.0), 1.0) # Clamp between 0 and 1

if __name__ == "__main__":
    swarm = SwarmIntelligence()
    
    # Simulate a massive liquidation dump on BTC
    swarm.register_ecosystem_shock("BTC", -8.0)
    
    # A local ML model predicts ETH is going UP with 0.80 confidence based on local features
    raw_eth_conf = 0.80
    adj_eth_conf = swarm.evaluate_signal_context("ETH", "UP", raw_eth_conf)
    
    logging.info(f"ETH Local Model Confidence: {raw_eth_conf:.2f}")
    logging.info(f"ETH Swarm Adjusted Confidence: {adj_eth_conf:.2f}")
    
    # A local ML model predicts SOL is going DOWN with 0.60 confidence
    raw_sol_conf = 0.60
    adj_sol_conf = swarm.evaluate_signal_context("SOL", "DOWN", raw_sol_conf)
    
    logging.info(f"SOL Local Model Confidence: {raw_sol_conf:.2f}")
    logging.info(f"SOL Swarm Adjusted Confidence: {adj_sol_conf:.2f}")
