import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Optional
from datetime import datetime, timezone

from core.state_vector import SymbolStateVector

logger = logging.getLogger(__name__)

class GraphIntelligenceLayer:
    """
    Representa el Ecosistema de Mercado como un Grafo Causal y Relacional.
    Evalúa el contagio de liquidez y la conectividad (correlación) entre criptomonedas.
    """
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.graph = nx.DiGraph()  # Directed Graph for Causal/Leader-Follower Relationships
        self.state_matrix: Dict[str, SymbolStateVector] = {
            sym: SymbolStateVector(symbol=sym) for sym in symbols
        }
        
        # Initialize isolated nodes
        self.graph.add_nodes_from(symbols)
        self.last_update = datetime.now(timezone.utc).timestamp()
        
    def update_graph_edges(self, correlation_matrix: np.ndarray, current_symbols: List[str] = None):
        """
        Recibe una matriz de correlación (ej. calculada a partir de los últimos X retornos)
        y actualiza las aristas del grafo matemático.
        """
        symbols_to_use = current_symbols if current_symbols is not None else self.symbols
        
        if correlation_matrix.shape != (len(symbols_to_use), len(symbols_to_use)):
            logger.error("GraphIntelligenceLayer: Dimension mismatch in correlation matrix.")
            return

        self.graph.clear_edges()
        
        for i, sym_a in enumerate(symbols_to_use):
            for j, sym_b in enumerate(symbols_to_use):
                if i != j:
                    weight = correlation_matrix[i, j]
                    # Solo conectamos si la correlación/causalidad es estadísticamente significativa
                    if abs(weight) >= 0.70:
                        # Si weight > 0, se mueven juntos. Si < 0, inverso.
                        # Asumiremos direccionalidad basada en liquidez/dominio en iteraciones futuras,
                        # por ahora conectamos bidireccionalmente la influencia.
                        self.graph.add_edge(sym_a, sym_b, weight=weight)
                        
        self._calculate_network_metrics()

    def _calculate_network_metrics(self):
        """Calcula métricas de centralidad y las inyecta en los state vectors."""
        try:
            # Eigenvector centrality indica quién es el verdadero líder (ej. BTC)
            centrality = nx.eigenvector_centrality(self.graph, max_iter=500, weight='weight', tol=1e-06)
            for sym, cent_val in centrality.items():
                if sym in self.state_matrix:
                    self.state_matrix[sym].eigenvector_centrality = cent_val
        except nx.PowerIterationFailedConvergence:
            logger.warning("GraphIntelligenceLayer: Eigenvector centrality failed to converge.")

    def get_contagion_risk(self, symbol: str) -> float:
        """
        Calcula la presión de contagio bajista. Si los predecesores de este símbolo
        están sufriendo una caída severa (orderflow negativo o trend negativo), este
        símbolo sufrirá gravedad matemática.
        """
        if symbol not in self.graph:
            return 0.0

        predecessors = list(self.graph.predecessors(symbol))
        if not predecessors:
            return 0.0

        risk = 0.0
        total_weight = 0.0
        
        for pred in predecessors:
            weight = abs(self.graph[pred][symbol]['weight'])
            pred_state = self.state_matrix[pred]
            
            if pred_state:
                # Si el líder tiene orderflow_imbalance < -0.3, suma al riesgo bajista
                if pred_state.orderflow_imbalance < -0.3:
                    risk += weight * abs(pred_state.orderflow_imbalance)
                # Si el líder tiene una tendencia fuertemente negativa
                if pred_state.trend_score_m5 < -0.5:
                    risk += weight * abs(pred_state.trend_score_m5)
            
            total_weight += weight
            
        if total_weight > 0:
            risk = risk / total_weight  # Normalizar
            
        # Actualizar el tensor de estado
        if symbol in self.state_matrix:
            self.state_matrix[symbol].contagion_pressure = risk
            
        return risk

    def get_ecosystem_gravity(self) -> float:
        """Devuelve el estado general del ecosistema (Bullish vs Bearish) basado en los nodos centrales."""
        total_gravity = 0.0
        for sym, state in self.state_matrix.items():
            total_gravity += state.trend_score_m5 * state.eigenvector_centrality
        return total_gravity

    def update_symbol_state(self, symbol: str, features: Dict[str, float]):
        """Actualiza el vector de estado para un símbolo específico."""
        if symbol in self.state_matrix:
            self.state_matrix[symbol].update_from_dict(features)
