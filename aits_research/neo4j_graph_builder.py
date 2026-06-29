"""
AITS Phase 3: Graph Representation Engine
Neo4j Hypergraph Builder

Transforms flat market data into a highly interconnected topological structure.
Creates:
1. Asset Nodes (e.g. BTC, ETH)
2. Liquidity Pool Nodes (representing high volume clusters at specific prices)
3. Event Nodes (Whale movements, Macro news)

Connects them dynamically via Cypher queries:
- (:Asset)-[:CORRELATED_TO {weight}]->(:Asset)
- (:Asset)-[:ATTRACTS {gravity}]->(:LiquidityPool)
"""

import logging
try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None

import networkx as nx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "aits_neo4j_pass")

class AITSGraphBuilder:
    def __init__(self, uri, auth):
        self.uri = uri
        self.auth = auth
        self.use_neo4j = False
        self.nx_graph = nx.DiGraph() # Fallback in-memory graph
        
        if GraphDatabase:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=self.auth)
                # Test connectivity
                self.driver.verify_connectivity()
                self.use_neo4j = True
                logging.info("🌟 Conectado a Neo4j exitosamente.")
            except Exception as e:
                logging.warning(f"⚠️ No se pudo conectar a Neo4j ({e}). Activando fallback In-Memory (NetworkX).")
                self.driver = None
        else:
            logging.warning("⚠️ Librería 'neo4j' no encontrada. Activando fallback In-Memory (NetworkX).")
            self.driver = None

    def close(self):
        if self.use_neo4j and self.driver:
            self.driver.close()

    def _execute_query(self, query, parameters=None):
        """Executes a Cypher query with optional parameters."""
        if not self.use_neo4j:
            return None
            
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                return [record for record in result]
        except Exception as e:
            logging.error(f"Cypher Execution Error: {e}")
            return None

    def initialize_schema(self):
        """Sets up uniqueness constraints to prevent duplicate nodes."""
        logging.info("Initializing Graph Schema...")
        
        if self.use_neo4j:
            queries = [
                "CREATE CONSTRAINT asset_symbol IF NOT EXISTS FOR (a:Asset) REQUIRE a.symbol IS UNIQUE",
                "CREATE CONSTRAINT pool_id IF NOT EXISTS FOR (p:LiquidityPool) REQUIRE p.pool_id IS UNIQUE",
                "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:MacroEvent) REQUIRE e.event_id IS UNIQUE"
            ]
            for q in queries:
                self._execute_query(q)
        else:
            # NetworkX does not need explicit schema constraints. Nodes are uniquely identified by their ID.
            pass
            
        logging.info("Schema Constraints Applied.")

    def seed_initial_ontology(self):
        """Creates the initial base graph topology."""
        logging.info("Seeding Phase 3 Hypergraph (Assets & Liquidity Pools)...")

        # 1. Create Core Assets
        assets = [
            {"symbol": "BTC", "market_cap": "high", "sector": "L1"},
            {"symbol": "ETH", "market_cap": "high", "sector": "L1"},
            {"symbol": "SOL", "market_cap": "mid", "sector": "L1"}
        ]
        
        for asset in assets:
            if self.use_neo4j:
                query = """
                MERGE (a:Asset {symbol: $symbol})
                SET a.market_cap = $market_cap, a.sector = $sector
                """
                self._execute_query(query, asset)
            else:
                self.nx_graph.add_node(asset['symbol'], type='Asset', **asset)

        # 2. Establish Cross-Asset Correlations (Simulated Dynamic Weights)
        correlations = [
            {"source": "BTC", "target": "ETH", "weight": 0.85},
            {"source": "BTC", "target": "SOL", "weight": 0.72},
            {"source": "ETH", "target": "SOL", "weight": 0.90}
        ]
        
        for corr in correlations:
            if self.use_neo4j:
                query = """
                MATCH (a:Asset {symbol: $source}), (b:Asset {symbol: $target})
                MERGE (a)-[r:CORRELATED_TO]->(b)
                SET r.weight = $weight, r.last_updated = timestamp()
                """
                self._execute_query(query, corr)
            else:
                self.nx_graph.add_edge(corr['source'], corr['target'], type='CORRELATED_TO', weight=corr['weight'])

        # 3. Create Liquidity Pools (Order Book Gravity Zones)
        pools = [
            {"pool_id": "BTC_50K", "asset": "BTC", "price_level": 50000, "volume_usd": 15000000},
            {"pool_id": "ETH_4K", "asset": "ETH", "price_level": 4000, "volume_usd": 8000000}
        ]

        for pool in pools:
            if self.use_neo4j:
                query = """
                MATCH (a:Asset {symbol: $asset})
                MERGE (p:LiquidityPool {pool_id: $pool_id})
                SET p.price_level = $price_level, p.volume_usd = $volume_usd
                MERGE (a)-[r:ATTRACTS]->(p)
                SET r.gravity_score = (p.volume_usd / 1000000) * 1.5
                """
                self._execute_query(query, pool)
            else:
                gravity_score = (pool['volume_usd'] / 1000000) * 1.5
                self.nx_graph.add_node(pool['pool_id'], type='LiquidityPool', **pool)
                self.nx_graph.add_edge(pool['asset'], pool['pool_id'], type='ATTRACTS', gravity_score=gravity_score)

    def run_graph_analytics(self):
        """Executes intelligent traversals to find market vulnerabilities."""
        logging.info("--- AITS Graph Analytics ---")
        
        if self.use_neo4j:
            # Query: Find the liquidity pool exerting the highest gravitational pull on the ecosystem
            query = """
            MATCH (a:Asset)-[r:ATTRACTS]->(p:LiquidityPool)
            RETURN a.symbol AS asset, p.price_level AS target_price, r.gravity_score AS gravity
            ORDER BY gravity DESC LIMIT 1
            """
            results = self._execute_query(query)
            if results:
                for record in results:
                    logging.info(f"Highest Gravity Target: {record['asset']} pulled towards ${record['target_price']} (Gravity: {record['gravity']})")
            else:
                logging.info("No gravity data found.")
        else:
            # NetworkX equivalent
            max_gravity = 0
            best_edge = None
            for u, v, data in self.nx_graph.edges(data=True):
                if data.get('type') == 'ATTRACTS':
                    if data['gravity_score'] > max_gravity:
                        max_gravity = data.get('gravity_score')
                        best_edge = (u, v)
            
            if best_edge:
                asset = best_edge[0]
                pool_data = self.nx_graph.nodes[best_edge[1]]
                logging.info(f"Highest Gravity Target: {asset} pulled towards ${pool_data.get('price_level')} (Gravity: {max_gravity})")
            else:
                logging.info("No gravity data found in NetworkX graph.")

    def propagate_shock(self, source_symbol: str, shock_magnitude: float) -> dict:
        """
        Cálculo de propagación causal. Si el source_symbol sufre un shock (ej. OFI fuertemente negativo),
        calcula el impacto propagado hacia los activos correlacionados en milisegundos.
        """
        impacts = {}
        if self.use_neo4j:
            query = """
            MATCH (a:Asset {symbol: $source})-[r:CORRELATED_TO]-(b:Asset)
            RETURN b.symbol AS target, r.weight AS weight
            """
            results = self._execute_query(query, {"source": source_symbol})
            if results:
                for record in results:
                    impacts[record['target']] = shock_magnitude * record['weight']
        else:
            # NetworkX graph is directed, but we assume correlation is mutual for shock propagation.
            if source_symbol in self.nx_graph:
                # Undirected approach for correlation
                neighbors = list(self.nx_graph.successors(source_symbol)) + list(self.nx_graph.predecessors(source_symbol))
                # Remove duplicates and self
                neighbors = set(n for n in neighbors if self.nx_graph.nodes[n].get('type') == 'Asset' and n != source_symbol)
                
                for target in neighbors:
                    # Check edge weight in either direction
                    weight = 0.0
                    if self.nx_graph.has_edge(source_symbol, target):
                        weight = self.nx_graph[source_symbol][target]['weight']
                    elif self.nx_graph.has_edge(target, source_symbol):
                        weight = self.nx_graph[target][source_symbol]['weight']
                    
                    impacts[target] = shock_magnitude * weight
                    
        return impacts

if __name__ == "__main__":
    builder = AITSGraphBuilder(URI, AUTH)
    builder.initialize_schema()
    builder.seed_initial_ontology()
    builder.run_graph_analytics()
    
    logging.info("--- Probando propagación de shock causal ---")
    # Simulate a massive drop in BTC (Magnitude -5.0)
    shocks = builder.propagate_shock("BTC", -5.0)
    for target, impact in shocks.items():
        logging.warning(f"⚠️ Alerta Swarm: El shock en BTC ha propagado un impacto de {impact:.2f} a {target}")
        
    builder.close()
