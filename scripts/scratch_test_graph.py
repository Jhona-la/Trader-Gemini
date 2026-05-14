import sys
import asyncio
from utils.logger import logger
from core.engine import Engine
from core.meta_arbitrator import meta_arbitrator

async def test_graph_layer():
    print("Testing Graph Layer Initialization...")
    engine = Engine()
    print("Engine instantiated.")
    print("Graph Layer attached:", meta_arbitrator.graph_layer is not None)
    print("Symbols in graph:", list(meta_arbitrator.graph_layer.graph.nodes()))
    
    print("SUCCESS: Engine and Graph Layer initialized without errors.")

if __name__ == "__main__":
    asyncio.run(test_graph_layer())
