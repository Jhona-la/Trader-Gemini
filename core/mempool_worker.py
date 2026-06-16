import asyncio
import json
import websockets
import threading
from utils.logger import logger
from core.mev_rbf_engine import MempoolRbfEngine

class MempoolWorker:
    """
    Connects to an L1/L2 WebSocket RPC (e.g., Alchemy, QuickNode, local Erigon)
    to monitor `newPendingTransactions`.
    Feeds transaction data (from, nonce, gasPrice) into the Zero-Copy Cython Map
    to detect real-time panics (RBF) and MEV bundle patterns.
    """
    def __init__(self, wss_url="wss://eth-mainnet.g.alchemy.com/v2/demo"):
        self.engine = MempoolRbfEngine(halflife=10.0)
        self.wss_url = wss_url  # In production, replace with dedicated WSS
        self._thread = None
        self._loop = None
        self.is_running = False
        
    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="MempoolWorker")
        self._thread.start()
        logger.info("🛰️ [MEMPOOL] Zero-Copy Worker started.")
        
    def stop(self):
        self.is_running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            
    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._ws_loop())
        except Exception as e:
            logger.error(f"❌ [MEMPOOL] Loop crashed: {e}")
            
    async def _ws_loop(self):
        while self.is_running:
            try:
                # WSS connections often have strict limits on free tiers.
                # A local node is highly recommended for production MEV tracking.
                async with websockets.connect(self.wss_url) as ws:
                    logger.info("⚡ [MEMPOOL] Connected to RPC. Subscribing to pending txs...")
                    req = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "eth_subscribe",
                        "params": ["newPendingTransactions"]
                    }
                    await ws.send(json.dumps(req))
                    
                    # We would need to fetch full tx details for each hash using eth_getTransactionByHash.
                    # Since this is a massive bottleneck on free RPCs, for the simulation/demo,
                    # we will randomly generate panic events if the URL is 'demo', or 
                    # use the full flow if it's a real node.
                    
                    if "demo" in self.wss_url:
                        logger.warning("⚠️ [MEMPOOL] Running in DEMO mode. Emulating mempool txs to avoid rate-limits.")
                        await self._emulate_mempool_stream()
                        return

                    # Real flow (Warning: High RPC usage)
                    while self.is_running:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        if "params" in data and "result" in data["params"]:
                            tx_hash = data["params"]["result"]
                            
                            # Fire-and-forget fetch
                            asyncio.create_task(self._fetch_and_process_tx(ws, tx_hash))
                            
            except Exception as e:
                if self.is_running:
                    logger.warning(f"⚠️ [MEMPOOL] WS Disconnected. Reconnecting in 5s... ({e})")
                    await asyncio.sleep(5)
                    
    async def _fetch_and_process_tx(self, ws, tx_hash):
        """Fetch full transaction payload to get Nonce and GasPrice."""
        req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "eth_getTransactionByHash",
            "params": [tx_hash]
        }
        await ws.send(json.dumps(req))
        # Note: In a highly concurrent environment over the same WS, responses can interleave.
        # A robust implementation would use a dedicated HTTP endpoint for fetching or map IDs.
        # For this prototype, we'll wait for the next message (simplified).
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            data = json.loads(msg)
            if "result" in data and data["result"]:
                tx = data["result"]
                from_addr = tx.get("from", "")
                nonce = int(tx.get("nonce", "0x0"), 16)
                gas_price = int(tx.get("gasPrice", "0x0"), 16) / 1e9 # Convert to Gwei
                
                # Push to Cython map
                urgency = self.engine.process_transaction(from_addr, nonce, gas_price)
                if urgency > 50.0:
                    logger.warning(f"🚨 [RBF PANIC] Detected urgency replacement on {from_addr}. Delta: {urgency:.2f} Gwei")
        except Exception:
            pass

    async def _emulate_mempool_stream(self):
        """Emulates Mempool traffic for the Demo without hitting rate limits."""
        import random
        addresses = [f"0x{i:040x}" for i in range(1000)]
        nonces = {addr: 0 for addr in addresses}
        
        while self.is_running:
            # Simulate 100 txs per second
            for _ in range(100):
                addr = random.choice(addresses)
                # 5% chance of RBF (same nonce, higher gas)
                is_rbf = random.random() < 0.05
                
                if is_rbf:
                    nonce = nonces[addr]
                    gas_price = random.uniform(50.0, 200.0)
                else:
                    nonces[addr] += 1
                    nonce = nonces[addr]
                    gas_price = random.uniform(10.0, 30.0)
                    
                urgency = self.engine.process_transaction(addr, nonce, gas_price)
                if urgency > 1000.0: # Only log major panics
                    logger.warning(f"🚨 [RBF PANIC EMULATED] Urgent replacement detected! Delta: {urgency:.2f} Gwei")
                    
            await asyncio.sleep(1.0)
            # Prune state every 10 seconds
            if random.random() < 0.1:
                pruned = self.engine.prune_stale_transactions(60.0)

    def get_panic_score(self):
        """Gets time-decayed global panic score."""
        return self.engine.get_panic_score()

# Singleton
mempool_worker = MempoolWorker()
