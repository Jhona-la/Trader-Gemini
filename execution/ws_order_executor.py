import asyncio
import json
import hmac
import hashlib
import urllib.parse
import time
import websockets
import ssl
from uuid import uuid4
from config import Config
from utils.logger import logger
from utils.metrics_exporter import metrics

class WSOrderExecutor:
    """
    ⚡ ZERO-LATENCY WEBSOCKET ORDER EXECUTOR
    
    Bypasses REST API completely. Connects to Binance WS-API for direct order placement.
    Expected Latency: <10ms (vs 50-100ms REST).
    """
    
    def __init__(self, api_key: str, secret_key: str, is_testnet: bool = False, is_futures: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.is_testnet = is_testnet
        self.is_futures = is_futures
        
        # Resolve Endpoints
        if self.is_futures:
            if self.is_testnet or getattr(Config, 'BINANCE_USE_DEMO', False):
                self.ws_url = "wss://testnet.binancefuture.com/ws-fapi/v1"
            else:
                self.ws_url = "wss://ws-fapi.binance.com/ws-fapi/v1"
        else:
            if self.is_testnet:
                self.ws_url = "wss://testnet.binance.vision/ws-api/v3"
            else:
                self.ws_url = "wss://ws-api.binance.com:443/ws-api/v3"
                
        self.ws = None
        self.running = False
        self.connection_task = None
        
        # Request Tracking (Futures resolve with the ID)
        self.pending_requests = {}
        self.req_id_counter = int(time.time() * 1000)
        
    async def start(self):
        """Spawns the background connection task."""
        self.running = True
        self.connection_task = asyncio.create_task(self._connection_loop())
        
    async def stop(self):
        """Graceful shutdown."""
        self.running = False
        if self.ws:
            await self.ws.close()
        if self.connection_task:
            self.connection_task.cancel()
            
    def is_ready(self) -> bool:
        """Checks if the WS is connected and open."""
        return self.ws is not None and self.ws.open
        
    async def _connection_loop(self):
        """Maintains the WS connection indefinitely."""
        ssl_context = ssl.create_default_context()
        reconnect_delay = 1.0
        
        while self.running:
            try:
                logger.info(f"⚡ [WS-EXEC] Connecting to {self.ws_url} ...")
                async with websockets.connect(self.ws_url, ssl=ssl_context, ping_interval=30) as ws:
                    self.ws = ws
                    reconnect_delay = 1.0
                    logger.info("✅ [WS-EXEC] Connected! Zero-Latency pipeline active.")
                    
                    while self.running and ws.open:
                        try:
                            msg_raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
                            asyncio.create_task(self._handle_response(msg_raw))
                        except asyncio.TimeoutError:
                            # Ping-pong handled natively, just verifying liveliness
                            continue
                            
            except websockets.ConnectionClosed:
                logger.warning("⚠️ [WS-EXEC] Connection dropped. Reconnecting...")
            except Exception as e:
                logger.error(f"❌ [WS-EXEC] Connection error: {e}")
                
            self.ws = None
            if self.running:
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 30.0)

    async def _handle_response(self, msg_raw: str):
        """Routes incoming ACK/RESP to the waiting future."""
        try:
            res = json.loads(msg_raw)
            req_id = str(res['id'])
            
            if req_id in self.pending_requests:
                future = self.pending_requests.pop(req_id)
                if not future.done():
                    future.set_result(res)
            elif 'event' not in res: # Ignore general events like listenKey
                logger.debug(f"🔍 [WS-EXEC] Untracked response: {msg_raw}")
        except Exception as e:
            logger.error(f"❌ [WS-EXEC] Parse error in response: {e}")

    def _sign_params(self, params: dict) -> dict:
        """Adds API key, Timestamp, and generates HMAC SHA256 Signature."""
        params['apiKey'] = self.api_key
        params['timestamp'] = int(time.time() * 1000)
        
        # Sort and urlencode
        query_string = urllib.parse.urlencode(dict(sorted(params.items())))
        
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        params['signature'] = signature
        return params

    async def place_order(self, params: dict) -> dict:
        """
        Submits an order over WS and waits for response.
        Expected params format depends on Spot vs Futures.
        """
        if not self.is_ready():
            raise ConnectionError("WS-API is not connected.")
            
        req_id = str(self.req_id_counter)
        self.req_id_counter += 1
        
        signed_params = self._sign_params(params.copy())
        
        payload = {
            "id": req_id,
            "method": "order.place",
            "params": signed_params
        }
        
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[req_id] = future
        
        start_ts = time.time()
        await self.ws.send(json.dumps(payload))
        
        # Wait up to 5 seconds for execution confirmation
        try:
            response = await asyncio.wait_for(future, timeout=5.0)
            latency_ms = (time.time() - start_ts) * 1000
            metrics.record_latency("ws_order_place", latency_ms)
            logger.info(f"⚡ [WS-EXEC] Order ACK in {latency_ms:.2f}ms")
            
            # Check for API Error
            if 'error' in response:
                raise Exception(f"Binance WS-API Error: {response['error']}")
                
            # Spot vs Futures response shape
            if 'result' in response:
                return response['result']
            return response
            
        except asyncio.TimeoutError:
            self.pending_requests.pop(req_id, None)
            raise TimeoutError("WS-API Order timeout")
            
    async def cancel_order(self, symbol: str, order_id: str = None, client_order_id: str = None) -> dict:
        """Cancels an existing order via WS."""
        if not self.is_ready():
            raise ConnectionError("WS-API is not connected.")
            
        params = {"symbol": symbol}
        if order_id:
            params["orderId"] = int(order_id)
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Must provide order_id or client_order_id")
            
        req_id = str(self.req_id_counter)
        self.req_id_counter += 1
        
        payload = {
            "id": req_id,
            "method": "order.cancel",
            "params": self._sign_params(params)
        }
        
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[req_id] = future
        
        start_ts = time.time()
        await self.ws.send(json.dumps(payload))
        
        try:
            response = await asyncio.wait_for(future, timeout=5.0)
            latency_ms = (time.time() - start_ts) * 1000
            logger.info(f"⚡ [WS-EXEC] Cancel ACK in {latency_ms:.2f}ms")
            
            if 'error' in response:
                raise Exception(f"Binance WS-API Error: {response['error']}")
            if 'result' in response:
                return response['result']
            return response
            
        except asyncio.TimeoutError:
            self.pending_requests.pop(req_id, None)
            raise TimeoutError("WS-API Cancel timeout")
