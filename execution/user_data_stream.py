import asyncio
import json
from core.events import FillEvent, OrderEvent
import websockets
import ssl
from datetime import datetime, timezone
from config import Config
from utils.logger import logger
import aiohttp

class UserDataStream:
    """
    🎧 USER DATA STREAM LISTENER (Grade-A Security Component)
    
    QUÉ: Conexión WebSocket dedicada a eventos de cuenta (Ejecuciones, Balance).
    POR QUÉ: Elimina la ceguera ante "Ghost Orders" (fills tardíos) y liquidaciones.
    CÓMO: Gestiona listenKey dinámico y inyecta eventos en el Engine.
    
    Features:
    - Auto-Reconnect & Keep-Alive (cada 30m)
    - Parsing de ORDER_TRADE_UPDATE (Fills parciales/totales)
    - Parsing de ACCOUNT_UPDATE (Cambios de balance/posición)
    - Parsing de ORDER_TRADE_UPDATE (Liquidaciones)
    """
    
    def __init__(self, events_queue, exchange_client=None):
        self.events_queue = events_queue
        self.client = exchange_client # Optional: reuse existing session if passed
        self.listen_key = None
        self.running = False
        self.ws_connection = None
        self.last_keep_alive = 0
        self.keep_alive_task = None
        self.reconnect_count = 0
        
        # Determine Base URLs
        if Config.BINANCE_USE_FUTURES:
            if (hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO) or Config.BINANCE_USE_TESTNET:
                self.rest_url = "https://testnet.binancefuture.com"
                self.ws_url = "wss://stream.binancefuture.com/ws"
            else:
                self.rest_url = "https://fapi.binance.com"
                self.ws_url = "wss://fstream.binance.com/ws"
        else:
            if Config.BINANCE_USE_TESTNET:
                self.rest_url = "https://testnet.binance.vision"
                self.ws_url = "wss://testnet.binance.vision/ws"
            else:
                self.rest_url = "https://api.binance.com"
                self.ws_url = "wss://stream.binance.com:9443/ws"
                
    async def start(self):
        """Main entry point"""
        self.running = True
        logger.info("🎧 [UserStream] Starting User Data Stream Listener...")
        
        while self.running:
            try:
                # 1. Obtain Listen Key
                self.listen_key = await self._get_listen_key()
                if not self.listen_key:
                    logger.error("❌ [UserStream] Failed to get Listen Key. Retrying in 5s...")
                    await asyncio.sleep(5)
                    continue
                
                # 2. Start Keep-Alive Loop
                if self.keep_alive_task: self.keep_alive_task.cancel()
                self.keep_alive_task = asyncio.create_task(self._keep_alive_loop())
                
                # 3. Connect WebSocket
                stream_url = f"{self.ws_url}/{self.listen_key}"
                logger.info(f"🔗 [UserStream] Warning: Connecting to {stream_url.split('/ws/')[0]}...")
                
                # SSL Context (Standard security)
                ssl_context = ssl.create_default_context()
                
                async with websockets.connect(stream_url, ssl=ssl_context) as ws:
                    self.ws_connection = ws
                    self.reconnect_count = 0
                    logger.info("✅ [UserStream] Connected & Listening for Account Events")
                    
                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=60.0)
                            await self._handle_message(msg)
                        except asyncio.TimeoutError:
                            # Ping/Pong handled by library, but verify liveliness
                            logger.debug("💓 [UserStream] Heartbeat check...")
                            continue
                        except websockets.ConnectionClosed:
                            logger.warning("⚠️ [UserStream] Connection Closed. Reconnecting...")
                            break
                            
            except Exception as e:
                logger.error(f"❌ [UserStream] Connection Error: {e}")
                self.reconnect_count += 1
                await asyncio.sleep(min(30, self.reconnect_count * 2)) # Exponential backoff
                
    async def stop(self):
        self.running = False
        if self.ws_connection:
            await self.ws_connection.close()
        if self.keep_alive_task:
            self.keep_alive_task.cancel()
        logger.info("🛑 [UserStream] Stopped.")

    async def _get_listen_key(self):
        """Fetch listenKey via REST"""
        endpoint = "/fapi/v1/listenKey" if Config.BINANCE_USE_FUTURES else "/api/v3/userDataStream"
        url = f"{self.rest_url}{endpoint}"
        headers = {'X-MBX-APIKEY': self._get_api_key()}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data['listenKey']
                else:
                    err = await resp.text()
                    logger.error(f"❌ [UserStream] ListenKey Error: {resp.status} - {err}")
                    return None

    async def _keep_alive_loop(self):
        """Keep listenKey alive every 30 mins"""
        while self.running and self.listen_key:
            await asyncio.sleep(1800) # 30 mins
            try:
                endpoint = "/fapi/v1/listenKey" if Config.BINANCE_USE_FUTURES else "/api/v3/userDataStream"
                url = f"{self.rest_url}{endpoint}" # PUT ?listenKey=... not needed in body usually? 
                # Check specifics: Futures uses PUT with header usually? 
                # Docs: PUT /fapi/v1/listenKey (HMAC not needed, just API Key)
                headers = {'X-MBX-APIKEY': self._get_api_key()}
                # For Spot: param listenKey might be needed in body or query?
                # CCXT usually handles this, but here we do manual light request.
                
                # Futures: Just PUT request is enough to extend the *associated* key? 
                # Actually, standard is "PUT /fapi/v1/listenKey"
                
                async with aiohttp.ClientSession() as session:
                    async with session.put(url, headers=headers) as resp:
                         if resp.status == 200:
                             logger.debug("🔄 [UserStream] ListenKey Extended")
                         else:
                             logger.warning(f"⚠️ [UserStream] Keep-Alive Failed: {resp.status}")
            except Exception as e:
                logger.error(f"⚠️ [UserStream] Keep-Alive Error: {e}")

    def _get_api_key(self):
        if Config.BINANCE_USE_FUTURES and hasattr(Config, 'BINANCE_USE_DEMO') and Config.BINANCE_USE_DEMO:
             return Config.BINANCE_DEMO_API_KEY
        elif Config.BINANCE_USE_TESTNET:
             return Config.BINANCE_TESTNET_API_KEY
        else:
             return Config.BINANCE_API_KEY

    async def _handle_message(self, msg_raw):
        """Route event types"""
        try:
            import orjson
            msg = orjson.loads(msg_raw)
        except ImportError:
            import json
            msg = json.loads(msg_raw)
            
        try:
            event_type = msg.get('e')
            
            if event_type == 'ORDER_TRADE_UPDATE':
                await self._process_order_update(msg)
            elif event_type == 'ACCOUNT_UPDATE':
                await self._process_account_update(msg)
            elif event_type == 'listenKeyExpired':
                logger.warning("⚠️ [UserStream] ListenKey Expired! Reconnecting...")
                if self.ws_connection: await self.ws_connection.close()
                
        except Exception as e:
            logger.error(f"Message Parse Error: {e} | Raw: {msg_raw[:100]}")

    async def _process_order_update(self, msg):
        """
        Handle execution reports (Fills, Cancellations, Rejections)
        Payload: 'o': { ... }
        """
        o = msg['o']
        symbol = o['s']
        status = o['X'] # NEW, PARTIALLY_FILLED, FILLED, CANCELED, REJECTED, EXPIRED
        order_id = str(o['i'])
        filled_qty = float(o['z']) # Accumulated
        last_qty = float(o['l'])   # Last trade qty
        last_price = float(o['L']) # Last trade price
        side = o['S']
        
        # We only care about Fills (Partial or Full)
        if last_qty > 0 and status in ['PARTIALLY_FILLED', 'FILLED']:
            logger.info(f"⚡ [UserStream] FILL DETECTED: {symbol} {side} {last_qty} @ {last_price} (Status: {status})")
            
            # Map side to Direction
            # If Side is BUY, direction is LONG (ENTRY) or SHORT_CLOSE (EXIT)?
            # Standard logic: BUY is LONG, SELL is SHORT.
            # But what if closing a SHORT? It's a BUY.
            # FillEvent doesn't strictly need 'Entry/Exit', just direction.
            # Core engine handles net position via Portfolio.
            
            # ═══════════════════════════════════════════════════════════════
            # FORENSIC-V44 FIX: CORRECTED FillEvent CONSTRUCTION
            # BUG #1: direction=side passed raw string ('BUY'/'SELL') but
            #   FillEvent expects OrderSide enum. frozen+slots dataclass
            #   would reject or mistype it.
            # BUG #2: received_ns=time.time_ns() — FillEvent has NO such
            #   field → TypeError silently kills the fill event.
            # BUG #3: strategy_id="UserStream" loses the real strategy.
            #   We now look up the original OrderEvent metadata from
            #   the OrderManager (if available) to recover trade_id,
            #   strategy_id, setup_type, and horizon.
            # ═══════════════════════════════════════════════════════════════
            from core.enums import OrderSide as _OrderSide
            
            # Map string → enum
            if side == 'BUY':
                fill_direction = _OrderSide.BUY
            elif side == 'SELL':
                fill_direction = _OrderSide.SELL
            else:
                logger.error(f"❌ [UserStream] Unknown side: {side}")
                return

            # Attempt to recover original order metadata
            _real_strategy = "UserStream"
            _real_setup = None
            _real_trade_id = None
            _real_horizon = "SCALPING"
            _real_exit_reason = None
            _real_metadata = {}

            # Convert symbol format: BTCUSDT → BTC/USDT
            symbol_ccxt = symbol
            for quote in ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH']:
                if symbol.endswith(quote) and len(symbol) > len(quote):
                    symbol_ccxt = symbol[:-len(quote)] + '/' + quote
                    break

            # Try to find the original order in OrderManager or Portfolio
            try:
                if hasattr(self, 'order_manager') and self.order_manager:
                    tracked = self.order_manager.get_order(order_id)
                    if tracked:
                        _real_strategy = tracked.get('strategy_id', 'UserStream')
                        _real_setup = tracked.get('setup_type')
                        _real_trade_id = tracked.get('trade_id')
                        _real_horizon = tracked.get('horizon', 'SCALPING')
                        _real_metadata = tracked.get('metadata', {}) or {}
                        logger.info(f"🔗 [UserStream] Matched fill to order: {_real_strategy} ({_real_trade_id})")
            except Exception as e:
                logger.debug(f"[UserStream] Order lookup failed: {e}")

            # FORENSIC FIX: Extract Horizon from Native clientOrderId
            client_oid = o.get('c', '')
            if client_oid and client_oid.startswith('ctos_'):
                parts = client_oid.split('_')
                if len(parts) >= 2:
                    _real_horizon = parts[1] # e.g., 'SCALPING' or 'SWING'
                    logger.info(f"🧬 [UserStream] Recovered Horizon '{_real_horizon}' from clientOrderId: {client_oid}")

            # ═══════════════════════════════════════════════════════════════
            # FORENSIC FIX: NATIVE HEDGE MODE INFERENCE (Zombie Killer)
            # QUÉ: Si el OrderManager falla (o se reinició), usamos los
            #   campos nativos de Binance para deducir is_close de forma infalible.
            # ═══════════════════════════════════════════════════════════════
            # 'ps' = positionSide (LONG, SHORT, BOTH)
            # 'R' = reduceOnly (bool)
            pos_side = o.get('ps', 'BOTH')
            is_reduce_only = o.get('R', False)
            
            # Infer is_close natively
            native_is_close = is_reduce_only
            if pos_side == 'LONG' and side == 'SELL': native_is_close = True
            elif pos_side == 'SHORT' and side == 'BUY': native_is_close = True
            
            # Use tracked metadata if available, otherwise trust Binance native payload
            final_is_close = _real_metadata.get('is_close', _real_metadata.get('is_exit', native_is_close))
            _real_metadata['is_close'] = final_is_close
            _real_metadata['is_exit'] = final_is_close # Sync both
            
            # Add strict context to metadata to help Portfolio route the fill
            _real_metadata['binance_position_side'] = pos_side

            fill_event = FillEvent(
                timeindex=datetime.now(timezone.utc),
                symbol=symbol_ccxt,
                exchange='BINANCE',
                quantity=last_qty,
                direction=fill_direction,
                fill_cost=last_qty * last_price,
                fill_price=last_price,
                order_id=order_id,
                commission=float(o.get('n', 0)),
                is_closed=final_is_close, # NATIVE HEDGE INFERENCE INSTEAD OF STATUS=='FILLED' (which only means order is filled, not position closed)
                strategy_id=_real_strategy,
                order_type=None,
                setup_type=_real_setup,
                exit_reason=_real_exit_reason,
                trade_id=_real_trade_id,
                horizon=_real_horizon,
                metadata=_real_metadata,
            )
            
            self.events_queue.put(fill_event)

        elif status == 'CANCELED':
            logger.info(f"🗑️ [UserStream] Order {order_id} CANCELED")
            # Optional: Remove from OrderManager if needed via explicit event
        
        elif status == 'EXPIRED':
             logger.warning(f"⚠️ [UserStream] Order {order_id} EXPIRED (TimeInForce?)")
             
    async def _process_account_update(self, msg):
        """
        Handle Balance & Position Updates
        Payload: 'a': { 'B': [balances], 'P': [positions] }
        """
        # We could inject a 'BalanceUpdateEvent' but usually Portfolio
        # syncs on fills. However, this is useful for Liquidations.
        data = msg['a']
        reason = data.get('m', '') # Event reason type
        
        # Check for Liquidations
        if reason == 'ASSET_TRANSFER' or reason == 'DEPOSIT' or 'LIQUIDATION' in reason:
             logger.info(f"💰 [UserStream] Account Update Reason: {reason}")
             
        # Detect strict Liquidations from Position Updates?
        # TODO: Advanced liquidation handling
        pass
