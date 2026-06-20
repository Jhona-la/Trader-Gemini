
import numpy as np
from typing import Dict, List, Optional, Tuple
from utils.logger import logger
from config import Config

class LiquidityGuardian:
    """
    PROTECCIÓN DE CAPITAL (Phase 7): Guardián de Liquidez
    ====================================================
    - Analiza el Order Book en tiempo real.
    - Detecta 'Muros' (Order Walls) institucionales.
    - Calcula el Slippage esperado para asegurar R/R positivo.
    """
    
    def __init__(self, exchange, async_exchange=None):
        self.exchange = exchange
        self.async_exchange = async_exchange
        self.wall_threshold_multiplier = 5.0 # Un muro es 5x el volumen promedio del book
        
        # Phase 33: Anti-Spoofing Memory
        # Guardamos snapshots previos para detectar muros que desaparecen
        self.book_history = {} # {symbol: {'bids': [], 'asks': [], 'timestamp': ...}}
        
        # 🏎️ [L-001] Cache Asíncrono de Latencia Cero (TTL = 1.0s)
        self.order_book_cache = {} # {symbol: {'data': order_book, 'timestamp': float}}
        self.cache_ttl = 1.0 # 1 Segundo máximo de vigencia
        
    async def get_fast_bid_ask(self, symbol: str) -> Tuple[float, float]:
        """
        [NANO-LATENCY] Fetches best BID and ASK from cache. 
        Falls back to Ticker if cache expired, avoiding blocking fetch_order_book in the critical path.
        """
        import time
        current_time = time.time()
        
        # 1. Use pure cache if available (0 latency)
        if symbol in self.order_book_cache and (current_time - self.order_book_cache[symbol]['timestamp'] < self.cache_ttl):
            bids = self.order_book_cache[symbol]['data']['bids']
            asks = self.order_book_cache[symbol]['data']['asks']
            if bids and asks:
                return bids[0][0], asks[0][0]
                
        # ⚡ FASE 10: Zero-Latency Websocket Fallback
        from data.data_provider import get_data_provider
        dp = get_data_provider()
        if dp:
            # Remover /USDT para obtener el symbol interno
            internal_sym = symbol.replace('/', '') if Config.BINANCE_USE_FUTURES else symbol
            ob = dp.get_orderbook(internal_sym)
            if ob:
                bid, ask = ob.get_best_bid_ask()
                if bid > 0 and ask > 0:
                    return bid, ask
            # Fallback a última vela si el OB no está listo
            bars = dp.get_latest_bars(internal_sym, 1)
            # Manejar tanto DataFrames de Pandas como Numpy Arrays
            has_data = False
            last_price = 0.0
            
            if bars is not None:
                if hasattr(bars, 'empty'): # Pandas
                    if not bars.empty and 'close' in bars.columns:
                        last_price = float(bars['close'].iloc[-1])
                        has_data = True
                elif isinstance(bars, np.ndarray): # Numpy Structured Array
                    if bars.size > 0 and 'close' in bars.dtype.names:
                        last_price = float(bars['close'][-1])
                        has_data = True
                        
            if has_data:
                return last_price, last_price
                
        # 3. Fallback to Ticker (REST API - Last Resort)
        try:
            if self.async_exchange:
                ticker = await self.async_exchange.fetch_ticker(symbol)
            else:
                ticker = self.exchange.fetch_ticker(symbol)
            bid = ticker['bid']
            ask = ticker['ask']
            if not bid or not ask:
                last = ticker['last']
                return last, last
            return float(bid), float(ask)
        except Exception as e:
            logger.warning(f"⚠️ [Fast Bid/Ask] Error fetching ticker for {symbol}: {e}")
            return 0.0, 0.0

    async def analyze_liquidity(self, symbol: str, quantity: float, side: str, bypass_guardian: bool = False) -> Dict:
        """
        Realiza el triple chequeo de liquidez antes de disparar.
        """
        # 🏎️ [L-001] Bypass de Makers: Salto a la velocidad de la luz
        if bypass_guardian:
            logger.info(f"⚡ [GUARDIAN BYPASSED] Limit-Maker order - Fast Tracked para {symbol}")
            return {
                "is_safe": True,
                "avg_fill_price": 0.0, # Bypass
                "slippage_pct": 0.0,
                "spread_pct": 0.0,
                "walls": {"bid_walls": [], "ask_walls": [], "avg_depth": 0},
                "reason": "Bypassed (Maker/Limit)"
            }
            
        try:
            # 1. Obtener el Snapshot (Top 20 Niveles) con CACHÉ
            symbol_api = symbol.replace('/', '') if Config.BINANCE_USE_FUTURES else symbol
            limit = 20
            
            import time
            current_time = time.time()
            if symbol in self.order_book_cache and (current_time - self.order_book_cache[symbol]['timestamp'] < self.cache_ttl):
                order_book = self.order_book_cache[symbol]['data']
            else:
                # ⚡ FASE 10: Zero-Latency Websocket OrderBook
                from data.data_provider import get_data_provider
                dp = get_data_provider()
                order_book = None
                
                if dp:
                    internal_sym = symbol.replace('/', '') if Config.BINANCE_USE_FUTURES else symbol
                    ob = dp.get_orderbook(internal_sym)
                    if ob:
                        # Extract Bids/Asks in ccxt format: [[price, qty], ...]
                        bids = [[p, q] for p, q in ob.bids.items()]
                        bids.sort(key=lambda x: x[0], reverse=True)
                        asks = [[p, q] for p, q in ob.asks.items()]
                        asks.sort(key=lambda x: x[0])
                        order_book = {'bids': bids[:limit], 'asks': asks[:limit]}
                
                if not order_book:
                    # Bloqueante REST API Call evitado mediante Async Exchange
                    if self.async_exchange:
                        order_book = await self.async_exchange.fetch_order_book(symbol, limit=limit)
                    else:
                        order_book = self.exchange.fetch_order_book(symbol, limit=limit)
                    
                self.order_book_cache[symbol] = {'data': order_book, 'timestamp': current_time}
            
            bids = order_book['bids'] # Compras [[price, qty], ...]
            asks = order_book['asks'] # Ventas
            
            if not bids or not asks:
                return {"is_safe": False, "reason": "Empty Order Book"}
            
            # 2. Detector de Muros (Walls)
            walls = self._detect_walls(bids, asks)
            
            # 3. Calculador de Precio Real (True Price)
            avg_fill_price, total_slippage_pct = self._calculate_slippage(
                bids if side.upper() == 'SELL' else asks, 
                quantity
            )
            
            # --- PHASE 9: SPREAD GUARD ---
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread_pct = abs(best_ask - best_bid) / best_bid
            
            max_spread = 0.001  # 0.1% Max Spread for Scalping
            is_toxic_spread = spread_pct > max_spread
            
            # --- PHASE 9: DEPTH PRESSURE ---
            # Si el volumen en el BBO es < 2% del volumen promedio del book, es liquidez 'fake' o tóxica
            best_bid_qty = bids[0][1]
            best_ask_qty = asks[0][1]
            avg_vol = np.mean([b[1] for b in bids] + [a[1] for a in asks])
            
            is_low_depth = (best_bid_qty < (avg_vol * 0.05)) or (best_ask_qty < (avg_vol * 0.05))
            
            # --- PHASE 33: SPOOFING DETECTION ---
            is_spoofing, spoofing_reason = self._detect_spoofing(symbol, bids, asks)
            
            # 4. Evaluación de Seguridad
            # Bloquear si hay un muro gigante en contra del movimiento
            is_blocked_by_wall = False
            wall_reason = ""
            
            if side.upper() == 'BUY' and walls['ask_walls']:
                # Muro en ventas (resistencia)
                closest_wall_price = walls['ask_walls'][0][0]
                if closest_wall_price < (avg_fill_price * 1.005): # Muro a menos de 0.5%
                    is_blocked_by_wall = True
                    wall_reason = f"Sell Wall detected at {closest_wall_price}"
            
            elif side.upper() == 'SELL' and walls['bid_walls']:
                # Muro en compras (soporte)
                closest_wall_price = walls['bid_walls'][0][0]
                if closest_wall_price > (avg_fill_price * 0.995): # Muro a menos de 0.5%
                    is_blocked_by_wall = True
                    wall_reason = f"Buy Wall detected at {closest_wall_price}"
            
            # Bloquear si el slippage es demasiado alto (> 0.2% para $13 capital es mortal)
            is_high_slippage = total_slippage_pct > 0.002 
            
            is_safe = not is_blocked_by_wall and not is_high_slippage and not is_toxic_spread and not is_low_depth
            
            reason = "Safe"
            if is_blocked_by_wall: reason = wall_reason
            elif is_high_slippage: reason = f"High Slippage ({total_slippage_pct:.4%})"
            elif is_toxic_spread: reason = f"Toxic Spread ({spread_pct:.4%})"
            elif is_low_depth: reason = "Flash/Low Depth"
            
            return {
                "is_safe": is_safe,
                "avg_fill_price": avg_fill_price,
                "slippage_pct": total_slippage_pct,
                "spread_pct": spread_pct,
                "walls": walls,
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"⚠️ LiquidityGuardian Error: {e}")
            return {"is_safe": True, "reason": "Error fallback (Safe mode)"} # Fail open to avoid blocking trades during API errors

    def check_order_book_imbalance(self, symbol: str, side: str, depth: int = 10, threshold_ratio: float = 2.5) -> Tuple[bool, str]:
        """
        🚀 FASE 22: Nano-Timeframe Validation
        Analiza el desbalance del Order Book para evitar entrar justo antes de un micro-dump o micro-pump.
        Retorna (is_safe, reason).
        """
        import time
        current_time = time.time()
        order_book = None
        
        if symbol in self.order_book_cache and (current_time - self.order_book_cache[symbol]['timestamp'] < self.cache_ttl):
            order_book = self.order_book_cache[symbol]['data']
        else:
            from data.data_provider import get_data_provider
            dp = get_data_provider()
            if dp:
                internal_sym = symbol.replace('/', '') if getattr(Config, 'BINANCE_USE_FUTURES', False) else symbol
                ob = dp.get_orderbook(internal_sym)
                if ob:
                    bids = [[p, q] for p, q in ob.bids.items()]
                    bids.sort(key=lambda x: x[0], reverse=True)
                    asks = [[p, q] for p, q in ob.asks.items()]
                    asks.sort(key=lambda x: x[0])
                    order_book = {'bids': bids, 'asks': asks}
        
        if not order_book or not order_book['bids'] or not order_book['asks']:
            return True, "No OB data, assuming safe"
            
        bids = order_book['bids'][:depth]
        asks = order_book['asks'][:depth]
        
        bid_vol = sum([b[1] for b in bids])
        ask_vol = sum([a[1] for a in asks])
        
        if bid_vol == 0 or ask_vol == 0:
            return True, "OB too thin"
            
        # Si vamos LONG, queremos que los BIDs apoyen. Si los ASKs (ventas) aplastan a los BIDs, es un muro en contra.
        if side.upper() == 'BUY':
            if ask_vol > bid_vol * threshold_ratio:
                return False, f"OB Imbalance AGAINST LONG: Asks({ask_vol:.2f}) > Bids({bid_vol:.2f}) x {threshold_ratio}"
        elif side.upper() == 'SELL':
            if bid_vol > ask_vol * threshold_ratio:
                return False, f"OB Imbalance AGAINST SHORT: Bids({bid_vol:.2f}) > Asks({ask_vol:.2f}) x {threshold_ratio}"
                
        return True, "OB Balanced"

    def _detect_walls(self, bids: List, asks: List) -> Dict:
        """
        Identifica niveles con volumen anormalmente alto (Phase 7).
        """
        avg_bid_vol = np.median([b[1] for b in bids]) if bids else 1.0
        avg_ask_vol = np.median([a[1] for a in asks]) if asks else 1.0
        
        bid_walls = [b for b in bids if b[1] > (avg_bid_vol * self.wall_threshold_multiplier)]
        ask_walls = [a for a in asks if a[1] > (avg_ask_vol * self.wall_threshold_multiplier)]
        
        return {
            "bid_walls": bid_walls[:3], # Top 3 muros de compra
            "ask_walls": ask_walls[:3], # Top 3 muros de venta
            "avg_depth": (avg_bid_vol + avg_ask_vol) / 2
        }

    def _detect_spoofing(self, symbol, current_bids, current_asks):
        """
        Phase 33: Spoofing Detection.
        Compara el libro actual con el anterior. Si un muro gigante desaparece SIN ser comido (tradeado),
        es probable spoofing.
        """
        is_spoofing = False
        reason = ""
        
        if symbol in self.book_history:
            prev = self.book_history[symbol]
            prev_bids = prev['bids']
            prev_asks = prev['asks']
            
            # Detectar desaparición de muros en BIDs
            if prev_bids and current_bids:
                prev_best_bid_vol = prev_bids[0][1]
                curr_best_bid_vol = current_bids[0][1]
                
                # Si el volumen cae drásticamente (> 50%) sin cambio de precio significativo
                if (prev_best_bid_vol > 5.0) and (curr_best_bid_vol < prev_best_bid_vol * 0.2):
                     is_spoofing = True
                     reason = f"Spoofing detected: Bid Wall vanished ({prev_best_bid_vol:.2f} -> {curr_best_bid_vol:.2f})"

            # Detectar desaparición de muros en ASKs
            if prev_asks and current_asks:
                prev_best_ask_vol = prev_asks[0][1]
                curr_best_ask_vol = current_asks[0][1]
                
                if (prev_best_ask_vol > 5.0) and (curr_best_ask_vol < prev_best_ask_vol * 0.2):
                     is_spoofing = True
                     reason = f"Spoofing detected: Ask Wall vanished ({prev_best_ask_vol:.2f} -> {curr_best_ask_vol:.2f})"

        # Actualizar memoria
        self.book_history[symbol] = {'bids': current_bids[:1], 'asks': current_asks[:1]}
        return is_spoofing, reason

    def _calculate_slippage(self, levels: List, quantity: float) -> Tuple[float, float]:
        """
        Camina por el libro de órdenes para calcular el precio real de ejecución.
        """
        remaining_qty = quantity
        total_cost = 0.0
        best_price = levels[0][0]
        
        for price, level_qty in levels:
            fill_qty = min(remaining_qty, level_qty)
            total_cost += fill_qty * price
            remaining_qty -= fill_qty
            
            if remaining_qty <= 0:
                break
        
        # Si la orden es más grande que todo el snapshot, el slippage es masivo
        if remaining_qty > 0:
            avg_price = total_cost / (quantity - remaining_qty)
            slippage = 0.01 # 1% penalización mínima por iliquidez
        else:
            avg_price = total_cost / quantity
            slippage = abs(avg_price - best_price) / best_price
            
        return avg_price, slippage
