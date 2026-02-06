
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
    
    def __init__(self, exchange):
        self.exchange = exchange
        self.wall_threshold_multiplier = 5.0 # Un muro es 5x el volumen promedio del book
        
    def analyze_liquidity(self, symbol: str, quantity: float, side: str) -> Dict:
        """
        Realiza el triple chequeo de liquidez antes de disparar.
        """
        try:
            # 1. Obtener el Snapshot (Top 20 Niveles)
            symbol_api = symbol.replace('/', '') if Config.BINANCE_USE_FUTURES else symbol
            # CCXT fetch_order_book handles the internal mapping
            limit = 20
            order_book = self.exchange.fetch_order_book(symbol, limit=limit)
            
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
