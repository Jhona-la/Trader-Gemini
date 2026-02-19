
import logging
import time
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("LiquidityGuardian")

class LiquidityGuardian:
    """
    🛡️ PHASE 12: LIQUIDITY GUARDIAN L3
    Protects execution by verifying Order Book integrity before trade submission.
    """
    
    def __init__(self, data_provider):
        self.data_provider = data_provider
        # Thresholds
        self.MAX_SPREAD_PCT = 0.0015  # 0.15% Max Spread allowed
        self.MIN_LIQUIDITY_RATIO = 0.1 # Need at least 10% of order size available at top
        self.IMBALANCE_THRESHOLD = 3.0 # If one side is 3x larger than other, caution.
        
    def check_liquidity(self, symbol, side, quantity, price=None):
        """
        Verifies if the market has enough quality liquidity to support the trade.
        Args:
            symbol: Trading pair
            side: 'BUY' or 'SELL'
            quantity: Amount to trade
            price: Intended price (optional)
        Returns:
            bool: True if safe, False if unsafe
            str: Reason if unsafe
        """
        snapshot = self.data_provider.get_liquidity_snapshot(symbol)
        
        if not snapshot:
            return False, "NO_LIQUIDITY_DATA"
            
        # Check staleness (latency guard)
        if time.time() - snapshot['ts'] > 2.0:
            return False, f"STALE_LIQUIDITY_DATA ({time.time() - snapshot['ts']:.2f}s ago)"
            
        bid = snapshot['bid']
        ask = snapshot['ask']
        bid_qty = snapshot['bid_qty']
        ask_qty = snapshot['ask_qty']
        
        # 1. Spread Check
        spread_pct = (ask - bid) / bid
        if spread_pct > self.MAX_SPREAD_PCT:
            return False, f"WIDE_SPREAD: {spread_pct*100:.4f}% > {self.MAX_SPREAD_PCT*100:.4f}%"
            
        # 2. Impact Cost / Liquidity Depth Check
        # If buying, we eat the ASK. If selling, we hit the BID.
        available_qty = ask_qty if side.upper() == 'BUY' else bid_qty
        
        # We want to be sure we are not eating the ENTIRE top level (slippage risk)
        # Ideally we take max 50% of the top level
        if quantity > available_qty * 0.5:
            return False, f"LOW_DEPTH: Order {quantity} > 50% of available {available_qty}"
            
        # 3. Order Book Imbalance (Spoofing/Collapse detection)
        # If we are BUYING, we want deep BIDS (support) vs ASKS.
        # But if Bids vanish (Collapse), price falls.
        total_depth = bid_qty + ask_qty
        bid_ratio = bid_qty / total_depth
        
        # Extreme imbalance detection
        # If Bids are 1% of depth, price will crash.
        if bid_ratio < 0.05:
            return False, "BID_LIQUIDITY_COLLAPSE (Bids < 5% of Book)"
        if bid_ratio > 0.95:
            return False, "ASK_LIQUIDITY_COLLAPSE (Asks < 5% of Book)"
            
        return True, "SAFE"

    def get_market_quality_score(self, symbol):
        """
        Returns a 0-100 score of market quality for this symbol.
        """
        snapshot = self.data_provider.get_liquidity_snapshot(symbol)
        if not snapshot: return 0
        
        bid = snapshot['bid']
        ask = snapshot['ask']
        spread_bps = ((ask - bid) / bid) * 10000
        
        score = 100
        
        # Spread Penalty
        # 1 bp = excellent, 10 bps = bad
        score -= (spread_bps * 5)
        
        # Imbalance Penalty
        ratio = snapshot['bid_qty'] / (snapshot['bid_qty'] + snapshot['ask_qty'])
        imbalance = abs(0.5 - ratio) # 0 is perfect balance, 0.5 is extreme
        score -= (imbalance * 40)
        
        return max(0, min(100, score))
