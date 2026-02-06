import os
import json
from typing import List, Dict, Any
from utils.logger import logger
from config import Config

class MarketScanner:
    def __init__(self, data_provider):
        self.data_provider = data_provider
        self.toxic_assets = ['RENDER/USDT'] 
        self.loyalty_file = os.path.join(Config.DATA_DIR, "scanner_loyalty.json")
        self.loyalty_data = self._load_loyalty()
        self.active_basket = [] # Symbols currently selected
        
        # Scoring weights
        self.VOL_WEIGHT = 0.6
        self.VOLATILITY_WEIGHT = 0.4
        self.LOYALTY_BONUS = 0.05 # 5% bonus for each time it stayed in Top 20

    def _load_loyalty(self) -> Dict[str, int]:
        if os.path.exists(self.loyalty_file):
            try:
                with open(self.loyalty_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_loyalty(self):
        try:
            os.makedirs(os.path.dirname(self.loyalty_file), exist_ok=True)
            with open(self.loyalty_file, 'w') as f:
                json.dump(self.loyalty_data, f)
        except Exception as e:
            logger.error(f"Scanner: Failed to save loyalty: {e}")

    def get_top_ranked_symbols(self, limit: int = 15) -> List[str]:
        """
        Ranks symbols with Loyalty and Hysteresis.
        limit: Target size of the basket (e.g. 15-20)
        """
        logger.info(f"🔍 Scanning market with Loyalty System (Limit: {limit})...")
        try:
            client = self.data_provider.client_sync
            if not client: return []
                
            tickers = client.get_ticker()
            futures_tickers = [
                t for t in tickers 
                if t['symbol'].endswith('USDT') 
                and not any(toxic in t['symbol'] for toxic in ['UPUSDT', 'DOWNUSDT', 'BULLUSDT', 'BEARUSDT'])
            ]
            
            ranked_data = []
            for t in futures_tickers:
                symbol_raw = t['symbol']
                internal_symbol = f"{symbol_raw[:-4]}/USDT"
                
                if internal_symbol in self.toxic_assets: continue
                    
                volume = float(t['quoteVolume'])
                high, low = float(t['highPrice']), float(t['lowPrice'])
                volatility = (high - low) / low if low > 0 else 0
                
                # Raw Score
                raw_score = (volume * self.VOL_WEIGHT) + (volatility * 1000000 * self.VOLATILITY_WEIGHT)
                
                # Apply Loyalty Bonus
                loyalty_count = self.loyalty_data.get(internal_symbol, 0)
                final_score = raw_score * (1 + (loyalty_count * self.LOYALTY_BONUS))
                
                ranked_data.append({
                    'symbol': internal_symbol,
                    'score': final_score,
                    'raw_score': raw_score
                })
            
            # Sort all items
            ranked_data.sort(key=lambda x: x['score'], reverse=True)
            
            # --- HYSTERESIS & PATIENCE LOGIC ---
            # 1. Take Top N (limit) as "Candidate Basket"
            candidates = [d['symbol'] for d in ranked_data[:limit]]
            
            # 2. Update Loyalty for candidates
            for sym in candidates:
                self.loyalty_data[sym] = self.loyalty_data.get(sym, 0) + 1
            
            # 3. Retention Check: If a symbol was already active, keep it unless it's way out (Top 30)
            final_selection = []
            
            # Mandatory symbols first
            mandatory = ['BTC/USDT', 'ETH/USDT']
            for m in mandatory:
                if m not in final_selection: final_selection.append(m)

            # Re-evaluate previous active_basket
            for sym in self.active_basket:
                if sym in mandatory: continue
                # Find current rank
                rank = next((i for i, d in enumerate(ranked_data) if d['symbol'] == sym), 999)
                if rank < (limit + 10): # Patience buffer: if it's still Top 25 (for a limit of 15)
                    if sym not in final_selection:
                        final_selection.append(sym)
                else:
                    logger.info(f"📉 Scanner: Dropping {sym} (Rank {rank} is too low).")
                    # Penalty: reduce loyalty if it drops out significantly
                    self.loyalty_data[sym] = max(0, self.loyalty_data.get(sym, 0) - 2)

            # Fill remaining slots with new candidates
            for sym in candidates:
                if len(final_selection) >= limit: break
                if sym not in final_selection:
                    final_selection.append(sym)
            
            self.active_basket = final_selection
            self._save_loyalty()
            
            logger.info(f"💎 Elite Basket ({len(final_selection)} assets): {', '.join(final_selection)}")
            return final_selection
            
        except Exception as e:
            logger.error(f"Scanner Error: {e}")
            return self.active_basket if self.active_basket else Config.CRYPTO_FUTURES_PAIRS[:limit]
