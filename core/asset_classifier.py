from enum import Enum
from typing import Dict, Any

class AssetClass(Enum):
    MAJOR = 1
    ALT = 2
    MEME = 3

class AssetClassifier:
    """
    Categorizes symbols into specific institutional classes to apply 
    different execution and risk policies.
    """
    # Hardcoded overrides for well-known assets
    KNOWN_MAJORS = {"BTC", "ETH"}
    KNOWN_MEMES = {"DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF"}
    
    def __init__(self):
        # We could inject graph centralities here if we wanted fully dynamic classification.
        # For now, we will rely on static mapping + dynamic overrides if needed.
        self._cache = {}

    def get_class(self, symbol: str) -> AssetClass:
        """
        Classifies the given symbol (e.g., 'BTC/USDT' or 'BTCUSDT') into an AssetClass.
        """
        if symbol in self._cache:
            return self._cache[symbol]
            
        base_asset = symbol.split('/')[0] if '/' in symbol else symbol.replace('USDT', '')
        
        if base_asset in self.KNOWN_MAJORS:
            asset_class = AssetClass.MAJOR
        elif base_asset in self.KNOWN_MEMES:
            asset_class = AssetClass.MEME
        else:
            asset_class = AssetClass.ALT
            
        self._cache[symbol] = asset_class
        return asset_class
