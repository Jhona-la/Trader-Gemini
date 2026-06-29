from dataclasses import dataclass
from typing import Optional
from core.events import SignalEvent
from core.asset_classifier import AssetClassifier, AssetClass

@dataclass
class SegmentPolicy:
    execution_type: str        # 'MAKER_ONLY', 'TAKER_TOLERANT'
    capital_allocation_pct: float 
    trailing_aggression: str   # 'TIGHT', 'STRUCTURED', 'MEAN_REV'
    max_hold_minutes: int
    veto: bool = False

class SegmentPolicyEngine:
    """
    Enforces distinct operational rules based on the Asset Class and Market Regime.
    Prevents memecoins from being traded like majors, and vice-versa.
    """
    def __init__(self, classifier: AssetClassifier):
        self.classifier = classifier
        
    def enforce_policy(self, signal: SignalEvent, current_regime_str: str) -> SignalEvent:
        """
        Injects the mathematical policy into the signal before it reaches the MetaArbitrator.
        """
        asset_class = self.classifier.get_class(signal.symbol)
        
        policy = self._matrix_lookup(asset_class, current_regime_str, signal.horizon)
        
        # Inject the policy into the event
        # (Since SignalEvent is frozen, we use object.__setattr__)
        object.__setattr__(signal, 'segment_policy', policy)
        return signal
        
    def _matrix_lookup(self, asset_class: AssetClass, regime: str, horizon: str) -> SegmentPolicy:
        """
        The core intelligence matrix defining how to treat different assets in different environments.
        """
        is_meme = (asset_class == AssetClass.MEME)
        is_major = (asset_class == AssetClass.MAJOR)
        
        # --- MEMECOIN POLICIES ---
        if is_meme:
            if "CHOPPY" in regime or "ZOMBIE" in regime:
                # VETO chopped memes. No liquidity, no trend = death by theta/spread.
                return SegmentPolicy("TAKER_TOLERANT", 0.0, "TIGHT", 0, veto=True)
                
            # Memecoins require aggressive entry, tight trailing, very short holds, and small capital.
            # Max hold for a meme scalp is 15 mins. For swing maybe 60 mins.
            max_hold = 15 if horizon == "SCALPING" else 60
            
            return SegmentPolicy(
                execution_type="TAKER_TOLERANT",  # Allow crossing the spread if needed
                capital_allocation_pct=0.30,      # Only use 30% of available margin per symbol
                trailing_aggression="TIGHT",      # Lock in profits at the slightest reversal
                max_hold_minutes=max_hold,
                veto=False
            )
            
        # --- MAJORS POLICIES ---
        if is_major:
            if "TRENDING" in regime:
                max_hold = 240 if horizon == "SCALPING" else 1440 # 4h scalp, 24h swing
                return SegmentPolicy(
                    execution_type="MAKER_ONLY",  # Strict BBO Post-Only
                    capital_allocation_pct=0.80,  # Trust the major with more capital
                    trailing_aggression="STRUCTURED", # Give it room to breathe
                    max_hold_minutes=max_hold,
                    veto=False
                )
            elif "RANGING" in regime:
                max_hold = 60 if horizon == "SCALPING" else 240
                return SegmentPolicy(
                    execution_type="MAKER_ONLY",
                    capital_allocation_pct=0.50,
                    trailing_aggression="MEAN_REV",
                    max_hold_minutes=max_hold,
                    veto=False
                )
            else: # Choppy / Zombie
                return SegmentPolicy(
                    execution_type="MAKER_ONLY",
                    capital_allocation_pct=0.20,
                    trailing_aggression="TIGHT",
                    max_hold_minutes=30,
                    veto=False
                )
                
        # --- ALTS POLICIES (Default) ---
        max_hold = 60 if horizon == "SCALPING" else 480
        return SegmentPolicy(
            execution_type="MAKER_ONLY", # Standard limit entries
            capital_allocation_pct=0.50, # Standard allocation
            trailing_aggression="STRUCTURED",
            max_hold_minutes=max_hold,
            veto=False
        )
