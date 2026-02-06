from enum import Enum, auto

class ResolutionState(str, Enum):
    """
    State Machine for Dynamic Capital Allocation (Phase 14)
    """
    STABLE = "STABLE"     # Normal operation (Kelly Criterion)
    GROWTH = "GROWTH"     # Aggressive compounding (Profit > 10%)
    RECOVERY = "RECOVERY" # Defensive rebuilding (Drawdown > 5%)
