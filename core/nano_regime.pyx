# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
import cython

cdef enum RegimeState:
    TRENDING_BULL = 0
    TRENDING_BEAR = 1
    RANGING = 2
    CHOPPY = 3
    MEAN_REVERTING = 4
    UNKNOWN = 5

cdef dict STATE_TO_STR = {
    RegimeState.TRENDING_BULL: 'TRENDING_BULL',
    RegimeState.TRENDING_BEAR: 'TRENDING_BEAR',
    RegimeState.RANGING: 'RANGING',
    RegimeState.CHOPPY: 'CHOPPY',
    RegimeState.MEAN_REVERTING: 'MEAN_REVERTING',
    RegimeState.UNKNOWN: 'UNKNOWN'
}

cdef dict STR_TO_STATE = {v: k for k, v in STATE_TO_STR.items()}

cdef class NanoRegimeDetector:
    """
    O(1) Memory & Nanosecond Latency Regime Voter.
    Replaces dict/string iterations with C Enums and Arrays.
    """
    cdef dict last_regime
    cdef dict regime_history
    cdef dict regime_confidence
    cdef int hysteresis_window
    
    def __init__(self, int hysteresis_window=3):
        self.last_regime = {}
        self.regime_history = {}
        self.regime_confidence = {}
        self.hysteresis_window = hysteresis_window
        
    cpdef void set_hysteresis(self, int window):
        self.hysteresis_window = window
        
    cpdef tuple process_consensus(self, str symbol, str r_1m, str r_5m, str r_15m, str r_1h):
        """
        Calculates regime using static arrays and fixed weight constants instead of dict loops.
        Returns: (final_regime: str, confidence: float, emergency_exit: bool)
        """
        cdef float[6] votes
        cdef int i
        for i in range(6):
            votes[i] = 0.0
            
        cdef float total_weight = 0.0
        
        # 1m (Weight: 0.1)
        if r_1m is not None:
            votes[STR_TO_STATE.get(r_1m, RegimeState.UNKNOWN)] += 0.1
            total_weight += 0.1
            
        # 5m (Weight: 0.2)
        if r_5m is not None:
            votes[STR_TO_STATE.get(r_5m, RegimeState.UNKNOWN)] += 0.2
            total_weight += 0.2
            
        # 15m (Weight: 0.3)
        if r_15m is not None:
            votes[STR_TO_STATE.get(r_15m, RegimeState.UNKNOWN)] += 0.3
            total_weight += 0.3
            
        # 1h (Weight: 0.4)
        if r_1h is not None:
            votes[STR_TO_STATE.get(r_1h, RegimeState.UNKNOWN)] += 0.4
            total_weight += 0.4
            
        if total_weight == 0.0:
            return self.last_regime.get(symbol, 'UNKNOWN'), 0.0, False
            
        cdef int best_idx = RegimeState.UNKNOWN
        cdef float max_vote = -1.0
        
        for i in range(5):
            if votes[i] > max_vote:
                max_vote = votes[i]
                best_idx = i
                
        cdef float consensus_score = max_vote / total_weight
        
        # Divergence override
        if consensus_score < 0.40 and (best_idx == RegimeState.TRENDING_BULL or best_idx == RegimeState.TRENDING_BEAR):
            best_idx = RegimeState.CHOPPY
            
        cdef str raw_regime = STATE_TO_STR[best_idx]
        
        # Hysteresis update
        if symbol not in self.regime_history:
            self.regime_history[symbol] = []
            
        cdef list hist = self.regime_history[symbol]
        hist.append(raw_regime)
        
        cdef int hw = self.hysteresis_window
        while len(hist) > hw:
            hist.pop(0)
            
        self.regime_confidence[symbol] = consensus_score
        
        cdef str final_regime = 'UNKNOWN'
        cdef str previous_regime = self.last_regime.get(symbol, 'UNKNOWN')
        
        cdef bint all_same = True
        if len(hist) >= hw:
            for i in range(hw):
                if hist[i] != raw_regime:
                    all_same = False
                    break
            if all_same:
                final_regime = raw_regime
            else:
                final_regime = previous_regime
        else:
            final_regime = previous_regime
            
        self.last_regime[symbol] = final_regime
        
        cdef bint emergency_exit = False
        if previous_regime == 'CHOPPY' and final_regime == 'TRENDING_BEAR':
            emergency_exit = True
            
        return final_regime, consensus_score, emergency_exit
