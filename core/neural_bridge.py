import numpy as np
from typing import Dict, Any, List, Tuple
from core.genotype import Genotype
from core.enums import SignalType
from utils.logger import logger

class NeuralBridge:
    """
    Puente Neuronal (Trinidad Omega - Phase 31).
    Traduce el caos del mercado en tensores ordenados para la 'Mente'.
    """
    
    def __init__(self, observation_window: int = 5):
        self.window = observation_window
        # Dimensions: 
        # Market (4 features * window) + VBI (1 * window) + Liq (1 * window) + Portfolio (3) + Genotype (2)
        self.input_dim = (6 * observation_window) + 3 + 2 

    def get_state_tensor(self, 
                         market_data: Any, # SimDataProvider window or equivalent
                         portfolio_state: Dict[str, float], 
                         genotype: Genotype) -> np.ndarray:
        """
        Construye el tensor de estado normalizado.
        """
        # Market Tensor: [Returns, Vol, RSI, MACD, VBI, Liq]
        market_tensor = np.zeros(6 * self.window)
        
        if market_data is not None:
            # We expect market_data to be a dict or structured array
            # If it's a dict, we extract fields
            closes = market_data.get('close', np.zeros(self.window + 1))
            volumes = market_data.get('volume', np.zeros(self.window))
            vbi = market_data.get('vbi', np.zeros(self.window))
            liq = market_data.get('liq', np.zeros(self.window))
            
            # A. Log Returns
            if len(closes) > 1:
                returns = np.diff(closes) / closes[:-1] 
                feat_returns = returns[-self.window:]
            else:
                feat_returns = np.zeros(self.window)
            
            # B. Volatility / Volume
            mean_vol = np.mean(volumes) if np.mean(volumes) > 0 else 1.0
            feat_vol = (volumes / mean_vol)[-self.window:]
            
            # C. RSI & MACD (Placeholders or passed in)
            feat_rsi = market_data.get('rsi', np.full(self.window, 0.5))[-self.window:]
            feat_macd = market_data.get('macd', np.full(self.window, 0.0))[-self.window:]
            
            # D. OMEGA MIND: VBI & Liq
            feat_vbi = vbi[-self.window:]
            feat_liq = liq[-self.window:]
            
            # Flatten Market Data
            market_tensor = np.concatenate([
                feat_returns, feat_vol, feat_rsi, feat_macd, 
                feat_vbi, feat_liq
            ])
        
        # 2. PORTFOLIO STATE
        # [HasPosition (0/1), UnrealizedPnL (clamped -1 to 1), TimeInTrade (normalized)]
        has_pos = 1.0 if portfolio_state.get('quantity', 0) != 0 else 0.0
        pnl_pct = portfolio_state.get('pnl_pct', 0.0)
        pnl_norm = np.clip(pnl_pct * 10, -1.0, 1.0) # Scale roughly 10% move = 1.0
        duration = portfolio_state.get('duration', 0)
        dur_norm = min(duration / 100.0, 1.0) # 100 bars max
        
        portfolio_tensor = np.array([has_pos, pnl_norm, dur_norm])
        
        # 3. GENOTYPE CONTEXT (Personality)
        # [RiskAversion (SL), Aggression (TP)] - Normalized
        # TP 1% -> 0.1, TP 10% -> 1.0
        sl_norm = min(genotype.genes.get('sl_pct', 0.02) * 10, 1.0)
        tp_norm = min(genotype.genes.get('tp_pct', 0.02) * 10, 1.0)
        
        gene_tensor = np.array([sl_norm, tp_norm])
        
        # COMBINE
        state = np.concatenate([market_tensor, portfolio_tensor, gene_tensor])
        
        # Final safety check for NaNs
        return np.nan_to_num(state, nan=0.0)

    def decode_action(self, action_probs: np.ndarray) -> Tuple[Any, float]:
        """
        Decodifica la salida de la Red Neuronal (Softmax/Logits) en una Señal.
        Action Space:
        0: HOLD
        1: BUY_LONG
        2: SELL_SHORT
        3: CLOSE_POSITION
        """
        # Ensure we have a 1D array
        if action_probs.ndim > 1:
            action_probs = action_probs.flatten()
            
        action_idx = np.argmax(action_probs)
        confidence = action_probs[action_idx]
        
        # Map index to Signal
        if action_idx == 0:
            return None, 0.0 # HOLD
        elif action_idx == 1:
            return SignalType.LONG, confidence
        elif action_idx == 2:
            return SignalType.SHORT, confidence
        elif action_idx == 3:
            # SignalType doesn't usually have CLOSE, but we can return None 
            # or a special flag. For compatibility, we might return None 
            # and let the Strategy handle 'Exit' logic if needed, 
            # OR we introduce a new signal type.
            # For Phase 32, let's treat it as a special metadata flag or negative strength?
            # Better: Return EXIT type if available, otherwise just handle as specialized logic
            # Let's import SignalType and see if we can extend or misuse it safely.
            # Assuming standard enum: LONG, SHORT.
            # If we enter LONG with strength 0, maybe that means close?
            # Let's return a specific tuple logic.
            # Let's return a specific tuple logic.
            return "CLOSE", confidence
            
        return None, 0.0

    # ------------------------------------------------------------------
    # PHASE 3: METAL-CORE BINARY PROTOCOL (MessagePack + Struct)
    # ------------------------------------------------------------------
    def pack_action_signal(self, signal: Any, confidence: float) -> bytes:
        """
        Pack Signal into 5 bytes (1B Type + 4B Float).
        0: None, 1: LONG, 2: SHORT, 3: CLOSE
        """
        import struct
        code = 0
        if signal == SignalType.LONG: code = 1
        elif signal == SignalType.SHORT: code = 2
        elif signal == "CLOSE": code = 3
        
        return struct.pack('Bf', code, confidence)

    def unpack_action_signal(self, payload: bytes) -> Tuple[Any, float]:
        """
        Unpack 5-byte Signal.
        """
        import struct
        if len(payload) != 5: return None, 0.0
        
        code, conf = struct.unpack('Bf', payload)
        
        if code == 1: return SignalType.LONG, conf
        elif code == 2: return SignalType.SHORT, conf
        elif code == 3: return "CLOSE", conf
        return None, conf

    def pack_tensor(self, tensor: np.ndarray) -> bytes:
        """
        Ultra-Fast Binary Serialization for IPC (Phase 3).
        Payload: {'d': bytes, 's': shape, 't': dtype}
        """
        import msgpack
        return msgpack.packb({
            'd': tensor.tobytes(),
            's': tensor.shape,
            't': str(tensor.dtype)
        }, use_bin_type=True)

    def unpack_tensor(self, payload: bytes) -> np.ndarray:
        """
        Zero-Allocation Unpack (Phase 3).
        """
        import msgpack
        data = msgpack.unpackb(payload, raw=False)
        # Verify dtype string to numpy dtype
        dt = np.dtype(data['t'])
        return np.frombuffer(data['d'], dtype=dt).reshape(data['s'])

    def publish_insight(self, strategy_id: str, symbol: str, insight: Dict[str, Any]):
        """
        [PHASE 3: Neural Insight]
        Broadcasts neural confidence and direction for monitoring.
        """
        # For now, pass to avoid blocking the hot path.
        pass

    def cleanup(self):
        """Phase 8: Cleanup neural bridge resources."""
        logger.info("🧠 [NeuralBridge] Cleaning up resources...")
        pass

# Global Instance
neural_bridge = NeuralBridge()
