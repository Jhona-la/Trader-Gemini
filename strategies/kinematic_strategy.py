import os
import json
import numpy as np
from datetime import datetime, timezone
from core.events import SignalEvent
from core.enums import SignalType
from strategies.strategy import Strategy
from config import Config
from numba import njit
from utils.logger import logger

@njit(fastmath=True, nogil=True)
def calculate_sma_numba(arr, window):
    n = len(arr)
    res = np.zeros(n, dtype=np.float64)
    if n > window:
        for i in range(window - 1, n):
            suma = 0.0
            for j in range(window):
                suma += arr[i - j]
            res[i] = suma / window
    return res

@njit(fastmath=True, nogil=True)
def calculate_bollinger_bands_numba(closes, window, num_std):
    n = len(closes)
    upper = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    sma = calculate_sma_numba(closes, window)
    
    if n > window:
        for i in range(window - 1, n):
            suma_sq = 0.0
            for j in range(window):
                diff = closes[i - j] - sma[i]
                suma_sq += diff * diff
            std = np.sqrt(suma_sq / window)
            upper[i] = sma[i] + (num_std * std)
            lower[i] = sma[i] - (num_std * std)
            
    return upper, lower

class KinematicStrategy(Strategy):
    """
    Estrategia Cinemática Pura (Zero ML).
    Solo opera símbolos que han sobrevivido la auditoría Walk-Forward OOS
    con el Risk Manager asimétrico.
    """
    def __init__(self, data_provider, events_queue, symbol=None, horizon="SCALPING"):
        super().__init__()
        self.data_provider = data_provider
        self.events_queue = events_queue
        self.symbol = symbol
        self.horizon = horizon
        self.strategy_id = f"KINEMATIC_{horizon}"
        
        self.matrix_path = os.path.join(Config.BASE_DIR, "config", "quantum_kinematic_matrix.json")
        self.edge_config = None
        self.is_active = False
        
        self.load_matrix()

    def load_matrix(self):
        if not os.path.exists(self.matrix_path):
            logger.warning(f"Kinematic Matrix not found at {self.matrix_path}. Strategy dormant.")
            return
            
        with open(self.matrix_path, "r") as f:
            matrix = json.load(f)
            
        if self.symbol in matrix and matrix[self.symbol]["status"] == "ACTIVE":
            self.edge_config = matrix[self.symbol]
            self.is_active = True
            logger.info(f"✅ [{self.strategy_id}] Edge Cinemático Cargado para {self.symbol}: SL={self.edge_config['sl_pct']:.4f}, Trailing={self.edge_config['kinematic_umbral']:.4f}")
        else:
            logger.debug(f"[{self.strategy_id}] No Edge for {self.symbol}. Staying dormant.")

    def calculate_signals(self, event):
        if not self.is_active:
            return
            
        if event.type.name != 'MARKET' or event.symbol != self.symbol:
            return
            
        # Obtener 45 velas para calcular las Bandas de Bollinger de 40 periodos de forma segura
        bars = self.data_provider.get_latest_bars(self.symbol, timeframe="1m", n=45)
        if bars is None or len(bars) < 45:
            return
            
        closes = np.array([b[4] for b in bars], dtype=np.float64)
        
        upper, lower = calculate_bollinger_bands_numba(closes, 40, 2.0)
        
        if upper[-2] == 0.0:
            return
            
        signal_type = None
        
        # Breakout LONG (cierre anterior rompió arriba, y el previo al anterior estaba debajo)
        if closes[-2] > upper[-3] and closes[-3] <= upper[-4]:
            signal_type = SignalType.LONG
            
        # Breakout SHORT
        elif closes[-2] < lower[-3] and closes[-3] >= lower[-4]:
            signal_type = SignalType.SHORT
            
        if signal_type is not None:
            sl_pct = self.edge_config["sl_pct"]
            
            signal = SignalEvent(
                strategy_id=self.strategy_id,
                symbol=self.symbol,
                datetime=datetime.now(timezone.utc),
                signal_type=signal_type,
                strength=1.0,
                sl_pct=sl_pct,
                tp_pct=0.0, # Sin TP, dejamos correr
            )
            
            # Use dynamically injected attributes for risk manager compatibility
            signal.kinematic_umbral = self.edge_config["kinematic_umbral"]
            signal.horizon = self.horizon
            
            self.events_queue.put(signal)
            logger.info(f"🚀 [{self.strategy_id}] {self.symbol} KINEMATIC {signal_type.name} GENERADO (SL: {sl_pct:.4f})")
