from numba.experimental import jitclass
from numba import float64, int64, boolean
import numpy as np

# 1. EMA Stateful
ema_spec = [
    ('period', int64),
    ('alpha', float64),
    ('ema', float64),
    ('initialized', boolean),
    ('count', int64),
    ('sum', float64)
]

@jitclass(ema_spec)
class StatefulEMA:
    def __init__(self, period):
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self.ema = 0.0
        self.initialized = False
        self.count = 0
        self.sum = 0.0

    def update(self, price: float) -> float:
        if not self.initialized:
            self.sum += price
            self.count += 1
            if self.count == self.period:
                self.ema = self.sum / self.period
                self.initialized = True
            return self.ema if self.initialized else np.nan
        else:
            self.ema = (price * self.alpha) + (self.ema * (1.0 - self.alpha))
            return self.ema

# 2. RSI Stateful
rsi_spec = [
    ('period', int64),
    ('avg_gain', float64),
    ('avg_loss', float64),
    ('last_price', float64),
    ('initialized', boolean),
    ('count', int64),
    ('gain_sum', float64),
    ('loss_sum', float64)
]

@jitclass(rsi_spec)
class StatefulRSI:
    def __init__(self, period):
        self.period = period
        self.avg_gain = 0.0
        self.avg_loss = 0.0
        self.last_price = np.nan
        self.initialized = False
        self.count = 0
        self.gain_sum = 0.0
        self.loss_sum = 0.0

    def update(self, price: float) -> float:
        if np.isnan(self.last_price):
            self.last_price = price
            return np.nan
            
        change = price - self.last_price
        self.last_price = price
        
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0
        
        if not self.initialized:
            self.gain_sum += gain
            self.loss_sum += loss
            self.count += 1
            if self.count == self.period:
                self.avg_gain = self.gain_sum / self.period
                self.avg_loss = self.loss_sum / self.period
                self.initialized = True
                
                if self.avg_loss == 0:
                    return 100.0
                rs = self.avg_gain / self.avg_loss
                return 100.0 - (100.0 / (1.0 + rs))
            return np.nan
        else:
            self.avg_gain = ((self.avg_gain * (self.period - 1)) + gain) / self.period
            self.avg_loss = ((self.avg_loss * (self.period - 1)) + loss) / self.period
            
            if self.avg_loss == 0:
                return 100.0
            rs = self.avg_gain / self.avg_loss
            return 100.0 - (100.0 / (1.0 + rs))

# 3. Bollinger Stateful (Welford's Method)
bollinger_spec = [
    ('period', int64),
    ('num_std', float64),
    ('count', int64),
    ('buffer', float64[:]),
    ('idx', int64),
    ('sum_x', float64),
    ('sum_x2', float64),
    ('initialized', boolean)
]

@jitclass(bollinger_spec)
class StatefulBollinger:
    def __init__(self, period, num_std=2.0):
        self.period = period
        self.num_std = num_std
        self.count = 0
        self.buffer = np.zeros(period, dtype=np.float64)
        self.idx = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.initialized = False

    def update(self, price: float):
        # returns (upper, mid, lower)
        if not self.initialized:
            self.buffer[self.idx] = price
            self.sum_x += price
            self.sum_x2 += price * price
            self.idx += 1
            self.count += 1
            
            if self.count == self.period:
                self.initialized = True
                self.idx = 0 # reset for ring buffer
                
                mid = self.sum_x / self.period
                var = (self.sum_x2 - (self.sum_x * self.sum_x) / self.period) / self.period
                std = np.sqrt(max(0.0, var))
                return mid + (std * self.num_std), mid, mid - (std * self.num_std)
            return np.nan, np.nan, np.nan
        else:
            old_val = self.buffer[self.idx]
            self.buffer[self.idx] = price
            self.idx = (self.idx + 1) % self.period
            
            self.sum_x += (price - old_val)
            self.sum_x2 += (price * price - old_val * old_val)
            
            mid = self.sum_x / self.period
            var = (self.sum_x2 - (self.sum_x * self.sum_x) / self.period) / self.period
            std = np.sqrt(max(0.0, var))
            
            return mid + (std * self.num_std), mid, mid - (std * self.num_std)

def generate_code():
    with open("strategies/quant_math.py", "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("# ==============================================================================\n")
        f.write("# ⚡ STATEFUL O(1) JITCLASSES (THERMODYNAMIC FRICTION KILLERS)\n")
        f.write("# ==============================================================================\n")
        f.write("from numba.experimental import jitclass\n")
        f.write("from numba import boolean\n\n")
        
        with open("scratch/_write_jitclasses.py", "r", encoding="utf-8") as me:
            lines = me.readlines()
            # Extract classes
            start = False
            for line in lines:
                if "# 1. EMA Stateful" in line:
                    start = True
                if "def generate_code" in line:
                    break
                if start:
                    f.write(line)
    print("✅ Appended JIT classes to quant_math.py")

if __name__ == "__main__":
    generate_code()
