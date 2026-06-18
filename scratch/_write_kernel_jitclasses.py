from numba.experimental import jitclass
from numba import float64, int64, boolean
import numpy as np

zscore_spec = [
    ('period', int64),
    ('count', int64),
    ('buffer', float64[:]),
    ('idx', int64),
    ('sum_x', float64),
    ('sum_x2', float64),
    ('initialized', boolean)
]

@jitclass(zscore_spec)
class StatefulZScore:
    def __init__(self, period):
        self.period = period
        self.count = 0
        self.buffer = np.zeros(period, dtype=np.float64)
        self.idx = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.initialized = False

    def update(self, price: float) -> float:
        if not self.initialized:
            self.buffer[self.idx] = price
            self.sum_x += price
            self.sum_x2 += price * price
            self.idx += 1
            self.count += 1
            
            if self.count == self.period:
                self.initialized = True
                self.idx = 0
                mean = self.sum_x / self.period
                var = (self.sum_x2 - (self.sum_x * self.sum_x) / self.period) / self.period
                std = np.sqrt(max(0.0, var))
                if std > 1e-8:
                    return (price - mean) / std
                return 0.0
            return 0.0
        else:
            old_val = self.buffer[self.idx]
            self.buffer[self.idx] = price
            self.idx = (self.idx + 1) % self.period
            
            self.sum_x += (price - old_val)
            self.sum_x2 += (price * price - old_val * old_val)
            
            mean = self.sum_x / self.period
            var = (self.sum_x2 - (self.sum_x * self.sum_x) / self.period) / self.period
            std = np.sqrt(max(0.0, var))
            
            if std > 1e-8:
                return (price - mean) / std
            return 0.0

def generate_code():
    with open("utils/math_kernel.py", "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("# ==============================================================================\n")
        f.write("# ⚡ STATEFUL O(1) QUANTUM FEATURES (THERMODYNAMIC FRICTION KILLERS)\n")
        f.write("# ==============================================================================\n")
        f.write("from numba.experimental import jitclass\n")
        f.write("from numba import boolean\n\n")
        
        with open("scratch/_write_kernel_jitclasses.py", "r", encoding="utf-8") as me:
            lines = me.readlines()
            start = False
            for line in lines:
                if "zscore_spec =" in line:
                    start = True
                if "def generate_code" in line:
                    break
                if start:
                    f.write(line)
    print("✅ Appended kernel JIT classes to math_kernel.py")

if __name__ == "__main__":
    generate_code()
