import sys
import os

os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from core.engine import Engine
import risk.risk_manager

print("Importing technical")
from strategies.technical import HybridScalpingStrategy
print("Importing ml_strategy")
from strategies.ml_strategy import MLStrategy
print("DONE")
