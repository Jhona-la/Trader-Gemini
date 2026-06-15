import sys
import os
import queue

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies.omni_strategy import OmniStrategy

class MockData:
    pass

try:
    events_queue = queue.Queue()
    data_provider = MockData()
    omni = OmniStrategy(data_provider, events_queue, horizon="SCALPING")
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
