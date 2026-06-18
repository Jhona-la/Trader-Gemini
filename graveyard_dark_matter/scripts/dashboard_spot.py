"""
Dashboard for Spot Trading Instance
Runs on port 8502 to avoid conflict with Futures dashboard
"""

import sys
import config_spot as Config
sys.modules['config'] = Config

# Now import and run the dashboard app
from dashboard.app import *

if __name__ == "__main__":
    # This will be run by streamlit
    pass
