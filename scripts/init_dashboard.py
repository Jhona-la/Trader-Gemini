import pandas as pd
import os
from datetime import datetime

# Define path
status_path = "dashboard/data/status.csv"
os.makedirs(os.path.dirname(status_path), exist_ok=True)

# Create initial data
status_data = {
    'timestamp': [datetime.now()],
    'total_equity': [10000.0],
    'cash': [10000.0],
    'realized_pnl': [0.0],
    'unrealized_pnl': [0.0],
    'positions': ["{}"]
}
df_status = pd.DataFrame(status_data)

# Write to CSV
df_status.to_csv(status_path, index=False)
print(f"✅ Initial status written to {status_path}")
