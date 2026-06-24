import re
import os

app_path = r"c:\Users\jhona\Documents\Proyectos\Trader Gemini\dashboard\app.py"

with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace import
content = re.sub(r'import pandas as pd', 'import polars as pl', content)

# Replace pd.DataFrame() with pl.DataFrame()
content = re.sub(r'pd\.DataFrame\(\)', 'pl.DataFrame()', content)

# Replace pd.DataFrame(...) with pl.DataFrame(...)
content = re.sub(r'pd\.DataFrame\(', 'pl.DataFrame(', content)

# Replace pd.read_csv with pl.read_csv
content = re.sub(r'pd\.read_csv\(', 'pl.read_csv(', content)

# Handle simple datetime conversions (rough approximation for Streamlit usage)
# df['datetime'] = pd.to_datetime(df['datetime']) -> df = df.with_columns(pl.col("datetime").cast(pl.Datetime))
content = re.sub(r"([a-zA-Z0-9_]+)\['([a-zA-Z0-9_]+)'\] = pd\.to_datetime\(\1\['\2'\]\)",
                 r"\1 = \1.with_columns(pl.col('\2').cast(pl.Datetime, strict=False))", content)

# Any other remaining pd.to_datetime:
content = re.sub(r"pd\.to_datetime\((.*?)\)", r"pl.Series(\1).cast(pl.Datetime, strict=False)", content)

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Dashboard app.py migrated to Polars.")
