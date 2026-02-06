from strategies import ml_strategy
try:
    print(f"✅ joblib in ml_strategy: {ml_strategy.joblib}")
except AttributeError:
    print("❌ joblib NOT found in ml_strategy module")
except Exception as e:
    print(f"💥 Error: {e}")
