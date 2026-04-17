
import sys
print("Starting import...")
try:
    import ccxt
    print("Import successful!")
    print(ccxt.__version__)
except Exception as e:
    print(f"Import failed: {e}")
except KeyboardInterrupt:
    print("Import interrupted by user.")
