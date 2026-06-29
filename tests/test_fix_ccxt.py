import ccxt
import sys

def test_connection():
    print("Testing CCXT Connection...")
    try:
        # Attempt 1: set_sandbox_mode (Expected to fail based on logs)
        print("\n--- Attempt 1: set_sandbox_mode(True) ---")
        exchange = ccxt.binance({'options': {'defaultType': 'future'}})
        try:
            exchange.set_sandbox_mode(True)
            print("set_sandbox_mode(True) executed (Unexpected success if it was failing)")
        except Exception as e:
            print(f"Caught expected error: {e}")

        # Attempt 2: Manual URL Configuration
        print("\n--- Attempt 2: Manual URL Config ---")
        exchange_manual = ccxt.binance({
            'apiKey': 'test',
            'secret': 'test',
            'options': {'defaultType': 'future'},
            'urls': {
                'api': {
                    'public': 'https://testnet.binancefuture.com/fapi/v1',
                    'private': 'https://testnet.binancefuture.com/fapi/v1',
                }
            }
        })
        # We won't call set_sandbox_mode here
        print(f"Manual Config URLs: {exchange_manual.urls['api']}")
        print("Manual config created without error.")
        
    except Exception as e:
        print(f"General Error: {e}")

if __name__ == "__main__":
    test_connection()
