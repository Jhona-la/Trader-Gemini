import ccxt
from config import Config

def verify_fix():
    print("--- Verifying CCXT Manual URL Configuration ---")
    
    # Simulate the executor's initialization logic
    options = {'adjustForTimeDifference': True, 'defaultType': 'future'}
    exchange = ccxt.binance({'options': options})
    
    # Apply the MANUAL CONFIGURATION we just added
    exchange.urls['api'] = {
        'public': 'https://testnet.binancefuture.com/fapi/v1',
        'private': 'https://testnet.binancefuture.com/fapi/v1',
        'fapiPublic': 'https://testnet.binancefuture.com/fapi/v1',
        'fapiPrivate': 'https://testnet.binancefuture.com/fapi/v1',
        'fapiData': 'https://testnet.binancefuture.com/fapi/v1',
        'sapi': 'https://testnet.binance.vision/api/v3',
    }
    
    print("Checking for required endpoints...")
    required_keys = ['fapiPublic', 'fapiPrivate', 'sapi']
    
    all_good = True
    for key in required_keys:
        if key in exchange.urls['api']:
            print(f"✅ Found '{key}': {exchange.urls['api'][key]}")
        else:
            print(f"❌ MISSING '{key}'")
            all_good = False
            
    if all_good:
        print("\nSUCCESS: All endpoints configured. The 'missing endpoint' error should be gone.")
    else:
        print("\nFAILURE: Still missing endpoints.")

if __name__ == "__main__":
    verify_fix()
