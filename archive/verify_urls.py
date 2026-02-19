import ccxt
from config import Config

def verify():
    print("--- Verifying Binance Futures Testnet Connection URLs ---")
    
    # Simulate the bot's initialization
    options = {'adjustForTimeDifference': True, 'defaultType': 'future'}
    exchange = ccxt.binance({'options': options})
    
    # Enable Sandbox Mode (as per our fix)
    exchange.set_sandbox_mode(True)
    
    print(f"Default Type: {exchange.options['defaultType']}")
    # print(f"Sandbox Mode: {exchange.sandbox}") # Removed to avoid AttributeError
    
    # Check URLs
    urls = exchange.urls
    api_urls = urls.get('api', {})
    
    print("\n[Active Endpoints]")
    if isinstance(api_urls, dict):
        for key, url in api_urls.items():
            print(f"{key}: {url}")
    else:
        print(f"API URL: {api_urls}")
        
    # VERIFICATION AGAINST DOCS
    # Expected: https://testnet.binancefuture.com
    
    is_valid = False
    target_url = "testnet.binancefuture.com"
    
    # CCXT structure for binance often has 'fapiPublic', 'fapiPrivate', etc.
    # or just 'public', 'private' pointing to the fapi domain.
    
    found_urls = str(api_urls)
    if target_url in found_urls:
        print(f"\n✅ SUCCESS: Found '{target_url}' in configuration.")
        is_valid = True
    else:
        print(f"\n❌ FAILURE: Did not find '{target_url}'. Connection might be wrong.")
        
    return is_valid

if __name__ == "__main__":
    verify()
