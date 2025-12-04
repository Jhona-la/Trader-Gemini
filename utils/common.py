"""
Common Utility Functions for Trader Gemini

This module consolidates duplicate code patterns found across the codebase
to follow the DRY (Don't Repeat Yourself) principle.
"""

def build_testnet_urls():
    """
    Generate standardized Binance Testnet/Demo URL configuration for CCXT.
    
    This function was previously duplicated in:
    - binance_loader.py (lines 33-62)
    - binance_executor.py (lines 63-93)
    
    Returns:
        dict: Complete URL configuration for Spot, USD-M Futures, and COIN-M Futures
    """
    # Base URLs (Official 2024 Binance Testnet URLs)
    spot_testnet_base = 'https://testnet.binance.vision/api' # SPOT Testnet
    futures_testnet_base = 'https://demo-fapi.binance.com'  # USD-M Testnet
    delivery_testnet_base = 'https://testnet.binancefuture.com'  # COIN-M Testnet
    
    return {
        # SPOT (Standard)
        'public': f'{spot_testnet_base}/v3',
        'private': f'{spot_testnet_base}/v3',
        'api': {
            'public': f'{spot_testnet_base}/v3',
            'private': f'{spot_testnet_base}/v3',
        },
        
        # FUTURES (USD-M)
        'fapiPublic': f'{futures_testnet_base}/fapi/v1',
        'fapiPrivate': f'{futures_testnet_base}/fapi/v1',
        'fapiData': f'{futures_testnet_base}/fapi/v1',
        'fapiPrivateV2': f'{futures_testnet_base}/fapi/v2',
        'fapiPrivateV3': f'{futures_testnet_base}/fapi/v3',
        
        # DELIVERY (COIN-M)
        'dapiPublic': f'{delivery_testnet_base}/dapi/v1',
        'dapiPrivate': f'{delivery_testnet_base}/dapi/v1',
        'dapiData': f'{delivery_testnet_base}/dapi/v1',
        
        # SAPI (Not supported on Testnet)
        'sapi': f'{spot_testnet_base}/v3',
    }


def validate_non_zero(value, name):
    """
    Validate that a value is not None and not zero.
    
    Args:
        value: The value to validate (typically a quantity or price)
        name: Descriptive name for error messages
        
    Returns:
        bool: True if valid, False otherwise
        
    Raises:
        ValueError: If value is None or zero
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")
    
    if isinstance(value, (int, float)) and value == 0:
        raise ValueError(f"{name} cannot be zero")
    
    return True


def format_position_for_display(position_dict):
    """
    Format a position dictionary for consistent display across the system.
    
    Args:
        position_dict: Dictionary with keys: quantity, avg_price, current_price
        
    Returns:
        str: Formatted position string
    """
    if not position_dict:
        return "No position"
    
    qty = position_dict.get('quantity', 0)
    avg_price = position_dict.get('avg_price', 0)
    current_price = position_dict.get('current_price', avg_price)
    
    # Calculate PnL
    if qty > 0:  # LONG position
        pnl_pct = ((current_price - avg_price) / avg_price) * 100 if avg_price > 0 else 0
        direction = "LONG"
    elif qty < 0:  # SHORT position
        pnl_pct = ((avg_price - current_price) / avg_price) * 100 if avg_price > 0 else 0
        direction = "SHORT"
    else:
        return "No position"
    
    pnl_sign = "+" if pnl_pct >= 0 else ""
    color = "🟢" if pnl_pct >= 0 else "🔴"
    
    return f"{color} {direction} {abs(qty):.4f} @ ${avg_price:.2f} ({pnl_sign}{pnl_pct:.2f}%)"


def safe_float_conversion(value, default=0.0):
    """
    Safely convert a value to float, handling None and invalid values.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        float: Converted value or default
    """
    if value is None:
        return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def calculate_position_value(quantity, price, leverage=1):
    """
    Calculate the notional value of a position.
    
    Args:
        quantity: Position size
        price: Entry/current price
        leverage: Leverage multiplier (default 1 for Spot)
        
    Returns:
        float: Position value in quote currency
    """
    if quantity is None or price is None:
        return 0.0
    
    return abs(quantity) * price / leverage
