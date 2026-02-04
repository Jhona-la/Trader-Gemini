import numpy as np
import pandas as pd

def safe_div(a, b, fill_value=0.0):
    """
    Safely divide a by b, replacing division by zero or NaN results with fill_value.
    Supports both scalar and array-like inputs (lists, numpy arrays, pandas Series).
    
    Args:
        a: Numerator
        b: Denominator
        fill_value: Value to return where division is invalid (default: 0.0)
        
    Returns:
        Result of division with invalid values replaced by fill_value.
    """
    # Handle scalar case specifically to avoid overhead/types issues if simple floats
    if np.isscalar(a) and np.isscalar(b):
        if b == 0 or pd.isna(b) or pd.isna(a):
            return fill_value
        return a / b

    # Convert to numpy arrays for vectorized operation
    # using strict float type to allow NaN/Inf handling
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    
    # Perform division ignoring invalid errors (handled by valid_mask later)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a_arr, b_arr)
    
    # Identify invalid indices:
    # 1. Infinite results (division by zero)
    # 2. NaN results (0/0 or NaN input)
    # mask = ~np.isfinite(result) # This catches Inf and NaN
    
    # More explicit replacement:
    result[~np.isfinite(result)] = fill_value
    
    return result
