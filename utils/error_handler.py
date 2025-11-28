"""
Binance-specific error handling with retry logic
Based on official Binance API error codes
"""
import time
from functools import wraps
from utils.logger import logger


# Binance API Error Codes (from official docs)
BINANCE_ERROR_CODES = {
    -1000: "UNKNOWN - An unknown error occurred",
    -1001: "DISCONNECTED - Internal error; unable to process your request",
    -1002: "UNAUTHORIZED - You are not authorized to execute this request",
    -1003: "TOO_MANY_REQUESTS - Too many requests; current limit is exceeded",
    -1006: "UNEXPECTED_RESP - Unexpected response from server",
    -1007: "TIMEOUT - Request timed out",
    -1013: "INVALID_MESSAGE - Invalid message",
    -1014: "UNKNOWN_ORDER_COMPOSITION - Unsupported order combination",
    -1015: "TOO_MANY_ORDERS - Too many new orders",
    -1016: "SERVICE_SHUTTING_DOWN - Service is shutting down",
    -1020: "UNSUPPORTED_OPERATION - Operation not supported",
    -1021: "INVALID_TIMESTAMP - Timestamp too far in the past/future",
    -1022: "INVALID_SIGNATURE - Signature for request is invalid",
    -2010: "NEW_ORDER_REJECTED - Order rejected by Binance",
    -2011: "CANCEL_REJECTED - Order cancel rejected",
    -2013: "NO_SUCH_ORDER - Order does not exist",
    -2014: "BAD_API_KEY_FMT - API-key format invalid",
    -2015: "REJECTED_MBX_KEY - Invalid API-key, IP, or permissions",
    -4000: "INVALID_ORDER_STATUS - Invalid order status",
    -4001: "PRICE_LESS_THAN_ZERO - Price less than 0",
    -4002: "PRICE_GREATER_THAN_MAX - Price greater than max",
    -4003: "QTY_LESS_THAN_ZERO - Quantity less than 0",
    -4004: "QTY_LESS_THAN_MIN_QTY - Quantity less than minimum",
    -4005: "QTY_GREATER_THAN_MAX_QTY - Quantity greater than maximum",
    -4006: "STOP_PRICE_LESS_THAN_ZERO - Stop price less than 0",
    -4007: "STOP_PRICE_GREATER_THAN_MAX - Stop price greater than max",
}


class BinanceAPIError(Exception):
    """Custom exception for Binance API errors"""
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(f"Binance API Error {code}: {message}")


def parse_binance_error(error):
    """
    Parse CCXT error to extract Binance error code
    CCXT wraps Binance errors, we need to extract the code
    """
    try:
        # CCXT might include error in different formats
        error_str = str(error)
        
        # Try to find error code in format: {"code":-1021,"msg":"..."}
        if '"code":' in error_str:
            import re
            match = re.search(r'"code":(-?\d+)', error_str)
            if match:
                code = int(match.group(1))
                msg = BINANCE_ERROR_CODES.get(code, f"Unknown error: {error_str}")
                return code, msg
        
        # If no code found, return generic error
        return None, str(error)
    except Exception as e:
        logger.debug(f"Could not parse Binance error: {e}")
        return None, str(error)


def retry_on_api_error(max_retries=3, base_delay=1.0):
    """
    Decorator to retry API calls on temporary errors (rate limits, timeouts)
    
    Uses exponential backoff: wait 1s, 2s, 4s, etc.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    code, msg = parse_binance_error(e)
                    
                    # Errors that should be retried
                    should_retry = code in [-1003, -1007, -1001, -1006] if code else False
                    
                    if should_retry and retries < max_retries - 1:
                        wait_time = base_delay * (2 ** retries)
                        logger.warning(f"API Error {code}: {msg}. Retrying in {wait_time}s... ({retries+1}/{max_retries})")
                        time.sleep(wait_time)
                        retries += 1
                    else:
                        # Don't retry - log and raise
                        logger.error(f"API Error {code}: {msg}")
                        raise
            
            # Max retries exceeded
            logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
            raise Exception(f"API call failed after {max_retries} retries")
        
        return wrapper
    return decorator


def handle_order_error(error, symbol, direction, quantity):
    """
    Specific handler for order execution errors
    Provides actionable feedback
    """
    error_str = str(error)
    
    # DO NOT SUPPRESS - All order execution errors are critical
    # Previous suppression of capital/config/getall was hiding real execution failures
    
    code, msg = parse_binance_error(error)
    
    if code == -2010:  # NEW_ORDER_REJECTED
        logger.error(f"Order REJECTED for {symbol}: {msg}")
        logger.info(f"  → Check: Sufficient balance, valid quantity, symbol status")
    elif code == -1013:  # INVALID_MESSAGE
        logger.error(f"Invalid order parameters for {symbol}")
        logger.info(f"  → Direction: {direction}, Quantity: {quantity}")
    elif code == -1021:  # INVALID_TIMESTAMP
        logger.error(f"Timestamp error - system clock may be out of sync")
        logger.info(f"  → Solution: Sync system time or adjust 'recvWindow' parameter")
    elif code == -2015:  # REJECTED_MBX_KEY
        logger.error(f"API Key rejected - check permissions and IP whitelist")
    elif code == -4004:  # QTY_LESS_THAN_MIN
        logger.error(f"Quantity {quantity} below minimum for {symbol}")
    elif code:
        logger.error(f"Order error {code}: {msg}")
    else:
        logger.error(f"Unknown order error: {error}")


def handle_balance_error(error):
    """Handle balance fetching errors"""
    code, msg = parse_binance_error(error)
    
    if code == -2015:
        logger.error("Cannot fetch balance: Invalid API key or permissions")
        logger.info("  → Check: API key has 'Enable Reading' permission")
    elif code == -1021:
        logger.error("Timestamp error when fetching balance")
        logger.info("  → Sync system clock")
    else:
        logger.error(f"Balance fetch error: {msg}")
