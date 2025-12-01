"""
Professional logging configuration for Trader Gemini
Replaces print() statements with structured logging
"""
import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime


def setup_logger(name='trader_gemini', log_dir='logs'):
    """
    Setup professional logger with rotating file handler
    
    Levels:
    - DEBUG: Detailed diagnostic info
    - INFO: General informational messages
    - WARNING: Warning messages (non-critical issues)
    - ERROR: Error messages (failures)
    - CRITICAL: Critical errors (system shutdown)
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Capture INFO and above
    
    # Prevent duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Console Handler (for terminal output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File Handler (rotating, max 10MB per file, keep 5 backups)
    # Allow suffix for Spot/Futures separation
    suffix = os.getenv('BOT_MODE', '') # e.g. 'spot' or 'futures'
    if suffix:
        filename = f'bot_{suffix}_{datetime.now().strftime("%Y%m%d")}.log'
    else:
        filename = f'bot_{datetime.now().strftime("%Y%m%d")}.log'
        
    log_file = os.path.join(log_dir, filename)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Capture everything to file
    file_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Create default logger for import
logger = setup_logger()


def log_trade(symbol, direction, quantity, price, strategy='Unknown'):
    """Helper to log trade execution"""
    logger.info(f"TRADE: {direction} {quantity} {symbol} @ ${price:.2f} (Strategy: {strategy})")


def log_error_with_context(error, context=''):
    """Helper to log errors with context"""
    logger.error(f"{context}: {str(error)}", exc_info=True)
