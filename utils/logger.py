"""
Professional logging configuration for Trader Gemini
Replaces print() statements with structured logging (JSON)
"""
import logging
import json
import traceback
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """
    Custom formatter to output logs in JSON format for machine ingestion (Splunk/ELK).
    """
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "path": record.pathname 
        }
        
        # Include exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        # Include stack trace if present
        if record.stack_info:
            log_record["stack_trace"] = self.formatStack(record.stack_info)
            
        return json.dumps(log_record)

def setup_logger(name='trader_gemini', log_dir='logs'):
    """
    Setup professional ASYNC logger with non-blocking I/O (QueueHandler).
    Critical for HFT to prevent disk writes from blocking the main loop.
    
    Levels:
    - CONSOLE: INFO+ (Text format)
    - FILE (MAIN): DEBUG+ (JSON format)
    - FILE (ERROR): ERROR+ (JSON format)
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture everything, handlers will filter
    
    # Prevent duplicate handlers if logger already exists
    if logger.handlers:
        return logger
        
    # --- 1. PREPARE HANDLERS (But don't attach them to logger directly yet) ---
    
    # A. Console Handler (Human Readable)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # Determine filenames based on mode
    suffix = os.getenv('BOT_MODE', '') # e.g. 'spot' or 'futures'
    date_str = datetime.now().strftime("%Y%m%d")
    
    if suffix:
        main_log_name = f'bot_{suffix}_{date_str}.json'
        error_log_name = f'error_{suffix}_{date_str}.json'
    else:
        main_log_name = f'bot_{date_str}.json'
        error_log_name = f'error_{date_str}.json'
        
    main_log_path = os.path.join(log_dir, main_log_name)
    error_log_path = os.path.join(log_dir, error_log_name)

    # B. Main File Handler (JSON, DEBUG+)
    file_handler = RotatingFileHandler(
        main_log_path,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter(datefmt='%Y-%m-%d %H:%M:%S'))
    
    # C. Error File Handler (JSON, ERROR+)
    error_handler = RotatingFileHandler(
        error_log_path,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR) # Only ERROR and CRITICAL
    error_handler.setFormatter(JSONFormatter(datefmt='%Y-%m-%d %H:%M:%S'))

    # --- 2. ASYNC INFRASTRUCTURE ---
    import queue
    from logging.handlers import QueueHandler, QueueListener
    
    # The Queue that will hold pending logs
    log_queue = queue.Queue(-1) # Infinite queue to prevent blocking
    
    # The QueueHandler pushes logs to the queue (Runs in Main Thread -> Fast)
    queue_handler = QueueHandler(log_queue)
    
    # The Listener reads from queue and writes to real handlers (Runs in Background Thread)
    # We include console_handler in listener too to prevent console blocking, 
    # though usually console is fast. Let's make EVERYTHING async.
    listener = QueueListener(
        log_queue, 
        console_handler, 
        file_handler, 
        error_handler,
        respect_handler_level=True
    )
    
    # Start the background thread
    listener.start()
    
    # Attach ONLY the QueueHandler to the main logger
    logger.addHandler(queue_handler)
    
    # Attach listener to logger so we can stop it later if needed (hacky but works)
    logger._listener = listener 
    
    # Phase 6: Cleanup old logs (>7 days)
    cleanup_old_logs(log_dir, days=7)
    
    return logger

def stop_logger():
    """Stop the async log listener gracefully"""
    if hasattr(logger, '_listener'):
        logger._listener.stop()


def cleanup_old_logs(log_dir: str, days: int = 7):
    """
    Remove log files older than specified days.
    """
    import glob
    from datetime import timedelta
    
    try:
        cutoff = datetime.now() - timedelta(days=days)
        pattern = os.path.join(log_dir, "*.json")
        
        for log_file in glob.glob(pattern):
            try:
                # Extract date from filename (bot_20260203.json)
                basename = os.path.basename(log_file)
                # Try to find date in filename
                for part in basename.split('_'):
                    if part.replace('.json', '').isdigit() and len(part.replace('.json', '')) == 8:
                        date_str = part.replace('.json', '')
                        file_date = datetime.strptime(date_str, "%Y%m%d")
                        if file_date < cutoff:
                            os.remove(log_file)
                            break
            except:
                pass  # Skip files that don't match expected format
    except Exception:
        pass  # Silent fail - log cleanup is not critical


# Create default logger for import
logger = setup_logger()


def log_trade(symbol, direction, quantity, price, strategy='Unknown'):
    """Helper to log trade execution"""
    # This renders nicely in console text format
    logger.info(f"TRADE: {direction} {quantity} {symbol} @ ${price:.2f} (Strategy: {strategy})")


def log_error_with_context(error, context=''):
    """Helper to log errors with context"""
    logger.error(f"{context}: {str(error)}", exc_info=True)
