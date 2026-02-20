"""
Professional logging configuration for Trader Gemini
Replaces print() statements with structured logging (JSON)
"""
import re
import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
# Try to use orjson for ultra-fast serialization (Phase 3: Hardware Opt)
try:
    import orjson
    def json_dumps(obj):
        return orjson.dumps(obj).decode('utf-8')
except ImportError:
    import json
    import json
    def json_dumps(obj):
        return json.dumps(obj)

class SensitiveDataFilter(logging.Filter):
    """
    🛡️ [PHASE I] VANGUARDIA-SOBERANA: Secure Logging
    Masks sensitive patterns (API Keys, Secrets) in log records.
    """
    def filter(self, record):
        if not isinstance(record.msg, str): return True
        msg = record.msg
        # Mask Binance API Keys (Typical 64 chars)
        # Simple heuristic: key=VALUE where VALUE is long alphanumeric
        msg = re.sub(r'(api_key|secret|token|password)=([a-zA-Z0-9\-\_]{8,})', r'\1=********************', msg, flags=re.IGNORECASE)
        record.msg = msg
        return True

class JSONFormatter(logging.Formatter):
    """
    Custom formatter to output logs in JSON format for machine ingestion (Splunk/ELK).
    """
    def format(self, record):
        # AEGIS-ULTRA: Loki-Compatible JSON Structure (Phase 8)
        log_record = {
            "ts": datetime.fromtimestamp(record.created).isoformat(), # ISO 8601
            "level": record.levelname,
            "component": getattr(record, "component", record.name),
            "trade_id": getattr(record, "trade_id", None),
            "strategy_id": getattr(record, "strategy_id", None),
            "msg": record.getMessage(),
            "line": record.lineno,
            "file": record.filename
        }
        
        # Prune None values (Bandwidth Opt)
        log_record = {k: v for k, v in log_record.items() if v is not None}
        
        # Include exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        return json_dumps(log_record)

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
    
    # 🛡️ [VANGUARDIA-SOBERANA] Attach Sensitive Data Filter
    secure_filter = SensitiveDataFilter()
    logger.addFilter(secure_filter)
    
    # Prevent duplicate handlers if logger already exists
    if logger.handlers:
        return logger
        
    # --- 1. PREPARE HANDLERS (But don't attach them to logger directly yet) ---
    
    # A. Console Handler (Human Readable)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # PHASE 46: HFT_STREAMING Telemetry (Microseconds)
    if os.getenv('HFT_LOG_MODE') == 'STREAMING':
        console_format = logging.Formatter(
            '%(asctime)s.%(msecs)03d %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
    console_handler.setFormatter(console_format)
    
    # Determine filenames based on mode and process
    suffix = os.getenv('BOT_MODE', '') # e.g. 'spot' or 'futures'
    date_str = datetime.now().strftime("%Y%m%d")
    
    # Process differentiation (Phase 17.1 Fix)
    import sys
    process_suffix = ""
    if "dashboard" in sys.argv[0] or "streamlit" in sys.argv[0]:
        process_suffix = "_dashboard"
    elif "check_oracle" in sys.argv[0]:
        process_suffix = "_oracle"
    elif "walk_forward" in sys.argv[0]:
        process_suffix = "_tester"
    
    if suffix:
        main_log_name = f'bot_{suffix}{process_suffix}_{date_str}.json'
        error_log_name = f'error_{suffix}{process_suffix}_{date_str}.json'
    else:
        main_log_name = f'bot{process_suffix}_{date_str}.json'
        error_log_name = f'error{process_suffix}_{date_str}.json'
        
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

    # 🌑 PHASE 24: IOPS OPTIMIZATION (Memory Buffered Logging)
    # Wraps FileHandlers in MemoryHandler to buffer logs in RAM and flush in chunks.
    # This reduces SSD wear and IO Interrupts (Nadir-Soberano).
    from logging.handlers import MemoryHandler
    
    # Buffer: 1000 records or flush on ERROR
    buffered_file_handler = MemoryHandler(
        capacity=1000, 
        flushLevel=logging.ERROR, 
        target=file_handler
    )
    
    buffered_error_handler = MemoryHandler(
        capacity=1000, 
        flushLevel=logging.CRITICAL, 
        target=error_handler
    )

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
        buffered_file_handler, 
        buffered_error_handler,
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
    """Stop the async log listener gracefully and flush the queue"""
    if hasattr(logger, '_listener'):
        try:
            # First, check if there are pending logs
            pending = logger._listener.queue.qsize()
            if pending > 0:
                print(f"⏳ Flushing {pending} logs before shutdown...")
            
            logger._listener.stop()
            # Remove handlers to prevent leaks on reload
            while logger.handlers:
                logger.removeHandler(logger.handlers[0])
            print("✅ Logging system stopped gracefully.")
        except Exception as e:
            print(f"⚠️ Error stopping logger: {e}")

def get_log_queue_status():
    """Returns the current size of the log queue to monitor backpressure"""
    if hasattr(logger, '_listener'):
        return logger._listener.queue.qsize()
    return 0


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
