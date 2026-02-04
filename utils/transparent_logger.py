from utils.logger import logger

class TransparentLogger:
    """
    Shim class for TransparentLogger.
    Wraps standard logger to resolve import errors.
    """
    @staticmethod
    def info(msg):
        logger.info(msg)
        
    @staticmethod
    def error(msg):
        logger.error(msg)
        
    @staticmethod
    def warning(msg):
        logger.warning(msg)
        
    @staticmethod
    def debug(msg):
        logger.debug(msg)
