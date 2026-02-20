from utils.logger import logger
from config import Config

class CostGuard:
    """
    💸 COMPONENT: Cost Guard (The Leak Preventer)
    QUÉ: Protege el capital de costos ocultos: Funding Rates excesivos y Slippage.
    POR QUÉ: En Futuros, un Funding del 0.1% puede comerse el 20% del profit de un scalp.
    """
    
    FUNDING_THRESHOLD = 0.0005 # 0.05% Funding Limit
    
    @staticmethod
    def check_funding_leak(exchange, symbol, side):
        """
        Verifica si el Funding Rate es adverso para la operación.
        Retorna: True si es seguro operar, False si hay riesgo de 'Leak'.
        """
        try:
            if not Config.BINANCE_USE_FUTURES:
                return True
                
            # Fetch Funding Rate
            funding_info = exchange.fetch_funding_rate(symbol)
            funding_rate = funding_info.get('fundingRate', 0)
            
            # Logic:
            # - Si vamos LONG y funding es POSITIVO -> Pagamos nosotros. (Malo si es muy alto)
            # - Si vamos SHORT y funding es NEGATIVO -> Pagamos nosotros. (Malo si es muy bajo)
            
            is_long = side.upper() == 'BUY'
            is_short = side.upper() == 'SELL'
            
            if is_long and funding_rate > CostGuard.FUNDING_THRESHOLD:
                logger.warning(f"💸 [CostGuard] High Funding Leak! Longing {symbol} with Rate {funding_rate*100:.4f}% > {CostGuard.FUNDING_THRESHOLD*100}%")
                return False
                
            if is_short and funding_rate < -CostGuard.FUNDING_THRESHOLD:
                logger.warning(f"💸 [CostGuard] High Funding Leak! Shorting {symbol} with Rate {funding_rate*100:.4f}% < -{CostGuard.FUNDING_THRESHOLD*100}%")
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ CostGuard Error: {e}")
            return True # Fail open (allow trade) to avoid paralysis
