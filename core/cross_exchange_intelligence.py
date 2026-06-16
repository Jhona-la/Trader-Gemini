import asyncio
from utils.logger import logger
from data.coinbase_loader import CoinbaseLoader
from data.bybit_loader import BybitLoader
from data.okx_loader import OKXLoader
from data.deribit_ws_loader import DeribitWSLoader
from core.global_state import global_state

class CrossExchangeIntelligenceEngine:
    """
    [PHASE V] - Multi-Source Intelligence Engine (Pre-Binance Price Discovery)
    
    QUÉ: Orquestador central de todos los websockets de exchanges externos.
    POR QUÉ: Centraliza la conexión y agrega inteligencia sobre el mercado global.
    PARA QUÉ: Proporcionar a Trader Gemini una ventaja injusta (Latencia Negativa) 
              frente a Binance mediante la lectura adelantada de líderes de precio y derivados.
    """
    
    def __init__(self, symbol_list):
        self.symbol_list = symbol_list
        self.coinbase = CoinbaseLoader(symbol_list)
        self.bybit = BybitLoader(symbol_list)
        self.okx = OKXLoader(symbol_list)
        self.deribit = DeribitWSLoader(symbol_list)
        
        # Iniciar el diccionario en el Global State para acceso O(1) desde el Engine
        if not hasattr(global_state, 'cross_exchange_metrics'):
            global_state.cross_exchange_metrics = {}
            for s in symbol_list:
                global_state.cross_exchange_metrics[s] = {
                    'cb_price': 0.0,
                    'cb_velocity': 0.0,
                    'cb_latency': 0,
                    'bybit_oi': 0.0,
                    'bybit_funding': 0.0,
                    'okx_funding': 0.0,
                    'deribit_gex': 0.0,
                    'deribit_pc_ratio': 1.0,
                    'pdc_signal': 0.0 # Price Discovery Coefficient
                }
                
        logger.info("🧠 [CrossExchangeEngine] Motor Multi-Fuente Inicializado.")

    async def start(self):
        """Inicia todos los WebSockets concurrentemente."""
        logger.info("🚀 [CrossExchangeEngine] Arrancando enjambre de WebSockets...")
        
        tasks = [
            asyncio.create_task(self.coinbase.start_websocket()),
            asyncio.create_task(self.bybit.start_websocket()),
            asyncio.create_task(self.okx.start_websocket()),
            asyncio.create_task(self.deribit.start_websocket())
        ]
        
        # Background task para calcular el PDC periódicamente
        tasks.append(asyncio.create_task(self._pdc_calculator_loop()))
        
        await asyncio.gather(*tasks)
        
    async def _pdc_calculator_loop(self):
        """
        Calcula el Price Discovery Coefficient (PDC) iterativamente en background.
        PDC = Coeficiente que indica si Coinbase y Bybit están adelantando a Binance (Lead-Lag).
        Rango: [-1.0, 1.0] donde 1.0 es Fuerte LEAD ALCISTA, -1.0 es Fuerte LEAD BAJISTA.
        """
        while True:
            try:
                for sym in self.symbol_list:
                    metrics = global_state.cross_exchange_metrics.get(sym, {})
                    cb_vel = metrics.get('cb_velocity', 0.0)
                    bybit_oi = metrics.get('bybit_oi', 0.0)
                    funding = metrics.get('bybit_funding', 0.0)
                    
                    # 1. Coinbase Lead-Lag (Velocity in % per sec)
                    # pdc_signal > 0 means strong upward lead from Coinbase
                    # pdc_signal < 0 means strong downward lead from Coinbase
                    pdc = 0.0
                    
                    # Umbral de aceleración (0.01% por segundo en Coinbase)
                    if cb_vel > 0.0001:
                        pdc += min(0.6, cb_vel * 1000) # Máx contribución 0.6
                    elif cb_vel < -0.0001:
                        pdc += max(-0.6, cb_vel * 1000)
                        
                    # 2. Open Interest Context (Bybit)
                    # Si el OI es muy alto o Funding es extremo, se suma Bias
                    # Altamente simplificado para el ejemplo
                    if funding > 0.001: # Funding extremadamente positivo (riesgo de long squeeze)
                        pdc -= 0.2
                    elif funding < -0.001: # Funding extremadamente negativo (riesgo de short squeeze)
                        pdc += 0.2
                        
                    # Limitar al rango estricto [-1.0, 1.0]
                    metrics['pdc_signal'] = max(-1.0, min(1.0, pdc))
                    
            except Exception as e:
                pass
            
            await asyncio.sleep(0.05) # Calculate every 50ms (Ultra-Low Latency)
            
    async def stop(self):
        await self.coinbase.stop()
        await self.bybit.stop()
        await self.okx.stop()
        await self.deribit.stop()
        logger.info("🛑 [CrossExchangeEngine] Apagado completo.")
