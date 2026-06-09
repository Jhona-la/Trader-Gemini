"""
🎯 MICRO OPTIMIZED STRATEGY — Wrapper para cuentas de $13 USD
==============================================================
QUÉ: Estrategia wrapper que adapta HybridScalpingStrategy para micro cuentas.
POR QUÉ: Con $13 USD, cada trade debe ser viable económicamente (cubrir fees).
PARA QUÉ: Filtrar trades donde el target profit < breakeven threshold.
CÓMO: Hereda de HybridScalpingStrategy y añade filtro MicroAccountAwareness.
CUÁNDO: Solo si se instancia explícitamente (NO registrada en main.py por defecto).
DÓNDE: strategies/micro_optimized.py
QUIÉN: MicroOptimizedStrategy → HybridScalpingStrategy → Portfolio

ESTADO: LEGACY — No registrada en producción. MicroAccountAwareness se usa
  directamente en main.py (L467) y BinanceExecutor (L504) sin este wrapper.
  Mantenida para compatibilidad y posible uso futuro.

DEPENDENCIAS CRÍTICAS:
- strategies/technical.py → HybridScalpingStrategy (parent)
- core/micro_awareness.py → MicroAccountAwareness (fee viability filter)
"""
from typing import Dict, Optional
from .technical import HybridScalpingStrategy
from core.micro_awareness import MicroAccountAwareness
from core.genotype import Genotype


class MicroOptimizedStrategy(HybridScalpingStrategy):
    """
    Wrapper que añade filtrado de viabilidad económica para micro cuentas.
    
    FORENSIC FIX: El __init__ original llamaba super().__init__() sin
    argumentos, causando TypeError instantáneo. Ahora pasa correctamente
    data_provider y events_queue al padre.
    """
    
    def __init__(self, data_provider, events_queue, micro_awareness: MicroAccountAwareness,
                 genotype: Genotype = None, horizon: str = "SCALPING"):
        # FORENSIC FIX: Pass required args to parent
        super().__init__(
            data_provider=data_provider,
            events_queue=events_queue,
            genotype=genotype,
            horizon=horizon
        )
        self.micro = micro_awareness
        
    def generate_micro_signal(self, symbol: str, data: Dict) -> Optional[Dict]:
        """
        Genera señal optimizada para micro cuenta.
        
        QUÉ: Filtra señales donde el profit esperado no cubre fees.
        POR QUÉ: Con $13 y fees de 0.06%, un TP de 0.10% deja solo 0.04% neto.
        PARA QUÉ: Solo ejecutar trades económicamente viables.
        """
        if not data or 'close' not in data:
            return None
            
        current_price = float(data['close'][-1]) if hasattr(data['close'], '__len__') else float(data['close'])
        
        # Verificar viabilidad para micro cuenta
        target_profit = self.TP_PCT  # Use horizon-aware TP
        is_viable, reason = self.micro.is_trade_viable(
            symbol, current_price, target_profit
        )
        
        if not is_viable:
            return None
            
        # Calcular tamaño viable
        size, adjusted = self.micro.calculate_viable_trade_size(
            symbol, current_price
        )
        
        return {
            'symbol': symbol,
            'size': size,
            'micro_optimized': True,
            'size_adjusted': adjusted,
            'min_target': self.micro.calculate_breakeven_threshold(size, current_price) * 1.5,
            'tp_pct': self.TP_PCT,
            'sl_pct': self.SL_PCT,
        }
