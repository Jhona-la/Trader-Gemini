"""
⚡ PHASE OMNI: ADAPTIVE WORKLOAD BALANCER
===========================================
QUÉ: Motor de distribución adaptativa de carga de CPU entre símbolos.
POR QUÉ: Símbolos de baja volatilidad desperdician ciclos de procesamiento;
         los de alta volatilidad necesitan prioridad inmediata.
PARA QUÉ: Maximizar la velocidad de reacción del engine en los mercados
           que ofrecen más oportunidades en cada instante.
CÓMO: Cada 30s, clasifica símbolos por ATR/precio (volatilidad relativa).
      Los Top-N reciben prioridad de procesamiento (0.1 a 1.0 weight).
CUÁNDO: Llamado periódicamente por Engine._optimize_ryzen_resources().
DÓNDE: core/adaptive_balancer.py → integrado en Engine event loop.
QUIÉN: Engine, MarketRegimeDetector, DataHandler.

DEPENDENCIAS CRÍTICAS:
- config.py → Config.TRADING_PAIRS (lista de símbolos activos)
- data/binance_loader.py → get_latest_bars() para obtener datos ATR
- core/engine.py → consume get_processing_order() para ordenar estrategias
"""

import time
import numpy as np
from typing import Dict, List, Optional
from utils.logger import setup_logger

logger = setup_logger("AdaptiveBalancer")


class AdaptiveBalancer:
    """
    🧠 Adaptive Workload Balancer (Phase OMNI)
    
    Distributes processing priority based on real-time volatility.
    Higher ATR/price ratio = higher processing priority.
    
    Rebalance interval: 30 seconds (configurable).
    Smoothing: Exponential Moving Average (alpha=0.3) to avoid jitter.
    """
    
    def __init__(self, symbols: List[str], rebalance_interval: float = 30.0):
        self.symbols = list(symbols)
        self.rebalance_interval = rebalance_interval
        self._last_rebalance = 0.0
        
        # Priority map: symbol → weight [0.1, 1.0]
        # 1.0 = highest priority (most volatile / most opportunity)
        # 0.1 = lowest priority (flat / no opportunity)
        self.priority_map: Dict[str, float] = {s: 1.0 for s in symbols}
        
        # Historical volatility tracking (EMA smoothed)
        self._vol_ema: Dict[str, float] = {s: 0.0 for s in symbols}
        self._ema_alpha = 0.3  # Smoothing factor
        
        # Metrics
        self._rebalance_count = 0
        self._last_top_symbols: List[str] = []
        
        logger.info(f"⚡ [AdaptiveBalancer] Initialized for {len(symbols)} symbols")
    
    def update_volatility(self, symbol: str, atr_pct: float):
        """
        Update the volatility estimate for a symbol.
        
        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            atr_pct: ATR as percentage of price (e.g. 0.005 = 0.5%)
        """
        if symbol not in self._vol_ema:
            self._vol_ema[symbol] = atr_pct
            return
        
        # EMA smoothing to prevent whipsaw
        old = self._vol_ema[symbol]
        self._vol_ema[symbol] = self._ema_alpha * atr_pct + (1 - self._ema_alpha) * old
    
    def rebalance(self, volatility_dict: Optional[Dict[str, float]] = None):
        """
        Recalculates priority weights based on current volatility.
        
        Args:
            volatility_dict: Optional dict {symbol: atr_pct}. 
                           If provided, updates all symbols at once.
        """
        now = time.time()
        if now - self._last_rebalance < self.rebalance_interval:
            return  # Too soon
        
        self._last_rebalance = now
        
        # Batch update if provided
        if volatility_dict:
            for sym, vol in volatility_dict.items():
                self.update_volatility(sym, vol)
        
        # Sort by smoothed volatility (descending)
        valid_syms = [(s, self._vol_ema[s]) for s in self.symbols]
        sorted_syms = sorted(valid_syms, key=lambda x: x[1], reverse=True)
        
        n = len(sorted_syms)
        if n == 0:
            return
        
        # Assign priority weights: top symbols → 1.0, bottom → 0.1
        for rank, (sym, vol) in enumerate(sorted_syms):
            # Linear decay from 1.0 to 0.1 based on rank
            weight = max(0.1, 1.0 - (rank / max(n, 1)) * 0.9)
            self.priority_map[sym] = round(weight, 3)
        
        self._rebalance_count += 1
        
        # Log top 5 for telemetry
        top_5 = sorted_syms[:5]
        self._last_top_symbols = [s for s, _ in top_5]
        
        if self._rebalance_count % 10 == 0:  # Log every 10th rebalance
            top_str = ", ".join(f"{s}({v:.4f})" for s, v in top_5)
            logger.info(f"⚡ [Balancer] Rebalance #{self._rebalance_count}: Top → {top_str}")
    
    def get_processing_order(self) -> List[str]:
        """
        Returns symbols sorted by priority (highest first).
        Used by Engine to determine strategy evaluation order.
        """
        return sorted(
            self.priority_map.keys(), 
            key=lambda s: self.priority_map[s], 
            reverse=True
        )
    
    def get_priority(self, symbol: str) -> float:
        """Returns the priority weight for a single symbol."""
        return self.priority_map[symbol]
    
    def should_skip(self, symbol: str, threshold: float = 0.15) -> bool:
        """
        Returns True if a symbol's priority is below threshold.
        Used for aggressive CPU savings in ZOMBIE/CHOPPY regimes.
        """
        return self.priority_map[symbol] < threshold
    
    def add_symbol(self, symbol: str):
        """Hot-add a new symbol to the balancer."""
        if symbol not in self.priority_map:
            self.symbols.append(symbol)
            self.priority_map[symbol] = 0.5  # Neutral priority until data arrives
            self._vol_ema[symbol] = 0.0
    
    def remove_symbol(self, symbol: str):
        """Hot-remove a symbol from the balancer."""
        if symbol in self.priority_map:
            self.symbols.remove(symbol)
            del self.priority_map[symbol]
            del self._vol_ema[symbol]
    
    def get_metrics(self) -> Dict:
        """Returns balancer metrics for dashboard/telemetry."""
        return {
            'rebalance_count': self._rebalance_count,
            'top_symbols': self._last_top_symbols[:5],
            'priority_map': dict(sorted(
                self.priority_map.items(), 
                key=lambda x: x[1], reverse=True
            )[:10]),  # Top 10 only
            'last_rebalance': self._last_rebalance,
        }
