"""
🌍 WORLD AWARENESS - Intelligence Layer for Global Session Sensitivity
========================================================================

PROFESSOR METHOD:
- QUÉ: Sistema de consciencia temporal y de liquidez global.
- POR QUÉ: Los mercados de Crypto operan 24/7 pero la liquidez y volatilidad varían drásticamente según la sesión abierta (Londres, NY, Tokyo).
- PARA QUÉ: Ajustar umbrales de confianza y tamaño de posición dinámicamente.
- CÓMO: Mapeo de horas UTC a puntajes de liquidez (0.0 - 1.0).
- CUÁNDO: Se consulta en cada MarketEvent para inyectar contexto a las estrategias.
- DÓNDE: core/world_awareness.py (Inyectado en Engine).
"""

from datetime import datetime, timezone
import numpy as np

class WorldAwareness:
    """
    Tracks global market sessions and calculates liquidity/activity factors.
    """
    
    # Session Hours (UTC)
    SESSIONS = {
        'SYDNEY': {'open': 22, 'close': 7},
        'TOKYO': {'open': 0, 'close': 9},
        'LONDON': {'open': 8, 'close': 17},
        'NEW_YORK': {'open': 13, 'close': 22}
    }
    
    def __init__(self):
        self.last_update = None
        self.current_score = 1.0  # Default to full activity
        
        # Cache for Optimization
        self._cached_context = None
        self._last_cache_time = 0.0
        self._cache_ttl = 60.0 # 1 minute TTL
        
    def get_market_context(self) -> dict:
        """
        Returns the current global market context.
        Optimized with 60s Cache to reduce CPU usage in Event Loop.
        """
        import time
        now_ts = time.time()
        
        # Check Cache
        if self._cached_context and (now_ts - self._last_cache_time) < self._cache_ttl:
            return self._cached_context
            
        now = datetime.now(timezone.utc)
        hour = now.hour
        
        active_sessions = []
        for name, times in self.SESSIONS.items():
            if times['open'] < times['close']:
                if times['open'] <= hour < times['close']:
                    active_sessions.append(name)
            else:  # Over midnight (e.g. Sydney)
                if hour >= times['open'] or hour < times['close']:
                    active_sessions.append(name)
        
        # Calculate Liquidity Score (LS)
        # LS = 1.0 (Prime: London/NY Overlap)
        # LS = 0.8 (Active: London or NY)
        # LS = 0.6 (Secondary: Tokyo/Sydney)
        # LS = 0.4 (Dead Zone: 22:00 - 00:00 UTC)
        
        ls = 0.4  # Base
        
        if 'LONDON' in active_sessions and 'NEW_YORK' in active_sessions:
            ls = 1.0
        elif 'LONDON' in active_sessions or 'NEW_YORK' in active_sessions:
            ls = 0.85
        elif 'TOKYO' in active_sessions:
            ls = 0.65
        elif 'SYDNEY' in active_sessions:
            ls = 0.55
            
        # Smoothing or Volatility Adjustment could go here
        self.current_score = ls
        
        context = {
            'timestamp': now.isoformat(),
            'active_sessions': active_sessions,
            'liquidity_score': ls,
            'is_prime': ls >= 0.85,
            'is_dead_zone': ls <= 0.45
        }
        
        # Update Cache
        self._cached_context = context
        self._last_cache_time = now_ts
        
        return context

# Global Instance
world_awareness = WorldAwareness()
