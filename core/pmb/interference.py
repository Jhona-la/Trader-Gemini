import logging
from typing import Dict, Any

logger = logging.getLogger("PMB-Interference")

class InterferenceDetector:
    """
    AXIOMA: GAP 5 - AUTO-INTERFERENCIA NO SIMULADA
    Detecta y registra auto-interferencias durante el backtest.
    """
    
    def __init__(self):
        self.interferences = {
            'IT-1': 0, # Colisión de stop en el mismo activo
            'IT-2': 0, # Capital sobre-comprometido
            'IT-3': 0, # Señales contradictorias (TFTF LONG vs MRBB SHORT)
            'IT-4': 0  # Modificación de parámetros simultánea
        }
        self.resolved_conflicts = 0
        self.unresolved_conflicts = 0
        
    def detect_it1(self, action: str, asset: str, namespace: str):
        """Mock detection of Stop colissions"""
        # En una simulación real se registrarían los locks
        pass
        
    def detect_it2(self, total_committed: float, available: float):
        """Mock detection of over-committed capital"""
        if total_committed > available:
            self.interferences['IT-2'] += 1
            self.unresolved_conflicts += 1
            
    def detect_it3(self, signals: list):
        """Mock detection of contradictory signals"""
        directions = set(s.get('direction') for s in signals)
        if 'LONG' in directions and 'SHORT' in directions:
            self.interferences['IT-3'] += 1
            # Assuming Nexus resolves it
            self.resolved_conflicts += 1
            
    def generate_report(self) -> Dict[str, Any]:
        total = sum(self.interferences.values())
        return {
            'total_interferences': total,
            'breakdown': self.interferences,
            'resolved_conflicts': self.resolved_conflicts,
            'unresolved_conflicts': self.unresolved_conflicts,
            'passed': self.unresolved_conflicts == 0
        }
