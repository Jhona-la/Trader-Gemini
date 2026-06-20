"""
Sistema de monitoreo para micro cuentas
"""
from typing import Dict

class MicroPerformanceMonitor:
    def __init__(self, micro_awareness):
        self.micro = micro_awareness
        self.metrics = {
            'total_trades': 0,
            'viable_trades': 0,
            'non_viable_trades': 0,
            'adjusted_trades': 0,
            'total_fees': 0.0,
            'total_slippage': 0.0
        }
    
    def update_metrics(self, trade_data: Dict):
        """Actualiza métricas con datos del trade"""
        self.metrics['total_trades'] += 1
        
        if trade_data['micro_optimized']:
            self.metrics['viable_trades'] += 1
        else:
            self.metrics['non_viable_trades'] += 1
            
        if trade_data['size_adjusted']:
            self.metrics['adjusted_trades'] += 1
            
        self.metrics['total_fees'] += trade_data['fees']
        self.metrics['total_slippage'] += trade_data['slippage']
    
    def get_micro_report(self) -> str:
        """Genera reporte de performance micro"""
        viable_ratio = (self.metrics['viable_trades'] / 
                       self.metrics['total_trades'] * 100) if self.metrics['total_trades'] > 0 else 0
        
        return f"""
📊 REPORTE MICRO CUENTA

Balance: ${self.micro.balance:.2f}
Trades Totales: {self.metrics['total_trades']}
Trades Viables: {self.metrics['viable_trades']} ({viable_ratio:.1f}%)
Trades Ajustados: {self.metrics['adjusted_trades']}

💸 Costos Totales:
- Fees: ${self.metrics['total_fees']:.4f}
- Slippage: ${self.metrics['total_slippage']:.4f}

🎯 Eficiencia Micro: {self._calculate_efficiency_score():.2f}/1.0
"""
    def _calculate_efficiency_score(self) -> float:
        if self.metrics['total_trades'] == 0:
            return 0.0
        viable_ratio = self.metrics['viable_trades'] / self.metrics['total_trades']
        return viable_ratio * 0.8 + 0.2 # Score base
