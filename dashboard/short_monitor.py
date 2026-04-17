import pandas as pd
import os
import json
from datetime import datetime, timedelta
from config import Config
from utils.logger import logger

class ShortMonitor:
    """
    Monitor asíncrono para evaluar métricas de operaciones SHORT.
    Calcula: short_avg_profit, short_expectancy, y compara con operaciones LONG.
    """
    def __init__(self, data_dir=Config.DATA_DIR):
        self.data_dir = data_dir
        self.trades_path = os.path.join(self.data_dir, "trades.csv")
        
    def get_short_metrics(self) -> dict:
        """Calcula métricas clave para posiciones SCALPING y SWING de tipo SHORT."""
        metrics = {
            'total_shorts': 0,
            'short_winrate': 0.0,
            'short_avg_profit': 0.0,
            'short_expectancy': 0.0,
            'short_pnl_total': 0.0,
            'best_short_symbol': "N/A"
        }
        
        if not os.path.exists(self.trades_path):
            return metrics
            
        try:
            df = pd.read_csv(self.trades_path)
            if df.empty or 'direction' not in df.columns:
                return metrics
                
            # Filter strictly for SHORT trades
            shorts = df[df['direction'].isin(['SHORT', 'SELL', -1, '-1'])]
            
            if shorts.empty:
                return metrics
                
            total_shorts = len(shorts)
            wins = shorts[shorts['net_pnl'] > 0]
            losses = shorts[shorts['net_pnl'] <= 0]
            
            win_rate = len(wins) / total_shorts if total_shorts > 0 else 0
            
            avg_win = wins['net_pnl'].mean() if not wins.empty else 0
            avg_loss = abs(losses['net_pnl'].mean()) if not losses.empty else 0
            
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            # Grupos
            sym_pnl = shorts.groupby('symbol')['net_pnl'].sum()
            best_sym = sym_pnl.idxmax() if not sym_pnl.empty and sym_pnl.max() > 0 else "N/A"
            
            metrics.update({
                'total_shorts': int(total_shorts),
                'short_winrate': float(win_rate),
                'short_avg_profit': float(shorts['net_pnl'].mean() if not shorts.empty else 0),
                'short_expectancy': float(expectancy),
                'short_pnl_total': float(shorts['net_pnl'].sum()),
                'best_short_symbol': str(best_sym)
            })
            
            logger.debug(f"📉 [ShortMonitor] Processed {total_shorts} short trades. WR: {win_rate:.0%}, Exp: ${expectancy:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating short metrics: {e}")
            return metrics

    def generate_report(self):
        """Generador de string para loggueo / debugging."""
        mets = self.get_short_metrics()
        report = (
            f"=== 📉 SHORT PERFORMANCE DIAGNOSTIC ==\n"
            f"Total Shorts    : {mets['total_shorts']}\n"
            f"Win Rate        : {mets['short_winrate']:.1%}\n"
            f"Expectancy      : ${mets['short_expectancy']:.3f}/trade\n"
            f"Total PnL       : ${mets['short_pnl_total']:.2f}\n"
            f"Best Symbol     : {mets['best_short_symbol']}\n"
            f"====================================="
        )
        return report

if __name__ == "__main__":
    monitor = ShortMonitor()
    print(monitor.generate_report())
