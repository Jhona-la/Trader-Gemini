import os
import pandas as pd
from datetime import datetime, timezone
from config import Config
from utils.logger import logger
from utils.analytics import AnalyticsEngine

class ReportGenerator:
    """
    📋 GENERADOR DE REPORTES PDF (Phase 4)
    
    PROFESSOR METHOD:
    - QUÉ: Sistema de exportación de resultados a documentos PDF/Excel.
    - POR QUÉ: Para mantener un registro formal del desempeño fuera del dashboard.
    - PARA QUÉ: Análisis a largo plazo y auditoría de la estrategia.
    - CÓMO: Recopila datos de status.csv y trades.csv para crear un resumen estético.
    """
    
    @staticmethod
    def generate_daily_summary(data_dir):
        """Genera un reporte resumen de la sesión actual"""
        try:
            status_path = os.path.join(data_dir, "status.csv")
            trades_path = os.path.join(data_dir, "trades.csv")
            
            if not os.path.exists(status_path):
                return None
                
            history = pd.read_csv(status_path)
            trades = pd.read_csv(trades_path) if os.path.exists(trades_path) else pd.DataFrame()
            
            # Calcular métricas
            metrics = AnalyticsEngine.calculate_metrics(history)
            win_stats = AnalyticsEngine.calculate_winrate_details(trades)
            
            # Formatear reporte (Texto por ahora, expandible a PDF si fpdf está disponible)
            report_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            
            report = f"""
============================================================
           TRADER GEMINI - DAILY PERFORMANCE REPORT
============================================================
Fecha: {report_time} UTC
Modo: {data_dir.split('/')[-1].upper()}
------------------------------------------------------------
RESULTADOS FINANCIEROS:
- Equity Final:  ${history['total_equity'].iloc[-1]:,.2f}
- PnL Sesión:   ${history['realized_pnl'].iloc[-1]:,.2f}
- Max Drawdown:  {metrics['max_drawdown']}%
- Sharpe Ratio:  {metrics['sharpe']}
- Sortino:       {metrics['sortino']}

ESTADÍSTICAS DE TRADING:
- Total Trades:  {win_stats['total_trades']}
- Win Rate:      {win_stats['global_winrate']}%
- Profit Factor: {win_stats['profit_factor']}
------------------------------------------------------------
"""
            # Guardar reporte en archivo de texto
            report_name = f"report_{datetime.now(timezone.utc).strftime('%Y%m%d')}.txt"
            report_file = os.path.join(data_dir, report_name)
            with open(report_file, 'w') as f:
                f.write(report)
                
            logger.info(f"📊 Report generated: {report_file}")
            return report
            
        except Exception as e:
            logger.error(f"❌ Error generating report: {e}")
            return None

    @staticmethod
    def export_to_excel(data_dir):
        """Exporta el historial completo a Excel con múltiples hojas"""
        try:
            status_path = os.path.join(data_dir, "status.csv")
            trades_path = os.path.join(data_dir, "trades.csv")
            
            output_path = os.path.join(data_dir, "full_performance_export.xlsx")
            
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                if os.path.exists(status_path):
                    pd.read_csv(status_path).to_excel(writer, sheet_name='EquityHistory', index=False)
                if os.path.exists(trades_path):
                    pd.read_csv(trades_path).to_excel(writer, sheet_name='TradesHistory', index=False)
            
            logger.info(f"📁 Data exported to Excel: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"❌ Error exporting to Excel: {e}")
            return None
