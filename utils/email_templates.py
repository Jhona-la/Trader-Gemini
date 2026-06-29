"""
📧 EMAIL TEMPLATES — Plantillas HTML para Notificaciones por Email
==================================================================

PROFESSOR METHOD:
- QUÉ: Plantillas HTML estilizadas para notificaciones de email.
- POR QUÉ: Emails en texto plano son difíciles de leer y no destacan datos críticos.
- PARA QUÉ: Presentación visual profesional con colores semánticos (verde=profit, rojo=loss).
- CÓMO: Funciones render_*() que devuelven HTML con CSS inline (compatible con clientes email).
- CUÁNDO: Cuando email SMTP está habilitado y se envían notificaciones.
- DÓNDE: Llamado desde Notifier.send_trade_open(), send_trade_close(), send_daily_report().
- QUIÉN: Módulo utils/email_templates.py, importado por Notifier.
"""

from typing import Dict, Any, List


# ═══════════════════════════════════════════════════════════════════════════
# BASE STYLES
# ═══════════════════════════════════════════════════════════════════════════
_BASE_STYLE = """
<style>
    body { font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; background: #f4f4f4; margin: 0; padding: 20px; }
    .container { max-width: 600px; margin: 0 auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); overflow: hidden; }
    .header { padding: 20px; color: white; text-align: center; }
    .header-profit { background: linear-gradient(135deg, #2ecc71, #27ae60); }
    .header-loss { background: linear-gradient(135deg, #e74c3c, #c0392b); }
    .header-neutral { background: linear-gradient(135deg, #3498db, #2980b9); }
    .header-warning { background: linear-gradient(135deg, #f39c12, #e67e22); }
    .header-critical { background: linear-gradient(135deg, #e74c3c, #c0392b); }
    .header h2 { margin: 0; font-size: 22px; }
    .header p { margin: 5px 0 0; opacity: 0.9; font-size: 14px; }
    .content { padding: 20px; }
    .section { margin-bottom: 20px; }
    .section-title { font-size: 16px; font-weight: bold; color: #2c3e50; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; margin-bottom: 10px; }
    .metric { display: flex; justify-content: space-between; padding: 4px 0; }
    .metric-label { color: #7f8c8d; font-size: 14px; }
    .metric-value { font-weight: bold; font-size: 14px; }
    .positive { color: #27ae60; }
    .negative { color: #e74c3c; }
    .neutral { color: #2980b9; }
    .highlight-box { background: #f8f9fa; border-left: 4px solid #3498db; padding: 12px; margin: 10px 0; border-radius: 0 4px 4px 0; }
    .highlight-profit { border-left-color: #27ae60; background: #f0faf0; }
    .highlight-loss { border-left-color: #e74c3c; background: #fdf0f0; }
    .highlight-warning { border-left-color: #f39c12; background: #fef9ed; }
    table { width: 100%; border-collapse: collapse; margin: 10px 0; }
    th { background: #ecf0f1; padding: 8px; text-align: left; font-size: 13px; color: #2c3e50; }
    td { padding: 8px; border-bottom: 1px solid #ecf0f1; font-size: 13px; }
    .footer { padding: 15px 20px; background: #f8f9fa; text-align: center; font-size: 12px; color: #95a5a6; }
</style>
"""


def _wrap_html(title: str, header_class: str, subtitle: str, body_html: str) -> str:
    """Wraps content in base HTML template."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    {_BASE_STYLE}
</head>
<body>
    <div class="container">
        <div class="header {header_class}">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
        <div class="content">
            {body_html}
        </div>
        <div class="footer">
            Trader Gemini — Sistema de Trading Automatizado<br>
            Este es un email automático. No responder.
        </div>
    </div>
</body>
</html>"""


def _metric_row(label: str, value: str, css_class: str = "") -> str:
    """Creates a metric row HTML."""
    val_class = f' class="{css_class}"' if css_class else ''
    return f'<div class="metric"><span class="metric-label">{label}</span><span class="metric-value"{val_class}>{value}</span></div>'


# ═══════════════════════════════════════════════════════════════════════════
# TRADE OPEN EMAIL
# ═══════════════════════════════════════════════════════════════════════════
def render_trade_open_email(data: Dict[str, Any]) -> str:
    """Renders HTML email for trade open notification."""
    direction = str(data['direction']).upper()
    symbol = data['symbol']
    strategy = data['strategy']
    horizon = data['horizon']

    header_class = "header-neutral"
    dir_str = "🟢 LONG" if direction in ("BUY", "LONG") else "🔴 SHORT"

    body = '<div class="section">'
    body += '<div class="section-title">📋 Detalles del Trade</div>'
    body += _metric_row("Estrategia", f"{strategy} ({horizon})")
    body += _metric_row("Par", symbol)
    body += _metric_row("Dirección", dir_str)
    body += _metric_row("Entrada", f"${data['fill_price']:,.4f}")
    body += _metric_row("Tamaño", f"{data['quantity']} (${data['size_usd']:,.2f} USD)")

    sl_pct = data['sl_pct']
    tp_pct = data['tp_pct']
    if sl_pct:
        body += _metric_row("Stop Loss", f"{sl_pct*100:.2f}%", "negative")
    if tp_pct:
        body += _metric_row("Take Profit", f"{tp_pct*100:.2f}%", "positive")

    rr = data['rr_ratio']
    if rr:
        body += _metric_row("Risk/Reward", f"1:{rr:.2f}")
    body += '</div>'

    # Viability
    body += '<div class="section">'
    body += '<div class="section-title">📊 Análisis de Viabilidad</div>'
    body += f'<div class="highlight-box">'
    body += _metric_row("Fees estimados", f"${data['commission']:,.4f}")
    body += _metric_row("Breakeven", f"{data['breakeven_pct']:,.3f}%")
    body += _metric_row("Neto mínimo viable", f"${data['min_viable_net']:,.4f}")
    body += '</div></div>'

    return _wrap_html(
        "🎯 Nuevo Trade Iniciado",
        header_class,
        f"{symbol} — {dir_str}",
        body
    )


# ═══════════════════════════════════════════════════════════════════════════
# TRADE CLOSE EMAIL
# ═══════════════════════════════════════════════════════════════════════════
def render_trade_close_email(data: Dict[str, Any]) -> str:
    """Renders HTML email for trade close notification."""
    net_pnl = data['net_pnl']
    symbol = data['symbol']
    is_profit = net_pnl > 0

    header_class = "header-profit" if is_profit else "header-loss"
    result_text = "GANANCIA" if is_profit else "PÉRDIDA"
    pnl_class = "positive" if is_profit else "negative"

    body = '<div class="section">'
    body += '<div class="section-title">📋 Resumen</div>'
    body += _metric_row("Estrategia", f"{data['strategy']} ({data['horizon']})")
    body += _metric_row("Par", symbol)
    body += _metric_row("Duración", str(data['duration']))
    body += _metric_row("Razón cierre", str(data['exit_reason']))
    body += '</div>'

    # Results
    body += '<div class="section">'
    body += '<div class="section-title">💰 Resultados</div>'
    hl_class = "highlight-profit" if is_profit else "highlight-loss"
    body += f'<div class="highlight-box {hl_class}">'
    body += _metric_row("PnL Bruto", f"${data['pnl']:,.4f}")
    body += _metric_row("Fees", f"-${data['commission']:,.4f}")
    sign = "+" if net_pnl >= 0 else ""
    body += _metric_row("PnL Neto", f"{sign}${net_pnl:,.4f} ({data['net_pnl_pct']:+.2f}%)", pnl_class)
    body += '</div></div>'

    # Prices
    body += '<div class="section">'
    body += '<div class="section-title">📈 Precios</div>'
    body += _metric_row("Entrada", f"${data['entry_price']:,.4f}")
    body += _metric_row("Salida", f"${data['exit_price']:,.4f}")
    body += '</div>'

    # Balance
    bal_before = data['balance_before']
    bal_after = data['balance_after']
    if bal_before > 0:
        body += '<div class="section">'
        body += '<div class="section-title">💵 Balance</div>'
        body += _metric_row("Antes", f"${bal_before:,.2f}")
        body += _metric_row("Después", f"${bal_after:,.2f}", pnl_class)
        body += _metric_row("Cambio", f"{data['balance_change_pct']:+.2f}%", pnl_class)
        body += '</div>'

    return _wrap_html(
        f"{'🟢' if is_profit else '🔴'} Trade Cerrado — {result_text}",
        header_class,
        f"{symbol} — PnL: {sign}${net_pnl:,.4f}",
        body
    )


# ═══════════════════════════════════════════════════════════════════════════
# DAILY REPORT EMAIL
# ═══════════════════════════════════════════════════════════════════════════
def render_daily_report_email(data: Dict[str, Any]) -> str:
    """Renders HTML email for daily report."""
    daily_pnl = data['daily_pnl']
    is_positive = daily_pnl >= 0
    header_class = "header-profit" if is_positive else "header-loss"
    pnl_class = "positive" if is_positive else "negative"

    body = '<div class="section">'
    body += '<div class="section-title">📊 Resumen del Día</div>'
    body += _metric_row("Fecha", str(data['date']))
    body += _metric_row("Balance Inicial", f"${data['start_balance']:,.2f}")
    body += _metric_row("Balance Final", f"${data['end_balance']:,.2f}")
    sign = "+" if daily_pnl >= 0 else ""
    body += _metric_row("PnL Diario", f"{sign}${daily_pnl:,.2f} ({data['daily_pnl_pct']:+.2f}%)", pnl_class)
    body += '</div>'

    # Trading metrics
    body += '<div class="section">'
    body += '<div class="section-title">📈 Métricas de Trading</div>'
    body += _metric_row("Total Trades", str(data['total_trades']))
    body += _metric_row("Ganadores", f"{data['winning_trades']} ({data['win_rate']:.1f}%)")
    body += _metric_row("Perdedores", str(data['losing_trades']))
    body += '</div>'

    # Strategy table
    strategies = data['strategies']
    if strategies:
        body += '<div class="section">'
        body += '<div class="section-title">🧠 Análisis por Estrategia</div>'
        body += '<table>'
        body += '<tr><th>Estrategia</th><th>Trades</th><th>Win Rate</th><th>PnL</th></tr>'
        for s in strategies:
            s_pnl = s['pnl']
            s_class = 'positive' if s_pnl >= 0 else 'negative'
            s_sign = "+" if s_pnl >= 0 else ""
            body += f'<tr><td>{s["name"]}</td><td>{s["trades"]}</td>'
            body += f'<td>{s["win_rate"]:.1f}%</td>'
            body += f'<td class="{s_class}">{s_sign}${s_pnl:,.2f}</td></tr>'
        body += '</table></div>'

    # Risk
    body += '<div class="section">'
    body += '<div class="section-title">🔒 Gestión de Riesgo</div>'
    body += _metric_row("Max Drawdown", f"{data['max_drawdown']:.2f}%")
    sharpe = data['sharpe_ratio']
    if sharpe:
        body += _metric_row("Sharpe Ratio", f"{sharpe:.2f}")
    sortino = data['sortino_ratio']
    if sortino:
        body += _metric_row("Sortino Ratio", f"{sortino:.2f}")
    body += '</div>'

    return _wrap_html(
        f"{'📈' if is_positive else '📉'} Reporte Diario",
        header_class,
        f"PnL: {sign}${daily_pnl:,.2f}",
        body
    )
