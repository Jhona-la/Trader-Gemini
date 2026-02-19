
import os
import glob
try:
    import ujson as json
except ImportError:
    import json
import time
import re
from datetime import datetime
import sys
from core.world_awareness import world_awareness

# Simple color shim to avoid dependencies
def colored(text, color, on_color=None, attrs=None):
    """Simple ANSI color wrapper"""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'dark_grey': '\033[90m',
        'reset': '\033[0m'
    }
    
    # Check if we should disable colors (dumb terminal)
    if not sys.stdout.isatty():
        return text
        
    code = colors.get(color, '')
    return f"{code}{text}{colors['reset']}"

def visible_len(text):
    """Calculate length of string excluding ANSI escape codes"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return len(ansi_escape.sub('', str(text)))

def pad_v(text, width, align='left'):
    """Pad string based on visible characters only"""
    v_len = visible_len(text)
    padding = " " * max(0, width - v_len)
    if align == 'right':
        return f"{padding}{text}"
    return f"{text}{padding}"

# Configuration
LOG_DIR = 'logs'

def get_latest_log_file():
    """Find the most recent bot log file."""
    # Pattern matches standard and futures logs
    patterns = [
        os.path.join(LOG_DIR, "bot_futures_*.json"),
        os.path.join(LOG_DIR, "bot_*.json")
    ]
    
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(p))
    
    if not candidates:
        return None
        
    # Sort by modification time
    return max(candidates, key=os.path.getmtime)

def get_live_status():
    """Read structured status data from live_status.json (More efficient than parsing logs)."""
    # Determine path based on mode (Futures/Spot)
    # For now, we assume Futures as per user context
    status_path = os.path.join("dashboard", "data", "futures", "live_status.json")
    if not os.path.exists(status_path):
        return {}
        
    try:
        with open(status_path, 'r') as f:
            return json.load(f)
    except:
        return {}

def parse_oracle_logs(log_file, last_pos=0):
    """
    Parse new lines for Oracle insights.
    Optimized to read only NEW lines (tail logic).
    Returns: (insights, new_file_pos)
    """
    insights = {}
    
    if not log_file or not os.path.exists(log_file):
        return {}, last_pos
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            # Efficiency: Seek to last known position
            f.seek(last_pos)
            lines = f.readlines()
            new_pos = f.tell()
            
            for line in lines:
                try:
                    line = line.strip()
                    if not line: continue
                    
                    try:
                        entry = json.loads(line)
                        msg = entry.get('message', '')
                    except json.JSONDecodeError:
                        msg = line
                        
                    # NOTE: We now get Portfolio Status from live_status.json, 
                    # so we focus mainly on ML/Oracle text insights here.

                    # 2. Check for DEBUG ML info
                    if 'DEBUG ML' in msg:
                        # Match: DEBUG ML [BTC/USDT]: Rows=...
                        debug_match = re.search(r'DEBUG ML\s+\[([A-Z0-9/]+)\]:\s*(.*)', msg)
                        if debug_match:
                            symbol = debug_match.group(1)
                            debug_info = debug_match.group(2)
                            
                            if symbol not in insights:
                                insights[symbol] = {'timestamp': entry.get('timestamp', '')}
                            
                            insights[symbol]['debug_ml'] = debug_info
                        continue

                    # 3. Check for Oracle Insights
                    if '[ML ORACLE]' in msg or '[UNIFIED ORACLE]' in msg:
                        sym_match = re.search(r'(?:ML|UNIFIED) ORACLE\]\s+([A-Z0-9/]+)', msg)
                        if not sym_match: continue
                        
                        symbol = sym_match.group(1)
                        if symbol not in insights:
                            insights[symbol] = {
                                'timestamp': entry.get('timestamp', datetime.now().strftime('%H:%M:%S')),
                                'type': 'UNIFIED' if 'UNIFIED' in msg else 'ML'
                            }
                        
                        # Basic fields
                        insights[symbol]['timestamp'] = entry.get('timestamp', insights[symbol]['timestamp'])
                        
                        # Scores
                        scores_match = re.search(r'Scores\s+->\s*ML:\s*([\d\.]+)\s*\|\s*SENT:\s*([\d\.]+)\s*\|\s*TECH:\s*([\d\.]+)', msg)
                        if scores_match:
                            insights[symbol].update({
                                'ml_score': float(scores_match.group(1)),
                                'sent_score': float(scores_match.group(2)),
                                'tech_score': float(scores_match.group(3))
                            })
                        
                        # Horizons
                        horiz_match = re.search(r'Horizon\s+->\s*H5:\s*([\d\.]+)\s*\|\s*H15:\s*([\d\.]+)\s*\|\s*H30:\s*([\d\.]+)', msg)
                        if horiz_match:
                            insights[symbol].update({
                                'h5': float(horiz_match.group(1)),
                                'h15': float(horiz_match.group(2)),
                                'h30': float(horiz_match.group(3))
                            })

                        # Verdict
                        dir_match = re.search(r'Verdict\s+->\s+Direction:\s*([A-Z1\-]+)\s*\|\s*Final Conf:\s*([\d\.]+)', msg)
                        if dir_match:
                            raw_dir = dir_match.group(1)
                            insights[symbol]['direction'] = 'LONG' if raw_dir == '1' else 'SHORT' if raw_dir == '-1' else raw_dir
                            insights[symbol]['confidence'] = float(dir_match.group(2))

                        # Phase
                        phase_match = re.search(r'Phase:\s*(.+)', msg)
                        if phase_match: insights[symbol]['phase'] = phase_match.group(1).strip()

                        # Concept
                        concept_match = re.search(r'Concept:\s*(.+)', msg)
                        if concept_match: insights[symbol]['concept'] = concept_match.group(1).strip()

                        # Stats
                        stats_match = re.search(r'Stats:\s*(.+)', msg)
                        if stats_match: insights[symbol]['stats'] = stats_match.group(1).strip()
                        
                        # Strategy
                        strat_match = re.search(r'Strategy:\s*(.+)', msg)
                        if strat_match: insights[symbol]['strategy'] = strat_match.group(1).strip()

                    # 4. Check for Separate CV Scores/Regime metrics
                    if 'CV Scores -' in msg or 'Regime:' in msg:
                        sym_match = re.search(r'📊\s+\[([A-Z0-9/]+)\]', msg)
                        if sym_match:
                            symbol = sym_match.group(1)
                            if symbol not in insights:
                                insights[symbol] = {'timestamp': entry.get('timestamp', '')}
                            
                            # CV Scores
                            cv_match = re.search(r'Ensemble:\s*([\d\.]+)', msg)
                            if cv_match: insights[symbol]['cv'] = float(cv_match.group(1))
                            
                            # Regime
                            reg_match = re.search(r'Regime:\s*([A-Za-z]+)', msg)
                            if reg_match: insights[symbol]['market_regime'] = reg_match.group(1)
                    
                    # 5. Check for Quality Guard alerts
                    if '[Quality Guard]' in msg:
                        sym_match = re.search(r'for\s+([A-Z0-9/]+)', msg)
                        if sym_match:
                            symbol = sym_match.group(1)
                            if symbol not in insights:
                                insights[symbol] = {'timestamp': entry.get('timestamp', '')}
                            
                            if 'Rejected' in msg:
                                insights[symbol]['qual_guard'] = 'REJECTED'
                            elif 'PASSED' in msg:
                                insights[symbol]['qual_guard'] = 'PASSED'

                except Exception:
                    continue
            
            return insights, new_pos
                    
    except Exception as e:
        print(colored(f"Error reading log file: {e}", "red"))
        return {}, last_pos


def display_dashboard(insights, status_data):
    """Print a expanded dashboard of insights."""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Context
    context = world_awareness.get_market_context()
    ls = context['liquidity_score']
    sessions = (", ".join(context['active_sessions']) if context['active_sessions'] else "NONE")
    status_color = "green" if context['is_prime'] else "yellow" if ls > 0.45 else "red"
    
    # Portfolio Data from JSON
    equity = status_data.get('total_equity', 0.0)
    balance = status_data.get('wallet_balance', 0.0)
    pnl = status_data.get('unrealized_pnl', 0.0)
    pnl_color = "green" if pnl >= 0 else "red"
    
    print("\n" + colored("="*145, "cyan"))
    print(f" {colored('🌍 TRADER GEMINI - PROFESSOR ORACLE', 'cyan', attrs=['bold'])} | Sessions: {colored(sessions, 'white')} | LS: {colored(f'{ls:.2f}', status_color)} | "
          f"Equity: ${colored(f'{equity:.2f}', 'white')} | Unr. PnL: {colored(f'${pnl:.2f}', pnl_color)}")
    print(colored("="*145, "cyan"))
    
    # --- SOVEREIGN BRAIN RANKINGS (Phase 7) ---
    rankings = status_data.get('strategy_rankings', {})
    if rankings:
        print(f" {colored('🧠 SOVEREIGN BRAIN RANKINGS:', 'cyan')} ", end="")
        for strat, info in rankings.items():
            rank = info.get('rank', '-')
            score = info.get('score', 0)
            color = "green" if score > 0.6 else "yellow" if score > 0.4 else "red"
            icon = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else "💀"))
            print(f"[{icon} {colored(strat, color)}: {score:.1%}]  ", end="")
        print("\n" + colored("-" * 145, "dark_grey"))

    if not insights:
        print(colored("\n   Waiting for data... (No recent insights found)", "yellow"))
        return

    # Sort symbols alphabetically
    sorted_symbols = sorted(insights.keys())
    
    # Header
    print(f" {colored('SYMBOL', 'white', attrs=['bold']):<10} | {colored('TYPE', 'white'):<8} | {colored('DIRECTION', 'white'):<9} | {colored('CONF', 'white'):<6} | {colored('CV', 'white'):<5} | {colored('REGIME', 'white'):<8} | {colored('ML', 'white'):<4} | {colored('SENT', 'white'):<4} | {colored('TECH', 'white'):<4} | {colored('HORIZON (5/15/30m)', 'white'):<20} | {colored('PHASE / CONCEPT', 'white')}")
    print(colored("-" * 145, "cyan"))

    for sym in sorted_symbols:
        data = insights[sym]
        
        # Determine Color based on Confidence & Direction
        directory = data.get('direction', 'N/A')
        conf = data.get('confidence', 0.0)
        
        row_color = "white"
        if directory == "LONG" and conf > 0.7: row_color = "green"
        elif directory == "SHORT" and conf > 0.7: row_color = "red"
        
        # Format metrics with vibrant colors
        ml_val = data.get('ml_score', 0.0)
        ml_color = "green" if ml_val > 0.6 else "yellow" if ml_val > 0.5 else "red"
        ml_score_c = colored(f"{ml_val:.2f}", ml_color)
        
        sent_val = data.get('sent_score', 0.0)
        sent_color = "cyan" if sent_val > 0.6 else "yellow" if sent_val > 0.5 else "red"
        sent_score_c = colored(f"{sent_val:.2f}", sent_color)
        
        tech_val = data.get('tech_score', 0.0)
        tech_color = "magenta" if tech_val > 0.6 else "yellow" if tech_val > 0.5 else "red"
        tech_score_c = colored(f"{tech_val:.2f}", tech_color)
        
        # CV Score / Training Health
        cv_val = data.get('cv', data.get('training_score', 0.0))
        cv_color = "green" if cv_val > 0.55 else "yellow" if cv_val > 0.45 else "red"
        cv_str = colored(f"{cv_val:.2f}", cv_color)
        
        # Qual Guard
        qg = data.get('qual_guard', 'OK')
        if qg == "REJECTED":
            cv_str = colored("REJ", "red")
        
        # Regime
        regime_raw = data.get('market_regime', data.get('phase', 'UNKNOWN')).replace(" (0.0%)", "").strip()
        reg_color = "cyan" if "TRENDING" in regime_raw else "magenta" if "VOLATILE" in regime_raw else "yellow" if "RANGING" in regime_raw else "dark_grey"
        regime_c = colored(pad_v(regime_raw, 8), reg_color)
        
        horizon = f"{data.get('h5', 0.0):.2f}/{data.get('h15', 0.0):.2f}/{data.get('h30', 0.0):.2f}"
        
        # Symbol with Active indicator
        is_active = data.get('active', False)
        prefix = colored("* ", "cyan") if is_active else "  "
        sym_c = colored(pad_v(sym, 8), row_color)
        sym_display = f"{prefix}{sym_c}"
        
        # Row format
        row = (
            f" {sym_display} | "
            f" {pad_v(data.get('type', 'ML'), 8)} | "
            f" {colored(pad_v(directory, 9), row_color)} | "
            f" {colored(f'{conf:.2f}', row_color):<6} | "
            f" {cv_str:<5} | "
            f" {regime_c} | "
            f" {ml_score_c:<4} | "
            f" {sent_score_c:<4} | "
            f" {tech_score_c:<4} | "
            f" {horizon:<20} | "
            f" {colored(data.get('phase', 'N/A'), 'white'):<8} / {colored(data.get('concept', 'Scanning...'), 'dark_grey'):<40}"
        )
        print(row)

    # --- LEYENDA DETALLADA (MODO PROFESOR) ---
    print("\n" + colored("="*145, "cyan"))
    print(f" {colored('👨‍🏫 GUÍA TÉCNICA TRADER GEMINI', 'cyan', attrs=['bold'])}")
    print(colored("-" * 145, "dark_grey"))
    
    legend = [
        ("CV", "Cross-Validation Score", "Validación interna del modelo (0-1). >0.55 es óptimo para operar.", "ml_strategy.py", "Valida salud del cerebro IA"),
        ("REGIME", "Estado del Mercado", " trending=Tendencia, ranging=Lateral, volatile=Caos.", "market_regime.py", "Adapta SL/TP y Riesgo"),
        ("ML", "Machine Learning Score", "Probabilidad basada en patrones históricos (RF+XGB+GB).", "ml_strategy.py", "Motor de predicción principal"),
        ("SENT", "Sentiment Score", "Ánimo del mercado (Noticias/Social). 0.5=Neutral.", "sentiment_loader.py", "Filtro de consciencia externa"),
        ("TECH", "Technical Score", "Indicadores clásicos (RSI, MACD, EMAs) en confluencia.", "strategies/technical.py", "Confirmación de estructura"),
        ("HORIZON", "Éxito Multi-temporal", "Probabilidad de acierto en 5m, 15m y 30m.", "ml_strategy.py", "Asegura tendencia sostenida"),
        ("CONF", "Ensemble Confidence", "Promedio ponderado de los 3 motores (Threshold: 0.75).", "strategies/ml_strategy.py", "Gatillo final de ejecución")
    ]
    
    for key, name, desc, loc, goal in legend:
        print(f" • {colored(pad_v(key, 8), 'cyan')} | {colored(pad_v(name, 22), 'white')} | {desc:<60} | {colored(f'Destino: {goal}', 'green'):<30} | {colored(f'Origen: {loc}', 'magenta')}")

    print(colored("-" * 145, "dark_grey"))
    print(f" {colored('*', 'cyan')} = Posición Activa en Portafolio | Actualizado: {datetime.now().strftime('%H:%M:%S')}")


def main():
    print("🔍 Searching for active logs...")
    log_file = get_latest_log_file()
    
    if not log_file:
        print(colored("❌ No log files found in logs/ directory!", "red"))
    else:
        print(f"📂 Reading: {log_file}")
    
    print("Hit CTRL+C to stop.")
    time.sleep(1)
    
    # State tracking
    last_pos = 0
    all_insights = {} # Accumulate insights over time (since we read delta)
    
    try:
        while True:
            # 1. Read JSON Status (Fast, Reliable)
            status_data = get_live_status()
            active_from_json = status_data.get('positions', {}).keys()
            
            # 2. Read Logs (Delta only)
            if log_file and os.path.exists(log_file):
                new_insights, last_pos = parse_oracle_logs(log_file, last_pos)
                if new_insights:
                    all_insights.update(new_insights)
            else:
                # Retry finding log file if missing
                log_file = get_latest_log_file()
            
            # 3. Update active status on insights
            for sym in all_insights:
                all_insights[sym]['active'] = (sym in active_from_json)

            display_dashboard(all_insights, status_data)
            time.sleep(2) # Refresh every 2 seconds
            
    except KeyboardInterrupt:
        print("\n👋 Exiting Oracle View.")

if __name__ == "__main__":
    main()
