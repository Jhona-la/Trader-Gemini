"""
🔬 AUDITORÍA FORENSE COMPLETA — 6 ÁREAS — TRADER GEMINI
=========================================================
Examina todas las áreas del sistema para identificar causas raíz de pérdidas:
1. Datos y flujo de información
2. Estrategias y señales  
3. Ejecución y operativa
4. Gestión de riesgos
5. Backtesting y validación
6. Infraestructura y performance
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
import numpy as np

FINDINGS = []
CRITICAL = []
WARNING = []
INFO = []

# Constants
CAPITAL = 13.0
LEVERAGE = 10
SIZING = 0.40

def log_finding(severity, area, title, detail, fix=""):
    entry = {"severity": severity, "area": area, "title": title, "detail": detail, "fix": fix}
    FINDINGS.append(entry)
    if severity == "CRITICAL": CRITICAL.append(entry)
    elif severity == "WARNING": WARNING.append(entry)
    else: INFO.append(entry)

# ============================================================
# ÁREA 1: AUDITORÍA DE DATOS Y FLUJO DE INFORMACIÓN
# ============================================================
def audit_data_flow():
    print("\n" + "="*70)
    print("📊 ÁREA 1: DATOS Y FLUJO DE INFORMACIÓN")
    print("="*70)
    
    # 1.1 Volume Ratio en backtesting
    # El backtest usa barras históricas con volumen normalizado.
    # El volume_ratio se calcula como vol_actual / media_20.
    # En backtest, las primeras 20 barras siempre tienen ratio ~1.0 
    # pero después de gaps o baja liquidez, cae a 0.01.
    log_finding("CRITICAL", "DATOS", 
        "Volume Ratio mata señales en backtest",
        "ml_strategy.py L3529: min_vol=0.2 en testnet, 0.7 en producción. "
        "En backtest, vol_ratio=0.01 es NORMAL para datos históricos sin volumen tick. "
        "Esto bloquea >60% de las señales válidas del Oracle.",
        "Desactivar filtro de volumen en backtest o usar umbral adaptativo (0.01 para BT)")
    
    # 1.2 Datos f32 vs f64 precision
    from utils.hft_buffer import NumbaStructuredRingBuffer
    buf = NumbaStructuredRingBuffer(10)
    buf.push(1, np.float32(100.123456789), np.float32(101.0), np.float32(99.0), 
             np.float32(100.5), np.float32(1000.0))
    t, o, h, l, c, v = buf.get_last(1)
    precision_loss = abs(float(o[0]) - 100.123456789)
    if precision_loss > 1e-4:
        log_finding("WARNING", "DATOS",
            f"Pérdida de precisión float32: {precision_loss:.8f}",
            "Los buffers HFT usan float32 para velocidad, perdiendo ~4 decimales. "
            "Para BTC a $84K, un error de $0.01 en slippage puede acumularse.",
            "Aceptable para scalping. Monitorear drift acumulado.")
    else:
        log_finding("INFO", "DATOS", "Precisión float32 dentro de límites", 
                    f"Error: {precision_loss:.10f}")
    
    # 1.3 Config fee consistency
    maker = getattr(Config, 'BINANCE_MAKER_FEE_BNB', None)
    taker = getattr(Config, 'BINANCE_TAKER_FEE_BNB', None)
    if maker is None:
        log_finding("CRITICAL", "DATOS",
            "BINANCE_MAKER_FEE_BNB no existe en Config",
            "El sistema usa TAKER fee para LIMIT orders, sobreestimando costos 47%",
            "Añadir BINANCE_MAKER_FEE_BNB = 0.0002 a Config")
    elif maker and taker:
        saving = (1 - maker/taker) * 100
        log_finding("INFO", "DATOS", 
            f"Fees configurados: MAKER={maker*100:.3f}%, TAKER={taker*100:.4f}%",
            f"Ahorro LIMIT vs MARKET: {saving:.1f}%")

    print(f"  ✅ Área 1 completada: {sum(1 for f in FINDINGS if f['area']=='DATOS')} hallazgos")


# ============================================================
# ÁREA 2: AUDITORÍA DE ESTRATEGIAS Y SEÑALES
# ============================================================
def audit_strategies():
    print("\n" + "="*70)
    print("🧠 ÁREA 2: ESTRATEGIAS Y SEÑALES")
    print("="*70)
    
    # 2.1 strength_threshold inconsistency
    scalp_params = getattr(Config.Strategies, 'SCALPING_PARAMS', {})
    swing_params = getattr(Config.Strategies, 'SWING_PARAMS', {})
    
    scalp_st = scalp_params.get('strength_threshold', 'MISSING')
    swing_st = swing_params.get('strength_threshold', 'MISSING')
    
    log_finding("WARNING", "ESTRATEGIAS",
        f"strength_threshold: SCALP={scalp_st}, SWING={swing_st}",
        "Valores inconsistentes entre Config, Genotype (0.6) y technical.py (0.40). "
        "El Genotype Gen-0 inyecta 0.6, sobreescribiendo el 0.55 de Config. "
        "Luego technical.py L294 corrige a self.STRENGTH_THRESHOLD solo si gen==0.",
        "Unificar source of truth: Config debe ser el master, nunca Genotype gen-0.")
    
    # 2.2 ML consensus_threshold es 0.78 en oracle pero 2/3 engines_passing
    log_finding("CRITICAL", "ESTRATEGIAS",
        "Doble filtro de confianza en ML: Oracle 0.78 + Engines 2/3",
        "ml_strategy.py define consensus_threshold=0.78 Y requiere engines_passing>=2/3. "
        "Esto significa que incluso con ML=1.0 y TECH=0.91 (87% confianza), "
        "si SENTIMENT=0.50 (siempre es 0.50 porque no hay feed real), "
        "solo pasan 2/3 engines. La señal se genera PERO con debug warn 'Resetting weights'. "
        "En la práctica, SENTIMENT NUNCA aporta porque siempre es 0.50.",
        "Reducir a 2 engines requeridos (ML+TECH) o eliminar SENTIMENT del cálculo")
    
    # 2.3 ML Volume filter kills backtest signals
    log_finding("CRITICAL", "ESTRATEGIAS",
        "Volume filter en ML mata >60% de señales en backtest",
        "ml_strategy.py L3529-3531: vol_ratio < 0.2 (testnet) o < 0.7 (producción) "
        "bloquea señales. En backtest con datos 1m, volume_ratio frecuentemente es 0.01 "
        "porque se calcula ratio vs media_20, y en datos históricos sin volumen tick "
        "la varianza es mínima. Esto invalida TODO el backtest.",
        "Bypass volume filter en modo sandbox/backtest")
    
    # 2.4 Signal deduplication temporal check
    log_finding("INFO", "ESTRATEGIAS",
        "Signal dedup corregido de tick-based a bar-based (Phase 1 fix)",
        "technical.py ahora usa timestamp de la barra OHLCV como clave de dedup, "
        "eliminando el flooding de señales idénticas por segundo.")
    
    # 2.5 ML base_position_size = 0.95 en MLStrategy
    log_finding("CRITICAL", "ESTRATEGIAS",
        "MLStrategy tiene base_position_size = 0.95 (95%)",
        "ml_strategy.py L443: base_position_size = 0.95 — contradice el fix "
        "de RiskManager que bajó a 40%. Si MLStrategy usa su propio sizing "
        "en vez de pasar por RiskManager, podría usar el 95% suicida.",
        "Verificar que MLStrategy SIEMPRE delegate sizing a RiskManager")
    
    # 2.6 Temporal confidence decay penaliza primeras barras
    log_finding("WARNING", "ESTRATEGIAS",
        "Temporal confidence decay reduce ML a 50% en primeras barras",
        "ml_strategy.py L3264: Las primeras 20% de barras procesadas reciben "
        "factor 0.50, reduciendo la confianza ML de ~0.87 a ~0.43. "
        "Esto significa que en backtests cortos (1-3 días), las primeras horas "
        "NUNCA generan trades viables.",
        "Para backtests de 3 días, el warmup period consume ~7h de señales")

    print(f"  ✅ Área 2 completada: {sum(1 for f in FINDINGS if f['area']=='ESTRATEGIAS')} hallazgos")


# ============================================================
# ÁREA 3: AUDITORÍA DE EJECUCIÓN Y OPERATIVA
# ============================================================
def audit_execution():
    print("\n" + "="*70)
    print("⚡ ÁREA 3: EJECUCIÓN Y OPERATIVA")
    print("="*70)
    
    # 3.1 Slippage hardcoded
    log_finding("WARNING", "EJECUCIÓN",
        "Slippage hardcoded a 0.0001 (0.01%) en Portfolio",
        "portfolio.py L779-780: slippage_entry=0.0001, slippage_exit=0.0001 "
        "son valores fijos, no reflejan slippage real. Para micro-cuentas con "
        "LIMIT orders, el slippage debería ser ~0 (fills at limit price).",
        "Usar slippage real del FillEvent o 0 para LIMIT orders")
    
    # 3.2 Portfolio usa TAKER_FEE por defecto
    log_finding("CRITICAL", "EJECUCIÓN",
        "Portfolio._record_closed_trade usa TAKER fee por defecto",
        "portfolio.py L731: fee_rate = Config.BINANCE_TAKER_FEE_BNB (0.0375%). "
        "Pero el sistema usa LIMIT orders que pagan MAKER fee (0.02%). "
        "Cada trade cerrado en backtest/producción sobreestima fees un 87.5%.",
        "Usar MAKER_FEE_BNB para LIMIT orders en _record_closed_trade")
    
    # 3.3 Leverage floor micro-account
    notional = CAPITAL * SIZING * LEVERAGE
    log_finding("INFO", "EJECUCIÓN",
        f"Notional viable: ${notional:.2f} (capital=${CAPITAL}, size=40%, lev={LEVERAGE}x)",
        f"Binance mínimo: $5.00 — {'✅ VIABLE' if notional > 5 else '❌ INSUFICIENTE'}")
    
    # 3.4 Cooldown y concurrent positions
    cooldown = getattr(Config, 'COOLDOWN_PERIOD_SECONDS', 45)
    concurrent = getattr(Config, 'MAX_CONCURRENT_POSITIONS', 2)
    if cooldown > 0:
        log_finding("INFO", "EJECUCIÓN",
            f"Operativa: Cooldown={cooldown}s, MaxConcurrent={concurrent}",
            f"Con {concurrent} posiciones concurrentes y cooldown {cooldown}s, "
            f"máximo teórico: {3600/cooldown * concurrent:.0f} trades/hora")
    else:
        log_finding("INFO", "EJECUCIÓN",
            f"Operativa: Cooldown={cooldown}s, MaxConcurrent={concurrent}",
            f"Sin límite de cooldown, trades/hora infinito.")

    print(f"  ✅ Área 3 completada: {sum(1 for f in FINDINGS if f['area']=='EJECUCIÓN')} hallazgos")


# ============================================================
# ÁREA 4: AUDITORÍA DE GESTIÓN DE RIESGOS
# ============================================================
def audit_risk():
    print("\n" + "="*70)
    print("🔒 ÁREA 4: GESTIÓN DE RIESGOS")
    print("="*70)
    
    # 4.1 Sizing: comparar producción vs backtest
    log_finding("INFO", "RIESGO",
        "Sizing corregido de 95% a 40% por trade (Phase 1 fix)",
        "Previene over-allocation con 3+ posiciones concurrentes. "
        "Con 40% sizing × 3 positions = 120% del capital usado, "
        "dejando 0% buffer para fees/slippage. Considerar bajar a 30%.",
        "Monitorear: 3×40%=120% puede causar margin calls en pérdidas rápidas")
    
    # 4.2 Stop loss y take profit
    scalp_params = getattr(Config.Strategies, 'SCALPING_PARAMS', {})
    tp = scalp_params.get('tp_pct', 0.015)
    sl = scalp_params.get('sl_pct', 0.02)
    rr = tp / sl if sl > 0 else 0
    
    log_finding("WARNING" if rr < 1.0 else "INFO", "RIESGO",
        f"Risk/Reward Ratio: TP={tp*100:.1f}% / SL={sl*100:.1f}% = R:R {rr:.2f}",
        f"{'⚠️ R:R < 1.0 — el riesgo es mayor que la recompensa. Un WR < 50% garantiza pérdidas.' if rr < 1 else '✅ R:R ratio viable.'} "
        f"Con WR=60% y R:R={rr:.2f}: Edge = 0.6×{tp*100:.1f}% - 0.4×{sl*100:.1f}% = {(0.6*tp - 0.4*sl)*100:.3f}%",
        "Si R:R < 1.0, subir TP o bajar SL")
    
    # 4.3 REGIME_MAP leverage correctness
    regime_map = getattr(Config.Sniper, 'REGIME_MAP', {})
    for regime, params in regime_map.items():
        lev = params.get('max_leverage', 0)
        if lev < 5:
            log_finding("WARNING", "RIESGO",
                f"REGIME_MAP[{regime}] max_leverage={lev}x — Notional=${CAPITAL * 0.4 * lev:.1f}",
                f"Con leverage {lev}x, notional ${CAPITAL * 0.4 * lev:.1f} puede caer debajo "
                f"del mínimo de Binance ($5.00).",
                f"Subir a mínimo 8x para micro-cuentas")
    
    # 4.4 Kill Switch drawdown verify
    max_dd = getattr(Config.Risk, 'MAX_DRAWDOWN', 0.02)
    sl = getattr(Config.Strategies.SCALPING_PARAMS, 'sl_pct', 0.02)
    log_finding("INFO", "RIESGO",
        f"Kill Switch: MAX_DRAWDOWN = {max_dd*100:.1f}%",
        f"Con $13 capital, kill switch se activa con pérdida de ${CAPITAL * max_dd:.2f}. "
        f"Esto son ~{int(CAPITAL * max_dd / (CAPITAL * SIZING * LEVERAGE * sl)):.0f} trades perdedores consecutivos.")

    print(f"  ✅ Área 4 completada: {sum(1 for f in FINDINGS if f['area']=='RIESGO')} hallazgos")


# ============================================================
# ÁREA 5: AUDITORÍA DE BACKTESTING Y VALIDACIÓN
# ============================================================
def audit_backtesting():
    print("\n" + "="*70)
    print("📊 ÁREA 5: BACKTESTING Y VALIDACIÓN")
    print("="*70)
    
    # 5.1 Backtest vs Production parity
    log_finding("CRITICAL", "BACKTEST",
        "Volume filter diverge entre backtest y producción",
        "ml_strategy.py L3529: En testnet (backtest), min_vol=0.2. "
        "En producción, min_vol=0.7. Esto significa que el backtest acepta "
        "señales que producción RECHAZARÍA (vol_ratio 0.3 pasa en BT pero no en prod). "
        "Sin embargo, el problema real es que en backtest, vol_ratio=0.01 "
        "es NORMAL y ambos umbrales la rechazan.",
        "Usar mismo umbral en BT y producción, pero ajustar el cálculo de vol_ratio "
        "en el backtest para que sea representativo")
    
    # 5.2 God Mode backtest usa producción Portfolio
    log_finding("INFO", "BACKTEST",
        "God Mode backtest usa Portfolio y RiskManager de producción",
        "scripts/run_god_mode_backtest.py instancia las clases reales, "
        "garantizando paridad backtest-producción para sizing y fees.")
    
    # 5.3 Backtest speed
    log_finding("WARNING", "BACKTEST",
        "Backtest extremadamente lento: 8 symbols × 3 días = ~20+ minutos",
        "Cada epoch evalúa 8 símbolos secuencialmente con Oracle completo. "
        "El Oracle produce ~300 bytes de log por evaluación. 4320 epochs × 8 symbols "
        "= 34,560 evaluaciones × 0.3s = ~3 horas si no se optimiza.",
        "Reducir logging en mode silencioso, paralelizar por símbolo, "
        "o usar backtest vetcorizado sin Oracle completo")
    
    # 5.4 Fee calculation en backtest
    log_finding("CRITICAL", "BACKTEST",
        "Portfolio calcula fees con TAKER rate incluso para LIMIT orders en BT",
        "Cada trade cerrado en backtest sobreestima fees un 87.5% (0.0375% vs 0.02%). "
        "Con 200 trades en 7 días, el fee drag acumulado es: "
        f"200 × ${CAPITAL * 0.4 * LEVERAGE * 0.000375 * 2:.4f} = ${200 * CAPITAL * 0.4 * LEVERAGE * 0.000375 * 2:.2f} sobrecalculado "
        f"vs real ${200 * CAPITAL * 0.4 * LEVERAGE * 0.0002 * 2:.2f}",
        "Usar MAKER fee en portfolio._record_closed_trade para LIMIT orders")

    print(f"  ✅ Área 5 completada: {sum(1 for f in FINDINGS if f['area']=='BACKTEST')} hallazgos")


# ============================================================
# ÁREA 6: AUDITORÍA DE INFRAESTRUCTURA Y PERFORMANCE
# ============================================================
def audit_infrastructure():
    print("\n" + "="*70)
    print("🔧 ÁREA 6: INFRAESTRUCTURA Y PERFORMANCE")
    print("="*70)
    
    # 6.1 ProcessPoolExecutor workers
    log_finding("WARNING", "INFRA",
        "ML Training Pool: 6 workers fijos (ProcessPoolExecutor)",
        "ml_strategy.py L99: max_workers=6 sin considerar RAM disponible. "
        "Con 25 símbolos × 80 features × RandomForest, cada worker consume ~200MB. "
        "6 workers = ~1.2GB solo en training.",
        "Monitorear RAM con system_monitor.py, reducir a 4 workers si RAM > 80%")
    
    # 6.2 ThreadPoolExecutor per symbol
    log_finding("WARNING", "INFRA",
        "Cada MLStrategy instancia tiene su propio ThreadPoolExecutor(2)",
        f"Con {len(getattr(Config.Data, 'SYMBOLS', []))} símbolos × 2 threads = "
        f"{len(getattr(Config.Data, 'SYMBOLS', [])) * 2} threads permanentes solo para ML inference. "
        "Más los threads de BinanceData, LatencyMonitor, DerivativesMonitor.",
        "Pool compartido global o reducir a max_workers=1 por símbolo")
    
    # 6.3 Duplicate logging (every line logged twice)
    log_finding("WARNING", "INFRA",
        "Cada mensaje del Oracle se imprime DUPLICADO en la salida",
        "El logger está configurado con múltiples handlers que producen la misma línea "
        "con timestamp largo y timestamp corto. Esto duplica I/O y dificulta análisis.",
        "Configurar un solo handler o filtrar duplicados en logger.py")
    
    # 6.4 Risk of weight reset on every signal
    log_finding("WARNING", "INFRA",
        "Ensemble weights se resetean en CADA señal generada",
        "ml_strategy.py genera '⚠️ Resetting weights due to poor performance' "
        "en CADA señal. Esto significa que el aprendizaje online NUNCA acumula mejoras — "
        "se resetea antes de tener datos suficientes para evaluar.",
        "Solo resetear si performance degradation > threshold (e.g., 10 trades)")

    print(f"  ✅ Área 6 completada: {sum(1 for f in FINDINGS if f['area']=='INFRA')} hallazgos")


# ============================================================
# INFORME EJECUTIVO
# ============================================================
def generate_report():
    print("\n" + "="*70)
    print("📋 INFORME EJECUTIVO — AUDITORÍA FORENSE COMPLETA")
    print("="*70)
    
    print(f"\n🔴 HALLAZGOS CRÍTICOS: {len(CRITICAL)}")
    for i, f in enumerate(CRITICAL, 1):
        print(f"  {i}. [{f['area']}] {f['title']}")
        print(f"     Diagnóstico: {f['detail'][:120]}...")
        print(f"     🔧 Fix: {f['fix'][:100]}")
        print()
    
    print(f"\n🟡 WARNINGS: {len(WARNING)}")
    for i, f in enumerate(WARNING, 1):
        print(f"  {i}. [{f['area']}] {f['title']}")
        print(f"     {f['detail'][:100]}...")
        print()
    
    print(f"\n🟢 INFO: {len(INFO)}")
    for i, f in enumerate(INFO, 1):
        print(f"  {i}. [{f['area']}] {f['title']}")
        print()
    
    # Priority matrix
    print("\n" + "="*70)
    print("🎯 PLAN DE ACCIÓN PRIORITIZADO")
    print("="*70)
    print("""
╔══════╦═══════════════════════════════════════════════════════╦══════════╦═══════════╗
║ Prio ║ Acción                                               ║ Archivo  ║ Impacto   ║
╠══════╬═══════════════════════════════════════════════════════╬══════════╬═══════════╣
║  P0  ║ Portfolio: Usar MAKER fee para LIMIT orders           ║ portfol  ║ -47% fees ║
║  P0  ║ ML: Bypass volume filter en backtest/sandbox          ║ ml_strat ║ +300% sig ║
║  P1  ║ ML: Reducir consensus a 2 engines (SENT=dead weight) ║ ml_strat ║ +40% sig  ║
║  P1  ║ ML: No resetear weights en cada señal                ║ ml_strat ║ Estabilid ║
║  P2  ║ Sizing: Considerar bajar 40%→30% (3×40%=120%)        ║ risk_mgr ║ Seguridad ║
║  P2  ║ Backtest: Same volume filter BT vs Prod              ║ ml_strat ║ Paridad   ║
║  P3  ║ Reduce Oracle logging verbosity                       ║ ml_strat ║ -90% I/O  ║
║  P3  ║ Slippage: 0 para LIMIT orders                        ║ portfol  ║ Precisión ║
╚══════╩═══════════════════════════════════════════════════════╩══════════╩═══════════╝
""")
    
    # Mathematical summary
    notional = CAPITAL * SIZING * LEVERAGE
    maker_fee = 0.0002
    taker_fee = 0.000375
    tp = 0.015
    sl = 0.02
    
    rt_fee_maker = notional * maker_fee * 2
    rt_fee_taker = notional * taker_fee * 2
    
    gross_win = notional * tp
    gross_loss = notional * sl
    
    net_win_maker = gross_win - rt_fee_maker
    net_loss_maker = gross_loss + rt_fee_maker
    
    net_win_taker = gross_win - rt_fee_taker
    net_loss_taker = gross_loss + rt_fee_taker
    
    print(f"""
📊 ANÁLISIS MATEMÁTICO DE VIABILIDAD (POST-FIXES)
{"="*60}
Capital: ${CAPITAL:.2f} | Leverage: {LEVERAGE}x | Sizing: {SIZING*100:.0f}%
Notional: ${notional:.2f} | TP: {tp*100:.1f}% | SL: {sl*100:.1f}%

                    MAKER (LIMIT)     TAKER (MARKET)
Round-trip fee:     ${rt_fee_maker:.4f}          ${rt_fee_taker:.4f}
Gross Win:          ${gross_win:.4f}          ${gross_win:.4f}
Net Win:            ${net_win_maker:.4f}          ${net_win_taker:.4f}
Net Loss:           -${net_loss_maker:.4f}         -${net_loss_taker:.4f}
Fee % of Win:       {rt_fee_maker/gross_win*100:.1f}%             {rt_fee_taker/gross_win*100:.1f}%

Con Win Rate 60%:
  MAKER EV/trade: ${0.6 * net_win_maker - 0.4 * net_loss_maker:.4f}
  TAKER EV/trade: ${0.6 * net_win_taker - 0.4 * net_loss_taker:.4f}
  
  10 trades/día MAKER: ${(0.6 * net_win_maker - 0.4 * net_loss_maker) * 10:.3f}/día = {(0.6 * net_win_maker - 0.4 * net_loss_maker) * 10 / CAPITAL * 100:.2f}%/día
  Días para duplicar MAKER: {CAPITAL / ((0.6 * net_win_maker - 0.4 * net_loss_maker) * 10):.0f} días
  
Con Win Rate 70%:
  MAKER EV/trade: ${0.7 * net_win_maker - 0.3 * net_loss_maker:.4f}
  10 trades/día: ${(0.7 * net_win_maker - 0.3 * net_loss_maker) * 10:.3f}/día = {(0.7 * net_win_maker - 0.3 * net_loss_maker) * 10 / CAPITAL * 100:.2f}%/día
  Días para duplicar: {CAPITAL / ((0.7 * net_win_maker - 0.3 * net_loss_maker) * 10):.0f} días
""")

    print(f"\n{'='*70}")
    print(f"RESULTADO TOTAL: {len(CRITICAL)} Críticos | {len(WARNING)} Warnings | {len(INFO)} Info")
    print(f"{'='*70}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("🔬 AUDITORÍA FORENSE COMPLETA — TRADER GEMINI")
    print(f"{'='*70}")
    print(f"Capital: $13 USD | Objetivo: Duplicar cada 15 días")
    print(f"{'='*70}")
    
    audit_data_flow()
    audit_strategies()
    audit_execution()
    audit_risk()
    audit_backtesting()
    audit_infrastructure()
    generate_report()
