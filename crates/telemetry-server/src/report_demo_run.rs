//! Informe por operación de una corrida Demo (SOL M1, ítem 2).
//!
//! Herramienta de mi área (`telemetry-server`) que **juzga la Demo de 50 USDT**:
//! lee el log de una corrida emitido con `TG_LOG_TS=1` (cada línea lleva la
//! marca `[ts=epoch_ms]`) y el `income` del exchange, y produce una tabla por
//! operación con:
//!
//! símbolo · lado · hora de la señal · hora de la orden · hora del fill ·
//! latencia orden→fill · precio de entrada · precio de salida · bruto ·
//! comisión · funding · neto · motivo del cierre · quién lo ordenó,
//!
//! más los **totales del embudo por compuerta** (`FunnelCounter`).
//!
//! Contrato del log (cada línea, con prefijo `[ts=MS]` obligatorio en líneas
//! `DEMO`; si falta `[ts=]` en una línea `DEMO`, el parseo falla fail-closed:
//! no se fabrica latencia ni tiempos):
//!
//! ```text
//! [ts=1700000000000] DEMO SIGNAL symbol=BCHUSDT side=BUY trade_id=a3b-c1
//! [ts=1700000003000] DEMO ORDER  symbol=BCHUSDT side=BUY trade_id=a3b-c1
//! [ts=1700000004000] DEMO FILL   symbol=BCHUSDT trade_id=a3b-c1 price=0.012345
//! [ts=1700000005000] DEMO CLOSE  symbol=BCHUSDT trade_id=a3b-c1 price=0.012300 reason=STOP_LOSS by=RISK_ENGINE
//! ```
//!
//! El dinero (bruto/comisión/funding/neto) viene del `income` del exchange por
//! `trade_id`; los tiempos, lado, precios y el cierre vienen del log. La unión
//! es por `(símbolo, trade_id)`. Sin `trade_id` se usa `tran_id` (igual que en
//! `trade_ledger`).

use crate::funnel::FunnelCounter;
use crate::trade_ledger::IncomeRow;
use std::collections::BTreeMap;

/// Una operación ya unida: tiempos/precios del log + dinero del income.
#[derive(Debug, Clone, PartialEq)]
pub struct DemoOperation {
    pub symbol: String,
    pub trade_id: String,
    pub side: String,
    pub signal_ts_ms: Option<u128>,
    pub order_ts_ms: Option<u128>,
    pub fill_ts_ms: Option<u128>,
    /// fill_ts - order_ts (ms). `None` si falta alguno de los dos.
    pub order_to_fill_ms: Option<u128>,
    pub entry_price: Option<f64>,
    pub exit_price: Option<f64>,
    /// REALIZED_PNL (NaN si no viajó en el income de la operación).
    pub gross: f64,
    /// COMMISSION (NaN si no viajó).
    pub commission: f64,
    /// FUNDING_FEE (0.0 medido si no hubo filas de funding).
    pub funding: f64,
    /// gross + commission + funding (NaN si falta alguno).
    pub net: f64,
    pub close_reason: String,
    pub who_ordered: String,
}

/// Error de parseo del log de corrida. Fail-closed.
#[derive(Debug, Clone, PartialEq)]
pub struct LogParseError {
    pub line: usize,
    pub message: String,
}

/// Estado intermedio por `(símbolo, trade_id)` mientras se parsea el log.
/// Público para que `parse_run_log` lo devuelva y `build_report` lo consuma.
#[derive(Debug, Default, Clone)]
pub struct LogOp {
    side: Option<String>,
    signal_ts_ms: Option<u128>,
    order_ts_ms: Option<u128>,
    fill_ts_ms: Option<u128>,
    entry_price: Option<f64>,
    exit_price: Option<f64>,
    close_reason: Option<String>,
    who_ordered: Option<String>,
}

impl LogOp {
    fn entry_price(&self) -> Option<f64> {
        self.entry_price
    }
}

/// Extrae `[ts=MS]` del principio de la línea y devuelve `(ms, resto)`.
/// El `resto` empieza justo tras el `]`.
fn extract_ts(line: &str) -> Option<(u128, &str)> {
    let s = line.find("[ts=")?;
    let rest = &line[s + 4..]; // tras "ts="
    let end = rest.find(']')?;
    let num = rest[..end].parse::<u128>().ok()?;
    let after = &line[s + 4 + end + 1..]; // tras "]"
    Some((num, after))
}

/// Parsea el log de una corrida Demo. Devuelve un vector de
/// `(símbolo, trade_id, LogOp)` por operación encontrada.
///
/// Fail-closed: cualquier línea que contenga el tag `DEMO` PERO carezca de la
/// marca `[ts=...]` produce `Err`. Las líneas que no son `DEMO` se ignoran.
pub fn parse_run_log(log: &str) -> Result<Vec<(String, String, LogOp)>, LogParseError> {
    let mut map: BTreeMap<(String, String), LogOp> = BTreeMap::new();
    for (i, raw) in log.lines().enumerate() {
        let lineno = i + 1;
        if !raw.contains("DEMO") {
            continue;
        }
        let (ts, rest) = match extract_ts(raw) {
            Some(x) => x,
            None => {
                return Err(LogParseError {
                    line: lineno,
                    message: "línea DEMO sin marca [ts=...]; no se fabrica latencia".to_string(),
                });
            }
        };
        let tokens: Vec<&str> = rest.split_whitespace().collect();
        if tokens.first().map(|t| *t) != Some("DEMO") || tokens.len() < 2 {
            return Err(LogParseError {
                line: lineno,
                message: "línea DEMO malformada (falta tag DEMO o tipo de evento)".to_string(),
            });
        }
        let kind = tokens[1];
        let mut fields: BTreeMap<&str, &str> = BTreeMap::new();
        for tok in &tokens[2..] {
            if let Some(eq) = tok.find('=') {
                let k = &tok[..eq];
                let v = &tok[eq + 1..];
                fields.insert(k, v);
            }
        }
        let symbol = match fields.get("symbol") {
            Some(s) if !s.is_empty() => (*s).to_string(),
            _ => {
                return Err(LogParseError {
                    line: lineno,
                    message: "línea DEMO sin symbol".to_string(),
                });
            }
        };
        let trade_id = fields
            .get("trade_id")
            .map(|s| (*s).to_string())
            .unwrap_or_default();
        let key = (symbol, trade_id);
        let op = map.entry(key.clone()).or_default();
        match kind {
            "SIGNAL" => {
                op.signal_ts_ms = Some(ts);
                if let Some(sd) = fields.get("side") {
                    op.side = Some((*sd).to_string());
                }
            }
            "ORDER" => {
                // orden más temprana
                op.order_ts_ms = Some(match op.order_ts_ms {
                    Some(prev) => prev.min(ts),
                    None => ts,
                });
                if let Some(sd) = fields.get("side") {
                    op.side = Some((*sd).to_string());
                }
            }
            "FILL" => {
                // primer fill = entrada
                let is_entry = match op.fill_ts_ms {
                    Some(prev) => ts < prev,
                    None => true,
                };
                if is_entry {
                    op.fill_ts_ms = Some(ts);
                    if let Some(p) = fields.get("price").and_then(|v| v.parse::<f64>().ok()) {
                        op.entry_price = Some(p);
                    }
                }
            }
            "CLOSE" => {
                // último cierre = salida
                if let Some(p) = fields.get("price").and_then(|v| v.parse::<f64>().ok()) {
                    op.exit_price = Some(p);
                }
                if let Some(r) = fields.get("reason") {
                    op.close_reason = Some((*r).to_string());
                }
                if let Some(w) = fields.get("by") {
                    op.who_ordered = Some((*w).to_string());
                }
            }
            other => {
                return Err(LogParseError {
                    line: lineno,
                    message: format!("tipo de evento DEMO desconocido: {}", other),
                });
            }
        }
    }
    Ok(map.into_iter().map(|(k, v)| (k.0, k.1, v)).collect())
}

/// Clave de operación del income (igual que `trade_ledger::operation_key`).
fn income_op_key(r: &IncomeRow) -> (String, String) {
    let op = if r.trade_id.trim().is_empty() {
        r.tran_id.to_string()
    } else {
        r.trade_id.clone()
    };
    (r.symbol.clone(), op)
}

fn add_finite_opt(acc: f64, x: f64) -> f64 {
    if !x.is_finite() {
        return acc;
    }
    if acc.is_nan() {
        x
    } else {
        acc + x
    }
}

fn add_finite_from_zero(acc: f64, x: f64) -> f64 {
    if !x.is_finite() {
        return acc;
    }
    acc + x
}

/// Agrupa el income por `(símbolo, trade_id)` -> (gross, commission, funding).
/// `gross`/`commission` arrancan en NaN (ausente) y `funding` en 0.0 medido.
fn group_income(rows: &[IncomeRow]) -> BTreeMap<(String, String), (f64, f64, f64)> {
    let mut m: BTreeMap<(String, String), (f64, f64, f64)> = BTreeMap::new();
    for r in rows {
        let k = income_op_key(r);
        let e = m.entry(k).or_insert((f64::NAN, f64::NAN, 0.0));
        match r.income_type.as_str() {
            "REALIZED_PNL" => e.0 = add_finite_opt(e.0, r.income),
            "COMMISSION" => e.1 = add_finite_opt(e.1, r.income),
            "FUNDING_FEE" => e.2 = add_finite_from_zero(e.2, r.income),
            _ => {}
        }
    }
    m
}

/// Une el log y el income en el informe por operación. El conjunto de filas es
/// el de operaciones del income (autoritativo para el dinero); se enriquece con
/// tiempos/precios del log cuando hay cruce por `(símbolo, trade_id)`. Sin cruce
/// => tiempos/preecios en `None`/"N/D" (no se inventan).
pub fn build_report(
    log_ops: &[(String, String, LogOp)],
    income: &[IncomeRow],
) -> Vec<DemoOperation> {
    let groups = group_income(income);
    let mut out = Vec::new();
    for (k, triple) in &groups {
        let (sym, tid) = k;
        let (gross, comm, fund) = *triple;
        let lo = log_ops
            .iter()
            .find(|(s, t, _)| s == sym && t == tid);
        let net = if gross.is_finite() && comm.is_finite() && fund.is_finite() {
            gross + comm + fund
        } else {
            f64::NAN
        };
        let (signal_ts, order_ts, fill_ts, side, entry, exit, reason, who) = match lo {
            Some((_, _, op)) => (
                op.signal_ts_ms,
                op.order_ts_ms,
                op.fill_ts_ms,
                op.side.clone().unwrap_or_default(),
                op.entry_price(),
                op.exit_price,
                op.close_reason.clone().unwrap_or_else(|| "N/D".to_string()),
                op.who_ordered.clone().unwrap_or_else(|| "N/D".to_string()),
            ),
            None => (
                None,
                None,
                None,
                String::new(),
                None,
                None,
                "N/D".to_string(),
                "N/D".to_string(),
            ),
        };
        let order_to_fill = match (order_ts, fill_ts) {
            (Some(o), Some(f)) if f >= o => Some(f - o),
            _ => None,
        };
        out.push(DemoOperation {
            symbol: sym.clone(),
            trade_id: tid.clone(),
            side,
            signal_ts_ms: signal_ts,
            order_ts_ms: order_ts,
            fill_ts_ms: fill_ts,
            order_to_fill_ms: order_to_fill,
            entry_price: entry,
            exit_price: exit,
            gross,
            commission: comm,
            funding: fund,
            net,
            close_reason: reason,
            who_ordered: who,
        });
    }
    out
}

/// Totales del embudo por compuerta (etiqueta -> conteo) desde un
/// `FunnelCounter` vivo. En la Demo los alimentan risk-engine (latencia) y
/// god_engine (sistema inmune) vía `record_latency_blocked` /
/// `record_immune_veto` / `record_death`.
pub fn funnel_totals_by_gate(c: &FunnelCounter) -> BTreeMap<String, u64> {
    let snap = c.snapshot_by_gate();
    let mut out = BTreeMap::new();
    for (g, v) in snap {
        out.insert(g.as_str().to_string(), v);
    }
    out
}

/// Parsea un JSONL de income del exchange en `IncomeRow`. Una línea no válida se
/// ignora (fail-closed por línea, no por archivo). Campos: `symbol`,
/// `income_type`, `income`, `time_ms`, `trade_id`, `tran_id`, `asset`.
pub fn parse_income_jsonl(text: &str) -> Vec<IncomeRow> {
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(x) => x,
            Err(_) => continue,
        };
        let symbol = v
            .get("symbol")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let income_type = v
            .get("income_type")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let income = v.get("income").and_then(|x| x.as_f64()).unwrap_or(f64::NAN);
        let time_ms = v.get("time_ms").and_then(|x| x.as_u64()).unwrap_or(0);
        let trade_id = v
            .get("trade_id")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let tran_id = v.get("tran_id").and_then(|x| x.as_u64()).unwrap_or(0);
        let asset = v
            .get("asset")
            .and_then(|x| x.as_str())
            .unwrap_or("USDT")
            .to_string();
        out.push(IncomeRow {
            symbol,
            income_type,
            income,
            time_ms,
            tran_id,
            trade_id,
            asset,
        });
    }
    out
}

fn fmt_ms(v: Option<u128>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => "N/D".to_string(),
    }
}

fn fmt_opt(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{:.6}", x),
        _ => "N/D".to_string(),
    }
}

fn fmt_money(v: f64) -> String {
    if v.is_finite() {
        format!("{:.6}", v)
    } else {
        "N/D".to_string()
    }
}

/// Tabla de texto legible del informe por operación.
pub fn render_report_table(ops: &[DemoOperation]) -> String {
    let mut s = String::new();
    s.push_str(
        "símbolo | trade | lado | señal_ms | orden_ms | fill_ms | ord→fill_ms | entrada | salida | bruto | comisión | funding | neto | cierre | quien\n",
    );
    s.push_str(
        "--------|-------|------|----------|----------|---------|-------------|---------|--------|-------|----------|---------|-------|--------|------\n",
    );
    for o in ops {
        s.push_str(&format!(
            "{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}\n",
            o.symbol,
            o.trade_id,
            if o.side.is_empty() { "N/D" } else { &o.side },
            fmt_ms(o.signal_ts_ms),
            fmt_ms(o.order_ts_ms),
            fmt_ms(o.fill_ts_ms),
            fmt_ms(o.order_to_fill_ms),
            fmt_opt(o.entry_price),
            fmt_opt(o.exit_price),
            fmt_money(o.gross),
            fmt_money(o.commission),
            fmt_money(o.funding),
            fmt_money(o.net),
            o.close_reason,
            o.who_ordered,
        ));
    }
    s
}

/// Render de los totales del embudo por compuerta.
pub fn render_funnel_totals(totals: &BTreeMap<String, u64>) -> String {
    let mut s = String::from("EMBUDO por compuerta:\n");
    if totals.is_empty() {
        s.push_str("  (sin registros)\n");
    } else {
        for (g, v) in totals {
            s.push_str(&format!("  {:22} {}\n", g, v));
        }
    }
    s
}

/// CLI: `demo_report --log <run.log> --income <income.jsonl> [--funnel <gate.json>]`.
/// El log SIN `[ts=]` aborta con código 2 (no se juzga la Demo con tiempos falsos).
pub fn cli_main() {
    let args: Vec<String> = std::env::args().collect();
    let mut log_path: Option<String> = None;
    let mut income_path: Option<String> = None;
    let mut funnel_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--log" => {
                i += 1;
                log_path = args.get(i).cloned();
            }
            "--income" => {
                i += 1;
                income_path = args.get(i).cloned();
            }
            "--funnel" => {
                i += 1;
                funnel_path = args.get(i).cloned();
            }
            _ => {}
        }
        i += 1;
    }
    let (log_path, income_path) = match (log_path, income_path) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            eprintln!("uso: demo_report --log <run.log> --income <income.jsonl> [--funnel <gate.json>]");
            std::process::exit(2);
        }
    };
    let log_text = match std::fs::read_to_string(&log_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("no lee {}: {}", log_path, e);
            std::process::exit(2);
        }
    };
    let log_ops = match parse_run_log(&log_text) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("LOG SIN [ts=]: línea {}: {}", e.line, e.message);
            std::process::exit(2);
        }
    };
    let income_text = match std::fs::read_to_string(&income_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("no lee {}: {}", income_path, e);
            std::process::exit(2);
        }
    };
    let income = parse_income_jsonl(&income_text);
    let ops = build_report(&log_ops, &income);
    println!("{}", render_report_table(&ops));

    if let Some(fp) = funnel_path {
        match std::fs::read_to_string(&fp) {
            Ok(t) => match serde_json::from_str::<BTreeMap<String, u64>>(&t) {
                Ok(map) => println!("\n{}", render_funnel_totals(&map)),
                Err(e) => eprintln!("funnel json inválido: {}", e),
            },
            Err(e) => eprintln!("no lee {}: {}", fp, e),
        }
    } else {
        println!(
            "\nTOTALES DEL EMBUDO: se proveen en vivo por el FunnelCounter de telemetry-server\n(vía --funnel <gate.json> si se vuelca el snapshot)."
        );
    }
}

    #[cfg(test)]
    mod tests_report_demo {
        use crate::funnel::FunnelGate;
        use super::*;

    /// Los 5 ciclos reales de Demo (glm.md), en forma de `IncomeRow`.
    /// Ciclos 2 y 3 infieren el bruto del patrón "ídem" declarado; ver F6.
    fn five_real_cycles() -> Vec<IncomeRow> {
        fn row(symbol: &str, income_type: &str, income: f64, trade_id: &str, time_ms: u64) -> IncomeRow {
            IncomeRow {
                symbol: symbol.to_string(),
                income_type: income_type.to_string(),
                income,
                time_ms,
                tran_id: 0,
                trade_id: trade_id.to_string(),
                asset: "USDT".to_string(),
            }
        }
        vec![
            // 1) BCHUSDT 11-06  bruto −0.0006  comisión −0.0043  neto −0.0049
            row("BCHUSDT", "REALIZED_PNL", -0.0006, "a3b-c1", 1_784_000_000_000),
            row("BCHUSDT", "COMMISSION", -0.0043, "a3b-c1", 1_784_000_000_000),
            // 2) DOGEUSDT 11-08 "ídem"  bruto −0.0006  comisión −0.0043
            row("DOGEUSDT", "REALIZED_PNL", -0.0006, "a3b-c2", 1_785_000_000_000),
            row("DOGEUSDT", "COMMISSION", -0.0043, "a3b-c2", 1_785_000_000_000),
            // 3) BCHUSDT 11-09 "ídem"  bruto −0.0001  comisión −0.0043
            row("BCHUSDT", "REALIZED_PNL", -0.0001, "a3c-c3", 1_786_000_000_000),
            row("BCHUSDT", "COMMISSION", -0.0043, "a3c-c3", 1_786_000_000_000),
            // 4) ARBUSDT 11-10  bruto −0.0033  comisión −0.0082  neto −0.0115
            row("ARBUSDT", "REALIZED_PNL", -0.0033, "a3g-c4", 1_787_000_000_000),
            row("ARBUSDT", "COMMISSION", -0.0082, "a3g-c4", 1_787_000_000_000),
            // 5) LDOUSDT 11-10  bruto 0  comisión −0.0041  neto −0.0041
            row("LDOUSDT", "REALIZED_PNL", 0.0, "a3g-c5", 1_788_000_000_000),
            row("LDOUSDT", "COMMISSION", -0.0041, "a3g-c5", 1_788_000_000_000),
        ]
    }

    /// Log de ejemplo con marcas `[ts=...]` para los 5 ciclos reales.
    /// Cada operación: SIGNAL, ORDER (+3000 ms), FILL (+1000 ms, entrada),
    /// CLOSE (salida, motivo, quien). ord→fill = 2000 ms.
    fn sample_ts_log() -> String {
        let mut s = String::new();
        // 1) BCHUSDT a3b-c1 — entrada 0.012345, salida 0.012300 (pérdida)
        s.push_str("[ts=1784000000000] DEMO SIGNAL symbol=BCHUSDT side=BUY trade_id=a3b-c1\n");
        s.push_str("[ts=1784000003000] DEMO ORDER  symbol=BCHUSDT side=BUY trade_id=a3b-c1\n");
        s.push_str("[ts=1784000004000] DEMO FILL   symbol=BCHUSDT trade_id=a3b-c1 price=0.012345\n");
        s.push_str("[ts=1784000005000] DEMO CLOSE  symbol=BCHUSDT trade_id=a3b-c1 price=0.012300 reason=STOP_LOSS by=RISK_ENGINE\n");
        // 2) DOGEUSDT a3b-c2
        s.push_str("[ts=1785000000000] DEMO SIGNAL symbol=DOGEUSDT side=BUY trade_id=a3b-c2\n");
        s.push_str("[ts=1785000003000] DEMO ORDER  symbol=DOGEUSDT side=BUY trade_id=a3b-c2\n");
        s.push_str("[ts=1785000004000] DEMO FILL   symbol=DOGEUSDT trade_id=a3b-c2 price=0.123450\n");
        s.push_str("[ts=1785000005000] DEMO CLOSE  symbol=DOGEUSDT trade_id=a3b-c2 price=0.123400 reason=TP_HIT by=STRATEGY\n");
        // 3) BCHUSDT a3c-c3
        s.push_str("[ts=1786000000000] DEMO SIGNAL symbol=BCHUSDT side=SELL trade_id=a3c-c3\n");
        s.push_str("[ts=1786000003000] DEMO ORDER  symbol=BCHUSDT side=SELL trade_id=a3c-c3\n");
        s.push_str("[ts=1786000004000] DEMO FILL   symbol=BCHUSDT trade_id=a3c-c3 price=0.012111\n");
        s.push_str("[ts=1786000005000] DEMO CLOSE  symbol=BCHUSDT trade_id=a3c-c3 price=0.012100 reason=FLATTEN by=GOD_ENGINE\n");
        // 4) ARBUSDT a3g-c4
        s.push_str("[ts=1787000000000] DEMO SIGNAL symbol=ARBUSDT side=BUY trade_id=a3g-c4\n");
        s.push_str("[ts=1787000003000] DEMO ORDER  symbol=ARBUSDT side=BUY trade_id=a3g-c4\n");
        s.push_str("[ts=1787000004000] DEMO FILL   symbol=ARBUSDT trade_id=a3g-c4 price=0.543210\n");
        s.push_str("[ts=1787000005000] DEMO CLOSE  symbol=ARBUSDT trade_id=a3g-c4 price=0.543000 reason=STOP_LOSS by=RISK_ENGINE\n");
        // 5) LDOUSDT a3g-c5
        s.push_str("[ts=1788000000000] DEMO SIGNAL symbol=LDOUSDT side=BUY trade_id=a3g-c5\n");
        s.push_str("[ts=1788000003000] DEMO ORDER  symbol=LDOUSDT side=BUY trade_id=a3g-c5\n");
        s.push_str("[ts=1788000004000] DEMO FILL   symbol=LDOUSDT trade_id=a3g-c5 price=2.345000\n");
        s.push_str("[ts=1788000005000] DEMO CLOSE  symbol=LDOUSDT trade_id=a3g-c5 price=2.345000 reason=TIME_EXIT by=GOD_ENGINE\n");
        s
    }

    #[test]
    fn demo_report_green_cinco_ciclos_reales_con_log_ts() {
        let log = sample_ts_log();
        let log_ops = parse_run_log(&log).expect("GREEN: el log con [ts=] debe parsear");
        assert_eq!(log_ops.len(), 5, "5 operaciones en el log");

        let income = five_real_cycles();
        let ops = build_report(&log_ops, &income);
        assert_eq!(ops.len(), 5, "5 operaciones en el informe");

        // Op 1: BCHUSDT a3b-c1
        let o1 = ops.iter().find(|o| o.trade_id == "a3b-c1").expect("a3b-c1");
        assert_eq!(o1.symbol, "BCHUSDT");
        assert_eq!(o1.side, "BUY");
        assert_eq!(o1.signal_ts_ms, Some(1_784_000_000_000));
        assert_eq!(o1.order_ts_ms, Some(1_784_000_003_000));
        assert_eq!(o1.fill_ts_ms, Some(1_784_000_004_000));
        assert_eq!(o1.order_to_fill_ms, Some(1000), "ord→fill = 1000 ms");
        assert_eq!(o1.entry_price, Some(0.012345));
        assert_eq!(o1.exit_price, Some(0.012300));
        assert!((o1.gross + 0.0006).abs() < 1e-12, "gross = -0.0006");
        assert!((o1.commission + 0.0043).abs() < 1e-12, "comisión = -0.0043");
        assert!(o1.funding.abs() < 1e-12, "funding = 0");
        assert!((o1.net + 0.0049).abs() < 1e-12, "neto = -0.0049");
        assert_eq!(o1.close_reason, "STOP_LOSS");
        assert_eq!(o1.who_ordered, "RISK_ENGINE");

        // Op 5: LDOUSDT a3g-c5 (bruto 0, comisión −0.0041, neto −0.0041)
        let o5 = ops.iter().find(|o| o.trade_id == "a3g-c5").expect("a3g-c5");
        assert!(o5.gross.abs() < 1e-12, "bruto = 0 medido");
        assert!((o5.commission + 0.0041).abs() < 1e-12);
        assert!((o5.net + 0.0041).abs() < 1e-12, "neto = -0.0041");

        // Totales del embudo (contador vivo de ejemplo).
        let f = FunnelCounter::new();
        f.record_death(FunnelGate::SpectrumEval, "BCHUSDT", 1_784_000);
        f.record_death(FunnelGate::MlGate, "DOGEUSDT", 1_785_000);
        f.record_death(FunnelGate::Flatten, "ARBUSDT", 1_787_000);
        f.record_latency_blocked("BTCUSDT", 1_700_000, 1675); // H-07
        f.record_immune_veto("BCHUSDT", 1_784_000, 2); // nota 14:45
        let totals = funnel_totals_by_gate(&f);
        assert_eq!(totals.get("SPECTRUM_EVAL").copied(), Some(1));
        assert_eq!(totals.get("ML_GATE").copied(), Some(1));
        assert_eq!(totals.get("FLATTEN").copied(), Some(1));
        assert_eq!(totals.get("LATENCY_KILL_SWITCH").copied(), Some(1675));
        assert_eq!(totals.get("IMMUNE_SYSTEM").copied(), Some(2));
    }

    #[test]
    fn demo_report_red_log_sin_ts_es_rechazado() {
        // Log de una corrida SIN TG_LOG_TS=1: líneas DEMO sin [ts=].
        let bad = "\
DEMO SIGNAL symbol=BCHUSDT side=BUY trade_id=a3b-c1
DEMO ORDER  symbol=BCHUSDT side=BUY trade_id=a3b-c1
DEMO FILL   symbol=BCHUSDT trade_id=a3b-c1 price=0.012345
";
        let r = parse_run_log(bad);
        assert!(r.is_err(), "RED: un log sin [ts=] debe rechazarse fail-closed");
        let e = r.unwrap_err();
        assert_eq!(e.line, 1);
        assert!(e.message.contains("[ts="));
    }

    #[test]
    fn demo_report_red_join_sin_log_deja_tiempos_nd() {
        // Income real (5 ciclos) pero log vacío: el dinero está, los tiempos
        // NO se inventan (quedan N/D / None).
        let log_ops = parse_run_log("").expect("log vacío parsea a 0 ops");
        let income = five_real_cycles();
        let ops = build_report(&log_ops, &income);
        assert_eq!(ops.len(), 5);
        for o in &ops {
            assert!(o.signal_ts_ms.is_none());
            assert!(o.order_ts_ms.is_none());
            assert!(o.fill_ts_ms.is_none());
            assert!(o.order_to_fill_ms.is_none());
            assert!(o.entry_price.is_none());
            assert!(o.exit_price.is_none());
            assert_eq!(o.close_reason, "N/D");
            assert_eq!(o.who_ordered, "N/D");
            // el dinero sí está
            assert!(o.gross.is_finite());
            assert!(o.commission.is_finite());
            assert!(o.net.is_finite());
        }
        // Verificación puntual del neto del ciclo 4 (ARB).
        let o4 = ops.iter().find(|o| o.trade_id == "a3g-c4").unwrap();
        assert!((o4.net + 0.0115).abs() < 1e-12);
    }

    #[test]
    fn demo_report_red_funnel_vacio_sin_registros() {
        let f = FunnelCounter::new();
        let totals = funnel_totals_by_gate(&f);
        assert!(totals.is_empty(), "sin registros => embudo vacío");
    }

    #[test]
    fn demo_report_parse_income_jsonl() {
        let jsonl = "\
{\"symbol\":\"BCHUSDT\",\"income_type\":\"REALIZED_PNL\",\"income\":-0.0006,\"time_ms\":1784000000000,\"trade_id\":\"a3b-c1\"}
{\"symbol\":\"BCHUSDT\",\"income_type\":\"COMMISSION\",\"income\":-0.0043,\"time_ms\":1784000000000,\"trade_id\":\"a3b-c1\"}
";
        let rows = parse_income_jsonl(jsonl);
        assert_eq!(rows.len(), 2);
        let ops = build_report(&[], &rows);
        assert_eq!(ops.len(), 1);
        let o = &ops[0];
        assert!((o.gross + 0.0006).abs() < 1e-12);
        assert!((o.commission + 0.0043).abs() < 1e-12);
        assert!((o.net + 0.0049).abs() < 1e-12);
    }
}
