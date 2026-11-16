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

/// M1 T2 (QRA-PARSER-M1, punto 1): prefijo de logger reconocido y ANCLADO al
/// principio de la línea: `[ts=<entero>]` seguido de espacio o fin de línea.
/// Devuelve `(marca, carga útil)`. Sin prefijo, la carga es la línea entera
/// (stdout de `println!`). Un `[ts=` inicial que no es un entero cerrado es un
/// prefijo mal formado: error con número de línea, nunca se adivina.
fn separar_prefijo(raw: &str) -> Result<(Option<u128>, &str), &'static str> {
    let s = raw.strip_prefix('\u{feff}').unwrap_or(raw);
    let Some(r) = s.strip_prefix("[ts=") else {
        return Ok((None, s));
    };
    let Some(end) = r.find(']') else {
        return Err("prefijo [ts=...] mal formado: sin cierre");
    };
    let num = &r[..end];
    if num.is_empty() || !num.bytes().all(|b| b.is_ascii_digit()) {
        return Err("prefijo [ts=...] mal formado: la marca no es un entero");
    }
    let Ok(ts) = num.parse::<u128>() else {
        return Err("prefijo [ts=...] mal formado: entero fuera de rango");
    };
    let rest = &r[end + 1..];
    if !(rest.is_empty() || rest.starts_with(' ') || rest.starts_with('\t')) {
        return Err("prefijo [ts=...] mal formado: sin separador tras la marca");
    }
    Ok((Some(ts), rest.trim_start()))
}

/// Primer token de la carga útil (tras el prefijo opcional).
fn primer_token(payload: &str) -> Option<&str> {
    payload.split_whitespace().next()
}

/// Parsea el log de una corrida Demo. Devuelve un vector de
/// `(símbolo, trade_id, LogOp)` por operación encontrada.
///
/// M1 T2 (QRA-PARSER-M1, punto 1): se clasifica por el PRIMER TOKEN de la carga
/// útil tras el prefijo de logger opcional y anclado. Sólo el token exacto
/// `DEMO` es del esquema legacy (SIGNAL/ORDER/FILL/CLOSE) y exige `[ts=entero]`
/// inicial. La palabra DEMO en texto humano (`Cayendo a DEMO`,
/// `[MODO WARMUP/DEMO ACTIVO]`), las líneas `DEMO_TEL` (lector propio:
/// [`parse_demo_tel`]) y la línea humana `💧 [FILL]` no activan este parser.
/// Fail-closed: línea `DEMO` sin marca inicial, tipo desconocido, prefijo mal
/// formado o campo repetido (metadatos ambiguos) dan `Err` con su número de línea.
pub fn parse_run_log(log: &str) -> Result<Vec<(String, String, LogOp)>, LogParseError> {
    let mut map: BTreeMap<(String, String), LogOp> = BTreeMap::new();
    for (i, raw) in log.lines().enumerate() {
        let lineno = i + 1;
        let (ts, payload) = match separar_prefijo(raw) {
            Ok(x) => x,
            Err(message) => {
                return Err(LogParseError { line: lineno, message: message.to_string() });
            }
        };
        if primer_token(payload) != Some("DEMO") {
            continue;
        }
        let Some(ts) = ts else {
            return Err(LogParseError {
                line: lineno,
                message: "línea DEMO sin marca [ts=...] inicial; no se fabrica latencia".to_string(),
            });
        };
        let tokens: Vec<&str> = payload.split_whitespace().collect();
        if tokens.len() < 2 {
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
                if fields.insert(k, v).is_some() {
                    return Err(LogParseError {
                        line: lineno,
                        message: format!("línea DEMO con el campo {k} repetido (metadatos ambiguos)"),
                    });
                }
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

// ===========================================================================
// M1 T2 (QRA-PARSER-M1, puntos 2 a 5): lector DEMO_TEL, línea humana del FILL,
// unión de fills por identidad nativa e informe mixto con cobertura.
// ===========================================================================

/// Ámbito acreditado de una fuente (venue y cuenta). Lo declara quien entrega
/// el archivo (manifiesto de entradas); el lector no lo deduce del texto.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ambito {
    pub venue: String,
    pub cuenta: String,
}

impl Ambito {
    pub fn new(venue: &str, cuenta: &str) -> Self {
        Self { venue: venue.to_string(), cuenta: cuenta.to_string() }
    }
}

/// Un evento DEMO_TEL reconocido y válido.
#[derive(Debug, Clone, PartialEq)]
pub struct EventoTel {
    pub linea: usize,
    pub tipo: String,
    /// `ts` propio del evento (campo de la línea). `None` sólo en SYMBOL_SPEC.
    pub ts_evento: Option<u64>,
    /// Marca del prefijo del logger (hora de recepción/escritura), si la hay.
    pub ts_recepcion: Option<u128>,
    pub ambito: Ambito,
    pub campos: BTreeMap<String, String>,
}

/// Diagnóstico de cobertura con número de línea.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticoTel {
    pub fuente: String,
    pub linea: usize,
    pub mensaje: String,
}

/// La línea humana `💧 [FILL]` tal cual (evidencia de diagnóstico: orden,
/// lado, cantidad acumulada). Su formato real no trae tradeId ni reloj: aquí
/// no se inventan, y nunca crea una operación ni un fill nativo.
#[derive(Debug, Clone, PartialEq)]
pub struct FillHumano {
    pub linea: usize,
    /// Bytes exactos de la línea.
    pub texto: String,
    pub symbol: String,
    pub side: String,
    pub order_type: String,
    pub client_order_id: String,
    pub price: Option<f64>,
    /// `acum a/b` tal cual.
    pub acum: String,
    pub status: String,
    /// Marca del prefijo del logger si la línea la trae (la del humo no).
    pub ts_recepcion: Option<u128>,
}

/// Lectura de una fuente (stdout o un segmento del logger).
#[derive(Debug, Clone, PartialEq)]
pub struct LecturaTel {
    pub fuente: String,
    pub ambito: Ambito,
    pub lineas: usize,
    pub eventos: Vec<EventoTel>,
    pub diagnosticos: Vec<DiagnosticoTel>,
    /// Tipos DEMO_TEL futuros/desconocidos: se reportan, no se interpretan.
    pub tipos_desconocidos: Vec<(usize, String)>,
    pub fills_humanos: Vec<FillHumano>,
}

/// Esquema DEMO_TEL del SHA: (tipo, campos requeridos, campo final que admite
/// espacios). `ts` es el reloj propio del evento; SYMBOL_SPEC no lo trae.
fn esquema_tel(tipo: &str) -> Option<(&'static [&'static str], Option<&'static str>)> {
    Some(match tipo {
        "SYMBOL_SPEC" => (&["symbol", "min_notional", "tick_size", "step_size", "fuente", "exchange_info"], None),
        "DECISION_MID" => (&["ts", "signal_id", "symbol", "slot", "dir", "mid", "bid", "ask"], None),
        "ORDER_SEND" => (
            &["tipo", "ts", "symbol", "is_long", "qty", "client_order_id", "lev", "route"],
            Some("route"),
        ),
        "ORDER_RESULT" => (&["ts", "symbol", "client_order_id", "resultado"], None),
        "FILL" => (
            &["ts", "symbol", "side", "price", "qty", "cum_qty", "client_order_id", "trade_id", "is_maker", "status"],
            None,
        ),
        "TRADE_FEE" => (
            &["ts", "symbol", "client_order_id", "trade_id", "fee", "fee_asset", "is_maker", "alcance"],
            None,
        ),
        "FUNDING" => (
            &[
                "ts", "symbol", "income", "asset", "tran_id", "income_type", "alcance", "ventana_desde_ms",
                "ventana_hasta_ms", "filtro", "paginas_max", "cobertura",
            ],
            None,
        ),
        "CAPITAL_UPDATE" => (
            &["ts", "wallet_balance", "fuente", "era", "socket", "marca_ms", "hora_evento_ms"],
            None,
        ),
        "UNREALIZED_PNL" => (&["ts", "total_upnl", "open_positions"], None),
        "TRADE_CLOSE" => (
            &["ts", "close_id", "symbol", "slot", "dir", "entry_price", "exit_price", "gross_pnl", "reason"],
            None,
        ),
        "IMMUNE_TRIGGER" => (
            &[
                "ts", "motivo", "alcance", "freno_capital", "freno_capital_plano", "peak_capital", "peak_plano",
                "equity_del_freno", "equity_del_freno_plano", "caida_capital_pct", "tope_capital_pct",
                "operator_lock", "insolvent", "dd_breach", "lat_breach", "latency_strikes", "revision",
                "lecturas_posteriores",
            ],
            None,
        ),
        "AUDIT_SUMMARY" => (&["ts", "status", "closed_positions", "cancelled_orders"], None),
        _ => return None,
    })
}

/// Campos decimales (finitos) y enteros; los que admiten ausencia explícita.
const DECIMALES: &[&str] = &[
    "min_notional", "tick_size", "step_size", "mid", "bid", "ask", "qty", "price", "cum_qty", "fee", "income",
    "wallet_balance", "total_upnl", "entry_price", "exit_price", "gross_pnl", "freno_capital", "peak_capital",
    "equity_del_freno", "caida_capital_pct", "tope_capital_pct",
];
const ENTEROS: &[&str] = &[
    "ts", "signal_id", "slot", "lev", "trade_id", "tran_id", "era", "socket", "marca_ms", "hora_evento_ms",
    "open_positions", "close_id", "latency_strikes", "closed_positions", "cancelled_orders", "ventana_desde_ms",
    "ventana_hasta_ms", "paginas_max",
];
/// (tipo, campo) que puede valer `AUSENTE` o `N/D` (ausencia explícita).
fn admite_ausencia(tipo: &str, campo: &str) -> bool {
    matches!(
        (tipo, campo),
        ("TRADE_FEE", "trade_id")
            | ("TRADE_FEE", "fee_asset")
            | ("FUNDING", "tran_id")
            | ("FUNDING", "cobertura")
            | ("SYMBOL_SPEC", "exchange_info")
            | ("CAPITAL_UPDATE", "marca_ms")
            | ("CAPITAL_UPDATE", "hora_evento_ms")
            | ("IMMUNE_TRIGGER", "equity_del_freno")
            | ("IMMUNE_TRIGGER", "revision")
    )
}

fn es_ausencia(v: &str) -> bool {
    v == "AUSENTE" || v == "N/D"
}

fn decimal_finito(v: &str) -> bool {
    !v.is_empty()
        && v.chars().all(|c| c.is_ascii_digit() || c == '.' || c == '-')
        && v.parse::<f64>().map(f64::is_finite).unwrap_or(false)
}

fn entero(v: &str) -> bool {
    !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()) && v.parse::<u64>().is_ok()
}

/// Valida un DEMO_TEL contra su esquema. `Ok(None)` = tipo desconocido.
fn validar_tel(
    tipo: &str,
    cuerpo: &[&str],
) -> Result<Option<BTreeMap<String, String>>, String> {
    let Some((requeridos, con_espacios)) = esquema_tel(tipo) else {
        return Ok(None);
    };
    let mut campos: BTreeMap<String, String> = BTreeMap::new();
    let mut ultimo: Option<String> = None;
    for tok in cuerpo {
        match tok.find('=') {
            Some(eq) if requeridos.contains(&&tok[..eq]) => {
                let k = &tok[..eq];
                if campos.contains_key(k) {
                    return Err(format!("{tipo}: campo {k} repetido (metadatos ambiguos)"));
                }
                campos.insert(k.to_string(), tok[eq + 1..].to_string());
                ultimo = Some(k.to_string());
            }
            _ => match (&ultimo, con_espacios) {
                (Some(u), Some(c)) if u == c => {
                    let v = campos.get_mut(c).expect("campo abierto");
                    v.push(' ');
                    v.push_str(tok);
                }
                _ => return Err(format!("{tipo}: texto fuera de esquema: {tok:?}")),
            },
        }
    }
    for k in requeridos {
        let Some(v) = campos.get(*k) else {
            return Err(format!("{tipo}: falta el campo {k} (línea truncada o esquema distinto)"));
        };
        if v.is_empty() {
            return Err(format!("{tipo}: campo {k} vacío"));
        }
        if admite_ausencia(tipo, k) && es_ausencia(v) {
            continue;
        }
        if v == FUERA_DE_DOMINIO {
            return Err(format!(
                "{tipo}: diagnóstico de cobertura: campo {k} fuera del dominio admitido por el emisor (sin valor)"
            ));
        }
        if DECIMALES.contains(k) && !decimal_finito(v) {
            return Err(format!("{tipo}: {k}={v} no es un decimal finito"));
        }
        if ENTEROS.contains(k) && !entero(v) {
            return Err(format!("{tipo}: {k}={v} no es un entero"));
        }
    }
    // Identidad nativa: un FILL sin tradeId no es un fill unible.
    if tipo == "FILL" && campos.get("trade_id").map(|t| t == "0").unwrap_or(true) {
        return Err("FILL: trade_id 0 (sin identidad nativa)".to_string());
    }
    validar_dominios(tipo, &campos)?;
    Ok(Some(campos))
}

/// DEMO_TEL T3 (T3-6): dominio de los campos textuales que el esquema cerraba
/// sólo por "no vacío". Un valor fuera de dominio es diagnóstico (fuente, línea,
/// campo y valor) y el evento no entra en los joins.
const BOOLEANOS: &[(&str, &str)] = &[
    ("FILL", "is_maker"),
    ("TRADE_FEE", "is_maker"),
    ("ORDER_SEND", "is_long"),
    ("IMMUNE_TRIGGER", "operator_lock"),
    ("IMMUNE_TRIGGER", "insolvent"),
    ("IMMUNE_TRIGGER", "dd_breach"),
    ("IMMUNE_TRIGGER", "lat_breach"),
];

/// Estados de `OrderStatus` que existen pero sólo describen la orden: con ellos
/// una línea FILL no acredita una ejecución.
const ESTADOS_SOLO_DE_ORDEN: &[&str] = &["New", "Canceled", "Expired", "Rejected"];

/// DEMO_TEL T3 (T3-5): códigos cortos de los planos de IMMUNE_TRIGGER y su
/// significado (compatibilidad de lectura: también se aceptan los textos
/// largos de T2).
pub const PLANOS_IMMUNE: &[(&str, &str, &str)] = &[
    ("freno_capital_plano", "FRENO", "drawdown.capital_del_freno(min(arena.unified_capital,arena.capital_exchange))"),
    ("peak_plano", "PICO", "drawdown.registrar_pico(arena.pico_capital,freno_capital)"),
    ("equity_del_freno_plano", "PUENTE", "puente.equity_del_freno"),
];

/// Marca del emisor para un valor fuera del dominio admitido (IMMUNE_TRIGGER).
pub const FUERA_DE_DOMINIO: &str = "FUERA_DE_DOMINIO";

fn fuera_de_dominio(tipo: &str, k: &str, v: &str, admitidos: &str) -> String {
    format!("{tipo}: dominio: campo {k}={v} fuera de dominio (admitidos: {admitidos})")
}

fn validar_dominios(tipo: &str, campos: &BTreeMap<String, String>) -> Result<(), String> {
    let v = |k: &str| campos.get(k).map(String::as_str).unwrap_or("");
    for (t, k) in BOOLEANOS {
        if *t == tipo && !matches!(v(k), "true" | "false") {
            return Err(fuera_de_dominio(tipo, k, v(k), "true, false"));
        }
    }
    match tipo {
        "FILL" => {
            if !matches!(v("side"), "BUY" | "SELL") {
                return Err(fuera_de_dominio(tipo, "side", v("side"), "BUY, SELL"));
            }
            match v("status") {
                "Filled" | "PartiallyFilled" => {}
                "Unknown" => {
                    return Err(format!(
                        "FILL: evidencia insuficiente: campo status=Unknown (estado no reconocido por el registro; no acredita ejecución)"
                    ));
                }
                s if ESTADOS_SOLO_DE_ORDEN.contains(&s) => {
                    return Err(format!(
                        "FILL: dominio: campo status={s} es un estado sólo de orden; no acredita un fill ejecutado"
                    ));
                }
                s => return Err(fuera_de_dominio(tipo, "status", s, "Filled, PartiallyFilled")),
            }
        }
        "ORDER_SEND" if v("tipo") != "INTENTO" => {
            return Err(fuera_de_dominio(tipo, "tipo", v("tipo"), "INTENTO"));
        }
        "ORDER_RESULT" if !matches!(v("resultado"), "ENTREGADA" | "RECHAZADA" | "AMBIGUA") => {
            return Err(fuera_de_dominio(tipo, "resultado", v("resultado"), "ENTREGADA, RECHAZADA, AMBIGUA"));
        }
        // T3-5: el campo terminal tiene un valor cerrado; un corte dentro de él
        // (`lecturas_posteriores=ning`) es una línea truncada, no un evento.
        "IMMUNE_TRIGGER" => {
            if v("lecturas_posteriores") != "ninguna" {
                return Err(format!(
                    "IMMUNE_TRIGGER: campo terminal lecturas_posteriores={} incompleto o fuera de dominio (línea truncada o esquema distinto)",
                    v("lecturas_posteriores")
                ));
            }
            if v("alcance") != "CUENTA" {
                return Err(fuera_de_dominio(tipo, "alcance", v("alcance"), "CUENTA"));
            }
            if !matches!(v("motivo"), "OPERATOR_LOCK" | "INSOLVENTE" | "DRAWDOWN" | "LATENCIA") {
                return Err(fuera_de_dominio(tipo, "motivo", v("motivo"), "OPERATOR_LOCK, INSOLVENTE, DRAWDOWN, LATENCIA"));
            }
        }
        _ => {}
    }
    Ok(())
}

/// Línea humana `💧 [FILL] SYM SIDE TYPE COID @ PRICE (acum a/b status S)`.
fn parse_fill_humano(linea: usize, raw: &str, ts: Option<u128>, payload: &str) -> Option<FillHumano> {
    let rest = payload.strip_prefix("💧 [FILL] ")?;
    let (cabeza, cola) = rest.split_once(" @ ")?;
    let mut c = cabeza.split_whitespace();
    let (symbol, side, order_type, client_order_id) = (c.next()?, c.next()?, c.next()?, c.next()?);
    let (precio, parentesis) = cola.split_once(" (acum ")?;
    let (acum, status) = parentesis.strip_suffix(')')?.split_once(" status ")?;
    Some(FillHumano {
        linea,
        texto: raw.to_string(),
        symbol: symbol.to_string(),
        side: side.to_string(),
        order_type: order_type.to_string(),
        client_order_id: client_order_id.to_string(),
        price: precio.parse::<f64>().ok().filter(|p| p.is_finite()),
        acum: acum.to_string(),
        status: status.to_string(),
        ts_recepcion: ts,
    })
}

/// Líneas humanas `💧 [FILL]` (conteo histórico: una observación por línea).
pub fn parse_human_fills(text: &str) -> Vec<FillHumano> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        if let Ok((ts, payload)) = separar_prefijo(raw) {
            if let Some(f) = parse_fill_humano(i + 1, raw, ts, payload) {
                out.push(f);
            }
        }
    }
    out
}

/// M1 T2 (QRA-PARSER-M1, punto 3): lector por eventos DEMO_TEL, separado del
/// esquema DEMO. Acepta la forma sin prefijo (stdout de `println!`) y la forma
/// con prefijo de logger; conserva el `ts` propio del evento y la hora de
/// recepción por separado. Un evento reconocido truncado o inválido queda como
/// diagnóstico de cobertura (no como evento); un tipo desconocido se reporta.
/// Una línea con prefijo mal formado también es diagnóstico. No deduce orden
/// causal entre fuentes distintas.
pub fn parse_demo_tel(text: &str, fuente: &str, ambito: &Ambito) -> LecturaTel {
    let mut l = LecturaTel {
        fuente: fuente.to_string(),
        ambito: ambito.clone(),
        lineas: 0,
        eventos: Vec::new(),
        diagnosticos: Vec::new(),
        tipos_desconocidos: Vec::new(),
        fills_humanos: Vec::new(),
    };
    for (i, raw) in text.lines().enumerate() {
        let linea = i + 1;
        l.lineas = linea;
        let diag = |mensaje: String| DiagnosticoTel { fuente: fuente.to_string(), linea, mensaje };
        let (ts, payload) = match separar_prefijo(raw) {
            Ok(x) => x,
            Err(m) => {
                if raw.contains("DEMO_TEL") {
                    l.diagnosticos.push(diag(m.to_string()));
                }
                continue;
            }
        };
        if let Some(f) = parse_fill_humano(linea, raw, ts, payload) {
            l.fills_humanos.push(f);
            continue;
        }
        if primer_token(payload) != Some("DEMO_TEL") {
            continue;
        }
        let tokens: Vec<&str> = payload.split_whitespace().collect();
        let Some(tipo) = tokens.get(1).and_then(|t| t.strip_prefix("event=")) else {
            l.diagnosticos.push(diag("DEMO_TEL sin event=<tipo>".to_string()));
            continue;
        };
        match validar_tel(tipo, &tokens[2..]) {
            Ok(Some(campos)) => {
                let ts_evento = campos.get("ts").and_then(|v| v.parse::<u64>().ok());
                l.eventos.push(EventoTel {
                    linea,
                    tipo: tipo.to_string(),
                    ts_evento,
                    ts_recepcion: ts,
                    ambito: ambito.clone(),
                    campos,
                });
            }
            Ok(None) => l.tipos_desconocidos.push((linea, tipo.to_string())),
            Err(m) => l.diagnosticos.push(diag(m)),
        }
    }
    l
}

/// Clave de identidad nativa de un fill: (venue, cuenta, símbolo, tradeId).
pub type ClaveFill = (String, String, String, String);

/// Un fill DEMO_TEL con identidad nativa.
#[derive(Debug, Clone, PartialEq)]
pub struct FillNativo {
    pub ambito: Ambito,
    pub symbol: String,
    pub trade_id: String,
    pub client_order_id: String,
    pub side: String,
    pub price: f64,
    pub qty: f64,
    pub cum_qty: f64,
    pub status: String,
    pub ts_evento: u64,
    pub ts_recepcion: Option<u128>,
    pub fuente: String,
    pub linea: usize,
}

impl FillNativo {
    fn mismo_hecho(&self, o: &FillNativo) -> bool {
        self.client_order_id == o.client_order_id
            && self.side == o.side
            && self.price.to_bits() == o.price.to_bits()
            && self.qty.to_bits() == o.qty.to_bits()
            && self.cum_qty.to_bits() == o.cum_qty.to_bits()
            && self.status == o.status
            && self.ts_evento == o.ts_evento
    }
}

/// Fills unidos por identidad nativa (QRA-PARSER-M1, punto 4).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FillsUnidos {
    pub fills: BTreeMap<ClaveFill, FillNativo>,
    /// Repeticiones idénticas del mismo fill (deduplicadas).
    pub repetidos: usize,
    /// Mismo tradeId con campos en conflicto: bloquea la conciliación de esa clave.
    pub conflictos: Vec<DiagnosticoTel>,
    pub claves_en_conflicto: Vec<ClaveFill>,
}

/// Une los FILL DEMO_TEL de todas las fuentes SÓLO por identidad nativa
/// (venue/cuenta/símbolo/tradeId). El clientOrderId identifica la orden y puede
/// agrupar varios fills; no es clave. No une por símbolo, slot ni cercanía.
pub fn unir_fills(lecturas: &[LecturaTel]) -> FillsUnidos {
    let mut u = FillsUnidos::default();
    for l in lecturas {
        for e in l.eventos.iter().filter(|e| e.tipo == "FILL") {
            let c = &e.campos;
            let num = |k: &str| c.get(k).and_then(|v| v.parse::<f64>().ok()).unwrap_or(f64::NAN);
            let f = FillNativo {
                ambito: e.ambito.clone(),
                symbol: c["symbol"].clone(),
                trade_id: c["trade_id"].clone(),
                client_order_id: c["client_order_id"].clone(),
                side: c["side"].clone(),
                price: num("price"),
                qty: num("qty"),
                cum_qty: num("cum_qty"),
                status: c["status"].clone(),
                ts_evento: e.ts_evento.unwrap_or(0),
                ts_recepcion: e.ts_recepcion,
                fuente: l.fuente.clone(),
                linea: e.linea,
            };
            let clave = (f.ambito.venue.clone(), f.ambito.cuenta.clone(), f.symbol.clone(), f.trade_id.clone());
            if u.claves_en_conflicto.contains(&clave) {
                continue;
            }
            match u.fills.get(&clave) {
                None => {
                    u.fills.insert(clave, f);
                }
                Some(previo) if previo.mismo_hecho(&f) => u.repetidos += 1,
                Some(previo) => {
                    u.conflictos.push(DiagnosticoTel {
                        fuente: l.fuente.clone(),
                        linea: e.linea,
                        mensaje: format!(
                            "FILL {} {} trade_id={}: campos en conflicto con {}:{} — conciliación bloqueada",
                            f.ambito.venue, f.symbol, f.trade_id, previo.fuente, previo.linea
                        ),
                    });
                    u.fills.remove(&clave);
                    u.claves_en_conflicto.push(clave);
                }
            }
        }
    }
    u
}

/// Ventana declarada del income (cobertura): filas fuera se excluyen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VentanaIncome {
    pub desde_ms: u64,
    pub hasta_ms: u64,
}

/// DEMO_TEL T3 (T3-4): un importe observado en el income, con su activo. Los
/// importes de activos distintos nunca se suman entre sí.
#[derive(Debug, Clone, PartialEq)]
pub struct ImporteObservado {
    pub tipo: String,
    pub activo: String,
    pub importe: f64,
    pub filas: usize,
}

/// Una fila del informe mixto: la fila legacy (dinero del income) más el join.
#[derive(Debug, Clone, PartialEq)]
pub struct FilaMixta {
    pub op: DemoOperation,
    /// `LEGACY` (líneas DEMO), `FILL_NATIVO` (DEMO_TEL por tradeId) o `N/D`.
    pub join: &'static str,
    pub fill: Option<FillNativo>,
    /// Funding y neto oficiales: sólo con cobertura acreditada. T3 no construye
    /// el certificador de cobertura del income (T3-3): siempre N/D (no cero).
    pub funding_acreditado: Option<f64>,
    pub neto_acreditado: Option<f64>,
    /// DEMO_TEL T3 (T3-2): `TRADE_ID` (tradeId explícito: identidad nativa) o
    /// `TRAN_ID` (el fallback sólo identifica la fila contable: no une).
    pub origen_clave: &'static str,
    /// DEMO_TEL T3 (T3-4): importes observados por (tipo, activo).
    pub observado: Vec<ImporteObservado>,
    /// DEMO_TEL T3 (T3-4): motivo por el que la operación está incompleta en el
    /// numerario del alcance (p. ej. una comisión en otro activo).
    pub incompleta: Option<String>,
}

/// Informe mixto con cobertura y estado de aceptación.
#[derive(Debug, Clone, PartialEq)]
pub struct InformeMixto {
    pub filas: Vec<FilaMixta>,
    pub ambito: Ambito,
    pub ventana: Option<VentanaIncome>,
    pub fuentes: Vec<(String, usize, usize)>,
    pub diagnosticos: Vec<DiagnosticoTel>,
    pub tipos_desconocidos: usize,
    pub fills_humanos: usize,
    pub fills_nativos: usize,
    pub fills_repetidos: usize,
    pub fills_fuera_de_ambito: usize,
    pub fills_sin_pareja: Vec<ClaveFill>,
    pub income_fuera_de_ventana: usize,
    pub income_duplicado: usize,
    pub apto: bool,
    pub motivos_no_apto: Vec<String>,
    /// DEMO_TEL T3 (T3-3): numerario declarado del alcance (`--activo`).
    pub activo: Option<String>,
    /// DEMO_TEL T3 (T3-3): defectos de las filas del income (línea y motivo).
    pub income_diagnosticos: Vec<DiagnosticoTel>,
    /// DEMO_TEL T3 (T3-4): operaciones cuya suma de la tabla LEGACY mezcla activos.
    pub legacy_sin_autoridad: Vec<String>,
}

/// DEMO_TEL T3 (T3-3): lo que se sabe del income además de sus filas.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EvidenciaIncome {
    /// Numerario del alcance (`--activo`); `None` = no declarado.
    pub activo: Option<String>,
    /// Diagnósticos del lector auditado ([`parse_income_jsonl_auditado`]).
    pub diagnosticos: Vec<DiagnosticoTel>,
    /// Líneas no vacías del archivo de income leído.
    pub lineas: usize,
    /// sha256 (hex) de los bytes del archivo de income leído.
    pub sha256: Option<String>,
    /// Recibo de cobertura (`--recibo-income`): `None` si no se dio, `Err` si
    /// no se pudo leer o no es JSON.
    pub recibo: Option<Result<serde_json::Value, String>>,
}

/// DEMO_TEL T3 (T3-3): income leído con sus defectos. Una fila inválida no
/// desaparece: queda como diagnóstico con su línea y bloquea la acreditación.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IncomeLeido {
    pub filas: Vec<IncomeRow>,
    pub diagnosticos: Vec<DiagnosticoTel>,
    /// Líneas no vacías.
    pub lineas: usize,
}

/// Activo de una fila sin activo válido: se muestra con esa marca, nunca se
/// deduce del sufijo del símbolo.
pub const ACTIVO_AUSENTE: &str = "AUSENTE";
pub const ACTIVO_VACIO: &str = "VACIO";
pub const ACTIVO_INVALIDO: &str = "INVALIDO";

fn numero_json(v: Option<&serde_json::Value>) -> Option<f64> {
    match v? {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) if decimal_finito(s) => s.parse::<f64>().ok(),
        _ => None,
    }
    .filter(|x| x.is_finite())
}

fn entero_json(v: Option<&serde_json::Value>) -> Option<u64> {
    match v? {
        serde_json::Value::Number(n) => n.as_u64(),
        serde_json::Value::String(s) if entero(s) => s.parse::<u64>().ok(),
        _ => None,
    }
}

/// DEMO_TEL T3 (T3-3/T3-4): lector auditado del JSONL de income, para el
/// informe oficial (la tabla LEGACY sigue con [`parse_income_jsonl`]). Cada
/// defecto es un diagnóstico con su línea:
/// - `integridad`: la línea no es un objeto JSON (truncada o corrupta), o le
///   falta `symbol`, `income_type`, un `income` finito o `time_ms`;
/// - `activo`: `asset` ausente, vacío o de tipo inválido (la fila se conserva
///   con la marca [`ACTIVO_AUSENTE`]/[`ACTIVO_VACIO`]/[`ACTIVO_INVALIDO`]);
/// - `ámbito`: la fila declara `venue`/`cuenta` distintos del alcance (la fila
///   es de otra cuenta y no entra).
pub fn parse_income_jsonl_auditado(text: &str, ambito: &Ambito) -> IncomeLeido {
    let mut l = IncomeLeido::default();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        l.lineas += 1;
        let linea = i + 1;
        let diag = |mensaje: String| DiagnosticoTel { fuente: "income".to_string(), linea, mensaje };
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v @ serde_json::Value::Object(_)) => v,
            _ => {
                l.diagnosticos.push(diag("integridad: la línea no es un objeto JSON (truncada o corrupta)".to_string()));
                continue;
            }
        };
        let texto = |k: &str| v.get(k).and_then(|x| x.as_str()).filter(|s| !s.is_empty()).map(str::to_string);
        let (Some(symbol), Some(income_type)) = (texto("symbol"), texto("income_type")) else {
            l.diagnosticos.push(diag("integridad: fila sin symbol o sin income_type".to_string()));
            continue;
        };
        let Some(income) = numero_json(v.get("income")) else {
            l.diagnosticos.push(diag(format!("integridad: {symbol} {income_type}: income ausente o no finito")));
            continue;
        };
        let Some(time_ms) = entero_json(v.get("time_ms")) else {
            l.diagnosticos.push(diag(format!("integridad: {symbol} {income_type}: time_ms ausente o inválido")));
            continue;
        };
        let mut fuera = false;
        for (k, esperado) in [("venue", &ambito.venue), ("cuenta", &ambito.cuenta)] {
            if let Some(x) = v.get(k) {
                if x.as_str() != Some(esperado.as_str()) {
                    l.diagnosticos.push(diag(format!(
                        "ámbito: {symbol} {income_type}: {k}={x} distinto del alcance {esperado}; la fila no entra"
                    )));
                    fuera = true;
                }
            }
        }
        if fuera {
            continue;
        }
        let asset = match v.get("asset") {
            None => ACTIVO_AUSENTE.to_string(),
            Some(serde_json::Value::String(s)) if s.trim().is_empty() => ACTIVO_VACIO.to_string(),
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(_) => ACTIVO_INVALIDO.to_string(),
        };
        if matches!(asset.as_str(), ACTIVO_AUSENTE | ACTIVO_VACIO | ACTIVO_INVALIDO) {
            l.diagnosticos.push(diag(format!(
                "activo: {symbol} {income_type}: asset {asset}; el sufijo del símbolo no acredita la moneda"
            )));
        }
        let trade_id = match v.get("trade_id") {
            Some(serde_json::Value::String(s)) => s.trim().to_string(),
            Some(serde_json::Value::Number(n)) => n.to_string(),
            _ => String::new(),
        };
        let tran_id = entero_json(v.get("tran_id")).unwrap_or(0);
        l.filas.push(IncomeRow { symbol, income_type, income, time_ms, tran_id, trade_id, asset });
    }
    l
}

/// sha256 (hex en minúsculas) de unos bytes. FIPS 180-4, sin dependencias.
pub fn sha256_hex(datos: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] =
        [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut m = datos.to_vec();
    let bits = (datos.len() as u64).wrapping_mul(8);
    m.push(0x80);
    while m.len() % 64 != 56 {
        m.push(0);
    }
    m.extend_from_slice(&bits.to_be_bytes());
    for bloque in m.chunks(64) {
        let mut w = [0u32; 64];
        for t in 0..16 {
            w[t] = u32::from_be_bytes([bloque[4 * t], bloque[4 * t + 1], bloque[4 * t + 2], bloque[4 * t + 3]]);
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16].wrapping_add(s0).wrapping_add(w[t - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for t in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[t]).wrapping_add(w[t]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

/// DEMO_TEL T3 (T3-3): motivos de cobertura del income. T3 no construye el
/// certificador de procedencia (decisión de alcance admitida por los criterios
/// T3-3): un recibo coherente sólo acredita lo que su checksum puede acreditar
/// (que el archivo es el descrito), nunca el origen ni la cobertura completa,
/// así que siempre queda un motivo de cobertura y el informe no es APTO.
fn motivos_de_cobertura(
    ambito: &Ambito,
    ventana: Option<VentanaIncome>,
    ev: &EvidenciaIncome,
) -> Vec<String> {
    let mut m = Vec::new();
    let declarado = |s: &str| !s.is_empty() && s != "N/D";
    if !declarado(&ambito.venue) || !declarado(&ambito.cuenta) {
        m.push("ámbito: venue/cuenta del alcance no declarados (--ambito)".to_string());
    }
    if ev.activo.is_none() {
        m.push("activo: numerario del alcance no declarado (--activo)".to_string());
    }
    match &ev.recibo {
        None => m.push("cobertura: sin recibo de cobertura del income (--recibo-income)".to_string()),
        Some(Err(e)) => m.push(format!("integridad: recibo de cobertura ilegible: {e}")),
        Some(Ok(r)) => {
            let s = |k: &str| r.get(k).and_then(|x| x.as_str());
            match (s("archivo_sha256"), ev.sha256.as_deref()) {
                (Some(a), Some(b)) if a.eq_ignore_ascii_case(b) => {}
                (Some(a), b) => m.push(format!(
                    "integridad: el sha256 del recibo ({a}) no corresponde al archivo de income leído ({})",
                    b.unwrap_or("N/D")
                )),
                (None, _) => m.push("integridad: el recibo no declara archivo_sha256".to_string()),
            }
            for (k, esperado) in [("venue", &ambito.venue), ("cuenta", &ambito.cuenta)] {
                match s(k) {
                    Some(x) if x == esperado => {}
                    Some(x) => m.push(format!("ámbito: el recibo declara {k}={x} y el alcance {esperado}")),
                    None => m.push(format!("ámbito: el recibo no declara {k}")),
                }
            }
            match (s("activo"), ev.activo.as_deref()) {
                (Some(a), Some(b)) if a == b => {}
                (Some(a), b) => m.push(format!("activo: el recibo declara activo={a} y el alcance {}", b.unwrap_or("N/D"))),
                (None, _) => m.push("activo: el recibo no declara activo".to_string()),
            }
            let tipos: Vec<&str> =
                r.get("tipos").and_then(|x| x.as_array()).map(|a| a.iter().filter_map(|t| t.as_str()).collect()).unwrap_or_default();
            for t in ["REALIZED_PNL", "COMMISSION", "FUNDING_FEE"] {
                if !tipos.contains(&t) {
                    m.push(format!("cobertura: el recibo no declara el tipo {t}"));
                }
            }
            let paginas = r.get("paginas").and_then(|x| x.as_array()).cloned().unwrap_or_default();
            let mut filas = 0u64;
            let mut defecto_pag = None;
            if paginas.is_empty() {
                defecto_pag = Some("el recibo no declara páginas".to_string());
            }
            for (i, p) in paginas.iter().enumerate() {
                let n = p.get("pagina").and_then(|x| x.as_u64());
                if n != Some(i as u64 + 1) {
                    defecto_pag.get_or_insert(format!("falta la página {} (secuencia {:?})", i + 1, n));
                }
                filas += p.get("filas").and_then(|x| x.as_u64()).unwrap_or(0);
                let ultima = i + 1 == paginas.len();
                let cont = p.get("continuacion").filter(|c| !c.is_null());
                if ultima && cont.is_some() {
                    defecto_pag.get_or_insert("la última página tiene continuación: no hay término".to_string());
                }
                if !ultima && cont.is_none() {
                    defecto_pag.get_or_insert(format!("la página {} no enlaza la siguiente", i + 1));
                }
            }
            if r.get("termino").and_then(|x| x.as_bool()) != Some(true) {
                defecto_pag.get_or_insert("el recibo no declara término comprobable".to_string());
            }
            if defecto_pag.is_none() && filas != ev.lineas as u64 {
                defecto_pag = Some(format!("las páginas suman {filas} filas y el archivo tiene {}", ev.lineas));
            }
            if let Some(d) = defecto_pag {
                m.push(format!("paginación: {d}"));
            }
            let intervalo = |k: &str| {
                let x = r.get(k)?;
                Some((x.get("desde_ms")?.as_u64()?, x.get("hasta_ms")?.as_u64()?))
            };
            for k in ["intervalo_solicitado", "intervalo_cubierto"] {
                match (intervalo(k), ventana) {
                    (Some((a, b)), Some(v)) if a <= v.desde_ms && b >= v.hasta_ms => {}
                    (Some((a, b)), Some(v)) => m.push(format!(
                        "cobertura: {k} {a}..{b} no cubre la ventana {}..{}: intervalo incompleto",
                        v.desde_ms, v.hasta_ms
                    )),
                    (Some(_), None) => {}
                    (None, _) => m.push(format!("cobertura: el recibo no declara {k}")),
                }
            }
            m.push(
                "cobertura: el recibo no acredita la procedencia de la captura (un checksum no es autenticidad; certificador de income fuera del alcance de T3): funding y neto oficiales N/D"
                    .to_string(),
            );
        }
    }
    m
}

/// Income filtrado por ventana y deduplicado por (símbolo, tipo, tranId, activo).
fn income_acreditado(
    income: &[IncomeRow],
    ventana: Option<VentanaIncome>,
) -> (Vec<IncomeRow>, usize, usize, Vec<String>) {
    let mut out: Vec<IncomeRow> = Vec::new();
    let (mut fuera, mut dup) = (0usize, 0usize);
    let mut conflictos = Vec::new();
    for r in income {
        if let Some(v) = ventana {
            if r.time_ms < v.desde_ms || r.time_ms > v.hasta_ms {
                fuera += 1;
                continue;
            }
        }
        if r.tran_id != 0 {
            if let Some(p) = out.iter().find(|p| {
                p.tran_id == r.tran_id && p.income_type == r.income_type && p.symbol == r.symbol && p.asset == r.asset
            }) {
                if p.income.to_bits() == r.income.to_bits() && p.time_ms == r.time_ms && p.trade_id == r.trade_id {
                    dup += 1;
                } else {
                    conflictos.push(format!(
                        "income {} {} tran_id={} repetido con otro contenido",
                        r.symbol, r.income_type, r.tran_id
                    ));
                }
                continue;
            }
        }
        out.push(r.clone());
    }
    (out, fuera, dup, conflictos)
}

/// M1 T2 (QRA-PARSER-M1, puntos 4 y 5): informe mixto. Las filas y el dinero
/// son los de [`build_report`] sobre el income acreditado (ventana + dedup);
/// el log legacy enriquece como siempre. Sin join legacy, un FILL DEMO_TEL del
/// ámbito acreditado con el MISMO (símbolo, tradeId) aporta su hora propia,
/// precio y cantidad de ESE fill (no el precio de la posición); latencia
/// orden→fill y salida quedan N/D (ORDER_SEND es intento y TRADE_CLOSE es
/// estimado). Ningún evento DEMO_TEL suma dinero. Sin ventana declarada, el
/// funding y el neto quedan N/D (no cero) y el informe no es apto.
pub fn build_report_mixto(
    legacy: &[(String, String, LogOp)],
    lecturas: &[LecturaTel],
    ambito: &Ambito,
    income: &[IncomeRow],
    ventana: Option<VentanaIncome>,
) -> InformeMixto {
    build_report_mixto_con(legacy, lecturas, ambito, income, ventana, &EvidenciaIncome::default())
}

/// DEMO_TEL T3 (T3-2): procedencia de la clave contable de una fila de income.
/// Sólo un tradeId explícito es identidad nativa; el tranId sólo identifica la
/// fila contable y nunca une con un fill ni con el log.
fn origen_y_clave(r: &IncomeRow) -> (&'static str, String) {
    if r.trade_id.trim().is_empty() {
        ("TRAN_ID", r.tran_id.to_string())
    } else {
        ("TRADE_ID", r.trade_id.clone())
    }
}

fn fmt_importe(x: f64) -> String {
    if x.is_finite() { format!("{x}") } else { "N/D".to_string() }
}

/// DEMO_TEL T3: [`build_report_mixto`] con la evidencia del income (numerario,
/// defectos de filas y recibo de cobertura).
///
/// - T3-2: las filas se agrupan por (símbolo, procedencia de la clave, id): un
///   income sin tradeId (omitido o vacío) con `tranId` igual a un tradeId nativo
///   es otra operación, sin join, y el fill sigue sin pareja.
/// - T3-3: funding y neto oficiales N/D mientras no haya cobertura acreditada
///   (T3 no construye el certificador); el informe nunca es APTO sin ella.
/// - T3-4: los importes se conservan por (tipo, activo); un activo fuera del
///   numerario del alcance deja la operación incompleta, y la suma nominal de la
///   tabla LEGACY queda señalada sin autoridad oficial.
pub fn build_report_mixto_con(
    legacy: &[(String, String, LogOp)],
    lecturas: &[LecturaTel],
    ambito: &Ambito,
    income: &[IncomeRow],
    ventana: Option<VentanaIncome>,
    ev: &EvidenciaIncome,
) -> InformeMixto {
    let (income_ok, fuera, dup, conflictos_income) = income_acreditado(income, ventana);
    let unidos = unir_fills(lecturas);
    let mut grupos: BTreeMap<(String, &'static str, String), Vec<IncomeRow>> = BTreeMap::new();
    for r in &income_ok {
        let (origen, clave) = origen_y_clave(r);
        grupos.entry((r.symbol.clone(), origen, clave)).or_default().push(r.clone());
    }
    let mut filas = Vec::new();
    let mut motivos = Vec::new();
    let mut legacy_sin_autoridad = Vec::new();
    for ((symbol, origen, clave), rows) in &grupos {
        let nativa = *origen == "TRADE_ID";
        let log: &[(String, String, LogOp)] = if nativa { legacy } else { &[] };
        let Some(mut op) = build_report(log, rows).into_iter().next() else {
            continue;
        };
        let tiene_legacy = nativa && legacy.iter().any(|(s, t, _)| s == symbol && t == clave);
        let fill = if nativa {
            unidos.fills.get(&(ambito.venue.clone(), ambito.cuenta.clone(), symbol.clone(), clave.clone())).cloned()
        } else {
            None
        };
        let join = if tiene_legacy {
            "LEGACY"
        } else if let Some(f) = &fill {
            op.fill_ts_ms = Some(u128::from(f.ts_evento));
            if op.side.is_empty() {
                op.side = f.side.clone();
            }
            "FILL_NATIVO"
        } else {
            "N/D"
        };
        if join == "N/D" {
            motivos.push(format!("{} trade_id={}: sin join nativo ni legacy", op.symbol, op.trade_id));
            if !nativa {
                motivos.push(format!(
                    "join: {symbol} tran_id={clave}: fila contable sin tradeId explícito; el tranId no es identidad nativa"
                ));
            }
        }
        let mut por_activo: BTreeMap<(String, String), (f64, usize)> = BTreeMap::new();
        for r in rows {
            let e = por_activo.entry((r.income_type.clone(), r.asset.clone())).or_insert((0.0, 0));
            e.0 += r.income;
            e.1 += 1;
        }
        let observado: Vec<ImporteObservado> = por_activo
            .into_iter()
            .map(|((tipo, activo), (importe, n))| ImporteObservado { tipo, activo, importe, filas: n })
            .collect();
        let mut activos: Vec<&str> = observado.iter().map(|o| o.activo.as_str()).collect();
        activos.sort_unstable();
        activos.dedup();
        let ajenos: Vec<&ImporteObservado> = observado
            .iter()
            .filter(|o| ev.activo.as_deref().map_or(activos.len() > 1, |a| o.activo != a))
            .collect();
        let incompleta = (!ajenos.is_empty()).then(|| {
            let cargos: Vec<String> =
                ajenos.iter().map(|o| format!("{} {} {}", o.tipo, fmt_importe(o.importe), o.activo)).collect();
            format!(
                "activo: {} trade_id={}: importes fuera del numerario {} ({}); operación incompleta, neto N/D",
                symbol,
                clave,
                ev.activo.as_deref().unwrap_or("N/D"),
                cargos.join(", ")
            )
        });
        if let Some(m) = &incompleta {
            motivos.push(m.clone());
        }
        if activos.len() > 1 {
            legacy_sin_autoridad.push(format!(
                "{symbol} {clave}: la tabla LEGACY suma nominalmente {}; sin autoridad oficial",
                activos.join(" + ")
            ));
        }
        filas.push(FilaMixta {
            fill: if join == "FILL_NATIVO" { fill } else { None },
            op,
            join,
            funding_acreditado: None,
            neto_acreditado: None,
            origen_clave: origen,
            observado,
            incompleta,
        });
    }
    let fills_sin_pareja: Vec<ClaveFill> = unidos
        .fills
        .keys()
        .filter(|(v, c, s, t)| {
            *v == ambito.venue
                && *c == ambito.cuenta
                && !income_ok.iter().any(|r| r.symbol == *s && origen_y_clave(r) == ("TRADE_ID", t.clone()))
        })
        .cloned()
        .collect();
    let fills_fuera = unidos.fills.keys().filter(|(v, c, _, _)| *v != ambito.venue || *c != ambito.cuenta).count();
    let mut diagnosticos: Vec<DiagnosticoTel> = lecturas.iter().flat_map(|l| l.diagnosticos.clone()).collect();
    diagnosticos.extend(unidos.conflictos.clone());
    let tipos_desconocidos: usize = lecturas.iter().map(|l| l.tipos_desconocidos.len()).sum();
    if ventana.is_none() {
        motivos.push("income sin ventana de cobertura declarada: funding y neto N/D".to_string());
    }
    if filas.is_empty() {
        motivos.push("sin filas de income en la ventana".to_string());
    }
    if !diagnosticos.is_empty() {
        motivos.push(format!("{} diagnósticos DEMO_TEL/FILL", diagnosticos.len()));
    }
    if tipos_desconocidos > 0 {
        motivos.push(format!("{tipos_desconocidos} eventos DEMO_TEL de tipo desconocido"));
    }
    if !fills_sin_pareja.is_empty() {
        motivos.push(format!("{} fills sin pareja en el income", fills_sin_pareja.len()));
    }
    motivos.extend(conflictos_income);
    // T3-3: las filas inválidas o contradictorias del income bloquean la acreditación.
    for d in &ev.diagnosticos {
        motivos.push(format!("income línea {}: {}", d.linea, d.mensaje));
    }
    motivos.extend(motivos_de_cobertura(ambito, ventana, ev));
    InformeMixto {
        activo: ev.activo.clone(),
        income_diagnosticos: ev.diagnosticos.clone(),
        legacy_sin_autoridad,
        apto: motivos.is_empty(),
        filas,
        ambito: ambito.clone(),
        ventana,
        fuentes: lecturas.iter().map(|l| (l.fuente.clone(), l.lineas, l.eventos.len())).collect(),
        diagnosticos,
        tipos_desconocidos,
        fills_humanos: lecturas.iter().map(|l| l.fills_humanos.len()).sum(),
        fills_nativos: unidos.fills.len(),
        fills_repetidos: unidos.repetidos,
        fills_fuera_de_ambito: fills_fuera,
        fills_sin_pareja,
        income_fuera_de_ventana: fuera,
        income_duplicado: dup,
        motivos_no_apto: motivos,
    }
}

fn fmt_opt_money(v: Option<f64>) -> String {
    v.map(fmt_money).unwrap_or_else(|| "N/D".to_string())
}

/// Sección de cobertura M1 (va DESPUÉS de la tabla legacy, que no cambia).
pub fn render_informe_mixto(i: &InformeMixto) -> String {
    let mut s = String::from("COBERTURA M1 (DEMO_TEL)\n");
    s.push_str(&format!("ambito: venue={} cuenta={}\n", i.ambito.venue, i.ambito.cuenta));
    s.push_str(&format!("activo del alcance: {}\n", i.activo.as_deref().unwrap_or("N/D")));
    match i.ventana {
        Some(v) => s.push_str(&format!("ventana income: {}..{} ms\n", v.desde_ms, v.hasta_ms)),
        None => s.push_str("ventana income: N/D\n"),
    }
    for (f, lineas, eventos) in &i.fuentes {
        s.push_str(&format!("fuente {f}: {lineas} lineas, {eventos} eventos DEMO_TEL\n"));
    }
    s.push_str(&format!(
        "fills nativos {} (repetidos {}, fuera de ambito {}, sin pareja {}), fills humanos {}\n",
        i.fills_nativos, i.fills_repetidos, i.fills_fuera_de_ambito, i.fills_sin_pareja.len(), i.fills_humanos
    ));
    s.push_str(&format!(
        "income fuera de ventana {}, duplicado {}, diagnosticos {}, tipos desconocidos {}\n",
        i.income_fuera_de_ventana,
        i.income_duplicado,
        i.diagnosticos.len(),
        i.tipos_desconocidos
    ));
    for d in &i.diagnosticos {
        s.push_str(&format!("  diagnostico {}:{}: {}\n", d.fuente, d.linea, d.mensaje));
    }
    s.push_str(&format!("income: diagnosticos de filas {}\n", i.income_diagnosticos.len()));
    for d in &i.income_diagnosticos {
        s.push_str(&format!("  diagnostico {}:{}: {}\n", d.fuente, d.linea, d.mensaje));
    }
    s.push_str(
        "símbolo | trade | join | fill_ms | fill_precio | fill_qty | ord→fill_ms | funding | neto | origen_clave | observado\n",
    );
    for f in &i.filas {
        let (fp, fq) = match &f.fill {
            Some(x) => (fmt_opt(Some(x.price)), fmt_opt(Some(x.qty))),
            None => ("N/D".to_string(), "N/D".to_string()),
        };
        let observado: Vec<String> =
            f.observado.iter().map(|o| format!("{}={} {}", o.tipo, fmt_importe(o.importe), o.activo)).collect();
        s.push_str(&format!(
            "{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}{}\n",
            f.op.symbol,
            f.op.trade_id,
            f.join,
            fmt_ms(f.op.fill_ts_ms),
            fp,
            fq,
            fmt_ms(f.op.order_to_fill_ms),
            fmt_opt_money(f.funding_acreditado),
            fmt_opt_money(f.neto_acreditado),
            f.origen_clave,
            observado.join("; "),
            if f.incompleta.is_some() { " (INCOMPLETA)" } else { "" },
        ));
    }
    for l in &i.legacy_sin_autoridad {
        s.push_str(&format!("LEGACY sin autoridad oficial: {l}\n"));
    }
    if i.apto {
        s.push_str("ACEPTACION: APTO\n");
    } else {
        s.push_str("ACEPTACION: NO APTO\n");
        for m in &i.motivos_no_apto {
            s.push_str(&format!("  motivo: {m}\n"));
        }
    }
    s
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
    // M1 T2: `--log` puede repetirse (stdout y segmentos del logger con
    // TG_LOG_FILE: cada archivo es una fuente con su recibo de cobertura).
    let mut log_paths: Vec<String> = Vec::new();
    let mut income_path: Option<String> = None;
    let mut funnel_path: Option<String> = None;
    let mut ventana: Option<VentanaIncome> = None;
    let mut ambito = Ambito::new("N/D", "N/D");
    let mut activo: Option<String> = None;
    let mut recibo_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--log" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    log_paths.push(p.clone());
                }
            }
            "--ventana" => {
                i += 1;
                let v = args.get(i).and_then(|v| v.split_once(':')).and_then(|(a, b)| {
                    Some(VentanaIncome { desde_ms: a.parse().ok()?, hasta_ms: b.parse().ok()? })
                });
                match v {
                    Some(v) if v.desde_ms <= v.hasta_ms => ventana = Some(v),
                    _ => {
                        eprintln!("--ventana <desde_ms>:<hasta_ms> inválida");
                        std::process::exit(2);
                    }
                }
            }
            "--ambito" => {
                i += 1;
                match args.get(i).and_then(|v| v.split_once(':')) {
                    Some((v, c)) if !v.is_empty() && !c.is_empty() => ambito = Ambito::new(v, c),
                    _ => {
                        eprintln!("--ambito <venue>:<cuenta> inválido");
                        std::process::exit(2);
                    }
                }
            }
            "--income" => {
                i += 1;
                income_path = args.get(i).cloned();
            }
            // DEMO_TEL T3 (T3-3): numerario del alcance y recibo de cobertura del income.
            "--activo" => {
                i += 1;
                match args.get(i) {
                    Some(a) if !a.trim().is_empty() => activo = Some(a.trim().to_string()),
                    _ => {
                        eprintln!("--activo <ASSET> inválido");
                        std::process::exit(2);
                    }
                }
            }
            "--recibo-income" => {
                i += 1;
                recibo_path = args.get(i).cloned();
            }
            "--funnel" => {
                i += 1;
                funnel_path = args.get(i).cloned();
            }
            _ => {}
        }
        i += 1;
    }
    let income_path = match (log_paths.is_empty(), income_path) {
        (false, Some(b)) => b,
        _ => {
            eprintln!(
                "uso: demo_report --log <run.log> [--log <otro segmento>] --income <income.jsonl> [--ventana <desde_ms>:<hasta_ms>] [--ambito <venue>:<cuenta>] [--activo <ASSET>] [--recibo-income <recibo.json>] [--funnel <gate.json>]"
            );
            std::process::exit(2);
        }
    };
    let mut log_ops = Vec::new();
    let mut lecturas = Vec::new();
    for log_path in &log_paths {
        let log_text = match std::fs::read_to_string(log_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("no lee {}: {}", log_path, e);
                std::process::exit(2);
            }
        };
        match parse_run_log(&log_text) {
            Ok(o) => log_ops.extend(o),
            Err(e) => {
                eprintln!("LOG DEMO RECHAZADO ({}): línea {}: {}", log_path, e.line, e.message);
                std::process::exit(2);
            }
        }
        lecturas.push(parse_demo_tel(&log_text, log_path, &ambito));
    }
    // DEMO_TEL T3 (T3-3): los bytes exactos que se consumen (para el sha256 del recibo).
    let income_bytes = match std::fs::read(&income_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("no lee {}: {}", income_path, e);
            std::process::exit(2);
        }
    };
    let income_text = String::from_utf8_lossy(&income_bytes).into_owned();
    let income = parse_income_jsonl(&income_text);
    let ops = build_report(&log_ops, &income);
    println!("{}", render_report_table(&ops));
    // M1 T2: cobertura y estado de aceptación, después de la tabla legacy.
    // T3: el informe oficial lee el income con el lector auditado.
    let leido = parse_income_jsonl_auditado(&income_text, &ambito);
    let recibo = recibo_path.map(|p| {
        std::fs::read_to_string(&p)
            .map_err(|e| format!("no lee {p}: {e}"))
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).map_err(|e| format!("{p}: {e}")))
    });
    let ev = EvidenciaIncome {
        activo,
        diagnosticos: leido.diagnosticos,
        lineas: leido.lineas,
        sha256: Some(sha256_hex(&income_bytes)),
        recibo,
    };
    let mixto = build_report_mixto_con(&log_ops, &lecturas, &ambito, &leido.filas, ventana, &ev);
    println!("{}", render_informe_mixto(&mixto));

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
