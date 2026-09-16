//! B3.7 — CONTABILIDAD DE CIERRES POR BRACKET (auditoría de disparos, roadmap).
//!
//! POR QUÉ: los cierres TP/SL disparados en el EXCHANGE llegaban como fill
//! + log 💧 y nada más — sin PnL atribuido, sin fees, sin slippage contra el
//! trigger, y SIN alimentar el posterior de Kelly (el risk_envelope sólo
//! aprendía de los cierres del core, la minoría: 10/10 trades del
//! 2026-09-15 cerraron por bracket). Esta pieza cierra el lazo:
//!
//! CÓMO: `on_order_trade_update` detecta fills de piernas de bracket
//! (STOP_MARKET / TAKE_PROFIT_MARKET / TRAILING_STOP_MARKET — siempre
//! cierres en este sistema), junta el contexto de entrada del arena local
//! (precio/fee de entrada) y produce un `BracketClose` que va a DOS sitios:
//!   1. data/trade_fills.jsonl — diario persistente inmediato (un renglón
//!      por disparo: trigger, fill, slippage en bps adversos, PnL bruto,
//!      fees, neto) — la estadística de disparos N>1 que faltaba.
//!   2. Cola en memoria que god_engine drena cada tick para alimentar
//!      risk_envelope.record_trade — Kelly aprende de TODOS los cierres.
//!
//! HONESTIDAD: si el arena local no tiene la posición (adoptada tras
//! reconexión), entry=0 y pnl_gross=0 — el registro queda como evidencia
//! del disparo (slippage/fees reales) sin inventar PnL; la contabilidad
//! neta del símbolo vive en el exchange (income, ya usada por el
//! fee-breaker B3.6).

use std::sync::LazyLock;
use std::sync::Mutex;

/// Un cierre por pierna de bracket, con todo el contexto disponible.
#[derive(Debug, Clone)]
pub struct BracketClose {
    pub ts_ms: u64,
    pub symbol: String,
    /// La posición que se cerró era long (el fill es SELL).
    pub was_long: bool,
    pub qty: f64,
    /// Precio de entrada conocido por el arena local (0 = desconocido).
    pub entry_price: f64,
    /// Fill real de salida.
    pub exit_price: f64,
    /// Precio trigger de la pierna (0 si el evento no lo trajo).
    pub stop_price: f64,
    /// PnL bruto del fill: (entry-exit)·qty·sign — 0 si entry desconocido.
    pub pnl_gross: f64,
    /// Comisión del fill de salida + entrada pro-rata (positivo).
    pub fees: f64,
    /// "TP" | "SL" | "TRAIL"
    pub trigger: &'static str,
    /// Slippage del disparo en bps ADVERSOS (positivo = peor que el trigger).
    pub slippage_bps: f64,
}

static PENDING: LazyLock<Mutex<Vec<BracketClose>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

/// Serializa los tests que tocan la cola global PENDING (cargo test corre
/// los módulos en paralelo dentro del mismo binario).
#[cfg(test)]
pub(crate) static TEST_QUEUE_LOCK: Mutex<()> = Mutex::new(());

/// ¿Es una pierna de cierre? En este sistema las entradas son
/// MARKET/LIMIT/GTX/ICEBERG; STOP_MARKET, TAKE_PROFIT_MARKET y
/// TRAILING_STOP_MARKET sólo existen como brackets de salida —
/// convención-independiente (los clientIds han cambiado de formato
/// `*_TP` a `wdTP_*` entre bloques).
pub fn is_closing_bracket_order(order_type: &str) -> bool {
    order_type.contains("STOP_MARKET") || order_type.contains("TAKE_PROFIT_MARKET")
}

/// Clasifica el trigger por tipo de orden.
pub fn trigger_kind(order_type: &str) -> &'static str {
    if order_type.contains("TAKE_PROFIT") {
        "TP"
    } else if order_type.contains("TRAILING") {
        "TRAIL"
    } else {
        "SL"
    }
}

/// B3.7 (repro demo_v26) — clasificación RESPALDO por clientOrderId.
///
/// El servicio de Algo (migración OCO-F5) puede convertir la pierna
/// condicional disparada en una orden MARKET normal: en ese caso el campo
/// `o` del ORDER_TRADE_UPDATE ya no identifica el cierre y la detección
/// por tipo lo pierde. Las piernas de este sistema llevan ids FIRMADOS:
///   · watchdog top-up:  "wdTP_<micros>_<intent>" / "wdSL_<micros>_<intent>"
///   · OCO de entrada:   "<base>_TP" / "<base>_TPR" / "<base>_SL" / "<base>_SLR"
/// Los ids de ENTRADA ("cL_", "cS_", "mc_", "iceberg_") y los reduce-only
/// de emergencia (UUID simple, sin guiones bajos) NO matchean — sus cierres
/// los contabiliza el core (o el despacho X-008), no este diario.
pub fn bracket_close_kind_by_client_id(client_order_id: &str) -> Option<&'static str> {
    if client_order_id.starts_with("wdTP_") {
        Some("TP")
    } else if client_order_id.starts_with("wdSL_") {
        Some("SL")
    } else if client_order_id.ends_with("_TPR") || client_order_id.ends_with("_TP") {
        Some("TP")
    } else if client_order_id.ends_with("_SLR") || client_order_id.ends_with("_SL") {
        Some("SL")
    } else {
        None
    }
}

/// ¿Este fill cierra posición vía pierna de bracket? Por tipo de orden
/// (canónico) o por clientOrderId firmado (respaldo ante conversión MARKET
/// del servicio Algo).
pub fn is_bracket_close_fill(order_type: &str, client_order_id: &str) -> bool {
    is_closing_bracket_order(order_type)
        || bracket_close_kind_by_client_id(client_order_id).is_some()
}

/// Registra un cierre: append al diario persistente + cola para Kelly.
pub fn record_bracket_close(rec: BracketClose) {
    // Diario primero: la evidencia sobrevive aunque el motor caiga.
    let net = rec.pnl_gross - rec.fees;
    let line = format!(
        "{{\"ts\":{},\"sym\":\"{}\",\"trig\":\"{}\",\"long\":{},\"qty\":{:.8},\"entry\":{:.6},\"exit\":{:.6},\"stop\":{:.6},\"pnl_gross\":{:.6},\"fees\":{:.6},\"net\":{:.6},\"slip_bps\":{:.2}}}\n",
        rec.ts_ms,
        rec.symbol,
        rec.trigger,
        rec.was_long,
        rec.qty,
        rec.entry_price,
        rec.exit_price,
        rec.stop_price,
        rec.pnl_gross,
        rec.fees,
        net,
        rec.slippage_bps
    );
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("data/trade_fills.jsonl")
    {
        use std::io::Write;
        let _ = f.write_all(line.as_bytes());
    }
    if let Ok(mut q) = PENDING.lock() {
        if q.len() < 1024 {
            q.push(rec);
        }
    }
}

/// Drena la cola de cierres pendientes (el host la consume cada tick).
pub fn drain_bracket_closes() -> Vec<BracketClose> {
    PENDING
        .lock()
        .map(|mut q| std::mem::take(&mut *q))
        .unwrap_or_default()
}

/// B3.10 — FILL DE ENTRADA al diario (evidencia, sin cola): con el flag
/// maker se mide la selección adversa de la ruta maker (¿los fills maker
/// entran en peor precio relativo que los taker?) una vez que existan
/// ambas poblaciones. `long` = dirección de la posición ABIERTA.
/// B3.21 — FALLBACK DE CONTEXTO para cierres de bracket: la reconciliación
/// (ciclo 60s) puede consumir la posición local — y poner entry_price=0 —
/// ANTES de que el fill del bracket llegue al stream. El diario de entradas
/// (position_journal.jsonl, B3.1) existe precisamente para cargar ese
/// contexto: devuelve el ÚLTIMO px de entrada registrado para símbolo+lado.
pub fn last_journal_entry_px(symbol: &str, was_long: bool) -> Option<f64> {
    let content = std::fs::read_to_string("data/position_journal.jsonl").ok()?;
    let mut best: Option<(u64, f64)> = None;
    for line in content.lines().rev() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let sym = v.get("sym").and_then(|x| x.as_str()).unwrap_or("");
        let long = v.get("long").and_then(|x| x.as_bool()).unwrap_or(false);
        if sym != symbol || long != was_long {
            continue;
        }
        let px = v.get("px").and_then(|x| x.as_f64()).unwrap_or(0.0);
        let ts = v.get("ts").and_then(|x| x.as_u64()).unwrap_or(0);
        if px > 0.0 && best.map(|(b, _)| ts >= b).unwrap_or(true) {
            best = Some((ts, px));
        }
    }
    best.map(|(_, px)| px)
}

pub fn record_entry_fill(
    ts_ms: u64,
    symbol: &str,
    long: bool,
    qty: f64,
    price: f64,
    is_maker: bool,
    commission: f64,
) {
    let line = format!(
        "{{\"kind\":\"ENTRY\",\"ts\":{ts_ms},\"sym\":\"{symbol}\",\"long\":{long},\"qty\":{qty:.8},\"px\":{price:.6},\"maker\":{is_maker},\"fee\":{commission:.6}}}\n"
    );
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("data/trade_fills.jsonl")
    {
        use std::io::Write;
        let _ = f.write_all(line.as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b3_7_clasifica_triggers_y_piernas() {
        assert!(is_closing_bracket_order("STOP_MARKET"));
        assert!(is_closing_bracket_order("TAKE_PROFIT_MARKET"));
        assert!(is_closing_bracket_order("TRAILING_STOP_MARKET"));
        assert!(!is_closing_bracket_order("MARKET"));
        assert!(!is_closing_bracket_order("LIMIT"));
        assert_eq!(trigger_kind("TAKE_PROFIT_MARKET"), "TP");
        assert_eq!(trigger_kind("TRAILING_STOP_MARKET"), "TRAIL");
        assert_eq!(trigger_kind("STOP_MARKET"), "SL");
    }

    /// B3.7 repro demo_v26: la detección por tipo NO es suficiente si el
    /// servicio Algo convierte la pierna disparada en MARKET — el respaldo
    /// por clientOrderId firmado debe atraparla. Y los cierres de emergencia
    /// (UUID simple) / entradas ("mc_", "cL_", "iceberg_") NO son bracket.
    #[test]
    fn b3_7_repro_respaldo_por_client_id() {
        // El fill real del repro: MARKET reduce-only con UUID simple — NO es
        // bracket (lo cuenta el core vía su despacho X-008).
        let repro_uuid = "01a0a929a0af73c28a7822088668d314";
        assert!(!is_bracket_close_fill("MARKET", repro_uuid));
        assert!(bracket_close_kind_by_client_id(repro_uuid).is_none());
        // Entradas: nunca bracket.
        assert!(!is_bracket_close_fill("LIMIT", "mc_1789544472577123"));
        assert!(!is_bracket_close_fill("MARKET", "cL_0123456789abcdef0123456789abcdef"));
        assert!(!is_bracket_close_fill("LIMIT", "iceberg_01"));
        // Pierna disparada convertida a MARKET: el id firmado la rescata.
        assert_eq!(
            bracket_close_kind_by_client_id("wdTP_1789544472_0"),
            Some("TP")
        );
        assert_eq!(
            bracket_close_kind_by_client_id("wdSL_1789544472_1"),
            Some("SL")
        );
        assert_eq!(
            bracket_close_kind_by_client_id("CONT_oco_1789544472577_1_TP"),
            Some("TP")
        );
        assert_eq!(
            bracket_close_kind_by_client_id("CONT_oco_1789544472577_2_SLR"),
            Some("SL")
        );
        assert!(is_bracket_close_fill("MARKET", "CONT_oco_1_1_TPR"));
        // Canónico por tipo sigue mandando cuando está disponible.
        assert!(is_bracket_close_fill("TAKE_PROFIT_MARKET", "otro_id"));
    }

    #[test]
    fn b3_7_cola_drena_y_vacia() {
        let _guard = TEST_QUEUE_LOCK.lock();
        record_bracket_close(BracketClose {
            ts_ms: 1,
            symbol: "TESTUSDT".into(),
            was_long: true,
            qty: 1.0,
            entry_price: 100.0,
            exit_price: 101.0,
            stop_price: 101.0,
            pnl_gross: 1.0,
            fees: 0.1,
            trigger: "TP",
            slippage_bps: 0.0,
        });
        let drained = drain_bracket_closes();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].symbol, "TESTUSDT");
        assert!(drain_bracket_closes().is_empty());
    }
}
