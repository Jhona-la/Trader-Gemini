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

    #[test]
    fn b3_7_cola_drena_y_vacia() {
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
