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

/// D-701 (DÉCIMA OLA · auditoría integral) — PnL BRUTO DE UN CIERRE, FUENTE ÚNICA.
///
/// Un largo gana cuando sale POR ENCIMA de su entrada y un corto cuando sale por
/// debajo. La contabilidad de brackets lo calculaba como
/// `(entrada − salida)·qty·signo` con `signo = +1` para el largo: el signo
/// quedaba invertido en las DOS direcciones, de modo que cada TP se apuntaba
/// como pérdida y cada SL como ganancia. Estaba latente sólo porque el contexto
/// de entrada nunca llegaba (`entry_price == 0`); en cuanto se cablea, Kelly y
/// el win-rate aprenden justo al revés.
///
/// `reconciliation.rs` ya tenía la fórmula correcta: aquí vive una sola vez y la
/// consumen ambos caminos. Devuelve 0 si falta el contexto de entrada (posición
/// adoptada): PnL desconocido no es PnL cero, y quien lo consuma debe mirar
/// `entry_price > 0` para distinguirlo.
#[inline]
pub fn gross_pnl(was_long: bool, entry_price: f64, exit_price: f64, qty: f64) -> f64 {
    if !(entry_price > 0.0 && exit_price > 0.0 && qty > 0.0) {
        return 0.0;
    }
    if was_long {
        (exit_price - entry_price) * qty
    } else {
        (entry_price - exit_price) * qty
    }
}

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
        } else {
            // D-710 (DÉCIMA OLA · auditoría integral): el descarte era SILENCIOSO.
            // Cada cierre perdido es una operación que no alimenta el posterior
            // de Kelly ni el win-rate, y la muestra queda censurada sin que nadie
            // pueda saberlo — justo el defecto que esta contabilidad existe para
            // cerrar. Se cuenta y se informa.
            let n = DESCARTADOS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n == 1 || n % 50 == 0 {
                println!(
                    "🚨 [CONTABILIDAD] Cola de cierres llena (1024): {} cierres DESCARTADOS — la estadística de Kelly y el win-rate quedan censurados hasta que se drene",
                    n
                );
            }
        }
    }
}

/// D-710: cierres perdidos por cola llena. Cualquier valor > 0 invalida la
/// muestra con la que aprende Kelly.
pub static DESCARTADOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// D-710: cuántos cierres se han descartado por cola llena desde el arranque.
pub fn cierres_descartados() -> u64 {
    DESCARTADOS.load(std::sync::atomic::Ordering::Relaxed)
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

    #[test]
    fn d701_el_largo_gana_subiendo_y_el_corto_bajando() {
        // Largo 100 → 110 con 1 unidad: +10. La fórmula anterior daba −10.
        assert!((gross_pnl(true, 100.0, 110.0, 1.0) - 10.0).abs() < 1e-9);
        // Largo 100 → 90: −10.
        assert!((gross_pnl(true, 100.0, 90.0, 1.0) + 10.0).abs() < 1e-9);
        // Corto 100 → 90: +10.
        assert!((gross_pnl(false, 100.0, 90.0, 1.0) - 10.0).abs() < 1e-9);
        // Corto 100 → 110: −10.
        assert!((gross_pnl(false, 100.0, 110.0, 1.0) + 10.0).abs() < 1e-9);
    }

    #[test]
    fn d701_sin_contexto_de_entrada_no_se_inventa_pnl() {
        assert_eq!(gross_pnl(true, 0.0, 110.0, 1.0), 0.0);
        assert_eq!(gross_pnl(true, 100.0, 0.0, 1.0), 0.0);
        assert_eq!(gross_pnl(true, 100.0, 110.0, 0.0), 0.0);
        assert_eq!(gross_pnl(false, f64::NAN, 110.0, 1.0), 0.0);
    }

    #[test]
    fn d701_es_antisimetrico_entre_direcciones() {
        for (e, x, q) in [(100.0, 103.5, 0.25), (58_000.0, 57_100.0, 0.003)] {
            let largo = gross_pnl(true, e, x, q);
            let corto = gross_pnl(false, e, x, q);
            assert!((largo + corto).abs() < 1e-9, "largo {largo} corto {corto}");
        }
    }


    #[test]
    fn d710_la_cola_llena_no_descarta_en_silencio() {
        let antes = cierres_descartados();
        for i in 0..1100u64 {
            record_bracket_close(BracketClose {
                ts_ms: i,
                symbol: "FULLUSDT".into(),
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
        }
        let descartados = cierres_descartados() - antes;
        let drenados = drain_bracket_closes().len() as u64;
        assert_eq!(drenados + descartados, 1100, "ni se pierden ni se inventan cierres");
        assert!(descartados > 0, "con 1100 cierres y cola de 1024 tiene que haber descartes contados");
    }

}
