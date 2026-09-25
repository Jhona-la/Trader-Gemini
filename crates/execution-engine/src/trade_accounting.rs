//! B3.7 — CONTABILIDAD DE CIERRES POR BRACKET (auditoría de disparos, roadmap).
//!
//! POR QUÉ: los cierres TP/SL disparados en el EXCHANGE llegaban como fill
//! + log 💧 y nada más — sin PnL atribuido, sin fees, sin slippage contra el
//! trigger, y SIN alimentar el posterior de Kelly (el risk_envelope sólo
//! aprendía de los cierres del core, la minoría: 10/10 trades del
//! 2026-09-15 cerraron por bracket). Esta pieza cierra el lazo:
//!
//! CÓMO: `on_order_trade_update` reconoce candidatos por tipo o nombre de
//! bracket y junta contexto local de entrada. Esa clasificación no prueba
//! propiedad ni terminalidad. `BracketClose` se entrega a DOS caminos:
//!   1. data/trade_fills.jsonl — serialización validada y escritura asíncrona
//!      sin ACK durable; un registro puede representar un fill parcial.
//!   2. Cola acotada que god_engine drena antes de validar la aritmética y
//!      actualizar estadísticas. Saturación, contexto ausente o invalidez
//!      pueden excluir evidencia; no se garantiza aprender de todos los cierres.
//!
//! HONESTIDAD: si el arena local no tiene la posición (adoptada tras
//! reconexión), entry=0 y pnl_gross=0 — el registro queda como evidencia
//! del disparo (slippage/fees reales) sin inventar PnL; la contabilidad
//! neta del símbolo vive en el exchange (income, ya usada por el
//! fee-breaker B3.6).

use std::sync::LazyLock;
use std::sync::Mutex;

/// Ruta del diario persistente de fills/entradas.
const FILLS_JOURNAL_PATH: &str = "data/trade_fills.jsonl";
/// Ruta del diario de contexto de posiciones (B3.1).
const POSITION_JOURNAL_PATH: &str = "data/position_journal.jsonl";

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
    /// Precio reportado de salida; la emergencia legacy puede usar una estimación.
    pub exit_price: f64,
    /// Precio trigger de la pierna (0 si el evento no lo trajo).
    pub stop_price: f64,
    /// Gross PnL: (exit-entry)*qty for long, opposite for short.
    /// Zero may still mean unknown in legacy producers; use numeric validation.
    pub pnl_gross: f64,
    /// Signed costs in one common currency: positive expense, negative rebate.
    /// The record does not yet carry currency or conversion provenance.
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
/// consumen ambos caminos. El wrapper legacy devuelve 0 ante contexto inválido
/// o aritmética no representable: no distingue esos casos de un cierre plano.
/// Los consumidores de aprendizaje deben usar la API checked y conservar el error.
#[inline]
pub fn gross_pnl(was_long: bool, entry_price: f64, exit_price: f64, qty: f64) -> f64 {
    // Compatibility only: zero is not proof of a breakeven outcome.
    checked_gross_pnl(was_long, entry_price, exit_price, qty).unwrap_or(0.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccountingError {
    InvalidSymbol,
    InvalidQuantity,
    InvalidEntryPrice,
    InvalidExitPrice,
    InvalidStopPrice,
    InvalidGrossPnl,
    InvalidFees,
    InvalidSlippage,
    NonFiniteCalculation,
    Underflow,
    Serialization,
}

/// Arithmetic validation only. Does not attest a fill, its owner or its currency.
pub fn checked_gross_pnl(was_long: bool, entry_price: f64, exit_price: f64, qty: f64)
    -> Result<f64, AccountingError> {
    if !entry_price.is_finite() || entry_price <= 0.0 { return Err(AccountingError::InvalidEntryPrice); }
    if !exit_price.is_finite() || exit_price <= 0.0 { return Err(AccountingError::InvalidExitPrice); }
    if !qty.is_finite() || qty <= 0.0 { return Err(AccountingError::InvalidQuantity); }
    let result = if was_long {
        (exit_price - entry_price) * qty
    } else {
        (entry_price - exit_price) * qty
    };
    if !result.is_finite() { return Err(AccountingError::NonFiniteCalculation); }
    if result == 0.0 && entry_price != exit_price { return Err(AccountingError::Underflow); }
    Ok(result)
}

impl BracketClose {
    /// Reject numeric corruption before statistics/learning. Missing entry (0)
    /// is not breakeven. This does NOT establish provenance or deduplication.
    pub fn checked_numeric_net_pnl(&self) -> Result<f64, AccountingError> {
        self.validate_journal_fields()?;
        checked_gross_pnl(self.was_long, self.entry_price, self.exit_price, self.qty)?;
        let net = self.pnl_gross - self.fees;
        if net.is_finite() { Ok(net) } else { Err(AccountingError::NonFiniteCalculation) }
    }

    fn validate_journal_fields(&self) -> Result<(), AccountingError> {
        if self.symbol.trim().is_empty() { return Err(AccountingError::InvalidSymbol); }
        if !self.qty.is_finite() || self.qty <= 0.0 { return Err(AccountingError::InvalidQuantity); }
        // Zero entry retains diagnostic records from legacy unknown-context producers.
        if !self.entry_price.is_finite() || self.entry_price < 0.0 { return Err(AccountingError::InvalidEntryPrice); }
        if !self.exit_price.is_finite() || self.exit_price <= 0.0 { return Err(AccountingError::InvalidExitPrice); }
        if !self.stop_price.is_finite() || self.stop_price < 0.0 { return Err(AccountingError::InvalidStopPrice); }
        if !self.pnl_gross.is_finite() { return Err(AccountingError::InvalidGrossPnl); }
        if !self.fees.is_finite() { return Err(AccountingError::InvalidFees); }
        if !self.slippage_bps.is_finite() { return Err(AccountingError::InvalidSlippage); }
        if !(self.pnl_gross - self.fees).is_finite() { return Err(AccountingError::NonFiniteCalculation); }
        Ok(())
    }

    /// Pure serializer, preserving the legacy keys without fixed decimal truncation.
    /// Serialization is not durable persistence; zero entry is diagnostic-only.
    pub fn journal_line(&self) -> Result<String, AccountingError> {
        self.validate_journal_fields()?;
        let value = serde_json::json!({
            "ts": self.ts_ms, "sym": self.symbol, "trig": self.trigger,
            "long": self.was_long, "qty": self.qty, "entry": self.entry_price,
            "exit": self.exit_price, "stop": self.stop_price,
            "pnl_gross": self.pnl_gross, "fees": self.fees,
            "net": self.pnl_gross - self.fees, "slip_bps": self.slippage_bps
        });
        serde_json::to_string(&value).map(|s| s + "\n").map_err(|_| AccountingError::Serialization)
    }
}
/// Serializa los tests que tocan la cola global PENDING (cargo test corre
/// los módulos en paralelo dentro del mismo binario).
#[cfg(test)]
pub(crate) static TEST_QUEUE_LOCK: Mutex<()> = Mutex::new(());

/// Reconoce los tres tipos canónicos usados como candidatos a bracket local.
/// El tipo exacto evita coincidencias por substring, pero no demuestra por sí
/// solo reduce-only, propiedad de la posición ni identidad del fill.
pub fn is_closing_bracket_order(order_type: &str) -> bool {
    matches!(order_type, "STOP_MARKET" | "TAKE_PROFIT_MARKET" | "TRAILING_STOP_MARKET")
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
/// por tipo lo pierde. Las piernas usan una convención de nombres, NO una firma:
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
/// (canónico) o por convención de clientOrderId (respaldo ante conversión MARKET
/// del servicio Algo). Es reconocimiento sintáctico, no autenticación del cierre.
pub fn is_bracket_close_fill(order_type: &str, client_order_id: &str) -> bool {
    is_closing_bracket_order(order_type)
        || bracket_close_kind_by_client_id(client_order_id).is_some()
}

/// MOD1/4-012 (INFORME-14): escritura de diario en hilo de FONDO.
///
/// Los handlers del user-data stream corrían `std::fs` write DENTRO del hilo
/// del WebSocket privado — cada disparo de TP/SL y cada fill de entrada
/// bloqueaba el runtime que debe procesar fills en microsegundos. Ahora las
/// líneas formateadas se envían por un canal mpsc que un hilo dedicado
/// (`trade-fills-io`) drena y escribe. El handle del archivo se abre UNA vez
/// y se conserva (antes se re-abría por línea). Si el hilo drenador muere
/// se cae a escritura síncrona. Esta ruta sigue sin ACK durable y puede perder
/// datos por errores de disco o caída del proceso; la cola de I/O no es acotada.
static FILLS_LINE_TX: LazyLock<std::sync::mpsc::Sender<String>> =
    LazyLock::new(|| {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        // Hilo demonio por diseño: el proceso vive mientras el motor viva.
        let spawned = std::thread::Builder::new()
            .name("trade-fills-io".to_string())
            .spawn(move || drain_fills_journal(rx));
        if let Err(ref e) = spawned {
            println!(
                "⚠️ [TRADE-FILLS] no se pudo spawn del hilo de I/O ({}): las escrituras del diario volverán al hilo llamador",
                e
            );
        }
        tx
    });

fn drain_fills_journal(rx: std::sync::mpsc::Receiver<String>) {
    use std::io::Write;
    let _ = std::fs::create_dir_all("data");
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(FILLS_JOURNAL_PATH)
        .ok();
    while let Ok(line) = rx.recv() {
        if file.is_none() {
            // Disco/directorio reapareció: reabrir antes de soltar la línea.
            let _ = std::fs::create_dir_all("data");
            file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(FILLS_JOURNAL_PATH)
                .ok();
        }
        match file.as_mut() {
            Some(f) => {
                let _ = f.write_all(line.as_bytes());
            }
            None => {
                // Sin handle: escritura directa de emergencia.
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(FILLS_JOURNAL_PATH)
                {
                    let _ = f.write_all(line.as_bytes());
                }
            }
        }
    }
}

/// Envía una línea al hilo de I/O; cae a escritura síncrona si el canal murió.
fn journal_append(line: String) {
    if FILLS_LINE_TX.send(line.clone()).is_err() {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(FILLS_JOURNAL_PATH)
        {
            let _ = f.write_all(line.as_bytes());
        }
    }
}

/// Registra un cierre: append al diario persistente (hilo de fondo) + cola para Kelly.
pub fn record_bracket_close(rec: BracketClose) {
    // Valid syntax/numbers before either queue. Enqueue is NOT durable commit.
    let line = match rec.journal_line() {
        Ok(line) => line,
        Err(reason) => {
            INVALID_RECORDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            eprintln!("[ACCOUNTING] rejected invalid close record: {:?}", reason);
            return;
        }
    };
    journal_append(line);
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

/// Numeric/serialization rejections, separate from queue-capacity losses.
pub static INVALID_RECORDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
/// B3.21 — FALLBACK DE CONTEXTO para cierres de bracket: la reconciliación
/// (ciclo 60s) puede consumir la posición local — y poner entry_price=0 —
/// ANTES de que el fill del bracket llegue al stream. El diario de entradas
/// (position_journal.jsonl, B3.1) existe precisamente para cargar ese
/// contexto: devuelve el ÚLTIMO px de entrada registrado para símbolo+lado.
/// MOD1/4-012 (INFORME-14): caché del diario de contexto con firma
/// (mtime, len). Antes, cada disparo de TP/SL re-leía y re-parseaba el
/// position_journal.jsonl COMPLETO dentro del hilo del WebSocket — a 10k
/// entradas eran 10k parses serde bloqueando el procesamiento de fills.
/// Ahora: un `metadata` barato por llamada; sólo si la firma cambió (append
/// nuevo o compactación del host) se re-lee y re-parsea.
struct JournalCache {
    loaded: bool,
    sig: Option<(std::time::SystemTime, u64)>,
    /// (sym, was_long, ts, px) en orden de archivo.
    entries: Vec<(String, bool, u64, f64)>,
}

static JOURNAL_CACHE: LazyLock<Mutex<JournalCache>> = LazyLock::new(|| {
    Mutex::new(JournalCache {
        loaded: false,
        sig: None,
        entries: Vec::new(),
    })
});

pub fn last_journal_entry_px(symbol: &str, was_long: bool) -> Option<f64> {
    let sig = std::fs::metadata(POSITION_JOURNAL_PATH)
        .ok()
        .and_then(|m| m.modified().ok().map(|t| (t, m.len())));
    let mut cache = JOURNAL_CACHE
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    if !cache.loaded || cache.sig != sig {
        let content = std::fs::read_to_string(POSITION_JOURNAL_PATH).unwrap_or_default();
        let mut entries = Vec::new();
        for line in content.lines() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let sym = v.get("sym").and_then(|x| x.as_str()).unwrap_or("");
            let long = v.get("long").and_then(|x| x.as_bool()).unwrap_or(false);
            let px = v.get("px").and_then(|x| x.as_f64()).unwrap_or(0.0);
            let ts = v.get("ts").and_then(|x| x.as_u64()).unwrap_or(0);
            if px > 0.0 {
                entries.push((sym.to_string(), long, ts, px));
            }
        }
        cache.sig = sig;
        cache.entries = entries;
        cache.loaded = true;
    }
    // Último px por ts (empate → el más tardío en orden de archivo), idéntico
    // a la semántica del escaneo .rev() original.
    let mut best: Option<(u64, f64)> = None;
    for (sym, long, ts, px) in cache.entries.iter() {
        if sym == symbol && *long == was_long && best.map(|(b, _)| *ts >= b).unwrap_or(true) {
            best = Some((*ts, *px));
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
    match entry_fill_journal_line(ts_ms, symbol, long, qty, price, is_maker, commission) {
        Ok(line) => journal_append(line),
        Err(reason) => {
            INVALID_RECORDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            eprintln!("[ACCOUNTING] rejected invalid entry record: {:?}", reason);
        }
    }
}

/// Pure entry serializer; signed commission must already have a currency contract.
pub fn entry_fill_journal_line(ts_ms: u64, symbol: &str, long: bool, qty: f64,
    price: f64, is_maker: bool, commission: f64) -> Result<String, AccountingError> {
    if symbol.trim().is_empty() { return Err(AccountingError::InvalidSymbol); }
    if !qty.is_finite() || qty <= 0.0 { return Err(AccountingError::InvalidQuantity); }
    if !price.is_finite() || price <= 0.0 { return Err(AccountingError::InvalidEntryPrice); }
    if !commission.is_finite() { return Err(AccountingError::InvalidFees); }
    let value = serde_json::json!({"kind":"ENTRY", "ts":ts_ms, "sym":symbol,
        "long":long, "qty":qty, "px":price, "maker":is_maker, "fee":commission});
    serde_json::to_string(&value).map(|s| s + "\n").map_err(|_| AccountingError::Serialization)
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
        // B3.x: la cola PENDING es global; los tests que la tocan se serializan.
        let _guard = TEST_QUEUE_LOCK.lock();
        let _ = drain_bracket_closes();
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
