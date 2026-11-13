//! DEMO_TEL T2 (QRA-TELEMETRIA-DEMO, fila T3): el sink de capital del host y la
//! línea `CAPITAL_UPDATE` que emite por cada revisión aceptada del saldo.
//!
//! `CapitalBridgeSink` vivía dentro de `main()` de `src/bin/god_engine.rs`, donde
//! ninguna prueba podía ejecutarlo: la emisión de `CAPITAL_UPDATE` sólo se podía
//! anclar por texto. Se mueve aquí SIN cambiar su comportamiento (mismo
//! `update_dynamic_capital`, mismo espejo `unified_capital` con la misma guarda,
//! mismo `on_equity` y mismo aviso de posiciones) para que la prueba lo corra
//! con el streamer y el puente reales. Sólo cambia el contenido de la línea:
//! ahora lleva la revisión del saldo aceptado (fuente, era, socket, marca `T`,
//! hora `E`), con `N/D` explícito cuando falta.
//!
//! `vincular_testigo` es un observador de SOLO lectura de las líneas (pruebas):
//! no altera la emisión ni ninguna decisión.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use execution_engine::user_data_stream::{AccountSink, RemotePosition, RevisionSaldo};

type Testigo = Box<dyn Fn(&str) + Send + Sync>;
static TESTIGO: OnceLock<Testigo> = OnceLock::new();

/// Registra un observador de las líneas DEMO_TEL emitidas aquí (sólo el primero vale).
pub fn vincular_testigo(f: impl Fn(&str) + Send + Sync + 'static) -> bool {
    TESTIGO.set(Box::new(f)).is_ok()
}

fn emitir(linea: String) {
    if let Some(t) = TESTIGO.get() {
        t(&linea);
    }
    telemetry_server::telemetry_log!("{}", linea);
}

fn o_nd(v: Option<u64>) -> String {
    v.map_or_else(|| "N/D".to_string(), |x| x.to_string())
}

/// La línea `CAPITAL_UPDATE` de una revisión aceptada del saldo.
pub fn linea_capital_update(ts_ms: u64, saldo: f64, revision: Option<&RevisionSaldo>) -> String {
    let (fuente, era, socket, marca, hora) = match revision {
        Some(r) => (r.fuente, Some(r.era), Some(r.socket), r.marca_ms, r.hora_evento_ms),
        None => ("N/D", None, None, None, None),
    };
    format!(
        "DEMO_TEL event=CAPITAL_UPDATE ts={} wallet_balance={:.4} fuente={} era={} socket={} marca_ms={} hora_evento_ms={}",
        ts_ms,
        saldo,
        fuente,
        o_nd(era),
        o_nd(socket),
        o_nd(marca),
        o_nd(hora)
    )
}

/// CAPITAL-UNICO V2: `on_capital` es el SALDO (sin PnL no realizado) y
/// `on_equity` la EQUITY; las dos salen del puente de la cuenta.
pub struct CapitalBridgeSink {
    pub unified_capital: Arc<AtomicU64>,
    pub equity_exchange: Arc<AtomicU64>,
}

impl CapitalBridgeSink {
    fn capital(&self, usdt: f64, revision: Option<&RevisionSaldo>) {
        audit_engine::telemetry::update_dynamic_capital(usdt);
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        emitir(linea_capital_update(ts, usdt, revision));
        // F2: permite propagar 0.0 observado de la cuenta real al puente de capital
        if usdt >= 0.0 && usdt.is_finite() {
            self.unified_capital.store(usdt.to_bits(), Ordering::Relaxed);
        }
    }
}

impl AccountSink for CapitalBridgeSink {
    fn on_capital(&self, usdt: f64) {
        self.capital(usdt, None);
    }
    fn on_capital_aceptado(&self, usdt: f64, revision: &RevisionSaldo) {
        self.capital(usdt, Some(revision));
    }
    fn on_equity(&self, usdt: f64) {
        if usdt >= 0.0 && usdt.is_finite() {
            self.equity_exchange.store(usdt.to_bits(), Ordering::Relaxed);
        }
    }
    fn on_positions(&self, positions: &[RemotePosition]) {
        if !positions.is_empty() {
            let summary: Vec<String> = positions
                .iter()
                .map(|p| format!("{} {}", p.symbol, p.position_amt))
                .collect();
            telemetry_server::telemetry_log!("📡 [USER-DATA] Posiciones vivas: {}", summary.join(", "));
        }
    }
    fn on_bracket_close(&self, _c: &execution_engine::trade_accounting::BracketClose) {}
}
