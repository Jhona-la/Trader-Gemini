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
use std::sync::{Arc, Mutex, OnceLock};

use execution_engine::puente_cuenta::{LecturaCuenta, Observacion, PuenteCuenta, Ticket};
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
///
/// DEMO_TEL T3 (T3-1): `wallet_balance` lleva el saldo aceptado con la
/// representación decimal más corta que lo reproduce exactamente (`Display` de
/// `f64`), sin redondear a cuatro decimales: dos saldos distintos que coinciden a
/// cuatro decimales salen distintos. No cambia la precisión del motor.
pub fn linea_capital_update(ts_ms: u64, saldo: f64, revision: Option<&RevisionSaldo>) -> String {
    let (fuente, era, socket, marca, hora) = match revision {
        Some(r) => (r.fuente, Some(r.era), Some(r.socket), r.marca_ms, r.hora_evento_ms),
        None => ("N/D", None, None, None, None),
    };
    format!(
        "DEMO_TEL event=CAPITAL_UPDATE ts={} wallet_balance={} fuente={} era={} socket={} marca_ms={} hora_evento_ms={}",
        ts_ms,
        saldo,
        fuente,
        o_nd(era),
        o_nd(socket),
        o_nd(marca),
        o_nd(hora)
    )
}

/// DEMO_TEL T3 (T3-1): la última revisión registrada (fuente, era, socket, `T`,
/// `E`, bits del saldo). La repetición exacta de una revisión ya registrada no
/// produce otra línea. Sólo afecta al registro.
static ULTIMA_REVISION: Mutex<Option<(&'static str, u64, u64, Option<u64>, Option<u64>, u64)>> = Mutex::new(None);

fn ts_ahora_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Registra una revisión aceptada (una línea por revisión distinta).
fn registrar_revision(saldo: f64, revision: Option<&RevisionSaldo>) {
    if let Some(r) = revision {
        let clave = (r.fuente, r.era, r.socket, r.marca_ms, r.hora_evento_ms, saldo.to_bits());
        let mut u = ULTIMA_REVISION.lock().unwrap_or_else(|p| p.into_inner());
        if *u == Some(clave) {
            return;
        }
        *u = Some(clave);
    }
    emitir(linea_capital_update(ts_ahora_ms(), saldo, revision));
}

/// DEMO_TEL T3 (T3-1): la lectura REST que el puente acaba de aceptar, con el
/// ticket con que se pidió. Emite `CAPITAL_UPDATE fuente=REST` sólo si el saldo
/// vigente del puente es el que trajo esa lectura (si el puente conservó otro
/// saldo más nuevo, esa lectura no fijó el saldo y no hay revisión REST que
/// registrar). `E` no existe en REST: `hora_evento_ms=N/D`. Sólo registro: no
/// toca el puente, el espejo, el libro ni el veto.
pub fn capital_rest_aceptado(ticket: &Ticket, lectura: &LecturaCuenta, aceptada: &Observacion) {
    if aceptada.saldo.to_bits() != lectura.saldo.to_bits() {
        return;
    }
    let revision = RevisionSaldo {
        fuente: "REST",
        era: ticket.era(),
        socket: ticket.socket(),
        marca_ms: lectura.tiempo_saldo_ms,
        hora_evento_ms: None,
    };
    registrar_revision(aceptada.saldo, Some(&revision));
}

/// DEMO_TEL T3 (T3-1): testigo de la lectura REST de la resincronización
/// periódica (`resincronizar_una_vez` sólo entrega la observación). Guarda el
/// ticket vigente al recibir la respuesta y la lectura, sin cambiarla.
#[derive(Default)]
pub struct LecturaRestAnotada {
    anotada: Mutex<Option<(Ticket, LecturaCuenta)>>,
}

impl LecturaRestAnotada {
    /// Envuelve el GET: devuelve exactamente lo que devolvió.
    pub async fn anotar<F>(&self, puente: &PuenteCuenta, lectura: F) -> Result<LecturaCuenta, String>
    where
        F: std::future::Future<Output = Result<LecturaCuenta, String>>,
    {
        let r = lectura.await;
        if let Ok(l) = &r {
            *self.anotada.lock().unwrap_or_else(|p| p.into_inner()) = Some((puente.ticket(), l.clone()));
        }
        r
    }

    /// Tras una pasada aceptada: registra su revisión REST.
    pub fn registrar(&self, aceptada: &Observacion) {
        let anotada = self.anotada.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some((ticket, l)) = anotada {
            capital_rest_aceptado(&ticket, &l, aceptada);
        }
    }
}

// ---------------------------------------------------------------------------
// DEMO_TEL T3 (T3-5): IMMUNE_TRIGGER íntegro por el logger acotado.
// ---------------------------------------------------------------------------

/// Cota del dominio admitido para los capitales de IMMUNE_TRIGGER (en USDT).
pub const IMMUNE_CAPITAL_MAX: f64 = 1e12;
/// Cota del dominio admitido para los porcentajes de IMMUNE_TRIGGER.
pub const IMMUNE_PCT_MAX: f64 = 1e6;
/// Marca que sustituye un valor fuera del dominio admitido: el lector la trata
/// como diagnóstico de cobertura, nunca como número.
pub const FUERA_DE_DOMINIO: &str = "FUERA_DE_DOMINIO";

/// Los operandos del disparo del inmune, tal como los calculó el host ANTES de
/// la decisión de este ciclo (sin relecturas).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisparoInmune {
    pub ts_ms: u64,
    /// OPERATOR_LOCK, INSOLVENTE, DRAWDOWN o LATENCIA (misma precedencia que `reason`).
    pub motivo: &'static str,
    pub freno_capital: f64,
    pub peak_capital: f64,
    /// `NaN` = el puente no tenía equity del freno: AUSENTE, sin sustituto.
    pub equity_del_freno: f64,
    pub caida_capital_pct: f64,
    pub tope_capital_pct: f64,
    pub operator_lock: bool,
    pub insolvent: bool,
    pub dd_breach: bool,
    pub lat_breach: bool,
    pub latency_strikes: u32,
}

/// Un decimal con `{:.4}` si está en el dominio admitido; si no, la marca.
struct Dominio(f64, f64);

impl std::fmt::Display for Dominio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_finite() && self.0.abs() < self.1 {
            write!(f, "{:.4}", self.0)
        } else {
            f.write_str(FUERA_DE_DOMINIO)
        }
    }
}

/// La equity del freno: AUSENTE si no hay, la marca si está fuera del dominio.
struct EquityDelFreno(f64);

impl std::fmt::Display for EquityDelFreno {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_finite() {
            Dominio(self.0, IMMUNE_CAPITAL_MAX).fmt(f)
        } else {
            f.write_str("AUSENTE")
        }
    }
}

/// Escribe la línea IMMUNE_TRIGGER. Los planos van con su código corto
/// (significado en `telemetry_server::report_demo_run::PLANOS_IMMUNE`):
/// `FRENO` = `drawdown.capital_del_freno(arena) [con libro derivado: equity del puente; sin el libro: min(unified_capital, capital_exchange)]`,
/// `PICO` = `drawdown.registrar_pico(arena.pico_capital,freno_capital)`,
/// `PUENTE` = `capital_del_freno(arena) [equity del puente si dimensionable_derivado; min(unified_capital, capital_exchange) en otro caso]`. Con el prefijo `[ts=]` de mayor
/// longitud representable y los máximos del dominio, la línea cabe en
/// `telemetry_server::macros::MAX_LOG_LEN` (lo comprueba la prueba T3-5).
pub fn escribir_immune_trigger(w: &mut impl std::fmt::Write, d: &DisparoInmune) -> std::fmt::Result {
    write!(
        w,
        "DEMO_TEL event=IMMUNE_TRIGGER ts={} motivo={} alcance=CUENTA freno_capital={} freno_capital_plano=FRENO peak_capital={} peak_plano=PICO equity_del_freno={} equity_del_freno_plano=PUENTE caida_capital_pct={} tope_capital_pct={} operator_lock={} insolvent={} dd_breach={} lat_breach={} latency_strikes={} revision=N/D lecturas_posteriores=ninguna",
        d.ts_ms,
        d.motivo,
        Dominio(d.freno_capital, IMMUNE_CAPITAL_MAX),
        Dominio(d.peak_capital, IMMUNE_CAPITAL_MAX),
        EquityDelFreno(d.equity_del_freno),
        Dominio(d.caida_capital_pct, IMMUNE_PCT_MAX),
        Dominio(d.tope_capital_pct, IMMUNE_PCT_MAX),
        d.operator_lock,
        d.insolvent,
        d.dd_breach,
        d.lat_breach,
        d.latency_strikes
    )
}

/// El emisor que llama la rama real del disparo (`god_engine.rs`): encola la
/// línea en el logger acotado (sin E/S síncrona ni espera; sin cola, `println!`
/// como cualquier `telemetry_log!`).
pub fn emitir_immune_trigger(d: &DisparoInmune) {
    struct Linea<'a>(&'a DisparoInmune);
    impl std::fmt::Display for Linea<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            escribir_immune_trigger(f, self.0)
        }
    }
    telemetry_server::telemetry_log!("{}", Linea(d));
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
        registrar_revision(usdt, revision);
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
