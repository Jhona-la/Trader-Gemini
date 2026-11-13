//! CAPITAL-UNICO V4F: RED adversaria de Codex QRA para P1-c (`qra_v4e_p1c_get_anterior.rs`,
//! dictamen de V4E de las 16:55, sha256 03C21E3813AFA81A1D7DB4DEF58AA5EFAB5EB32BC7DA693CAD606C184EB015F8),
//! adoptada por el integrador. Contra 5b974f70 falla igual que en el dictamen
//! (`v4f2_logs\red.contra-5b974f70.log`). Cambios, uno por uno:
//! 1. Esta cabecera y las rutas: el puente y las fuentes del host salen del arbol del repo, no de
//!    `v4e_p1c_fuentes/`.
//! 2. Enlace: el puente recibe el testigo de la sesion OMS (`vincular_testigo_de_sesion`) leyendo
//!    la misma sesion que el publicador del arnes, como hace el host con `oms.snapshot_token()`.
//!    Su nota lo pide: conectar las mismas aserciones a la autoridad de produccion.
//! 3. Anclas del host: la era nueva la abre `relevar_sesion`, que envuelve el relevo del OMS
//!    (`paso_transicion`), y el host vincula el testigo. Las del OMS, del publicador y del
//!    resincronizador no cambian.
//! Las aserciones y el `ResultadoDelCorte` esperado no cambian.
//! Se agrega `qra_p1c_..._por_el_relevo_del_host`: el mismo corte por el camino real del host.

//! P1-c, old request completing after OMS replacement and before bridge handoff.
//! The bridge is production code. Scalar state represents the two OMS
//! generations. Source anchors tie the scheduler cut and the two publication
//! callbacks to the host. Runtime host/transport and disk are outside the harness.
//! If the fix seals this cut in the host, wire the same assertions to that
//! production authority instead of driving the bridge in isolation.

#[path = "../src/puente_cuenta.rs"]
#[allow(dead_code)]
mod puente_cuenta;

use puente_cuenta::{LecturaCuenta, Observacion, Publicacion, PuenteCuenta, Rechazo};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

const HOST: &str = include_str!("../../../src/bin/god_engine.rs");
const OMS: &str = include_str!("../../../src/host_oms.rs");
const RESYNC: &str = include_str!("../src/resincronizar_cuenta.rs");

const ANTERIOR: u64 = 10;
const NUEVA: u64 = 11;
const OBSERVACION_ANTES: Observacion = Observacion { saldo: 100.0, equity: 100.0 };

struct LectorHost {
    sesion: AtomicU64,
    dimensionable: AtomicU64,
    espejo: AtomicU64,
    veto: AtomicBool,
    rechazadas_por_sesion: AtomicUsize,
}

impl LectorHost {
    fn nuevo() -> Self {
        Self {
            sesion: AtomicU64::new(ANTERIOR),
            dimensionable: AtomicU64::new(0.0f64.to_bits()),
            espejo: AtomicU64::new(0.0f64.to_bits()),
            veto: AtomicBool::new(true),
            rechazadas_por_sesion: AtomicUsize::new(0),
        }
    }

    fn publicar(&self, p: &Publicacion) -> bool {
        // V4G (adaptador): como el host, publica el saldo vigente en cada cambio sin veto.
        let suelto = match (p.veta, p.vigente) {
            (true, _) | (false, None) => false,
            (false, Some((saldo, _))) => {
                if p.sesion == self.sesion.load(Ordering::Acquire) {
                    self.espejo.store(saldo.to_bits(), Ordering::Relaxed);
                    self.dimensionable.store(saldo.to_bits(), Ordering::Relaxed);
                    true
                } else {
                    self.rechazadas_por_sesion.fetch_add(1, Ordering::Relaxed);
                    false
                }
            }
        };
        self.veto.store(!suelto, Ordering::Release);
        suelto
    }

    fn saldos(&self) -> (f64, f64) {
        (
            f64::from_bits(self.dimensionable.load(Ordering::Relaxed)),
            f64::from_bits(self.espejo.load(Ordering::Relaxed)),
        )
    }
}

fn lectura(saldo: f64, marca: u64) -> LecturaCuenta {
    LecturaCuenta {
        saldo,
        tiempo_saldo_ms: Some(marca),
        posiciones: Some(Default::default()),
        rechazo_arranque: None,
        upnl_del_activo: Some(0.0),
    }
}

fn orden(source: &str, markers: &[&str]) -> Result<(), String> {
    let mut rest = source;
    for &marker in markers {
        let (_, after) = rest.split_once(marker)
            .ok_or_else(|| format!("reanchor the production cut at {marker}"))?;
        rest = after;
    }
    Ok(())
}

fn comprobar_camino_real() -> Result<(), String> {
    orden(OMS, &["pub fn replace_executor(", "state.generation = next",
        "self.executors.store(executor)"])?;
    // V4F (anclas adaptadas): el host vincula el testigo del OMS al puente, y el
    // relevo del OMS corre dentro de `relevar_sesion`, que abre la era.
    orden(HOST, &["PuenteCuenta::con_sesion(",
        "puente_cuenta.vincular_testigo_de_sesion(move || oms_testigo.snapshot_token().generation());",
        "let token_arranque = oms_session.snapshot_token();"])?;
    orden(HOST, &["let (paso, era_del_relevo) = puente_cuenta.relevar_sesion(|| {",
        "let paso = oms_session.paso_transicion(",
        "ResultadoTransicion::Publicada { generacion, .. } => Some(*generacion)",
        "ResultadoTransicion::Publicada { generacion, ejecutor }",
        "let Some(era_mainnet) = era_del_relevo else {"])?;
    // V4G (ancla adaptada): el publicador entrega el saldo vigente (`p.vigente`).
    orden(HOST, &["fn publicar_capital_de_sesion(", "(false, Some((saldo, marca)))",
        "oms.snapshot_token()", "token.generation() == p.sesion",
        "oms.publish_current_balance(&token, saldo", "arena.capital_incompleto.store(!suelto"])?;
    orden(HOST, &["let token = oms_rs.snapshot_token();", "executor.fetch_account_lectura()",
        "|obs| saldo_rs.store(obs.saldo.to_bits()"])?;
    orden(RESYNC, &["let ticket = puente.ticket();", "tokio::time::timeout(tope, lectura).await",
        "puente.observar_rest_de_sesion(ticket, sesion, &l)",
        ".con_era_vigente(ticket.era(), |vigente| publicar("])
}

#[derive(Debug, PartialEq)]
struct ResultadoDelCorte {
    retorno_get: Option<Observacion>,
    observacion_era_vieja: Option<Observacion>,
    freno_era_vieja: Option<f64>,
    veto_del_puente_era_vieja: bool,
    observacion_era_nueva: Option<Observacion>,
    freno_heredado_era_nueva: Option<f64>,
    dimensionable: f64,
    espejo: f64,
}

/// V4F (enlace): el testigo del puente lee la sesion OMS del host del arnes.
fn vincular_testigo(puente: &PuenteCuenta, host: &Arc<LectorHost>) {
    let testigo = Arc::clone(host);
    puente.vincular_testigo_de_sesion(move || testigo.sesion.load(Ordering::Acquire));
}

#[test]
fn qra_p1c_get_anterior_no_contamina_la_equity_heredada_en_handoff() -> Result<(), String> {
    comprobar_camino_real()?;

    let puente = PuenteCuenta::con_sesion(ANTERIOR);
    let host = Arc::new(LectorHost::nuevo());
    vincular_testigo(&puente, &host);
    let publicador = Arc::clone(&host);
    puente.vincular_publicador(move |p| publicador.publicar(p));

    let era_vieja = puente.era();
    puente.abrir_socket(era_vieja).map_err(|e| e.to_string())?;
    puente.observar_rest_de_sesion(puente.ticket(), ANTERIOR, &lectura(100.0, 100))
        .map_err(|e| e.to_string())?;
    assert_eq!(puente.observacion(), Some(OBSERVACION_ANTES));
    assert_eq!(host.saldos(), (100.0, 100.0));
    assert!(!host.veto.load(Ordering::Acquire));

    // A private frame was lost on a still-open socket. This reaches the
    // periodic GET branch with both token and ticket from the old session.
    puente.evidencia_perdida(era_vieja).map_err(|e| e.to_string())?;
    assert!(puente.necesita_resincronizar());
    assert!(host.veto.load(Ordering::Acquire));
    let ticket = puente.ticket();

    // OMS publishes the successor. Pause the transition before new bridge era.
    host.sesion.store(NUEVA, Ordering::Release);
    assert_eq!(puente.sesion(), ANTERIOR);
    assert_eq!(puente.era(), era_vieja);
    let resultado = puente.observar_rest_de_sesion(ticket, ANTERIOR, &lectura(97.0, 200));
    if let Ok(obs) = &resultado {
        // V4E periodic helper's second callback. Even though the dimensioner
        // refused the old session, this callback checks only the old era.
        let _ = puente.con_era_vigente(ticket.era(), |vigente| {
            host.espejo.store(vigente.unwrap_or(*obs).saldo.to_bits(), Ordering::Relaxed);
        });
    }
    let observacion_era_vieja = puente.observacion();
    let freno_era_vieja = puente.equity_del_freno();
    let veto_del_puente_era_vieja = puente.veta_entradas();

    let era_nueva = puente.nueva_era_de_sesion(NUEVA);
    puente.sin_socket(era_nueva).map_err(|e| e.to_string())?;
    assert!(host.veto.load(Ordering::Acquire));
    assert_eq!(puente.sesion(), NUEVA);
    let (dimensionable, espejo) = host.saldos();
    let observado = ResultadoDelCorte {
        retorno_get: resultado.ok(),
        observacion_era_vieja,
        freno_era_vieja,
        veto_del_puente_era_vieja,
        observacion_era_nueva: puente.observacion(),
        freno_heredado_era_nueva: puente.equity_del_freno(),
        dimensionable,
        espejo,
    };
    let intentos_rehusados = host.rechazadas_por_sesion.load(Ordering::Relaxed);
    assert_eq!(observado, ResultadoDelCorte {
        retorno_get: None,
        observacion_era_vieja: Some(OBSERVACION_ANTES),
        freno_era_vieja: Some(100.0),
        veto_del_puente_era_vieja: true,
        observacion_era_nueva: None,
        freno_heredado_era_nueva: Some(100.0),
        dimensionable: 100.0,
        espejo: 100.0,
    }, "no old-session mutation after OMS handoff; refused sizing callbacks={intentos_rehusados}");
    Ok(())
}

/// V4F (agregada por el integrador): el mismo corte por la autoridad de produccion. El host
/// publica el ejecutor DENTRO de `relevar_sesion` (aqui, el cierre sube la sesion del OMS), asi
/// que el GET anterior completa despues del cambio de era. Mismos importes y mismas salidas
/// esperadas que la RED, salvo las de la era vieja, que ya no son observables tras el relevo.
#[test]
fn qra_p1c_get_anterior_no_contamina_la_equity_heredada_por_el_relevo_del_host() -> Result<(), String> {
    comprobar_camino_real()?;
    let puente = PuenteCuenta::con_sesion(ANTERIOR);
    let host = Arc::new(LectorHost::nuevo());
    vincular_testigo(&puente, &host);
    let publicador = Arc::clone(&host);
    puente.vincular_publicador(move |p| publicador.publicar(p));
    let era_vieja = puente.era();
    puente.abrir_socket(era_vieja).map_err(|e| e.to_string())?;
    puente.observar_rest_de_sesion(puente.ticket(), ANTERIOR, &lectura(100.0, 100))
        .map_err(|e| e.to_string())?;
    puente.evidencia_perdida(era_vieja).map_err(|e| e.to_string())?;
    let ticket = puente.ticket();

    let oms = Arc::clone(&host);
    let ((), era_nueva) = puente.relevar_sesion(|| {
        oms.sesion.store(NUEVA, Ordering::Release);
        ((), Some(NUEVA))
    });
    let era_nueva = era_nueva.ok_or("el relevo publicado abre era")?;
    assert_ne!(era_nueva, era_vieja);
    let resultado = puente.observar_rest_de_sesion(ticket, ANTERIOR, &lectura(97.0, 200));
    assert_eq!(resultado, Err(Rechazo::OtraEra));
    if let Ok(obs) = &resultado {
        let _ = puente.con_era_vigente(ticket.era(), |vigente| {
            host.espejo.store(vigente.unwrap_or(*obs).saldo.to_bits(), Ordering::Relaxed);
        });
    }
    assert_eq!(puente.con_era_vigente(ticket.era(), |_| ()), None, "el espejo no se escribe con la era vieja");
    assert!(host.veto.load(Ordering::Acquire));
    assert!(puente.veta_entradas());
    assert_eq!(puente.sesion(), NUEVA);
    assert_eq!(puente.observacion(), None);
    assert_eq!(puente.equity_del_freno(), Some(100.0));
    assert_eq!(host.saldos(), (100.0, 100.0));
    Ok(())
}
