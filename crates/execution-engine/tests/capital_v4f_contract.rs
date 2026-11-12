//! CAPITAL-UNICO V4F: pruebas del integrador para la ventana entre el relevo del OMS y el cambio
//! de era (dictamen de Codex QRA sobre V4E, P1-c) y sus hermanas.
//!
//! Toda escritura del puente (saldo, equity del freno, posiciones, veto, socket, generacion) y la
//! publicacion al espejo (`con_era_vigente`) pasan por `PuenteCuenta::vigente`, que comprueba era
//! y testigo de la sesion OMS bajo el mutex del puente, en el mismo paso que la escritura. Con la
//! sesion del OMS ya relevada, cada una se rechaza entera: no cambia nada ni publica nada.
//! Importes del fixture.

#[path = "../src/puente_cuenta.rs"]
#[allow(dead_code)]
mod puente_cuenta;

use puente_cuenta::{EventoStream, LecturaCuenta, Observacion, PuenteCuenta, Rechazo};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};

const ANTERIOR: u64 = 10;
const NUEVA: u64 = 11;

fn rest(saldo: f64, t: u64) -> LecturaCuenta {
    LecturaCuenta {
        saldo,
        tiempo_saldo_ms: Some(t),
        posiciones: Some(HashMap::new()),
        rechazo_arranque: None,
        upnl_del_activo: Some(0.0),
    }
}

fn ws(t: u64, saldo: f64) -> EventoStream {
    EventoStream { tiempo_ms: Some(t), saldo: Some(saldo), posiciones: vec![] }
}

/// Lo que cualquier escritura cambiaria.
#[derive(Debug, PartialEq)]
struct Foto {
    era: u64,
    sesion: u64,
    socket: u64,
    generacion: u64,
    observacion: Option<Observacion>,
    freno: Option<f64>,
    veta: bool,
    resincronizar: bool,
    posiciones: usize,
}

fn foto(p: &PuenteCuenta) -> Foto {
    let t = p.ticket();
    Foto {
        era: p.era(),
        sesion: p.sesion(),
        socket: p.socket(),
        generacion: t.generacion(),
        observacion: p.observacion(),
        freno: p.equity_del_freno(),
        veta: p.veta_entradas(),
        resincronizar: p.necesita_resincronizar(),
        posiciones: p.posiciones_en_cache(),
    }
}

/// Puente admitido en la sesion ANTERIOR (saldo 100), con testigo y publicador contados.
fn puente_admitido() -> (PuenteCuenta, Arc<AtomicU64>, Arc<AtomicUsize>) {
    let p = PuenteCuenta::con_sesion(ANTERIOR);
    let oms = Arc::new(AtomicU64::new(ANTERIOR));
    let testigo = Arc::clone(&oms);
    p.vincular_testigo_de_sesion(move || testigo.load(Ordering::Acquire));
    let llamadas = Arc::new(AtomicUsize::new(0));
    let cuenta = Arc::clone(&llamadas);
    p.vincular_publicador(move |pb| {
        cuenta.fetch_add(1, Ordering::Relaxed);
        !pb.veta
    });
    p.abrir_socket(p.era()).unwrap();
    p.observar_rest(p.ticket(), &rest(100.0, 100)).unwrap();
    assert!(!p.veta_entradas());
    (p, oms, llamadas)
}

fn otra_sesion<T: std::fmt::Debug>(r: Result<T, Rechazo>) {
    assert!(
        matches!(r, Err(Rechazo::OtraSesion { sesion: ANTERIOR, vigente: NUEVA })),
        "se esperaba OtraSesion {{ {ANTERIOR}, {NUEVA} }}: {r:?}"
    );
}

#[test]
fn v4f_testigo_rechaza_entera_cada_escritura_de_la_sesion_anterior() {
    let (p, oms, llamadas) = puente_admitido();
    let era = p.era();
    let ticket = p.ticket();
    oms.store(NUEVA, Ordering::Release); // relevo del OMS; la era sigue siendo la anterior
    let antes = foto(&p);
    let publicadas = llamadas.load(Ordering::Relaxed);

    otra_sesion(p.observar_rest_de_sesion(ticket, ANTERIOR, &rest(97.0, 200)));
    otra_sesion(p.observar_rest(ticket, &rest(97.0, 200)));
    let espejo = AtomicUsize::new(0);
    otra_sesion(p.observar_stream_del_socket(era, p.socket(), Some(201), &ws(200, 97.0), |_| {
        espejo.fetch_add(1, Ordering::Relaxed);
    }));
    otra_sesion(p.observar_stream_y_publicar(era, &ws(200, 97.0), |_| {
        espejo.fetch_add(1, Ordering::Relaxed);
    }));
    otra_sesion(p.observar_stream(era, &ws(200, 97.0)));
    otra_sesion(p.abrir_socket(era));
    otra_sesion(p.reconexion(era));
    otra_sesion(p.perdida_socket(era));
    otra_sesion(p.sin_socket(era));
    otra_sesion(p.evidencia_perdida(era));
    assert_eq!(p.con_era_vigente(era, |_| espejo.fetch_add(1, Ordering::Relaxed)), None);

    assert_eq!(espejo.load(Ordering::Relaxed), 0, "el espejo no se escribe");
    assert_eq!(llamadas.load(Ordering::Relaxed), publicadas, "el publicador no corre");
    assert_eq!(foto(&p), antes, "ni saldo, ni equity del freno, ni veto, ni socket, ni generacion");

    // Control: con la sesion del OMS de vuelta en la de la era, la misma lectura entra.
    oms.store(ANTERIOR, Ordering::Release);
    assert_eq!(p.observar_rest(ticket, &rest(97.0, 200)).unwrap().saldo, 97.0);
    assert_eq!(p.equity_del_freno(), Some(97.0));
}

#[test]
fn v4f_control_sin_testigo_el_puente_conserva_el_contrato_v4e() {
    // Sin testigo vinculado (pruebas y caminos sin OMS) decide solo la era y la sesion declarada.
    let p = PuenteCuenta::con_sesion(ANTERIOR);
    p.abrir_socket(p.era()).unwrap();
    assert_eq!(p.observar_rest(p.ticket(), &rest(100.0, 100)).unwrap().saldo, 100.0);
    otra_sesion_declarada(p.observar_rest_de_sesion(p.ticket(), NUEVA, &rest(97.0, 200)));
    assert_eq!(p.observacion().map(|o| o.saldo), Some(100.0));
}

fn otra_sesion_declarada<T: std::fmt::Debug>(r: Result<T, Rechazo>) {
    assert!(matches!(r, Err(Rechazo::OtraSesion { sesion: NUEVA, vigente: ANTERIOR })), "{r:?}");
}

#[test]
fn v4f_relevo_publicado_abre_la_era_sin_socket_en_el_mismo_paso() {
    let (p, oms, _) = puente_admitido();
    let era_vieja = p.era();
    let socket_viejo = p.socket();
    let ticket = p.ticket();
    let testigo = Arc::clone(&oms);
    let (r, era) = p.relevar_sesion(|| {
        testigo.store(NUEVA, Ordering::Release);
        ("publicada", Some(NUEVA))
    });
    assert_eq!(r, "publicada");
    let era = era.expect("era nueva");
    assert_eq!(era, era_vieja + 1);
    assert_eq!(p.era(), era);
    assert_eq!(p.sesion(), NUEVA);
    assert!(p.socket() > socket_viejo);
    assert!(p.veta_entradas());
    assert_eq!(p.observacion(), None);
    assert_eq!(p.equity_del_freno(), Some(100.0), "la equity anterior queda solo como evidencia del freno");
    assert_eq!(p.observar_rest(ticket, &rest(97.0, 200)), Err(Rechazo::OtraEra));
    // Nace sin socket: una REST de la era nueva llega al freno pero no admite.
    p.observar_rest_de_sesion(p.ticket(), NUEVA, &rest(90.0, 300)).unwrap();
    assert_eq!(p.equity_del_freno(), Some(90.0));
    assert!(p.veta_entradas(), "sin handshake del socket de la era nueva no se admite");
    p.abrir_socket(era).unwrap();
    p.observar_rest_de_sesion(p.ticket(), NUEVA, &rest(90.0, 300)).unwrap();
    assert!(!p.veta_entradas());
}

#[test]
fn v4f_relevo_no_publicado_no_cambia_nada() {
    let (p, _oms, llamadas) = puente_admitido();
    let antes = foto(&p);
    let publicadas = llamadas.load(Ordering::Relaxed);
    let ((), era) = p.relevar_sesion(|| ((), None));
    assert_eq!(era, None);
    assert_eq!(foto(&p), antes);
    assert_eq!(llamadas.load(Ordering::Relaxed), publicadas);
}

/// El GET de la sesion anterior que completa mientras corre el relevo espera al mutex del puente y
/// despues encuentra otra era: en ningun intercalado escribe saldo, freno ni espejo.
#[test]
fn v4f_get_anterior_concurrente_con_el_relevo_no_escribe_en_ningun_intercalado() {
    for _ in 0..200 {
        let (p, oms, _) = puente_admitido();
        p.evidencia_perdida(p.era()).unwrap();
        let p = Arc::new(p);
        let ticket = p.ticket();
        let espejo = Arc::new(AtomicU64::new(100.0f64.to_bits()));
        let salida = Arc::new(Barrier::new(2));
        let get = {
            let p = Arc::clone(&p);
            let espejo = Arc::clone(&espejo);
            let salida = Arc::clone(&salida);
            std::thread::spawn(move || {
                salida.wait();
                let r = p.observar_rest_de_sesion(ticket, ANTERIOR, &rest(97.0, 200));
                if let Ok(obs) = &r {
                    let _ = p.con_era_vigente(ticket.era(), |v| {
                        espejo.store(v.unwrap_or(*obs).saldo.to_bits(), Ordering::Relaxed)
                    });
                }
                r
            })
        };
        let testigo = Arc::clone(&oms);
        let (_, era) = p.relevar_sesion(|| {
            salida.wait();
            testigo.store(NUEVA, Ordering::Release);
            ((), Some(NUEVA))
        });
        let r = get.join().unwrap();
        // Antes del relevo la lectura era de la sesion vigente y podia entrar; aqui el GET sale
        // de la barrera dentro del relevo, con el mutex tomado: siempre llega tarde.
        assert_eq!(r, Err(Rechazo::OtraEra), "intercalado con escritura de la sesion anterior");
        assert_eq!(p.era(), era.unwrap());
        assert_eq!(p.equity_del_freno(), Some(100.0));
        assert_eq!(f64::from_bits(espejo.load(Ordering::Relaxed)), 100.0);
    }
}

#[test]
fn v4f_contrato_de_fuente_un_solo_punto_para_las_escrituras() {
    let puente = include_str!("../src/puente_cuenta.rs").replace("\r\n", "\n");
    let cuerpo = |firma: &str| -> String {
        let desde = puente.find(firma).unwrap_or_else(|| panic!("falta {firma}"));
        let resto = &puente[desde..];
        let fin = resto[1..].find("\n    pub fn ").or_else(|| resto[1..].find("\n    fn ")).unwrap_or(resto.len() - 1);
        resto[..fin + 1].to_string()
    };
    for firma in [
        "pub fn observar_rest_de_sesion<",
        "fn observar_evento(",
        "pub fn abrir_socket(",
        "pub fn perdida_socket(",
        "pub fn sin_socket(",
        "pub fn evidencia_perdida(",
        "pub fn con_era_vigente<",
    ] {
        assert!(cuerpo(firma).contains("self.vigente(&e,"), "{firma} no pasa por `vigente`");
    }
    let host = include_str!("../../../src/bin/god_engine.rs").replace("\r\n", "\n");
    assert!(!host.contains("puente_cuenta.nueva_era_de_sesion("), "la era solo la abre el relevo");
    assert!(!host.contains("puente_cuenta.nueva_era()"));
    assert_eq!(host.matches("puente_cuenta.relevar_sesion(").count(), 1);
    assert_eq!(host.matches(".vincular_testigo_de_sesion(").count(), 1);
    let testigo = host.find(".vincular_testigo_de_sesion(").unwrap();
    assert!(testigo < host.find("let token_arranque = oms_session.snapshot_token();").unwrap(), "antes de la REST de arranque");
    assert!(testigo < host.find("puente_cuenta.vincular_publicador(").unwrap());
}
