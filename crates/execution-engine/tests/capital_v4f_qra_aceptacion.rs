//! CAPITAL-UNICO V4E: aceptacion V4F de Codex QRA (`qra_capital_v4f_aceptacion.rs`, 16:05,
//! sha256 19C0372B07AB42EAF297C9FA31D8FD8B05164383492C4B98E029A748EFB8B35D), TAL COMO ESTA
//! ESCRITA. Cambios: esta cabecera, la ruta del puente y del host al arbol del repo y, como pide su
//! nota, el adaptador `observar_arranque` conectado al punto real nuevo (el publicador: con el
//! contrato V4E `vincular_veto` ya no recibe `false`). Las aserciones no cambian.

//! Contratos de aceptación de V4F. Ejecutar contra el SHA que se va a juzgar.
//! Colocar este archivo junto a puente_cuenta.rs y god_engine.rs del mismo SHA.
//! El adaptador de arranque reconoce la marca síncrona del host. Si cambia
//! la API, conectar ese adaptador al nuevo punto real y conservar los asertos.

#[path = "../src/puente_cuenta.rs"]
#[allow(dead_code)]
mod puente_cuenta;

use puente_cuenta::{DatoPosicion, EventoStream, LecturaCuenta, Observacion, PosicionRest, PuenteCuenta, Rechazo};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const HOST: &str = include_str!("../../../src/bin/god_engine.rs");

fn foto(saldo: f64, t: u64) -> LecturaCuenta {
    LecturaCuenta {
        saldo,
        tiempo_saldo_ms: Some(t),
        posiciones: Some(HashMap::new()),
        rechazo_arranque: None,
        upnl_del_activo: Some(0.0),
    }
}

fn ws(t: u64, saldo: f64) -> EventoStream {
    EventoStream { tiempo_ms: Some(t), saldo: Some(saldo), posiciones: Vec::new() }
}

fn ancla<'a>(fuente: &'a str, desde: &str, hasta: &str) -> Result<&'a str, String> {
    let (_, inicio) = fuente.split_once(desde).ok_or_else(|| format!("Falta ancla inicial {desde}"))?;
    inicio.split_once(hasta).map(|(tramo, _)| tramo).ok_or_else(|| format!("Falta ancla final {hasta}"))
}

fn marca_sin_socket(tramo: &str, llamada: &str) -> bool {
    let codigo = tramo.lines()
        .map(|linea| linea.split_once("//").map_or(linea, |(codigo, _)| codigo))
        .flat_map(str::split_whitespace)
        .collect::<String>();
    codigo.split_once(llamada).is_some_and(|(antes, _)| !antes.contains(".await") && !antes.contains("spawn("))
}

fn observar_arranque(p: &PuenteCuenta) -> Arc<AtomicBool> {
    let publicado = Arc::new(AtomicBool::new(true));
    let salida = Arc::clone(&publicado);
    // V4E (adaptador conectado al publicador): publica el saldo y despues suelta.
    p.vincular_publicador(move |pb| {
        salida.store(pb.veta, Ordering::Release);
        !pb.veta
    });
    publicado
}

fn rest_del_hueco_y_confirmacion(
    p: &PuenteCuenta,
    publicado: &AtomicBool,
    saldo: f64,
    t: u64,
) -> Result<(), Rechazo> {
    let anterior = p.ticket();
    assert_eq!(p.observar_rest(anterior, &foto(saldo, t))?, Observacion { saldo, equity: saldo });
    assert_eq!(p.equity_del_freno(), Some(saldo));
    assert!(publicado.load(Ordering::Acquire), "REST antes del primer poll no admite sin handshake");
    p.abrir_socket(p.era())?;
    assert!(matches!(p.observar_rest(anterior, &foto(saldo, t)), Err(Rechazo::OtroSocket { .. })));
    assert!(publicado.load(Ordering::Acquire));
    p.observar_rest(p.ticket(), &foto(saldo, t))?;
    assert!(!publicado.load(Ordering::Acquire), "handshake y REST del socket vigente recuperan");
    Ok(())
}

#[test]
fn p1a_arranque_frio_no_autoriza_antes_del_primer_poll() -> Result<(), String> {
    let tramo = ancla(HOST, "let puente_cuenta =", "let (initial_capital, initial_equity)")?;
    let p = PuenteCuenta::new();
    assert_eq!(p.equity_del_freno(), None, "sin evidencia, el freno no inventa equity cero");
    if marca_sin_socket(tramo, "puente_cuenta.sin_socket(puente_cuenta.era())") {
        p.sin_socket(p.era()).map_err(|e| e.to_string())?;
    }
    let publicado = observar_arranque(&p);
    rest_del_hueco_y_confirmacion(&p, &publicado, 100.0, 100).map_err(|e| e.to_string())?;
    Ok(())
}

#[test]
fn p1a_transicion_no_autoriza_antes_del_primer_poll() -> Result<(), String> {
    let tramo = ancla(HOST, "let era_mainnet =", "let oms_transition =")?;
    let p = PuenteCuenta::new();
    p.observar_rest(p.ticket(), &foto(120.0, 50)).map_err(|e| e.to_string())?;
    let publicado = observar_arranque(&p);
    let era_anterior = p.ticket();
    let nueva = p.nueva_era();
    assert!(matches!(p.observar_rest(era_anterior, &foto(120.0, 50)), Err(Rechazo::OtraEra)));
    assert_eq!(p.equity_del_freno(), Some(120.0));
    if marca_sin_socket(tramo, "puente_cuenta.sin_socket(era_mainnet)") {
        p.sin_socket(nueva).map_err(|e| e.to_string())?;
    }
    rest_del_hueco_y_confirmacion(&p, &publicado, 80.0, 100).map_err(|e| e.to_string())?;
    Ok(())
}

fn saldo_3a() -> LecturaCuenta {
    let mut lectura = foto(0.0, 100);
    lectura.rechazo_arranque = Some("3a".to_string());
    lectura
}

#[test]
fn inv2_ticket_anterior_a_ws_aceptado_conserva_veto_3a() -> Result<(), Rechazo> {
    let p = PuenteCuenta::new();
    p.observar_rest(p.ticket(), &saldo_3a())?;
    let previo = p.ticket();
    p.observar_stream_del_socket(p.era(), p.socket(), Some(200), &ws(100, 0.0), |_| {})?;
    assert!(p.veta_entradas(), "el WS igual conserva 3a");
    let resultado = p.observar_rest(previo, &foto(0.0, 100));
    eprintln!("REST con ticket anterior={resultado:?}, veto={}", p.veta_entradas());
    assert_eq!(p.equity_del_freno(), Some(0.0));
    assert!(p.veta_entradas(), "REST anterior a WS aceptado quitó el veto 3a");
    let posterior = p.observar_rest(p.ticket(), &foto(0.0, 100))?;
    assert_eq!(posterior, Observacion { saldo: 0.0, equity: 0.0 });
    assert!(!p.veta_entradas(), "un cero ordinario completo despeja el veto de evidencia");
    Ok(())
}

#[test]
fn control_3a_vigente_y_cero_ordinario_completo_se_distinguen() -> Result<(), Rechazo> {
    let p = PuenteCuenta::new();
    p.observar_rest(p.ticket(), &saldo_3a())?;
    assert_eq!(p.equity_del_freno(), Some(0.0));
    assert!(p.veta_entradas());
    assert!(!p.necesita_resincronizar(), "3a es completo para el freno");
    p.observar_rest(p.ticket(), &foto(0.0, 100))?;
    assert!(!p.veta_entradas());
    assert_eq!(p.equity_del_freno(), Some(0.0));
    Ok(())
}

fn con_testigo_ws() -> Result<PuenteCuenta, Rechazo> {
    let p = PuenteCuenta::new();
    p.observar_rest(p.ticket(), &foto(101.0, 99))?;
    p.observar_stream_del_socket(p.era(), p.socket(), Some(200), &ws(100, 100.0), |_| {})?;
    Ok(p)
}

fn rechazo_no_quita_testigo(e: Option<u64>, socket_nuevo: bool) -> Result<(), Rechazo> {
    let p = con_testigo_ws()?;
    let socket = if socket_nuevo { p.abrir_socket(p.era())? } else { p.socket() };
    let rechazado = p.observar_stream_del_socket(p.era(), socket, e, &ws(100, 90.0), |_| {});
    assert!(matches!(rechazado, Err(Rechazo::Contradiccion { ref componente, tiempo_ms: 100 }) if componente == "saldo"));
    assert_eq!((p.observacion().map(|o| o.saldo), p.equity_del_freno(), p.veta_entradas()),
        (Some(100.0), Some(90.0), true));
    let rest = p.observar_rest(p.ticket(), &foto(110.0, 100));
    let estado = (p.observacion().map(|o| o.saldo), p.equity_del_freno(), p.veta_entradas());
    eprintln!("E={e:?}; WS={rechazado:?}; REST={rest:?}; (saldo, freno, veto)={estado:?}");
    assert!(matches!(rest, Err(Rechazo::Contradiccion { ref componente, tiempo_ms: 100 }) if componente == "saldo"));
    assert_eq!(estado, (Some(100.0), Some(90.0), true));
    assert_eq!(p.observar_rest(p.ticket(), &foto(100.0, 101))?.saldo, 100.0);
    assert!(!p.veta_entradas(), "la marca posterior y completa recupera");
    Ok(())
}

#[test]
fn inv7_rechazo_de_saldo_con_e_igual_conserva_testigo() -> Result<(), Rechazo> {
    rechazo_no_quita_testigo(Some(200), false)
}

#[test]
fn inv7_rechazo_de_saldo_con_e_omitida_conserva_testigo_tras_reconexion() -> Result<(), Rechazo> {
    rechazo_no_quita_testigo(None, true)
}

#[test]
fn control_f2_socket_nuevo_requiere_ticket_y_rest_corroborante() -> Result<(), Rechazo> {
    let p = con_testigo_ws()?;
    let viejo = p.socket();
    let anterior = p.ticket();
    let nuevo = p.abrir_socket(p.era())?;
    assert!(matches!(p.observar_rest(anterior, &foto(100.0, 100)), Err(Rechazo::OtroSocket { .. })));
    let antes_del_empate = p.ticket();
    let conflicto = p.observar_stream_del_socket(p.era(), nuevo, Some(201), &ws(100, 90.0), |_| {});
    assert!(matches!(conflicto, Err(Rechazo::Contradiccion { .. })));
    assert!(p.observar_rest(antes_del_empate, &foto(90.0, 100)).is_err());
    let _ = p.observar_rest(p.ticket(), &foto(100.0, 100));
    assert!(p.veta_entradas(), "el saldo intermedio no autoriza");
    assert_eq!(p.equity_del_freno(), Some(90.0));
    let valido = p.ticket();
    assert!(matches!(p.observar_stream_del_socket(p.era(), viejo, Some(250), &ws(101, 50.0), |_| {}),
        Err(Rechazo::OtroSocket { .. })));
    assert_eq!(p.observar_rest(valido, &foto(90.0, 100))?.saldo, 90.0);
    assert!(!p.veta_entradas());
    Ok(())
}

#[test]
fn control_cierre_confirmado_y_cero_llegan_al_freno_durante_el_hueco() -> Result<(), Rechazo> {
    let p = PuenteCuenta::new();
    let btc = ("BTCUSDT".to_string(), "BOTH".to_string());
    let mut abierta = foto(100.0, 100);
    abierta.upnl_del_activo = Some(-10.0);
    abierta.posiciones = Some(HashMap::from([(btc.clone(), PosicionRest {
        dato: DatoPosicion::Abierta { cantidad: 1.0, upnl: -10.0 }, tiempo_ms: Some(100)
    })]));
    p.observar_rest(p.ticket(), &abierta)?;
    p.perdida_socket(p.era())?;
    let mut cerrada = foto(0.0, 101);
    cerrada.posiciones = Some(HashMap::from([(btc, PosicionRest {
        dato: DatoPosicion::Cerrada, tiempo_ms: Some(101)
    })]));
    assert_eq!(p.observar_rest(p.ticket(), &cerrada)?, Observacion { saldo: 0.0, equity: 0.0 });
    assert_eq!(p.posiciones_en_cache(), 0);
    assert_eq!(p.equity_del_freno(), Some(0.0));
    assert!(p.veta_entradas());
    Ok(())
}

fn lectura(marca_saldo: u64) -> LecturaCuenta {
    LecturaCuenta {
        saldo: 100.0,
        tiempo_saldo_ms: Some(marca_saldo),
        posiciones: Some(HashMap::from([(("BTCUSDT".to_string(), "BOTH".to_string()), PosicionRest {
            dato: DatoPosicion::Abierta { cantidad: 1.0, upnl: -10.0 },
            tiempo_ms: Some(100),
        })])),
        rechazo_arranque: None,
        upnl_del_activo: Some(-10.0),
    }
}

#[test]
fn lote_ws_rechazado_por_posicion_no_impide_confirmar_saldo_vigente() -> Result<(), Rechazo> {
    let p = PuenteCuenta::new();
    p.observar_rest(p.ticket(), &lectura(99))?;
    let s = p.socket();
    let sube_marca = EventoStream { tiempo_ms: Some(100), saldo: Some(100.0), posiciones: vec![] };
    p.observar_stream_del_socket(p.era(), s, Some(200), &sube_marca, |_| {})?;
    let conflicto = EventoStream {
        tiempo_ms: Some(100),
        saldo: Some(90.0),
        posiciones: vec![(("BTCUSDT".to_string(), "BOTH".to_string()), DatoPosicion::Abierta { cantidad: 0.5, upnl: -5.0 })],
    };
    let rechazo = p.observar_stream_del_socket(p.era(), s, Some(201), &conflicto, |_| {});
    assert!(matches!(rechazo, Err(Rechazo::Contradiccion { componente, .. }) if componente.contains("BTCUSDT")));
    assert_eq!(p.observacion().map(|o| o.saldo), Some(100.0));
    assert_eq!(p.equity_del_freno(), Some(80.0));
    assert!(p.veta_entradas());
    let posterior = p.observar_rest(p.ticket(), &lectura(100));
    assert!(posterior.is_ok(), "foto completa que confirma saldo y posición vigentes: {posterior:?}");
    assert!(!p.veta_entradas());
    assert_eq!(p.equity_del_freno(), Some(90.0));
    Ok(())
}
