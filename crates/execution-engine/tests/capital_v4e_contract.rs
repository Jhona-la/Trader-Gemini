//! CAPITAL-UNICO V4E: pruebas del integrador.
//!
//! - INV-7 (variante sin prueba de `qoder_p1d_lectura.md`): un lote del stream
//!   RECHAZADO con la misma `T` y `E` igual u omitida no borra el ultimo saldo
//!   aceptado del stream. Si lo borra, una foto REST atrasada con esa misma `T`
//!   (el saldo de antes de los fills) cierra el empate y suelta el veto.

#[path = "../src/puente_cuenta.rs"]
#[allow(dead_code)]
mod puente_cuenta;

use puente_cuenta::{EventoStream, LecturaCuenta, PuenteCuenta, Rechazo};
use std::collections::HashMap;

fn rest(saldo: f64, t: u64) -> LecturaCuenta {
    LecturaCuenta {
        saldo,
        tiempo_saldo_ms: Some(t),
        posiciones: Some(HashMap::new()),
        rechazo_arranque: None,
        upnl_del_activo: Some(0.0),
    }
}

fn ws(p: &PuenteCuenta, t: u64, e: Option<u64>, saldo: f64) -> Result<(), Rechazo> {
    let ev = EventoStream { tiempo_ms: Some(t), saldo: Some(saldo), posiciones: vec![] };
    p.observar_stream_del_socket(p.era(), p.socket(), e, &ev, |_| {}).map(|_| ())
}

fn saldo(p: &PuenteCuenta) -> Option<f64> {
    p.observacion().map(|o| o.saldo)
}

// ---- INV-7 ------------------------------------------------------------------

#[test]
fn v4e_inv7_rechazo_con_e_igual_no_borra_el_saldo_aceptado() {
    let p = PuenteCuenta::new();
    p.observar_rest(p.ticket(), &rest(100.0, 100)).unwrap();
    assert!(!p.veta_entradas());
    ws(&p, 200, Some(201), 99.0).unwrap();
    // Mismo T y misma E con otro contenido: contradiccion (aceptado = false).
    let r = ws(&p, 200, Some(201), 98.0);
    assert!(matches!(r, Err(Rechazo::Contradiccion { .. })), "{r:?}");
    assert!(p.veta_entradas());
    // Foto atrasada con la misma T (el saldo de antes de los fills), pedida despues.
    let r = p.observar_rest(p.ticket(), &rest(100.0, 200));
    assert!(
        matches!(r, Err(Rechazo::Contradiccion { .. })),
        "INV-7: el rechazo borro el ultimo saldo aceptado y la foto atrasada se aplico: {r:?}"
    );
    assert!(p.veta_entradas(), "la foto atrasada no suelta el veto");
    assert_eq!(saldo(&p), Some(99.0));
    assert!(p.equity_del_freno().is_some_and(|f| f <= 98.0), "el freno ve el menor");
    // La REST que confirma el saldo aceptado lo suelta.
    assert_eq!(p.observar_rest(p.ticket(), &rest(99.0, 200)).unwrap().saldo, 99.0);
    assert!(!p.veta_entradas());
}

#[test]
fn v4e_inv7_rechazo_con_e_omitida_no_borra_el_saldo_aceptado() {
    let p = PuenteCuenta::new();
    p.observar_rest(p.ticket(), &rest(100.0, 100)).unwrap();
    ws(&p, 200, Some(201), 99.0).unwrap();
    // Una REST con la misma T y otro valor deja el empate abierto.
    let r = p.observar_rest(p.ticket(), &rest(97.0, 200));
    assert!(matches!(r, Err(Rechazo::Contradiccion { .. })), "{r:?}");
    // Evento sin E con la misma T y el empate abierto: contradiccion (aceptado = false).
    let r = ws(&p, 200, None, 98.0);
    assert!(matches!(r, Err(Rechazo::Contradiccion { .. })), "{r:?}");
    let r = p.observar_rest(p.ticket(), &rest(100.0, 200));
    assert!(
        matches!(r, Err(Rechazo::Contradiccion { .. })),
        "INV-7: el rechazo sin E borro el ultimo saldo aceptado: {r:?}"
    );
    assert!(p.veta_entradas());
    assert_eq!(saldo(&p), Some(99.0));
    assert_eq!(p.observar_rest(p.ticket(), &rest(99.0, 200)).unwrap().saldo, 99.0);
    assert!(!p.veta_entradas());
}

#[test]
fn v4e_inv7_control_la_rest_posterior_sigue_arbitrando_al_valor_rechazado() {
    // Contrato V4B (`v4b_mismas_marcas_con_otro_contenido_pide_reconciliacion`):
    // ante dos contenidos con las mismas marcas, la REST posterior decide, tambien
    // a favor del contenido que el stream rechazo.
    let p = PuenteCuenta::new();
    p.observar_rest(p.ticket(), &rest(100.0, 100)).unwrap();
    ws(&p, 200, Some(201), 99.0).unwrap();
    assert!(ws(&p, 200, Some(201), 98.0).is_err());
    assert_eq!(p.observar_rest(p.ticket(), &rest(98.0, 200)).unwrap().saldo, 98.0);
    assert!(!p.veta_entradas());
}

// ---- P1-b (INV-5): el veto solo se suelta publicando el saldo -----------------

use puente_cuenta::Publicacion;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// El saldo y el veto que leen Risk y el OMS, con el orden de Risk en V4E:
/// adquiere el veto y despues lee el saldo.
struct Arena {
    saldo: AtomicU64,
    veto: AtomicBool,
}

impl Arena {
    fn nueva(saldo: f64) -> Arc<Self> {
        Arc::new(Self { saldo: AtomicU64::new(saldo.to_bits()), veto: AtomicBool::new(true) })
    }
    fn saldo_local(&self) -> f64 {
        f64::from_bits(self.saldo.load(Ordering::Relaxed))
    }
    fn saldo_si_admite(&self) -> Option<f64> {
        if self.veto.load(Ordering::Acquire) {
            return None;
        }
        let s = self.saldo_local();
        (s.is_finite() && s > 0.0).then_some(s)
    }
}

/// Publicador con la forma del host (`publicar_capital_de_sesion`): en el alza
/// publica el saldo y despues suelta el veto. `sesion_oms`: la sesion vigente
/// del OMS; si no es la de la era, no publica ni suelta. Comprueba con el lector
/// intercalado dentro del paso que nadie admite antes de tener el saldo nuevo.
fn vincular(p: &PuenteCuenta, a: &Arc<Arena>, sesion_oms: Arc<AtomicU64>) -> Arc<Mutex<Vec<Publicacion>>> {
    let vistas = Arc::new(Mutex::new(Vec::new()));
    let (arena, registro) = (Arc::clone(a), Arc::clone(&vistas));
    p.vincular_publicador(move |pb| {
        registro.lock().unwrap().push(*pb);
        // V4G (INV-B): como el host, publica el saldo vigente en CADA cambio sin veto; las
        // comprobaciones del lector intercalado valen en el alza (antes de soltar el veto).
        let suelto = match (pb.veta, pb.vigente) {
            (true, _) | (false, None) => false,
            (false, Some((s, _))) => {
                if sesion_oms.load(Ordering::Relaxed) != pb.sesion {
                    false
                } else {
                    let alza = pb.saldo.is_some();
                    if alza {
                        assert_eq!(arena.saldo_si_admite(), None, "nadie admite antes del alza");
                    }
                    arena.saldo.store(s.to_bits(), Ordering::Relaxed);
                    if alza {
                        assert_eq!(arena.saldo_si_admite(), None, "ni entre publicar y soltar");
                    }
                    true
                }
            }
        };
        arena.veto.store(!suelto, Ordering::Release);
        suelto
    });
    vistas
}

fn transicion(saldo_anterior: f64) -> (PuenteCuenta, Arc<Arena>, Arc<Mutex<Vec<Publicacion>>>, Arc<AtomicU64>) {
    let p = PuenteCuenta::new();
    let a = Arena::nueva(saldo_anterior);
    let sesion = Arc::new(AtomicU64::new(0));
    let vistas = vincular(&p, &a, Arc::clone(&sesion));
    let era = p.nueva_era();
    sesion.store(p.sesion(), Ordering::Relaxed);
    p.reconexion(era).unwrap();
    assert!(a.veto.load(Ordering::Acquire));
    (p, a, vistas, sesion)
}

#[test]
fn v4e_p1b_saldo_positivo_menor_se_publica_antes_de_soltar_el_veto() {
    let (p, a, vistas, _) = transicion(100.0);
    p.observar_rest(p.ticket(), &rest(10.0, 100)).unwrap();
    assert_eq!(a.saldo_si_admite(), Some(10.0), "se admite con el saldo nuevo, nunca con el anterior");
    let alzas: Vec<f64> = vistas.lock().unwrap().iter().filter_map(|v| v.saldo).collect();
    assert_eq!(alzas, vec![10.0], "el saldo se ofrece una vez, en el alza");
    // Ya admitido: un evento posterior no es otro alza (`saldo` es solo del alza)...
    ws(&p, 200, Some(201), 9.0).unwrap();
    let ultima = *vistas.lock().unwrap().last().unwrap();
    assert_eq!((ultima.veta, ultima.saldo), (false, None));
    // ...pero V4G (INV-B, residual (a) de V4E): su saldo llega al dimensionable en el mismo paso.
    // Antes esta linea fijaba el defecto: `Some(10.0)` con 9 en la cuenta.
    assert_eq!(ultima.vigente.map(|(s, _)| s), Some(9.0));
    assert_eq!(a.saldo_si_admite(), Some(9.0));
}

#[test]
fn v4e_p1b_cero_de_cuenta_vacia_se_publica_y_la_admision_lo_veta() {
    // Paso 4: sin atajo. El puente no veta un saldo valido; el alza publica 0 y
    // la regla de Risk (saldo > 0) no admite. El freno ve el 0.
    let (p, a, _, _) = transicion(100.0);
    p.observar_rest(p.ticket(), &rest(0.0, 100)).unwrap();
    assert!(!p.veta_entradas(), "el puente no veta el cero (contratos V3 y V4B)");
    assert_eq!(a.saldo_local(), 0.0, "el cero se publica");
    assert!(!a.veto.load(Ordering::Acquire));
    assert_eq!(a.saldo_si_admite(), None, "la cuenta vacia no hereda capacidad de entrada");
    assert_eq!(p.equity_del_freno(), Some(0.0), "el cero llega al freno");
}

#[test]
fn v4e_p1b_sin_publicar_no_se_suelta_y_el_alza_se_vuelve_a_ofrecer() {
    let (p, a, vistas, sesion) = transicion(100.0);
    sesion.store(p.sesion() + 1, Ordering::Relaxed); // el OMS ya cambio de sesion
    p.observar_rest(p.ticket(), &rest(10.0, 100)).unwrap();
    assert!(a.veto.load(Ordering::Acquire), "sin publicar el saldo, el veto sigue");
    assert_eq!(a.saldo_local(), 100.0);
    sesion.store(p.sesion(), Ordering::Relaxed);
    ws(&p, 200, Some(201), 9.0).unwrap();
    let ultima = *vistas.lock().unwrap().last().unwrap();
    assert_eq!(ultima.saldo, Some(9.0), "el cambio siguiente vuelve a ofrecer el alza");
    assert_eq!(a.saldo_si_admite(), Some(9.0));
}

#[test]
fn v4e_p1b_el_observador_de_solo_veto_nunca_recibe_false() {
    let p = PuenteCuenta::new();
    let falsos = Arc::new(AtomicUsize::new(0));
    let verdaderos = Arc::new(AtomicUsize::new(0));
    let (f, v) = (Arc::clone(&falsos), Arc::clone(&verdaderos));
    p.vincular_veto(move |veta| {
        if veta { v.fetch_add(1, Ordering::Relaxed) } else { f.fetch_add(1, Ordering::Relaxed) };
    });
    p.observar_rest(p.ticket(), &rest(100.0, 100)).unwrap();
    assert!(!p.veta_entradas());
    p.reconexion(p.era()).unwrap();
    p.observar_rest(p.ticket(), &rest(100.0, 100)).unwrap();
    assert_eq!(falsos.load(Ordering::Relaxed), 0, "soltar el veto exige publicar el saldo");
    assert!(verdaderos.load(Ordering::Relaxed) >= 2);
}

#[test]
fn v4e_p1b_contrato_de_fuente_del_host_y_de_risk() {
    let host = include_str!("../../../src/bin/god_engine.rs").replace("\r\n", "\n");
    // Un solo escritor del veto en el host: el publicador del puente.
    // V4G (asercion adaptada, misma intencion): el publicador escribe el veto dos veces (lo alza
    // mientras el saldo aceptado espera la puerta y lo resuelve al final); todas las escrituras del
    // host siguen dentro de `publicar_capital_de_sesion`.
    assert!(!host.contains("puente_cuenta.vincular_veto("), "el host no usa el observador de solo veto");
    let f = &host[host.find("fn publicar_capital_de_sesion(").unwrap()..];
    let f = &f[..f.find("\nfn ").unwrap()];
    assert_eq!(host.matches("capital_incompleto").count(), f.matches("capital_incompleto").count());
    assert_eq!(f.matches("capital_incompleto").count(), 2);
    let sesion = f.find("token.generation() == p.sesion").expect("la sesion de la era");
    let publica = f.find("oms.publish_current_balance(&token, saldo,").expect("el saldo por el testigo del OMS");
    // V4G (ancla adaptada): el saldo llega al arena por el libro del dimensionable.
    let arena = f.find(".saldo_aceptado(arena, balance, marca);").expect("el saldo del arena");
    let veto = f.find("arena.capital_incompleto.store(!suelto, Ordering::Release)").expect("el veto");
    assert!(sesion < publica && publica < arena && arena < veto, "saldo antes que veto, en el mismo paso");
    let risk = include_str!("../../risk-engine/src/lib.rs").replace("\r\n", "\n");
    let ev = &risk[risk.find("fn evaluate_quantum_order_impl(").unwrap()..];
    let veto = ev.find("let capital_vetado = arena.capital_incompleto.load(Ordering::Acquire);").unwrap();
    let saldo = ev.find("let current_capital = arena.unified_capital.load(").unwrap();
    let decision = ev.find("if capital_vetado {").unwrap();
    let pico = ev.find("if arena.pico_ilegible.load(").unwrap();
    assert!(veto < saldo, "Risk adquiere el veto antes de leer el saldo");
    assert!(pico < decision, "la decision sigue en el mismo sitio (mismo orden de rechazos)");
}

// ---- P1-c (INV-6): testigo comun de era y sesion OMS ---------------------------

#[test]
fn v4e_p1c_lectura_de_otra_sesion_no_toca_nada() {
    let p = PuenteCuenta::new();
    p.observar_rest(p.ticket(), &rest(100.0, 100)).unwrap();
    p.reconexion(p.era()).unwrap();
    let antes = (p.observacion(), p.equity_del_freno(), p.veta_entradas());
    let r = p.observar_rest_de_sesion(p.ticket(), p.sesion() + 1, &rest(50.0, 200));
    assert!(matches!(r, Err(Rechazo::OtraSesion { .. })), "{r:?}");
    assert_eq!((p.observacion(), p.equity_del_freno(), p.veta_entradas()), antes);
    assert!(p.tomar_diagnostico().ultimo_rechazo.unwrap().contains("OtraSesion"));
    // La de su sesion si.
    p.observar_rest_de_sesion(p.ticket(), p.sesion(), &rest(50.0, 200)).unwrap();
    assert!(!p.veta_entradas());
}

#[test]
fn v4e_p1c_la_era_nueva_queda_ligada_a_la_sesion_publicada() {
    let p = PuenteCuenta::con_sesion(7);
    assert_eq!((p.sesion(), p.ticket().sesion()), (7, 7));
    let era = p.nueva_era_de_sesion(9);
    assert_eq!((p.era(), p.sesion(), p.ticket().sesion()), (era, 9, 9));
    assert!(matches!(
        p.observar_rest_de_sesion(p.ticket(), 7, &rest(10.0, 1)),
        Err(Rechazo::OtraSesion { sesion: 7, vigente: 9 })
    ));
    assert_eq!(p.observacion(), None);
}

#[test]
fn v4e_p1c_contrato_de_fuente_host_resincronizador_y_publicador() {
    let host = include_str!("../../../src/bin/god_engine.rs").replace("\r\n", "\n");
    // Ningun camino de produccion usa la lectura sin testigo.
    assert!(!host.contains(".observar_rest(ticket"), "el host lee siempre con la sesion del ejecutor");
    assert!(!host.contains("exec_rs.load_full()"), "el GET periodico no carga el ejecutor suelto");
    let rs = &host[host.find("let puente_rs = Arc::clone(&puente_cuenta);").unwrap()..];
    let rs = &rs[..rs.find("match &r {").unwrap()];
    let token = rs.find("let token = oms_rs.snapshot_token();").unwrap();
    let ejecutor = rs.find("let executor = Arc::clone(token.executor());").unwrap();
    let sesion = rs.find("token.generation(),").unwrap();
    assert!(token < ejecutor && ejecutor < sesion);
    assert!(host.contains("PuenteCuenta::con_sesion(\n        oms_session.snapshot_token().generation(),"));
    // V4F (ancla adaptada): la era de la sesion publicada la abre el relevo.
    assert!(host.contains("let (paso, era_del_relevo) = puente_cuenta.relevar_sesion(|| {"));
    assert!(host.contains("let Some(era_mainnet) = era_del_relevo else {"));
    assert!(host.contains("observar_rest_de_sesion(ticket, balance_token.generation(), l)"));
    assert!(host.contains("observar_rest_de_sesion(ticket, token_arranque.generation(), &l)"));
    let helper = include_str!("../src/resincronizar_cuenta.rs").replace("\r\n", "\n");
    let cuerpo = &helper[helper.find("pub async fn resincronizar_una_vez").unwrap()..];
    let cuerpo = &cuerpo[..cuerpo.find("pub struct Pausa").unwrap()];
    assert!(cuerpo.contains("puente.observar_rest_de_sesion(ticket, sesion, &l)"));
    assert!(!cuerpo.contains("puente.observar_rest(ticket"));
}