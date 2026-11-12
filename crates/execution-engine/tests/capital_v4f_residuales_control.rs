//! CAPITAL-UNICO V4F: controles documentados de los residuales (a) y (b) de V4E, para que Codex
//! QRA los dictamine. NO cambian el comportamiento: lo fijan con su efecto contable exacto. Si
//! alguien corrige un residual, su control falla y hay que reescribirlo con el dictamen.
//!
//! Piezas reales: el puente de produccion, el `GlobalArena` y `drawdown::capital_del_freno`. El
//! publicador es el de `publicar_capital_de_sesion` del host (vive en el binario) y las dos
//! escrituras del nucleo y de la reserva son las mismas sumas atomicas del codigo; las anclas de
//! fuente de cada prueba atan el modelo a esas lineas. Importes del fixture.
//!
//! Lo que mide Risk para dimensionar es `arena.unified_capital` (`risk-engine/src/lib.rs`, ancla
//! abajo); el freno mide con `capital_del_freno` (minimo de arena y equity del puente).

#[path = "../src/puente_cuenta.rs"]
#[allow(dead_code)]
mod puente_cuenta;

use puente_cuenta::{LecturaCuenta, Publicacion, PuenteCuenta};
use quantum_arena::GlobalArena;
use risk_engine::drawdown;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

const HOST: &str = include_str!("../../../src/bin/god_engine.rs");
const RISK: &str = include_str!("../../risk-engine/src/lib.rs");
const NUCLEO: &str = include_str!("../../god-engine-core/src/lib.rs");
const RESERVA: &str = include_str!("../../god-engine-core/src/entry_reservation.rs");

const SESION: u64 = 3;

fn rest(saldo: f64, t: u64) -> LecturaCuenta {
    LecturaCuenta {
        saldo,
        tiempo_saldo_ms: Some(t),
        posiciones: Some(HashMap::new()),
        rechazo_arranque: None,
        upnl_del_activo: Some(0.0),
    }
}

fn orden(fuente: &str, pasos: &[&str]) {
    let mut resto = fuente;
    for paso in pasos {
        resto = resto.split_once(paso).unwrap_or_else(|| panic!("ancla ausente o fuera de orden: {paso}")).1;
    }
}

/// `publicar_capital_de_sesion` del host, con el OMS en la sesion de la era.
fn publicar(arena: &GlobalArena, espejo: &AtomicU64, p: &Publicacion) -> bool {
    let suelto = match (p.veta, p.saldo) {
        (true, _) => false,
        (false, None) => return true,
        (false, Some(saldo)) => {
            espejo.store(saldo.to_bits(), Ordering::Relaxed);
            arena.unified_capital.store(saldo, Ordering::Relaxed);
            true
        }
    };
    arena.capital_incompleto.store(!suelto, Ordering::Release);
    suelto
}

fn anclas_del_publicador_y_de_risk() {
    orden(HOST, &[
        "fn publicar_capital_de_sesion(",
        "(true, _) => false,",
        "(false, None) => return true,",
        "(false, Some(saldo)) => {",
        "espejo.store(balance.to_bits(), Ordering::Relaxed);",
        "arena.unified_capital.store(balance, Ordering::Relaxed);",
        "arena.capital_incompleto.store(!suelto, Ordering::Release);",
    ]);
    orden(RISK, &[
        "let capital_vetado = arena.capital_incompleto.load(Ordering::Acquire);",
        "let current_capital = arena.unified_capital.load(Ordering::Relaxed);",
    ]);
    // El puente entrega saldo solo en el alza.
    orden(include_str!("../src/puente_cuenta.rs"), &[
        "let alza = !veta && !self.admitido.load(Ordering::Relaxed);",
        "saldo: if alza { e.saldo } else { None },",
    ]);
}

struct Montaje {
    puente: PuenteCuenta,
    arena: Arc<GlobalArena>,
    espejo: Arc<AtomicU64>,
}

fn montaje() -> Montaje {
    let arena = GlobalArena::build_in_own_stack(0.0);
    let espejo = Arc::new(AtomicU64::new(0.0f64.to_bits()));
    let puente = PuenteCuenta::con_sesion(SESION);
    puente.vincular_testigo_de_sesion(|| SESION);
    {
        let arena = Arc::clone(&arena);
        let espejo = Arc::clone(&espejo);
        puente.vincular_publicador(move |p| publicar(&arena, &espejo, p));
    }
    puente.abrir_socket(puente.era()).unwrap();
    Montaje { puente, arena, espejo }
}

impl Montaje {
    /// Lo que el inmune deja en el arena cada 5 s (`capital_exchange` = equity del freno).
    fn ciclo_del_inmune(&self) {
        let e = self.puente.equity_del_freno().unwrap_or(f64::NAN);
        self.arena.capital_exchange.store(e, Ordering::Relaxed);
    }
    fn dimensionable(&self) -> f64 {
        self.arena.unified_capital.load(Ordering::Relaxed)
    }
    fn espejo(&self) -> f64 {
        f64::from_bits(self.espejo.load(Ordering::Relaxed))
    }
    fn admitido(&self) -> bool {
        !self.arena.capital_incompleto.load(Ordering::Acquire)
    }
    /// GET periodico (resincronizador): ticket, lectura y espejo bajo el mutex.
    fn get(&self, ticket: puente_cuenta::Ticket, l: &LecturaCuenta) {
        let obs = self.puente.observar_rest_de_sesion(ticket, SESION, l).unwrap();
        self.puente
            .con_era_vigente(ticket.era(), |v| self.espejo.store(v.unwrap_or(obs).saldo.to_bits(), Ordering::Relaxed))
            .unwrap();
    }
}

/// Residual (a). Dos GET en vuelo con marcas crecientes: el primero (saldo 100) habilita; el
/// segundo observa 90 (una baja del exchange que el arena no contabilizo, p. ej. funding).
/// Efecto exacto: el exchange y el freno dicen 90, Risk dimensiona con 100. Hay 10 USDT (el 11,1 %
/// del saldo real) en el capital dimensionable que no existen en la cuenta.
#[test]
fn control_residual_a_saldo_menor_con_admision_abierta_no_llega_al_dimensionable() {
    anclas_del_publicador_y_de_risk();
    let m = montaje();
    let primero = m.puente.ticket();
    let segundo = m.puente.ticket();
    m.get(primero, &rest(100.0, 100));
    assert!(m.admitido());
    assert_eq!((m.dimensionable(), m.espejo()), (100.0, 100.0));

    m.get(segundo, &rest(90.0, 200));
    m.ciclo_del_inmune();
    let saldo_real = m.puente.observacion().unwrap().saldo;
    assert_eq!(saldo_real, 90.0, "el puente acepta el saldo nuevo");
    assert_eq!(m.espejo(), 90.0, "el espejo lo ve");
    assert_eq!(drawdown::capital_del_freno(&m.arena), 90.0, "el freno lo ve");
    assert!(m.admitido(), "la admision sigue abierta");
    assert_eq!(m.dimensionable(), 100.0, "Risk dimensiona con el saldo anterior");
    let inexistente = m.dimensionable() - saldo_real;
    assert_eq!(inexistente, 10.0);
    assert!(inexistente > 0.0, "RESIDUAL (a): se dimensiona con {inexistente} USDT que la cuenta no tiene");

    // Control del sentido contrario: una subida tampoco llega (conservador).
    let m = montaje();
    m.get(m.puente.ticket(), &rest(100.0, 100));
    m.get(m.puente.ticket(), &rest(110.0, 200));
    assert_eq!(m.dimensionable(), 100.0);
}

const COMISION: f64 = 0.04; // fixture: comision de entrada de la reserva

fn anclas_de_la_reserva() {
    orden(NUCLEO, &[
        "let _publication = publication_arena.config.entry_publication_gate",
        "self.arena.unified_capital.fetch_add(-fee_paid, Ordering::Relaxed);",
    ]);
    orden(RESERVA, &[
        "pub fn cancel(&self, arena: &GlobalArena)",
        "arena.unified_capital.fetch_add(fee, Ordering::AcqRel);",
        "pub fn confirmar_llenado(",
    ]);
}

/// Residual (b). Reserva en vuelo durante un alza tras un corte. Con saldo 100 admitido, el
/// nucleo reserva una entrada y descuenta su comision (99,96). El socket se corta (veto), se
/// reabre y la REST del alza publica el saldo del exchange, 100: la orden no llego a ejecutarse.
/// Despues la reserva se cancela y devuelve su comision SOBRE el saldo recien publicado.
/// Efecto exacto: dimensionable 100,04 con 100 en la cuenta. Sobran 0,04 USDT (la comision).
#[test]
fn control_residual_b_cancelacion_tras_el_alza_devuelve_la_comision_sobre_el_saldo_publicado() {
    anclas_del_publicador_y_de_risk();
    anclas_de_la_reserva();
    let m = montaje();
    m.get(m.puente.ticket(), &rest(100.0, 100));
    assert_eq!(m.dimensionable(), 100.0);

    m.arena.unified_capital.fetch_add(-COMISION, Ordering::Relaxed); // reserva del nucleo
    let con_reserva = m.dimensionable();
    assert!((con_reserva - 99.96).abs() < 1e-9);

    m.puente.perdida_socket(m.puente.era()).unwrap();
    assert!(!m.admitido());
    m.puente.abrir_socket(m.puente.era()).unwrap();
    m.get(m.puente.ticket(), &rest(100.0, 300));
    assert!(m.admitido());
    assert_eq!(m.dimensionable(), 100.0, "el alza publica el saldo del exchange");

    m.arena.unified_capital.fetch_add(COMISION, Ordering::AcqRel); // cancelacion de la reserva
    let saldo_real = m.puente.observacion().unwrap().saldo;
    let inexistente = m.dimensionable() - saldo_real;
    assert_eq!(saldo_real, 100.0);
    assert!((m.dimensionable() - 100.04).abs() < 1e-9, "dimensionable {}", m.dimensionable());
    assert!((inexistente - COMISION).abs() < 1e-9);
    m.ciclo_del_inmune();
    assert_eq!(drawdown::capital_del_freno(&m.arena), 100.0, "el freno mide 100 (minimo de los dos)");
}

/// Control de (b): si la orden si se ejecuto, el exchange ya cobro la comision; el alza publica
/// 99,96 y la reserva se confirma sin devolucion. No sobra nada.
#[test]
fn control_residual_b_con_la_orden_ejecutada_no_sobra_nada() {
    anclas_de_la_reserva();
    let m = montaje();
    m.get(m.puente.ticket(), &rest(100.0, 100));
    m.arena.unified_capital.fetch_add(-COMISION, Ordering::Relaxed);
    m.puente.perdida_socket(m.puente.era()).unwrap();
    m.puente.abrir_socket(m.puente.era()).unwrap();
    m.get(m.puente.ticket(), &rest(100.0 - COMISION, 300));
    assert!((m.dimensionable() - (100.0 - COMISION)).abs() < 1e-9);
    assert!((m.dimensionable() - m.puente.observacion().unwrap().saldo).abs() < 1e-9);
}
