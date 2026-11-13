//! CAPITAL-UNICO V4F: controles de los residuales (a) y (b) de V4E. V4G: dejan de fijar el defecto
//! y exigen las invariantes (INV-A, INV-B, INV-C de "V4G - DISENO"): el dimensionable nunca supera
//! el saldo aceptado; un saldo aceptado con la admision abierta llega en el mismo paso; reservar y
//! cancelar deja exactamente el saldo aceptado. Contra c001f15b fallan (los numeros del defecto
//! estan en cada prueba).
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
use quantum_arena::dimensionable::ClaveReserva;
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

/// `publicar_capital_de_sesion` del host, con el OMS en la sesion de la era (V4G: en cada cambio
/// sin veto entrega el saldo aceptado al libro del dimensionable; contra c001f15b solo publicaba
/// en el alza con `arena.unified_capital.store`).
fn publicar(arena: &GlobalArena, espejo: &AtomicU64, p: &Publicacion) -> bool {
    let suelto = match (p.veta, p.vigente) {
        (true, _) | (false, None) => false,
        (false, Some((saldo, marca))) => {
            arena.capital_incompleto.store(true, Ordering::Release);
            arena
                .config
                .entry_publication_gate
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .saldo_aceptado(arena, saldo, marca);
            espejo.store(saldo.to_bits(), Ordering::Relaxed);
            true
        }
    };
    arena.capital_incompleto.store(!suelto, Ordering::Release);
    suelto
}

fn anclas_del_publicador_y_de_risk() {
    orden(HOST, &[
        "fn publicar_capital_de_sesion(",
        "(true, _) | (false, None) => false,",
        "(false, Some((saldo, marca))) => {",
        "arena.capital_incompleto.store(true, Ordering::Release);",
        ".saldo_aceptado(arena, balance, marca);",
        "espejo.store(balance.to_bits(), Ordering::Relaxed);",
        "arena.capital_incompleto.store(!suelto, Ordering::Release);",
    ]);
    orden(HOST, &[".activar(&arena_real);", "puente_cuenta.vincular_publicador(move |p| {"]);
    orden(RISK, &[
        "let capital_vetado = arena.capital_incompleto.load(Ordering::Acquire);",
        "let current_capital = arena.unified_capital.load(Ordering::Relaxed);",
    ]);
    // V4G: el puente entrega el saldo vigente en cada cambio sin veto (antes, solo en el alza).
    orden(include_str!("../src/puente_cuenta.rs"), &[
        "let alza = !veta && !self.admitido.load(Ordering::Relaxed);",
        "saldo: if alza { e.saldo } else { None },",
        "vigente: if veta { None } else { e.saldo.map(|s| (s, e.marca_saldo)) },",
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
    arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner()).activar(&arena);
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
    // ADAPTADOR (V4G): la reserva del nucleo y su cancelacion o ejecucion pasan por el libro del
    // dimensionable, bajo la puerta (anclas en `anclas_de_la_reserva`). Contra c001f15b eran las
    // sumas atomicas de `god-engine-core` (`fetch_add(-fee)`, `fetch_add(fee)` y nada al ejecutar).
    fn reservar(&self, comision: f64) -> ClaveReserva {
        let clave = ClaveReserva { coin_id: 0, slot: 0, generation: 1 };
        self.libro().reservar(&self.arena, clave, comision);
        clave
    }
    fn cancelar(&self, clave: ClaveReserva) {
        self.libro().cancelar(&self.arena, clave, COMISION);
    }
    fn ejecutar(&self, clave: ClaveReserva, marca_fill: u64) {
        self.libro().ejecutar(&self.arena, clave, 0.0, Some(marca_fill));
    }
    fn libro(&self) -> std::sync::MutexGuard<'_, quantum_arena::dimensionable::LibroDimensionable> {
        self.arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner())
    }
    // FIN DEL ADAPTADOR
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

/// Residual (a), INV-B. Dos GET en vuelo con marcas crecientes: el primero (saldo 100) habilita; el
/// segundo observa 90 (una baja del exchange que el arena no contabilizo, p. ej. funding). Exige:
/// el dimensionable es 90 en el mismo paso (en c001f15b quedaba en 100: 10 USDT que la cuenta no
/// tiene). La subida llega por el mismo camino (en c001f15b quedaba en 100).
#[test]
fn inv_b_residual_a_saldo_menor_con_admision_abierta_llega_al_dimensionable() {
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
    let inexistente = m.dimensionable() - saldo_real;
    assert!(inexistente <= 0.0, "RESIDUAL (a): se dimensiona con {inexistente} USDT que la cuenta no tiene");
    assert_eq!(m.dimensionable(), 90.0, "Risk dimensiona con el saldo aceptado");

    // Sentido contrario: la subida llega por el mismo camino.
    let m = montaje();
    m.get(m.puente.ticket(), &rest(100.0, 100));
    m.get(m.puente.ticket(), &rest(110.0, 200));
    assert_eq!(m.dimensionable(), 110.0);
}

const COMISION: f64 = 0.04; // fixture: comision de entrada de la reserva

fn anclas_de_la_reserva() {
    // V4G: reserva, cancelacion y ejecucion por el libro, bajo la puerta (contra c001f15b eran
    // `self.arena.unified_capital.fetch_add(-fee_paid, ..)` y `arena.unified_capital.fetch_add(fee, ..)`).
    orden(NUCLEO, &[
        "let mut _publication = publication_arena.config.entry_publication_gate",
        "_publication.reservar(&self.arena, clave_reserva, fee_paid);",
    ]);
    orden(RESERVA, &[
        "pub fn cancel(&self, arena: &GlobalArena)",
        ".entry_publication_gate",
        ".cancelar(arena, self.clave(), fee);",
        "pub fn confirmar_llenado_con_marca(",
        ".ejecutar(arena, self.clave(), devuelta, marca_fill);",
    ]);
}

/// Residual (b), INV-C. Reserva en vuelo durante un alza tras un corte. Con saldo 100 admitido, el
/// nucleo reserva una entrada (comision 0,04). El socket se corta (veto), se reabre y la REST del
/// alza trae el saldo del exchange, 100: la orden no llego a ejecutarse. Exige: el alza cuenta la
/// reserva viva (100 - 0,04) y la cancelacion la quita: el dimensionable vuelve EXACTAMENTE a 100
/// (en c001f15b el alza dejaba 100 y la cancelacion sumaba 0,04 encima: 100,04 con 100 en la cuenta).
#[test]
fn inv_c_residual_b_cancelacion_tras_el_alza_vuelve_al_saldo_exacto() {
    anclas_del_publicador_y_de_risk();
    anclas_de_la_reserva();
    let m = montaje();
    m.get(m.puente.ticket(), &rest(100.0, 100));
    assert_eq!(m.dimensionable(), 100.0);

    let reserva = m.reservar(COMISION);
    assert_eq!(m.dimensionable(), 100.0 - COMISION);

    m.puente.perdida_socket(m.puente.era()).unwrap();
    assert!(!m.admitido());
    m.puente.abrir_socket(m.puente.era()).unwrap();
    m.get(m.puente.ticket(), &rest(100.0, 300));
    assert!(m.admitido());
    assert_eq!(m.dimensionable(), 100.0 - COMISION, "el alza cuenta la reserva viva");

    m.cancelar(reserva);
    let saldo_real = m.puente.observacion().unwrap().saldo;
    let inexistente = m.dimensionable() - saldo_real;
    assert_eq!(saldo_real, 100.0);
    assert!(inexistente <= 0.0, "RESIDUAL (b): dimensionable {} con {saldo_real} en la cuenta", m.dimensionable());
    assert_eq!(m.dimensionable().to_bits(), 100.0f64.to_bits(), "dimensionable {}", m.dimensionable());
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
    let reserva = m.reservar(COMISION);
    m.ejecutar(reserva, 250);
    m.puente.perdida_socket(m.puente.era()).unwrap();
    m.puente.abrir_socket(m.puente.era()).unwrap();
    m.get(m.puente.ticket(), &rest(100.0 - COMISION, 300));
    assert_eq!(m.dimensionable(), 100.0 - COMISION);
    assert_eq!(m.dimensionable().to_bits(), m.puente.observacion().unwrap().saldo.to_bits());
}
