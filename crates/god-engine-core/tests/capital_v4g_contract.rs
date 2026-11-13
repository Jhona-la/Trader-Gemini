//! CAPITAL-UNICO V4G: el saldo dimensionable (`arena.unified_capital`, lo que lee Risk) contra el
//! saldo de la cuenta aceptado por el puente. Invariantes (seccion "V4G - DISENO" de
//! `integrador_capital_unico.md`):
//!
//! - INV-A: en toda decision de tamano, dimensionable = ultimo saldo aceptado menos lo reservado
//!   por ordenes vivas que el exchange aun no refleja. Nunca mas.
//! - INV-B: todo saldo aceptado con la admision abierta llega al dimensionable en el mismo paso;
//!   si no puede llegar, la admision se veta hasta que llegue. Bajadas y subidas.
//! - INV-C: reservar y cancelar sin ejecucion deja exactamente el saldo aceptado; tras una
//!   ejecucion, el saldo que publique el exchange. Ninguna comision dos veces.
//!
//! Piezas reales: el puente de produccion, el `GlobalArena`, `EntryReservation` (cancelar y
//! confirmar) y `drawdown::capital_del_freno`. Lo que vive en linea en el host (el publicador) y en
//! el nucleo (la reserva al abrir y el cierre local) esta en el ADAPTADOR de abajo, atado a sus
//! lineas con anclas de fuente. Importes del fixture. Comparaciones exactas: sin tolerancias.

use execution_engine::puente_cuenta::{EventoStream, LecturaCuenta, Publicacion, PuenteCuenta, Ticket};
use god_engine_core::entry_reservation::EntryReservation;
use quantum_arena::{position::PositionHorizon, symbol_registry, GlobalArena};
use risk_engine::drawdown;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const HOST: &str = include_str!("../../../src/bin/god_engine.rs");
const RISK: &str = include_str!("../../risk-engine/src/lib.rs");
const NUCLEO: &str = include_str!("../src/lib.rs");
const SIMBOLO: &str = "V4GUSDT";
const SESION: u64 = 3;
const COMISION: f64 = 0.04; // fixture: comision de entrada estimada por el nucleo

static REGISTRO: Mutex<()> = Mutex::new(());

fn texto(fuente: &str) -> String {
    fuente.replace("\r\n", "\n")
}

fn orden(fuente: &str, pasos: &[&str]) {
    let fuente = texto(fuente);
    let mut resto = fuente.as_str();
    for paso in pasos {
        resto = resto.split_once(paso).unwrap_or_else(|| panic!("ancla ausente o fuera de orden: {paso}")).1;
    }
}

// ============================ ADAPTADOR (V4G) ============================
// Lo que hacen el host y el nucleo en linea. Una vuelta que lo cambie cambia SOLO este bloque
// (y sus anclas), nunca las aserciones de abajo. Contra c001f15b este bloque era el de c001f15b
// (publicador solo en el alza, sumas atomicas del nucleo y de la reserva, confirmar sin marca):
// ver la tabla de pruebas de "INTEGRADOR - CAPITAL-UNICO V4G".

fn anclas_del_adaptador() {
    orden(HOST, &[
        "fn publicar_capital_de_sesion(",
        "(true, _) | (false, None) => false,",
        "(false, Some((saldo, marca))) => {",
        "arena.capital_incompleto.store(true, Ordering::Release);",
        "token.generation() == p.sesion",
        ".entry_publication_gate",
        ".saldo_aceptado(arena, balance, marca);",
        "espejo.store(balance.to_bits(), Ordering::Relaxed);",
        "arena.capital_incompleto.store(!suelto, Ordering::Release);",
    ]);
    orden(HOST, &[".activar(&arena_real);", "puente_cuenta.vincular_publicador(move |p| {"]);
    orden(HOST, &[
        "reservation.confirmar_llenado_con_marca(&arena_clone, Some(ejecutada), entry_executor.registry().get(&client_id).map(|o| o.marca_fill_ms).filter(|m| *m > 0))",
    ]);
    orden(NUCLEO, &[
        "let mut _publication = publication_arena.config.entry_publication_gate",
        "_publication.reservar(&self.arena, clave_reserva, fee_paid);",
        "if !target_pos.open_with_tau_and_fee(",
        "_publication.cancelar(&self.arena, clave_reserva, fee_paid);",
    ]);
    orden(NUCLEO, &[
        "let net_realized_pnl = gross_pnl - close_fee;",
        "if !self.arena.config.dimensionable_derivado.load(Ordering::Acquire) {",
        ".fetch_add(net_realized_pnl, Ordering::Relaxed);",
    ]);
}

/// El host activa el dimensionable del arena vivo antes de vincular el publicador.
fn activar(arena: &GlobalArena) {
    arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner()).activar(arena);
}

/// `publicar_capital_de_sesion` del host; `oms` es la generacion vigente del testigo del OMS.
fn publicar(oms: &AtomicU64, arena: &GlobalArena, espejo: &AtomicU64, p: &Publicacion) -> bool {
    let suelto = match (p.veta, p.vigente) {
        (true, _) | (false, None) => false,
        (false, Some((saldo, marca))) => {
            arena.capital_incompleto.store(true, Ordering::Release);
            oms.load(Ordering::SeqCst) == p.sesion && {
                arena
                    .config
                    .entry_publication_gate
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .saldo_aceptado(arena, saldo, marca);
                espejo.store(saldo.to_bits(), Ordering::Relaxed);
                true
            }
        }
    };
    arena.capital_incompleto.store(!suelto, Ordering::Release);
    suelto
}

/// La reserva del nucleo al abrir la ranura (`god-engine-core/src/lib.rs`, bajo la puerta).
fn reservar_del_nucleo(arena: &GlobalArena, slot: usize, comision: f64) -> EntryReservation {
    let pos = arena.coins[0].positions.get_slot(slot);
    let generation = pos.generation.load(Ordering::Acquire).wrapping_add(1);
    {
        let mut _publication = arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner());
        let clave = quantum_arena::dimensionable::ClaveReserva { coin_id: 0, slot, generation };
        _publication.reservar(arena, clave, comision);
        assert!(pos.open_with_tau_and_fee(
            true, 100.0, 1.0, 10.0, 1000, 101.0, 99.0, PositionHorizon::Continuous, 0.6, 0.7, comision, 30_000,
        ));
    }
    EntryReservation { coin_id: 0, slot, generation, symbol: SIMBOLO.into() }
}

/// El host confirma la reserva con lo ejecutado y la marca del exchange del ultimo fill.
fn confirmar(r: &EntryReservation, arena: &GlobalArena, ejecutada: f64, marca_fill: u64) {
    r.confirmar_llenado_con_marca(arena, Some(ejecutada), Some(marca_fill)).unwrap();
}

/// Cierre local de una posicion confirmada (`god-engine-core/src/lib.rs`, PnL neto).
fn cierre_local(arena: &GlobalArena, pnl_neto: f64) {
    if !arena.config.dimensionable_derivado.load(Ordering::Acquire) {
        arena.unified_capital.fetch_add(pnl_neto, Ordering::Relaxed);
    }
}
// ========================== FIN DEL ADAPTADOR ==========================

fn rest(saldo: f64, t: u64) -> LecturaCuenta {
    LecturaCuenta {
        saldo,
        tiempo_saldo_ms: Some(t),
        posiciones: Some(HashMap::new()),
        rechazo_arranque: None,
        upnl_del_activo: Some(0.0),
    }
}

struct Montaje {
    puente: PuenteCuenta,
    arena: Arc<GlobalArena>,
    espejo: Arc<AtomicU64>,
    oms: Arc<AtomicU64>,
}

fn registrar_simbolo() {
    quantum_arena::symbols::update_dynamic_universe(vec![SIMBOLO.into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(SIMBOLO)]);
}

/// `con_testigo`: el puente comprueba la sesion OMS (host V4F). Sin el, solo el publicador la ve.
fn montaje_con(con_testigo: bool) -> Montaje {
    registrar_simbolo();
    let arena = GlobalArena::build_in_own_stack(0.0);
    let espejo = Arc::new(AtomicU64::new(0.0f64.to_bits()));
    let oms = Arc::new(AtomicU64::new(SESION));
    let puente = PuenteCuenta::con_sesion(SESION);
    if con_testigo {
        let oms = Arc::clone(&oms);
        puente.vincular_testigo_de_sesion(move || oms.load(Ordering::SeqCst));
    }
    activar(&arena);
    {
        let (arena, espejo, oms) = (Arc::clone(&arena), Arc::clone(&espejo), Arc::clone(&oms));
        puente.vincular_publicador(move |p| publicar(&oms, &arena, &espejo, p));
    }
    puente.abrir_socket(puente.era()).unwrap();
    Montaje { puente, arena, espejo, oms }
}

fn montaje() -> Montaje {
    montaje_con(true)
}

impl Montaje {
    fn dimensionable(&self) -> f64 {
        self.arena.unified_capital.load(Ordering::Relaxed)
    }
    fn saldo(&self) -> f64 {
        self.puente.observacion().unwrap().saldo
    }
    fn admitido(&self) -> bool {
        !self.arena.capital_incompleto.load(Ordering::Acquire)
    }
    fn get(&self, ticket: Ticket, l: &LecturaCuenta) {
        self.puente.observar_rest_de_sesion(ticket, self.puente.sesion(), l).unwrap();
        self.puente
            .con_era_vigente(ticket.era(), |v| self.espejo.store(v.unwrap().saldo.to_bits(), Ordering::Relaxed))
            .unwrap();
    }
    fn rest(&self, saldo: f64, t: u64) {
        self.get(self.puente.ticket(), &rest(saldo, t));
    }
    fn stream(&self, saldo: f64, t: u64) {
        let ev = EventoStream { tiempo_ms: Some(t), saldo: Some(saldo), posiciones: Vec::new() };
        self.puente
            .observar_stream_del_socket(self.puente.era(), self.puente.socket(), Some(t), &ev, |_| {})
            .unwrap();
    }
    fn corte_y_reapertura(&self) {
        self.puente.perdida_socket(self.puente.era()).unwrap();
        assert!(!self.admitido(), "el corte veta");
        self.puente.abrir_socket(self.puente.era()).unwrap();
    }
    fn freno(&self) -> f64 {
        let e = self.puente.equity_del_freno().unwrap_or(f64::NAN);
        self.arena.capital_exchange.store(e, Ordering::Relaxed);
        drawdown::capital_del_freno(&self.arena)
    }
    /// INV-A sin reservas: el dimensionable es el saldo aceptado, bit a bit.
    fn exacto(&self, que: &str) {
        assert_eq!(self.dimensionable().to_bits(), self.saldo().to_bits(), "{que}: dimensionable {} con saldo aceptado {}", self.dimensionable(), self.saldo());
    }
}

fn anclas_de_risk() {
    orden(RISK, &[
        "let capital_vetado = arena.capital_incompleto.load(Ordering::Acquire);",
        "let current_capital = arena.unified_capital.load(Ordering::Relaxed);",
    ]);
}

// ---------------------------------------------------------------- INV-B

/// Residual (a) de V4F, ahora exigiendo lo correcto: con la admision abierta, un saldo menor
/// aceptado llega al dimensionable en el mismo paso.
#[test]
fn v4g_inv_b_saldo_menor_con_admision_abierta_llega_al_dimensionable() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    anclas_de_risk();
    let m = montaje();
    let primero = m.puente.ticket();
    let segundo = m.puente.ticket();
    m.get(primero, &rest(100.0, 100));
    assert!(m.admitido());
    m.exacto("primer saldo");
    m.get(segundo, &rest(90.0, 200));
    assert_eq!(m.saldo(), 90.0);
    assert_eq!(m.freno(), 90.0);
    assert!(m.admitido());
    m.exacto("segundo GET, saldo menor");
}

/// Subidas por el mismo camino (el pedido no las deja fuera).
#[test]
fn v4g_inv_b_saldo_mayor_con_admision_abierta_llega_al_dimensionable() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let m = montaje();
    m.rest(100.0, 100);
    m.rest(110.0, 200);
    assert!(m.admitido());
    m.exacto("subida");
}

/// Dos saldos seguidos del stream: cada uno llega en su paso.
#[test]
fn v4g_inv_b_dos_saldos_seguidos_del_stream() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let m = montaje();
    m.rest(100.0, 100);
    m.stream(95.0, 200);
    assert!(m.admitido());
    m.exacto("primer evento");
    m.stream(92.5, 300);
    m.exacto("segundo evento");
}

/// Si el saldo aceptado no puede llegar (el OMS ya es otra sesion y el puente no lo sabe), la
/// admision se veta: no se sigue dimensionando con el saldo anterior.
#[test]
fn v4g_inv_b_si_no_puede_llegar_la_admision_se_veta() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let m = montaje_con(false);
    m.rest(100.0, 100);
    assert!(m.admitido());
    m.oms.store(SESION + 1, Ordering::SeqCst);
    m.rest(90.0, 200);
    assert!(!m.admitido(), "el saldo 90 no llego y la admision siguio abierta con {}", m.dimensionable());
    // Control: de vuelta a la sesion de la era, el siguiente cambio lo entrega y suelta.
    m.oms.store(SESION, Ordering::SeqCst);
    m.rest(90.0, 300);
    assert!(m.admitido());
    m.exacto("tras volver la sesion");
}

// ---------------------------------------------------------------- INV-C

/// Residual (b) de V4F, ahora exigiendo lo correcto: reserva viva durante corte, reapertura y alza
/// (la orden no se ejecuto). El alza cuenta la reserva viva y la cancelacion la quita: el
/// dimensionable vuelve a ser exactamente el saldo aceptado.
#[test]
fn v4g_inv_c_cancelacion_tras_el_alza_vuelve_al_saldo_exacto() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let m = montaje();
    m.rest(100.0, 100);
    let r = reservar_del_nucleo(&m.arena, 0, COMISION);
    assert_eq!(m.dimensionable(), 100.0 - COMISION);
    m.corte_y_reapertura();
    m.rest(100.0, 300);
    assert!(m.admitido());
    assert_eq!(m.dimensionable(), 100.0 - COMISION, "el alza cuenta la reserva viva");
    r.cancel(&m.arena).unwrap();
    m.exacto("cancelacion tras el alza");
    assert!(m.dimensionable() <= m.saldo(), "RESIDUAL (b): dimensionable {} con {} en la cuenta", m.dimensionable(), m.saldo());
}

/// Control de (b): la orden se ejecuto; el exchange publica el saldo con su comision real y el
/// dimensionable es ese saldo (la reserva sale al aceptarlo).
#[test]
fn v4g_inv_c_con_la_orden_ejecutada_es_el_saldo_del_exchange() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let m = montaje();
    m.rest(100.0, 100);
    let r = reservar_del_nucleo(&m.arena, 0, COMISION);
    confirmar(&r, &m.arena, 1.0, 250);
    m.corte_y_reapertura();
    m.rest(100.0 - COMISION, 300);
    m.exacto("alza tras la ejecucion");
}

/// Saldo nuevo aceptado con una reserva viva, y la cancelacion que llega despues de ese saldo.
#[test]
fn v4g_inv_a_saldo_aceptado_con_reserva_viva_y_cancelacion_posterior() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let m = montaje();
    m.rest(100.0, 100);
    let r = reservar_del_nucleo(&m.arena, 0, COMISION);
    m.rest(90.0, 200); // funding: la orden sigue sin ejecutarse
    assert_eq!(m.dimensionable(), 90.0 - COMISION, "INV-A: saldo aceptado menos la reserva viva");
    r.cancel(&m.arena).unwrap();
    m.exacto("cancelacion despues del saldo nuevo");
    // La repeticion no lo sube (idempotente: la ranura ya no es de esa generacion).
    assert!(r.cancel(&m.arena).is_err());
    m.exacto("cancelacion repetida");
}

/// Ejecucion parcial: queda reservada solo la comision de lo ejecutado; el saldo del exchange que
/// la incluye (con la comision real, distinta de la estimada) es el dimensionable.
#[test]
fn v4g_inv_c_ejecucion_parcial() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let m = montaje();
    m.rest(100.0, 100);
    let r = reservar_del_nucleo(&m.arena, 0, COMISION);
    confirmar(&r, &m.arena, 0.4, 150);
    assert!(m.dimensionable() < 100.0, "la parte ejecutada sigue reservada hasta que el exchange la refleje");
    assert!(m.dimensionable() > 100.0 - COMISION, "la parte no ejecutada ya no esta reservada");
    m.rest(99.99, 150);
    m.exacto("saldo del exchange tras el parcial");
    // Confirmacion repetida: no vuelve a escalar ni acredita nada.
    r.confirmar_llenado_con_marca(&m.arena, Some(0.2), Some(150)).unwrap();
    m.exacto("confirmacion repetida");
    // Variante: el saldo ya neto del parcial llega ANTES del ack. La devolucion del remanente no
    // acredita dinero sobre ese saldo.
    let m = montaje();
    m.rest(100.0, 100);
    let r = reservar_del_nucleo(&m.arena, 0, COMISION);
    m.stream(99.99, 150);
    assert!(m.dimensionable() <= m.saldo());
    confirmar(&r, &m.arena, 0.4, 150);
    m.exacto("saldo neto antes del ack del parcial");
}

/// Comision del nucleo y despues el saldo del exchange que ya la incluye (stream antes del ack):
/// sin doble resta, y con la comision que cobro el exchange (0,05), no la estimada (0,04).
#[test]
fn v4g_inv_c_sin_doble_resta_de_la_comision() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    // a) el saldo del exchange llega antes de que el host confirme el fill.
    let m = montaje();
    m.rest(100.0, 100);
    let r = reservar_del_nucleo(&m.arena, 0, COMISION);
    m.stream(99.95, 150);
    assert!(m.dimensionable() <= m.saldo(), "INV-A antes de la confirmacion");
    confirmar(&r, &m.arena, 1.0, 150);
    m.exacto("stream antes del ack");
    // b) el host confirma antes del saldo.
    let m = montaje();
    m.rest(100.0, 100);
    let r = reservar_del_nucleo(&m.arena, 0, COMISION);
    confirmar(&r, &m.arena, 1.0, 150);
    assert_eq!(m.dimensionable(), 100.0 - COMISION, "ejecutada y aun no reflejada");
    m.stream(99.95, 150);
    m.exacto("ack antes del stream");
}

/// Reserva viva durante el relevo de sesion: el OMS rehusa el relevo con la ranura sin confirmar
/// (`host_oms.rs`, `ocupada_locked`); la cancelacion deja el saldo exacto y el relevo siguiente
/// abre la era nueva, cuyo alza publica el saldo de esa sesion.
#[test]
fn v4g_inv_c_reserva_viva_durante_el_relevo_de_sesion() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    orden(include_str!("../../../src/host_oms.rs"), &[
        "fn ocupada_locked(",
        ".any(|slot| slot.is_open() && !slot.exchange_confirmed.load(Ordering::Acquire))",
    ]);
    let m = montaje();
    m.rest(100.0, 100);
    let r = reservar_del_nucleo(&m.arena, 0, COMISION);
    let relevo = |m: &Montaje| {
        m.puente.relevar_sesion(|| {
            let ocupada = m.arena.coins.iter().any(|c| {
                c.positions.slots().iter().any(|s| s.is_open() && !s.exchange_confirmed.load(Ordering::Acquire))
            });
            if ocupada {
                ((), None)
            } else {
                m.oms.store(SESION + 1, Ordering::SeqCst);
                ((), Some(SESION + 1))
            }
        })
    };
    assert_eq!(relevo(&m).1, None, "el OMS rehusa con la reserva viva");
    assert!(m.admitido());
    assert_eq!(m.dimensionable(), 100.0 - COMISION);
    r.cancel(&m.arena).unwrap();
    m.exacto("cancelada la reserva del relevo");
    // Reserva ejecutada antes del relevo, y saldo de la era nueva que ya la incluye.
    let r = reservar_del_nucleo(&m.arena, 1, COMISION);
    confirmar(&r, &m.arena, 1.0, 250);
    assert!(relevo(&m).1.is_some());
    assert!(!m.admitido(), "la era nueva veta hasta su saldo");
    m.puente.abrir_socket(m.puente.era()).unwrap();
    m.rest(100.0 - 0.05, 300);
    assert!(m.admitido());
    m.exacto("alza de la sesion nueva");
}

/// INV-A con un cierre local con ganancia: el PnL local no sube el dimensionable por encima del
/// saldo aceptado; llega con el saldo del exchange.
#[test]
fn v4g_inv_a_cierre_local_no_sube_el_dimensionable_sobre_la_cuenta() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let m = montaje();
    m.rest(100.0, 100);
    let r = reservar_del_nucleo(&m.arena, 0, COMISION);
    confirmar(&r, &m.arena, 1.0, 150);
    m.stream(100.0 - COMISION, 150);
    m.exacto("entrada reflejada");
    cierre_local(&m.arena, 5.0);
    assert!(m.dimensionable() <= m.saldo(), "dimensionable {} con {} en la cuenta", m.dimensionable(), m.saldo());
    m.stream(100.0 - COMISION + 5.0, 400);
    m.exacto("cierre reflejado por el exchange");
}

/// Control: sin activar el libro (backtest, replay, simulador) el arena sigue siendo el acumulador
/// de siempre: la reserva resta la comision, la cancelacion la devuelve y el parcial devuelve la
/// parte no pagada, y el cierre local suma su PnL.
#[test]
fn control_sin_activar_el_arena_sigue_siendo_acumulador() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    registrar_simbolo();
    let arena = GlobalArena::build_in_own_stack(100.0);
    assert!(!arena.config.dimensionable_derivado.load(Ordering::Acquire));
    let r = reservar_del_nucleo(&arena, 0, COMISION);
    assert_eq!(arena.unified_capital.load(Ordering::Relaxed), 100.0 - COMISION);
    r.cancel(&arena).unwrap();
    assert_eq!(arena.unified_capital.load(Ordering::Relaxed), 100.0 - COMISION + COMISION);
    let antes = arena.unified_capital.load(Ordering::Relaxed);
    let r = reservar_del_nucleo(&arena, 1, COMISION);
    confirmar(&r, &arena, 0.4, 150);
    let despues = arena.unified_capital.load(Ordering::Relaxed);
    assert!(despues > antes - COMISION && despues < antes, "el parcial devuelve la parte no pagada: {despues}");
    let antes = arena.unified_capital.load(Ordering::Relaxed);
    cierre_local(&arena, 5.0);
    assert_eq!(arena.unified_capital.load(Ordering::Relaxed), antes + 5.0);
}

// ---------------------------------------------------------------- RED de Codex QRA (QRA-CAPITAL-V4G-TRES-RED)
// `qra_v4g_causal.rs` (19:48), adoptadas con las piezas reales del arena y el adaptador del host.
// Mismas aserciones e importes. Las dos del libro siguen abiertas (pasan a V4H, INV-E, segun el
// aviso del coordinador de las 19:32): se marcan `ignore` con su motivo para que la suite se
// pueda correr; la del publicador se cierra aqui.

const QRA_INICIAL: f64 = 100.0;
const QRA_COMISION_TOTAL: f64 = 0.75;
const QRA_COMISION_PARCIAL: f64 = 0.25;
const QRA_CLAVE: quantum_arena::dimensionable::ClaveReserva =
    quantum_arena::dimensionable::ClaveReserva { coin_id: 0, slot: 0, generation: 1 };

fn qra_libro_activo(arena: &GlobalArena) -> std::sync::MutexGuard<'_, quantum_arena::dimensionable::LibroDimensionable> {
    let mut libro = arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner());
    libro.activar(arena);
    libro
}

#[test]
#[ignore = "RED de Codex QRA abierta: T igual no prueba inclusion (pasa a V4H, INV-E)"]
fn qra_red_mismo_t_parcial_y_final_no_libera_comision_sin_cobertura() {
    let mut cortes = Vec::new();
    for observacion_antes_de_confirmar in [false, true] {
        let arena = GlobalArena::build_in_own_stack(0.0);
        let mut libro = qra_libro_activo(&arena);
        libro.saldo_aceptado(&arena, QRA_INICIAL, Some(100));
        libro.reservar(&arena, QRA_CLAVE, QRA_COMISION_TOTAL);
        let saldo_parcial = QRA_INICIAL - QRA_COMISION_PARCIAL;
        if observacion_antes_de_confirmar {
            libro.saldo_aceptado(&arena, saldo_parcial, Some(200));
        }
        libro.ejecutar(&arena, QRA_CLAVE, 0.0, Some(200));
        if !observacion_antes_de_confirmar {
            libro.saldo_aceptado(&arena, saldo_parcial, Some(200));
        }
        let maximo_con_cobertura = QRA_INICIAL - QRA_COMISION_TOTAL;
        let observado = arena.unified_capital.load(Ordering::Relaxed);
        cortes.push((observacion_antes_de_confirmar, observado, maximo_con_cobertura, libro.reservado()));
    }
    assert!(
        cortes.iter().all(|(_, observado, maximo, _)| observado <= maximo),
        "saldo parcial T=200, fill final T=200; (saldo_antes_del_ack, dimensionable, maximo, reservado)={cortes:?}"
    );
}

#[test]
#[ignore = "RED de Codex QRA abierta: fill sin T y marca no correlacionada (pasa a V4H, INV-E)"]
fn qra_red_fill_sin_t_no_libera_comision_por_una_marca_no_correlacionada() {
    let arena = GlobalArena::build_in_own_stack(0.0);
    let mut libro = qra_libro_activo(&arena);
    libro.saldo_aceptado(&arena, QRA_INICIAL, Some(100));
    libro.reservar(&arena, QRA_CLAVE, QRA_COMISION_TOTAL);
    libro.ejecutar(&arena, QRA_CLAVE, 0.0, None);
    libro.saldo_aceptado(&arena, QRA_INICIAL, Some(101));
    let observado = arena.unified_capital.load(Ordering::Relaxed);
    assert!(
        observado <= QRA_INICIAL - QRA_COMISION_TOTAL,
        "fill sin T, saldo no correlacionado T=101: dimensionable {observado}, reservado {}",
        libro.reservado()
    );
}

/// El publicador espera la puerta con un saldo menor ya aceptado: mientras tanto Risk no admite
/// con el saldo anterior. Sin plazos: el hilo del GET deja huella (veto o espejo) antes de
/// bloquearse en la puerta, y esa huella es la que se espera.
#[test]
fn qra_red_risk_no_admite_saldo_anterior_si_publicador_espera_puerta() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    anclas_de_risk();
    let nuevo_saldo = 90.0;
    let m = Arc::new(montaje());
    m.rest(QRA_INICIAL, 100);
    let lectura_risk = |m: &Montaje| (m.arena.capital_incompleto.load(Ordering::Acquire), m.dimensionable());
    assert_eq!(lectura_risk(&m), (false, QRA_INICIAL));

    let retenida = m.arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner());
    let m_get = Arc::clone(&m);
    let get = std::thread::spawn(move || {
        let ticket = m_get.puente.ticket();
        m_get.puente.observar_rest_de_sesion(ticket, SESION, &rest(nuevo_saldo, 200)).map(|o| o.saldo)
    });
    loop {
        let espejo = f64::from_bits(m.espejo.load(Ordering::Relaxed));
        if m.arena.capital_incompleto.load(Ordering::Acquire) || espejo == nuevo_saldo || get.is_finished() {
            break;
        }
        std::thread::yield_now();
    }
    let lectura_durante_publicacion = lectura_risk(&m);
    let espejo_durante_publicacion = f64::from_bits(m.espejo.load(Ordering::Relaxed));
    drop(retenida);
    let observacion = get.join().expect("el GET no entro en panico").expect("el GET se acepta");

    assert_eq!(f64::from_bits(m.espejo.load(Ordering::Relaxed)), nuevo_saldo);
    assert_eq!(observacion, nuevo_saldo);
    assert_eq!(lectura_risk(&m), (false, nuevo_saldo), "control: despues del commit");
    let (vetado, dimensionable) = lectura_durante_publicacion;
    assert!(
        vetado || dimensionable <= nuevo_saldo,
        "Risk durante la escritura: vetado={vetado}, dimensionable={dimensionable}, espejo={espejo_durante_publicacion}, saldo entregado al publicador={nuevo_saldo}"
    );
}
