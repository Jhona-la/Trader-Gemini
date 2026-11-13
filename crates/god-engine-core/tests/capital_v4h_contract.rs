//! CAPITAL-UNICO V4H: las invariantes INV-D a INV-G (seccion "V4H - DISENO" de
//! `integrador_capital_unico.md`) sobre el saldo dimensionable (`arena.unified_capital`, lo que
//! lee Risk). INV-A, INV-B e INV-C siguen en `capital_v4g_contract.rs`.
//!
//! - INV-D: el saldo vigente es el de MAYOR marca del exchange; marca menor no mueve nada; marca
//!   igual o sin marca, el menor.
//! - INV-E: un debito sale solo con un saldo de marca ESTRICTAMENTE mayor que la de sus fills.
//! - INV-F: la decision se revalida al reservar y al enviar.
//! - INV-G: debitos locales no observados restan en el acto; ningun credito suma; cierre
//!   externo veta hasta el saldo siguiente.
//!
//! Piezas reales: el puente de produccion, el `GlobalArena` con su libro, `EntryReservation` y el
//! `OrderRegistry`. Lo que vive en linea en el host, el stream, el OMS y el nucleo esta en el
//! ADAPTADOR de abajo, atado a sus lineas con anclas de fuente. Una vuelta que lo cambie cambia
//! SOLO ese bloque; contra 4a652d76 (V4G) el adaptador era el de V4G (copia en
//! `Desktop\PC-Turbo\integrador\v4h_logs\red_v4g\`). Importes del fixture; sin tolerancias.

use execution_engine::puente_cuenta::{EventoStream, LecturaCuenta, Publicacion, PuenteCuenta, Ticket};
use god_engine_core::entry_reservation::EntryReservation;
use quantum_arena::dimensionable::ClaveReserva;
use quantum_arena::{position::PositionHorizon, symbol_registry, GlobalArena};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const HOST: &str = include_str!("../../../src/bin/god_engine.rs");
const OMS: &str = include_str!("../../../src/host_oms.rs");
const RISK: &str = include_str!("../../risk-engine/src/lib.rs");
const NUCLEO: &str = include_str!("../src/lib.rs");
const RESERVA: &str = include_str!("../src/entry_reservation.rs");
const RECONCILIACION: &str = include_str!("../../execution-engine/src/reconciliation.rs");
const STREAM: &str = include_str!("../../execution-engine/src/user_data_stream.rs");
const LIBRO: &str = include_str!("../../quantum-arena/src/dimensionable.rs");
const SIMBOLO: &str = "V4HUSDT";
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

// ============================ ADAPTADOR (V4H) ============================

fn anclas_del_adaptador() {
    orden(HOST, &[
        "fn publicar_capital_de_sesion(",
        "if p.veta {",
        ".veto_del_puente();",
        "arena.capital_incompleto.store(true, Ordering::Release);",
        "token.generation() == p.sesion",
        ".entry_publication_gate",
        "libro.fijar_alcance(p.era, p.sesion);",
        "libro.saldo_aceptado(arena, balance, marca);",
        "espejo.store(balance.to_bits(), Ordering::Relaxed);",
        "arena.capital_incompleto.store(!suelto, Ordering::Release);",
    ]);
    orden(NUCLEO, &[
        "let capital_de_la_decision = self.risk_engine.capital_evaluado();",
        "let mut _publication = publication_arena.config.entry_publication_gate",
        "|| !_publication.admite(&publication_arena, capital_de_la_decision) {",
        "continue;",
        "_publication.reservar_con_base(&self.arena, clave_reserva, fee_paid, capital_de_la_decision);",
        "if !target_pos.open_with_tau_and_fee(",
        "_publication.cancelar(&self.arena, clave_reserva, fee_paid);",
    ]);
    orden(NUCLEO, &[
        "let was_exchange_confirmed =",
        "let net_realized_pnl = gross_pnl - close_fee;",
        "if !self.arena.config.dimensionable_derivado.load(Ordering::Acquire) {",
        ".fetch_add(net_realized_pnl, Ordering::Relaxed);",
        "} else if was_exchange_confirmed && net_realized_pnl < 0.0 {",
        ".debito_cierre_local(",
        "slot: slot_idx,",
        "generation: position_generation,",
        "-net_realized_pnl,",
    ]);
    orden(RISK, &[
        "let capital_vetado = arena.capital_incompleto.load(Ordering::Acquire);",
        "let current_capital = arena.unified_capital.load(Ordering::Relaxed);",
        "self.capital_evaluado = Some(current_capital);",
        "let current_cap = arena.unified_capital.load(Ordering::Relaxed);",
        "self.capital_evaluado = self.capital_evaluado.map(|c| c.max(current_cap));",
    ]);
    orden(STREAM, &[
        "#[serde(rename = \"R\", default)]",
        "#[serde(rename = \"rp\", default, deserialize_with = \"crate::order_types::string_or_f64\")]",
        "if new_fill && (o.reduce_only || close_kind.is_some() || o.realized_profit != 0.0) {",
        "update.commission_asset.eq_ignore_ascii_case(\"USDT\")",
        "let externo = coid.starts_with(\"autoclose\") || coid.starts_with(\"adl_autoclose\");",
        "OrderStatus::Filled | OrderStatus::Canceled | OrderStatus::Expired",
        "libro.fill_de_cierre(",
        "comision_usdt - o.realized_profit,",
        "update.trade_time_ms,",
        "if externo {",
        "libro.cierre_externo(arena);",
        "drop(libro);",
        "let _ = self.puente.evidencia_perdida(self.era);",
    ]);
    orden(HOST, &[
        "\"FUNDING_FEE\" => a.funding += e.income,",
        "if e.income_type == \"FUNDING_FEE\" && e.income < 0.0 {",
        "libro.debito_funding(&arena_brk, id, -e.income, e.time);",
    ]);
    orden(OMS, &[
        "let mut _publication = arena",
        "let arena_report = reconciliation::reconcile_arena_checked(&positions, arena, now_ms);",
        "let mut cierre_externo = arena_report.cierres_externos > 0;",
        "_publication.cierre_externo(arena);",
        "for &coin_id in &arena_report.monedas_planas {",
        "cierre_externo |= _publication.moneda_plana(arena, coin_id);",
        "let outcome = self.apply_snapshot(&token, positions, orders, fees, arena, now_ms());",
        "if outcome.cierre_externo {",
        "if let Some(f) = self.al_cierre_externo.get() {",
    ]);
    orden(HOST, &[
        "oms_session.vincular_cierre_externo(move || {",
        "let _ = puente_cierre.evidencia_perdida(puente_cierre.era());",
    ]);
    orden(OMS, &[
        "fn revalidar_capital(&self) -> Result<(), String> {",
        ".envio_valido_en(self.arena, clave);",
        "async fn submit_entry(&self, request: &EntryRequest<'_>) -> Result<(), String> {",
        "let _permit = self.session.admit_transport(self.token, self.arena)?;",
        "self.revalidar_capital()?;",
        "self.transport.submit_entry(request).await",
    ]);
    orden(HOST, &[
        "entry_oms.dispatch_entry_reservada(",
        "Some(reservation.clave()),",
    ]);
    orden(RECONCILIACION, &[
        "report.monedas_planas.push(coin_idx);",
        "for slot in local_open {",
        "report.cierres_externos += 1;",
    ]);
}

fn activar(arena: &GlobalArena) {
    arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner()).activar(arena);
}

/// `publicar_capital_de_sesion` del host; `oms` es la generacion vigente del testigo del OMS.
fn publicar(oms: &AtomicU64, arena: &GlobalArena, espejo: &AtomicU64, p: &Publicacion) -> bool {
    if p.veta {
        arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner()).veto_del_puente();
    }
    let suelto = match (p.veta, p.vigente) {
        (true, _) | (false, None) => false,
        (false, Some((saldo, marca))) => {
            arena.capital_incompleto.store(true, Ordering::Release);
            oms.load(Ordering::SeqCst) == p.sesion && {
                let mut libro = arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner());
                libro.fijar_alcance(p.era, p.sesion);
                libro.saldo_aceptado(arena, saldo, marca);
                drop(libro);
                espejo.store(saldo.to_bits(), Ordering::Relaxed);
                true
            }
        }
    };
    arena.capital_incompleto.store(!suelto, Ordering::Release);
    suelto
}

/// La reserva del nucleo bajo la puerta, con el dimensionable con el que Risk decidio. `None`:
/// la revalidacion la descarto (sin reserva ni orden).
fn reservar_del_nucleo(arena: &GlobalArena, slot: usize, comision: f64, capital: Option<f64>) -> Option<EntryReservation> {
    let pos = arena.coins[0].positions.get_slot(slot);
    let generation = pos.generation.load(Ordering::Acquire).wrapping_add(1);
    let mut _publication = arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner());
    if !_publication.admite(arena, capital) {
        return None;
    }
    let clave = ClaveReserva { coin_id: 0, slot, generation };
    _publication.reservar_con_base(arena, clave, comision, capital);
    assert!(pos.open_with_tau_and_fee(
        true, 100.0, 1.0, 10.0, 1000, 101.0, 99.0, PositionHorizon::Continuous, 0.6, 0.7, comision, 30_000,
    ));
    Some(EntryReservation { coin_id: 0, slot, generation, symbol: SIMBOLO.into() })
}

/// El guardia de transporte del OMS justo antes de la llamada al exchange.
fn enviar(arena: &GlobalArena, r: &EntryReservation) -> bool {
    arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner()).envio_valido_en(arena, r.clave())
}

/// Cierre local del nucleo (`god-engine-core/src/lib.rs`): PnL neto de una posicion.
fn cierre_local(arena: &GlobalArena, clave: ClaveReserva, pnl_neto: f64, existio_en_exchange: bool) {
    if !arena.config.dimensionable_derivado.load(Ordering::Acquire) {
        arena.unified_capital.fetch_add(pnl_neto, Ordering::Relaxed);
    } else if existio_en_exchange && pnl_neto < 0.0 {
        arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner()).debito_cierre_local(arena, clave, -pnl_neto);
    }
}

/// Un fill de cierre del stream (`user_data_stream.rs`).
fn fill_de_cierre(m: &Montaje, trade_id: u64, comision: f64, pnl_realizado: f64, t: u64, terminal: bool, externo: bool) {
    let mut libro = m.arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner());
    libro.fill_de_cierre(&m.arena, 0, trade_id, comision - pnl_realizado, t, terminal);
    if externo {
        libro.cierre_externo(&m.arena);
    }
    drop(libro);
    if externo {
        let _ = m.puente.evidencia_perdida(m.puente.era());
    }
}

/// Una fila FUNDING_FEE del historial de ingresos (host, `FEE-BREAKER`).
fn funding(arena: &GlobalArena, id: u64, income: f64, t: u64) {
    if income < 0.0 {
        arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner()).debito_funding(arena, id, -income, t);
    }
}

/// `apply_snapshot` del OMS tras reconciliar, y despues (`refresh`, sin candados) la accion del
/// host: evidencia perdida al puente.
fn reconciliacion(m: &Montaje, cierres_externos: usize, monedas_planas: &[usize]) {
    let mut _publication = m.arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner());
    let mut cierre_externo = cierres_externos > 0;
    if cierre_externo {
        _publication.cierre_externo(&m.arena);
    }
    for &coin_id in monedas_planas {
        cierre_externo |= _publication.moneda_plana(&m.arena, coin_id);
    }
    drop(_publication);
    if cierre_externo {
        let _ = m.puente.evidencia_perdida(m.puente.era());
    }
}

/// INV-G en el libro: el veto propio y el cierre externo.
fn vetado(libro: &quantum_arena::dimensionable::LibroDimensionable) -> bool {
    libro.vetado()
}
fn cierre_externo_libro(libro: &mut quantum_arena::dimensionable::LibroDimensionable, arena: &GlobalArena) {
    libro.cierre_externo(arena);
}

/// El alcance (era, sesion) de la publicacion, como lo fija el publicador.
fn fijar_alcance(libro: &mut quantum_arena::dimensionable::LibroDimensionable, era: u64, sesion: u64) {
    libro.fijar_alcance(era, sesion);
}

/// Modo unico: lo que el nucleo llama al reservar y la unica escritura del libro activo.
const RESERVA_DEL_NUCLEO: &str = "_publication.reservar_con_base(&self.arena, clave_reserva, fee_paid, capital_de_la_decision);";
const ESCRITURA_DEL_LIBRO_ACTIVO: &str = "if!self.derivado{return;}ifletSome(d)=self.dimensionable(){arena.";

/// INV-E: el alcance con el que el libro registro la reserva.
fn alcance_de(arena: &GlobalArena, clave: ClaveReserva) -> Option<(u64, u64)> {
    arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner()).alcance_de(clave).flatten()
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

fn montaje() -> Montaje {
    registrar_simbolo();
    let arena = GlobalArena::build_in_own_stack(0.0);
    let espejo = Arc::new(AtomicU64::new(0.0f64.to_bits()));
    let oms = Arc::new(AtomicU64::new(SESION));
    let puente = PuenteCuenta::con_sesion(SESION);
    {
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
    /// Lo que lee Risk: el veto y, si esta suelto, el dimensionable.
    fn capital_de_risk(&self) -> Option<f64> {
        let vetado = self.arena.capital_incompleto.load(Ordering::Acquire);
        let c = self.arena.unified_capital.load(Ordering::Relaxed);
        (!vetado).then_some(c)
    }
    fn get(&self, ticket: Ticket, l: &LecturaCuenta) {
        let _ = self.puente.observar_rest_de_sesion(ticket, self.puente.sesion(), l);
        let _ = self.puente.con_era_vigente(ticket.era(), |v| {
            if let Some(v) = v {
                self.espejo.store(v.saldo.to_bits(), Ordering::Relaxed)
            }
        });
    }
    fn rest(&self, saldo: f64, t: u64) {
        self.get(self.puente.ticket(), &rest(saldo, t));
    }
    fn stream(&self, saldo: f64, t: u64) {
        let ev = EventoStream { tiempo_ms: Some(t), saldo: Some(saldo), posiciones: Vec::new() };
        let _ = self.puente.observar_stream_del_socket(self.puente.era(), self.puente.socket(), Some(t), &ev, |_| {});
    }
    fn reconectar(&self) {
        self.puente.perdida_socket(self.puente.era()).unwrap();
        assert!(!self.admitido(), "el corte veta");
        self.puente.abrir_socket(self.puente.era()).unwrap();
    }
    fn exacto(&self, que: &str) {
        assert_eq!(self.dimensionable().to_bits(), self.saldo().to_bits(), "{que}: dimensionable {} con saldo aceptado {}", self.dimensionable(), self.saldo());
    }
    fn reservar(&self, slot: usize, comision: f64) -> EntryReservation {
        let capital = self.capital_de_risk();
        reservar_del_nucleo(&self.arena, slot, comision, capital).expect("la reserva cabe")
    }
}

fn clave(slot: usize, generation: u64) -> ClaveReserva {
    ClaveReserva { coin_id: 0, slot, generation }
}

// ---------------------------------------------------------------- RED de Codex QRA (TRES-RED)
// `qra_v4g_causal.rs` (19:48), mismas aserciones e importes; ya sin `ignore`.

const QRA_INICIAL: f64 = 100.0;
const QRA_COMISION_TOTAL: f64 = 0.75;
const QRA_COMISION_PARCIAL: f64 = 0.25;
const QRA_CLAVE: ClaveReserva = ClaveReserva { coin_id: 0, slot: 0, generation: 1 };

fn qra_libro_activo(arena: &GlobalArena) -> std::sync::MutexGuard<'_, quantum_arena::dimensionable::LibroDimensionable> {
    let mut libro = arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner());
    libro.activar(arena);
    libro
}

#[test]
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
/// con el saldo anterior (mismo intercalado que `capital_v4g_contract`, aqui con el publicador
/// de V4H). Y la propuesta que Risk calculo con el saldo anterior no se reserva despues.
#[test]
fn qra_red_risk_no_admite_saldo_anterior_si_publicador_espera_puerta() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let nuevo_saldo = 90.0;
    let m = Arc::new(montaje());
    m.rest(QRA_INICIAL, 100);
    let lectura_risk = |m: &Montaje| (m.arena.capital_incompleto.load(Ordering::Acquire), m.dimensionable());
    assert_eq!(lectura_risk(&m), (false, QRA_INICIAL));
    // Propuesta preparada con la revision anterior.
    let capital_previo = m.capital_de_risk();

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
    // INV-F: la propuesta de 100 se descarta al reservar con 90 vigente.
    assert!(reservar_del_nucleo(&m.arena, 0, COMISION, capital_previo).is_none(), "propuesta de {capital_previo:?} reservada con {nuevo_saldo} vigente");
    // Control: recalculada con la revision nueva, cabe.
    assert!(reservar_del_nucleo(&m.arena, 0, COMISION, m.capital_de_risk()).is_some());
}

// ---------------------------------------------------------------- Casos de reserva de Codex
// `qra_p1b_residual.md` y `qra_p1b_residual_v4g_cases.rs`, adoptados con INV-E: el saldo que
// "ya incluye" el fill trae marca estrictamente mayor que la del fill.

const FEE: f64 = 0.5;

#[test]
fn qra_parcial_con_saldo_ya_neto_antes_del_ack_no_acredita_dinero() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    for marca_conocida in [true, false] {
        let m = montaje();
        m.rest(100.0, 100);
        let r = m.reservar(0, FEE);
        assert_eq!(m.dimensionable(), 100.0 - FEE);
        let fee_ejecutada = FEE * 0.25;
        m.reconectar();
        m.rest(100.0 - fee_ejecutada, 300);
        let marca = marca_conocida.then_some(250);
        r.confirmar_llenado_con_marca(&m.arena, Some(0.25), marca).unwrap();
        let tras_ack = m.dimensionable();
        r.confirmar_llenado_con_marca(&m.arena, Some(0.25), marca).unwrap();
        assert_eq!(m.dimensionable(), tras_ack, "ack repetido vuelve a abonar");
        assert!(tras_ack <= m.saldo(), "la devolucion del remanente acredita sobre un saldo que nunca la pago: {tras_ack}");
        if marca_conocida {
            m.exacto("saldo T=300 posterior al fill T=250");
        } else {
            assert_eq!(tras_ack, m.saldo() - fee_ejecutada, "sin marca del fill no hay prueba de inclusion: se conserva");
        }
    }
}

#[test]
fn qra_relevo_rehusa_reserva_viva_y_no_mezcla_eras() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let m = montaje();
    m.rest(100.0, 100);
    let r = m.reservar(0, FEE);
    let ocupada = |m: &Montaje| m.arena.coins.iter().any(|c| c.positions.slots().iter().any(|s| s.is_open() && !s.exchange_confirmed.load(Ordering::Acquire)));
    let relevo = |m: &Montaje| {
        m.puente.relevar_sesion(|| {
            if ocupada(m) {
                ((), None)
            } else {
                m.oms.store(SESION + 1, Ordering::SeqCst);
                ((), Some(SESION + 1))
            }
        })
    };
    let era = m.puente.era();
    assert_eq!(relevo(&m).1, None);
    assert_eq!(m.puente.era(), era);
    assert_eq!(m.dimensionable(), 100.0 - FEE);
    r.cancel(&m.arena).unwrap();
    assert!(relevo(&m).1.is_some());
    assert!(!m.admitido());
    m.puente.abrir_socket(m.puente.era()).unwrap();
    m.rest(80.0, 300);
    assert!(r.cancel(&m.arena).is_err());
    assert!(m.admitido());
    m.exacto("saldo de la era nueva");
}

#[test]
fn qra_fee_del_nucleo_y_wallet_ws_contabilizan_una_vez() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let neto = 100.0 - FEE;
    for ws_primero in [true, false] {
        let m = montaje();
        m.rest(100.0, 100);
        let r = m.reservar(0, FEE);
        if ws_primero {
            m.stream(neto, 200);
            r.confirmar_llenado_con_marca(&m.arena, Some(1.0), Some(150)).unwrap();
        } else {
            r.confirmar_llenado_con_marca(&m.arena, Some(1.0), Some(150)).unwrap();
            m.stream(neto, 200);
        }
        assert_eq!(m.dimensionable(), neto, "doble fee, WS primero={ws_primero}");
    }
    // Con la MISMA T en el fill y en el WS no hay prueba de inclusion: se conserva (nunca mas).
    let m = montaje();
    m.rest(100.0, 100);
    let r = m.reservar(0, FEE);
    m.stream(neto, 200);
    r.confirmar_llenado_con_marca(&m.arena, Some(1.0), Some(200)).unwrap();
    assert_eq!(m.dimensionable(), neto - FEE, "T igual: el debito se conserva");
    // El siguiente cambio de la cuenta (marca mayor) lo retira.
    m.stream(neto, 201);
    m.exacto("saldo posterior a la T del fill");
}

#[test]
fn qra_dos_saldos_actualizan_el_dimensionable_o_vetan_en_el_mismo_paso() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    for segundo in [80.0, 105.0] {
        let m = montaje();
        let (a, b) = (m.puente.ticket(), m.puente.ticket());
        m.get(a, &rest(100.0, 100));
        m.get(b, &rest(segundo, 200));
        assert!(!m.admitido() || m.dimensionable() == segundo, "REST {segundo}, dimensionable {}", m.dimensionable());
    }
}

#[test]
fn qra_cancelacion_tras_saldo_nuevo_solo_quita_la_reserva() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let m = montaje();
    m.rest(100.0, 100);
    let r = m.reservar(0, FEE);
    m.reconectar();
    m.rest(80.0, 300);
    assert_eq!(m.dimensionable(), 80.0 - FEE);
    r.cancel(&m.arena).unwrap();
    m.exacto("cancelacion tras saldo nuevo");
    assert!(r.cancel(&m.arena).is_err());
    m.exacto("cancelacion repetida");
}

// ---------------------------------------------------------------- INV-D (Qoder 1)

/// Dos GET que llegan en orden inverso de marca: el de marca mayor manda.
#[test]
fn v4h_inv_d_dos_saldos_en_orden_inverso_de_marca() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let m = montaje();
    let (primero, segundo) = (m.puente.ticket(), m.puente.ticket());
    m.get(segundo, &rest(90.0, 200));
    m.get(primero, &rest(100.0, 100));
    assert_eq!(m.dimensionable(), 90.0, "el saldo viejo (T=100) movio el dimensionable");
    // El libro lo exige tambien por su cuenta (es el ultimo que escribe el campo).
    let arena = GlobalArena::build_in_own_stack(0.0);
    let mut libro = qra_libro_activo(&arena);
    libro.saldo_aceptado(&arena, 90.0, Some(200));
    libro.saldo_aceptado(&arena, 100.0, Some(100));
    assert_eq!(arena.unified_capital.load(Ordering::Relaxed), 90.0, "INV-D en el libro: marca menor al alza");
    libro.saldo_aceptado(&arena, 80.0, Some(150));
    assert_eq!(arena.unified_capital.load(Ordering::Relaxed), 90.0, "INV-D en el libro: marca menor a la baja");
    // Control: una marca mayor si lo mueve.
    libro.saldo_aceptado(&arena, 95.0, Some(300));
    assert_eq!(arena.unified_capital.load(Ordering::Relaxed), 95.0);
}

/// Stream que repite una ventana vieja tras reconectar, y marcas iguales con otro valor.
#[test]
fn v4h_inv_d_stream_que_repite_una_ventana_vieja() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let m = montaje();
    m.rest(100.0, 100);
    m.stream(95.0, 200);
    m.stream(92.5, 300);
    m.reconectar();
    m.rest(92.5, 300);
    m.stream(95.0, 200); // repeticion de la ventana vieja
    assert_eq!(m.dimensionable(), 92.5);
    let arena = GlobalArena::build_in_own_stack(0.0);
    let mut libro = qra_libro_activo(&arena);
    libro.saldo_aceptado(&arena, 92.5, Some(300));
    libro.saldo_aceptado(&arena, 95.0, Some(200));
    assert_eq!(arena.unified_capital.load(Ordering::Relaxed), 92.5, "la ventana vieja movio el libro");
    // Marca igual con otro valor: no hay orden, el menor.
    libro.saldo_aceptado(&arena, 92.0, Some(300));
    libro.saldo_aceptado(&arena, 92.5, Some(300));
    assert_eq!(arena.unified_capital.load(Ordering::Relaxed), 92.0, "marca igual: el menor");
    // Sin marca: tampoco hay orden.
    libro.saldo_aceptado(&arena, 99.0, None);
    assert_eq!(arena.unified_capital.load(Ordering::Relaxed), 92.0, "sin marca: el menor");
    // Control: con marca mayor, el nuevo.
    libro.saldo_aceptado(&arena, 99.0, Some(400));
    assert_eq!(arena.unified_capital.load(Ordering::Relaxed), 99.0);
}

/// Un alcance nuevo (relevo) empieza con su primer saldo, aunque su marca sea menor.
#[test]
fn v4h_inv_d_el_orden_vale_dentro_de_la_era_y_la_sesion() {
    let arena = GlobalArena::build_in_own_stack(0.0);
    let mut libro = qra_libro_activo(&arena);
    fijar_alcance(&mut libro, 1, SESION);
    libro.saldo_aceptado(&arena, 100.0, Some(500));
    fijar_alcance(&mut libro, 2, SESION + 1);
    libro.saldo_aceptado(&arena, 80.0, Some(300));
    assert_eq!(arena.unified_capital.load(Ordering::Relaxed), 80.0);
    libro.saldo_aceptado(&arena, 100.0, Some(200));
    assert_eq!(arena.unified_capital.load(Ordering::Relaxed), 80.0, "dentro del alcance nuevo rige INV-D");
}

// ---------------------------------------------------------------- INV-E (Qoder 2 y 3)

/// La confirmacion REST que llega despues de un saldo que ya incluyo el fill: no hace nada.
#[test]
fn v4h_inv_e_confirmacion_tardia_de_un_fill_ya_incluido() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let m = montaje();
    m.rest(100.0, 100);
    let r = m.reservar(0, COMISION);
    m.stream(100.0 - COMISION, 200); // incluye el fill de T=150
    assert!(m.dimensionable() <= m.saldo());
    r.confirmar_llenado_con_marca(&m.arena, Some(1.0), Some(150)).unwrap();
    m.exacto("ack tardio");
    r.confirmar_llenado_con_marca(&m.arena, Some(1.0), Some(150)).unwrap();
    m.exacto("ack repetido");
    assert!(r.cancel(&m.arena).is_err());
    m.exacto("cancelacion tras el ack");
}

/// Fills parciales desordenados: la marca es la MAXIMA (registro y libro), no la ultima llegada.
#[test]
fn v4h_inv_e_fills_parciales_desordenados_marca_maxima() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    use execution_engine::order_registry::{OrderRegistry, OrderStatus, TradeUpdate};
    let registro = OrderRegistry::new();
    let fill = |t: u64, acumulado: f64| TradeUpdate {
        client_order_id: "cL_v4h".into(),
        symbol: SIMBOLO.into(),
        side: "BUY".into(),
        position_side: "BOTH".into(),
        order_type: "LIMIT".into(),
        execution_type: "TRADE".into(),
        order_id: 7,
        status: OrderStatus::PartiallyFilled,
        orig_qty: 1.0,
        cumulative_filled_qty: acumulado,
        last_filled_qty: 0.5,
        last_filled_price: 100.0,
        avg_price: 100.0,
        commission: 0.02,
        commission_asset: "USDT".into(),
        trade_id: t,
        trade_time_ms: t,
    };
    registro.apply_trade_update(&fill(300, 1.0), 1);
    registro.apply_trade_update(&fill(250, 0.5), 2);
    let marca = registro.get("cL_v4h").map(|o| o.marca_fill_ms);
    assert_eq!(marca, Some(300), "registro: marca maxima de los fills");

    let m = montaje();
    m.rest(100.0, 100);
    let r = m.reservar(0, COMISION);
    r.confirmar_llenado_con_marca(&m.arena, Some(1.0), Some(300)).unwrap();
    // Una confirmacion repetida con una marca menor no la baja.
    r.confirmar_llenado_con_marca(&m.arena, Some(1.0), Some(250)).unwrap();
    m.stream(100.0 - COMISION, 280); // posterior al fill de 250, anterior al de 300
    assert_eq!(m.dimensionable(), m.saldo() - COMISION, "un saldo entre los dos fills no prueba inclusion");
    m.stream(100.0 - COMISION, 301);
    m.exacto("saldo posterior al ultimo fill");
}

/// La reserva se identifica por sesion, ranura y generacion.
#[test]
fn v4h_inv_e_la_reserva_lleva_su_sesion() {
    let arena = GlobalArena::build_in_own_stack(0.0);
    {
        let mut libro = qra_libro_activo(&arena);
        fijar_alcance(&mut libro, 1, SESION);
        libro.saldo_aceptado(&arena, 100.0, Some(100));
        libro.reservar(&arena, QRA_CLAVE, COMISION);
    }
    assert_eq!(alcance_de(&arena, QRA_CLAVE), Some((1, SESION)));
    // Otra generacion de la misma ranura es otra reserva.
    let otra = ClaveReserva { generation: 2, ..QRA_CLAVE };
    let mut libro = arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner());
    libro.cancelar(&arena, otra, COMISION);
    assert_eq!(libro.reservas(), 1, "cancelar otra generacion quito esta reserva");
}

// ---------------------------------------------------------------- INV-F (dictamen QRA V4G)

/// La propuesta calculada antes de que bajara el saldo y reservada despues se descarta; la que
/// ya estaba reservada no sale al exchange si su base ya no cabe.
#[test]
fn v4h_inv_f_propuesta_con_la_revision_anterior() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    // Al reservar.
    let m = montaje();
    m.rest(100.0, 100);
    let decidida = m.capital_de_risk();
    m.rest(90.0, 200);
    assert!(m.admitido(), "la admision volvio a abrirse tras el cambio");
    assert!(reservar_del_nucleo(&m.arena, 0, COMISION, decidida).is_none(), "reservada con 90 la decision de 100");
    assert!(reservar_del_nucleo(&m.arena, 0, COMISION, m.capital_de_risk()).is_some(), "control: recalculada, cabe");
    // Veto del puente al reservar: descarta.
    let m = montaje();
    m.rest(100.0, 100);
    let decidida = m.capital_de_risk();
    m.arena.capital_incompleto.store(true, Ordering::Release);
    assert!(reservar_del_nucleo(&m.arena, 0, COMISION, decidida).is_none(), "reservada con el veto del puente");
    // Al enviar.
    let m = montaje();
    m.rest(100.0, 100);
    let r = m.reservar(0, COMISION);
    assert!(enviar(&m.arena, &r));
    // Una reserva posterior no invalida la anterior (se decidio contando con ella).
    let r2 = m.reservar(1, COMISION);
    assert!(enviar(&m.arena, &r), "la reserva posterior invalido la anterior");
    assert!(enviar(&m.arena, &r2));
    m.rest(90.0, 200);
    assert!(m.admitido());
    assert!(!enviar(&m.arena, &r), "enviada con 90 la decision de 100");
    // La revision es identificable: 100, 90 y otra vez 100 no certifican la decision anterior.
    m.rest(100.0, 300);
    assert!(!enviar(&m.arena, &r), "la decision de la revision T=100 sobrevivio a T=200 y T=300");
    // Veto del puente que entra despues del permiso del OMS: la ultima comprobacion lo ve.
    let m = montaje();
    m.rest(100.0, 100);
    let r = m.reservar(0, COMISION);
    assert!(enviar(&m.arena, &r), "control: misma revision, se envia");
    m.arena.capital_incompleto.store(true, Ordering::Release);
    assert!(!enviar(&m.arena, &r), "enviada con el veto del puente");
    // Una entrada posterior YA EJECUTADA cuenta como debito antes del envio de la anterior.
    let m = montaje();
    m.rest(100.0, 100);
    let a = m.reservar(0, COMISION);
    let b = m.reservar(1, COMISION);
    assert!(enviar(&m.arena, &a), "la posterior viva no invalida la anterior");
    b.confirmar_llenado_con_marca(&m.arena, Some(1.0), Some(150)).unwrap();
    assert!(!enviar(&m.arena, &a), "el envio ignoro la comision de una orden posterior ya ejecutada");
    // Comision cero: la reserva conserva su identidad (si no, el envio no la encontraria).
    let m = montaje();
    m.rest(100.0, 100);
    let r = m.reservar(0, 0.0);
    assert!(enviar(&m.arena, &r), "la reserva de comision cero perdio su identidad");
}

// ---------------------------------------------------------------- INV-G (Qoder 4)

/// Cierre local con perdida y entrada inmediata antes del saldo siguiente.
#[test]
fn v4h_inv_g_cierre_local_con_perdida_y_entrada_inmediata() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let m = montaje();
    m.rest(100.0, 100);
    let r = m.reservar(0, COMISION);
    r.confirmar_llenado_con_marca(&m.arena, Some(1.0), Some(150)).unwrap();
    m.stream(100.0 - COMISION, 200);
    m.exacto("entrada reflejada");
    let saldo = m.saldo();
    // El nucleo cierra con perdida neta de 5 (perdida + comision de cierre).
    cierre_local(&m.arena, clave(0, r.generation), -5.0, true);
    let capital = m.capital_de_risk().expect("admision abierta");
    assert!(capital <= saldo - 5.0, "entrada inmediata dimensionada con {capital}; la cuenta tendra {}", saldo - 5.0);
    // Una ganancia local no suma; una perdida de papel (nunca en el exchange) tampoco resta.
    cierre_local(&m.arena, clave(1, 1), 3.0, true);
    cierre_local(&m.arena, clave(2, 1), -7.0, false);
    assert_eq!(m.dimensionable(), saldo - 5.0);
    // El fill de cierre del stream (comision 0,25, PnL realizado -4,75): debito exacto. No dice
    // de que ranura es, asi que el estimado se conserva (conservador: los dos restan).
    fill_de_cierre(&m, 9, 0.25, -4.75, 400, true, false);
    assert!(m.dimensionable() <= saldo - 5.0);
    // El mismo fill repetido no vuelve a restar.
    let antes = m.dimensionable();
    fill_de_cierre(&m, 9, 0.25, -4.75, 400, true, false);
    assert_eq!(m.dimensionable(), antes, "fill repetido");
    // Su saldo con la MISMA T no prueba inclusion del exacto; el siguiente cambio si.
    m.stream(saldo - 5.0, 400);
    assert!(m.dimensionable() <= m.saldo() - 5.0, "T igual: se conserva");
    m.stream(saldo - 5.0, 401);
    assert_eq!(m.dimensionable(), m.saldo() - 5.0, "queda el estimado del cierre local");
    // La reconciliacion ve la moneda plana: el estimado sale y la admision espera una REST
    // pedida despues; esa REST lo suelta y el dimensionable es el saldo.
    reconciliacion(&m, 0, &[0]);
    assert!(reservar_del_nucleo(&m.arena, 1, COMISION, m.capital_de_risk()).is_none(), "moneda plana sin REST posterior");
    m.rest(saldo - 5.0, 401);
    m.exacto("cierre reflejado por el exchange");
}

/// Dos ranuras de la misma moneda con perdidas 5 y 7; el terminal de una no quita el estimado
/// de la otra (RED de Codex V4H: 95 con maximo 88).
#[test]
fn v4h_inv_g_terminal_de_una_ranura_conserva_el_estimado_de_otra() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let m = montaje();
    m.rest(100.0, 100);
    cierre_local(&m.arena, clave(0, 1), -5.0, true);
    cierre_local(&m.arena, clave(1, 1), -7.0, true);
    assert_eq!(m.dimensionable(), 88.0);
    fill_de_cierre(&m, 17, 0.25, -4.75, 200, true, false);
    assert!(m.dimensionable() <= 88.0, "terminal de A deja dimensionable {}", m.dimensionable());
}

/// Funding pagado visto antes que su saldo; cobrado no suma.
#[test]
fn v4h_inv_g_funding_pagado() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let m = montaje();
    m.rest(100.0, 100);
    funding(&m.arena, 1, -0.3, 150);
    assert_eq!(m.dimensionable(), 100.0 - 0.3, "funding pagado no observado");
    funding(&m.arena, 1, -0.3, 150);
    assert_eq!(m.dimensionable(), 100.0 - 0.3, "repetido");
    funding(&m.arena, 2, 0.3, 160);
    assert_eq!(m.dimensionable(), 100.0 - 0.3, "funding cobrado no suma");
    funding(&m.arena, 3, -0.2, 90);
    assert_eq!(m.dimensionable(), 100.0 - 0.3, "funding anterior al saldo vigente ya incluido");
    m.rest(99.7, 170);
    m.exacto("saldo posterior al funding");
}

/// Cierre externo o liquidacion detectados sin saldo: la admision se veta hasta un saldo pedido
/// despues de detectarlo; una foto anterior al hecho, aunque traiga marca mayor, no lo suelta.
#[test]
fn v4h_inv_g_cierre_externo_detectado_sin_saldo() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    // a) la reconciliacion cierra una ranura que el exchange ya tiene plana.
    let m = montaje();
    m.rest(100.0, 100);
    let pedida_antes = m.puente.ticket();
    reconciliacion(&m, 1, &[]);
    assert!(reservar_del_nucleo(&m.arena, 0, COMISION, m.capital_de_risk()).is_none(), "admitida con un cierre externo sin saldo");
    m.get(pedida_antes, &rest(100.0, 200)); // foto pedida antes del hecho, marca mayor
    assert!(reservar_del_nucleo(&m.arena, 0, COMISION, m.capital_de_risk()).is_none(), "una foto anterior al cierre solto el veto");
    m.rest(98.0, 300);
    m.exacto("saldo pedido despues del cierre externo");
    assert!(reservar_del_nucleo(&m.arena, 0, COMISION, m.capital_de_risk()).is_some(), "control: el saldo posterior suelta");
    // El libro solo, sin el puente: un saldo con marca mayor no lo suelta.
    let arena = GlobalArena::build_in_own_stack(0.0);
    let mut libro = qra_libro_activo(&arena);
    libro.saldo_aceptado(&arena, 100.0, Some(100));
    cierre_externo_libro(&mut libro, &arena);
    libro.saldo_aceptado(&arena, 100.0, Some(200));
    assert!(vetado(&libro), "una foto tomada antes del cierre levanto el veto");
    drop(libro);
    // b) liquidacion en el stream: debito exacto y veto hasta una REST posterior.
    let m = montaje();
    m.rest(100.0, 100);
    let r = m.reservar(0, COMISION);
    fill_de_cierre(&m, 11, 0.5, -9.5, 150, true, true);
    assert_eq!(m.dimensionable(), m.saldo() - (COMISION + 10.0), "debito exacto de la liquidacion");
    assert!(!enviar(&m.arena, &r), "enviada con la liquidacion sin saldo");
    m.stream(100.0 - 10.0, 151);
    assert!(reservar_del_nucleo(&m.arena, 1, COMISION, m.capital_de_risk()).is_none(), "el stream solo no reemplaza la REST posterior");
    m.rest(90.0, 151);
    assert!(reservar_del_nucleo(&m.arena, 1, COMISION, m.capital_de_risk()).is_some(), "la REST posterior suelta el veto");
    // c) moneda plana con un cierre local pendiente: sale el estimado y se veta hasta la REST.
    let m = montaje();
    m.rest(100.0, 100);
    cierre_local(&m.arena, clave(0, 1), -5.0, true);
    assert_eq!(m.dimensionable(), 95.0);
    reconciliacion(&m, 0, &[0]);
    assert!(reservar_del_nucleo(&m.arena, 0, COMISION, m.capital_de_risk()).is_none(), "moneda plana sin saldo posterior");
    m.rest(95.0, 200);
    m.exacto("saldo con el cierre");
}

/// INV-D y la marca prestada (RED de Codex V4H: 90 con maximo 89): el saldo elegido conserva su
/// propia marca; la mayor marca conocida ordena pero no prueba inclusion.
#[test]
fn v4h_inv_d_marca_prestada_y_foto_vieja_tras_relevo() {
    let arena = GlobalArena::build_in_own_stack(0.0);
    let mut libro = qra_libro_activo(&arena);
    libro.saldo_aceptado(&arena, 90.0, None);
    libro.reservar(&arena, QRA_CLAVE, 1.0);
    libro.ejecutar(&arena, QRA_CLAVE, 0.0, Some(250));
    libro.saldo_aceptado(&arena, 100.0, Some(300));
    assert!(vetado(&libro) || arena.unified_capital.load(Ordering::Relaxed) <= 89.0, "marca prestada: {:?}", libro.saldo());
    // T200, luego sin T, luego T150: ni el sin T ni el viejo elevan o liberan.
    let arena = GlobalArena::build_in_own_stack(0.0);
    let mut libro = qra_libro_activo(&arena);
    libro.saldo_aceptado(&arena, 90.0, Some(200));
    libro.reservar(&arena, QRA_CLAVE, 1.0);
    libro.ejecutar(&arena, QRA_CLAVE, 0.0, Some(200));
    libro.saldo_aceptado(&arena, 95.0, None);
    libro.saldo_aceptado(&arena, 110.0, Some(150));
    assert_eq!(arena.unified_capital.load(Ordering::Relaxed), 89.0);
    drop(libro);
    // Relevo de la misma cuenta con la primera foto retrasada: no la hace actual.
    let arena = GlobalArena::build_in_own_stack(0.0);
    let mut libro = qra_libro_activo(&arena);
    fijar_alcance(&mut libro, 1, 1);
    libro.saldo_aceptado(&arena, 90.0, Some(200));
    fijar_alcance(&mut libro, 2, 2);
    libro.saldo_aceptado(&arena, 100.0, Some(150));
    assert!(vetado(&libro) || arena.unified_capital.load(Ordering::Relaxed) <= 90.0, "foto vieja tras relevo: {:?}", libro.saldo());
}
// ---------------------------------------------------------------- Modo unico

/// Escrituras de `unified_capital` (sin espacios ni saltos) que no van detras de la guarda del
/// modo. Devuelve (escrituras, sin_guarda).
fn escrituras_sin_guarda(fuente: &str, guardas: &[&str]) -> (usize, Vec<String>) {
    let plano: String = texto(fuente)
        .lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<String>()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let mut n = 0;
    let mut sin = Vec::new();
    for op in [".store(", ".fetch_add(", ".fetch_sub(", ".swap(", ".fetch_update("] {
        let patron = format!("unified_capital{op}");
        for (i, _) in plano.match_indices(&patron) {
            n += 1;
            let antes = &plano[..i];
            if !guardas.iter().any(|g| antes.ends_with(g)) {
                let desde = i.saturating_sub(80);
                sin.push(plano[desde..i + patron.len()].to_string());
            }
        }
    }
    (n, sin)
}

#[test]
fn v4h_modo_unico_las_seis_escrituras_no_ocurren_en_modo_derivado() {
    // Fuera de las pruebas (`#[cfg(test)]`) de cada archivo.
    let sin_pruebas = |f: &str| texto(f).split("#[cfg(test)]").next().unwrap_or("").to_string();
    let nucleo = sin_pruebas(NUCLEO);
    let reserva = sin_pruebas(RESERVA);
    let reconciliacion = sin_pruebas(RECONCILIACION);
    // god-engine-core/src/lib.rs: reserva y compensacion (:10506, :10528 en V4F) pasan por el
    // libro; el cierre local (:5632-5634) solo con el modo apagado.
    let (n, sin) = escrituras_sin_guarda(&nucleo, &["if!self.arena.config.dimensionable_derivado.load(Ordering::Acquire){self.arena."]);
    assert_eq!((n, sin.clone()), (1, Vec::<String>::new()), "nucleo: escrituras {n}, sin guarda {sin:?}");
    orden(&nucleo, &[RESERVA_DEL_NUCLEO, "_publication.cancelar(&self.arena, clave_reserva, fee_paid);"]);
    // entry_reservation.rs (:53, :86 en V4F): ninguna escritura directa.
    let (n, sin) = escrituras_sin_guarda(&reserva, &[]);
    assert_eq!(n, 0, "la reserva vuelve a escribir el campo: {sin:?}");
    // reconciliation.rs (:628-630 en V4F): solo con el modo apagado.
    let (n, sin) = escrituras_sin_guarda(&reconciliacion, &["if!arena.config.dimensionable_derivado.load(std::sync::atomic::Ordering::Acquire){arena."]);
    assert_eq!((n, sin.clone()), (1, Vec::<String>::new()), "reconciliacion: escrituras {n}, sin guarda {sin:?}");
    // El libro: las sumas de siempre solo sin activar; con el libro activo, solo `publicar`.
    let (n, sin) = escrituras_sin_guarda(LIBRO, &[
        ESCRITURA_DEL_LIBRO_ACTIVO,
        "if!self.derivado{arena.",
        "if!self.derivado{ifcomision.is_finite()&&comision!=0.0{arena.",
        "if!self.derivado{ifdevuelta.is_finite()&&devuelta!=0.0{arena.",
    ]);
    assert_eq!((n, sin.clone()), (4, Vec::<String>::new()), "libro: escrituras {n}, sin guarda {sin:?}");
    // En produccion el host activa el libro antes de vincular el publicador.
    orden(HOST, &[".activar(&arena_real);", "puente_cuenta.vincular_publicador(move |p| {"]);
    // Control: la prueba muerde si una escritura vuelve sin la guarda.
    let mutado = format!("{nucleo}\nfn x(a: &A) {{ a.unified_capital.fetch_add(1.0, Ordering::Relaxed); }}");
    let (_, sin) = escrituras_sin_guarda(&mutado, &["if!self.arena.config.dimensionable_derivado.load(Ordering::Acquire){self.arena."]);
    assert_eq!(sin.len(), 1, "el control no detecto la escritura sin guarda");
    // Control de comportamiento: con el libro activo, el cierre local y una reserva cancelada
    // dejan el dimensionable en el saldo (nadie mas escribe).
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    let m = montaje();
    m.rest(100.0, 100);
    cierre_local(&m.arena, clave(0, 1), 5.0, true);
    let r = m.reservar(1, COMISION);
    r.cancel(&m.arena).unwrap();
    m.exacto("modo derivado: solo escribe el libro");
}
