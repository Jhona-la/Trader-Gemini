//! CAPITAL-UNICO V4I: los criterios de aceptacion de Codex QRA (`qra_v4i_pedido.md`) y los
//! puntos 1 a 6 del pedido 20261112-0948f sobre el libro del dimensionable, el puente y las
//! anclas del host, el stream y el nucleo (seccion "V4I - DISENO" de
//! `integrador_capital_unico.md`).
//!
//! Piezas reales: el puente de produccion, el `GlobalArena` con su libro, `EntryReservation` y el
//! `OrderRegistry`. Lo que vive en linea en el host o el stream esta en el ADAPTADOR, atado a sus
//! lineas con anclas de fuente; contra la punta de V4H (774f97e2) el adaptador es el de V4H (copia
//! en `Desktop\PC-Turbo\integrador\v4i_logs\red_v4h\`). Importes del fixture; sin tolerancias.

use execution_engine::puente_cuenta::{LecturaCuenta, Publicacion, PuenteCuenta};
use god_engine_core::entry_reservation::EntryReservation;
use quantum_arena::dimensionable::{ClaveReserva, LibroDimensionable};
use quantum_arena::{position::PositionHorizon, symbol_registry, GlobalArena};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

const HOST: &str = include_str!("../../../src/bin/god_engine.rs");
const OMS: &str = include_str!("../../../src/host_oms.rs");
const NUCLEO: &str = include_str!("../src/lib.rs");
const STREAM: &str = include_str!("../../execution-engine/src/user_data_stream.rs");
const EJECUTOR: &str = include_str!("../../execution-engine/src/executor.rs");
const DESPACHO: &str = include_str!("../../execution-engine/src/entry_dispatch.rs");
const API: &str = include_str!("../../execution-engine/src/binance_api.rs");
const SIMBOLO: &str = "V4IUSDT";
const SESION: u64 = 3;
const COMISION: f64 = 0.04;
const CLAVE: ClaveReserva = ClaveReserva { coin_id: 0, slot: 0, generation: 1 };
const ORDEN: &str = "cL_v4i";

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

// ============================ ADAPTADOR (V4I) ============================

fn anclas_del_adaptador() {
    // Stream: testigo de era y candado en el mismo paso; fills de entrada por identidad.
    orden(STREAM, &[
        "let aplicado = self.puente.con_era_vigente(self.era, |_| {",
        "libro.fill_de_cierre(",
        "libro.cierre_externo(arena);",
        "if veta_cierre && aplicado.is_some() {",
        "let _ = self.puente.evidencia_perdida(self.era);",
        "let orden_terminal = matches!(",
        "let aplicado = self.puente.con_era_vigente(self.era, |_| {",
        // V4J (punto 2): con cantidad y comision tal como llegan; sin `T`, sin marca.
        "marca: marca_fill,",
        "libro.fill_de_entrada_del_stream(",
        "if let (true, false, Some(t)) = (orden_terminal, coid.starts_with(\"mc_\"), marca_fill) {",
        "libro.marca_de_orden_con_acumulado(arena, coin_id, coid, t, acumulado_e8);",
    ]);
    // Host: funding con testigo; confirmaciones con la marca terminal; cuenta; evidencia.
    orden(HOST, &[
        "let era_brk = puente_brk.era();",
        "let executor = exec_brk.load_full();",
        ".fetch_income_paged(&[], window_start, 4)",
        "let _ = puente_brk.con_era_vigente(era_brk, |_| {",
        "libro.debito_funding(&arena_brk, id, -e.income, e.time);",
    ]);
    orden(HOST, &[".activar(&arena_real);", ".fijar_cuenta(&arena_real, cuenta_de_arranque);", "pedir_evidencia_si_espera(&arena_veto, &puente_evidencia);"]);
    orden(HOST, &["let sesion_publicada = match &paso.resultado {", ".fijar_cuenta(&arena_real, cuenta_del_relevo);", "(paso, sesion_publicada)"]);
    // V4M (D1, autorizado 20261115-0050): las dos confirmaciones del host (envio ambiguo resuelto y
    // brazo Ok) pasan por la UNICA confirmacion de `host_oms::resolver_resultado_de_entrada`. Misma
    // propiedad: toda confirmacion de entrada usa la marca terminal; el host no confirma por otra via.
    assert_eq!(texto(HOST).matches("confirmar_llenado_con_marca(").count(), 0, "el host confirma fuera de la resolucion");
    assert_eq!(texto(OMS).matches("reservation.confirmar_llenado_con_marca(arena, Some(ejecutada), marca)").count(), 1, "una sola confirmacion en la resolucion");
    assert_eq!(texto(OMS).matches("let marca = executor.registry().marca_terminal(client_id);").count(), 1, "la confirmacion usa la marca terminal");
    assert!(texto(HOST).contains("quantum_engine::host_oms::resolver_resultado_de_entrada("), "el host no usa la resolucion");
    assert!(!texto(HOST).contains("map(|o| o.marca_fill_ms)"), "una confirmacion vuelve a usar la marca del ACK");
    orden(OMS, &[
        ".vincular_orden_con_cantidad(clave, request.client_order_id, request.quantity);",
        "async fn configure_leverage(&self, symbol: &str, leverage: u32) -> Result<(), String> {",
        "self.revalidar_capital()?;",
        "self.transport.configure_leverage(symbol, leverage).await",
        "async fn submit_entry(&self, request: &EntryRequest<'_>) -> Result<(), String> {",
        "self.revalidar_capital()?;",
        "let frontera = quantum_arena::dimensionable::FronteraDeEnvio::nueva();",
        "self.transport.submit_entry(request).await",
        // V4J (punto 1): el cruce ejecuta el validador de la sesion y del libro.
        "if !frontera.cruzada() {",
        "frontera.paso(&validar, || std::future::Future::poll(envio.as_mut(), cx))",
    ]);
    orden(NUCLEO, &[
        "} else if was_exchange_confirmed && net_realized_pnl < 0.0 {",
        "if qty.is_finite() && qty > 0.0 {",
        ".debito_cierre_local(",
    ]);
}

fn activar(arena: &GlobalArena) {
    arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner()).activar(arena);
}

fn libro(arena: &GlobalArena) -> MutexGuard<'_, LibroDimensionable> {
    arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner())
}

/// `publicar_capital_de_sesion` del host.
fn publicar(oms: &AtomicU64, arena: &GlobalArena, espejo: &AtomicU64, p: &Publicacion) -> bool {
    if p.veta {
        libro(arena).veto_del_puente();
    }
    let suelto = match (p.veta, p.vigente) {
        (true, _) | (false, None) => false,
        (false, Some((saldo, marca))) => {
            arena.capital_incompleto.store(true, Ordering::Release);
            oms.load(Ordering::SeqCst) == p.sesion && {
                let mut l = libro(arena);
                l.fijar_alcance(p.era, p.sesion);
                l.saldo_aceptado(arena, saldo, marca);
                drop(l);
                espejo.store(saldo.to_bits(), Ordering::Relaxed);
                true
            }
        }
    };
    arena.capital_incompleto.store(!suelto, Ordering::Release);
    suelto
}

/// La reserva del nucleo bajo la puerta (`admite` y `reservar_con_base`). `None`: descartada.
fn reservar_del_nucleo(arena: &GlobalArena, slot: usize, comision: f64, capital: Option<f64>) -> Option<EntryReservation> {
    let pos = arena.coins[0].positions.get_slot(slot);
    let generation = pos.generation.load(Ordering::Acquire).wrapping_add(1);
    let mut l = libro(arena);
    if !l.admite(arena, capital) {
        return None;
    }
    let clave = ClaveReserva { coin_id: 0, slot, generation };
    l.reservar_con_base(arena, clave, comision, capital);
    assert!(pos.open_with_tau_and_fee(
        true, 100.0, 1.0, 10.0, 1000, 101.0, 99.0, PositionHorizon::Continuous, 0.6, 0.7, comision, 30_000,
    ));
    Some(EntryReservation { coin_id: 0, slot, generation, symbol: SIMBOLO.into() })
}

/// El host fija la orden de la reserva antes de enviarla (`dispatch_entry_reservada`).
fn vincular(arena: &GlobalArena, clave: ClaveReserva, orden: &str) {
    libro(arena).vincular_orden(clave, orden);
}

/// Un fill de entrada del stream (`user_data_stream.rs`), comision en USDT.
fn fill_de_entrada(arena: &GlobalArena, orden: &str, trade_id: u64, comision: f64, t: Option<u64>) {
    libro(arena).fill_de_entrada(arena, 0, orden, trade_id, Some(comision), t);
}

/// El evento terminal de la orden en el stream.
fn terminal(arena: &GlobalArena, orden: &str, t: u64) {
    libro(arena).marca_de_orden(arena, 0, orden, t);
}

/// La cuenta del ejecutor (host, arranque y relevo).
fn cuenta(arena: &GlobalArena, id: u64) {
    libro(arena).fijar_cuenta(arena, id);
}

fn en_suelo(arena: &GlobalArena) -> bool {
    libro(arena).en_suelo()
}

/// Un fill de cierre del stream (`user_data_stream.rs`), con el testigo de era del socket.
fn fill_de_cierre_del_socket(m: &Montaje, era: u64, trade_id: u64, debito: f64, t: u64) {
    let _ = m.puente.con_era_vigente(era, |_| {
        libro(&m.arena).fill_de_cierre(&m.arena, 0, trade_id, debito, t, true);
    });
}

/// El funding del host (`FEE-BREAKER`) con la era tomada antes de la consulta.
fn funding_de_la_era(m: &Montaje, era: u64, id: u64, pagado: f64, t: u64) {
    let _ = m.puente.con_era_vigente(era, |_| {
        libro(&m.arena).debito_funding(&m.arena, id, pagado, t);
    });
}
// ========================== FIN DEL ADAPTADOR ==========================

fn rest(saldo: f64, t: Option<u64>) -> LecturaCuenta {
    LecturaCuenta {
        saldo,
        tiempo_saldo_ms: t,
        posiciones: Some(HashMap::new()),
        rechazo_arranque: None,
        upnl_del_activo: Some(0.0),
    }
}

struct Montaje {
    puente: PuenteCuenta,
    arena: Arc<GlobalArena>,
    oms: Arc<AtomicU64>,
}

fn montaje() -> Montaje {
    quantum_arena::symbols::update_dynamic_universe(vec![SIMBOLO.into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(SIMBOLO)]);
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
    Montaje { puente, arena, oms }
}

impl Montaje {
    fn dimensionable(&self) -> f64 {
        self.arena.unified_capital.load(Ordering::Relaxed)
    }
    fn capital_de_risk(&self) -> Option<f64> {
        let vetado = self.arena.capital_incompleto.load(Ordering::Acquire);
        let c = self.arena.unified_capital.load(Ordering::Relaxed);
        (!vetado).then_some(c)
    }
    fn rest(&self, saldo: f64, t: Option<u64>) {
        let _ = self.puente.observar_rest_de_sesion(self.puente.ticket(), self.puente.sesion(), &rest(saldo, t));
    }
}

fn arena_activa() -> Arc<GlobalArena> {
    let arena = GlobalArena::build_in_own_stack(0.0);
    activar(&arena);
    arena
}

fn dim(arena: &GlobalArena) -> f64 {
    arena.unified_capital.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------- punto 1, INV-D

/// T200, caida sin T, luego T150 (y su repeticion) y T201: ni elevan el saldo ni retiran cargos.
/// La salida del suelo exige evidencia positiva (el primer saldo tras un veto del puente).
#[test]
fn v4i_inv_d_suelo_sin_t_no_lo_elevan_fotos_con_marca() {
    anclas_del_adaptador();
    let arena = arena_activa();
    let mut l = libro(&arena);
    l.saldo_aceptado(&arena, 100.0, Some(100));
    l.reservar(&arena, CLAVE, 1.0);
    l.ejecutar(&arena, CLAVE, 0.0, Some(180));
    l.saldo_aceptado(&arena, 90.0, None); // caida sin T
    for _ in 0..2 {
        l.saldo_aceptado(&arena, 110.0, Some(150));
    }
    l.saldo_aceptado(&arena, 100.0, Some(201)); // foto con marca mayor, tomada antes de la caida
    assert!(l.vetado() || dim(&arena) <= 89.0, "una foto con marca subio el suelo: {:?}", l.saldo());
    assert_eq!(l.reservado(), 1.0, "una foto que no es la vigente retiro el cargo");
    // Una lectura menor si baja el suelo; con SU marca (T202) incluye el cargo de T180.
    l.saldo_aceptado(&arena, 85.0, Some(202));
    assert_eq!(dim(&arena), 85.0);
    l.saldo_aceptado(&arena, 100.0, Some(300));
    assert_eq!(dim(&arena), 85.0, "sigue en el suelo: solo baja");
    // Evidencia positiva: el puente veto y el primer saldo que entrega al soltar manda.
    l.veto_del_puente();
    l.saldo_aceptado(&arena, 95.0, Some(400));
    assert_eq!(dim(&arena), 95.0, "evidencia positiva: saldo nuevo y el cargo de T180 incluido");
    l.saldo_aceptado(&arena, 99.0, Some(500));
    assert_eq!(dim(&arena), 99.0, "fuera del suelo rige otra vez el orden por marca");
}

/// El mismo corte por el puente de produccion. El puente descarta una REST sin marca que llega
/// despues de una con marca (`SinOrden`); un saldo sin marca solo le llega al libro como primera
/// lectura de una era. Desde ahi una REST con marca mayor no lo sube, y solo la REST pedida
/// despues de la evidencia perdida lo levanta.
#[test]
fn v4i_inv_d_suelo_sin_t_por_el_puente_y_salida_con_evidencia() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let m = montaje();
    m.rest(90.0, None);
    assert_eq!(m.capital_de_risk(), Some(90.0));
    m.rest(100.0, Some(201));
    assert!(m.capital_de_risk().is_none_or(|c| c <= 90.0), "foto anterior a la caida elevo el suelo: {:?}", m.capital_de_risk());
    assert!(en_suelo(&m.arena));
    // El host pide evidencia perdida al puente (veta hasta una REST pedida despues).
    m.puente.evidencia_perdida(m.puente.era()).unwrap();
    assert!(m.capital_de_risk().is_none(), "la evidencia perdida veta");
    m.rest(100.0, Some(202));
    assert_eq!(m.capital_de_risk(), Some(100.0), "la REST posterior es evidencia positiva");
    assert!(!en_suelo(&m.arena));
}

/// Relevo: el orden por marca empieza de nuevo en el alcance nuevo y los debitos pasan intactos.
#[test]
fn v4i_inv_d_relevo_reinicia_el_orden_y_traspasa_los_debitos() {
    let arena = arena_activa();
    let mut l = libro(&arena);
    l.fijar_alcance(1, 1);
    l.saldo_aceptado(&arena, 100.0, Some(500));
    l.reservar(&arena, CLAVE, 1.0);
    l.ejecutar(&arena, CLAVE, 0.0, None);
    l.debito_funding(&arena, 7, 0.5, 600);
    let antes = l.reservado();
    l.fijar_alcance(2, 2);
    l.saldo_aceptado(&arena, 80.0, Some(300));
    assert_eq!(l.reservado(), antes, "el relevo perdio o libero debitos");
    assert_eq!(dim(&arena), 80.0 - antes);
    l.saldo_aceptado(&arena, 100.0, Some(200));
    assert_eq!(dim(&arena), 80.0 - antes, "dentro del alcance nuevo rige el orden por marca");
    l.saldo_aceptado(&arena, 79.0, Some(700));
    assert_eq!(l.reservado(), 1.0, "el funding de T600 queda incluido con T700; la entrada sin marca no");
}

/// Otra cuenta (otro venue o clave): el libro empieza sin saldo y nada de la anterior ordena.
#[test]
fn v4i_inv_d_otra_cuenta_empieza_sin_saldo() {
    let arena = arena_activa();
    cuenta(&arena, 1);
    {
        let mut l = libro(&arena);
        l.saldo_aceptado(&arena, 100.0, Some(500));
        l.reservar(&arena, CLAVE, 1.0);
    }
    cuenta(&arena, 1);
    assert_eq!(dim(&arena), 99.0, "la misma cuenta no cambia nada");
    cuenta(&arena, 2);
    {
        let mut l = libro(&arena);
        assert!(l.saldo().is_none(), "la cuenta nueva hereda el saldo de la anterior");
        assert!(!l.admite(&arena, Some(99.0)), "admitida sin saldo de la cuenta nueva");
        l.saldo_aceptado(&arena, 50.0, Some(100));
    }
    // V4K (A5, dictamen V4J :37): la cuenta ajena no soporta el debito de la original (antes 49,
    // que sumaba la reserva de la cuenta 1 al saldo de la 2); el debito se conserva y vuelve a
    // restar al volver a su cuenta.
    assert_eq!(dim(&arena), 50.0, "su primer saldo vale (aunque su marca sea menor) y no soporta la reserva ajena");
    assert_eq!(libro(&arena).reservado_de_otras_cuentas(), 1.0, "el debito de la cuenta 1 se perdio");
    cuenta(&arena, 1);
    libro(&arena).saldo_aceptado(&arena, 100.0, Some(600));
    assert_eq!(dim(&arena), 99.0, "al volver a la cuenta 1 su reserva no sigue restando");
}

// ---------------------------------------------------------------- punto 2, INV-E

/// Saldo T201 que cubre solo el parcial T200; segundo parcial sin T; ACK acumulado. En los dos
/// ordenes (ACK antes y despues del saldo) el segundo cargo se conserva.
#[test]
fn v4i_inv_e_parcial_marcado_segundo_sin_t_y_ack_acumulado() {
    for ack_antes in [false, true] {
        let arena = arena_activa();
        {
            let mut l = libro(&arena);
            l.saldo_aceptado(&arena, 100.0, Some(100));
            l.reservar(&arena, CLAVE, 0.75);
        }
        vincular(&arena, CLAVE, ORDEN);
        fill_de_entrada(&arena, ORDEN, 1, 0.25, Some(200));
        fill_de_entrada(&arena, ORDEN, 2, 0.5, None);
        // El host confirma sin marca: el ACK (updateTime) no cubre cada ejecucion.
        if ack_antes {
            libro(&arena).ejecutar(&arena, CLAVE, 0.0, None);
        }
        libro(&arena).saldo_aceptado(&arena, 99.75, Some(201));
        if !ack_antes {
            libro(&arena).ejecutar(&arena, CLAVE, 0.0, None);
        }
        assert_eq!(dim(&arena), 99.25, "ack_antes={ack_antes}: el segundo cargo sin T salio del libro");
    }
}

/// El registro: la marca que cubre la orden es la del evento terminal del stream, nunca el
/// `updateTime` de un ACK acumulado.
#[test]
fn v4i_inv_e_el_ack_acumulado_no_aporta_marca() {
    use execution_engine::order_registry::{OrderRegistry, OrderStatus, TradeUpdate};
    use execution_engine::order_types::OrderAck;
    let registro = OrderRegistry::new();
    let ack: OrderAck = serde_json::from_str(r#"{"symbol":"V4IUSDT","clientOrderId":"cL_v4i","orderId":9,"side":"BUY","status":"FILLED","origQty":"1.0","executedQty":"1.0","avgPrice":"100.0","price":0,"updateTime":300}"#).unwrap();
    registro.apply_ack(&ack, 1);
    assert_eq!(registro.marca_terminal(ORDEN), None, "el ACK acumulado aporto marca");
    let evento = |t: u64, status: OrderStatus| TradeUpdate {
        client_order_id: ORDEN.into(),
        symbol: SIMBOLO.into(),
        side: "BUY".into(),
        position_side: "BOTH".into(),
        order_type: "LIMIT".into(),
        execution_type: "TRADE".into(),
        order_id: 9,
        status,
        orig_qty: 1.0,
        cumulative_filled_qty: 0.5,
        last_filled_qty: 0.5,
        last_filled_price: 100.0,
        avg_price: 100.0,
        commission: 0.02,
        commission_asset: "USDT".into(),
        trade_id: t,
        trade_time_ms: t,
    };
    registro.apply_trade_update(&evento(200, OrderStatus::PartiallyFilled), 2);
    assert_eq!(registro.marca_terminal(ORDEN), None, "un parcial no es terminal");
    registro.apply_trade_update(&evento(250, OrderStatus::Filled), 3);
    assert_eq!(registro.marca_terminal(ORDEN), Some(250));
}

/// Fill repetido, ACK repetido, repeticion tras reconectar y tras retirar el cargo: sin doble
/// cargo. El mismo `trade_id` en otro simbolo es otra ejecucion.
#[test]
fn v4i_inv_e_repeticiones_por_identidad_sin_doble_cargo() {
    let arena = arena_activa();
    {
        let mut l = libro(&arena);
        l.saldo_aceptado(&arena, 100.0, Some(100));
        l.reservar(&arena, CLAVE, 0.5);
    }
    vincular(&arena, CLAVE, ORDEN);
    fill_de_entrada(&arena, ORDEN, 1, 0.5, Some(200));
    fill_de_entrada(&arena, ORDEN, 1, 0.5, Some(200));
    libro(&arena).ejecutar(&arena, CLAVE, 0.0, None);
    libro(&arena).ejecutar(&arena, CLAVE, 0.0, None);
    assert_eq!(dim(&arena), 99.5, "fill o ACK repetidos cargaron dos veces");
    // Reconexion: el stream repite el fill.
    fill_de_entrada(&arena, ORDEN, 1, 0.5, Some(200));
    assert_eq!(dim(&arena), 99.5);
    // Mismo trade_id en otro simbolo: otra ejecucion, sin reserva que la respalde: se retiene.
    libro(&arena).fill_de_entrada(&arena, 1, "cL_otro", 1, Some(0.25), Some(210));
    assert_eq!(dim(&arena), 99.25, "el trade_id de otro simbolo se tomo por el mismo fill");
    // Cobertura acreditada: el terminal T220 y un saldo con marca mayor retiran UNA vez.
    terminal(&arena, ORDEN, 220);
    libro(&arena).saldo_aceptado(&arena, 99.25, Some(221));
    assert_eq!(dim(&arena), 99.25, "retirado una sola vez");
    fill_de_entrada(&arena, ORDEN, 1, 0.5, Some(200));
    libro(&arena).ejecutar(&arena, CLAVE, 0.0, Some(220));
    assert_eq!(dim(&arena), 99.25, "una repeticion tras retirar el cargo volvio a cargar");
}

/// Reserva estimada 0,25 y comision real 0,75 en el stream: el disponible cae 0,75 antes de
/// otra apertura; la cancelacion solo libera la parte no ejecutada.
#[test]
fn v4i_inv_e_comision_real_mayor_y_cancelacion_parcial() {
    let arena = arena_activa();
    {
        let mut l = libro(&arena);
        l.saldo_aceptado(&arena, 100.0, Some(100));
        l.reservar(&arena, CLAVE, 0.25);
    }
    vincular(&arena, CLAVE, ORDEN);
    fill_de_entrada(&arena, ORDEN, 5, 0.75, Some(150));
    assert_eq!(dim(&arena), 99.25, "el disponible no descuenta la comision real");
    libro(&arena).ejecutar(&arena, CLAVE, 0.0, None);
    assert_eq!(dim(&arena), 99.25);
    // Cancelacion con un parcial ya ejecutado: queda su comision real.
    let otra = ClaveReserva { slot: 1, ..CLAVE };
    {
        let mut l = libro(&arena);
        l.reservar(&arena, otra, 1.0);
        l.vincular_orden(otra, "cL_parcial");
        l.fill_de_entrada(&arena, 0, "cL_parcial", 6, Some(0.5), Some(160));
        l.cancelar(&arena, otra, 1.0);
    }
    assert_eq!(dim(&arena), 99.25 - 0.5, "la cancelacion libero la comision de lo ejecutado");
}

// ---------------------------------------------------------------- punto 3, INV-F al reservar

/// Propuesta R0 con 100; caida a 90 y vuelta a 100 / veto transitorio / cambio de sesion, cada
/// uno por separado: 0 reservas con la decision vieja; la recalculada con la revision nueva
/// reserva 1; el control sin cambio reserva 1.
#[test]
fn v4i_inv_f_al_reservar_la_revision_es_identificable() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    // Control sin cambio.
    let m = montaje();
    m.rest(100.0, Some(100));
    let r0 = m.capital_de_risk();
    assert!(reservar_del_nucleo(&m.arena, 0, COMISION, r0).is_some(), "control: misma revision");
    for corte in ["100-90-100", "veto transitorio", "cambio de sesion"] {
        let m = montaje();
        m.rest(100.0, Some(100));
        let r0 = m.capital_de_risk();
        match corte {
            "100-90-100" => {
                m.rest(90.0, Some(200));
                m.rest(100.0, Some(300));
            }
            "veto transitorio" => {
                m.puente.evidencia_perdida(m.puente.era()).unwrap();
                m.rest(100.0, Some(100));
            }
            _ => {
                let nueva = SESION + 1;
                let (_, era) = m.puente.relevar_sesion(|| {
                    m.oms.store(nueva, Ordering::SeqCst);
                    ((), Some(nueva))
                });
                m.puente.abrir_socket(era.unwrap()).unwrap();
                m.rest(100.0, Some(100));
            }
        }
        assert_eq!(m.capital_de_risk(), r0, "{corte}: fixture, el importe vuelve a ser el mismo");
        assert!(reservar_del_nucleo(&m.arena, 0, COMISION, r0).is_none(), "{corte}: reservada con la decision de la revision anterior");
        // La decision se recalcula con lo que publica la revision nueva.
        let r1 = m.capital_de_risk();
        assert!(reservar_del_nucleo(&m.arena, 0, COMISION, r1).is_some(), "{corte}: la recalculada no reservo ({r1:?})");
    }
}

// ---------------------------------------------------------------- punto 4, INV-G

/// Dos ranuras de la misma moneda con perdidas 5 y 7; terminal de A con debito 5: quedan al
/// menos 12 retenidos (el estimado de A no se atribuye al fill: se conservan los dos).
/// Terminal duplicado, fuera de orden y sin `trade_id`.
#[test]
fn v4i_inv_g_atribucion_y_fills_sin_identidad() {
    let arena = arena_activa();
    let mut l = libro(&arena);
    l.saldo_aceptado(&arena, 100.0, Some(100));
    l.debito_cierre_local(&arena, CLAVE, 5.0);
    l.debito_cierre_local(&arena, ClaveReserva { slot: 1, ..CLAVE }, 7.0);
    l.fill_de_cierre(&arena, 0, 17, 5.0, 200, true);
    assert!(l.reservado() >= 12.0, "retenido {}", l.reservado());
    let retenido = l.reservado();
    l.fill_de_cierre(&arena, 0, 17, 5.0, 200, true);
    assert_eq!(l.reservado(), retenido, "terminal duplicado");
    l.fill_de_cierre(&arena, 0, 16, 1.0, 150, false); // fuera de orden
    assert_eq!(l.reservado(), retenido + 1.0);
    l.fill_de_cierre(&arena, 0, 0, 0.5, 0, false);
    l.fill_de_cierre(&arena, 0, 0, 0.5, 0, false);
    assert_eq!(l.reservado(), retenido + 2.0, "sin trade_id: ruta conservadora, cada uno cuenta");
}

/// Liquidacion detectada tras T300 con saldo conocido T100: una foto en vuelo T200 no abre
/// entradas; la REST pedida despues si. Las salidas defensivas no pasan por la frontera.
#[test]
fn v4i_inv_g_liquidacion_y_foto_en_vuelo() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    let m = montaje();
    m.rest(100.0, Some(100));
    let en_vuelo = m.puente.ticket();
    fill_de_cierre_del_socket(&m, m.puente.era(), 31, 9.0, 300);
    libro(&m.arena).cierre_externo(&m.arena);
    m.puente.evidencia_perdida(m.puente.era()).unwrap();
    let _ = m.puente.observar_rest_de_sesion(en_vuelo, m.puente.sesion(), &rest(100.0, Some(200)));
    assert!(reservar_del_nucleo(&m.arena, 0, COMISION, m.capital_de_risk()).is_none(), "la foto en vuelo abrio entradas");
    assert!(quantum_arena::dimensionable::cruzar_frontera_de_envio().is_ok(), "una salida defensiva quedo bloqueada por la frontera");
    m.rest(90.5, Some(301));
    assert!(reservar_del_nucleo(&m.arena, 0, COMISION, m.capital_de_risk()).is_some(), "la REST posterior no solto el veto");
}

/// Coste de INV-F al reservar: cuando el saldo siguiente confirma los debitos, el dimensionable
/// vuelve a un importe ya publicado en la revision anterior. Esa decision no identifica su
/// revision: se rechaza UNA vez, se publica el double inmediato inferior y la recalculada entra.
#[test]
fn v4i_inv_f_coste_saldo_que_confirma_los_debitos_recalcula_una_vez() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    let m = montaje();
    m.rest(100.0, Some(100));
    fill_de_cierre_del_socket(&m, m.puente.era(), 51, 9.0, 150);
    assert_eq!(m.capital_de_risk(), Some(91.0));
    m.rest(91.0, Some(151)); // el exchange ya refleja el debito: mismo importe, otra revision
    let vieja = m.capital_de_risk();
    assert_eq!(vieja, Some(91.0));
    assert!(reservar_del_nucleo(&m.arena, 0, COMISION, vieja).is_none(), "un importe de la revision anterior identifico la decision");
    let nueva = m.capital_de_risk();
    assert_eq!(nueva, Some(91.0f64.next_down()), "el valor publicado para recalcular");
    assert!(reservar_del_nucleo(&m.arena, 0, COMISION, nueva).is_some(), "la decision recalculada no entro");
}

// ---------------------------------------------------------------- punto 5

/// (a) A lista para enviar; B posterior viva no la invalida; B ejecutada con comision real
/// pendiente si reduce el disponible de A; comision cero conserva la identidad.
#[test]
fn v4i_ruta_hermana_a_exencion_solo_de_reservas_vivas() {
    let arena = arena_activa();
    let a = ClaveReserva { slot: 0, ..CLAVE };
    let b = ClaveReserva { slot: 1, ..CLAVE };
    let mut l = libro(&arena);
    l.saldo_aceptado(&arena, 100.0, Some(100));
    l.reservar_con_base(&arena, a, 0.0, Some(100.0));
    assert!(l.envio_valido(a), "comision cero perdio la identidad");
    l.reservar_con_base(&arena, b, 0.1, Some(100.0));
    assert!(l.envio_valido(a), "una posterior viva invalido a A");
    l.vincular_orden(b, "cL_b");
    l.fill_de_entrada(&arena, 0, "cL_b", 9, Some(0.3), Some(150));
    l.ejecutar(&arena, b, 0.0, None);
    assert!(!l.envio_valido(a), "el envio de A ignoro la comision real de B ya ejecutada");
}

/// (b) Relevo: el fill privado y el funding de la sesion anterior, entregados por sus caminos
/// despues de publicar el sucesor, dejan libro, espejo y veto intactos.
#[test]
fn v4i_ruta_hermana_b_callbacks_de_la_sesion_anterior() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_del_adaptador();
    let m = montaje();
    m.rest(100.0, Some(100));
    let era_vieja = m.puente.era();
    let nueva = SESION + 1;
    let (_, era) = m.puente.relevar_sesion(|| {
        m.oms.store(nueva, Ordering::SeqCst);
        ((), Some(nueva))
    });
    m.puente.abrir_socket(era.unwrap()).unwrap();
    m.rest(100.0, Some(150));
    let (antes, veto_antes) = (m.dimensionable(), m.capital_de_risk().is_none());
    let reservado = libro(&m.arena).reservado();
    fill_de_cierre_del_socket(&m, era_vieja, 41, 3.0, 160);
    funding_de_la_era(&m, era_vieja, 42, 0.5, 170);
    assert_eq!(libro(&m.arena).reservado(), reservado, "el socket o el historial viejo escribieron el libro");
    assert_eq!((m.dimensionable(), m.capital_de_risk().is_none()), (antes, veto_antes));
    // Control: por la era vigente si escriben.
    fill_de_cierre_del_socket(&m, m.puente.era(), 41, 3.0, 160);
    funding_de_la_era(&m, m.puente.era(), 42, 0.5, 170);
    assert_eq!(m.dimensionable(), antes - 3.5);
}

/// (c) La ruta maker y la iceberg son inalcanzables en el host (fijadas en la fuente); aun asi
/// cada POST de apertura del ejecutor, el MARKET del remanente incluido, pasa la frontera.
#[test]
fn v4i_ruta_hermana_c_maker_inalcanzable_y_frontera_en_cada_post() {
    assert!(texto(HOST).contains("let force_maker = false;"), "el host volvio a habilitar la ruta maker");
    assert!(texto(API).contains("pub const SUPPORTS_NATIVE_ICEBERG: bool = false;"), "iceberg nativo habilitado");
    let e = texto(EJECUTOR);
    // V4J (punto 1): el cruce esta en el ultimo instante antes de la entrega al cliente HTTP
    // (despues de registrar la intencion y firmar); entre el cruce y la entrega, nada.
    for (funcion, envio, cruce) in [
        ("async fn execute_raw_qty_internal(", "let res = self.client.execute_order_payload_typed(buf.as_str()).await;", "self.cruzar_frontera(&client_order_id, timestamp)?;"),
        ("async fn execute_limit_order(", "let res = self.client.execute_order_payload(buf.as_str()).await;", "quantum_arena::dimensionable::cruzar_frontera_de_envio()?;"),
        ("async fn execute_ioc_order(", "match self.client.execute_order_payload_body(buf.as_str()).await {", "self.cruzar_frontera(client_order_id, timestamp)?;"),
        ("async fn execute_iceberg_limit(", "let res = self.client.execute_order_payload(buf.as_str()).await;", "quantum_arena::dimensionable::cruzar_frontera_de_envio()?;"),
    ] {
        let cuerpo = e.rsplit_once(funcion).unwrap_or_else(|| panic!("falta {funcion}")).1;
        let (antes, _) = cuerpo.split_once(envio).unwrap_or_else(|| panic!("{funcion}: falta el envio"));
        assert!(antes.contains(cruce), "{funcion}: el POST no pasa la frontera");
        let tras = antes.rsplit_once(cruce).unwrap().1;
        assert!(tras.trim().lines().all(|l| l.trim().is_empty()), "{funcion}: hay codigo entre la frontera y la entrega: {tras:?}");
    }
    // El remanente del maker sale por la ruta MARKET (que pasa la frontera), despues de cancelar.
    orden(&e, &["async fn execute_maker_chase(", "self.cancel_order(symbol, client_order_id).await", "self.query_order(symbol, client_order_id).await", "self.execute_raw_qty_with_client_id(symbol, is_long, remaining, step_size, remnant_id)"]);
    orden(DESPACHO, &["EntryRoute::Maker { price } => {", "self.execute_maker_chase("]);
}

// ---------------------------------------------------------------- punto 6, Qoder

/// El libro solo, con observaciones ENTRANTES desordenadas (alta, baja y repetida), sin apoyarse
/// en que el puente ordena.
#[test]
fn v4i_qoder_libro_con_entradas_desordenadas() {
    let arena = arena_activa();
    let mut l = libro(&arena);
    for (saldo, t) in [(90.0, 200), (110.0, 150), (80.0, 150), (90.0, 200), (110.0, 150)] {
        l.saldo_aceptado(&arena, saldo, Some(t));
        assert_eq!(l.saldo(), Some((90.0, Some(200))), "entrada ({saldo}, T{t})");
    }
}

/// Un cierre local solo con cantidad ejecutada mayor que cero deja debito estimado (ancla).
#[test]
fn v4i_qoder_cierre_local_solo_con_cantidad_ejecutada() {
    orden(NUCLEO, &[
        "let was_exchange_confirmed =",
        "} else if was_exchange_confirmed && net_realized_pnl < 0.0 {",
        "if qty.is_finite() && qty > 0.0 {",
        ".debito_cierre_local(",
    ]);
    // El margen de cada reserva entra antes que la siguiente: el orden de las reservas no
    // sustituye el limite de margen (lo valida el exchange al recibir cada orden).
    orden(NUCLEO, &["self.arena.used_margin.fetch_add(margin_req, Ordering::Relaxed);", "_publication.reservar_con_base(&self.arena, clave_reserva, fee_paid, capital_de_la_decision);"]);
}

// ---------------------------------------------------------------- corte de cuenta (anadido B)

/// Un cargo sin marca no sale con el saldo siguiente. V4I lo soltaba con un corte de cuenta
/// conciliado; V4J (punto 4, salida A4b de `qra_v4j_criterios.md`) retiro ese corte del API
/// productivo: la misma secuencia queda retenida con cualquier foto posterior (ninguna marca, por
/// mayor que sea, prueba la inclusion de un cargo sin la suya). Ver `capital_v4j_corte_contract.rs`.
#[test]
fn v4i_corte_de_cuenta_conciliado_unica_salida_sin_marca() {
    let arena = arena_activa();
    {
        let mut l = libro(&arena);
        l.saldo_aceptado(&arena, 100.0, Some(100));
        l.reservar(&arena, CLAVE, 0.25);
        l.vincular_orden(CLAVE, ORDEN);
        l.fill_de_entrada(&arena, 0, ORDEN, 77, Some(0.25), None); // sin T
        l.ejecutar(&arena, CLAVE, 0.0, None);
        l.saldo_aceptado(&arena, 99.75, Some(300));
    }
    assert_eq!(dim(&arena), 99.5, "el cargo sin marca sigue retenido con el saldo siguiente");
    libro(&arena).saldo_aceptado(&arena, 99.75, Some(10_000));
    assert_eq!(dim(&arena), 99.5, "una marca mayor cualquiera libero el cargo sin marca");
}