//! CAPITAL-UNICO V4L (pedido `codex_app_pedido_v4l.md` con la adenda de QODER-APP): las ocho pruebas
//! de la matriz de aceptacion, sobre las piezas productivas.
//!
//! Piezas reales: `OmsSession::dispatch_entry_reservada` (guardia, frontera y compromiso local), el
//! `OrderExecutor` de produccion con su cliente HTTP contra un servidor local (127.0.0.1, sin red
//! externa ni claves), el libro del dimensionable, `EntryReservation` y `EntrySettlement`, la
//! resolucion del host `host_oms::resolver_resultado_de_entrada`, el puente de produccion
//! (`PuenteCuenta`), el plano del freno (`drawdown::EscrituraDelPlano`, `vista_del_freno`), la
//! decision real de Risk (`RiskEngine::evaluate_quantum_order`), la autoria del disparo del inmune
//! (`aplanado_inmune::comprometer_disparo`, `relevar_causa_patrimonial`) y el registro de flujos
//! (`host_oms::registrar_flujos_externos_de_era`). Lo que vive en linea en `god_engine.rs` se ata
//! con anclas de fuente (texto con CRLF normalizado) y se reporta como ancla, no como ejecucion.
//! Barreras deterministas (barreras del compromiso, `std::sync::Barrier`, `Notify` del servidor);
//! ningun plazo de calendario decide un resultado.
use arc_swap::ArcSwap;
use execution_engine::entry_dispatch::{EntryRequest, EntryRoute};
use execution_engine::executor::{ExecutionProvider, OrderExecutor};
use execution_engine::puente_cuenta::{LecturaCuenta, PuenteCuenta};
use execution_engine::reconciliation::PositionRiskEntry;
use god_engine_core::entry_reservation::EntryReservation;
use quantum_arena::dimensionable::{ClaveReserva, FaseDeEntrega};
use quantum_arena::position::PositionHorizon;
use quantum_arena::{FeeSchedule, GlobalArena, SymbolFees};
use quantum_engine::host_oms::{DesenlaceDeEntrada, EntrySettlement, OmsSession, SessionToken};
use risk_engine::drawdown::{self, FlujosExternos, OrigenDeLaCaida, RegistroDeFlujo};
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::Notify;

/// Candado unico de la serie: todas las pruebas de este target lo toman (estado global del
/// universo de simbolos y del registro de settlements por lado).
static UNIVERSO: Mutex<()> = Mutex::new(());
const SIMBOLO: &str = "BTCUSDT";
const HOST: &str = include_str!("../src/bin/god_engine.rs");
const OMS: &str = include_str!("../src/host_oms.rs");
const PLAZO: std::time::Duration = std::time::Duration::from_secs(2);

fn texto(f: &str) -> String {
    f.replace("\r\n", "\n")
}

fn orden(fuente: &str, pasos: &[&str]) {
    let fuente = texto(fuente);
    let mut resto = fuente.as_str();
    for p in pasos {
        resto = resto.split_once(p).unwrap_or_else(|| panic!("ancla ausente o fuera de orden: {p}")).1;
    }
}

// ------------------------------------------------------------------ servidor local

#[derive(Debug, Clone)]
struct Peticion {
    metodo: String,
    ruta: String,
    params: HashMap<String, String>,
}

#[derive(Default)]
struct Servidor {
    peticiones: Mutex<Vec<Peticion>>,
    /// Cantidad ejecutada que devuelve la consulta (GET) de una orden.
    ejecutada_get: Mutex<String>,
    estado_get: Mutex<String>,
    /// Rechazo firme del POST de la orden de apertura (400 con codigo).
    rechazar_post: AtomicBool,
    /// ACK perdido: el POST de la orden llega y la conexion se cierra sin respuesta.
    perder_ack: AtomicBool,
    /// El POST de la orden llega y su respuesta se retiene hasta `seguir_post`.
    retener_post: AtomicBool,
    llego_post: Notify,
    seguir_post: Notify,
    /// El GET de income llega y su respuesta se retiene hasta `seguir_income`.
    retener_income: AtomicBool,
    llego_income: Notify,
    seguir_income: Notify,
    income: Mutex<String>,
    /// Cantidad ejecutada de un cierre reduce-only.
    ejecutada_cierre: Mutex<String>,
}

impl Servidor {
    fn ordenes(&self) -> Vec<Peticion> {
        self.peticiones.lock().unwrap().iter().filter(|p| p.metodo == "POST" && p.ruta == "/fapi/v1/order").cloned().collect()
    }
    fn aperturas(&self) -> usize {
        self.ordenes().iter().filter(|p| p.params.get("reduceOnly").map(String::as_str) != Some("true")).count()
    }
    fn contar(&self, metodo: &str, ruta: &str) -> usize {
        self.peticiones.lock().unwrap().iter().filter(|p| p.metodo == metodo && p.ruta == ruta).count()
    }
    fn apalancamientos(&self) -> usize {
        self.contar("POST", "/fapi/v1/leverage")
    }
}

async fn leer(socket: &mut TcpStream) -> Option<Peticion> {
    let mut lector = BufReader::new(socket);
    let mut linea = String::new();
    if lector.read_line(&mut linea).await.ok()? == 0 {
        return None;
    }
    let primera = linea.trim_end().to_string();
    let mut largo = 0usize;
    loop {
        linea.clear();
        let n = lector.read_line(&mut linea).await.ok()?;
        if n == 0 || linea == "\r\n" || linea == "\n" {
            break;
        }
        if let Some((k, v)) = linea.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                largo = v.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut cuerpo = vec![0u8; largo];
    lector.read_exact(&mut cuerpo).await.ok()?;
    let mut partes = primera.split_whitespace();
    let metodo = partes.next()?.to_string();
    let objetivo = partes.next()?.to_string();
    let (ruta, consulta) = objetivo.split_once('?').unwrap_or((objetivo.as_str(), ""));
    let params = url::form_urlencoded::parse(consulta.as_bytes()).into_owned().collect();
    Some(Peticion { metodo, ruta: ruta.to_string(), params })
}

async fn responder(mut socket: TcpStream, estado: &str, cuerpo: String) {
    let r = format!(
        "HTTP/1.1 {estado}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{cuerpo}",
        cuerpo.len()
    );
    let _ = socket.write_all(r.as_bytes()).await;
    let _ = socket.shutdown().await;
}

fn orden_json(p: &Peticion, estado: &str, ejecutada: &str) -> String {
    let id = p.params.get("newClientOrderId").or(p.params.get("origClientOrderId")).cloned().unwrap_or_default();
    let tipo = p.params.get("type").cloned().unwrap_or_else(|| "LIMIT".into());
    let qty = p.params.get("quantity").cloned().unwrap_or_else(|| "0.010".into());
    let lado = p.params.get("side").cloned().unwrap_or_else(|| "BUY".into());
    let reduce = p.params.get("reduceOnly").map(String::as_str) == Some("true");
    format!(
        r#"{{"symbol":"{SIMBOLO}","clientOrderId":"{id}","orderId":4242,"side":"{lado}","type":"{tipo}","status":"{estado}","origQty":"{qty}","executedQty":"{ejecutada}","avgPrice":"100.0","price":"100.0","cumQuote":"1.0","reduceOnly":{reduce},"positionSide":"BOTH","updateTime":1000}}"#
    )
}

async fn servir(listener: tokio::net::TcpListener, s: Arc<Servidor>) {
    loop {
        let Ok((mut socket, _)) = listener.accept().await else { return };
        let s = Arc::clone(&s);
        tokio::spawn(async move {
            let Some(p) = leer(&mut socket).await else { return };
            s.peticiones.lock().unwrap().push(p.clone());
            let ok = "200 OK";
            match (p.metodo.as_str(), p.ruta.as_str()) {
                ("POST", "/fapi/v1/leverage") => {
                    let l = p.params.get("leverage").cloned().unwrap_or_default();
                    responder(socket, ok, format!(r#"{{"symbol":"{SIMBOLO}","leverage":{l},"maxNotionalValue":"1000000"}}"#)).await
                }
                ("POST", "/fapi/v1/order") => {
                    let cierre = p.params.get("reduceOnly").map(String::as_str) == Some("true");
                    if cierre {
                        let ej = s.ejecutada_cierre.lock().unwrap().clone();
                        let estado = if ej == p.params.get("quantity").cloned().unwrap_or_default() { "FILLED" } else { "EXPIRED" };
                        return responder(socket, ok, orden_json(&p, estado, &ej)).await;
                    }
                    if s.retener_post.load(Ordering::Acquire) {
                        s.llego_post.notify_one();
                        s.seguir_post.notified().await;
                    }
                    if s.perder_ack.load(Ordering::Acquire) {
                        drop(socket);
                        return;
                    }
                    if s.rechazar_post.load(Ordering::Acquire) {
                        return responder(socket, "400 Bad Request", r#"{"code":-2019,"msg":"Margin is insufficient."}"#.to_string()).await;
                    }
                    let cuerpo = match p.params.get("timeInForce").map(String::as_str) {
                        Some("GTX") => orden_json(&p, "NEW", "0"),
                        Some("IOC") => orden_json(&p, "FILLED", "0.010"),
                        _ => orden_json(&p, "FILLED", p.params.get("quantity").map(String::as_str).unwrap_or("0.010")),
                    };
                    responder(socket, ok, cuerpo).await
                }
                ("DELETE", "/fapi/v1/order") => responder(socket, ok, orden_json(&p, "CANCELED", "0")).await,
                ("GET", "/fapi/v1/order") => {
                    let (estado, ej) = (s.estado_get.lock().unwrap().clone(), s.ejecutada_get.lock().unwrap().clone());
                    responder(socket, ok, orden_json(&p, &estado, &ej)).await
                }
                ("GET", "/fapi/v1/income") => {
                    if s.retener_income.load(Ordering::Acquire) {
                        s.llego_income.notify_one();
                        s.seguir_income.notified().await;
                    }
                    let cuerpo = s.income.lock().unwrap().clone();
                    responder(socket, ok, cuerpo).await
                }
                ("GET", "/fapi/v2/positionRisk") => responder(
                    socket,
                    ok,
                    format!(r#"[{{"symbol":"{SIMBOLO}","positionAmt":"0.010","entryPrice":"100.0","unRealizedProfit":"0.0","positionSide":"BOTH","leverage":"1","marginType":"cross","updateTime":1000}}]"#),
                )
                .await,
                _ => responder(socket, ok, r#"{"code":-9999,"msg":"ruta inesperada del montaje"}"#.to_string()).await,
            }
        });
    }
}

// ------------------------------------------------------------------ montaje del OMS

struct Montaje {
    _universo: MutexGuard<'static, ()>,
    arena: Arc<GlobalArena>,
    sesion: Arc<OmsSession>,
    token: SessionToken,
    ejecutor: Arc<OrderExecutor>,
    servidor: Arc<Servidor>,
}

fn filas_de_comision() -> Vec<SymbolFees> {
    quantum_arena::symbols::get_active_universe_ref()
        .iter()
        .map(|s| SymbolFees { symbol: s.to_string(), maker: 0.0, taker: 0.0003 })
        .collect()
}

fn posiciones_planas() -> Vec<PositionRiskEntry> {
    quantum_arena::symbols::get_active_universe_ref()
        .iter()
        .map(|s| PositionRiskEntry { symbol: s.to_uppercase(), position_side: "BOTH".into(), ..Default::default() })
        .collect()
}

async fn montaje() -> Montaje {
    let universo = UNIVERSO.lock().unwrap_or_else(|e| e.into_inner());
    quantum_arena::symbols::update_dynamic_universe(vec![SIMBOLO.into(), "ETHUSDT".into()]);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let servidor = Arc::new(Servidor::default());
    *servidor.ejecutada_get.lock().unwrap() = "0.010".into();
    *servidor.estado_get.lock().unwrap() = "CANCELED".into();
    *servidor.income.lock().unwrap() = "[]".into();
    *servidor.ejecutada_cierre.lock().unwrap() = "0.010".into();
    tokio::spawn(servir(listener, Arc::clone(&servidor)));
    let mut ejecutor = OrderExecutor::new(String::new(), "offline-contract".into(), false).con_base_de_prueba(&base);
    ejecutor.set_paper_trading(false);
    ejecutor.is_hedge_mode.store(false, Ordering::Relaxed);
    let ejecutor = Arc::new(ejecutor);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let sesion = Arc::new(OmsSession::new(Arc::new(ArcSwap::from(Arc::clone(&ejecutor))), Arc::new(AtomicBool::new(false))));
    let token = sesion.snapshot_token();
    let fees = FeeSchedule::new(token.generation(), filas_de_comision()).map_err(|e| format!("{e:?}"));
    sesion.apply_snapshot(&token, Ok(posiciones_planas()), Ok(Vec::new()), fees, &arena, 1_000);
    {
        let mut libro = arena.config.entry_publication_gate.lock().unwrap();
        libro.activar(&arena);
        libro.saldo_aceptado(&arena, 100.0, Some(100));
        arena.capital_incompleto.store(false, Ordering::Release);
    }
    Montaje { _universo: universo, arena, sesion, token, ejecutor, servidor }
}

impl Montaje {
    /// La reserva del nucleo bajo la puerta (`admite` y `reservar_con_base`).
    fn reservar_clave(&self, clave: ClaveReserva) {
        let mut libro = self.arena.config.entry_publication_gate.lock().unwrap();
        let mut capital = Some(self.arena.unified_capital.load(Ordering::Relaxed));
        if !libro.admite(&self.arena, capital) {
            capital = Some(self.arena.unified_capital.load(Ordering::Relaxed));
            assert!(libro.admite(&self.arena, capital), "fixture: la decision recalculada se reserva");
        }
        libro.reservar_con_base(&self.arena, clave, 0.04, capital);
    }
    fn reservar(&self, generation: u64) -> ClaveReserva {
        let clave = ClaveReserva { coin_id: 0, slot: 0, generation };
        self.reservar_clave(clave);
        clave
    }
    /// Una reserva REAL del nucleo: ranura abierta sin confirmar con su margen, y la reserva del
    /// libro con la misma clave (coin, ranura, generacion).
    fn reserva_real(&self, slot: usize) -> EntryReservation {
        let p = self.arena.coins[0].positions.get_slot(slot);
        assert!(p.open_with_tau_and_fee(true, 100.0, 0.01, 10.0, 1_000, 110.0, 90.0, PositionHorizon::Continuous, 0.7, 0.8, 0.04, 60_000));
        self.arena.used_margin.fetch_add(10.0, Ordering::Relaxed);
        let generation = p.generation.load(Ordering::Acquire);
        let r = EntryReservation { coin_id: 0, slot, generation, symbol: SIMBOLO.into() };
        self.reservar_clave(r.clave());
        r
    }
    fn reservado(&self) -> f64 {
        self.arena.config.entry_publication_gate.lock().unwrap().reservado()
    }
    fn compromisos(&self) -> Vec<quantum_arena::dimensionable::CompromisoDeEntrega> {
        self.arena.config.entry_publication_gate.lock().unwrap().compromisos()
    }
    fn revision(&self) -> u64 {
        self.arena.config.entry_publication_gate.lock().unwrap().revision()
    }
}

/// El settlement real del lado; con plazo para que un caso fallido que no lo solto no cuelgue el resto.
async fn adquirir(r: &EntryReservation) -> EntrySettlement {
    tokio::time::timeout(PLAZO, EntrySettlement::acquire(r.clone(), true)).await.expect("settlement de otro caso sin soltar (montaje)")
}

fn pedido(ruta: EntryRoute, id: &'static str) -> EntryRequest<'static> {
    EntryRequest { symbol: SIMBOLO, is_long: true, quantity: 0.01, step_size: 0.001, tick_size: 0.1, leverage: 1, client_order_id: id, route: ruta }
}

const CORTES: [&str; 6] = ["revision", "importe ABA", "veto del libro", "veto del puente", "caida y reapertura", "sesion"];

fn aplicar_corte_en(a: &GlobalArena, sesion: &OmsSession, token: &SessionToken, corte: &str) {
    match corte {
        "revision" => {
            let mut l = a.config.entry_publication_gate.lock().unwrap();
            a.capital_incompleto.store(true, Ordering::Release);
            l.saldo_aceptado(a, 90.0, Some(200));
            a.capital_incompleto.store(false, Ordering::Release);
        }
        "importe ABA" => {
            let mut l = a.config.entry_publication_gate.lock().unwrap();
            l.saldo_aceptado(a, 90.0, Some(200));
            l.saldo_aceptado(a, 100.0, Some(300));
        }
        "veto del libro" => a.config.entry_publication_gate.lock().unwrap().cierre_externo(a),
        "veto del puente" => a.capital_incompleto.store(true, Ordering::Release),
        "caida y reapertura" => {
            a.capital_incompleto.store(true, Ordering::Release);
            let mut l = a.config.entry_publication_gate.lock().unwrap();
            l.saldo_aceptado(a, 90.0, Some(200));
            drop(l);
            a.capital_incompleto.store(false, Ordering::Release);
        }
        "sesion" => assert!(sesion.begin_refresh(token, a)),
        otro => panic!("corte desconocido {otro}"),
    }
}

/// Barrera que aplica `corte` en el k-esimo compromiso (antes o despues de el) y cuenta llamadas.
fn barrera(m: &Montaje, corte: Option<&'static str>, en: usize, tras: bool, rev_previa: Arc<AtomicU64>) -> Arc<AtomicUsize> {
    let n = Arc::new(AtomicUsize::new(0));
    let (a, ses, tok, n2) = (Arc::clone(&m.arena), Arc::clone(&m.sesion), m.token.clone(), Arc::clone(&n));
    let f: Box<dyn Fn()> = Box::new(move || {
        let k = n2.fetch_add(1, Ordering::SeqCst) + 1;
        if let (Some(c), true) = (corte, k == en) {
            rev_previa.store(a.config.entry_publication_gate.lock().unwrap().revision(), Ordering::SeqCst);
            aplicar_corte_en(&a, &ses, &tok, c);
        }
    });
    if tras {
        quantum_arena::dimensionable::fijar_barrera_tras_compromiso(Some(f));
    } else {
        quantum_arena::dimensionable::fijar_barrera_de_compromiso(Some(f));
    }
    n
}

fn quitar_barreras() {
    quantum_arena::dimensionable::fijar_barrera_de_compromiso(None);
    quantum_arena::dimensionable::fijar_barrera_tras_compromiso(None);
}

// ================================================================== A1

/// L-A1 (matriz :37). Por ruta REST de apertura (MARKET, IOC, GTX) y por corte (revision, importe
/// ABA, veto del libro, veto del puente, caida y reapertura, sesion):
/// - el corte gana ANTES del compromiso del apalancamiento: 0 apalancamientos / 0 ordenes;
/// - gana antes del compromiso de la orden, con el apalancamiento ya comprometido: 1 / 0;
/// - gana DESPUES del compromiso de la orden (barrera entre el compromiso y la entrega): el
///   paquete sale UNA vez y su testigo conserva la revision y la sesion que lo aceptaron;
/// - control sin corte: 1 / 1 y dos testigos (apalancamiento y orden).
/// Ademas: remanente MARKET del maker con corte despues de su compromiso (sale una vez), iceberg
/// no soportado sin mutacion ni testigo, salidas sin compromiso, y una publicacion concurrente
/// mientras el ACK de un envio valido esta retenido (ningun candado durante la red).
#[tokio::test]
async fn v4l_a1_compromiso_local_host_revalida_sin_esperar_red() {
    let rutas = [("MARKET", EntryRoute::Market), ("IOC", EntryRoute::Ioc { price: 101.0 }), ("GTX", EntryRoute::Maker { price: 99.0 })];
    let mut casos = 0usize;
    for (nombre, ruta) in rutas {
        // Control sin corte.
        {
            let m = montaje().await;
            let clave = m.reservar(1);
            let n = barrera(&m, None, 0, false, Arc::new(AtomicU64::new(0)));
            let res = m.sesion.dispatch_entry_reservada(&m.token, m.ejecutor.as_ref(), &pedido(ruta, "cL_v4l_ctl"), &m.arena, Some(clave)).await;
            quitar_barreras();
            assert!(n.load(Ordering::SeqCst) >= 2, "{nombre}: la barrera del compromiso no corrio (montaje)");
            assert_eq!((m.servidor.apalancamientos(), m.servidor.aperturas()), (1, 1), "{nombre}: control ({res:?})");
            let c = m.compromisos();
            assert!(c.len() >= 2 && c[0].fase == FaseDeEntrega::Apalancamiento && c[1].fase == FaseDeEntrega::Orden, "{nombre}: testigos {c:?}");
            assert!(c.iter().all(|t| t.reserva == Some(clave) && t.sesion == m.token.generation()), "{nombre}: testigo sin identidad: {c:?}");
            casos += 1;
        }
        for corte in CORTES {
            // Antes del compromiso: del apalancamiento (1) o de la orden (2).
            for en in [1usize, 2] {
                let m = montaje().await;
                let clave = m.reservar(1);
                let n = barrera(&m, Some(corte), en, false, Arc::new(AtomicU64::new(0)));
                let res = m.sesion.dispatch_entry_reservada(&m.token, m.ejecutor.as_ref(), &pedido(ruta, "cL_v4l_antes"), &m.arena, Some(clave)).await;
                quitar_barreras();
                assert!(n.load(Ordering::SeqCst) >= en, "{nombre}/{corte}: la barrera no alcanzo el compromiso {en} (montaje)");
                assert!(res.is_err(), "{nombre}/{corte}/{en}: entrega aceptada tras el corte");
                let esperado = if en == 1 { (0, 0) } else { (1, 0) };
                assert_eq!((m.servidor.apalancamientos(), m.servidor.aperturas()), esperado, "{nombre}/{corte}/{en}");
                let c = m.compromisos();
                assert_eq!(c.len(), en - 1, "{nombre}/{corte}/{en}: testigo de un paquete retenido: {c:?}");
                casos += 1;
            }
            // Despues del compromiso de la orden: el paquete ya es irrevocable.
            let m = montaje().await;
            let clave = m.reservar(1);
            let rev_previa = Arc::new(AtomicU64::new(u64::MAX));
            let n = barrera(&m, Some(corte), 2, true, Arc::clone(&rev_previa));
            let res = m.sesion.dispatch_entry_reservada(&m.token, m.ejecutor.as_ref(), &pedido(ruta, "cL_v4l_tras"), &m.arena, Some(clave)).await;
            quitar_barreras();
            assert!(n.load(Ordering::SeqCst) >= 2, "{nombre}/{corte}: la barrera tras el compromiso no corrio (montaje)");
            assert_eq!((m.servidor.apalancamientos(), m.servidor.aperturas()), (1, 1), "{nombre}/{corte}: el paquete comprometido no salio una sola vez ({res:?})");
            let c = m.compromisos();
            let ordenes: Vec<_> = c.iter().filter(|t| t.fase == FaseDeEntrega::Orden).collect();
            assert_eq!(ordenes.len(), 1, "{nombre}/{corte}: testigos de orden {c:?}");
            let rev = rev_previa.load(Ordering::SeqCst);
            assert_eq!(ordenes[0].revision, rev, "{nombre}/{corte}: el testigo se reetiqueto con otra revision");
            assert_eq!(ordenes[0].sesion, m.token.generation(), "{nombre}/{corte}: el testigo es de otra sesion");
            if matches!(corte, "revision" | "importe ABA" | "veto del libro" | "caida y reapertura") {
                assert!(m.revision() > rev, "{nombre}/{corte}: el corte no abrio revision (montaje)");
            }
            casos += 1;
        }
    }
    // Remanente MARKET de la ruta maker (GTX parcial 0.005): corte DESPUES de su compromiso (el
    // tercero: apalancamiento, GTX, remanente): el remanente ya comprometido sale una vez.
    for corte in CORTES {
        let m = montaje().await;
        *m.servidor.ejecutada_get.lock().unwrap() = "0.005".into();
        let clave = m.reservar(1);
        let rev = Arc::new(AtomicU64::new(u64::MAX));
        let n = barrera(&m, Some(corte), 3, true, Arc::clone(&rev));
        let _ = m.sesion.dispatch_entry_reservada(&m.token, m.ejecutor.as_ref(), &pedido(EntryRoute::Maker { price: 99.0 }, "mc_v4l_rem"), &m.arena, Some(clave)).await;
        quitar_barreras();
        assert!(n.load(Ordering::SeqCst) >= 3, "{corte}: el remanente no llego a su compromiso (montaje)");
        let ordenes = m.servidor.ordenes();
        let market = ordenes.iter().filter(|p| p.params.get("type").map(String::as_str) == Some("MARKET")).count();
        assert_eq!(market, 1, "{corte}: el remanente comprometido no salio una sola vez");
        let c = m.compromisos();
        assert_eq!(c.iter().filter(|t| t.fase == FaseDeEntrega::Orden).count(), 2, "{corte}: testigos {c:?}");
        assert_eq!(c.last().unwrap().revision, rev.load(Ordering::SeqCst), "{corte}: el remanente se reetiqueto");
        casos += 1;
    }
    // Iceberg nativo no soportado: sin apalancamiento, sin orden, sin testigo (exclusion antes de
    // cualquier mutacion). WS de ordenes: stub deshabilitado (regresion
    // `capital_v4j_admision_contract::veto_y_revision_en_la_entrega_local_por_ruta`).
    {
        let m = montaje().await;
        let clave = m.reservar(1);
        let antes = m.reservado();
        let res = m
            .sesion
            .dispatch_entry_reservada(&m.token, m.ejecutor.as_ref(), &pedido(EntryRoute::Iceberg { price: 99.0, visible_quantity: 0.005 }, "cL_v4l_ice"), &m.arena, Some(clave))
            .await;
        assert!(res.is_err(), "iceberg no soportado acepto la entrega");
        assert_eq!((m.servidor.apalancamientos(), m.servidor.ordenes().len(), m.compromisos().len()), (0, 0, 0), "iceberg muto algo");
        assert_eq!(m.reservado(), antes, "iceberg toco la reserva");
        casos += 1;
    }
    // Salidas sin cruce: ni el cierre defensivo ni la cancelacion pasan por el compromiso.
    {
        let m = montaje().await;
        aplicar_corte_en(&m.arena, &m.sesion, &m.token, "veto del puente");
        let n = barrera(&m, None, 0, false, Arc::new(AtomicU64::new(0)));
        let _ = m.ejecutor.execute_reduce_only_market_with_reconciliation(SIMBOLO, true, 0.01, 0.001).await;
        let _ = m.ejecutor.cancel_order(SIMBOLO, "cL_v4l_cualquiera").await;
        quitar_barreras();
        assert_eq!(n.load(Ordering::SeqCst), 0, "una salida paso por el compromiso de entrada");
        assert!(m.servidor.ordenes().iter().any(|p| p.params.get("reduceOnly").map(String::as_str) == Some("true")));
        assert_eq!(m.servidor.contar("DELETE", "/fapi/v1/order"), 1);
        assert!(m.compromisos().is_empty(), "una salida dejo testigo de entrada");
        casos += 1;
    }
    // ACK retenido: con el POST valido ya entregado y su respuesta retenida, un escritor publica
    // un saldo nuevo (puerta del libro) y el refresco de la sesion corre: ningun candado se
    // sostiene durante la red.
    {
        let m = montaje().await;
        let clave = m.reservar(1);
        m.servidor.retener_post.store(true, Ordering::Release);
        let pedido_ack = pedido(EntryRoute::Market, "cL_v4l_ack");
        let envio = m.sesion.dispatch_entry_reservada(&m.token, m.ejecutor.as_ref(), &pedido_ack, &m.arena, Some(clave));
        let publicar = async {
            m.servidor.llego_post.notified().await;
            // `try_lock`: si un candado se sostuviera durante la red, falla aqui (no cuelga el hilo).
            let publicado = m.arena.config.entry_publication_gate.try_lock().map(|mut l| {
                l.saldo_aceptado(&m.arena, 98.0, Some(500));
                l.revision()
            });
            let refresco = m.sesion.begin_refresh(&m.token, &m.arena);
            m.servidor.seguir_post.notify_one();
            (publicado.map_err(|_| "puerta del libro tomada durante la red"), refresco)
        };
        let (res, (publicado, refresco)) = tokio::join!(envio, publicar);
        assert!(publicado.is_ok() && refresco, "la publicacion o el refresco esperaron al ACK retenido ({publicado:?}, {refresco})");
        assert!(res.is_ok(), "el envio valido no termino: {res:?}");
        let c = m.compromisos();
        assert!(c.last().unwrap().revision < publicado.unwrap(), "el testigo tomo la revision publicada durante la red");
        casos += 1;
    }
    println!("v4l_a1_compromiso_local: {casos} casos");
}

/// L-A1H (matriz :38). Flujo de resultado del HOST (`resolver_resultado_de_entrada`) con reserva
/// REAL (ranura abierta sin confirmar, margen y reserva del libro) y settlement real. Sin cancelar
/// a mano: el error real del envio llega a la resolucion productiva.
#[tokio::test]
async fn v4l_a1_host_revierte_una_vez_y_resuelve_maker_parcial() {
    // 1. Rechazo firme sin compromiso ni ejecucion: cancela la generacion exacta UNA vez, devuelve
    //    solo lo aun reservado y suelta el settlement; la otra posicion queda intacta.
    {
        let m = montaje().await;
        let otra = m.reserva_real(1);
        otra.confirm(&m.arena).unwrap();
        let r = m.reserva_real(0);
        let (margen0, reservado0) = (m.arena.used_margin.load(Ordering::Relaxed), m.reservado());
        let _settlement = adquirir(&r).await;
        m.servidor.rechazar_post.store(true, Ordering::Release);
        let res = m.sesion.dispatch_entry_reservada(&m.token, m.ejecutor.as_ref(), &pedido(EntryRoute::Market, "cL_v4l_firme"), &m.arena, Some(r.clave())).await;
        assert!(res.as_ref().is_err_and(|e| !e.starts_with("AMBIGUOUS")), "fixture: rechazo firme ({res:?})");
        let d = quantum_engine::host_oms::resolver_resultado_de_entrada(res, &m.sesion, &m.token, &m.arena, &m.ejecutor, &r, "cL_v4l_firme", 0.01, PLAZO).await;
        assert_eq!(d, DesenlaceDeEntrada::Revertida);
        assert!(!m.arena.coins[0].positions.get_slot(0).is_open(), "la ranura rechazada sigue abierta");
        assert_eq!(m.arena.used_margin.load(Ordering::Relaxed), margen0 - 10.0, "margen devuelto distinto del de la reserva");
        assert!((reservado0 - m.reservado() - 0.04).abs() < 1e-12, "el libro devolvio otra cosa que la reserva debida");
        assert!(!EntrySettlement::release_for(&r), "el settlement no se solto (o se solto dos veces)");
        assert!(!quantum_engine::host_oms::revertir_entrada(&m.arena, &r), "segunda reversion de la misma generacion");
        assert_eq!(m.arena.used_margin.load(Ordering::Relaxed), margen0 - 10.0, "debito/credito duplicado");
        assert!(m.arena.coins[0].positions.get_slot(1).is_open(), "la otra posicion se toco");
    }
    println!("a1h: caso 1 hecho");
    // 2. Borde de A2: rechazo firme con ejecucion ACREDITADA por el stream (fill atribuido a la
    //    reserva): no revierte, conserva reserva, cargo y settlement.
    {
        let m = montaje().await;
        let r = m.reserva_real(0);
        let _settlement = adquirir(&r).await;
        {
            let mut l = m.arena.config.entry_publication_gate.lock().unwrap();
            l.vincular_orden_con_cantidad(r.clave(), "cL_v4l_a2", 0.01);
            let fill = quantum_arena::dimensionable::FillDelStream {
                trade_id: 77,
                cantidad: Some(0.004),
                cantidad_e8: Some(400_000),
                acumulado_e8: Some(400_000),
                comision: quantum_arena::dimensionable::ComisionDeFill::Activo(0.012),
                marca: Some(150),
            };
            assert!(!l.fill_de_entrada_del_stream(&m.arena, 0, "cL_v4l_a2", &fill), "fixture: comision conocida (sin veto)");
            assert!(l.ejecucion_observada(r.clave()), "fixture: fill atribuido a la reserva");
        }
        m.servidor.rechazar_post.store(true, Ordering::Release);
        let res: Result<(), String> = Err("Binance API Error: {\"code\":-2019}".into());
        let d = quantum_engine::host_oms::resolver_resultado_de_entrada(res, &m.sesion, &m.token, &m.arena, &m.ejecutor, &r, "cL_v4l_a2", 0.01, PLAZO).await;
        assert_eq!(d, DesenlaceDeEntrada::Conservada, "terminal contradictoria con ejecucion acreditada solto la reserva");
        assert!(m.arena.coins[0].positions.get_slot(0).is_open());
        assert!(m.reservado() > 0.0);
        assert!(EntrySettlement::release_for(&r), "el settlement se solto con ejecucion acreditada");
    }
    println!("a1h: caso 2 hecho");
    // 3. ACK perdido (envio AMBIGUO): reconsulta ESA identidad, sin reenvio. Terminal con
    //    ejecucion: confirma lo ejecutado; terminal sin ejecucion: revierte una vez.
    for (estado, ej, esperado) in [("FILLED", "0.010", "confirma"), ("CANCELED", "0", "revierte")] {
        let m = montaje().await;
        let r = m.reserva_real(0);
        let _settlement = adquirir(&r).await;
        println!("a1h: caso 3 {estado} empieza");
        m.servidor.perder_ack.store(true, Ordering::Release);
        *m.servidor.estado_get.lock().unwrap() = estado.into();
        *m.servidor.ejecutada_get.lock().unwrap() = ej.into();
        let res = m.sesion.dispatch_entry_reservada(&m.token, m.ejecutor.as_ref(), &pedido(EntryRoute::Market, "cL_v4l_ack"), &m.arena, Some(r.clave())).await;
        assert!(res.as_ref().is_err_and(|e| e.starts_with("AMBIGUOUS")), "fixture: ACK perdido ({res:?})");
        let d = quantum_engine::host_oms::resolver_resultado_de_entrada(res, &m.sesion, &m.token, &m.arena, &m.ejecutor, &r, "cL_v4l_ack", 0.01, PLAZO).await;
        assert_eq!(m.servidor.aperturas(), 1, "{estado}: reenvio a ciegas");
        assert!(m.servidor.contar("GET", "/fapi/v1/order") >= 1, "{estado}: sin reconsulta de la identidad");
        match esperado {
            "confirma" => {
                assert_eq!(d, DesenlaceDeEntrada::Confirmada { ejecutada: 0.01, envio_ambiguo: true });
                assert!(m.arena.coins[0].positions.get_slot(0).exchange_confirmed.load(Ordering::Acquire));
                // Confirmada: el settlement sigue retenido hasta la proteccion (lo suelta el host).
                assert!(EntrySettlement::release_for(&r), "el settlement se solto antes de la proteccion");
            }
            _ => {
                assert_eq!(d, DesenlaceDeEntrada::Revertida);
                assert!(!EntrySettlement::release_for(&r));
            }
        }
    }
    println!("a1h: caso 3 hecho");
    // 4. GTX parcialmente ejecutada (0.005 de 0.010) con el remanente RETENIDO antes de su
    //    compromiso (veto del puente): el padre ejecuto, el hijo no salio. La resolucion conserva
    //    reserva, cargo, settlement y proteccion; no confirma el total ni reenvia.
    {
        let m = montaje().await;
        *m.servidor.ejecutada_get.lock().unwrap() = "0.005".into();
        let r = m.reserva_real(0);
        let _settlement = adquirir(&r).await;
        let n = barrera(&m, Some("veto del puente"), 3, false, Arc::new(AtomicU64::new(0)));
        let res = m.sesion.dispatch_entry_reservada(&m.token, m.ejecutor.as_ref(), &pedido(EntryRoute::Maker { price: 99.0 }, "mc_v4l_parcial"), &m.arena, Some(r.clave())).await;
        quitar_barreras();
        assert!(n.load(Ordering::SeqCst) >= 3, "fixture: el remanente no llego a su compromiso");
        assert!(res.as_ref().is_err_and(|e| e.starts_with("MAKER_CHASE_UNVERIFIED")), "fixture: {res:?}");
        let ordenes_antes = m.servidor.ordenes().len();
        let d = quantum_engine::host_oms::resolver_resultado_de_entrada(res, &m.sesion, &m.token, &m.arena, &m.ejecutor, &r, "mc_v4l_parcial", 0.01, PLAZO).await;
        assert_eq!(d, DesenlaceDeEntrada::Conservada, "el padre parcial se resolvio como total o se revirtio");
        assert_eq!(m.servidor.ordenes().len(), ordenes_antes, "reenvio tras la resolucion");
        assert!(m.arena.coins[0].positions.get_slot(0).is_open() && !m.arena.coins[0].positions.get_slot(0).exchange_confirmed.load(Ordering::Acquire));
        assert!(m.reservado() > 0.0 && EntrySettlement::release_for(&r), "cargo o settlement perdidos");
    }
    println!("a1h: caso 4 hecho");
    // 5. GTX parcial con remanente IMPOSIBLE (menor que el paso): el host confirma SOLO lo
    //    ejecutado acreditado por el terminal del padre.
    {
        let m = montaje().await;
        *m.servidor.ejecutada_get.lock().unwrap() = "0.0095".into();
        let r = m.reserva_real(0);
        let _settlement = adquirir(&r).await;
        let res = m.sesion.dispatch_entry_reservada(&m.token, m.ejecutor.as_ref(), &pedido(EntryRoute::Maker { price: 99.0 }, "mc_v4l_casi"), &m.arena, Some(r.clave())).await;
        let d = quantum_engine::host_oms::resolver_resultado_de_entrada(res, &m.sesion, &m.token, &m.arena, &m.ejecutor, &r, "mc_v4l_casi", 0.01, PLAZO).await;
        assert_eq!(d, DesenlaceDeEntrada::Confirmada { ejecutada: 0.0095, envio_ambiguo: false }, "confirmo otra cantidad que la del padre");
        let q = m.arena.coins[0].positions.get_slot(0).quantity.load(Ordering::Relaxed).abs();
        assert!((q - 0.0095).abs() < 1e-12, "la ranura no quedo con lo ejecutado: {q}");
        assert!(EntrySettlement::release_for(&r), "settlement suelto antes de la proteccion");
    }
    println!("a1h: caso 5 hecho");
    // 6. Control vigente: IOC ejecutada entera por el terminal de la propia orden.
    {
        let m = montaje().await;
        let r = m.reserva_real(0);
        let _settlement = adquirir(&r).await;
        let res = m.sesion.dispatch_entry_reservada(&m.token, m.ejecutor.as_ref(), &pedido(EntryRoute::Ioc { price: 101.0 }, "cL_v4l_ioc"), &m.arena, Some(r.clave())).await;
        let d = quantum_engine::host_oms::resolver_resultado_de_entrada(res, &m.sesion, &m.token, &m.arena, &m.ejecutor, &r, "cL_v4l_ioc", 0.01, PLAZO).await;
        assert_eq!(d, DesenlaceDeEntrada::Confirmada { ejecutada: 0.01, envio_ambiguo: false });
        assert!(EntrySettlement::release_for(&r), "settlement suelto antes de la proteccion");
    }
    // El host corre los MISMOS pasos en linea (anclas): consulta de la identidad, borde de A2 antes
    // de revertir, reversion exacta y settlement, y confirmacion con la marca terminal.
    // V4M (D1): el host llama a la MISMA resolucion (ya no hay brazo propio en linea).
    orden(HOST, &["let entry_result = entry_oms.dispatch_entry_reservada(", "let desenlace = quantum_engine::host_oms::resolver_resultado_de_entrada(", "entry_result,", "DesenlaceDeEntrada::Confirmada { ejecutada, envio_ambiguo: false } => ejecutada,"]);
    orden(OMS, &["pub async fn resolver_resultado_de_entrada(", "execution_engine::entry_dispatch::resolve_terminal_entry_until(", "entrada_con_ejecucion_acreditada(arena, executor, reservation, client_id)", "revertir_entrada(arena, reservation)", "reservation.confirmar_llenado_con_marca(arena, Some(ejecutada), marca)"]);
}

// ================================================================== plano del freno

struct Plano {
    puente: Arc<PuenteCuenta>,
    arena: Arc<GlobalArena>,
    oms: Arc<AtomicU64>,
    riesgo: RiskEngine,
}



fn plano(cuenta: u64) -> Plano {
    plano_con_arena(cuenta, GlobalArena::build_in_own_stack(100.0))
}

fn plano_con_arena(cuenta: u64, arena: Arc<GlobalArena>) -> Plano {
    let oms = Arc::new(AtomicU64::new(1));
    let puente = Arc::new(PuenteCuenta::con_sesion(1));
    {
        let oms = Arc::clone(&oms);
        puente.vincular_testigo_de_sesion(move || oms.load(Ordering::SeqCst));
    }
    {
        let mut libro = arena.config.entry_publication_gate.lock().unwrap();
        libro.activar(&arena);
        libro.fijar_cuenta(&arena, cuenta);
    }
    puente.anunciar_cuenta(cuenta);
    {
        let arena = Arc::clone(&arena);
        puente.vincular_freno(move |equity| drawdown::publicar_equity_del_freno(&arena, equity));
    }
    {
        let (arena, oms) = (Arc::clone(&arena), Arc::clone(&oms));
        puente.vincular_publicador(move |p| {
            if p.veta {
                arena.config.entry_publication_gate.lock().unwrap().veto_del_puente();
                arena.capital_incompleto.store(true, Ordering::Release);
                return false;
            }
            let Some((saldo, marca)) = p.vigente else {
                arena.capital_incompleto.store(true, Ordering::Release);
                return false;
            };
            if oms.load(Ordering::SeqCst) != p.sesion {
                return false;
            }
            let mut l = arena.config.entry_publication_gate.lock().unwrap();
            l.fijar_alcance(p.era, p.sesion);
            l.saldo_aceptado(&arena, saldo, marca);
            drop(l);
            arena.capital_incompleto.store(false, Ordering::Release);
            true
        });
    }
    puente.abrir_socket(puente.era()).unwrap();
    Plano { puente, arena, oms, riesgo: RiskEngine::new(100.0) }
}

fn lectura(saldo: f64, upnl: Option<f64>, t: u64, completa: bool) -> LecturaCuenta {
    use execution_engine::puente_cuenta::{DatoPosicion, PosicionRest};
    let mut posiciones = HashMap::new();
    if let Some(u) = upnl {
        posiciones.insert((SIMBOLO.to_string(), "BOTH".to_string()), PosicionRest { dato: DatoPosicion::Abierta { cantidad: 0.01, upnl: u }, tiempo_ms: Some(t) });
    }
    LecturaCuenta { saldo, tiempo_saldo_ms: Some(t), posiciones: completa.then_some(posiciones), rechazo_arranque: None, upnl_del_activo: Some(upnl.unwrap_or(0.0)) }
}

impl Plano {
    fn rest(&self, saldo: f64, upnl: Option<f64>, t: u64) {
        let _ = self.puente.observar_rest_de_sesion(self.puente.ticket(), self.puente.sesion(), &lectura(saldo, upnl, t, true));
    }
    fn rest_incompleta(&self, saldo: f64, upnl: f64, t: u64) {
        let _ = self.puente.observar_rest_de_sesion(self.puente.ticket(), self.puente.sesion(), &lectura(saldo, Some(upnl), t, false));
    }
    /// La decision REAL de Risk: (vetada por drawdown, rechazo, vista con la que decidio).
    fn decision(&mut self) -> (bool, Option<usize>, Option<drawdown::VistaDelFreno>) {
        let intento = SignalIntent { signal: SignalType::Long, confidence: 0.9, win_probability: 0.9, expected_duration_ms: 60_000, ..Default::default() };
        let r = self.riesgo.evaluate_quantum_order(0, &intento, &self.arena);
        let rechazo = risk_engine::ultimo_rechazo();
        (r.signal == SignalType::Flat && rechazo == Some(risk_engine::REJ_DRAWDOWN), rechazo, self.riesgo.vista_del_freno())
    }
    /// ADAPTADOR del paso del lazo del inmune (anclado en `anclas_v4l`): una revision del plano bajo
    /// el mutex del puente, con su testigo (la revision que deja al cerrarse).
    fn paso_del_host(&self) -> (f64, f64, u64, u64) {
        self.puente.con_equity_del_freno(|equity| {
            let plano = drawdown::EscrituraDelPlano::abrir(&self.arena);
            let cap_exchange = equity.unwrap_or(f64::NAN);
            let pendiente = self.arena.pico_pendiente.load(Ordering::Acquire);
            self.arena.capital_exchange.store(if cap_exchange.is_finite() && !pendiente { cap_exchange } else { f64::NAN }, Ordering::Release);
            let cap = drawdown::capital_del_freno(&self.arena);
            let pico = drawdown::registrar_pico(&self.arena.pico_capital, cap);
            (cap, pico, plano.revision_al_cerrar(), drawdown::ambito_del_plano(&self.arena))
        })
    }
    /// ADAPTADOR del arranque del pico del host (`aplicar_arranque_pico` bajo el mutex del puente y
    /// dentro de una escritura del plano, anclado): pico vacio, el persistido compatible o ilegible.
    fn arranque(&self, rutas: &drawdown::RutasPico) {
        self.puente.con_equity_del_freno(|_| {
            let _plano = drawdown::EscrituraDelPlano::abrir(&self.arena);
            self.arena.pico_capital.store(0.0, Ordering::Release);
            match drawdown::decidir_arranque_pico(rutas).carga {
                drawdown::CargaPico::Leido(g) => {
                    drawdown::registrar_pico(&self.arena.pico_capital, g);
                }
                drawdown::CargaPico::Ilegible(_) => self.arena.pico_ilegible.store(true, Ordering::Relaxed),
                _ => {}
            }
            drawdown::pico_restaurado(&self.arena);
        });
    }
    /// ADAPTADOR del relevo del host (anclado): con otra cuenta, ausencia, pico vacio y ambito nuevo
    /// en UNA revision del plano, bajo el mutex del puente.
    fn relevo(&self, cuenta: u64) -> u64 {
        let sesion = self.oms.load(Ordering::SeqCst) + 1;
        let (_, era) = self.puente.relevar_sesion(|| {
            self.oms.store(sesion, Ordering::SeqCst);
            self.puente.anunciar_cuenta(cuenta);
            let mut libro = self.arena.config.entry_publication_gate.lock().unwrap();
            if libro.cuenta().is_some_and(|c| c != cuenta) {
                drawdown::escribir_plano_del_freno(&self.arena, || {
                    self.arena.capital_exchange.store(f64::NAN, Ordering::Release);
                    self.arena.pico_capital.store(0.0, Ordering::Release);
                    drawdown::preparar_ambito_del_pico(&self.arena);
                });
            }
            libro.fijar_cuenta(&self.arena, cuenta);
            ((), Some(sesion))
        });
        let era = era.unwrap();
        self.puente.abrir_socket(era).unwrap();
        era
    }
}

fn anclas_v4l() {
    orden(HOST, &[
        "let (mut pico_ilegible, mut marca_pendiente) = puente_imm.con_equity_del_freno(|_| {",
        "let _plano = risk_engine::drawdown::EscrituraDelPlano::abrir(&arena_imm);",
        "aplicar_arranque_pico(&arena_imm, &rutas_pico, false)",
    ]);
    orden(HOST, &[
        "let (cap, peak_capital, ruta_del_pico, ambito_de_la_evidencia) = puente_imm.con_equity_del_freno(|equity| {",
        "let _plano = risk_engine::drawdown::EscrituraDelPlano::abrir(&arena_imm);",
        "let pendiente = arena_imm.pico_pendiente.load(Ordering::Acquire);",
        "risk_engine::drawdown::registrar_pico(&arena_imm.pico_capital, cap);",
        "revision_de_la_evidencia = _plano.revision_al_cerrar();",
    ]);
    orden(HOST, &[
        "if libro_relevo.cuenta().is_some_and(|c| c != cuenta_del_relevo) {",
        "risk_engine::drawdown::escribir_plano_del_freno(&arena_real, || {",
        "arena_real.capital_exchange.store(f64::NAN, Ordering::Release);",
        "arena_real.pico_capital.store(0.0, Ordering::Release);",
        "risk_engine::drawdown::preparar_ambito_del_pico(&arena_real);",
        "ambito_relevado = true;",
        "libro_relevo.fijar_cuenta(&arena_real, cuenta_del_relevo);",
        "execution_engine::aplanado_inmune::relevar_causa_patrimonial(",
    ]);
}

/// L-F2 (matriz :39). Serie de Criterios V4J:24 (dimensionable 95; equity 100, 103, 100, 87):
/// el paso del host etiqueta su decision con la revision que dejo y Risk decide con la MISMA
/// revision (consumidor real). Con el escritor real retenido a mitad (otro hilo) Risk no decide con
/// la mezcla (veta) y un SEGUNDO escritor (el observador del puente publicando la siguiente muestra)
/// no se intercala: queda en espera del escritor unico; al soltar, la vista es la de la ultima muestra
/// con su revision par. Control sin escritura: Risk no veta por drawdown la propuesta valida.
#[test]
fn v4l_f2_host_y_risk_con_decision_y_revision_observadas() {
    let _g = UNIVERSO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_v4l();
    let mut m = plano(1);
    m.rest(100.0, Some(0.0), 100);
    m.arena.config.entry_publication_gate.lock().unwrap().debito_cierre_local(&m.arena, ClaveReserva { coin_id: 0, slot: 1, generation: 1 }, 5.0);
    assert_eq!(m.arena.unified_capital.load(Ordering::Relaxed), 95.0, "fixture: dimensionable");
    let _ = m.paso_del_host();
    let (vetada, _, vista) = m.decision();
    assert!(!vetada && vista.is_some_and(|v| v.lista), "control sin escritura: la propuesta valida se veto por drawdown ({vista:?})");
    for (upnl, t) in [(3.0, 200), (0.0, 300), (-13.0, 400)] {
        m.rest(100.0, Some(upnl), t);
        let (cap, pico, revision, ambito) = m.paso_del_host();
        let (vetada, _, vista) = m.decision();
        let vista = vista.expect("Risk sin revision estable");
        assert_eq!((vista.revision, vista.ambito), (revision, ambito), "la revision del host etiqueta otra muestra");
        assert_eq!((vista.equity, vista.pico), (cap, pico));
        assert_eq!((cap, pico), (100.0 + upnl, 103.0), "serie");
        let freno = drawdown::freno_duro(&m.arena, pico, cap);
        assert_eq!(vetada, freno.echado(), "Risk y el host decidieron distinto con la misma revision");
    }
    // Escritor real retenido a mitad en otro hilo; un segundo escritor (observador) espera.
    let a = Arc::clone(&m.arena);
    let (dentro, seguir) = (Arc::new(std::sync::Barrier::new(2)), Arc::new(std::sync::Barrier::new(2)));
    let primero = {
        let (dentro, seguir, a) = (Arc::clone(&dentro), Arc::clone(&seguir), Arc::clone(&a));
        std::thread::spawn(move || {
            let _p = drawdown::EscrituraDelPlano::abrir(&a);
            drawdown::registrar_pico(&a.pico_capital, 120.0);
            dentro.wait();
            seguir.wait();
            a.capital_exchange.store(120.0, Ordering::Release);
        })
    };
    dentro.wait();
    let segundo_listo = Arc::new(AtomicBool::new(false));
    let segundo = {
        let (a, listo) = (Arc::clone(&a), Arc::clone(&segundo_listo));
        std::thread::spawn(move || {
            drawdown::publicar_equity_del_freno(&a, Some(87.0));
            listo.store(true, Ordering::SeqCst);
        })
    };
    for _ in 0..50 {
        std::thread::yield_now();
    }
    assert!(!segundo_listo.load(Ordering::SeqCst), "un segundo escritor abrio su revision mientras el primero estaba a mitad");
    let (vetada, _, vista) = m.decision();
    assert!(vetada && vista.is_none(), "Risk decidio con la mezcla: {vista:?}");
    seguir.wait();
    primero.join().unwrap();
    segundo.join().unwrap();
    let (_, _, vista) = m.decision();
    let vista = vista.expect("revision estable");
    assert_eq!((vista.equity, vista.pico, vista.revision % 2), (87.0, 120.0, 0), "la vista final no es la del ultimo escritor");
}

/// L-F5 (matriz :41). Cota 90 con pico 100 (evidencia incompleta), ganancia no conciliada que no
/// reabre, vuelta de la misma cuenta que conserva la perdida, foto acreditada 104 y revision real:
/// en cada fase el host y Risk comparten revision. Con un escritor retenido durante la cota, Risk
/// veta (no decide con la mezcla). Fase previa (equity 100 completa) como control verde.
#[test]
fn v4l_f5_cota_sin_ascenso_y_restauracion_productiva() {
    let _g = UNIVERSO.lock().unwrap_or_else(|p| p.into_inner());
    let mut m = plano(1);
    m.rest(100.0, None, 100);
    let (cap, pico, revision, _) = m.paso_del_host();
    let (vetada, _, vista) = m.decision();
    assert_eq!((cap, pico, vista.map(|v| v.revision)), (100.0, 100.0, Some(revision)), "control previo");
    assert!(!vetada, "control previo vetado por drawdown");
    m.puente.evidencia_perdida(m.puente.era()).unwrap();
    m.rest_incompleta(100.0, -10.0, 150);
    let (cap, pico, revision, _) = m.paso_del_host();
    let (vetada, _, vista) = m.decision();
    assert_eq!((cap, pico), (90.0, 100.0), "la cota menor no actuo");
    assert_eq!(vista.map(|v| (v.equity, v.pico, v.revision)), Some((90.0, 100.0, revision)), "Risk y el host no comparten la cota");
    assert!(vetada, "con la evidencia perdida Risk admitio");
    // Escritor retenido durante la cota.
    {
        let a = Arc::clone(&m.arena);
        let (dentro, seguir) = (Arc::new(std::sync::Barrier::new(2)), Arc::new(std::sync::Barrier::new(2)));
        let h = {
            let (dentro, seguir) = (Arc::clone(&dentro), Arc::clone(&seguir));
            std::thread::spawn(move || {
                let _p = drawdown::EscrituraDelPlano::abrir(&a);
                dentro.wait();
                seguir.wait();
            })
        };
        dentro.wait();
        let (vetada, _, vista) = m.decision();
        assert!(vetada && vista.is_none(), "Risk decidio con un escritor a mitad durante la cota");
        seguir.wait();
        h.join().unwrap();
    }
    m.rest_incompleta(100.0, 20.0, 160);
    let (cap, pico, _, _) = m.paso_del_host();
    assert_eq!((cap, pico), (90.0, 100.0), "una foto incompleta mas alta gano evidencia");
    let _ = m.relevo(1);
    let (cap, pico, _, _) = m.paso_del_host();
    assert_eq!((cap, pico), (90.0, 100.0), "la vuelta de la misma cuenta perdio la perdida conservada");
    m.rest(104.0, None, 300);
    let (cap, pico, revision, _) = m.paso_del_host();
    let (vetada, _, vista) = m.decision();
    assert_eq!((cap, pico), (104.0, 104.0));
    assert_eq!(vista.map(|v| (v.equity, v.pico, v.revision)), Some((104.0, 104.0, revision)));
    assert!(!vetada, "la foto acreditada no restauro la admision");
}

/// L-F6 (matriz :42). Arranque: el pico persistido compatible (120) se restaura ANTES de cualquier
/// equity; restaurado pero sin equity comparable la vista NO esta lista (K la daba lista con NaN y
/// el freno medía caida 0): Risk veta por drawdown y el libro no admite (cero entregas). La primera
/// equity comparable (100) deja la vista lista con el pico del ambito y la caida real. Controles:
/// pico ilegible (veta), pico de otro espacio/legado (se ignora: nace del ambito), retorno a la
/// cuenta original con su pico, y reset valido (pico ausente con marca: nace de la equity).
#[test]
fn v4l_f6_restauracion_atomica_con_equity_vigente() {
    let _g = UNIVERSO.lock().unwrap_or_else(|p| p.into_inner());
    anclas_v4l();
    let dir = std::env::temp_dir().join(format!("tg_v4l_f6_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let rutas = drawdown::RutasPico::del_espacio("v4l_cuenta1", |n| dir.join(n));
    drawdown::guardar_pico(&rutas.pico, 120.0).unwrap();
    drawdown::guardar_marca_definicion(&rutas.marca).unwrap();
    let arena = GlobalArena::build_in_own_stack(100.0);
    drawdown::preparar_ambito_del_pico(&arena);
    let mut m = plano_con_arena(1, arena);
    m.arranque(&rutas);
    let (vetada, rechazo, vista) = m.decision();
    let v = vista.expect("vista");
    assert!(!v.lista && v.pico == 120.0 && !v.equity.is_finite(), "restaurado sin equity se dio por listo: {v:?}");
    assert!(vetada, "el falso listo esquivo la guarda: rechazo {rechazo:?}");
    let capital = Some(m.arena.unified_capital.load(Ordering::Relaxed));
    assert!(!m.arena.config.entry_publication_gate.lock().unwrap().admite(&m.arena, capital), "el libro admitio sin saldo (entrega posible)");
    // Primera equity comparable del ambito.
    m.rest(100.0, None, 100);
    let (cap, pico, revision, ambito) = m.paso_del_host();
    let (vetada, _, vista) = m.decision();
    let v = vista.expect("vista lista");
    assert!(v.lista && (v.equity, v.pico, v.revision, v.ambito) == (100.0, 120.0, revision, ambito), "{v:?}");
    assert_eq!((cap, pico), (100.0, 120.0));
    assert_eq!(vetada, drawdown::freno_duro(&m.arena, 120.0, 100.0).echado(), "la caida real no gobierna");
    // Relevo a otra cuenta y vuelta: el sucesor no hereda ni la equity ni el pico; al volver, su
    // pico se restaura antes de la evidencia.
    let _ = m.relevo(2);
    let (_, _, vista) = m.decision();
    assert!(vista.is_some_and(|v| !v.lista), "el sucesor quedo listo con la evidencia de la cuenta anterior");
    let rutas2 = drawdown::RutasPico::del_espacio("v4l_cuenta2", |n| dir.join(n));
    m.arranque(&rutas2);
    m.rest(50.0, None, 10);
    let (cap, pico, _, _) = m.paso_del_host();
    assert_eq!((cap, pico), (50.0, 50.0), "reset valido: el pico ausente del espacio nuevo nace de su equity (legado ajeno ignorado)");
    let _ = m.relevo(1);
    m.arranque(&rutas);
    m.rest(110.0, None, 500);
    let (cap, pico, _, _) = m.paso_del_host();
    assert_eq!((cap, pico), (110.0, 120.0), "retorno: el pico de la cuenta original no gobierna");
    // Pico ilegible: veta.
    let rutas3 = drawdown::RutasPico::del_espacio("v4l_ilegible", |n| dir.join(n));
    std::fs::write(&rutas3.pico, "no-es-un-numero").unwrap();
    drawdown::guardar_marca_definicion(&rutas3.marca).unwrap();
    let _ = m.relevo(3);
    m.arranque(&rutas3);
    m.rest(80.0, None, 20);
    let _ = m.paso_del_host();
    let (vetada, _, vista) = m.decision();
    assert!(vetada && vista.is_some_and(|v| v.pico_ilegible), "pico ilegible sin veto: {vista:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// L-F6 (matriz :43). Autoria patrimonial: el disparo de la cuenta A compromete sus efectos bajo el
/// mutex del relevo. Relevo ANTES del compromiso: cero efectos. Relevo DESPUES: la causa de A no se
/// exporta al sucesor B (latch de arena y heredado por el ejecutor sucesor sueltos, sin consumo de
/// confirmaciones de B, reintento no vigente), mientras el aplanado de A conserva su sujeto. Mismo
/// ambito y causas globales: conservan latch y reintento (control).
#[test]
fn v4l_f6_disparo_y_reintento_no_saltan_de_ambito() {
    let _g = UNIVERSO.lock().unwrap_or_else(|p| p.into_inner());
    use execution_engine::aplanado_inmune::{comprometer_disparo, permiso_de_consumo, relevar_causa_patrimonial};
    let ejecutor = |a: &Arc<GlobalArena>| {
        let e = OrderExecutor::new(String::new(), "offline-contract".into(), false);
        e.arena.store(Some(Arc::clone(a)));
        Arc::new(e)
    };
    let confirmar_ranura = |a: &GlobalArena| {
        let p = a.coins[0].positions.get_slot(0);
        assert!(p.open_with_tau_and_fee(true, 100.0, 0.01, 10.0, 1_000, 110.0, 90.0, PositionHorizon::Continuous, 0.7, 0.8, 0.04, 60_000));
        p.exchange_confirmed.store(true, Ordering::Release);
    };
    // 1. Relevo antes del compromiso.
    {
        let m = plano(1);
        let a_exec = ejecutor(&m.arena);
        let ambito = drawdown::ambito_del_plano(&m.arena);
        let _ = m.relevo(2);
        let r = m.puente.con_equity_del_freno(|_| comprometer_disparo(&m.arena, || Arc::clone(&a_exec), Some(ambito)));
        assert!(r.is_none(), "la decision del ambito anterior comprometio efectos");
        assert!(!m.arena.kill_switch_active.load(Ordering::Acquire) && !a_exec.is_kill_switch_active(), "efectos sin compromiso");
    }
    // 2. Relevo despues del compromiso (otra cuenta).
    {
        let m = plano(1);
        let a_exec = ejecutor(&m.arena);
        let ambito = drawdown::ambito_del_plano(&m.arena);
        let (e, autoria) = m.puente.con_equity_del_freno(|_| comprometer_disparo(&m.arena, || Arc::clone(&a_exec), Some(ambito))).expect("control: mismo ambito");
        assert!(Arc::ptr_eq(&e, &a_exec) && a_exec.is_kill_switch_active() && m.arena.kill_switch_active.load(Ordering::Acquire));
        let (b_exec, _) = OrderExecutor::sucesor_de(&a_exec, String::new(), "offline-b".into(), false, Arc::clone(&m.arena));
        assert!(b_exec.is_kill_switch_active(), "fixture N58: el sucesor hereda el latch");
        let _ = m.relevo(2);
        let nuevo = drawdown::ambito_del_plano(&m.arena);
        assert!(relevar_causa_patrimonial(&m.arena, &b_exec, nuevo), "el relevo no solto la causa patrimonial anterior");
        assert!(!m.arena.kill_switch_active.load(Ordering::Acquire), "la arena del sucesor sigue vetada por la causa de A");
        assert!(!b_exec.is_kill_switch_active(), "el ejecutor sucesor sigue armado por la causa de A");
        assert!(a_exec.is_kill_switch_active(), "el sujeto A perdio su propio latch");
        assert!(!autoria.vigente(&m.arena), "el reintento de A seguiria vigente sobre B");
        confirmar_ranura(&m.arena);
        assert!(permiso_de_consumo(&m.arena).is_none(), "el aplanado de A consumiria confirmaciones de B");
        assert!(m.arena.coins[0].positions.get_slot(0).exchange_confirmed.load(Ordering::Acquire));
    }
    // 3. Control: mismo ambito (relevo de la misma cuenta) conserva latch y reintento.
    {
        let m = plano(1);
        let a_exec = ejecutor(&m.arena);
        let ambito = drawdown::ambito_del_plano(&m.arena);
        let (_, autoria) = m.puente.con_equity_del_freno(|_| comprometer_disparo(&m.arena, || Arc::clone(&a_exec), Some(ambito))).unwrap();
        let (b_exec, _) = OrderExecutor::sucesor_de(&a_exec, String::new(), "offline-b".into(), false, Arc::clone(&m.arena));
        let _ = m.relevo(1);
        assert!(!relevar_causa_patrimonial(&m.arena, &b_exec, drawdown::ambito_del_plano(&m.arena)));
        assert!(m.arena.kill_switch_active.load(Ordering::Acquire) && b_exec.is_kill_switch_active() && autoria.vigente(&m.arena));
        confirmar_ranura(&m.arena);
        assert!(permiso_de_consumo(&m.arena).is_some(), "mismo ambito: el aplanado no consumiria las confirmaciones");
    }
    // 4. Control: causa global (operador o latencia) cruza el relevo a otra cuenta.
    {
        let m = plano(1);
        let a_exec = ejecutor(&m.arena);
        let (_, autoria) = m.puente.con_equity_del_freno(|_| comprometer_disparo(&m.arena, || Arc::clone(&a_exec), None)).unwrap();
        let (b_exec, _) = OrderExecutor::sucesor_de(&a_exec, String::new(), "offline-b".into(), false, Arc::clone(&m.arena));
        let _ = m.relevo(2);
        assert!(!relevar_causa_patrimonial(&m.arena, &b_exec, drawdown::ambito_del_plano(&m.arena)), "una causa global se solto");
        assert!(m.arena.kill_switch_active.load(Ordering::Acquire) && b_exec.is_kill_switch_active() && autoria.vigente(&m.arena));
    }
    // El host: compromiso del disparo bajo el mutex, ACTIVADO tras el efecto, latch relevado antes
    // del cerrojo y del reintento, relevo que no exporta la causa (anclas).
    orden(HOST, &[
        "let mut autoria_del_latch: Option<execution_engine::aplanado_inmune::AutoriaDelDisparo> = None;",
        "if latched && autoria_del_latch.is_some_and(|a| !a.vigente(&arena_imm)) {",
        "reintento_aplanado.olvidar();",
        "latched = false;",
        "if latched {",
        "if reintento_aplanado.toca(ahora_reintento) {",
        "execution_engine::aplanado_inmune::sujeto_del_reintento(",  // V4M (F6): ancla nueva, autoria y ejecutor en un paso
        "let Some((executor, autoria)) = puente_imm.con_equity_del_freno(|_| {",
        "execution_engine::aplanado_inmune::comprometer_disparo(",
        "[SISTEMA INMUNE] ACTIVADO",
        "autoria_del_latch = Some(autoria);",
        "latched = true;",
    ]);
    orden(HOST, &["async fn aplanado_inmune(", "execution_engine::aplanado_inmune::permiso_de_consumo(arena)", "Some(_permiso) => arena.consume_exchange_confirmations(),", "executor.flatten_all_positions().await"]);
}

/// L-F8 (matriz :44). GET de income RETENIDO por el servidor local con el ejecutor real; relevo
/// mientras espera; la respuesta llega despues. Con el testigo de la consulta (era tomada antes
/// del ejecutor) no registra nada en el sucesor (tracker vacio, libro sin funding nuevo). Control:
/// sin relevo, la misma respuesta registra una vez; replay por identidad acreditada no cuenta
/// otra vez; otro activo y otra cuenta conservan su ambito; un flujo SIN identidad nativa queda
/// incierto (sin cifra ni deduplicacion inferidas por tipo y hora); el pico no cambia.
#[tokio::test]
async fn v4l_f8_income_antes_del_relevo_no_altera_sucesor() {
    // El montaje toma el candado unico de la serie (UNIVERSO).
    let o = montaje().await;
    *o.servidor.income.lock().unwrap() = r#"[{"symbol":"","incomeType":"TRANSFER","income":"-35","asset":"USDT","time":150,"tranId":9001,"tradeId":""},
        {"symbol":"","incomeType":"TRANSFER","income":"-1","asset":"BNB","time":151,"tranId":9002,"tradeId":""},
        {"symbol":"","incomeType":"TRANSFER","income":"-2","asset":"USDT","time":152,"tranId":0,"tradeId":""}]"#
        .into();
    let m = plano(1);
    m.rest(100.0, None, 100);
    let _ = m.paso_del_host();
    let pico0 = m.arena.pico_capital.load(Ordering::Acquire);
    let flujos = Mutex::new(FlujosExternos::default());
    // Consulta retenida, relevo de la misma cuenta mientras espera.
    o.servidor.retener_income.store(true, Ordering::Release);
    let era = m.puente.era();
    let consulta = o.ejecutor.fetch_income_paged(&[], 0, 1);
    let relevar = async {
        o.servidor.llego_income.notified().await;
        let _ = m.relevo(1);
        o.servidor.seguir_income.notify_one();
    };
    let (entradas, ()) = tokio::join!(consulta, relevar);
    let mut entradas = entradas.expect("fixture: la respuesta retenida llega");
    // El transporte productivo (`income_evidence`, fuera de los archivos autorizados) descarta el
    // registro sin `tranId` como invalido: llegan 2 de 3. El registro sin identidad se entrega
    // abajo al registro productivo directamente para observar su atribucion.
    assert_eq!(entradas.len(), 2, "fixture: el transporte cambio su filtro de identidad");
    let sin_id: Vec<execution_engine::order_types::IncomeEntry> = serde_json::from_str(r#"[{"symbol":"","incomeType":"TRANSFER","income":"-2","asset":"USDT","time":152,"tranId":0,"tradeId":""}]"#).unwrap();
    entradas.extend(sin_id.clone());
    let r = quantum_engine::host_oms::registrar_flujos_externos_de_era(&m.puente, era, &m.arena, &flujos, &entradas);
    assert!(r.is_none(), "la respuesta caducada registro en el sucesor: {r:?}");
    {
        let tracker = flujos.lock().unwrap();
        assert_eq!(tracker.cantidad() + tracker.sin_identidad(), 0, "tracker del sucesor modificado");
    }
    let efecto = m.puente.con_era_vigente(era, |_| ());
    assert!(efecto.is_none(), "el testigo de la consulta seguiria vigente para profiler, breaker, certificado o suspension");
    // Control con la era vigente: una vez, replay sin efecto.
    o.servidor.retener_income.store(false, Ordering::Release);
    let era = m.puente.era();
    let mut entradas = o.ejecutor.fetch_income_paged(&[], 0, 1).await.unwrap();
    entradas.extend(sin_id);
    let r = quantum_engine::host_oms::registrar_flujos_externos_de_era(&m.puente, era, &m.arena, &flujos, &entradas).expect("control");
    assert_eq!(r, vec![(9001, RegistroDeFlujo::Nuevo), (9002, RegistroDeFlujo::OtroActivo), (0, RegistroDeFlujo::SinIdentidad)]);
    let r = quantum_engine::host_oms::registrar_flujos_externos_de_era(&m.puente, era, &m.arena, &flujos, &entradas).unwrap();
    assert_eq!(r[..2], [(9001, RegistroDeFlujo::Repetido), (9002, RegistroDeFlujo::Repetido)], "replay por identidad");
    assert_eq!(r[2], (0, RegistroDeFlujo::SinIdentidad), "sin identidad: no se deduplica por inferencia");
    assert_eq!(flujos.lock().unwrap().neto_incierto(), -35.0, "el flujo sin identidad sumo una cifra inferida");
    assert_eq!(flujos.lock().unwrap().origen_de_la_caida(), OrigenDeLaCaida::FlujoIncierto);
    // Otra cuenta: la respuesta del sucesor no mezcla flujos de la cuenta 1.
    let _ = m.relevo(2);
    let r = quantum_engine::host_oms::registrar_flujos_externos_de_era(&m.puente, m.puente.era(), &m.arena, &flujos, &entradas).unwrap();
    assert!(r.iter().all(|(_, x)| *x == RegistroDeFlujo::OtroAmbito), "otra cuenta mezclo flujos: {r:?}");
    assert_eq!(m.arena.pico_capital.load(Ordering::Acquire) <= pico0, true, "un flujo ajusto el pico");
    // El host compromete profiler, breaker y suspension con el testigo de la consulta (anclas).
    orden(HOST, &[
        "let era_brk = puente_brk.era();",
        "let executor = exec_brk.load_full();",
        ".fetch_income_paged(&[], window_start, 4)",
        "let _ = puente_brk.con_era_vigente(era_brk, |_| {",
        "quantum_engine::host_oms::registrar_flujos_externos_de_era(",
        "let aplicada = puente_brk.con_era_vigente(era_brk, |_| {",
        "telemetry_server::profiler::record_exchange_income(",
        "registrar_breaker_ingreso(&mut *orch, x_closes, x_gross, x_fees)",
        "let Some(disparo_del_breaker) = aplicada else {",
        "continue;",
        "quantum_engine::host_oms::certificado_demo_de_era(",  // V4M (F8): ancla adaptada, el certificado con testigo de era
        "if puente_brk.con_era_vigente(era_brk, |_| suspend_symbol_until(&sym, until_ms)).is_none() {",
        "persist_fee_breaker(now_breaker);",
    ]);
}

/// L-F4 (matriz :40). Posicion ABIERTA reconciliada (snapshot del exchange con 0.010 y ranura
/// confirmada), una orden protectora viva con identidad y una reserva valida. Con ausencia, cero
/// ordinario y deuda: el libro rechaza reservar, la reserva previa no revalida (0 apalancamientos,
/// 0 aperturas). La cancelacion de la protectora (DELETE con ACK) y el cierre reduce-only (resultado
/// terminal correlacionado) recorren sus rutas reales; un cierre PARCIAL no se da por completo
/// (la posicion y su registro se conservan). Control sano con el mismo montaje: la entrada sale.
/// El caso 3a (cero con totalWalletBalance distinto) sigue en su regresion separada.
#[tokio::test]
async fn v4l_f4_cero_ausencia_deuda_protegen_posicion_real() {
    for variante in ["sano", "ausente", "cero", "deuda"] {
        let m = montaje().await;
        // Posicion abierta reconciliada.
        let abierta = vec![PositionRiskEntry { symbol: SIMBOLO.into(), position_side: "BOTH".into(), position_amt: 0.01, entry_price: 100.0, leverage: 1.0, ..Default::default() }];
        let mut filas = posiciones_planas();
        filas.retain(|p| p.symbol != SIMBOLO);
        filas.extend(abierta);
        // El snapshot del exchange (0.010 abierta) la ADOPTA en una ranura local: posicion
        // abierta reconciliada.
        let token = m.sesion.snapshot_token();
        let fees = FeeSchedule::new(token.generation(), filas_de_comision()).map_err(|e| format!("{e:?}"));
        let adopcion = m.sesion.apply_snapshot(&token, Ok(filas.clone()), Ok(Vec::new()), fees, &m.arena, 2_000);
        // Segunda foto del mismo exchange: la ranura adoptada ya es la local que la respalda.
        let token = m.sesion.snapshot_token();
        let fees = FeeSchedule::new(token.generation(), filas_de_comision()).map_err(|e| format!("{e:?}"));
        let filas_diag = filas.clone();
        let foto = m.sesion.apply_snapshot(&token, Ok(filas), Ok(Vec::new()), fees, &m.arena, 3_000);
        println!("{variante}: snapshot con la posicion abierta: {adopcion:?} / {foto:?}");
        let diag = execution_engine::reconciliation::reconcile_arena_checked(&filas_diag, &m.arena, 3_500);
        println!("{variante}: diagnostico de la reconciliacion: {:?}", diag.unresolved);
        assert!(adopcion.adopted == 1 && foto.unresolved == 0, "{variante}: fixture sin posicion reconciliada: {adopcion:?} / {foto:?}");
        let abierta_local = |m: &Montaje| m.arena.coins[0].positions.slots().iter().any(|p| p.is_open() && p.exchange_confirmed.load(Ordering::Acquire));
        assert!(abierta_local(&m), "{variante}: fixture sin posicion abierta");
        let token = m.sesion.snapshot_token();
        let clave = m.reservar(9);
        {
            let mut l = m.arena.config.entry_publication_gate.lock().unwrap();
            match variante {
                "ausente" => {
                    l.fijar_cuenta(&m.arena, 1);
                    l.fijar_cuenta(&m.arena, 2);
                }
                "cero" => l.saldo_aceptado(&m.arena, 0.0, Some(200)),
                "deuda" => l.saldo_aceptado(&m.arena, -5.0, Some(200)),
                _ => {}
            }
            if variante != "sano" {
                let capital = Some(m.arena.unified_capital.load(Ordering::Relaxed));
                assert!(!l.admite(&m.arena, capital), "{variante}: el libro admitio reservar");
            }
        }
        let res = m.sesion.dispatch_entry_reservada(&token, m.ejecutor.as_ref(), &pedido(EntryRoute::Market, "cL_v4l_f4"), &m.arena, Some(clave)).await;
        if variante == "sano" {
            assert!(res.is_ok() && (m.servidor.apalancamientos(), m.servidor.aperturas()) == (1, 1), "control sano: {res:?}");
        } else {
            assert!(res.is_err(), "{variante}: la reserva previa revalido");
            assert_eq!((m.servidor.apalancamientos(), m.servidor.aperturas()), (0, 0), "{variante}: salio apalancamiento o apertura");
        }
        // Protectora viva con identidad: cancelacion con ACK efectivo.
        let cancelada = m.ejecutor.cancel_order(SIMBOLO, "CONT_sl_v4l").await;
        assert!(cancelada.is_ok(), "{variante}: cancelacion sin ACK: {cancelada:?}");
        assert!(m.servidor.peticiones.lock().unwrap().iter().any(|p| p.metodo == "DELETE" && p.params.get("origClientOrderId").map(String::as_str) == Some("CONT_sl_v4l")), "{variante}: DELETE sin identidad");
        // Cierre reduce-only completo: terminal correlacionado.
        let cierre = m.ejecutor.execute_reduce_only_market_with_reconciliation(SIMBOLO, true, 0.01, 0.001).await;
        assert!(cierre.is_ok(), "{variante}: cierre reductor no terminal: {cierre:?}");
        // Cierre parcial: no se da por completo; la ranura y su registro se conservan.
        *m.servidor.ejecutada_cierre.lock().unwrap() = "0.004".into();
        let parcial = m.ejecutor.execute_reduce_only_market_with_reconciliation(SIMBOLO, true, 0.01, 0.001).await;
        assert!(parcial.is_err(), "{variante}: un cierre parcial se dio por completo");
        assert!(abierta_local(&m), "{variante}: la posicion local se perdio sin evidencia");
    }
}
