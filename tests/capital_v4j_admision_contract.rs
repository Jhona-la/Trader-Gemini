//! CAPITAL-UNICO V4J, punto 1 (A1 de `qra_v4j_criterios.md`): la aceptacion local del transporte
//! queda serializada con la revision vigente, los vetos y la sesion.
//!
//! Piezas reales: `OmsSession::dispatch_entry_reservada` (su guardia y la frontera), el
//! `OrderExecutor` de produccion con su cliente HTTP, y el libro del dimensionable bajo la puerta.
//! El exchange es un servidor local (127.0.0.1) que solo registra lo que le LLEGA: cada paquete
//! que cuenta la prueba es una entrega irrevocable local del POST real. Las barreras son
//! deterministas: el cambio se intercala dentro del mismo `poll`, despues de la revalidacion del
//! guardia y antes del cruce del ejecutor (adaptador que envuelve al ejecutor real), o en el
//! servidor mientras el ejecutor espera una respuesta (ruta maker). Sin red externa ni claves.
use arc_swap::ArcSwap;
use execution_engine::entry_dispatch::{EntryRequest, EntryRoute, EntryTransport};
use execution_engine::executor::{ExecutionProvider, OrderExecutor};
use execution_engine::reconciliation::PositionRiskEntry;
use quantum_arena::dimensionable::{cruzar_frontera_de_envio, ClaveReserva};
use quantum_arena::{FeeSchedule, GlobalArena, SymbolFees};
use quantum_engine::host_oms::{OmsSession, SessionToken};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::Notify;

static UNIVERSO: Mutex<()> = Mutex::new(());
const SIMBOLO: &str = "BTCUSDT";

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
    /// Ruta maker: el servidor retiene la respuesta de la consulta (GET) hasta `seguir_get`.
    retener_get: AtomicBool,
    llego_get: Notify,
    seguir_get: Notify,
    /// Cantidad ejecutada que devuelve la consulta del GTX.
    ejecutada_gtx: Mutex<String>,
    /// Control: el servidor recibe el POST de la orden y retiene su respuesta hasta `seguir_post`.
    retener_post: AtomicBool,
    llego_post: Notify,
    seguir_post: Notify,
}

impl Servidor {
    fn ordenes(&self) -> Vec<Peticion> {
        self.peticiones
            .lock()
            .unwrap()
            .iter()
            .filter(|p| p.metodo == "POST" && p.ruta == "/fapi/v1/order")
            .cloned()
            .collect()
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

async fn responder(mut socket: TcpStream, cuerpo: String) {
    let r = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{cuerpo}",
        cuerpo.len()
    );
    let _ = socket.write_all(r.as_bytes()).await;
    let _ = socket.shutdown().await;
}

fn orden_json(p: &Peticion, estado: &str, ejecutada: &str) -> String {
    let id = p.params.get("newClientOrderId").or(p.params.get("origClientOrderId")).cloned().unwrap_or_default();
    let tipo = p.params.get("type").cloned().unwrap_or_else(|| "LIMIT".into());
    format!(
        r#"{{"symbol":"{SIMBOLO}","clientOrderId":"{id}","orderId":4242,"side":"BUY","type":"{tipo}","status":"{estado}","origQty":"0.010","executedQty":"{ejecutada}","avgPrice":"100.0","price":"100.0","cumQuote":"1.0","updateTime":1000}}"#
    )
}

async fn servir(listener: tokio::net::TcpListener, s: Arc<Servidor>) {
    loop {
        let Ok((mut socket, _)) = listener.accept().await else { return };
        let s = Arc::clone(&s);
        tokio::spawn(async move {
            let Some(p) = leer(&mut socket).await else { return };
            s.peticiones.lock().unwrap().push(p.clone());
            let cuerpo = match (p.metodo.as_str(), p.ruta.as_str()) {
                ("POST", "/fapi/v1/leverage") => {
                    let l = p.params.get("leverage").cloned().unwrap_or_default();
                    format!(r#"{{"symbol":"{SIMBOLO}","leverage":{l},"maxNotionalValue":"1000000"}}"#)
                }
                ("POST", "/fapi/v1/order") => {
                    if s.retener_post.load(Ordering::Acquire) {
                        s.llego_post.notify_one();
                        s.seguir_post.notified().await;
                    }
                    match p.params.get("timeInForce").map(String::as_str) {
                    Some("GTX") => orden_json(&p, "NEW", "0"),
                    Some("IOC") => orden_json(&p, "FILLED", "0.010"),
                    _ => orden_json(&p, "FILLED", p.params.get("quantity").map(String::as_str).unwrap_or("0.010")),
                    }
                }
                ("DELETE", "/fapi/v1/order") => orden_json(&p, "CANCELED", "0"),
                ("GET", "/fapi/v1/order") => {
                    if s.retener_get.load(Ordering::Acquire) {
                        s.llego_get.notify_one();
                        s.seguir_get.notified().await;
                    }
                    let ejecutada = s.ejecutada_gtx.lock().unwrap().clone();
                    orden_json(&p, "CANCELED", &ejecutada)
                }
                _ => r#"{"code":-9999,"msg":"ruta inesperada del montaje"}"#.to_string(),
            };
            responder(socket, cuerpo).await;
        });
    }
}

// ------------------------------------------------------------------ montaje

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
    *servidor.ejecutada_gtx.lock().unwrap() = "0.010".into();
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
    fn reservar(&self, generation: u64) -> ClaveReserva {
        let clave = ClaveReserva { coin_id: 0, slot: 0, generation };
        let mut libro = self.arena.config.entry_publication_gate.lock().unwrap();
        let mut capital = Some(self.arena.unified_capital.load(Ordering::Relaxed));
        if !libro.admite(&self.arena, capital) {
            // V4I (INV-F): un importe ya publicado en una revision anterior se recalcula una vez.
            capital = Some(self.arena.unified_capital.load(Ordering::Relaxed));
            assert!(libro.admite(&self.arena, capital), "fixture: la decision recalculada se reserva");
        }
        libro.reservar_con_base(&self.arena, clave, 0.04, capital);
        clave
    }
    fn reservado(&self) -> f64 {
        self.arena.config.entry_publication_gate.lock().unwrap().reservado()
    }
}

fn pedido(ruta: EntryRoute, id: &'static str) -> EntryRequest<'static> {
    EntryRequest {
        symbol: SIMBOLO,
        is_long: true,
        quantity: 0.01,
        step_size: 0.001,
        tick_size: 0.1,
        leverage: 1,
        client_order_id: id,
        route: ruta,
    }
}

/// Adaptador sobre el ejecutor REAL: corre `antes` dentro del mismo `poll`, despues de la
/// revalidacion del guardia y antes de que el ejecutor llegue a su cruce (apalancamiento o POST).
struct Intercalado<'a> {
    real: &'a OrderExecutor,
    antes_de_apalancar: Option<Box<dyn Fn() + 'a>>,
    antes_de_enviar: Option<Box<dyn Fn() + 'a>>,
}

impl EntryTransport for Intercalado<'_> {
    fn supports_native_iceberg(&self) -> bool {
        self.real.supports_native_iceberg()
    }
    async fn configure_leverage(&self, symbol: &str, leverage: u32) -> Result<(), String> {
        if let Some(f) = &self.antes_de_apalancar {
            f();
        }
        self.real.configure_leverage(symbol, leverage).await
    }
    async fn submit_entry(&self, request: &EntryRequest<'_>) -> Result<(), String> {
        if let Some(f) = &self.antes_de_enviar {
            f();
        }
        self.real.submit_entry(request).await
    }
}

/// Los cortes del criterio A1, cada uno por separado.
const CORTES: [&str; 6] = ["revision", "importe ABA", "veto del libro", "veto del puente", "caida y reapertura", "sesion"];

fn aplicar_corte(m: &Montaje, corte: &str) {
    aplicar_corte_en(&m.arena, &m.sesion, &m.token, corte)
}

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

// ------------------------------------------------------------------ pruebas

/// A1: por cada ruta alcanzable (MARKET, IOC y GTX por REST) y cada corte (revision, importe ABA,
/// veto del libro, veto del puente, caida seguida de reapertura y sesion), intercalado DENTRO del
/// mismo `poll` despues de la revalidacion del guardia: 0 paquetes de entrada y 0 apalancamientos
/// cuando el corte precede al apalancamiento. Control por ruta sin corte: un solo paquete.
/// Ruta maker a MARKET: el GTX ya entregado pertenece a su revision; el remanente con la revision
/// vencida (corte mientras el ejecutor espera la consulta) no sale. Iceberg y WS: el enrutador y
/// el stub las excluyen antes de cualquier efecto.
#[tokio::test]
async fn veto_y_revision_en_la_entrega_local_por_ruta() {
    let rutas = [
        ("MARKET", EntryRoute::Market, "MARKET"),
        ("IOC", EntryRoute::Ioc { price: 101.0 }, "IOC"),
        ("GTX", EntryRoute::Maker { price: 99.0 }, "GTX"),
    ];
    for (nombre, ruta, tif) in rutas {
        // Control: sin corte, una sola entrega.
        {
            let m = montaje().await;
            let clave = m.reservar(1);
            let r = pedido(ruta, "cL_v4j_control");
            let t = Intercalado { real: &m.ejecutor, antes_de_apalancar: None, antes_de_enviar: None };
            let _ = m.sesion.dispatch_entry_reservada(&m.token, &t, &r, &m.arena, Some(clave)).await;
            let ordenes = m.servidor.ordenes();
            assert_eq!((m.servidor.apalancamientos(), ordenes.len()), (1, 1), "{nombre}: control sin corte");
            let tif_visto = ordenes[0].params.get("timeInForce").map(String::as_str).unwrap_or("MARKET");
            assert_eq!(tif_visto, tif, "{nombre}: la ruta del paquete");
        }
        for corte in CORTES {
            for en_apalancamiento in [false, true] {
                let m = montaje().await;
                let clave = m.reservar(1);
                let r = pedido(ruta, "cL_v4j_corte");
                let cambio: Box<dyn Fn() + '_> = Box::new(|| aplicar_corte(&m, corte));
                let t = if en_apalancamiento {
                    Intercalado { real: &m.ejecutor, antes_de_apalancar: Some(cambio), antes_de_enviar: None }
                } else {
                    Intercalado { real: &m.ejecutor, antes_de_apalancar: None, antes_de_enviar: Some(cambio) }
                };
                let res = m.sesion.dispatch_entry_reservada(&m.token, &t, &r, &m.arena, Some(clave)).await;
                assert!(res.is_err(), "{nombre} / {corte} / apalancamiento={en_apalancamiento}: entrega aceptada");
                assert_eq!(m.servidor.ordenes().len(), 0, "{nombre} / {corte} / apalancamiento={en_apalancamiento}: paquete de entrada con testigo vencido");
                if en_apalancamiento {
                    assert_eq!(m.servidor.apalancamientos(), 0, "{nombre} / {corte}: apalancamiento con testigo vencido");
                } else {
                    assert_eq!(m.servidor.apalancamientos(), 1, "{nombre} / {corte}: fixture");
                }
            }
        }
    }

    // Barrera entre el registro de la intencion y la entrega local (MARKET e IOC): el corte lo
    // publica OTRO hilo, dentro del registro. 0 paquetes y la intencion cerrada (nunca salio).
    for (nombre, ruta) in [("MARKET", EntryRoute::Market), ("IOC", EntryRoute::Ioc { price: 101.0 })] {
        for corte in CORTES {
            let m = montaje().await;
            let clave = m.reservar(1);
            let (a, ses, tok) = (Arc::clone(&m.arena), Arc::clone(&m.sesion), m.token.clone());
            m.ejecutor.registry().observar_intenciones(move |_| {
                let (a, ses, tok) = (Arc::clone(&a), Arc::clone(&ses), tok.clone());
                std::thread::spawn(move || aplicar_corte_en(&a, &ses, &tok, corte)).join().unwrap();
            });
            let r = pedido(ruta, "cL_v4j_registro");
            let t = Intercalado { real: &m.ejecutor, antes_de_apalancar: None, antes_de_enviar: None };
            let res = m.sesion.dispatch_entry_reservada(&m.token, &t, &r, &m.arena, Some(clave)).await;
            assert!(res.is_err(), "{nombre} / {corte}: entrega aceptada tras el corte en el registro");
            assert_eq!(m.servidor.ordenes().len(), 0, "{nombre} / {corte}: paquete con el testigo vencido entre registro y entrega");
            let estado = m.ejecutor.registry().get("cL_v4j_registro").map(|o| o.status);
            assert_eq!(estado, Some(execution_engine::OrderStatus::Rejected), "{nombre} / {corte}: la intencion retenida quedo abierta");
        }
    }

    // Control con el ACK retenido: un envio valido espera la red y, mientras tanto, los
    // escritores (saldo, veto del puente y del libro) publican sin esperar al envio.
    {
        let m = montaje().await;
        m.servidor.retener_post.store(true, Ordering::Release);
        let clave = m.reservar(1);
        let r = pedido(EntryRoute::Market, "cL_v4j_ack");
        let t = Intercalado { real: &m.ejecutor, antes_de_apalancar: None, antes_de_enviar: None };
        let envio = m.sesion.dispatch_entry_reservada(&m.token, &t, &r, &m.arena, Some(clave));
        let escritores = async {
            m.servidor.llego_post.notified().await;
            // Los candados del cruce ya se soltaron: estos escritores no esperan la red.
            for corte in ["revision", "veto del libro", "veto del puente"] {
                aplicar_corte(&m, corte);
            }
            assert!(m.arena.config.entry_publication_gate.try_lock().is_ok(), "la puerta quedo tomada durante la red");
            m.servidor.seguir_post.notify_one();
        };
        let (_res, ()) = tokio::join!(envio, escritores);
        assert_eq!(m.servidor.ordenes().len(), 1, "control: el envio valido no salio una vez");
    }

    // Ruta maker a MARKET: GTX entregado, corte mientras el ejecutor espera la consulta, remanente.
    for corte in CORTES {
        let m = montaje().await;
        *m.servidor.ejecutada_gtx.lock().unwrap() = "0.005".into();
        m.servidor.retener_get.store(true, Ordering::Release);
        let clave = m.reservar(1);
        let r = pedido(EntryRoute::Maker { price: 99.0 }, "mc_v4j_remanente");
        let t = Intercalado { real: &m.ejecutor, antes_de_apalancar: None, antes_de_enviar: None };
        let envio = m.sesion.dispatch_entry_reservada(&m.token, &t, &r, &m.arena, Some(clave));
        let barrera = async {
            m.servidor.llego_get.notified().await;
            aplicar_corte(&m, corte);
            m.servidor.seguir_get.notify_one();
        };
        let (res, ()) = tokio::join!(envio, barrera);
        let ordenes = m.servidor.ordenes();
        let gtx = ordenes.iter().filter(|p| p.params.get("timeInForce").map(String::as_str) == Some("GTX")).count();
        let market = ordenes.iter().filter(|p| p.params.get("type").map(String::as_str) == Some("MARKET")).count();
        assert_eq!(gtx, 1, "{corte}: el GTX valido ya entregado pertenece a su revision");
        assert_eq!(market, 0, "{corte}: el remanente salio con la revision vencida ({res:?})");
        assert_eq!(m.servidor.contar("DELETE", "/fapi/v1/order"), 1, "{corte}: la cancelacion del GTX se conserva");
        // El GTX ejecuto 0,005: el error retenido no lleva al rollback ordinario; la reserva se
        // conserva para la reconciliacion (destino del host).
        let e = res.expect_err("fixture: el remanente retenido");
        assert!(e.starts_with("MAKER_CHASE_UNVERIFIED"), "{corte}: {e}");
        assert_eq!(
            execution_engine::ioc_evidence::destino_tras_consulta(&e, execution_engine::OrderResolution::Timeout, None),
            execution_engine::ioc_evidence::DestinoReserva::Conservar,
            "{corte}: el host revertiria una reserva con ejecucion"
        );
        assert!(m.reservado() > 0.0, "{corte}: la reserva con ejecucion se perdio");
    }
    // Control maker a MARKET sin corte: el remanente sale una vez.
    {
        let m = montaje().await;
        *m.servidor.ejecutada_gtx.lock().unwrap() = "0.005".into();
        let clave = m.reservar(1);
        let r = pedido(EntryRoute::Maker { price: 99.0 }, "mc_v4j_control");
        let t = Intercalado { real: &m.ejecutor, antes_de_apalancar: None, antes_de_enviar: None };
        let _ = m.sesion.dispatch_entry_reservada(&m.token, &t, &r, &m.arena, Some(clave)).await;
        let market = m.servidor.ordenes().iter().filter(|p| p.params.get("type").map(String::as_str) == Some("MARKET")).count();
        assert_eq!(market, 1, "control: el remanente vigente sale una vez");
    }

    // Iceberg: el enrutador del host no la elige sin capacidad nativa y el despacho la rechaza
    // antes del apalancamiento (0 paquetes de cualquier tipo).
    assert!(!execution_engine::entry_dispatch::admite_iceberg(execution_engine::binance_api::SUPPORTS_NATIVE_ICEBERG, 1e9, 1.0));
    {
        let m = montaje().await;
        let clave = m.reservar(1);
        let r = pedido(EntryRoute::Iceberg { price: 99.0, visible_quantity: 0.005 }, "cL_v4j_iceberg");
        let t = Intercalado { real: &m.ejecutor, antes_de_apalancar: None, antes_de_enviar: None };
        assert!(m.sesion.dispatch_entry_reservada(&m.token, &t, &r, &m.arena, Some(clave)).await.is_err());
        assert_eq!(m.servidor.peticiones.lock().unwrap().len(), 0, "iceberg: ningun efecto");
    }
    // WS: la via es un stub sin transporte (siempre desconectada); el unico intento alcanzable de
    // cada entrada es el REST de arriba.
    let ws = execution_engine::ws_executor::WsExecutor::new(String::new(), String::new(), false);
    assert!(!ws.is_connected());
    assert!(ws.send_order_payload("", "", SIMBOLO, "BUY", None, "MARKET", 0.01, None, None, false, "x", 1).is_err());
    let host = include_str!("../src/bin/god_engine.rs").replace("\r\n", "\n");
    assert!(host.contains("let force_maker = false;"), "el host habilito la ruta maker");
}

/// A1: riesgo obsoleto en el reintento y en el apalancamiento. Un corte antes del apalancamiento
/// no cambia el apalancamiento ni envia; reintentar la MISMA reserva despues del rechazo no sale
/// (revision vencida); la decision vieja no se vuelve a reservar; la recalculada sale una vez.
#[tokio::test]
async fn riesgo_obsoleto_en_reintento_y_leverage() {
    let m = montaje().await;
    let capital_viejo = Some(m.arena.unified_capital.load(Ordering::Relaxed));
    let clave = m.reservar(1);
    let r = pedido(EntryRoute::Ioc { price: 101.0 }, "cL_v4j_reintento");
    let cambio: Box<dyn Fn() + '_> = Box::new(|| aplicar_corte(&m, "revision"));
    let t = Intercalado { real: &m.ejecutor, antes_de_apalancar: Some(cambio), antes_de_enviar: None };
    assert!(m.sesion.dispatch_entry_reservada(&m.token, &t, &r, &m.arena, Some(clave)).await.is_err());
    assert_eq!((m.servidor.apalancamientos(), m.servidor.ordenes().len()), (0, 0), "apalancamiento o entrega con el riesgo obsoleto");
    // Reintento de la misma reserva: sigue vencida.
    let t = Intercalado { real: &m.ejecutor, antes_de_apalancar: None, antes_de_enviar: None };
    assert!(m.sesion.dispatch_entry_reservada(&m.token, &t, &r, &m.arena, Some(clave)).await.is_err());
    assert_eq!((m.servidor.apalancamientos(), m.servidor.ordenes().len()), (0, 0), "el reintento salio con la revision vieja");
    // La decision vieja no se reserva otra vez.
    {
        let mut libro = m.arena.config.entry_publication_gate.lock().unwrap();
        assert!(!libro.admite(&m.arena, capital_viejo), "Risk reservo con el calculo de la revision anterior");
        libro.cancelar(&m.arena, clave, 0.04);
    }
    // La recalculada (revision nueva) sale una vez.
    let nueva = m.reservar(2);
    assert!(m.sesion.dispatch_entry_reservada(&m.token, &t, &r, &m.arena, Some(nueva)).await.is_ok() || m.servidor.ordenes().len() == 1);
    assert_eq!((m.servidor.apalancamientos(), m.servidor.ordenes().len()), (1, 1), "la recalculada no salio una vez");
}

/// A1: el relevo invalida el testigo sin reusar el importe. Un refresco de la sesion intercalado
/// antes del cruce retiene la entrega; reconciliada otra vez la sesion y con el saldo de vuelta al
/// MISMO importe (100), la reserva vieja sigue vencida; tras sustituir el ejecutor, el testigo
/// viejo no entrega nada; la reserva nueva con el testigo nuevo, una vez.
#[tokio::test]
async fn relevo_invalida_token_sin_reusar_importe() {
    let m = montaje().await;
    let clave = m.reservar(1);
    let r = pedido(EntryRoute::Market, "cL_v4j_relevo");
    let cambio: Box<dyn Fn() + '_> = Box::new(|| aplicar_corte(&m, "sesion"));
    let t = Intercalado { real: &m.ejecutor, antes_de_apalancar: None, antes_de_enviar: Some(cambio) };
    assert!(m.sesion.dispatch_entry_reservada(&m.token, &t, &r, &m.arena, Some(clave)).await.is_err());
    assert_eq!(m.servidor.ordenes().len(), 0, "entrega con la sesion en refresco");
    // La sesion se reconcilia y el saldo vuelve al mismo importe con otra revision.
    let fees = FeeSchedule::new(m.token.generation(), filas_de_comision()).map_err(|e| format!("{e:?}"));
    m.sesion.apply_snapshot(&m.token, Ok(posiciones_planas()), Ok(Vec::new()), fees, &m.arena, 2_000);
    {
        let mut l = m.arena.config.entry_publication_gate.lock().unwrap();
        l.saldo_aceptado(&m.arena, 90.0, Some(150));
        l.saldo_aceptado(&m.arena, 100.0, Some(160));
    }
    assert_eq!(m.arena.unified_capital.load(Ordering::Relaxed), 100.0 - 0.04, "fixture: mismo importe");
    let t = Intercalado { real: &m.ejecutor, antes_de_apalancar: None, antes_de_enviar: None };
    assert!(m.sesion.dispatch_entry_reservada(&m.token, &t, &r, &m.arena, Some(clave)).await.is_err());
    assert_eq!(m.servidor.ordenes().len(), 0, "el importe repetido revalido la reserva vieja");
    m.arena.config.entry_publication_gate.lock().unwrap().cancelar(&m.arena, clave, 0.04);
    // Ejecutor sustituido: el testigo viejo no entrega.
    let mut sucesor = OrderExecutor::new(String::new(), "offline-contract".into(), false);
    sucesor.set_paper_trading(false);
    m.sesion.replace_executor(Arc::new(sucesor), &m.arena).unwrap();
    let nueva = m.reservar(2);
    assert!(m.sesion.dispatch_entry_reservada(&m.token, &t, &r, &m.arena, Some(nueva)).await.is_err());
    assert_eq!(m.servidor.ordenes().len(), 0, "el testigo viejo entrego tras el relevo");
    // Testigo nuevo, sesion reconciliada: la reserva vigente sale una vez (por el ejecutor real
    // del montaje, que es el transporte del adaptador).
    let token = m.sesion.snapshot_token();
    let fees = FeeSchedule::new(token.generation(), filas_de_comision()).map_err(|e| format!("{e:?}"));
    m.sesion.apply_snapshot(&token, Ok(posiciones_planas()), Ok(Vec::new()), fees, &m.arena, 3_000);
    let _ = m.sesion.dispatch_entry_reservada(&token, &t, &r, &m.arena, Some(nueva)).await;
    assert_eq!(m.servidor.ordenes().len(), 1, "la reserva vigente con el testigo nuevo no salio una vez");
}

/// A1: con cualquier veto, el cierre defensivo (reduce-only) y la cancelacion conservan su
/// salida (llegan al servidor), la entrada se retiene, y la reserva rechazada se libera una sola
/// vez (la cancelacion repetida no suma).
#[tokio::test]
async fn veto_conserva_cierre_cancelacion_y_libera_rechazo() {
    for corte in ["veto del libro", "veto del puente", "revision", "sesion"] {
        let m = montaje().await;
        let clave = m.reservar(1);
        let saldo = 100.0;
        aplicar_corte(&m, corte);
        let r = pedido(EntryRoute::Market, "cL_v4j_vetada");
        let t = Intercalado { real: &m.ejecutor, antes_de_apalancar: None, antes_de_enviar: None };
        assert!(m.sesion.dispatch_entry_reservada(&m.token, &t, &r, &m.arena, Some(clave)).await.is_err(), "{corte}");
        assert_eq!(m.servidor.ordenes().len(), 0, "{corte}: entrada con veto");
        // Salidas defensivas: fuera de cualquier entrada guardada la frontera cruza siempre.
        assert!(cruzar_frontera_de_envio().is_ok(), "{corte}: la frontera bloquea fuera de una entrada");
        let _ = m.ejecutor.execute_reduce_only_market_with_reconciliation(SIMBOLO, true, 0.01, 0.001).await;
        let cierres = m.servidor.ordenes().iter().filter(|p| p.params.get("reduceOnly").map(String::as_str) == Some("true")).count();
        assert!(cierres >= 1, "{corte}: el cierre defensivo no salio");
        let _ = m.ejecutor.cancel_order(SIMBOLO, "cL_v4j_vetada").await;
        assert_eq!(m.servidor.contar("DELETE", "/fapi/v1/order"), 1, "{corte}: la cancelacion no salio");
        // La reserva rechazada se libera una vez.
        assert!(m.reservado() > 0.0, "{corte}: fixture");
        let mut libro = m.arena.config.entry_publication_gate.lock().unwrap();
        libro.cancelar(&m.arena, clave, 0.04);
        let tras = libro.reservado();
        libro.cancelar(&m.arena, clave, 0.04);
        assert_eq!((tras, libro.reservado()), (0.0, 0.0), "{corte}: la reserva no se libero una sola vez");
        if let Some((s, _)) = libro.saldo() {
            assert!(libro.dimensionable().is_some_and(|d| d <= s && s <= saldo), "{corte}: la cancelacion sumo dinero");
        }
    }
}

// ------------------------------------------------------------------ V4K (A1)

/// V4K (A1, dictamen V4J :32 y :52): el intervalo entre el cruce de la frontera y el COMPROMISO
/// LOCAL de la entrega (cabeceras cargadas y solicitud construida en el cliente HTTP). Una barrera
/// determinista en ese intervalo (mismo hilo, mismo `poll`) aplica cada corte por separado: con el
/// corte antes del compromiso del apalancamiento, 0 apalancamientos y 0 ordenes; antes del
/// compromiso del POST de la orden, el apalancamiento ya comprometido sale una vez y la orden no.
/// Control por ruta sin corte: 1 y 1 (la barrera corre y no cambia nada). Rutas: MARKET, IOC, GTX
/// y el remanente MARKET de la ruta maker.
#[tokio::test]
async fn v4k_corte_entre_cruce_y_compromiso_local_por_ruta() {
    let rutas = [
        ("MARKET", EntryRoute::Market),
        ("IOC", EntryRoute::Ioc { price: 101.0 }),
        ("GTX", EntryRoute::Maker { price: 99.0 }),
    ];
    for (nombre, ruta) in rutas {
        for corte in std::iter::once(None).chain(CORTES.iter().copied().map(Some)) {
            for en_compromiso in [1usize, 2] {
                let m = montaje().await;
                let clave = m.reservar(1);
                let r = pedido(ruta, "cL_v4k_compromiso");
                let llamadas = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                {
                    let (a, ses, tok, n) = (Arc::clone(&m.arena), Arc::clone(&m.sesion), m.token.clone(), Arc::clone(&llamadas));
                    quantum_arena::dimensionable::fijar_barrera_de_compromiso(Some(Box::new(move || {
                        let k = n.fetch_add(1, Ordering::SeqCst) + 1;
                        if let (Some(c), true) = (corte, k == en_compromiso) {
                            aplicar_corte_en(&a, &ses, &tok, c);
                        }
                    })));
                }
                let t = Intercalado { real: &m.ejecutor, antes_de_apalancar: None, antes_de_enviar: None };
                let res = m.sesion.dispatch_entry_reservada(&m.token, &t, &r, &m.arena, Some(clave)).await;
                quantum_arena::dimensionable::fijar_barrera_de_compromiso(None);
                let (apal, ordenes) = (m.servidor.apalancamientos(), m.servidor.ordenes().len());
                assert!(llamadas.load(Ordering::SeqCst) >= 1, "{nombre}: la barrera del compromiso no corrio");
                match corte {
                    None => assert_eq!((apal, ordenes), (1, 1), "{nombre}: control sin corte ({res:?})"),
                    Some(c) => {
                        assert!(res.is_err(), "{nombre} / {c} / compromiso {en_compromiso}: entrega aceptada");
                        let esperado = if en_compromiso == 1 { (0, 0) } else { (1, 0) };
                        assert_eq!((apal, ordenes), esperado, "{nombre} / {c} / compromiso {en_compromiso}: paquete con el testigo vencido tras el cruce");
                    }
                }
            }
        }
    }
    // Ruta maker a MARKET: el GTX valido sale y se cancela; un corte en el compromiso del
    // remanente MARKET (tercer compromiso: apalancamiento, GTX, remanente) lo retiene.
    for corte in CORTES {
        let m = montaje().await;
        *m.servidor.ejecutada_gtx.lock().unwrap() = "0.005".into();
        let clave = m.reservar(1);
        let r = pedido(EntryRoute::Maker { price: 99.0 }, "mc_v4k_remanente");
        let n = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        {
            let (a, ses, tok, n) = (Arc::clone(&m.arena), Arc::clone(&m.sesion), m.token.clone(), Arc::clone(&n));
            quantum_arena::dimensionable::fijar_barrera_de_compromiso(Some(Box::new(move || {
                if n.fetch_add(1, Ordering::SeqCst) + 1 == 3 {
                    aplicar_corte_en(&a, &ses, &tok, corte);
                }
            })));
        }
        let t = Intercalado { real: &m.ejecutor, antes_de_apalancar: None, antes_de_enviar: None };
        let res = m.sesion.dispatch_entry_reservada(&m.token, &t, &r, &m.arena, Some(clave)).await;
        quantum_arena::dimensionable::fijar_barrera_de_compromiso(None);
        let ordenes = m.servidor.ordenes();
        let gtx = ordenes.iter().filter(|p| p.params.get("timeInForce").map(String::as_str) == Some("GTX")).count();
        let market = ordenes.iter().filter(|p| p.params.get("type").map(String::as_str) == Some("MARKET")).count();
        assert_eq!(gtx, 1, "{corte}: el GTX valido ya comprometido pertenece a su revision");
        assert_eq!(market, 0, "{corte}: el remanente salio tras un corte en su compromiso ({res:?})");
        assert_eq!(m.servidor.contar("DELETE", "/fapi/v1/order"), 1, "{corte}: la cancelacion del GTX se conserva");
        assert!(m.reservado() > 0.0, "{corte}: la reserva con ejecucion se perdio");
    }
    // Salidas sin cruce: el cierre defensivo y la cancelacion no pasan por el compromiso (la
    // barrera no corre) y salen aunque haya veto.
    {
        let m = montaje().await;
        aplicar_corte(&m, "veto del puente");
        let n = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        {
            let n = Arc::clone(&n);
            quantum_arena::dimensionable::fijar_barrera_de_compromiso(Some(Box::new(move || {
                n.fetch_add(1, Ordering::SeqCst);
            })));
        }
        let _ = m.ejecutor.execute_reduce_only_market_with_reconciliation(SIMBOLO, true, 0.01, 0.001).await;
        let _ = m.ejecutor.cancel_order(SIMBOLO, "cL_v4k_cualquiera").await;
        quantum_arena::dimensionable::fijar_barrera_de_compromiso(None);
        assert_eq!(n.load(Ordering::SeqCst), 0, "una salida defensiva paso por el compromiso de entrada");
        let cierres = m.servidor.ordenes().iter().filter(|p| p.params.get("reduceOnly").map(String::as_str) == Some("true")).count();
        assert!(cierres >= 1, "el cierre defensivo no salio");
        assert_eq!(m.servidor.contar("DELETE", "/fapi/v1/order"), 1, "la cancelacion no salio");
    }
}

/// V4K F4 (dictamen V4J :42, criterios :26): con una reserva valida ya hecha, cada variante
/// (ausencia: cuenta nueva sin saldo; cero; deuda) veta EFECTIVAMENTE la entrada, sin disyuncion:
/// el libro no admite ninguna entrada nueva, la reserva previa no entrega (0 paquetes por el
/// transporte local) y el cierre defensivo y la cancelacion salen.
#[tokio::test]
async fn v4k_f4_ausente_cero_y_deuda_vetan_y_conservan_salidas() {
    for variante in ["ausente", "cero", "deuda"] {
        let m = montaje().await;
        let clave = m.reservar(1);
        {
            let mut l = m.arena.config.entry_publication_gate.lock().unwrap();
            match variante {
                "ausente" => {
                    l.fijar_cuenta(&m.arena, 1);
                    l.fijar_cuenta(&m.arena, 2);
                }
                "cero" => l.saldo_aceptado(&m.arena, 0.0, Some(200)),
                _ => l.saldo_aceptado(&m.arena, -5.0, Some(200)),
            }
            let capital = Some(m.arena.unified_capital.load(Ordering::Relaxed));
            assert!(!l.admite(&m.arena, capital), "{variante}: el libro admitio una entrada nueva");
            assert!(!l.admite(&m.arena, capital), "{variante}: admitida al recalcular");
        }
        let r = pedido(EntryRoute::Market, "cL_v4k_f4");
        let t = Intercalado { real: &m.ejecutor, antes_de_apalancar: None, antes_de_enviar: None };
        let res = m.sesion.dispatch_entry_reservada(&m.token, &t, &r, &m.arena, Some(clave)).await;
        assert!(res.is_err(), "{variante}: la reserva previa entrego");
        assert_eq!(m.servidor.ordenes().len(), 0, "{variante}: paquete de entrada");
        let _ = m.ejecutor.execute_reduce_only_market_with_reconciliation(SIMBOLO, true, 0.01, 0.001).await;
        let cierres = m.servidor.ordenes().iter().filter(|p| p.params.get("reduceOnly").map(String::as_str) == Some("true")).count();
        assert!(cierres >= 1, "{variante}: el cierre defensivo no salio");
        let _ = m.ejecutor.cancel_order(SIMBOLO, "cL_v4k_f4").await;
        assert_eq!(m.servidor.contar("DELETE", "/fapi/v1/order"), 1, "{variante}: la cancelacion no salio");
    }
}
