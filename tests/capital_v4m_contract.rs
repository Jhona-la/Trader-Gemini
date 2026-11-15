//! CAPITAL-UNICO V4M (dictamen NO FAVORABLE de CODEX-CLI sobre L, `codex_cli_dictamen_v4l.md`):
//! cada garantia atraviesa la ruta real (transporte local por 127.0.0.1 con el `OrderExecutor` de
//! produccion, ejecutor del sucesor capturado por su propio servidor, certificado escrito en disco,
//! resolucion productiva del resultado de entrada, ventana de income del transporte).
//!
//! Lo que el host hace EN LINEA en `god_engine.rs` y cambia entre pines vive en
//! `capital_v4m/adaptador.rs` (el de este commit es el de M). La MISMA prueba corre sobre L con el
//! adaptador de L (copia de sus lineas, instrumentacion sin rastrear): debe fallar en L y pasar en M.
use arc_swap::ArcSwap;
use execution_engine::entry_dispatch::{EntryRequest, EntryRoute};
use execution_engine::executor::OrderExecutor;
use execution_engine::puente_cuenta::PuenteCuenta;
use execution_engine::reconciliation::PositionRiskEntry;
use god_engine_core::entry_reservation::EntryReservation;
use quantum_arena::position::PositionHorizon;
use quantum_arena::{FeeSchedule, GlobalArena, SymbolFees};
use quantum_engine::host_oms::{DesenlaceDeEntrada, EntrySettlement, OmsSession};
use risk_engine::drawdown::{self, FlujosExternos, OrigenDeLaCaida};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

#[path = "capital_v4m/adaptador.rs"]
mod adaptador;
use adaptador::*;

static SERIE: Mutex<()> = Mutex::new(());
const SIMBOLO: &str = "BTCUSDT";
const PLAZO: std::time::Duration = std::time::Duration::from_secs(2);

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
    /// Consulta (GET) del padre y de la hija `mcT_`: (estado, ejecutada).
    padre: Mutex<(String, String)>,
    hija: Mutex<(String, String)>,
    /// El POST de la hija `mcT_` llega y la conexion se cierra sin respuesta (ACK perdido).
    perder_ack_hija: AtomicBool,
    income: Mutex<String>,
}

impl Servidor {
    fn total(&self) -> usize {
        self.peticiones.lock().unwrap().len()
    }
    fn cierres(&self) -> usize {
        self.peticiones.lock().unwrap().iter().filter(|p| p.params.get("reduceOnly").map(String::as_str) == Some("true")).count()
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
    let r = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{cuerpo}", cuerpo.len());
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
            let id = p.params.get("newClientOrderId").or(p.params.get("origClientOrderId")).cloned().unwrap_or_default();
            let cuerpo = match (p.metodo.as_str(), p.ruta.as_str()) {
                ("POST", "/fapi/v1/leverage") => {
                    let l = p.params.get("leverage").cloned().unwrap_or_default();
                    format!(r#"{{"symbol":"{SIMBOLO}","leverage":{l},"maxNotionalValue":"1000000"}}"#)
                }
                ("POST", "/fapi/v1/order") => {
                    if id.starts_with("mcT_") && s.perder_ack_hija.load(Ordering::Acquire) {
                        drop(socket);
                        return;
                    }
                    match p.params.get("timeInForce").map(String::as_str) {
                        Some("GTX") => orden_json(&p, "NEW", "0"),
                        _ => orden_json(&p, "FILLED", p.params.get("quantity").map(String::as_str).unwrap_or("0.010")),
                    }
                }
                ("DELETE", "/fapi/v1/order") => orden_json(&p, "CANCELED", "0"),
                ("GET", "/fapi/v1/order") => {
                    if id.starts_with("mcT_") {
                        // La hija se pidio por el remanente (0.005): su consulta responde con su cantidad.
                        let (estado, ej) = s.hija.lock().unwrap().clone();
                        let mut q = p.clone();
                        q.params.insert("quantity".into(), "0.005".into());
                        orden_json(&q, &estado, &ej)
                    } else {
                        let (estado, ej) = s.padre.lock().unwrap().clone();
                        orden_json(&p, &estado, &ej)
                    }
                }
                ("GET", "/fapi/v2/positionRisk") => format!(
                    r#"[{{"symbol":"{SIMBOLO}","positionAmt":"0.010","entryPrice":"100.0","unRealizedProfit":"0.0","positionSide":"BOTH","leverage":"1","marginType":"cross","updateTime":1000}}]"#
                ),
                ("GET", "/fapi/v1/income") => s.income.lock().unwrap().clone(),
                _ => r#"{"code":-9999,"msg":"ruta inesperada del montaje"}"#.to_string(),
            };
            responder(socket, cuerpo).await;
        });
    }
}

async fn servidor() -> (Arc<Servidor>, String) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let s = Arc::new(Servidor::default());
    *s.padre.lock().unwrap() = ("CANCELED".into(), "0.005".into());
    *s.hija.lock().unwrap() = ("FILLED".into(), "0.005".into());
    *s.income.lock().unwrap() = "[]".into();
    tokio::spawn(servir(listener, Arc::clone(&s)));
    (s, base)
}

fn ejecutor(base: &str, arena: Option<&Arc<GlobalArena>>) -> Arc<OrderExecutor> {
    let mut e = OrderExecutor::new(String::new(), "offline-contract".into(), false).con_base_de_prueba(base);
    e.set_paper_trading(false);
    e.is_hedge_mode.store(false, Ordering::Relaxed);
    if let Some(a) = arena {
        e.arena.store(Some(Arc::clone(a)));
    }
    Arc::new(e)
}

fn puente(a: &Arc<GlobalArena>) -> (Arc<PuenteCuenta>, Arc<AtomicU64>) {
    let oms = Arc::new(AtomicU64::new(1));
    let p = Arc::new(PuenteCuenta::con_sesion(1));
    {
        let oms = Arc::clone(&oms);
        p.vincular_testigo_de_sesion(move || oms.load(Ordering::SeqCst));
    }
    {
        let mut l = a.config.entry_publication_gate.lock().unwrap();
        l.activar(a);
        l.fijar_cuenta(a, 1);
    }
    p.anunciar_cuenta(1);
    p.abrir_socket(p.era()).unwrap();
    (p, oms)
}

/// El relevo del host a otra cuenta (dentro de `relevar_sesion`): el ejecutor de B pasa a ser el
/// vigente; con otra cuenta, el plano en una revision con ambito nuevo y la causa patrimonial de A
/// no exportada (como el host de M; el de L hace lo mismo).
fn relevo(p: &PuenteCuenta, a: &Arc<GlobalArena>, oms: &AtomicU64, vigente: &ArcSwap<OrderExecutor>, b: Arc<OrderExecutor>) {
    let sesion = oms.load(Ordering::SeqCst) + 1;
    let (_, era) = p.relevar_sesion(|| {
        vigente.store(Arc::clone(&b));
        oms.store(sesion, Ordering::SeqCst);
        p.anunciar_cuenta(2);
        let mut libro = a.config.entry_publication_gate.lock().unwrap();
        drawdown::escribir_plano_del_freno(a, || {
            a.capital_exchange.store(f64::NAN, Ordering::Release);
            a.pico_capital.store(0.0, Ordering::Release);
            drawdown::preparar_ambito_del_pico(a);
        });
        let _ = execution_engine::aplanado_inmune::relevar_causa_patrimonial(a, &b, drawdown::ambito_del_plano(a));
        libro.fijar_cuenta(a, 2);
        ((), Some(sesion))
    });
    p.abrir_socket(era.unwrap()).unwrap();
}

// ================================================================== F6: reintento

/// F6 (dictamen V4L :48-63; ventana G:2983-3000 de L). Disparo patrimonial de A comprometido bajo
/// el mutex; reintento pendiente. Un relevo a B gana entre la comprobacion de la autoria y la toma
/// del ejecutor del reintento. Oraculo, por TRANSPORTE: el servidor de B no recibe ninguna peticion
/// del reintento (ni lectura ni cierre). Control: sin relevo, el reintento opera sobre A (su servidor
/// recibe el aplanado). Y si el relevo llega DESPUES de tomar el sujeto, el aplanado sigue en A.
#[tokio::test]
async fn m_f6_reintento_no_toma_el_ejecutor_del_sucesor() {
    let _g = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    for caso in ["relevo en la ventana", "control sin relevo", "relevo tras tomar el sujeto"] {
        let a = GlobalArena::build_in_own_stack(100.0);
        let (p, oms) = puente(&a);
        let (sa, base_a) = servidor().await;
        let (sb, base_b) = servidor().await;
        let ej_a = ejecutor(&base_a, Some(&a));
        let ej_b = ejecutor(&base_b, Some(&a));
        let vigente = ArcSwap::from(Arc::clone(&ej_a));
        let ambito = drawdown::ambito_del_plano(&a);
        let (_, autoria) = p
            .con_equity_del_freno(|_| execution_engine::aplanado_inmune::comprometer_disparo(&a, || vigente.load_full(), Some(ambito)))
            .expect("fixture: disparo comprometido en su ambito");
        let entre = || {
            if caso == "relevo en la ventana" {
                relevo(&p, &a, &oms, &vigente, Arc::clone(&ej_b));
            }
        };
        let sujeto = sujeto_del_reintento_del_host(&p, &a, &vigente, Some(autoria), entre);
        if caso == "relevo tras tomar el sujeto" {
            relevo(&p, &a, &oms, &vigente, Arc::clone(&ej_b));
        }
        if let Some(e) = &sujeto {
            let _ = e.flatten_open_positions().await;
        }
        let (a_total, b_total) = (sa.total(), sb.total());
        println!("V4M-F6 pin={PIN} caso={caso} peticiones_A={a_total} peticiones_B={b_total} cierres_A={} cierres_B={}", sa.cierres(), sb.cierres());
        assert_eq!(b_total, 0, "{PIN}/{caso}: el reintento de A actuo sobre el ejecutor del sucesor B");
        match caso {
            "relevo en la ventana" => assert!(sujeto.is_none() && a_total == 0, "{PIN}: el reintento del ambito relevado siguio"),
            _ => assert!(a_total > 0 && sa.cierres() >= 1, "{PIN}/{caso}: el reintento no aplano su sujeto (montaje)"),
        }
    }
}

// ================================================================== F8: certificado

/// F8 (dictamen V4L :65-73; G:3988-4009 de L). La respuesta de income de la era A supera la guarda
/// del profiler; un relevo a B gana antes de publicar el certificado. Oraculo, por DISCO: el
/// certificado real (`evidencia_demo.json` en un directorio de prueba) no se escribe con la
/// evidencia de A. Control: sin relevo, se escribe una vez con su contenido.
#[tokio::test]
async fn m_f8_certificado_con_testigo_de_era() {
    let _g = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    let dir = std::env::temp_dir().join(format!("tg_v4m_f8_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for caso in ["relevo tras la guarda", "control sin relevo"] {
        let ruta = dir.join(format!("evidencia_demo_{}.json", caso.replace(' ', "_")));
        let a = GlobalArena::build_in_own_stack(100.0);
        let (p, oms) = puente(&a);
        let (_s, base) = servidor().await;
        let vigente = ArcSwap::from(ejecutor(&base, Some(&a)));
        let era = p.era();
        let entre = || {
            if caso == "relevo tras la guarda" {
                relevo(&p, &a, &oms, &vigente, ejecutor(&base, Some(&a)));
            }
        };
        let publicado = certificado_del_host(&p, era, &ruta, 30, 10.0, -1.0, 20, entre);
        let existe = ruta.exists();
        println!("V4M-F8 pin={PIN} caso={caso} publicado={publicado} certificado_en_disco={existe}");
        match caso {
            "relevo tras la guarda" => assert!(!existe, "{PIN}: el certificado se escribio con la evidencia de la era anterior"),
            _ => {
                assert!(existe, "{PIN}: control: el certificado vigente no se escribio");
                let texto = std::fs::read_to_string(&ruta).unwrap();
                assert!(texto.contains("\"cierres\": 30"), "{PIN}: certificado con otro contenido: {texto}");
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ================================================================== A1H: padre e hija

struct Oms {
    _u: MutexGuard<'static, ()>,
    arena: Arc<GlobalArena>,
    sesion: Arc<OmsSession>,
    token: quantum_engine::host_oms::SessionToken,
    ejecutor: Arc<OrderExecutor>,
    servidor: Arc<Servidor>,
}

static UNIVERSO: Mutex<()> = Mutex::new(());

async fn oms() -> Oms {
    let u = UNIVERSO.lock().unwrap_or_else(|e| e.into_inner());
    quantum_arena::symbols::update_dynamic_universe(vec![SIMBOLO.into(), "ETHUSDT".into()]);
    let (servidor, base) = servidor().await;
    let ejecutor = ejecutor(&base, None);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let sesion = Arc::new(OmsSession::new(Arc::new(ArcSwap::from(Arc::clone(&ejecutor))), Arc::new(AtomicBool::new(false))));
    let token = sesion.snapshot_token();
    let filas: Vec<SymbolFees> = quantum_arena::symbols::get_active_universe_ref().iter().map(|s| SymbolFees { symbol: s.to_string(), maker: 0.0, taker: 0.0003 }).collect();
    let planas: Vec<PositionRiskEntry> = quantum_arena::symbols::get_active_universe_ref().iter().map(|s| PositionRiskEntry { symbol: s.to_uppercase(), position_side: "BOTH".into(), ..Default::default() }).collect();
    let fees = FeeSchedule::new(token.generation(), filas).map_err(|e| format!("{e:?}"));
    sesion.apply_snapshot(&token, Ok(planas), Ok(Vec::new()), fees, &arena, 1_000);
    {
        let mut l = arena.config.entry_publication_gate.lock().unwrap();
        l.activar(&arena);
        l.saldo_aceptado(&arena, 100.0, Some(100));
        arena.capital_incompleto.store(false, Ordering::Release);
    }
    Oms { _u: u, arena, sesion, token, ejecutor, servidor }
}

impl Oms {
    fn reserva_real(&self) -> EntryReservation {
        let p = self.arena.coins[0].positions.get_slot(0);
        assert!(p.open_with_tau_and_fee(true, 100.0, 0.01, 10.0, 1_000, 110.0, 90.0, PositionHorizon::Continuous, 0.7, 0.8, 0.04, 60_000));
        self.arena.used_margin.fetch_add(10.0, Ordering::Relaxed);
        let r = EntryReservation { coin_id: 0, slot: 0, generation: p.generation.load(Ordering::Acquire), symbol: SIMBOLO.into() };
        let mut l = self.arena.config.entry_publication_gate.lock().unwrap();
        let c = Some(self.arena.unified_capital.load(Ordering::Relaxed));
        let c = if l.admite(&self.arena, c) { c } else { Some(self.arena.unified_capital.load(Ordering::Relaxed)) };
        l.reservar_con_base(&self.arena, r.clave(), 0.04, c);
        r
    }
}

/// A1H (dictamen V4L :82, :189): la resolucion PRODUCTIVA que ejecuta el host (D1), con reserva y
/// settlement reales, sobre la ruta maker GTX existente: padre parcial 0.005 y remanente hijo
/// `mcT_` por el transporte local. Oraculos:
/// - hija ejecutada (Ok): se confirma lo ejecutado por los DOS terminales, 0.010 (no solo el padre);
/// - ACK de la hija perdido y su terminal FILLED: se confirma 0.010 tras consultar la identidad de
///   la hija (evidencia terminal correlacionada), sin reenviar;
/// - ACK de la hija perdido y la hija sin terminal (NEW): se conserva todo, sin reenvio.
#[tokio::test]
async fn m_a1h_padre_e_hija_por_evidencia_terminal() {
    let _g = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    for (caso, ack_hija, hija, esperado) in [
        ("hija ejecutada", false, ("FILLED", "0.005"), Some(0.01)),
        ("ACK de la hija perdido, terminal", true, ("FILLED", "0.005"), Some(0.01)),
        ("ACK de la hija perdido, sin terminal", true, ("NEW", "0"), None),
    ] {
        let m = oms().await;
        m.servidor.perder_ack_hija.store(ack_hija, Ordering::Release);
        *m.servidor.hija.lock().unwrap() = (hija.0.into(), hija.1.into());
        let r = m.reserva_real();
        let _settlement = tokio::time::timeout(PLAZO, EntrySettlement::acquire(r.clone(), true)).await.expect("settlement");
        let pedido = EntryRequest { symbol: SIMBOLO, is_long: true, quantity: 0.01, step_size: 0.001, tick_size: 0.1, leverage: 1, client_order_id: "mc_v4m_padre", route: EntryRoute::Maker { price: 99.0 } };
        let res = m.sesion.dispatch_entry_reservada(&m.token, m.ejecutor.as_ref(), &pedido, &m.arena, Some(r.clave())).await;
        let posts = m.servidor.peticiones.lock().unwrap().iter().filter(|p| p.metodo == "POST" && p.ruta == "/fapi/v1/order").count();
        let d = quantum_engine::host_oms::resolver_resultado_de_entrada(res.clone(), &m.sesion, &m.token, &m.arena, &m.ejecutor, &r, "mc_v4m_padre", 0.01, PLAZO).await;
        let posts_despues = m.servidor.peticiones.lock().unwrap().iter().filter(|p| p.metodo == "POST" && p.ruta == "/fapi/v1/order").count();
        println!("V4M-A1H pin={PIN} caso={caso} envio={res:?} desenlace={d:?}");
        assert_eq!(posts, posts_despues, "{PIN}/{caso}: la resolucion reenvio");
        match esperado {
            Some(q) => {
                assert!(matches!(d, DesenlaceDeEntrada::Confirmada { ejecutada, .. } if (ejecutada - q).abs() < 1e-12), "{PIN}/{caso}: se confirmo otra cantidad que padre + hija: {d:?}");
                assert!(EntrySettlement::release_for(&r), "{PIN}/{caso}: settlement suelto antes de la proteccion");
            }
            None => {
                assert_eq!(d, DesenlaceDeEntrada::Conservada, "{PIN}/{caso}: sin evidencia de la hija no se conservo");
                assert!(m.arena.coins[0].positions.get_slot(0).is_open() && EntrySettlement::release_for(&r), "{PIN}/{caso}: reserva o settlement perdidos");
            }
        }
    }
}

// ================================================================== D2: sin tranId

/// D2 (dictamen V4L :161): un flujo externo SIN `tranId` atraviesa el transporte productivo
/// (`fetch_income_paged`, la API que consume el host) y el registro de flujos del host. Oraculo: la
/// atribucion de la caida queda incierta en el ambito (`sin_identidad` = 1, origen FlujoIncierto) y
/// la fila no suma cifra ni entra en los agregados de comisiones/funding. Sin reinsertar filas.
#[tokio::test]
async fn m_d2_sin_tranid_atraviesa_transporte_y_host() {
    let _g = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    let a = GlobalArena::build_in_own_stack(100.0);
    let (p, _) = puente(&a);
    let (s, base) = servidor().await;
    *s.income.lock().unwrap() = r#"[{"symbol":"","incomeType":"TRANSFER","income":"-2","asset":"USDT","time":152,"tranId":0,"tradeId":""},
        {"symbol":"BTCUSDT","incomeType":"COMMISSION","income":"-0.1","asset":"USDT","time":153,"tranId":7001,"tradeId":"1"}]"#
        .into();
    let e = ejecutor(&base, Some(&a));
    let entradas = e.fetch_income_paged(&[], 1, 1).await.expect("fixture: ventana agotada");
    assert_eq!(entradas.len(), 1, "{PIN}: la fila sin identidad entro en la lista de los agregados");
    let flujos = Mutex::new(FlujosExternos::default());
    let filas = flujos_de_la_consulta(&entradas);
    let r = quantum_engine::host_oms::registrar_flujos_externos_de_era(&p, p.era(), &a, &flujos, &filas).expect("era vigente");
    let f = flujos.lock().unwrap();
    println!("V4M-D2 pin={PIN} registros={r:?} sin_identidad={} neto={} origen={:?}", f.sin_identidad(), f.neto_incierto(), f.origen_de_la_caida());
    assert_eq!(f.sin_identidad(), 1, "{PIN}: la incertidumbre del flujo sin tranId se perdio en el transporte");
    assert_eq!(f.neto_incierto(), 0.0, "{PIN}: el flujo sin identidad sumo una cifra");
    assert_eq!(f.origen_de_la_caida(), OrigenDeLaCaida::FlujoIncierto);
}
