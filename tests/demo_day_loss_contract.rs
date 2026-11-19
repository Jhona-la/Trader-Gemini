//! DEMO-B (especificacion Demo §B.4, casos B4-B5; revision N1 de NVIDIA, cambio 7): el veto de
//! perdida diaria en la sesion OMS REAL (`OmsSession` y su despachador guardado) con un transporte
//! espia en memoria y un reloj inyectado. Ninguna red ni disco salvo una carpeta temporal.
//!
//! La especificacion propone el filtro `demo_day_loss_` dentro de `tests/host_oms_contract.rs`
//! (archivo del area de Codex QRA); aqui va en un target propio con el mismo prefijo.

use arc_swap::ArcSwap;
use execution_engine::entry_dispatch::{EntryRequest, EntryRoute, EntryTransport};
use execution_engine::executor::OrderExecutor;
use quantum_arena::GlobalArena;
use quantum_engine::host_oms::{GuardaDiariaHost, OmsSession};
use risk_engine::perdida_diaria::{Transicion, MS_POR_DIA_UTC};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

const HOST_OMS: &str = include_str!("../src/host_oms.rs");
const GMDD_GEN5: f64 = 0.50;

static UNIVERSO: Mutex<()> = Mutex::new(());
static RELOJ: AtomicU64 = AtomicU64::new(0);

fn reloj() -> u64 {
    RELOJ.load(Ordering::SeqCst)
}

fn sesion_papel() -> Arc<OmsSession> {
    let mut e = OrderExecutor::new(String::new(), "offline-contract".into(), true);
    e.set_paper_trading(true);
    Arc::new(OmsSession::new(Arc::new(ArcSwap::from(Arc::new(e))), Arc::new(AtomicBool::new(true))))
}

fn ruta(nombre: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "tg_demo_day_loss_{nombre}_{}_{}.txt",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ))
}

/// Una guarda con el dia abierto en el corte (E0 = P0 = 100, frio, gen 5).
fn guarda_con_dia(cuenta: u64, r: std::path::PathBuf, corte: u64) -> Arc<GuardaDiariaHost> {
    RELOJ.store(corte - 1_000, Ordering::SeqCst);
    let g = Arc::new(GuardaDiariaHost::recuperar(cuenta, r, reloj));
    g.publicar_evidencia(vec![(0.0, 0.0)]);
    let _ = g.observar(100.0, 100.0, GMDD_GEN5);
    RELOJ.store(corte + 1_000, Ordering::SeqCst);
    let (t, persistido) = g.observar(100.0, 100.0, GMDD_GEN5);
    assert!(matches!(t, Transicion::DiaAbierto(_)), "{t:?}");
    assert!(persistido.is_ok());
    g
}

#[derive(Default)]
struct Espia {
    arena: Option<Arc<GlobalArena>>,
    equity_al_configurar: Option<f64>,
    configurado: AtomicUsize,
    enviado: AtomicUsize,
}

impl EntryTransport for Espia {
    fn supports_native_iceberg(&self) -> bool {
        false
    }
    async fn configure_leverage(&self, _symbol: &str, _leverage: u32) -> Result<(), String> {
        self.configurado.fetch_add(1, Ordering::SeqCst);
        // Cruce del suelo entre etapas del MISMO intento (antes del primer POST de apertura).
        if let (Some(a), Some(e)) = (&self.arena, self.equity_al_configurar) {
            a.capital_exchange.store(e, Ordering::SeqCst);
        }
        Ok(())
    }
    async fn submit_entry(&self, _request: &EntryRequest<'_>) -> Result<(), String> {
        self.enviado.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

fn pedido() -> EntryRequest<'static> {
    EntryRequest {
        symbol: "BTCUSDT",
        is_long: true,
        quantity: 0.01,
        step_size: 0.001,
        tick_size: 0.1,
        leverage: 1,
        client_order_id: "demo_day_loss_entry",
        route: EntryRoute::Market,
    }
}

#[tokio::test]
async fn demo_day_loss_b4_cruce_del_suelo_antes_del_post_retiene_la_apertura() {
    let _u = UNIVERSO.lock().unwrap_or_else(|e| e.into_inner());
    let corte = 20_600 * MS_POR_DIA_UTC;
    let arena = GlobalArena::build_in_own_stack(100.0);
    let r = ruta("b4");
    let g = guarda_con_dia(41, r.clone(), corte);
    let estado = g.estado().expect("dia");
    let suelo = estado.e0 - estado.b_d;
    let sesion = sesion_papel();
    sesion.initialize_arena(&arena);
    assert!(sesion.vincular_guarda_diaria(Arc::clone(&g)));
    assert!(!sesion.vincular_guarda_diaria(Arc::clone(&g)), "una sola guarda");
    arena.capital_exchange.store(100.0, Ordering::SeqCst);
    let token = sesion.snapshot_token();
    assert!(sesion.check_entry(&token, &arena).is_ok(), "por encima del suelo la entrada es elegible");
    // Control: el mismo despacho sin cruce llega al POST.
    let sano = Espia::default();
    assert!(sesion.dispatch_entry(&token, &sano, &pedido(), &arena).await.is_ok());
    assert_eq!(sano.enviado.load(Ordering::SeqCst), 1);
    // El suelo se cruza en configure_leverage: el POST de apertura queda retenido.
    let espia = Espia { arena: Some(Arc::clone(&arena)), equity_al_configurar: Some(suelo), ..Default::default() };
    let r_envio = sesion.dispatch_entry(&token, &espia, &pedido(), &arena).await;
    assert_eq!(espia.configurado.load(Ordering::SeqCst), 1);
    assert_eq!(espia.enviado.load(Ordering::SeqCst), 0, "cero POST de apertura");
    let motivo = r_envio.expect_err("retenida");
    assert!(motivo.contains("DAY_LOSS_LOCKED"), "{motivo}");
    // Enclavado: un rebote no rearma, ni en la entrada ni en la frontera.
    arena.capital_exchange.store(1_000.0, Ordering::SeqCst);
    let e = sesion.check_entry(&token, &arena).expect_err("enclavado");
    assert!(e.contains("DAY_LOSS_LOCKED"), "{e}");
    let otro = Espia::default();
    assert!(sesion.dispatch_entry(&token, &otro, &pedido(), &arena).await.is_err());
    assert_eq!((otro.configurado.load(Ordering::SeqCst), otro.enviado.load(Ordering::SeqCst)), (0, 0));
    // El latch quedo persistido: un reinicio en el mismo dia lo recupera.
    RELOJ.store(corte + 12 * 3_600_000, Ordering::SeqCst);
    let reinicio = GuardaDiariaHost::recuperar(41, r.clone(), reloj);
    assert_eq!(reinicio.estado().map(|e| e.bloqueado), Some(true));
    assert!(reinicio.permite_apertura(&arena).is_err());
    let _ = std::fs::remove_file(&r);
}

#[tokio::test]
async fn demo_day_loss_b4_sin_ancla_o_equity_invalida_retiene_y_sin_guarda_no_veta() {
    let _u = UNIVERSO.lock().unwrap_or_else(|e| e.into_inner());
    let corte = 20_700 * MS_POR_DIA_UTC;
    let arena = GlobalArena::build_in_own_stack(100.0);
    // Arranque despues del corte sin estado de hoy: DAY_EQUITY_ND.
    RELOJ.store(corte + 5_000, Ordering::SeqCst);
    let r = ruta("nd");
    let g = Arc::new(GuardaDiariaHost::recuperar(42, r.clone(), reloj));
    let sesion = sesion_papel();
    sesion.initialize_arena(&arena);
    assert!(sesion.vincular_guarda_diaria(Arc::clone(&g)));
    arena.capital_exchange.store(100.0, Ordering::SeqCst);
    let e = sesion.check_entry(&sesion.snapshot_token(), &arena).expect_err("sin ancla");
    assert!(e.contains("DAY_EQUITY_ND causa=SIN_ANCLA"), "{e}");
    // Dia abierto pero equity no finita: N/D (no se enclava, se retiene).
    let g2 = guarda_con_dia(43, ruta("nan"), corte);
    let s2 = sesion_papel();
    s2.initialize_arena(&arena);
    s2.vincular_guarda_diaria(Arc::clone(&g2));
    arena.capital_exchange.store(f64::NAN, Ordering::SeqCst);
    let e2 = s2.check_entry(&s2.snapshot_token(), &arena).expect_err("NaN");
    assert!(e2.contains("DAY_EQUITY_ND causa=EQUITY_INVALIDA"), "{e2}");
    assert_eq!(g2.estado().map(|e| e.bloqueado), Some(false));
    // Control: una sesion sin guarda (otros perfiles) no recibe el veto.
    let s3 = sesion_papel();
    s3.initialize_arena(&arena);
    arena.capital_exchange.store(1.0, Ordering::SeqCst);
    assert!(s3.check_entry(&s3.snapshot_token(), &arena).is_ok());
    let _ = std::fs::remove_file(&r);
}

/// Ancla del cableado del binario (en linea en `god_engine.rs`): la guarda nace solo en
/// `demo-testnet`, se vincula a la sesion OMS, observa la equity y el pico del MISMO paso del
/// freno duro, recibe los flujos externos vigentes y la evidencia de la envolvente.
#[test]
fn demo_day_loss_b4_cableado_del_binario() {
    let h = include_str!("../src/bin/god_engine.rs").replace("\r\n", "\n");
    let nace = h.find("if contexto_arranque.es_demo_testnet() {\n        let ruta = std::path::PathBuf::from(quantum_arena::paths::env_data_path_for(").expect("solo demo-testnet");
    let vincula = h.find("let _ = oms_session.vincular_guarda_diaria(Arc::clone(&g));").expect("vinculada");
    assert!(nace < vincula);
    let freno = h.find("let freno = risk_engine::drawdown::freno_duro(&arena_imm, peak_capital, cap);").expect("freno");
    let observa = h.find("g.observar_con_aviso(\n                            cap,\n                            peak_capital,").expect("observa");
    assert!(freno < observa, "la misma equity y pico del paso del freno");
    assert!(h.contains("g.registrar_flujo(e.time, e.tran_id, &e.asset, e.income)"));
    assert_eq!(h.matches("publicar_evidencia_diaria(&guarda_diaria, &risk_envelope);").count(), 3, "tras restaurar y tras cada cierre");
}

#[test]
fn demo_day_loss_b5_el_veto_solo_esta_en_las_aperturas() {
    let h = HOST_OMS.replace("\r\n", "\n");
    let produccion = h.split("#[cfg(test)]").next().unwrap_or("");
    // Una definicion y dos usos: la admision de entrada y el validador de la frontera del POST.
    assert_eq!(produccion.matches("self.veto_diario(arena)").count(), 2);
    let entrada = produccion.find("fn check_entry_locked(").expect("admision");
    let frontera = produccion.find("pub fn entrega_vigente(").expect("frontera");
    let primer = produccion.find("self.veto_diario(arena)").expect("uso");
    assert!(entrada < primer && primer < frontera);
    // El freno duro, el kill-switch, el STOP y el fee breaker no leen la guarda diaria.
    for ajeno in ["fn paso_del_freno_del_inmune(", "pub fn breaker_de_ingreso("] {
        let i = produccion.find(ajeno).expect(ajeno);
        let tramo = &produccion[i..(i + 3_000).min(produccion.len())];
        assert!(!tramo.contains("guarda_diaria"), "{ajeno}");
    }
}
