//! I-12 (Línea I, fila N48 del censo C-F7b): la transición del ejecutor del host al terminar
//! el calentamiento.
//!
//! Con la sesión OMS real, la arena real y reservas reales: sin red, sin claves, sin modelos.
//! La reserva de entrada que el núcleo deja en el tick de la transición es una ranura abierta
//! sin confirmar, y con ella `replace_executor` rehúsa (`OMS_EXECUTOR_BUSY`). El host de main
//! reintentaba en cada mensaje WS: construía otro ejecutor (cliente reqwest nuevo), revocaba la
//! sesión y encolaba otro `refresh` (instantáneas REST), mientras el núcleo siguiera proponiendo.
use arc_swap::ArcSwap;
use execution_engine::executor::OrderExecutor;
use god_engine_core::entry_reservation::EntryReservation;
use quantum_arena::{position::PositionHorizon, GlobalArena};
use quantum_engine::host_oms::{
    OmsSession, ReintentoTransicion, ResultadoTransicion, SnapshotStatus, OMS_EXECUTOR_BUSY,
};
use std::cell::Cell;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, MutexGuard,
};

static UNIVERSE_LOCK: Mutex<()> = Mutex::new(());

struct UniverseGuard {
    previous: Vec<String>,
    _lock: MutexGuard<'static, ()>,
}

impl Drop for UniverseGuard {
    fn drop(&mut self) {
        quantum_arena::symbols::update_dynamic_universe(std::mem::take(&mut self.previous));
    }
}

fn universe() -> UniverseGuard {
    let lock = UNIVERSE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let previous = quantum_arena::symbols::get_active_universe();
    quantum_arena::symbols::update_dynamic_universe(vec!["BTCUSDT".into(), "ETHUSDT".into()]);
    UniverseGuard {
        previous,
        _lock: lock,
    }
}

fn executor(testnet: bool, paper: bool) -> Arc<OrderExecutor> {
    let mut executor = OrderExecutor::new(String::new(), "offline-contract".into(), testnet);
    executor.set_paper_trading(paper);
    Arc::new(executor)
}

/// Sesión del calentamiento ya reconciliada (lista para entradas), como la deja el arranque.
fn sesion_lista(arena: &GlobalArena) -> (Arc<OmsSession>, Arc<AtomicBool>) {
    let ready = Arc::new(AtomicBool::new(false));
    let executors = Arc::new(ArcSwap::from(executor(true, false)));
    let session = Arc::new(OmsSession::new(executors, Arc::clone(&ready)));
    let token = session.snapshot_token();
    let flat = quantum_arena::symbols::get_active_universe_ref()
        .iter()
        .map(
            |symbol| execution_engine::reconciliation::PositionRiskEntry {
                symbol: symbol.to_uppercase(),
                position_side: "BOTH".into(),
                ..Default::default()
            },
        )
        .collect();
    let fees = quantum_arena::symbols::get_active_universe_ref()
        .iter()
        .map(|symbol| quantum_arena::SymbolFees {
            symbol: symbol.to_string(),
            maker: 0.0002,
            taker: 0.0004,
        })
        .collect();
    let schedule =
        quantum_arena::FeeSchedule::new(token.generation(), fees).map_err(|e| format!("{e:?}"));
    let outcome = session.apply_snapshot(&token, Ok(flat), Ok(Vec::new()), schedule, arena, 1_000);
    assert_eq!(outcome.status, SnapshotStatus::Ready, "{outcome:?}");
    assert!(ready.load(Ordering::Acquire));
    (session, ready)
}

/// Lo que el núcleo deja en un tick con entrada propuesta: la ranura abierta sin confirmar y
/// su identidad (la reserva).
fn reservar(arena: &GlobalArena, coin_id: usize, slot: usize, symbol: &str) -> EntryReservation {
    let ranura = arena.coins[coin_id].positions.get_slot(slot);
    assert!(ranura.open_with_horizon(
        true,
        100.0,
        0.01,
        1.0,
        1_000,
        0.0,
        0.0,
        PositionHorizon::Continuous,
    ));
    arena.used_margin.fetch_add(1.0, Ordering::AcqRel);
    EntryReservation {
        coin_id,
        slot,
        generation: ranura.generation.load(Ordering::Acquire),
        symbol: symbol.into(),
    }
}

#[test]
fn i12_1_la_reserva_del_tick_se_suelta_y_la_transicion_publica_al_primer_intento() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = sesion_lista(&arena);
    let generacion_previa = session.snapshot_token().generation();
    let reserva = reservar(&arena, 0, 2, "BTCUSDT");
    // El mecanismo: con la reserva del tick abierta, la sesión está ocupada.
    assert!(session.ocupada(&arena));

    let reintento = ReintentoTransicion::new();
    let construidos = Cell::new(0_u32);
    let paso = session.paso_transicion(&arena, Some(&reserva), &reintento, || {
        construidos.set(construidos.get() + 1);
        executor(false, false)
    });
    assert!(paso.reserva_no_cancelable.is_none());
    let ResultadoTransicion::Publicada {
        generacion,
        ejecutor,
    } = paso.resultado
    else {
        panic!(
            "la transición debe publicar al primer intento: {:?}",
            paso.resultado
        );
    };
    assert_eq!(construidos.get(), 1);
    assert_eq!(generacion, generacion_previa + 1);
    assert!(Arc::ptr_eq(session.snapshot_token().executor(), &ejecutor));
    // La reserva del tick quedó suelta exactamente una vez: ni ranura ni margen fantasma.
    assert!(!arena.coins[0].positions.get_slot(2).is_open());
    assert_eq!(arena.used_margin.load(Ordering::Relaxed), 0.0);
    assert!(reserva.cancel(&arena).is_err());
    // La sustitución revocó la sesión anterior: el ejecutor nuevo reconcilia antes de operar.
    assert!(!ready.load(Ordering::Acquire));
    assert!(!reintento.refresco_en_curso());
}

/// Lo medido: el núcleo propone una entrada en cada uno de 100 mensajes seguidos.
#[test]
fn i12_2_una_senal_persistente_ya_no_construye_ni_encola_un_refresco_por_mensaje() {
    const MENSAJES: u32 = 100;
    let _universe = universe();

    // Protocolo de main: construir, sustituir, y tras el rechazo soltar la reserva y encolar
    // un refresco. Con la reserva del tick abierta la sustitución rehúsa siempre.
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, _ready) = sesion_lista(&arena);
    let mut construidos_main = 0_u32;
    let mut refrescos_main = 0_u32;
    let mut publicada_main = false;
    for _ in 0..MENSAJES {
        let reserva = reservar(&arena, 0, 2, "BTCUSDT");
        construidos_main += 1;
        match session.replace_executor(executor(false, false), &arena) {
            Ok(_) => {
                publicada_main = true;
                break;
            }
            Err(motivo) => {
                assert!(motivo.starts_with("OMS_EXECUTOR_BUSY"));
                assert!(reserva.cancel(&arena).is_ok());
                refrescos_main += 1;
            }
        }
    }
    assert!(!publicada_main);
    assert_eq!((construidos_main, refrescos_main), (MENSAJES, MENSAJES));

    // I-12: el mismo flujo de reservas pasa por `paso_transicion`.
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, _ready) = sesion_lista(&arena);
    let reintento = ReintentoTransicion::new();
    let construidos = Cell::new(0_u32);
    let mut refrescos = 0_u32;
    let mut publicada_en = None;
    for mensaje in 0..MENSAJES {
        let reserva = reservar(&arena, 0, 2, "BTCUSDT");
        let paso = session.paso_transicion(&arena, Some(&reserva), &reintento, || {
            construidos.set(construidos.get() + 1);
            executor(false, false)
        });
        assert!(paso.reserva_no_cancelable.is_none());
        match paso.resultado {
            ResultadoTransicion::Publicada { .. } => {
                publicada_en = Some(mensaje);
                break;
            }
            ResultadoTransicion::Rechazada {
                refresco: Some(_), ..
            } => refrescos += 1,
            otro => panic!("mensaje {mensaje}: {otro:?}"),
        }
    }
    assert_eq!(publicada_en, Some(0));
    assert_eq!((construidos.get(), refrescos), (1, 0));
}

/// Una causa ajena al tick (otra ranura sin confirmar) aplaza sin construir ni revocar.
#[test]
fn i12_3_con_la_sesion_ocupada_por_otra_ranura_no_se_construye_ni_se_revoca() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = sesion_lista(&arena);
    let ajena = reservar(&arena, 1, 0, "ETHUSDT");
    let reintento = ReintentoTransicion::new();
    let construidos = Cell::new(0_u32);
    let mut avisos = 0_u32;
    for _ in 0..50 {
        let reserva = reservar(&arena, 0, 2, "BTCUSDT");
        let paso = session.paso_transicion(&arena, Some(&reserva), &reintento, || {
            construidos.set(construidos.get() + 1);
            executor(false, false)
        });
        assert!(paso.reserva_no_cancelable.is_none());
        let ResultadoTransicion::Ocupada { primer_aviso } = paso.resultado else {
            panic!("debe aplazar: {:?}", paso.resultado);
        };
        avisos += u32::from(primer_aviso);
        // La reserva del tick se soltó aunque la transición se aplace.
        assert!(!arena.coins[0].positions.get_slot(2).is_open());
    }
    assert_eq!(construidos.get(), 0);
    assert_eq!(avisos, 1);
    // Aplazar no revoca: la sesión del calentamiento sigue reconciliada.
    assert!(ready.load(Ordering::Acquire));
    assert!(!reintento.refresco_en_curso());

    // Confirmada la ranura ajena por el exchange, la transición publica al primer intento.
    let ranura = arena.coins[1].positions.get_slot(0);
    assert!(ranura.confirm_generation(ajena.generation).is_ok());
    let paso = session.paso_transicion(&arena, None, &reintento, || {
        construidos.set(construidos.get() + 1);
        executor(false, false)
    });
    assert!(matches!(
        paso.resultado,
        ResultadoTransicion::Publicada { .. }
    ));
    assert_eq!(construidos.get(), 1);
    // Tras publicar, un aplazamiento futuro vuelve a avisar una vez.
    let otra = reservar(&arena, 1, 1, "ETHUSDT");
    let paso = session.paso_transicion(&arena, None, &reintento, || executor(false, false));
    assert!(matches!(
        paso.resultado,
        ResultadoTransicion::Ocupada { primer_aviso: true }
    ));
    assert!(otra.cancel(&arena).is_ok());
}

/// Si la sesión cambia entre la comprobación y la sustitución (aquí, una reserva que aparece
/// mientras se construye), la sustitución rehúsa: un solo testigo de refresco, y hasta que se
/// suelta no se construye otro ejecutor.
#[test]
fn i12_4_un_rechazo_entrega_un_solo_refresco_y_espera_a_que_termine() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, ready) = sesion_lista(&arena);
    let reintento = ReintentoTransicion::new();
    let carrera = Cell::new(None);
    let paso = session.paso_transicion(&arena, None, &reintento, || {
        carrera.set(Some(reservar(&arena, 1, 0, "ETHUSDT")));
        executor(false, false)
    });
    let ResultadoTransicion::Rechazada { motivo, refresco } = paso.resultado else {
        panic!("la sustitución debe rehusar: {:?}", paso.resultado);
    };
    assert_eq!(motivo, OMS_EXECUTOR_BUSY);
    let testigo = refresco.expect("el primer rechazo entrega el testigo del refresco");
    assert!(reintento.refresco_en_curso());
    // El rechazo ya revocó la sesión (lo hace `replace_executor`).
    assert!(!ready.load(Ordering::Acquire));
    if let Some(reserva) = carrera.take() {
        assert!(reserva.cancel(&arena).is_ok());
    }

    let construidos = Cell::new(0_u32);
    for _ in 0..20 {
        let reserva = reservar(&arena, 0, 2, "BTCUSDT");
        let paso = session.paso_transicion(&arena, Some(&reserva), &reintento, || {
            construidos.set(construidos.get() + 1);
            executor(false, false)
        });
        assert!(matches!(paso.resultado, ResultadoTransicion::Esperando));
        assert!(!arena.coins[0].positions.get_slot(2).is_open());
    }
    assert_eq!(construidos.get(), 0);

    // Termina el refresco (o su tarea muere): se suelta el testigo y vuelve a intentar.
    drop(testigo);
    assert!(!reintento.refresco_en_curso());
    let paso = session.paso_transicion(&arena, None, &reintento, || {
        construidos.set(construidos.get() + 1);
        executor(false, false)
    });
    assert!(matches!(
        paso.resultado,
        ResultadoTransicion::Publicada { .. }
    ));
    assert_eq!(construidos.get(), 1);
}

/// La reserva que no se puede soltar (identidad ajena) no se oculta: se informa y la
/// protección queda sucia; la transición no se publica con la ranura abierta.
#[test]
fn i12_5_una_reserva_no_cancelable_se_informa_y_no_se_publica_encima() {
    let _universe = universe();
    let arena = GlobalArena::build_in_own_stack(100.0);
    let (session, _ready) = sesion_lista(&arena);
    let reserva = reservar(&arena, 0, 2, "BTCUSDT");
    let ajena = EntryReservation {
        generation: reserva.generation + 1,
        ..reserva.clone()
    };
    let reintento = ReintentoTransicion::new();
    let paso = session.paso_transicion(&arena, Some(&ajena), &reintento, || executor(false, false));
    assert!(paso.reserva_no_cancelable.is_some());
    assert!(matches!(
        paso.resultado,
        ResultadoTransicion::Ocupada { .. }
    ));
    assert!(arena.coins[0].positions.get_slot(2).is_open());
    assert!(reserva.cancel(&arena).is_ok());
}

/// Guarda de fuente: el host transita por `paso_transicion`, soltando antes la entrada del tick,
/// y el refresco de reintento retiene su testigo.
#[test]
fn i12_6_el_host_transita_por_el_paso_con_un_solo_refresco() {
    let host = include_str!("../src/bin/god_engine.rs");
    assert!(
        !host.contains(".replace_executor("),
        "el host no debe sustituir el ejecutor fuera de paso_transicion"
    );
    let bloque = host
        .split("if newly_transitioned {")
        .nth(2)
        .expect("bloque de la transición del ejecutor");
    let paso = bloque
        .find("oms_session.paso_transicion(")
        .expect("la transición pasa por OmsSession::paso_transicion");
    let soltar = bloque
        .find("new_order = None;")
        .expect("el tick de la transición no abre exposición");
    assert!(soltar < paso);
    assert!(bloque.contains("entry_reservation.as_ref()"));
    assert!(bloque.contains("ResultadoTransicion::Esperando"));
    assert!(bloque.contains("ResultadoTransicion::Ocupada"));
    assert!(bloque.contains("let _refresco = refresco;"));
    assert_eq!(
        host.matches("oms_retry.refresh(").count(),
        1,
        "un solo refresco de reintento, bajo el testigo"
    );
    assert!(host.contains("ReintentoTransicion::new()"));
}
