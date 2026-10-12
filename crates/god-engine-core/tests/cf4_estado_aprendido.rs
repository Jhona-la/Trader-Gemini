//! C-F8 · estado aprendido (Línea C, CL-108; decisión del dueño «Guardar
//! todo», 2026-10-12): el recuento de cierres, la evidencia por moneda, los
//! calibradores, el freno de la estrategia y lo medido sobreviven al
//! reinicio.
//!
//! RED. Un núcleo nuevo (como tras reiniciar el host) arranca con
//! `trade_count = 0` en todas las monedas, calibradores vacíos y el freno
//! suelto: es lo que pasaba antes. GREEN. Restaurando el estado guardado, el
//! núcleo nuevo tiene lo mismo que el viejo.
//!
//! Fixture. Dos símbolos registrados (BTCUSDT y ETHUSDT); el núcleo viejo
//! cierra 7 operaciones en ETH (sale de la fase de sonda de D4), alimenta 40
//! observaciones a sus calibradores y registra 30 cierres en el freno de la
//! estrategia.

use god_engine_core::estado_aprendido::{cargar, guardar, VERSION_ESTADO};
use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering::Relaxed;

fn registrar_simbolos() {
    quantum_arena::symbol_registry::update_registry(
        ["BTCUSDT", "ETHUSDT"]
            .iter()
            .map(|s| quantum_arena::symbol_registry::SymbolSpec {
                symbol: (*s).into(),
                step_size: 0.001,
                tick_size: 0.01,
                min_qty: 0.001,
                min_notional: 5.0,
                max_leverage: 20,
                maker_fee: 0.0002,
                taker_fee: 0.0004,
                is_shadow: false,
            })
            .collect(),
    );
}

fn nucleo() -> GodEngineCore {
    GodEngineCore::new(GlobalArena::build_in_own_stack(1_000.0))
}

/// El núcleo «viejo», con historia.
fn con_historia() -> GodEngineCore {
    let mut c = nucleo();
    let m = &c.arena.coins[1].metrics;
    m.trade_count.store(7, Relaxed);
    m.win_rate.store(0.571, Relaxed);
    m.profit_factor.store(1.4, Relaxed);
    m.kelly_fraction.store(0.08, Relaxed);
    m.gross_wins.store(3.2, Relaxed);
    m.gross_losses.store(2.3, Relaxed);
    m.pnl_realized.store(0.9, Relaxed);
    for i in 0..40u64 {
        let score = 0.35 + 0.01 * (i % 30) as f64;
        c.alimentar_calibradores(1, score, i % 3 == 0, 1_000_000 + i * 60_000);
    }
    for i in 0..30 {
        let r = if i % 3 == 0 { 1.96 } else { -1.29 };
        c.risk_engine.freno_estrategia.registrar(r, 1.96, 1.29);
    }
    c.arena.riesgo_por_operacion.store(0.002, Relaxed);
    c.arena.ganancia_en_stops.store(1.96, Relaxed);
    c.arena.perdida_en_stops.store(1.29, Relaxed);
    c
}

/// Lo que el núcleo sabe de ETH: (cierres, acierto, p calibrada de 0,5 a las
/// 2 h del último dato, valor del freno, freno echado, riesgo medido).
fn huella(c: &GodEngineCore) -> (usize, f64, f64, f64, bool, f64) {
    (
        c.arena.coins[1].metrics.trade_count.load(Relaxed),
        c.arena.coins[1].metrics.win_rate.load(Relaxed),
        c.calibrator_by_coin[1].calibrate_at(0.5, 1_000_000 + 41 * 60_000),
        c.risk_engine.freno_estrategia.ville.e_value,
        c.risk_engine.freno_estrategia.rechazada(),
        c.arena.riesgo_por_operacion.load(Relaxed),
    )
}

#[test]
fn cl108_el_estado_aprendido_sobrevive_al_reinicio() {
    registrar_simbolos();
    let viejo = con_historia();
    let ruta = std::env::temp_dir().join(format!("cl108_{}_estado.json", std::process::id()));
    guardar(&ruta, &viejo.estado_aprendido()).expect("guardar");

    // Reinicio sin restaurar: lo aprendido se pierde.
    let sin = nucleo();
    // Reinicio restaurando.
    let mut con = nucleo();
    let estado = cargar(&ruta).expect("leer el estado guardado");
    let n = con.restaurar_estado_aprendido(&estado);
    let _ = std::fs::remove_file(&ruta);

    let (h_viejo, h_sin, h_con) = (huella(&viejo), huella(&sin), huella(&con));
    println!("observado: viejo {h_viejo:?}\n           sin restaurar {h_sin:?}\n           restaurando ({n} monedas) {h_con:?}");
    assert_eq!(h_sin.0, 0, "precondición: sin restaurar, el recuento vuelve a 0");
    assert!(risk_engine::sonda::en_fase_de_sonda(&sin.arena, 1), "precondición: ETH vuelve a la fase de sonda");
    assert_eq!(h_con.0, 7);
    assert!(!risk_engine::sonda::en_fase_de_sonda(&con.arena, 1), "con 7 cierres guardados, ETH no es sonda");
    assert_eq!(h_con.1, h_viejo.1);
    assert!((h_con.2 - h_viejo.2).abs() < 1e-9, "calibrador: {} frente a {}", h_con.2, h_viejo.2);
    assert!((h_sin.2 - h_viejo.2).abs() > 1e-6, "precondición: el calibrador vacío da otra p");
    assert_eq!(h_con.3, h_viejo.3, "freno de la estrategia");
    assert_eq!(h_con.4, h_viejo.4);
    assert_eq!(h_con.5, h_viejo.5);
    // El calibrador global también.
    let p = |c: &GodEngineCore| c.confidence_calibrator.calibrate_at(0.6, 1_000_000 + 41 * 60_000);
    assert!((p(&con) - p(&viejo)).abs() < 1e-9);
}

/// Un freno echado sigue echado tras el reinicio.
#[test]
fn cl108_un_freno_echado_sigue_echado() {
    registrar_simbolos();
    let mut viejo = nucleo();
    let mut n = 0;
    while !viejo.risk_engine.freno_estrategia.rechazada() && n < 5_000 {
        // Todo pérdidas: H0 (+0,30 R) cae enseguida.
        viejo.risk_engine.freno_estrategia.registrar(-1.29, 1.96, 1.29);
        n += 1;
    }
    assert!(viejo.risk_engine.freno_estrategia.rechazada());
    let mut con = nucleo();
    con.restaurar_estado_aprendido(&viejo.estado_aprendido());
    assert!(con.risk_engine.freno_estrategia.rechazada());
}

/// Un archivo corrupto, ausente o de otra versión no se restaura.
#[test]
fn cl108_un_archivo_invalido_no_se_restaura() {
    registrar_simbolos();
    let ruta = std::env::temp_dir().join(format!("cl108_{}_corrupto.json", std::process::id()));
    std::fs::write(&ruta, "{ esto no es json").unwrap();
    assert!(cargar(&ruta).is_none());
    let mut e = nucleo().estado_aprendido();
    e.version = VERSION_ESTADO + 1;
    guardar(&ruta, &e).unwrap();
    assert!(cargar(&ruta).is_none(), "otra versión");
    let _ = std::fs::remove_file(&ruta);
    assert!(cargar(&ruta).is_none(), "sin archivo");
}

/// CL-108d (T2-3 de la revisión de la Línea G) — el PnL de sesiones anteriores
/// no entra en el acumulador de la sesión. `pnl_realized` lo leen como PnL de
/// ESTA sesión el demonio en línea (retornos por diferencias), la deriva vivo
/// frente al control y el validador de paridad (capital = inicial + Σ PnL).
/// Esperado: tras restaurar, el acumulador de la sesión está en 0 y el
/// histórico se conserva en lo que se guarda (el de antes + el de la sesión).
#[test]
fn cl108d_el_pnl_historico_no_entra_en_el_acumulador_de_la_sesion() {
    registrar_simbolos();
    let viejo = con_historia();
    let estado = viejo.estado_aprendido();
    let mut con = nucleo();
    con.restaurar_estado_aprendido(&estado);
    let sesion = con.arena.coins[1].metrics.pnl_realized.load(Relaxed);
    println!("observado: PnL de la sesión tras restaurar {sesion} (histórico 0,9)");
    assert_eq!(sesion, 0.0, "el acumulador de la sesión empieza en 0");
    // Un cierre de esta sesión de +0,25: lo guardado es el histórico + la sesión.
    con.arena.coins[1].metrics.pnl_realized.store(0.25, Relaxed);
    let guardado = con
        .estado_aprendido()
        .monedas
        .iter()
        .find(|m| m.simbolo == "ETHUSDT")
        .map(|m| m.metricas.pnl_realized)
        .unwrap();
    assert!((guardado - 1.15).abs() < 1e-12, "guardado {guardado}");
}

/// CL-108e (T2-2 de la revisión de la Línea G) — se restaura todo lo que
/// aprende el cierre por moneda, no sólo el recuento: el sesgo epigenético
/// (multiplica la exposición en el riesgo y no se olvida con el tiempo), el
/// umbral epigenético con su ancla (el último cierre) y el anillo de
/// siniestros de Lundberg (aprieta el tope del grupo).
/// Escenario: tres pérdidas netas del −2 % a τ = 30 min bajan el sesgo de
/// ETH de 1,0 a 0,55. Esperado: tras reiniciar y restaurar, el sesgo, el
/// umbral y el último cierre son los mismos.
#[test]
fn cl108e_se_restaura_la_epigenetica_y_lundberg() {
    registrar_simbolos();
    let mut viejo = con_historia();
    // 300 cierres con deriva positiva: el anillo de 256 da la vuelta.
    for i in 0..300 {
        viejo.observar_siniestro(1, if i % 3 == 2 { -0.010 } else { 0.012 });
    }
    let eth = &viejo.arena.coins[1];
    for i in 0..3u64 {
        eth.apply_spectral_epigenetic_feedback_with_time(-0.02, 60_000, 1_800_000.0, 2_000_000 + i * 60_000);
    }
    let epi = |c: &GodEngineCore| {
        let m = &c.arena.coins[1];
        (
            m.epigenetic_bias.load(Relaxed),
            m.epigenetic_threshold_modifier.load(Relaxed),
            m.last_close_ts.load(Relaxed),
            m.last_close_was_win.load(Relaxed),
        )
    };
    let mut con = nucleo();
    con.restaurar_estado_aprendido(&viejo.estado_aprendido());
    let (e_viejo, e_con) = (epi(&viejo), epi(&con));
    println!("observado: epigenética vieja {e_viejo:?}; restaurada {e_con:?}");
    assert!((e_viejo.0 - 0.55).abs() < 1e-9, "precondición: el sesgo bajó a 0,55");
    assert_eq!(e_con, e_viejo, "sesgo, umbral, ancla y último resultado");
    // Lundberg: la cota publicada es la misma (el anillo restaurado empieza
    // en otra posición: las sumas cambian de orden, de ahí la tolerancia).
    let r = |c: &GodEngineCore| c.arena.registry.get_for_coin_or(1, "lundberg_r_nocional", -1.0);
    println!("observado: R de Lundberg viejo {}; restaurado {}; sin restaurar {}", r(&viejo), r(&con), r(&nucleo()));
    assert!(r(&viejo) > 0.0, "precondición: hay R medido");
    assert!((r(&con) - r(&viejo)).abs() <= 1e-9 * r(&viejo), "R restaurado");
}
