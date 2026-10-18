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

use god_engine_core::estado_aprendido::{cargar, guardar, Arranque, Carga, VERSION_ESTADO};
use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering::Relaxed;

/// Las pruebas cambian el registro global de símbolos (una lo reordena): se
/// serializan.
static ENTORNO: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn registrar_simbolos() -> std::sync::MutexGuard<'static, ()> {
    let g = ENTORNO.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    registrar(&["BTCUSDT", "ETHUSDT"]);
    g
}

fn registrar(simbolos: &[&str]) {
    quantum_arena::symbols::update_dynamic_universe(simbolos.iter().map(|s| s.to_string()).collect());
    quantum_arena::symbol_registry::update_registry(
        simbolos
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

/// Lo que el núcleo sabe de ETH: (cierres, acierto, p calibrada de 0,5 a
/// 2 min del último dato, valor del freno, freno echado, riesgo medido). El
/// olvido a +7 días lo prueba `cl119_olvido_freno_completo_y_universo_reordenado`.
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
    let _g = registrar_simbolos();
    let viejo = con_historia();
    let ruta = std::env::temp_dir().join(format!("cl108_{}_estado.json", std::process::id()));
    guardar(&ruta, &viejo.estado_aprendido()).expect("guardar");

    // Reinicio sin restaurar: lo aprendido se pierde.
    let sin = nucleo();
    // Reinicio restaurando.
    let mut con = nucleo();
    let Carga::Leido(estado) = cargar(&ruta) else {
        panic!("leer el estado guardado");
    };
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
    let _g = registrar_simbolos();
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

/// PR70-T2 de la revisión de la Línea G — la caída del freno duro en stops
/// (CL-130) se guarda con el estado aprendido y vuelve al reiniciar
/// (ADR-0017 §2b). Ninguna prueba la cubría: borrar su captura o su
/// restauración no ponía nada en rojo.
#[test]
fn cl130_la_caida_en_stops_sobrevive_al_reinicio() {
    let _g = registrar_simbolos();
    let viejo = nucleo();
    for r in [2.4, -1.3, -1.3, 2.4, -1.3, -1.3, -1.3] {
        risk_engine::drawdown::registrar_resultado_en_stops(&viejo.arena, r);
    }
    let caida = risk_engine::drawdown::caida_en_stops(&viejo.arena);
    assert!((caida - 4.1).abs() < 1e-9, "precondición: 2,4 de pico menos -1,7: {caida}");
    let ruta = std::env::temp_dir().join(format!("cl130_{}_estado.json", std::process::id()));
    guardar(&ruta, &viejo.estado_aprendido()).expect("guardar");
    let Carga::Leido(estado) = cargar(&ruta) else {
        panic!("leer el estado guardado");
    };
    let _ = std::fs::remove_file(&ruta);
    let sin = nucleo();
    let mut con = nucleo();
    con.restaurar_estado_aprendido(&estado);
    println!(
        "observado: caída en stops vieja {caida:.2}, sin restaurar {:.2}, restaurada {:.2}",
        risk_engine::drawdown::caida_en_stops(&sin.arena),
        risk_engine::drawdown::caida_en_stops(&con.arena)
    );
    assert_eq!(risk_engine::drawdown::caida_en_stops(&sin.arena), 0.0, "precondición");
    assert_eq!(con.arena.resultado_en_stops.load(Relaxed), viejo.arena.resultado_en_stops.load(Relaxed));
    assert_eq!(con.arena.pico_en_stops.load(Relaxed), viejo.arena.pico_en_stops.load(Relaxed));
    assert_eq!(risk_engine::drawdown::caida_en_stops(&con.arena), caida);
}

/// CL-108f (T2-1 de la revisión de la Línea G) — un archivo ilegible no es
/// un archivo ausente. Antes `cargar` devolvía `None` en los dos casos y el
/// host arrancaba en cero: el freno de la estrategia suelto y, en el primer
/// cierre, la única copia de la evidencia sobrescrita.
/// Esperado:
/// - sin archivo ⇒ `Ausente` y arranque en cero;
/// - corrupto, truncado o con un valor fuera de rango ⇒ `Ilegible`; de otra
///   versión ⇒ `OtraVersion`; en ambos el arranque queda `Bloqueado`, con el
///   freno echado, sin permiso para guardar y sin tocar el archivo;
/// - `guardar` no escribe un estado inválido y deja el anterior intacto.
#[test]
fn cl108f_un_archivo_ilegible_no_es_un_archivo_ausente() {
    let _g = registrar_simbolos();
    let dir = std::env::temp_dir().join(format!("cl108f_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ruta = dir.join("estado_aprendido.json");
    let _ = std::fs::remove_file(&ruta);

    // Sin archivo: se empieza de cero, con el freno suelto.
    assert!(matches!(cargar(&ruta), Carga::Ausente));
    let mut c = nucleo();
    let a = c.arrancar_con_estado(cargar(&ruta));
    assert_eq!(a, Arranque::SinEstado);
    assert!(a.permite_guardar() && !c.risk_engine.freno_estrategia.rechazada());

    // Un estado válido, para truncarlo y estropearlo.
    let bueno = serde_json::to_string(&con_historia().estado_aprendido()).unwrap();
    let mut fuera_de_rango = con_historia().estado_aprendido();
    fuera_de_rango.monedas[1].metricas.win_rate = 1.5;
    let mut otra_version = con_historia().estado_aprendido();
    otra_version.version = VERSION_ESTADO + 1;
    for (caso, texto) in [
        ("corrupto", "{ esto no es json".to_string()),
        ("truncado", bueno[..bueno.len() / 2].to_string()),
        ("fuera de rango", serde_json::to_string(&fuera_de_rango).unwrap()),
        ("otra versión", serde_json::to_string(&otra_version).unwrap()),
    ] {
        std::fs::write(&ruta, &texto).unwrap();
        let carga = cargar(&ruta);
        println!("observado: {caso} ⇒ {:?}", match &carga {
            Carga::Leido(_) => "Leido".to_string(),
            otra => format!("{otra:?}").chars().take(90).collect(),
        });
        let como_se_espera = if caso == "otra versión" {
            matches!(carga, Carga::OtraVersion(_))
        } else {
            matches!(carga, Carga::Ilegible(_))
        };
        assert!(como_se_espera, "{caso}");
        let mut c = nucleo();
        let a = c.arrancar_con_estado(carga);
        assert!(matches!(a, Arranque::Bloqueado(_)), "{caso}: {a:?}");
        assert!(!a.permite_guardar(), "{caso}: esta sesión no guarda");
        assert!(c.risk_engine.freno_estrategia.rechazada(), "{caso}: freno echado");
        assert_eq!(std::fs::read_to_string(&ruta).unwrap(), texto, "{caso}: el archivo queda intacto");
    }

    // `guardar` no escribe un estado que no se podría leer.
    guardar(&ruta, &con_historia().estado_aprendido()).unwrap();
    let antes = std::fs::read_to_string(&ruta).unwrap();
    let mut nan = con_historia().estado_aprendido();
    nan.freno_estrategia.e_value = f64::NAN;
    assert!(guardar(&ruta, &nan).is_err(), "un NaN no se guarda");
    assert_eq!(std::fs::read_to_string(&ruta).unwrap(), antes, "el archivo anterior queda intacto");
    assert!(matches!(cargar(&ruta), Carga::Leido(_)));
    let _ = std::fs::remove_dir_all(&dir);
}

/// CL-108d (T2-3 de la revisión de la Línea G) — el PnL de sesiones anteriores
/// no entra en el acumulador de la sesión. `pnl_realized` lo leen como PnL de
/// ESTA sesión el demonio en línea (retornos por diferencias), la deriva vivo
/// frente al control y el validador de paridad (capital = inicial + Σ PnL).
/// Esperado: tras restaurar, el acumulador de la sesión está en 0 y el
/// histórico se conserva en lo que se guarda (el de antes + el de la sesión).
#[test]
fn cl108d_el_pnl_historico_no_entra_en_el_acumulador_de_la_sesion() {
    let _g = registrar_simbolos();
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
    let _g = registrar_simbolos();
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
    // F3.4-07 (Ola Ω109): Relajación continua Ornstein-Uhlenbeck (tau = 30 min)
    // entre cierres a dt=60s: el sesgo baja de 1.0 a ~0.5646 en lugar del valor estático 0.55.
    assert!((e_viejo.0 - 0.5645915371729564).abs() < 1e-9, "precondición: el sesgo bajó con relajación OU");
    assert_eq!(e_con, e_viejo, "sesgo, umbral, ancla y último resultado");
    // Lundberg: la cota publicada es la misma (el anillo restaurado empieza
    // en otra posición: las sumas cambian de orden, de ahí la tolerancia).
    let r = |c: &GodEngineCore| c.arena.registry.get_for_coin_or(1, "lundberg_r_nocional", -1.0);
    println!("observado: R de Lundberg viejo {}; restaurado {}; sin restaurar {}", r(&viejo), r(&con), r(&nucleo()));
    assert!(r(&viejo) > 0.0, "precondición: hay R medido");
    assert!((r(&con) - r(&viejo)).abs() <= 1e-9 * r(&viejo), "R restaurado");
}

/// CL-118 (T2-5 de la revisión de la Línea G) — la evidencia de una moneda
/// que no está en el universo de este arranque no se pierde al guardar.
/// Antes la foto sólo llevaba las monedas del universo y `guardar` reescribía
/// el archivo entero: en el primer cierre desaparecía.
#[test]
fn cl118_la_evidencia_de_una_moneda_fuera_del_universo_se_conserva() {
    let _g = registrar_simbolos();
    let mut estado = con_historia().estado_aprendido();
    let mut ausente = estado.monedas[1].clone();
    ausente.simbolo = "NOESTAUSDT".into();
    ausente.metricas.trade_count = 42;
    estado.monedas.push(ausente.clone());
    let mut c = nucleo();
    c.restaurar_estado_aprendido(&estado);
    let guardado = c.estado_aprendido();
    let vuelve = guardado.monedas.iter().find(|m| m.simbolo == "NOESTAUSDT");
    println!("observado: la moneda fuera del universo se guarda: {:?}", vuelve.map(|m| m.metricas.trade_count));
    assert_eq!(vuelve, Some(&ausente));
}

/// CL-119 (T2-8 y T3-05 de la revisión de la Línea G) — lo que la prueba de
/// CL-108 no cubría:
/// - el olvido del calibrador (semivida de 12 h) sobrevive a la
///   restauración: a +7 días el restaurado y el viejo dan la misma p (ya
///   ≈ la puntuación), lo que exige restaurar las marcas de tiempo;
/// - el freno de la estrategia se restaura entero, no sólo e y el veredicto;
/// - con el universo en otro orden (ETH en otro slot), ETH recupera lo suyo.
#[test]
fn cl119_olvido_freno_completo_y_universo_reordenado() {
    let g = registrar_simbolos();
    let viejo = con_historia();
    let estado = viejo.estado_aprendido();
    let mut con = nucleo();
    con.restaurar_estado_aprendido(&estado);
    let semana = 1_000_000 + 39 * 60_000 + 7 * 24 * 3_600_000;
    let (p_viejo, p_con) = (
        viejo.calibrator_by_coin[1].calibrate_at(0.5, semana),
        con.calibrator_by_coin[1].calibrate_at(0.5, semana),
    );
    println!("observado: p a +7 días, vieja {p_viejo}, restaurada {p_con}");
    assert!((p_con - p_viejo).abs() < 1e-9);
    assert!((p_viejo - 0.5).abs() < 0.01, "a +7 días el calibrador ya olvidó: {p_viejo}");
    assert_eq!(con.estado_aprendido().freno_estrategia, estado.freno_estrategia);

    // Universo en otro orden: ETH en el slot 0.
    registrar(&["ETHUSDT", "BTCUSDT"]);
    let mut otro = nucleo();
    otro.restaurar_estado_aprendido(&estado);
    let eth_nuevo = quantum_arena::symbols::get_coin_id("ETHUSDT").expect("ETH registrada");
    println!("observado: ETH pasa al slot {eth_nuevo}");
    assert_eq!(eth_nuevo, 0);
    assert_eq!(otro.arena.coins[0].metrics.trade_count.load(Relaxed), 7);
    assert_eq!(otro.arena.coins[1].metrics.trade_count.load(Relaxed), 0, "BTC no hereda lo de ETH");
    drop(g);
}

/// CL-119 (T3-05): la restauración reproduce las observaciones que anota el
/// CIERRE del núcleo. Contrato de fuente (sin comentarios): el cierre
/// alimenta los calibradores por `estado_aprendido::alimentar`, que anota
/// cada observación; si los alimentara directamente, no se guardarían.
#[test]
fn cl119_el_cierre_anota_lo_que_alimenta() {
    let fuente: String = include_str!("../src/lib.rs")
        .lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<String>()
        .split_whitespace()
        .collect();
    assert!(fuente.contains(
        "estado_aprendido::alimentar(&mutself.calibrator_by_coin,&mutself.registro_calibrador,&mutself.confidence_calibrator,&mutself.registro_calibrador_global,"
    ));
    assert!(!fuente.contains(".update_at(score_at_entry"), "nadie alimenta los calibradores sin anotar");
}
