//! C-F6 · Censo de estado sin cota del proceso vivo (Línea C, CL-139…CL-145).
//!
//! El censo (`docs/audit/linea_c/estado_sin_cota_2026-10-18.tsv`) examinó 539
//! estructuras alcanzables desde `god_engine` y no encontró memoria que crezca
//! sin cota en el núcleo. Lo que crece sin tope son los avisos del log, cuyo
//! volumen escalaba con la tasa del feed o con los núcleos que el proceso
//! construye. Este archivo fija los arreglos del host y del núcleo; las piezas
//! puras tienen sus pruebas en su crate (estrangulador, auditor de trayectoria,
//! recarga de modelos, base forense, log rotativo, nombres del orquestador).

use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use std::sync::{atomic::Ordering::Relaxed, Arc, Mutex};

static ENTORNO: Mutex<()> = Mutex::new(());

const HOST: &str = include_str!("../../../src/bin/god_engine.rs");
const DEMONIO: &str = include_str!("../../evolution-engine/src/online_daemon.rs");
const SOMBRAS: &str = include_str!("../../evolution-engine/src/random_forest.rs");
const DARWIN: &str = include_str!("../src/darwin.rs");
const NUCLEO: &str = include_str!("../src/lib.rs");

/// El código sin comentarios de línea, para no aceptar un comentario como uso.
fn codigo(fuente: &str) -> String {
    fuente
        .lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Las `n` líneas de código que preceden a cada aparición de `marca`.
fn contextos(fuente: &str, marca: &str, n: usize) -> Vec<String> {
    let lineas: Vec<&str> = fuente.lines().collect();
    lineas
        .iter()
        .enumerate()
        .filter(|(_, l)| l.contains(marca))
        .map(|(i, _)| lineas[i.saturating_sub(n)..=i].join("\n"))
        .collect()
}

/// Esperado: cada aviso que escalaba con la tasa del feed pasa por su
/// estrangulador. Antes, LATENCY_PANIC y LIQUIDATION REJECT escribían una
/// línea por evento y el espectro una tanda cada 100 mensajes.
#[test]
fn cl139_los_avisos_del_host_tienen_tope_por_tiempo() {
    let host = codigo(HOST);
    for (marca, estrangulador) in [
        ("[LATENCY_PANIC]", "aviso_latencia.admitir("),
        ("[LIQUIDATION REJECT]", "aviso_liquidacion.admitir("),
        ("[KILL_SWITCH]", "aviso_kill_switch.admitir("),
    ] {
        let sitios = contextos(&host, marca, 2);
        assert!(!sitios.is_empty(), "{marca} sigue existiendo");
        for s in &sitios {
            assert!(s.contains(estrangulador), "{marca} sin estrangulador:\n{s}");
        }
    }
    assert!(host.contains("cadencia_espectro.admitir("));
    assert!(host.contains(".filter(|_| toca_espectro)"), "el espectro sólo cuando toca");
}

/// Esperado: la traza sin posición se descarta y sólo una divergencia nueva
/// se escribe; el rechazo repetido de un modelo no se escribe.
#[test]
fn cl140_cl141_trayectoria_y_recarga_avisan_lo_nuevo() {
    let host = codigo(HOST);
    // CL-140c: una traza por ranura. El estado de las ranuras se toma antes
    // del evento y la conciliación (TCE de lo que cerró el núcleo, descarte
    // de lo demás) va antes de evaluar el tick.
    let antes = host.find("let ranuras_antes = ranuras_abiertas(").expect("ranuras antes del evento");
    let evento = host.find("engine_real.process_event(").expect("evento");
    let conciliacion = host.find("trajectory_auditor.conciliar(").expect("conciliación");
    let evaluacion = host.find("trajectory_auditor.evaluate_tick(").expect("evaluación");
    assert!(antes < evento && evento < conciliacion && conciliacion < evaluacion);
    assert!(!host.contains("trajectory_auditor.record_exit("), "el cierre sale de conciliar");
    assert!(!host.contains("trajectory_auditor.descartar("), "el descarte sale de conciliar");
    // La traza nace con la ranura y la generación de la reserva y sólo si la
    // entrada se envía.
    let decision = host.find("let decision_envio: Option<u32>").expect("decisión de envío");
    let entrada = host.find("trajectory_auditor.record_entry(").expect("entrada");
    assert!(decision < entrada, "la traza nace tras decidir el envío");
    assert_eq!(host.matches("trajectory_auditor.record_entry(").count(), 1);
    let tras = host.get(entrada..entrada + 200).unwrap_or_default();
    assert!(tras.contains("reservation.slot") && tras.contains("reservation.generation"), "{tras}");
    let s = contextos(&host, "[TRAYECTORIA] {} ranura {} divergente", 3);
    assert_eq!(s.len(), 1);
    assert!(s[0].contains("divergencia_nueva(coin_id, ranura, &status)"), "{}", s[0]);
    assert!(host.contains("Err(_) if event.repetido => {}"));
    assert!(host.contains("tracker.error_de_escaneo_nuevo(&error)"));
}

/// Esperado: el log del motor rota y se comprime cuando el entorno lo pide.
#[test]
fn cl144_el_host_usa_el_log_rotativo_si_hay_ruta() {
    let host = codigo(HOST);
    assert!(host.contains("Rotacion::desde_entorno()"));
    assert!(host.contains("init_telemetry_logger_rotativo(16_384, rot)"));
}

/// RET-51b (CL-160, censo C-F7b, F155). Esperado: el host conecta
/// `telemetry!`/`telemetry_err!` a la cola del log antes de la primera línea
/// de esa familia en `main` y ya no tiene la cola binaria ni el hilo que la
/// pasaba a la del log. RED en main be67d521: el host no conecta nada y
/// conserva la cola binaria.
#[test]
fn ret51b_el_host_conecta_telemetry_a_la_cola_del_log() {
    let host = codigo(HOST);
    let main = host.find("async fn main()");
    assert!(main.is_some(), "main del host");
    let main = main.unwrap_or(0);
    let conecta =
        host.find("telemetry_engine::conectar(telemetry_server::macros::linea_de_telemetria)");
    assert!(conecta.is_some(), "el host conecta telemetry! a la cola del log");
    let conecta = conecta.unwrap_or(usize::MAX);
    let primera = ["telemetry!(", "telemetry_err!("]
        .iter()
        .filter_map(|m| host[main..].find(m))
        .min()
        .map(|i| main + i);
    assert!(primera.is_some(), "main emite telemetry!");
    let primera = primera.unwrap_or(0);
    assert!(main < conecta && conecta < primera, "conecta en {conecta}, primera línea en {primera}");
    for retirado in ["telemetry_engine::pop_telemetry", "telemetry_engine::init_telemetry("] {
        assert!(!host.contains(retirado), "el host conserva {retirado}");
    }
}

/// Esperado: el examen del demonio y los universos sombra construyen núcleos
/// auxiliares. Antes cada uno de los 25 núcleos por ronda del examen repetía
/// la carga de DarkAlpha y sus 100 primeras OPEN/CLOSE TRACE, que el monitor
/// contaba como órdenes de la cuenta.
#[test]
fn cl142_examen_y_sombras_son_nucleos_auxiliares() {
    for (nombre, fuente) in [
        ("online_daemon.rs", DEMONIO),
        ("random_forest.rs", SOMBRAS),
        ("darwin.rs", DARWIN),
    ] {
        let c = codigo(fuente);
        assert!(c.contains("GodEngineCore::new_auxiliar("), "{nombre} no usa new_auxiliar");
        assert!(!c.contains("GodEngineCore::new(arena"), "{nombre} construye un núcleo con trazas");
    }
    // CL-142b: el examen y las sombras construyen el arena desde su genoma,
    // sin leer ni anunciar el genoma activo del disco en cada núcleo.
    for (nombre, fuente) in [("online_daemon.rs", DEMONIO), ("random_forest.rs", SOMBRAS)] {
        let c = codigo(fuente);
        let produccion = c.split("#[cfg(test)]").next().unwrap_or("");
        assert!(produccion.contains("GlobalArena::from_genome_in_own_stack("), "{nombre}");
        assert!(!produccion.contains("GlobalArena::build_in_own_stack("), "{nombre}");
    }
}

/// Esperado: los avisos del núcleo que se contaban en eventos tienen tope por
/// tiempo (`avisos`), y VIABILIDAD calla en los núcleos auxiliares. Antes
/// LATENCY PANIC salía cada 1 000 ticks, ML-GATE uno de cada 50 vetos, el
/// latido en 26 de cada 5 000 ticks y ML-DIAG cada 100 ticks.
#[test]
fn cl142b_los_avisos_del_nucleo_tienen_tope_por_tiempo() {
    let nucleo = codigo(NUCLEO);
    for (marca, tope) in [
        ("[LATENCY PANIC] Latencia", "self.avisos.latencia()"),
        ("[ML-GATE]", "self.avisos.veto_ml()"),
        ("[ML-HEARTBEAT]", "self.avisos.latido_ml(coin_id)"),
        ("predict=None (modelo", "self.avisos.modelo_degenerado(coin_id)"),
    ] {
        let sitios = contextos(&nucleo, marca, 10);
        assert_eq!(sitios.len(), 1, "{marca}");
        assert!(sitios[0].contains(tope), "{marca} sin tope por tiempo:\n{}", sitios[0]);
    }
    assert!(!nucleo.contains("diag_ml_vetoes % 50"));
    assert!(!nucleo.contains("% 5_000 < 26"));
    let viabilidad = contextos(&nucleo, "[VIABILIDAD] tick=", 6);
    assert_eq!(viabilidad.len(), 1);
    assert!(viabilidad[0].contains("self.trazas_diagnosticas"), "{}", viabilidad[0]);
}

/// Cierres de la serie T-1, sin cambiar coeficientes (ver `cf3_una_sola_tau`).
fn serie_t1(n: usize) -> Vec<f64> {
    let mut seed = 0x5DEECE66Du64;
    let mut p = 60_000.0f64;
    (0..n)
        .map(|i| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((seed >> 33) as f64 / u32::MAX as f64) - 0.5;
            p *= 1.0 + (i as f64 / 180.0).sin() * 0.0030 + u * 0.0040 + 0.00004;
            p
        })
        .collect()
}

fn nucleo(auxiliar: bool) -> (Arc<GlobalArena>, GodEngineCore) {
    nucleo_con(auxiliar, GlobalArena::build_in_own_stack)
}

fn nucleo_con(
    auxiliar: bool,
    arena_de: impl FnOnce(f64) -> Arc<GlobalArena>,
) -> (Arc<GlobalArena>, GodEngineCore) {
    symbols::update_dynamic_universe(vec!["BTCUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec("BTCUSDT")]);
    let datos = god_engine_core::ml_inference::NanoForestData {
        children_left: vec![1, -1, -1],
        children_right: vec![2, -1, -1],
        feature: vec![0, -1, -1],
        threshold: vec![0.0, 0.0, 0.0],
        value: vec![0.0, -3.0, 3.0],
        tree_offsets: vec![0, 3],
        init_score: 0.0,
    };
    let bosque = god_engine_core::ml_inference::NanoForest::from_data(datos)
        .expect("bosque sintético fuera de contrato");
    god_engine_core::ml_inference::NanoForest::store_global("BTCUSDT_MOTOR", bosque);
    let arena = arena_de(1_000.0);
    let mut core = if auxiliar {
        GodEngineCore::new_auxiliar(arena.clone())
    } else {
        GodEngineCore::new(arena.clone())
    };
    core.swing_nn = None;
    (arena, core)
}

/// Huella de lo que decidió el núcleo: eventos con orden, capital y posiciones.
type Huella = (Vec<usize>, u64, Vec<(bool, u64, u64)>);

fn recorrido(auxiliar: bool) -> Huella {
    let (arena, core) = nucleo(auxiliar);
    assert_eq!(core.trazas_diagnosticas, !auxiliar);
    recorrer(arena, core)
}

fn recorrer(arena: Arc<GlobalArena>, mut core: GodEngineCore) -> Huella {
    let cierres = serie_t1(2 * 24 * 60);
    let (mut t, mut previo, mut ordenes) = (1_700_000_000_000u64, cierres[0], Vec::new());
    let mut i = 0usize;
    for &p in &cierres {
        for j in 0..12u64 {
            let w = (j + 1) as f64 / 12.0;
            let px = (previo.ln() * (1.0 - w) + p.ln() * w).exp();
            t += 5_000;
            let (entrada, _) = core.process_event(
                0, true, i % 12 == 11, true, px, 1.0, px * 0.9998, px * 1.0002, 5.0, 5.0,
                0.0, 0.0, t, false, &[0.0; 54], px < previo,
            );
            if entrada.is_some() {
                ordenes.push(i);
            }
            i += 1;
        }
        previo = p;
    }
    let posiciones = arena.coins[0]
        .positions
        .slots()
        .iter()
        .map(|s| (s.is_open(), s.entry_price.load(Relaxed).to_bits(), s.sl_price.load(Relaxed).to_bits()))
        .collect();
    (ordenes, arena.unified_capital.load(Relaxed).to_bits(), posiciones)
}

/// Esperado: el núcleo auxiliar decide bit a bit lo mismo que el normal sobre
/// la cinta T-1: callar las trazas no cambia ninguna decisión.
#[test]
fn cl142_el_nucleo_auxiliar_decide_lo_mismo() {
    let _g = ENTORNO.lock().unwrap_or_else(|p| p.into_inner());
    let normal = recorrido(false);
    let auxiliar = recorrido(true);
    println!("observado: {} órdenes en la cinta", normal.0.len());
    assert!(normal.0.len() >= 2, "la cinta tiene que abrir órdenes: {:?}", normal.0);
    assert_eq!(normal, auxiliar);
}

/// Esperado: el arena auxiliar construido desde el genoma decide bit a bit lo
/// mismo que el de antes (genoma del disco y luego el del candidato encima),
/// con un genoma mutado que no es el de partida.
#[test]
fn cl142b_el_arena_desde_el_genoma_decide_lo_mismo() {
    use quantum_arena::genome::SuperGenotype;
    let _g = ENTORNO.lock().unwrap_or_else(|p| p.into_inner());
    let genoma = SuperGenotype::new_baseline(0.0002, 0.0005).mutate_cmaes_seeded(0.15, 20261018);
    let g1 = genoma.clone();
    let (arena, core) = nucleo_con(true, move |capital| {
        let arena = GlobalArena::build_in_own_stack(capital);
        g1.apply_to_arena(&arena);
        arena
    });
    let antes = recorrer(arena, core);
    let (arena, core) = nucleo_con(true, move |capital| {
        let arena = GlobalArena::from_genome_in_own_stack(capital, &genoma);
        genoma.apply_to_arena(&arena);
        arena
    });
    let ahora = recorrer(arena, core);
    println!("observado: {} órdenes con el genoma mutado", antes.0.len());
    assert!(antes.0.len() >= 2, "la cinta tiene que abrir órdenes: {:?}", antes.0);
    assert_eq!(antes, ahora);
}
