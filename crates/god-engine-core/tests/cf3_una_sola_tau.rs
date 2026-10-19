//! C-F3 · F3.2 (Línea C): una sola τ de punta a punta, del juicio de las
//! puertas a la ranura que se abre.
//!
//! Teoría que se prueba:
//! - QO-586 (`banda_paga_friccion` dentro de `puertas_del_continuo`) decide
//!   si una banda paga la fricción A SU τ: σ(τ) crece con τ y el suelo de
//!   fricción es monótono, así que el veredicto sólo es válido para la τ con
//!   la que se evaluó. La orden que se abre debe vivir a esa misma τ
//!   (F3.2-01). Hasta CL-74, la modulación de Hurst (×[0,5; 2]) y la media
//!   geométrica con el centroide reescribían `expected_duration_ms` tras las
//!   puertas; desde CL-74 se aplican antes (`tau_de_la_orden`).
//! - La rama 11 decide su dirección con el veredicto espectral a τ_k; el
//!   contrato #624 dice que «la posición vive a la escala que habló», así
//!   que su intención debe viajar a min(τ_k, 12 h) (F3.2-05). Hasta CL-85
//!   viajaba a la micro τ (`fast_duration_ms`), movida además por Hurst y
//!   el centroide. CL-85 aplica la decisión del dueño (tarjeta del
//!   2026-10-12, «escala que habló»).
//!
//! Los eventos de libro puro no abren (`allow_entries = !is_depth ||
//! is_trade`), así que los eventos que deben poder abrir son de trade.
//!
//! Convención (fase C-F3): un test que hoy falla por la causa de un hallazgo
//! lleva `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto.

use god_engine_core::GodEngineCore;
use quantum_arena::temporal_spectrum::SPECTRUM_SCALES_MS;
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use signal_engine::SignalType;
use std::sync::{atomic::Ordering::Relaxed, Arc, Mutex};

const SIM: &str = "BTCUSDT";

/// El universo, el registro de símbolos y el almacén de bosques son globales.
static ENTORNO: Mutex<()> = Mutex::new(());

/// Un evento: (mid, OBI del libro que lo acompaña). Spread ±2 pb.
type Evento = (f64, f64);

/// Cierres de la serie T-1 (backtest-engine/tests/support/t1_measurement.rs),
/// sin cambiar coeficientes: mismo LCG 0x5DEECE66D, ciclo sin(i/180)·0,003,
/// ruido 0,004 (no centrado: GO) y deriva 0,00004.
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

/// Cinta T-1: 4 días de velas de 1 min, 12 eventos por minuto (cada 5 s)
/// interpolados en log entre cierres, libro equilibrado (5, 5).
fn cinta_t1() -> Vec<Evento> {
    let cierres = serie_t1(4 * 24 * 60);
    let mut previo = cierres[0];
    let mut out = Vec::with_capacity(cierres.len() * 12);
    for &p in &cierres {
        for j in 0..12u64 {
            let w = (j + 1) as f64 / 12.0;
            out.push(((previo.ln() * (1.0 - w) + p.ln() * w).exp(), 0.0));
        }
        previo = p;
    }
    out
}

/// Martingala con impulsos de libro comprador: 2 días a 1 evento cada 5 s,
/// paso de ln p gaussiano de 3 pb. Con probabilidad 0,2 el libro llega con
/// un impulso OBI = 0,85 (el lado del veredicto inyectado: la puerta de la
/// rama 11 exige |OBI| y |compuesto| en su percentil 80/85 medido y
/// alineados con el tensor); si no, OBI ~ N(0; 0,05) recortado a ±0,15. El
/// precio no tiene deriva, para que la estructura lenta no esté en contra ni
/// estirada.
fn cinta_impulsos_de_libro() -> Vec<Evento> {
    let mut seed = 0xCF3_2005u64;
    let mut u = || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((seed >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let n = 2 * 24 * 720usize;
    let mut out = Vec::with_capacity(n);
    let mut ln_p = 60_000f64.ln();
    for _ in 0..n {
        let (u1, u2) = (u(), u());
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        ln_p += 3e-4 * z;
        let obi = if u() < 0.2 {
            0.85
        } else {
            let (v1, v2) = (u(), u());
            let w = (-2.0 * v1.ln()).sqrt() * (2.0 * std::f64::consts::PI * v2).cos();
            (0.05 * w).clamp(-0.15, 0.15)
        };
        out.push((ln_p.exp(), obi));
    }
    out
}

/// Núcleo de un solo símbolo. Con `con_bosque`, el bosque sintético
/// direccional del oráculo T-1 (base 0,5, ±3 según el signo de
/// price_change) hace que la puerta del roster y el gate por lift cooperen.
fn nucleo(con_bosque: bool) -> (Arc<GlobalArena>, GodEngineCore) {
    symbols::update_dynamic_universe(vec![SIM.into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(SIM)]);
    if con_bosque {
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
    } else {
        // CL-95 (cl85-2): sin bosque también en el almacén global. Lo dejaba
        // ahí la prueba anterior que pidió bosque (el almacén es del
        // proceso), y el resultado de esta prueba dependía del orden.
        god_engine_core::ml_inference::GLOBAL_FORESTS.rcu(|m| {
            let mut m = (**m).clone();
            m.remove("BTCUSDT_MOTOR");
            m
        });
    }
    let arena = GlobalArena::build_in_own_stack(1_000.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    if !con_bosque {
    }
    (arena, core)
}

/// Pasa los eventos por `process_event` como trades, cada 5 s; cada 12.º
/// cierra la vela de 1 min. Con `con_libro` cada evento trae además el libro
/// (is_depth: el núcleo compone el consenso espectral); sin él, sólo trade
/// con las cantidades del libro (el consenso espectral no se recompone).
/// `antes(core)` se llama antes de cada evento y `despues(core, hubo_orden)`
/// después.
fn recorrer(
    core: &mut GodEngineCore,
    eventos: &[Evento],
    con_libro: bool,
    mut antes: impl FnMut(&mut GodEngineCore),
    mut despues: impl FnMut(&mut GodEngineCore, bool),
) {
    let mut t = 1_700_000_000_000u64;
    let mut previo = eventos[0].0;
    for (i, &(px, obi)) in eventos.iter().enumerate() {
        t += 5_000;
        let (bid, ask) = (px * 0.9998, px * 1.0002);
        let (bq, aq) = (5.0 * (1.0 + obi), 5.0 * (1.0 - obi));
        antes(core);
        let (entrada, _) = core.process_event(
            0,
            true,
            i % 12 == 11,
            con_libro,
            px,
            1.0,
            bid,
            ask,
            bq,
            aq,
            obi,
            0.0,
            t,
            false,
            &[0.0; 54],
            px < previo,
        );
        previo = px;
        despues(core, entrada.is_some());
    }
}

/// Verdad: cada ranura que se abre lleva `entry_tau_ms` igual a la τ que
/// juzgaron QO-586 y el co-despacho: `last_fast_intent[0]` o
/// `last_slow_intent[0].expected_duration_ms`, guardados justo después de
/// `puertas_del_continuo` (lib.rs, D-743). Error relativo ≤ 1e-6: las dos
/// son enteras en ms y τ ≥ 30 s, así que basta la igualdad salvo el
/// redondeo de la conversión a f64. Precondición: la cinta abre al menos
/// una orden.
///
/// Fixture del borrador: la cinta T-1 con trade + libro en cada evento y el
/// bosque direccional. Abría una sola orden (la cinta T-1 cae de 60 000 a
/// ≈ 2 000 en 4 días: su ruido no está centrado), de la rama 15, a una τ
/// que las puertas no juzgaron. Con CL-74 abre dos, las dos a la τ juzgada.
#[test]
fn cf3_la_tau_que_juzga_qo586_es_la_tau_de_la_orden() {
    let _g = ENTORNO.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, mut core) = nucleo(true);
    let mut ordenes = 0usize;
    let mut fallos: Vec<(f64, Vec<f64>)> = Vec::new();
    let mut abiertas: Vec<bool> = arena.coins[0]
        .positions
        .slots()
        .iter()
        .map(|s| s.is_open())
        .collect();
    recorrer(
        &mut core,
        &cinta_t1(),
        true,
        |_| {},
        |core, hubo_orden| {
            let ahora: Vec<bool> = arena.coins[0]
                .positions
                .slots()
                .iter()
                .map(|s| s.is_open())
                .collect();
            if hubo_orden {
                let juzgadas: Vec<f64> = [&core.last_fast_intent[0], &core.last_slow_intent[0]]
                    .iter()
                    .filter(|i| i.signal != SignalType::Flat && i.expected_duration_ms > 0)
                    .map(|i| i.expected_duration_ms as f64)
                    .collect();
                for (k, s) in arena.coins[0].positions.slots().iter().enumerate() {
                    if ahora[k] && !abiertas[k] {
                        ordenes += 1;
                        let tau_orden = s.entry_tau_ms.load(Relaxed) as f64;
                        if !juzgadas
                            .iter()
                            .any(|&tj| (tau_orden - tj).abs() <= 1e-6 * tj)
                        {
                            fallos.push((tau_orden, juzgadas.clone()));
                        }
                    }
                }
            }
            abiertas = ahora;
        },
    );
    println!(
        "observado: {ordenes} órdenes abiertas, {} con τ distinta de la juzgada; primeras: {:?}",
        fallos.len(),
        &fallos[..fallos.len().min(5)]
    );
    assert!(
        ordenes > 0,
        "precondición: la cinta T-1 debe abrir al menos una orden"
    );
    assert!(
        fallos.is_empty(),
        "{}/{ordenes} órdenes viven a una τ que las puertas no juzgaron; primera: τ_orden = {} ms, juzgadas {:?}",
        fallos.len(),
        fallos[0].0,
        fallos[0].1
    );
}

/// Verdad (contrato #624): una intención de la rama 11 (`volume_flow_rate`
/// 11) tomada con el veredicto espectral activo (dominante ≠ 0 a τ_k ≥ 30 s)
/// viaja a min(τ_k, 12 h), con error relativo ≤ 1e-6 (enteros en ms).
///
/// Fixture: el veredicto espectral se fija en el registro, que es la entrada
/// del orquestador (dominante 0,9 a τ_k = SPECTRUM_SCALES_MS[19] = 274,9 s,
/// media de banda 0,9 ⇒ coherencia 1 y tensor Long 0,9). Los eventos son
/// trades sin libro puro, así que el núcleo no recompone el consenso ni lo
/// caduca (su TTL cuenta desde la última composición, que no existe). Así
/// la prueba no depende de que una cinta produzca por azar un dominante en
/// banda: con cintas T-1 o martingala a 5 s el dominante cae casi siempre
/// en τ = 17,2 s (< 30 s) y la rama 11 decide con el respaldo escalar. El
/// libro trae impulsos compradores (OBI 0,85) para que pase su puerta.
/// Precondición: al menos una intención de la rama 11.
/// Antes de CL-85: 128/128 intenciones de la rama 11 viajaban a otra τ (las
/// primeras, a 54 919, 91 512 y 203 931 ms). CL-94: sustituye también al
/// contrato de fuente de F3.3-03.
#[test]
fn cf3_la_rama_11_vive_a_la_escala_que_hablo() {
    let _g = ENTORNO.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, mut core) = nucleo(false);
    let tau_k = SPECTRUM_SCALES_MS[19];
    let esperado = tau_k.min(43_200_000.0).round();
    let (mut vistos, mut fallos) = (0usize, Vec::<f64>::new());
    let reg = arena.clone();
    recorrer(
        &mut core,
        &cinta_impulsos_de_libro(),
        false,
        |_| {
            reg.registry
                .set_for_coin(0, "consenso_espectral_dominante", 0.9);
            reg.registry
                .set_for_coin(0, "consenso_espectral_tau", tau_k);
            reg.registry
                .set_for_coin(0, "consenso_espectral_media", 0.9);
        },
        |core, _| {
            let fi = core.last_fast_intent[0];
            if fi.signal == SignalType::Flat || fi.volume_flow_rate != 11.0 {
                return;
            }
            vistos += 1;
            let actual = fi.expected_duration_ms as f64;
            if (actual - esperado).abs() > 1e-6 * esperado {
                fallos.push(actual);
            }
        },
    );
    let d = &core.diag_dir;
    println!(
        "diag rama 11: puerta alcanzada [L,S] {:?}, pasada {:?}; observado: {vistos} intenciones de la rama 11 con τ_k = {tau_k} ms, {} viajan a otra τ; primeras: {:?}",
        d.gate_reached,
        d.gate_passed,
        fallos.len(),
        &fallos[..fallos.len().min(5)]
    );
    assert!(
        vistos > 0,
        "precondición: alguna intención de la rama 11 con el veredicto espectral activo"
    );
    assert!(
        fallos.is_empty(),
        "la rama 11 decide a τ_k = {tau_k} ms y viaja a {} ms ({}/{vistos})",
        fallos[0],
        fallos.len()
    );
}

/// CL-85: `GodEngineCore::escala_que_hablo` usa la condición del orquestador
/// para consumir el veredicto espectral (#652/H2) y su misma τ (#624). Con
/// veredicto, `expected_lifetime_ms` del orquestador (que trunca a ms) y la
/// τ de la rama 11 (que redondea) difieren a lo sumo en 1 ms. Sin veredicto
/// (dominante nulo o no finito, τ < 30 s o no finita) no hay escala.
#[test]
fn cl85_la_escala_que_hablo_es_la_del_orquestador() {
    let _g = ENTORNO.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, core) = nucleo(false);
    let casos: [(f64, f64, Option<u64>); 8] = [
        (0.9, SPECTRUM_SCALES_MS[19], Some(274_878)),
        (-0.4, 3_600_000.0, Some(3_600_000)),
        (0.9, 86_400_000.0, Some(43_200_000)),
        (0.9, 30_000.0, Some(30_000)),
        (0.9, 29_999.0, None),
        (0.0, 3_600_000.0, None),
        (f64::NAN, 3_600_000.0, None),
        (0.9, f64::NAN, None),
    ];
    for (v_dom, tau, esperado) in casos {
        arena.registry.set_for_coin(0, "consenso_espectral_dominante", v_dom);
        arena.registry.set_for_coin(0, "consenso_espectral_tau", tau);
        arena.registry.set_for_coin(0, "consenso_espectral_media", v_dom);
        let escala = core.escala_que_hablo(0);
        assert_eq!(escala, esperado, "v_dom = {v_dom}, τ = {tau}");
        if let Some(tau_k) = escala {
            let vida = core
                .tensor_orchestrator
                .evaluate_continuous_consensus_for_coin(0, SIM)
                .expected_lifetime_ms;
            assert!(
                vida.abs_diff(tau_k) <= 1,
                "orquestador {vida} ms frente a la escala que habló {tau_k} ms"
            );
        }
    }
}

/// CL-95 (cl85-2, revisión del #56). Verdad: `nucleo(false)` es un núcleo
/// sin bosque, también en el almacén global de bosques, aunque una prueba
/// anterior del mismo proceso haya pedido uno. Si no, el resultado de las
/// pruebas sin bosque depende del orden en que corren.
#[test]
fn cl95_nucleo_sin_bosque_no_hereda_el_bosque_global() {
    let _g = ENTORNO.lock().unwrap_or_else(|p| p.into_inner());
    let _ = nucleo(true);
    assert!(
        god_engine_core::ml_inference::NanoForest::get_global("BTCUSDT_MOTOR").is_some(),
        "precondición: nucleo(true) publica el bosque"
    );
    let _ = nucleo(false);
    assert!(
        god_engine_core::ml_inference::NanoForest::get_global("BTCUSDT_MOTOR").is_none(),
        "nucleo(false) hereda el bosque global de la prueba anterior"
    );
}
