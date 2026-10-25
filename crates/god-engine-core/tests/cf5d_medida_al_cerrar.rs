//! C-F5d · PR70-INT-04 de la revisión de la Línea G (CL-135): lo medido de
//! una orden entra en las medias de la cuenta al CERRARLA, y sólo si esa
//! operación existió.
//!
//! Teoría. Los dos frenos leen medias móviles de las órdenes (CL-131): el
//! riesgo al stop (`riesgo_por_operacion`, en fracción y en USD) y la
//! geometría en stops `w`, `l`, de la que sale el umbral K del freno duro y
//! la escala del proceso e. Son la muestra de las operaciones de la cuenta.
//! CL-131 las actualizaba cuando el núcleo abría la orden, pero esa apertura
//! es LOCAL y ocurre antes de los vetos del host (envolvente, breaker de
//! fees, guardia de margen, filtro del símbolo, rechazo del exchange). Una
//! intención que el host revierte vuelve a abrirse en el evento siguiente,
//! sin enfriamiento, y se medía cada vez. Desde CL-135 la medida viaja con la
//! ranura, atada a su generación, y entra al cerrar la posición tras la
//! puerta de aprendizaje: en el contexto del exchange, sólo si la entrada se
//! confirmó.
//!
//! Fixture: la cinta T-1 de `cf3_una_sola_tau` (velas de 1 min, un trade con
//! libro cada 5 s) con el bosque direccional sintético, que abre órdenes por
//! el camino real del núcleo.

use god_engine_core::{outcome_context::OutcomeContext, GodEngineCore};
use quantum_arena::position::PositionHorizon;
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use std::sync::{atomic::Ordering::Relaxed, Arc, Mutex};

const SIM: &str = "BTCUSDT";

/// El universo, el registro de símbolos y el almacén de bosques son globales.
static ENTORNO: Mutex<()> = Mutex::new(());

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

/// 4 días de velas de 1 min, 12 eventos por minuto interpolados en log.
fn cinta_t1() -> Vec<f64> {
    let cierres = serie_t1(4 * 24 * 60);
    let mut previo = cierres[0];
    let mut out = Vec::with_capacity(cierres.len() * 12);
    for &p in &cierres {
        for j in 0..12u64 {
            let w = (j + 1) as f64 / 12.0;
            out.push((previo.ln() * (1.0 - w) + p.ln() * w).exp());
        }
        previo = p;
    }
    out
}

fn nucleo(contexto: OutcomeContext) -> (Arc<GlobalArena>, GodEngineCore) {
    symbols::update_dynamic_universe(vec![SIM.into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(SIM)]);
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
    let arena = GlobalArena::build_in_own_stack(1_000.0);
    let mut core = GodEngineCore::new_with_outcome_context(arena.clone(), contexto);
    core.swing_nn = None;
    (arena, core)
}

/// Un evento de trade con libro (±2 pb, equilibrado); cada 12.º cierra la
/// vela de 1 min. Devuelve si el núcleo emitió una orden.
fn evento(core: &mut GodEngineCore, i: usize, px: f64, previo: f64, t: u64) -> bool {
    let (entrada, _) = core.process_event(
        0,
        true,
        i % 12 == 11,
        true,
        px,
        1.0,
        px * 0.9998,
        px * 1.0002,
        5.0,
        5.0,
        0.0,
        0.0,
        t,
        false,
        &[0.0; 54],
        px < previo,
    );
    entrada.is_some()
}

/// Las cuatro medias de las órdenes: riesgo al stop (fracción y USD) y
/// geometría en stops (w, l).
fn medido(arena: &GlobalArena) -> [f64; 4] {
    [
        arena.riesgo_por_operacion.load(Relaxed),
        arena.riesgo_por_operacion_usd.load(Relaxed),
        arena.ganancia_en_stops.load(Relaxed),
        arena.perdida_en_stops.load(Relaxed),
    ]
}

fn abiertas(arena: &GlobalArena) -> Vec<bool> {
    arena.coins[0].positions.slots().iter().map(|s| s.is_open()).collect()
}

/// Esperado: en el contexto del exchange, un host que revierte cada apertura
/// local del núcleo (como `rollback_positions` en god_engine.rs) deja las
/// medias en cero por muchas veces que el núcleo reabra. Antes (CL-131) cada
/// apertura revertida entraba en la EWMA.
#[test]
fn cl135_una_apertura_local_que_el_host_revierte_no_se_mide() {
    let _g = ENTORNO.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, mut core) = nucleo(OutcomeContext::ExchangeLocalEstimate);
    let cinta = cinta_t1();
    let (mut t, mut previo, mut revertidas) = (1_700_000_000_000u64, cinta[0], 0usize);
    for (i, &px) in cinta.iter().enumerate() {
        t += 5_000;
        let antes = abiertas(&arena);
        evento(&mut core, i, px, previo, t);
        previo = px;
        for (k, s) in arena.coins[0].positions.slots().iter().enumerate() {
            if s.is_open() && !antes[k] {
                let generacion = s.generation.load(Relaxed);
                s.cancel_unconfirmed_generation(generacion)
                    .expect("una reserva sin confirmar se revierte");
                revertidas += 1;
            }
        }
    }
    println!("observado: {revertidas} aperturas locales revertidas; medido {:?}", medido(&arena));
    assert!(revertidas >= 2, "la cinta tiene que abrir y reabrir: {revertidas}");
    assert_eq!(
        medido(&arena),
        [0.0; 4],
        "{revertidas} aperturas revertidas por el host no son operaciones de la cuenta"
    );
}

/// Esperado: en simulación aislada, la orden que el núcleo abre deja su
/// medida en la ranura, atada a su generación, y las medias no se mueven al
/// abrir. Al cerrarla por el camino de cierre del núcleo entran una vez.
/// Una ranura revertida y reocupada por otra posición (adoptada, sin orden
/// del riesgo) no hereda la medida de la anterior.
#[test]
fn cl135_lo_medido_entra_al_cerrar_y_no_lo_hereda_otra_posicion() {
    let _g = ENTORNO.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, mut core) = nucleo(OutcomeContext::IsolatedSimulation);
    let cinta = cinta_t1();
    let (mut t, mut previo) = (1_700_000_000_000u64, cinta[0]);
    let mut abierta = None;
    for (i, &px) in cinta.iter().enumerate() {
        t += 5_000;
        let antes = abiertas(&arena);
        evento(&mut core, i, px, previo, t);
        previo = px;
        abierta = (0..antes.len()).find(|&k| !antes[k] && arena.coins[0].positions.slots()[k].is_open());
        if abierta.is_some() {
            break;
        }
    }
    let k = abierta.expect("la cinta T-1 abre al menos una orden por el camino real");
    let ranura = arena.coins[0].positions.slots()[k];
    let generacion = ranura.generation.load(Relaxed);
    let medida = core.medida_de_entrada[0][k].expect("la apertura guarda la medida de su orden");
    assert_eq!(medida.generation, generacion, "atada a la generación de la ranura que abrió");
    assert_eq!(medida.is_long, ranura.is_long.load(Relaxed));
    assert_eq!(medido(&arena), [0.0; 4], "abrir no mueve las medias de la cuenta");

    // Cierre por el camino del núcleo: el precio cruza el stop de la ranura.
    let (entrada, sl) = (ranura.entry_price.load(Relaxed), ranura.sl_price.load(Relaxed));
    assert!(sl > 0.0 && entrada > 0.0, "la orden lleva su stop: entrada {entrada}, SL {sl}");
    let apalancamiento = ranura.quantity.load(Relaxed) * entrada / ranura.margin_used.load(Relaxed);
    let salida = if medida.is_long { sl * 0.999 } else { sl * 1.001 };
    t += 5_000;
    let (_, cierre, _) =
        core.process_tick_dual(0, salida, salida * 1.0001, 5.0, 5.0, t, &[0.0; 54], true, false);
    assert!(cierre.is_some() && !ranura.is_open(), "el stop se toca en {salida}");
    let (_, pnl_1, qty_1) = cierre.unwrap_or((false, f64::NAN, f64::NAN));
    let retorno_1 = pnl_1 / (qty_1 * entrada).max(1.0);
    let tras_cierre = medido(&arena);
    println!("observado: medido tras el cierre {tras_cierre:?}");
    assert!(
        tras_cierre[0] > 0.0 && tras_cierre[1] > 0.0,
        "la operación cerrada entra en el riesgo medido: {tras_cierre:?}"
    );
    assert!(core.medida_de_entrada[0][k].is_none(), "el cierre consume la medida");
    // CL-164: la sombra del Kelly legado frente a LH-04 aprende del mismo
    // cierre, una vez, con un log finito (un stop tocado no es una ruina).
    // En este arena la orden es la primera (n = 0 < lh04_muestras_min): f_lh04
    // y el mínimo valen 0 y su log es 0 para cualquier R, así que la unidad la
    // fija la suma de R de CL-166.
    let sombra = |nombre: &str| arena.registry.get_for_coin_strict(0, nombre);
    assert_eq!(sombra("sombra_kelly_n"), Some(1.0), "el cierre entra en la sombra");
    for regla in ["lh04", "legado", "min"] {
        let logw = sombra(&format!("sombra_kelly_logw_{regla}"));
        assert!(
            logw.is_some_and(f64::is_finite),
            "sombra {regla}: {logw:?} (ruina {:?})",
            sombra(&format!("sombra_kelly_ruina_{regla}"))
        );
    }
    // CL-166: el núcleo pasa el retorno neto sobre el nocional de entrada y el
    // riesgo lo divide por SL + c: un cierre en el stop (un 0,1 % más allá,
    // más la fricción de la salida) es R ≈ −1, no R·100 (en porcentaje).
    // RV8-T: un R sobre el margen sería R·L, y en este fixture la orden sale
    // con L = 1 (se observa 1,0), así que margen y nocional coinciden y este
    // cierre no distingue ese error. Lo distingue el tercero, con L = 10.
    let r = sombra("sombra_kelly_suma_r").unwrap_or(f64::NAN);
    println!("observado: R del cierre en el stop {r}, apalancamiento {apalancamiento}");
    assert!((-2.0..=-0.9).contains(&r), "un cierre en el stop es R ≈ −1 en unidades de SL + c: {r}");

    // La ranura revertida no deja su medida a la siguiente ocupante.
    core.medida_de_entrada[0][k] = Some(medida);
    let px = salida;
    assert!(ranura.open_with_horizon(
        medida.is_long,
        px,
        0.01,
        px * 0.01 / 10.0,
        t,
        if medida.is_long { px * 1.02 } else { px * 0.98 },
        if medida.is_long { px * 0.99 } else { px * 1.01 },
        PositionHorizon::Continuous,
    ));
    let cerca = if medida.is_long { px * 0.989 } else { px * 1.011 };
    t += 5_000;
    let (_, cierre, _) =
        core.process_tick_dual(0, cerca, cerca * 1.0001, 5.0, 5.0, t, &[0.0; 54], true, false);
    assert!(cierre.is_some() && !ranura.is_open(), "la adoptada también se cierra en su stop");
    assert_eq!(
        medido(&arena),
        tras_cierre,
        "una posición sin orden del riesgo no hereda la medida de la generación anterior"
    );
    assert_eq!(sombra("sombra_kelly_n"), Some(1.0), "ni la sombra");

    // RV9-T: la unidad del retorno que el núcleo pasa a la sombra, con L = 10.
    // La misma medida (su SL + c) se ata a una posición de margen nocional/10
    // y se cierra en su stop por el camino del núcleo. Con R = retorno sobre
    // el nocional / (SL + c), el R de este cierre es el del primero escalado
    // por la razón de sus retornos sobre el nocional; un retorno sobre el
    // margen lo multiplicaría por 10. close_outcome_contract fija aparte la
    // unidad de `realized_ret` (recompensa del PPO, L = 10).
    let px = cerca;
    let (qty, margen) = (0.01, px * 0.01 / 10.0);
    assert!(ranura.open_with_horizon(
        medida.is_long,
        px,
        qty,
        margen,
        t,
        if medida.is_long { px * 1.02 } else { px * 0.98 },
        if medida.is_long { px * 0.99 } else { px * 1.01 },
        PositionHorizon::Continuous,
    ));
    let apalancamiento_3 = qty * px / margen;
    assert!((apalancamiento_3 - 10.0).abs() < 1e-9, "premisa: L = 10, obtenido {apalancamiento_3}");
    let mut con_l = medida;
    con_l.generation = ranura.generation.load(Relaxed);
    core.medida_de_entrada[0][k] = Some(con_l);
    let r_antes = sombra("sombra_kelly_suma_r").unwrap_or(f64::NAN);
    let cerca = if medida.is_long { px * 0.989 } else { px * 1.011 };
    t += 5_000;
    let (_, cierre, _) =
        core.process_tick_dual(0, cerca, cerca * 1.0001, 5.0, 5.0, t, &[0.0; 54], true, false);
    assert!(cierre.is_some() && !ranura.is_open(), "la tercera también se cierra en su stop");
    let (_, pnl_3, qty_3) = cierre.unwrap_or((false, f64::NAN, f64::NAN));
    let retorno_3 = pnl_3 / (qty_3 * px).max(1.0);
    assert_eq!(sombra("sombra_kelly_n"), Some(2.0), "la medida atada a su generación entra en la sombra");
    let r_3 = sombra("sombra_kelly_suma_r").unwrap_or(f64::NAN) - r_antes;
    let esperado = r * retorno_3 / retorno_1;
    println!("observado: con L = 10, R {r_3}, esperado {esperado}");
    assert!(
        (r_3 - esperado).abs() < 1e-9 * esperado.abs(),
        "con L = 10 la sombra recibe el retorno sobre el nocional: R {r_3}, esperado {esperado}"
    );
}
