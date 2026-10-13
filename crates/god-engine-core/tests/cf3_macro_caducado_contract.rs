//! C-F3.5 (Línea C), hallazgo F3.5-02: el bosque opina con el macro de
//! semilla, fuera de su soporte de entrenamiento; la caducidad del macro no
//! se consulta en la inferencia.
//!
//! Teoría que se prueba:
//! - El entrenador (`train_forest`) sólo emite muestras con macro vigente
//!   (`macro_vigente`): el bosque nunca vio una fila sin macro observado.
//! - En servicio, `macro_ml_features` convierte los niveles de
//!   `omni_features` a las unidades del bosque (dims 44..48: VIX, SP500, DXY,
//!   NASDAQ) y sólo neutraliza x ≤ 0 o no finito. Las semillas de
//!   `OmniState` (DXY 104, SP500 5 100, NASDAQ 18 000, VIX 15) se sirven como
//!   dato: SP500 vale (5 100 − 5 000)/500 = 0,2 en unidades del bosque.
//! - El host publica `macro_staleness_ms` (u64::MAX si nunca hubo un éxito);
//!   su único lector en el núcleo es el consejo, no la inferencia del bosque.
//!
//! Solución conocida: un bosque de un solo split en la dimensión 45 (SP500)
//! con umbral 4,5 y hojas −2/+2, sin sesgo. Con SP500 = 5 100 (unidades 0,2)
//! cae a la izquierda: σ(−2) = 0,1192; con SP500 = 9 000 (unidades 8,0), a la
//! derecha: σ(+2) = 0,8808. Sin spot, `ml_prob_motor` es la salida del bosque
//! (único participante del ensamble).
//!
//! Propiedad: sin macro vigente el bosque no opina con el nivel macro
//! servido. Control: con macro fresco distingue los dos niveles
//! (|Δp| = 0,762 > 0,5). Con `macro_staleness_ms = u64::MAX` las dos salidas
//! deben coincidir (|Δp| < 1e-9: misma ruta, ningún redondeo distinto entre
//! las dos corridas si el macro no entra). Hoy siguen siendo 0,119 y 0,881.
//!
//! Notas de los escépticos (no cambian el assert): el defecto de fondo es
//! servir semillas inventadas como dato; el remedio puede ser frescura por
//! serie o semillas NaN, siempre que con «nunca observado» el bosque no
//! opine. Si el bosque deja de opinar, `ml_prob` debe valer la base (o la
//! moneda contar sin modelo): un 0,5 contra la base del bosque fabricaría un
//! lift largo. El backtest publica `macro_staleness_ms = 0` con niveles fijos
//! (pedido a Línea E).
//!
//! Convención de la fase C-F3 (docs/audit/HALLAZGOS_C-F3_2026-10-11.md): un
//! test que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto;
//! documenta el defecto y se activa con el arreglo. Para correrlo:
//! `cargo test ... -- --include-ignored`.

use god_engine_core::ml_inference::{NanoForest, NanoForestData};
use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};

const SIM: &str = "MCRAUSDT";

fn bosque_sp500() -> NanoForest {
    NanoForest::from_data(NanoForestData {
        children_left: vec![1, -1, -1],
        children_right: vec![2, -1, -1],
        feature: vec![45, -1, -1],
        threshold: vec![4.5, 0.0, 0.0],
        value: vec![0.0, -2.0, 2.0],
        tree_offsets: vec![0, 3],
        init_score: 0.0,
    })
    .expect("bosque sintético fuera de contrato")
}

/// Un núcleo nuevo por lectura: ensamble, espectro y registro sin estado
/// previo. Devuelve `ml_prob_motor` tras un tick con el SP500 dado.
fn ml_prob_con(sp500: f64, staleness_ms: f64) -> f64 {
    let arena = GlobalArena::build_in_own_stack(100.0);
    arena.registry.set("macro_staleness_ms", staleness_ms);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
    let mut omni = [0.0f64; 54];
    omni[21] = 104.0; // DXY (semilla)
    omni[22] = sp500;
    omni[23] = 18_000.0; // NASDAQ (semilla)
    omni[24] = 15.0; // VIX (semilla)
    let _ = core.process_tick_dual(0, 100.0, 100.01, 5.0, 5.0, 10_000, &omni, true, false);
    arena
        .registry
        .get_for_coin_strict_or(0, "ml_prob_motor", f64::NAN)
}

#[test]
#[ignore = "OPEN F3.5-02 (Línea C, núcleo; pedido a Línea E para el backtest): la inferencia del bosque no consulta la vigencia del macro (macro_staleness_ms sólo lo lee el consejo) y opina con las semillas de OmniState fuera de su soporte de entrenamiento"]
fn cf3_bosque_no_opina_con_macro_caducado() {
    symbols::update_dynamic_universe(vec![SIM.into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(SIM)]);
    NanoForest::store_global(&format!("{SIM}_MOTOR"), bosque_sp500());

    let fresco_semilla = ml_prob_con(5_100.0, 0.0);
    let fresco_alto = ml_prob_con(9_000.0, 0.0);
    println!(
        "observado: macro fresco: p(SP500 5100) = {fresco_semilla:.4}, p(SP500 9000) = {fresco_alto:.4}"
    );
    assert!(
        (fresco_alto - fresco_semilla).abs() > 0.5,
        "control: con macro fresco el bosque distingue niveles ({fresco_semilla} vs {fresco_alto})"
    );

    let caducado_semilla = ml_prob_con(5_100.0, u64::MAX as f64);
    let caducado_alto = ml_prob_con(9_000.0, u64::MAX as f64);
    println!(
        "observado: macro caducado: p(SP500 5100) = {caducado_semilla:.4}, p(SP500 9000) = {caducado_alto:.4}"
    );
    assert!(
        (caducado_semilla - caducado_alto).abs() < 1e-9,
        "con macro caducado el bosque opina con el nivel servido: {caducado_semilla} vs {caducado_alto}"
    );
}
