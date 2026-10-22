//! Contrato F1-010 y F1-011: Ventana Deslizante de Caída en Stops y Presupuesto Vivo de Drawdown Global.
//!
//! F1-010: El gen/config `global_max_drawdown` gobierna activamente el tope de capital en `freno_duro`,
//!         erradicando aristas muertas donde sólo regía el d*=0.6316 y garantizando corte en Darwin.
//! F1-011: La caída en stops se mide sobre la ventana deslizante de HORIZONTE_FRENO_DURO (200) operaciones,
//!         preservando el nivel de falsación alpha=0.05 a lo largo de toda la vida de la cuenta y
//!         erradicando estados absorbentes permanentes.

use quantum_arena::GlobalArena;
use risk_engine::drawdown::{
    caida_en_stops, freno_duro, registrar_resultado_en_stops, reiniciar_la_cuenta,
    drawdown_de_falsacion, ALFA_FALSACION, FRACCION_KELLY_MAXIMA,
};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

#[test]
fn test_f1_010_global_max_drawdown_active_governance() {
    let arena: Arc<GlobalArena> = GlobalArena::build_in_own_stack(100.0);
    let d_star = drawdown_de_falsacion(FRACCION_KELLY_MAXIMA, ALFA_FALSACION);

    // 1. Cuando global_max_drawdown es alto (p. ej. 0.96 > d*), rige d* = ~0.6316
    arena.config.global_max_drawdown.store(0.96, Relaxed);
    let f_def = freno_duro(&arena, 100.0, 100.0);
    assert!((f_def.tope_capital - d_star).abs() < 1e-6, "Tope cuando gen > d* debe ser d*");

    // 2. Establecer un presupuesto estricto: 20% max drawdown
    arena.config.global_max_drawdown.store(0.20, Relaxed);
    let f_estricto = freno_duro(&arena, 100.0, 100.0);
    assert!((f_estricto.tope_capital - 0.20).abs() < 1e-6, "Tope debe reflejar global_max_drawdown = 0.20");

    // 3. Evaluar con caída de 15% (capital = 85): NO debe disparar freno por capital
    let f_safe = freno_duro(&arena, 100.0, 85.0);
    assert!(!f_safe.por_capital(), "Caída 15% no debe disparar tope de 20%");

    // 4. Evaluar con caída de 22% (capital = 78): SÍ debe disparar freno por capital
    let f_breach = freno_duro(&arena, 100.0, 78.0);
    assert!(f_breach.por_capital(), "Caída 22% debe disparar tope de 20%");
    assert!(f_breach.echado(), "Freno duro debe estar echado");
}

#[test]
fn test_f1_011_rolling_window_drawdown_non_absorbing() {
    let arena: Arc<GlobalArena> = GlobalArena::build_in_own_stack(100.0);

    // 1. Registrar 30 cierres perdedores consecutivos de -1.0 stops
    for _ in 0..30 {
        registrar_resultado_en_stops(&arena, -1.0);
    }
    let caida_30 = caida_en_stops(&arena);
    assert!((caida_30 - 30.0).abs() < 1e-9, "Caída tras 30 pérdidas debe ser 30.0 stops: {caida_30}");

    // 2. Registrar 200 cierres ganadores consecutivos de +1.0 stops
    // Al pasar 200 operaciones, las 30 pérdidas iniciales ruedan fuera de la ventana deslizante
    for _ in 0..200 {
        registrar_resultado_en_stops(&arena, 1.0);
    }

    let caida_post_recuperacion = caida_en_stops(&arena);
    assert_eq!(
        caida_post_recuperacion, 0.0,
        "Tras 200 operaciones netas ganadoras, la caída en ventana debe ser 0.0 (no absorbente)"
    );

    // 3. Probar rearme explícito
    for _ in 0..10 {
        registrar_resultado_en_stops(&arena, -1.0);
    }
    assert!(caida_en_stops(&arena) > 0.0);

    reiniciar_la_cuenta(&arena, 100.0);
    assert_eq!(caida_en_stops(&arena), 0.0, "reiniciar_la_cuenta debe reiniciar la ventana");
    assert_eq!(arena.resultado_en_stops.load(Relaxed), 0.0);
    assert_eq!(arena.pico_en_stops.load(Relaxed), 0.0);
}
