//! TRIAJE B (GLM 109): (c), (d), (e) DRENADOS — sus contratos ahora
//! verifican la semántica honesta. (a) piso 0.70 y (b) boost por clones
//! siguen como diagnóstico: recalibran toda decisión del ensamble y la
//! auditoría XIV exige análisis de tasas de aceptación con datos —
//! decisión del consejo, no fix de ciclo.
use omniscient_registry::OmniscientRegistry;
use signal_engine::{
    flow_excitation_confluence::FlowExcitationConfluenceEngine,
    orchestrator::TensorVoteOrchestrator, trend_runner::HighPayoffTrendRunner,
};
use std::sync::{atomic::Ordering, Arc};
use strategy_core::{momentum_booster::VolatileMomentumBooster, QuantumStrategy};

struct Constant(f64);
impl QuantumStrategy for Constant {
    fn name(&self) -> &str {
        "constant"
    }
    fn init(&mut self, _: Arc<OmniscientRegistry>) -> Result<(), String> {
        Ok(())
    }
    fn evaluate(&self) -> f64 {
        self.0
    }
}

fn conviction(value: f64, copies: usize) -> f64 {
    let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
    let mut engine = TensorVoteOrchestrator::new(arena);
    for _ in 0..copies {
        engine.add_strategy(Box::new(Constant(value)));
    }
    engine.evaluate_continuous_consensus().net_confidence
}

#[test]
fn diagnostic_tiny_positive_scores_still_have_point_seven_floor() {
    // (a) — ABIERTO por decisión del consejo: el piso 0.70 es la
    // normalización D-101/D-425; retirarlo recalibra TODO el ensamble
    // (auditoría XIV §14 exige comparación de tasas de aceptación).
    assert!(conviction(1e-12, 1) >= 0.7);
    assert_eq!(conviction(0.0, 1), 0.0);
}

#[test]
fn diagnostic_clones_change_conviction_without_new_evidence() {
    // (b) — ABIERTO por decisión del consejo: invariancia a clones exige
    // preservar masa/prior por fuente (auditoría XIV §14), no sólo
    // quitar el boost.
    assert!(conviction(0.2, 5) > conviction(0.2, 1));
}

/// (c) DRENADO (GLM 109, CL-15): sin base medida el motor ML NO OPINA —
/// el fallback 0.5 fabricaba lift (con ml_prob 0.36 y base inventada, un
/// voto short ≈ −0.33 que no existiría con la base honesta).
#[test]
fn missing_model_base_means_ml_abstains_not_fabricated_lift() {
    let registry = Arc::new(OmniscientRegistry::new());
    registry.set("hawkes_intensity", 2.0);
    registry.set("order_book_imbalance", -0.5);
    registry.set("ml_prob_motor", 0.36);
    let mut trigger = FlowExcitationConfluenceEngine::new();
    trigger.init(registry.clone()).unwrap();
    // Sin base: AUSENCIA de voto (CL-15), jamás lift fabricado.
    assert_eq!(
        trigger.evaluate(),
        0.0,
        "sin base medida el motor no opina (no fabrica 0.5)"
    );
    // Con base honesta 0.3: ml_prob 0.36 está dentro del lift ±0.05
    // (rampa en 0.01/0.05) — voto ~0, y sobre todo JAMÁS negativo-fuerte.
    registry.set("ml_model_base", 0.3);
    let v = trigger.evaluate();
    assert!(v >= 0.0 || v.abs() < 0.05, "sin lift medido no hay voto fuerte: {v}");
    // Sin PROB (sin modelo): también ausencia, aunque la base exista.
    let registry2 = Arc::new(OmniscientRegistry::new());
    registry2.set("hawkes_intensity", 2.0);
    registry2.set("order_book_imbalance", -0.5);
    registry2.set("ml_model_base", 0.3);
    let mut trigger2 = FlowExcitationConfluenceEngine::new();
    trigger2.init(registry2).unwrap();
    assert_eq!(trigger2.evaluate(), 0.0, "sin ml_prob el motor no opina");
}

/// (d) DRENADO (GLM 109): el "límite superior suave" ya no COMPRIME lo
/// que está bajo el bound — saturación logística que preserva x ≪ bound.
#[test]
fn trend_extension_never_shrinks_the_base_target() {
    // Hurst neutro / vpin 0: expansión nula — el target es el base PISO
    // (identidad exacta: CERO compresión bajo el 80% del bound).
    let target = HighPayoffTrendRunner::calculate_expanded_tp(0.02, 0.5, 0.0, 0.01, 0.1);
    assert_eq!(
        target, 0.02,
        "bajo el punto de empalme el soft-cap es IDENTIDAD exacta"
    );
    // Sobre el bound: satura hacia él sin kink pero sigue CRECIENTE.
    let expandido = HighPayoffTrendRunner::calculate_expanded_tp(0.02, 0.9, 0.0, 0.001, 0.1);
    assert!(expandido > target, "más persistencia ⇒ más target");
    assert!(expandido <= 0.1 + 1e-9, "satura hacia el bound: {expandido}");
}

/// (e) DRENADO (GLM 109): PnL negativo NO activa la extensión de
/// momentum — sin ganancia realizada no hay evidencia de alineación a
/// favor (la regresión P15 alineaba vía dirección de posición).
#[test]
fn negative_pnl_disables_momentum_extension_even_in_cascade() {
    let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
    arena
        .config
        .dynamic_ofi_threshold
        .store(0.5, Ordering::Relaxed);
    arena.config.tensor_poly_a.store(0.1, Ordering::Relaxed);
    arena.config.tensor_poly_b.store(0.1, Ordering::Relaxed);
    arena
        .config
        .explosive_leverage_multiplier
        .store(2.0, Ordering::Relaxed);
    assert_eq!(
        VolatileMomentumBooster::calculate_tp_extension(1.0, -0.01, 3.0, 0.01, &arena),
        1.0,
        "PnL negativo ni con cascada: la extensión exige ganancia"
    );
}
