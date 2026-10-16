//! TRIAJE B (GLM 109): (a), (c), (d), (e) DRENADOS — sus contratos ahora
//! verifican la semántica honesta. (a) drenado en Ola Ω97 (F3.1-03): un voto infinitesimal
//! no compra convicción y escala con la evidencia. (b) DRENADO (GLM 127):
//! invariancia a clones — el consenso agrupa por fuente (`name()`) y k
//! copias aportan la masa de UNA (media, XIV §202).
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

/// Fuente con NOMBRE propio — para distinguir clones (mismo name) de
/// fuentes independientes (names distintos) en los contratos de (b).
struct NamedSource(&'static str, f64);
impl QuantumStrategy for NamedSource {
    fn name(&self) -> &str {
        self.0
    }
    fn init(&mut self, _: Arc<OmniscientRegistry>) -> Result<(), String> {
        Ok(())
    }
    fn evaluate(&self) -> f64 {
        self.1
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

fn conviction_of(sources: &[(&'static str, f64)]) -> f64 {
    let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
    let mut engine = TensorVoteOrchestrator::new(arena);
    for (name, value) in sources {
        engine.add_strategy(Box::new(NamedSource(name, *value)));
    }
    engine.evaluate_continuous_consensus().net_confidence
}

/// (a) DRENADO (Ola Ω97 / F3.1-03): un voto infinitesimal no compra convicción.
/// Se erradicó el piso artificial de 0.70; la confianza escala monótonamente
/// con la evidencia y valores infinitesimales quedan sub-corte (Flat).
#[test]
fn tiny_positive_scores_do_not_have_false_floor() {
    assert_eq!(conviction(1e-12, 1), 0.0);
    assert_eq!(conviction(0.0, 1), 0.0);
}

/// (b) DRENADO (GLM 127): INVARIANCIA A CLONES — k copias de una misma
/// fuente aportan la masa de UNA (media del grupo). XIV §202: "clonar
/// un experto debe dividir su masa previa en lugar de crearla". Antes:
/// conviction(0.2, 5) > conviction(0.2, 1) — el boost por instancias
/// compraba concordancia sin evidencia nueva. (0.5 para estar holgado
/// sobre el corte de señal, no en su borde.)
#[test]
fn clones_divide_mass_instead_of_creating_it() {
    let una = conviction(0.5, 1);
    let cinco = conviction(0.5, 5);
    assert!(
        (cinco - una).abs() < 1e-15,
        "5 clones == 1 fuente: {cinco} vs {una}"
    );
    assert!(una > 0.4, "la fuente sola sí opina: {una}");
}

/// La concordancia REAL sigue premiada: 5 fuentes INDEPENDIENTES con el
/// mismo voto elevan la convicción efectiva frente a 1 sola (el boost
/// cuenta fuentes únicas, no representaciones).
#[test]
fn distinct_concordant_sources_still_boost() {
    let una = conviction_of(&[("src_a", 0.2)]);
    let cinco_fuentes = conviction_of(&[
        ("src_a", 0.2),
        ("src_b", 0.2),
        ("src_c", 0.2),
        ("src_d", 0.2),
        ("src_e", 0.2),
    ]);
    assert!(
        cinco_fuentes > una,
        "5 fuentes independientes > 1: {cinco_fuentes} > {una}"
    );
}

/// Discordancia interna de una fuente: copias con signos opuestos se
/// cancelan por la media — la masa del grupo es la del veredicto medio,
/// no la suma de gritos.
#[test]
fn mixed_signs_in_one_source_attenuate_by_mean() {
    let split = conviction_of(&[("src_a", 0.2), ("src_a", -0.2)]);
    assert_eq!(split, 0.0, "media 0 ⇒ masa 0: {split}");
}

// ── GLM 130: HEDGE POR FUENTE (exponential-weights) ──────────────────

fn hedge_engine(
    fuentes: &[(&'static str, f64)],
) -> TensorVoteOrchestrator {
    let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
    let mut engine = TensorVoteOrchestrator::new(arena);
    for (name, value) in fuentes {
        engine.add_strategy(Box::new(NamedSource(name, *value)));
    }
    // Un consenso para poblar ultimo_voto (registro del voto con signo).
    let _ = engine.evaluate_continuous_consensus();
    engine
}

/// (i) La fuente que CONTRADICE la realización decae; la que acierta
/// sube. La dirección positiva sostenida separa los pesos.
#[test]
fn hedge_devalua_al_contradictorio_y_premia_al_acertivo() {
    let engine = hedge_engine(&[("adivino", 0.5), ("contrario", -0.5)]);
    for _ in 0..50 {
        engine.actualizar_hedge_con_realizacion(1.0);
        // Re-poblar votos: el update consume ultimo_voto sin tocarlo.
        let _ = engine.evaluate_continuous_consensus();
    }
    let pesos: std::collections::HashMap<_, _> =
        engine.pesos_hedge_snapshot().into_iter().collect();
    let w_adivino = *pesos.get("adivino").unwrap();
    let w_contrario = *pesos.get("contrario").unwrap();
    assert!(
        w_adivino > w_contrario,
        "acertivo > contradictorio: {w_adivino} > {w_contrario}"
    );
    assert!(w_contrario < 1.0, "el contradictorio decae: {w_contrario}");
}

/// (ii) Sin explosión ni muerte eterna: tras 500 aciertos seguidos el
/// peso vive dentro de los bounds suaves (renormalización a media 1.0).
#[test]
fn hedge_aciertos_persistentes_no_explotan() {
    let engine = hedge_engine(&[("siempre_long", 0.5), ("siempre_short", -0.5)]);
    for _ in 0..500 {
        engine.actualizar_hedge_con_realizacion(1.0);
        let _ = engine.evaluate_continuous_consensus();
    }
    for (_n, w) in engine.pesos_hedge_snapshot() {
        assert!(
            w.is_finite() && (0.04..=21.0).contains(&w),
            "peso fuera de bounds suaves: {w}"
        );
    }
}

/// (iii) Arranque frío = pesos uniformes 1.0 ⇒ el consenso es
/// idéntico al régimen sin updates de hedge (paridad del primer estado).
#[test]
fn hedge_arranque_frio_es_paridad_uniforme() {
    let fuentes = [("a", 0.4), ("b", 0.4), ("c", 0.4)];
    let engine = hedge_engine(&fuentes);
    for (_n, w) in engine.pesos_hedge_snapshot() {
        assert_eq!(w, 1.0, "arranque: todos 1.0");
    }
    let con_hedge_sin_updates = engine.evaluate_continuous_consensus().net_confidence;
    let plano = hedge_engine(&fuentes);
    let sin_updates = plano.evaluate_continuous_consensus().net_confidence;
    assert!(
        (con_hedge_sin_updates - sin_updates).abs() < 1e-15,
        "paridad arranque frío: {con_hedge_sin_updates} == {sin_updates}"
    );
}

/// (iv) La invariancia a clones (127) SE PRESERVA con el hedge: k
/// copias de una fuente pesan como una (el peso es por nombre/grupo).
#[test]
fn hedge_preserva_invariancia_a_clones() {
    let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
    let mut engine = TensorVoteOrchestrator::new(arena);
    for _ in 0..5 {
        engine.add_strategy(Box::new(Constant(0.5)));
    }
    let cinco = engine.evaluate_continuous_consensus().net_confidence;
    let una = conviction(0.5, 1);
    assert!(
        (cinco - una).abs() < 1e-15,
        "5 clones == 1 fuente (con hedge): {cinco} vs {una}"
    );
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
