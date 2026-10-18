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
    arena.registry.set("hedge_eta", 0.05);
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

/// G8-OMEGA-09 (OLA Ω148): Las corrientes físicas opuestas (Iceberg, Dirac, Kähler,
/// BSD, Yang-Mills, Wasserstein) pueden atenuar el voto de una escala hasta 0.0,
/// pero NUNCA invierten el signo del desplazamiento espectral `x(τ_k)` ni fabrican
/// un voto direccional cuando `x(τ_k) == 0.0`.
#[test]
fn g8_omega_09_opposing_currents_attenuate_without_sign_inversion() {
    let mut x_pos = [0.0f64; 32];
    for v in &mut x_pos {
        *v = 0.05; // Desplazamiento positivo moderado en todas las escalas
    }
    // Corrientes físicas fuertemente negativas (-1.0) oponiéndose a x > 0
    let voto_opuesto = FlowExcitationConfluenceEngine::voto_espectral_con_wasserstein_dirac_bsd_riemann_kahler_gauge_y_helmholtz(
        &x_pos,
        3.5,  // excitación alta
        -1.0, // p_ice opuesta
        0.85, // laminar_fraction
        -1.0, // j_ym opuesta
        -1.0, // j_kh opuesta
        1.0,  // riemann_gue_conviction
        -1.0, // j_bsd opuesta
        1.0,  // bsd_conviction_modulation
        -1.0, // j_dirac opuesta
        1.0,  // dirac_conviction_modulation
        -1.0, // v_wk opuesta
        1.0,  // wasserstein_conviction_modulation
    );
    for k in 0..32 {
        let vk = voto_opuesto.en_escala(k);
        assert!(
            vk >= 0.0,
            "G8-OMEGA-09: Corrientes opuestas jamás pueden invertir el signo de x(τ_{k}) > 0: vk={vk}"
        );
        assert_eq!(
            vk, 0.0,
            "Con oposición total mayor que |x|, el voto en escala {k} se atenúa a 0.0: vk={vk}"
        );
    }

    // Y cuando x == 0.0, ninguna corriente física puede fabricar un voto direccional
    let x_cero = [0.0f64; 32];
    let voto_en_cero = FlowExcitationConfluenceEngine::voto_espectral_con_wasserstein_dirac_bsd_riemann_kahler_gauge_y_helmholtz(
        &x_cero, 3.5, 1.0, 0.85, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    );
    for k in 0..32 {
        assert_eq!(
            voto_en_cero.en_escala(k),
            0.0,
            "Sin desplazamiento espectral (x=0), las corrientes no fabrican voto en escala {k}"
        );
    }
}

/// LG8-145-06 (OLA Ω148): En el régimen de fallback escalar (`tau_dom < 30_000 ms` o sin dominante),
/// `FeynmanPropagatorEngine` es estructuralmente inoperable y no diluye `n_fuentes`.
#[test]
fn lg8_145_06_feynman_inoperable_in_scalar_fallback_does_not_dilute_ensemble() {
    use signal_engine::feynman_propagator::FeynmanPropagatorEngine;

    let arena_solo = quantum_arena::GlobalArena::build_in_own_stack(13.0);
    let mut orch_solo = TensorVoteOrchestrator::new(Arc::clone(&arena_solo));
    orch_solo.add_strategy(Box::new(NamedSource("alpha_voter", 0.6)));
    let conf_solo = orch_solo
        .evaluate_continuous_consensus_for_coin(0, "BTCUSDT")
        .net_confidence;

    let arena_with_feynman = quantum_arena::GlobalArena::build_in_own_stack(13.0);
    let mut orch_with_feynman = TensorVoteOrchestrator::new(Arc::clone(&arena_with_feynman));
    orch_with_feynman.add_strategy(Box::new(NamedSource("alpha_voter", 0.6)));
    orch_with_feynman.add_strategy(Box::new(FeynmanPropagatorEngine::new()));
    let conf_with_feynman = orch_with_feynman
        .evaluate_continuous_consensus_for_coin(0, "BTCUSDT")
        .net_confidence;

    assert!(
        (conf_solo - conf_with_feynman).abs() < 1e-12,
        "FeynmanPropagatorEngine sin dominante espectral operable no debe diluir n_fuentes: solo={conf_solo} vs con_feynman={conf_with_feynman}"
    );
}

/// CSG-2 (OLA Ω148): En la cadencia de 1024 consensos, Hedge evalúa el retorno relativo
/// adimensional `((mid - prev) / prev) / atr_pct` contra el voto anclado en `t_prev`
/// (`voto_ancla_hedge`), NO contra el voto recién emitido en `t_now`.
#[test]
fn csg_2_hedge_cadence_scores_anchored_t_prev_vote_with_dimensionless_return() {
    use std::sync::atomic::AtomicI64;

    struct FlippingVoter {
        name: &'static str,
        sign: Arc<AtomicI64>,
    }
    impl QuantumStrategy for FlippingVoter {
        fn name(&self) -> &str {
            self.name
        }
        fn init(&mut self, _: Arc<OmniscientRegistry>) -> Result<(), String> {
            Ok(())
        }
        fn evaluate(&self) -> f64 {
            (self.sign.load(Ordering::Relaxed) as f64) * 0.8
        }
    }

    let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
    arena.registry.set("hedge_eta", 0.05);
    arena.registry.set_for_coin(0, "atr_pct", 0.001); // 10 bps
    arena.coins[0]
        .current_price
        .store(60_000.0, Ordering::Relaxed);

    let sign_flipper = Arc::new(AtomicI64::new(1)); // En t=0 vota Long (+0.8)
    let sign_bear = Arc::new(AtomicI64::new(-1));   // En t=0 vota Short (-0.8)

    let mut orch = TensorVoteOrchestrator::new(Arc::clone(&arena));
    orch.add_strategy(Box::new(FlippingVoter {
        name: "flipper_correct_at_t0",
        sign: Arc::clone(&sign_flipper),
    }));
    orch.add_strategy(Box::new(FlippingVoter {
        name: "bear_wrong_at_t0",
        sign: Arc::clone(&sign_bear),
    }));

    // Paso n = 0: ancla prev_mid = 60_000.0 y voto_ancla_hedge = [+0.8, -0.8]
    let _ = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");

    // Pasos n = 1..1023
    for _ in 1..1024 {
        let _ = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
    }

    // Justo en n = 1024: el precio subió +10 bps (de 60_000 a 60_060, retorno +0.001 == +1x ATR),
    // lo que valida el voto Long (+0.8) emitido en t=0.
    // PERO en n = 1024 `flipper_correct_at_t0` cambia su nuevo voto a Short (-0.8) para la SIGUIENTE ventana.
    arena.coins[0]
        .current_price
        .store(60_060.0, Ordering::Relaxed);
    sign_flipper.store(-1, Ordering::Relaxed);
    sign_bear.store(1, Ordering::Relaxed);

    let _ = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");

    let pesos: std::collections::HashMap<_, _> =
        orch.pesos_hedge_snapshot().into_iter().collect();
    let w_flipper = *pesos.get("flipper_correct_at_t0").unwrap();
    let w_bear = *pesos.get("bear_wrong_at_t0").unwrap();

    assert!(
        w_flipper > 1.0 && w_bear < 1.0 && w_flipper > w_bear,
        "CSG-2: Hedge debe premiar el voto emitido en t=0 (ancla) frente al retorno [0, 1024], no el voto de t=1024: w_flipper={w_flipper}, w_bear={w_bear}"
    );
}

