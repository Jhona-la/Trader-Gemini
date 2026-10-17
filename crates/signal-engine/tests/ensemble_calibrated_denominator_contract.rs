//! Contrato Formal: Calibración Dinámica de Denominador Efectivo ($n_{\text{hablantes}} / n_{\text{elegibles}}$)
//! y Eliminación de Dilución por Fuentes Inoperables (Ola Ω145).
//!
//! # F3.1-05: El denominador del ensamble debe distinguir rigurosamente entre:
//! 1. Votante Activo (|voto| > 1e-9): Aporta convicción direccional.
//! 2. Abstención Operable (voto = 0.0, is_operable = true): Observador válido que concluye ausencia de edge;
//!    SÍ diluye legítimamente la convicción promedio del ensamble.
//! 3. Inelegibilidad Contextual (voto = 0.0, is_operable = false): Estrategia que carece de condiciones
//!    de datos o dimensionales (e.g. sin feed spot, o monoactivo N=1 para Yang-Mills);
//!    NO diluye ni penaliza la convicción de los motores operables.
//! 4. Invariante Anti-Phantom-Boost: Un votante solitario entre fuentes elegibles NUNCA recibe boost de ensamble.

use omniscient_registry::OmniscientRegistry;
use quantum_arena::GlobalArena;
use signal_engine::coaxial_breakout::CoaxialBreakoutEngine;
use signal_engine::orchestrator::TensorVoteOrchestrator;
use std::sync::Arc;
use strategy_core::stat_arb::StatArbEngine;
use strategy_core::vecm_arbitrage::JohansenVecmEngine;
use strategy_core::yang_mills_gauge::YangMillsGaugeEngine;
use strategy_core::QuantumStrategy;

struct MockOperableStrategy {
    name: &'static str,
    val: f64,
    operable: bool,
}

impl QuantumStrategy for MockOperableStrategy {
    fn name(&self) -> &str {
        self.name
    }
    fn init(&mut self, _registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        Ok(())
    }
    fn evaluate(&self) -> f64 {
        self.val
    }
    fn evaluate_for_coin(&self, _coin_id: usize, _symbol: &str) -> f64 {
        self.val
    }
    fn is_operable_for_coin(&self, _coin_id: usize, _symbol: &str) -> bool {
        self.operable
    }
}

#[test]
fn test_invariante_anti_phantom_boost_preservado() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    arena
        .config
        .min_confidence_btc
        .store(0.50, std::sync::atomic::Ordering::Relaxed);

    let mut orch = TensorVoteOrchestrator::new(arena);

    // 1 activa y 4 silentes plenamente operables
    orch.add_strategy(Box::new(MockOperableStrategy { name: "Activa", val: 0.50, operable: true }));
    orch.add_strategy(Box::new(MockOperableStrategy { name: "Silente1", val: 0.0, operable: true }));
    orch.add_strategy(Box::new(MockOperableStrategy { name: "Silente2", val: 0.0, operable: true }));
    orch.add_strategy(Box::new(MockOperableStrategy { name: "Silente3", val: 0.0, operable: true }));
    orch.add_strategy(Box::new(MockOperableStrategy { name: "Silente4", val: 0.0, operable: true }));

    let decision = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");

    // Con 1 votante de 0.50 sobre 5 estrategias operables:
    // avg_conviction = 0.50 / 5 = 0.10.
    // active_voters = 1 -> boost = 1.0 (anti-phantom boost estricto).
    // effective_conviction = 0.10 * 1.0 = 0.10.
    assert!(
        (decision.net_confidence - 0.10).abs() < 1e-9,
        "La confianza esperada es 0.10, pero fue {}",
        decision.net_confidence
    );
}

#[test]
fn test_invariante_exclusion_inoperable() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    arena
        .config
        .min_confidence_btc
        .store(0.50, std::sync::atomic::Ordering::Relaxed);

    let mut orch = TensorVoteOrchestrator::new(arena);

    // 1 activa (0.50), 2 silentes operables (0.0), y 2 contextualmente inoperables (0.0, operable=false)
    orch.add_strategy(Box::new(MockOperableStrategy { name: "Activa", val: 0.50, operable: true }));
    orch.add_strategy(Box::new(MockOperableStrategy { name: "Silente1", val: 0.0, operable: true }));
    orch.add_strategy(Box::new(MockOperableStrategy { name: "Silente2", val: 0.0, operable: true }));
    orch.add_strategy(Box::new(MockOperableStrategy { name: "Inoperable1", val: 0.0, operable: false }));
    orch.add_strategy(Box::new(MockOperableStrategy { name: "Inoperable2", val: 0.0, operable: false }));

    let decision = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");

    // Denominador efectivo: 3 fuentes operables (Activa + Silente1 + Silente2).
    // Inoperable1 e Inoperable2 NO diluyen.
    // avg_conviction = 0.50 / 3 = 0.16666666666666666.
    // active_voters = 1 -> boost = 1.0.
    // net_confidence = 0.50 / 3.
    let expected = 0.50 / 3.0;
    assert!(
        (decision.net_confidence - expected).abs() < 1e-9,
        "La confianza esperada es {}, pero fue {}",
        expected,
        decision.net_confidence
    );
}

#[test]
fn test_consenso_unanime_operable_con_inoperables_presentes() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    arena
        .config
        .min_confidence_btc
        .store(0.50, std::sync::atomic::Ordering::Relaxed);

    let mut orch = TensorVoteOrchestrator::new(arena);

    // 3 operables unánimes en 0.50 + 2 inoperables (0.0)
    orch.add_strategy(Box::new(MockOperableStrategy { name: "Operable1", val: 0.50, operable: true }));
    orch.add_strategy(Box::new(MockOperableStrategy { name: "Operable2", val: 0.50, operable: true }));
    orch.add_strategy(Box::new(MockOperableStrategy { name: "Operable3", val: 0.50, operable: true }));
    orch.add_strategy(Box::new(MockOperableStrategy { name: "Inoperable1", val: 0.0, operable: false }));
    orch.add_strategy(Box::new(MockOperableStrategy { name: "Inoperable2", val: 0.0, operable: false }));

    let decision = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");

    // Denominador efectivo = 3.
    // avg_conviction = (3 * 0.50) / 3 = 0.50.
    // active_voters = 3 -> boost = 1.0 + (3 - 1) * 0.1 = 1.20.
    // effective_conviction = 0.50 * 1.20 = 0.60.
    // prob_long = 1.0, prob_short = 0.0 -> net = 0.60.
    assert!(
        (decision.net_confidence - 0.60).abs() < 1e-9,
        "La confianza esperada para 3 votos unánimes sobre 3 operables es 0.60, pero fue {}",
        decision.net_confidence
    );
}

#[test]
fn test_invarianza_a_clones_con_operabilidad() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    arena
        .config
        .min_confidence_btc
        .store(0.50, std::sync::atomic::Ordering::Relaxed);

    let mut orch = TensorVoteOrchestrator::new(arena);

    // 3 clones de la misma fuente "CloneMaster"
    orch.add_strategy(Box::new(MockOperableStrategy { name: "CloneMaster", val: 0.50, operable: true }));
    orch.add_strategy(Box::new(MockOperableStrategy { name: "CloneMaster", val: 0.50, operable: true }));
    orch.add_strategy(Box::new(MockOperableStrategy { name: "CloneMaster", val: 0.50, operable: true }));

    let decision = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");

    // Los 3 clones se agrupan en 1 sola fuente única.
    // avg_conviction = 0.50 / 1 = 0.50.
    // active_voters = 1 -> boost = 1.0 (clones NO compran boost).
    // net_confidence = 0.50 * 1.0 = 0.50.
    assert!(
        (decision.net_confidence - 0.50).abs() < 1e-9,
        "La confianza esperada para clones es 0.50 (invarianza a clones), pero fue {}",
        decision.net_confidence
    );
}

#[test]
fn test_active_vote_overrides_inoperability() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    arena
        .config
        .min_confidence_btc
        .store(0.50, std::sync::atomic::Ordering::Relaxed);

    let mut orch = TensorVoteOrchestrator::new(arena);

    // Estrategia que declara operable=false pero emite voto activo (0.40)
    orch.add_strategy(Box::new(MockOperableStrategy { name: "SurpriseActive", val: 0.40, operable: false }));

    let decision = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");

    // Si emitió voto activo no-cero, es inequívocamente operable y cuenta
    assert!(
        (decision.net_confidence - 0.40).abs() < 1e-9,
        "El voto activo debe tener convicción 0.40, pero fue {}",
        decision.net_confidence
    );
}

#[test]
fn test_estrategias_concretas_operabilidad_contextual() {
    let reg = Arc::new(OmniscientRegistry::new());

    // 1. YangMillsGaugeEngine
    let mut ym_single = YangMillsGaugeEngine::new(1);
    assert!(ym_single.init(reg.clone()).is_ok());
    // Sin corriente y N=1 -> inoperable
    assert!(!ym_single.is_operable_for_coin(0, "BTCUSDT"));

    // Con N=3 y corriente presente
    let mut ym_multi = YangMillsGaugeEngine::new(3);
    assert!(ym_multi.init(reg.clone()).is_ok());
    reg.set_scoped("BTCUSDT", "yang_mills_current", 0.5);
    reg.set_scoped("BTCUSDT", "yang_mills_current_absent", 0.0);
    reg.set_scoped("BTCUSDT", "yang_mills_active_triangles", 2.0);
    assert!(ym_multi.is_operable_for_coin(0, "BTCUSDT"));

    // 2. StatArbEngine
    let mut statarb = StatArbEngine::new(30, 1.5);
    let reg_sa = Arc::new(OmniscientRegistry::new());
    assert!(statarb.init(reg_sa.clone()).is_ok());
    // Sin claves -> inoperable
    assert!(!statarb.is_operable_for_coin(0, "BTCUSDT"));
    // Con claves spot-futuro válidas
    reg_sa.set_scoped("BTCUSDT", "statarb_ou_zscore", 1.8);
    reg_sa.set_scoped("BTCUSDT", "statarb_half_life_ms", 45_000.0);
    assert!(statarb.is_operable_for_coin(0, "BTCUSDT"));

    // 3. CoaxialBreakoutEngine
    let mut coax = CoaxialBreakoutEngine::new();
    let reg_coax = Arc::new(OmniscientRegistry::new());
    assert!(coax.init(reg_coax.clone()).is_ok());
    // Sin claves atr_1s/5s/1m -> inoperable
    assert!(!coax.is_operable_for_coin(0, "BTCUSDT"));
    // Con las 3 claves publicadas -> operable
    reg_coax.set_scoped("BTCUSDT", "atr_1s", 0.0005);
    reg_coax.set_scoped("BTCUSDT", "atr_5s", 0.005);
    reg_coax.set_scoped("BTCUSDT", "atr_1m", 0.050);
    assert!(coax.is_operable_for_coin(0, "BTCUSDT"));

    // 4. JohansenVecmEngine
    let mut vecm = JohansenVecmEngine::default();
    let reg_vecm = Arc::new(OmniscientRegistry::new());
    assert!(vecm.init(reg_vecm.clone()).is_ok());
    // Sin claves -> inoperable
    assert!(!vecm.is_operable_for_coin(0, "BTCUSDT"));
    // Con vecm_zscore -> operable
    reg_vecm.set_scoped("BTCUSDT", "vecm_zscore", 2.1);
    assert!(vecm.is_operable_for_coin(0, "BTCUSDT"));
}
