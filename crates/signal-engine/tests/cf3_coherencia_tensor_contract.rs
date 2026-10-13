//! C-F3 / F3.3-05 (Línea C): un espectro en contra de la escala dominante
//! debe bajar la confianza tensorial, no dar la misma que uno neutro.
//!
//! Teoría que se prueba: más evidencia en contra no puede dar ni más ni la
//! misma confianza que la ausencia de evidencia (monotonía de la confianza
//! en la evidencia). Con veredicto espectral activo, el orquestador hace
//! net = v_dom·(0,70 + 0,30·coherencia) con
//! coherencia = (media_banda/v_dom).clamp(0, 1). El recorte en 0 hace que una
//! banda de signo opuesto (media −0,8 con v_dom 0,8) dé coherencia 0, igual
//! que una banda neutra: las dos decisiones salen con 0,8·0,70 = 0,56.
//!
//! Solución conocida (mismo montaje que qo_624 en orchestrator.rs): con
//! media −0,8 el resultado debe ser Flat o tener confianza estrictamente menor
//! que con media 0. Sin tolerancia: la desigualdad es estricta y la
//! diferencia esperada tras un arreglo es de orden 0,1.
//!
//! Convención: un test que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto.

use omniscient_registry::OmniscientRegistry;
use signal_engine::orchestrator::TensorVoteOrchestrator;
use signal_engine::SignalType;
use std::sync::{atomic::Ordering::Relaxed, Arc};
use strategy_core::QuantumStrategy;

struct Fija(f64);
impl QuantumStrategy for Fija {
    fn name(&self) -> &str {
        "Fija"
    }
    fn init(&mut self, _r: Arc<OmniscientRegistry>) -> Result<(), String> {
        Ok(())
    }
    fn evaluate(&self) -> f64 {
        self.0
    }
}

/// Decisión con un dominante espectral operable de 0,8 a 1 h y la media de
/// banda dada.
fn decision(media: f64) -> (SignalType, f64) {
    let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
    arena.config.min_confidence_btc.store(0.51, Relaxed);
    let mut orch = TensorVoteOrchestrator::new(Arc::clone(&arena));
    orch.add_strategy(Box::new(Fija(0.9)));
    arena
        .registry
        .set_for_coin(0, "consenso_espectral_dominante", 0.8);
    arena
        .registry
        .set_for_coin(0, "consenso_espectral_tau", 3_600_000.0);
    arena
        .registry
        .set_for_coin(0, "consenso_espectral_media", media);
    let d = orch.evaluate_continuous_consensus_for_coin(0, "BTCUSDT");
    (d.signal, d.net_confidence)
}

#[test]
fn cf3_espectro_opuesto_baja_la_confianza_tensorial() {
    let (lado_neutro, neutro) = decision(0.0);
    let (lado, opuesto) = decision(-0.8);
    println!(
        "observado: media 0 -> {lado_neutro:?} {neutro:.4}; media −0,8 -> {lado:?} {opuesto:.4}"
    );
    assert!(
        lado == SignalType::Flat || opuesto < neutro,
        "banda en contra ({opuesto}) pesa igual que banda neutra ({neutro})"
    );
}
