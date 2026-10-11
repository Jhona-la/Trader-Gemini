//! C-F3 / F3.1-03 (Línea C): en el consenso escalar del orquestador, un voto
//! infinitesimal no puede comprar convicción.
//!
//! Teoría que se prueba: la confianza de una decisión debe crecer con la
//! evidencia. En el camino escalar (sin `consenso_espectral_dominante`, el
//! modo de arranque y de espectro plano) el orquestador calcula
//! net = (Σv⁺ − Σ|v⁻|)/Σ|v| · (0,70 + 0,30·eff), con
//! eff = min(1, Σ|v|/N · (1 + 0,1·(min(N, 5) − 1))). El primer factor sólo
//! mira signos (es invariante a una escala común de los votos) y el segundo
//! tiene un piso de 0,70. Por eso un único voto de 2·10⁻⁹ da Long con 0,70,
//! y dos votos de 0,002 dan 0,7007 frente a 0,997 de dos votos de 0,9: la
//! evidencia ×450 sólo mueve la confianza un 42 %.
//!
//! Solución conocida, aritmética cerrada:
//! - un voto de 2·10⁻⁹: Flat, o confianza < 10⁻⁶ (≈ la magnitud del voto);
//! - con (0,002; 0,002) frente a (0,9; 0,9), la confianza debe escalar con la
//!   evidencia: c_débil ≤ 0,1·c_fuerte (la razón de magnitudes es 1/450).
//!
//! Estrategias de salida fija sobre una arena nueva, sin claves espectrales:
//! el modo es el escalar.
//!
//! Convención: un test que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto.

use omniscient_registry::OmniscientRegistry;
use quantum_arena::GlobalArena;
use signal_engine::orchestrator::TensorVoteOrchestrator;
use signal_engine::SignalType;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// Estrategia de salida fija: aísla la regla de agregación del orquestador.
struct Fija(f64, &'static str);
impl QuantumStrategy for Fija {
    fn name(&self) -> &str {
        self.1
    }
    fn init(&mut self, _r: Arc<OmniscientRegistry>) -> Result<(), String> {
        Ok(())
    }
    fn evaluate(&self) -> f64 {
        self.0
    }
}

/// (es_plano, net_confidence) de una decisión escalar con los votos dados.
fn decide(votos: &[f64]) -> (bool, f64) {
    const NOMBRES: [&str; 4] = ["cf3_fija_a", "cf3_fija_b", "cf3_fija_c", "cf3_fija_d"];
    let mut o = TensorVoteOrchestrator::new(GlobalArena::build_in_own_stack(13.0));
    for (i, &v) in votos.iter().enumerate() {
        o.add_strategy(Box::new(Fija(v, NOMBRES[i])));
    }
    let d = o.evaluate_continuous_consensus_for_coin(0, "CF3ESCUSDT");
    (d.signal == SignalType::Flat, d.net_confidence)
}

#[test]
#[ignore = "OPEN F3.1-03 (Qoder, consenso): net = Σv/Σ|v|·(0,70 + 0,30·eff) sólo mira signos y tiene piso 0,70; un único voto de 2e-9 da Long con 0,70 y (0,002; 0,002) da 0,70 frente a 0,997"]
fn cf3_un_voto_infinitesimal_no_compra_conviccion() {
    let (plano, c) = decide(&[2e-9]);
    let (_, c_debil) = decide(&[0.002, 0.002]);
    let (_, c_fuerte) = decide(&[0.9, 0.9]);
    println!(
        "observado: voto 2e-9 -> plano = {plano}, c = {c:.4}; (0,002; 0,002) -> {c_debil:.4}; \
         (0,9; 0,9) -> {c_fuerte:.4}"
    );
    assert!(
        plano || c < 1e-6,
        "un único voto de 2e-9 produce una decisión (plano = {plano}) con confianza {c:.3}"
    );
    assert!(
        c_debil <= 0.1 * c_fuerte,
        "votos 0,002 dan {c_debil:.3} y votos 0,9 dan {c_fuerte:.3}: la convicción no escala con la evidencia"
    );
}
