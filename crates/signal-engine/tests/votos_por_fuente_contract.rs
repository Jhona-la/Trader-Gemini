//! GLM 146 — contrato de `votos_por_fuente_snapshot`: el voto medio CON
//! SIGNO por fuente del último consenso, media de instancias de la misma
//! fuente. Alimento certificado de la medición SVD de ortogonalidad
//! (ortogonalidad_voces_medicion): si este snapshot mintiera, toda la
//! matriz de correlación del ensamble sería ficción.
use omniscient_registry::OmniscientRegistry;
use signal_engine::orchestrator::TensorVoteOrchestrator;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

struct Constante(&'static str, f64);
impl QuantumStrategy for Constante {
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

fn orquestador_con(instancias: &[(&'static str, f64)]) -> TensorVoteOrchestrator {
    // D-714: pila suficiente para construir el arena.
    let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
    let mut engine = TensorVoteOrchestrator::new(arena);
    for (nombre, valor) in instancias {
        engine.add_strategy(Box::new(Constante(nombre, *valor)));
    }
    engine
}

#[test]
fn votos_por_fuente_media_con_signo_de_instancias() {
    let engine = orquestador_con(&[
        ("alpha", 0.6),
        ("alpha", 0.2), // 2ª instancia de alpha: media 0.4
        ("beta", -0.4),
        ("gamma", 0.0), // muda: voto 0, pero presente con signo honesto
    ]);
    let _ = engine.evaluate_continuous_consensus();
    let votos = engine.votos_por_fuente_snapshot();
    let dato = |n: &str| votos.iter().find(|(k, _)| *k == n).map(|(_, v)| *v);
    let alpha = dato("alpha").expect("alpha presente");
    assert!((alpha - 0.4).abs() < 1e-12, "alpha = media de instancias: {alpha}");
    let beta = dato("beta").expect("beta presente");
    assert!((beta - (-0.4)).abs() < 1e-12, "beta conserva el SIGNO: {beta}");
    let gamma = dato("gamma").expect("gamma presente (muda con 0.0 honesto)");
    assert_eq!(gamma, 0.0);
    assert_eq!(votos.len(), 3, "una fila por FUENTE distinta, no por instancia");
}

#[test]
fn votos_por_fuente_sin_consenso_previo_son_cero_honesto() {
    let engine = orquestador_con(&[("alpha", 0.7), ("beta", -0.2)]);
    // Sin evaluar: nunca hubo consenso ⇒ último_voto intacto en 0.0
    // (arranque). El snapshot reporta la verdad del arranque, no inventa.
    let votos = engine.votos_por_fuente_snapshot();
    assert_eq!(votos.len(), 2);
    for (_, v) in &votos {
        assert_eq!(*v, 0.0, "arranque sin consenso ⇒ 0.0 inicial");
    }
}
