//! #611 (Ola 33) — CENSO EMPÍRICO DE VOTANTES: contrato del contador por
//! estrategia. Un voto MUDO (siempre 0.0) nunca suma no-cero; un voto VIVO
//! (siempre 0.9) suma en cada evaluación. Telemetría pura — el censo no
//! toca la decisión.

use omniscient_registry::OmniscientRegistry;
use signal_engine::orchestrator::TensorVoteOrchestrator;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

struct VotoFijo {
    nombre: &'static str,
    voto: f64,
}
impl QuantumStrategy for VotoFijo {
    fn name(&self) -> &str {
        self.nombre
    }
    fn init(&mut self, _: Arc<OmniscientRegistry>) -> Result<(), String> {
        Ok(())
    }
    fn evaluate(&self) -> f64 {
        self.voto
    }
    fn evaluate_for_coin(&self, _: usize, _: &str) -> f64 {
        self.voto
    }
}

#[test]
fn qo_611_el_censo_mide_vida_por_estrategia() {
    let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
    let mut orch = TensorVoteOrchestrator::new(arena);
    orch.add_strategy(Box::new(VotoFijo { nombre: "Mudo", voto: 0.0 }));
    orch.add_strategy(Box::new(VotoFijo { nombre: "Vivo", voto: 0.9 }));

    for _ in 0..7 {
        let _ = orch.evaluate_continuous_consensus_for_coin(0, "TESTUSDT");
        let _ = orch.evaluate_continuous_consensus_for_coin(1, "TESTUSDT");
    }
    let censo = orch.censo_snapshot();
    assert_eq!(censo.len(), 2, "dos estrategias registradas");
    let (nombre_m, total_m, nc_m) = censo[0];
    let (nombre_v, total_v, nc_v) = censo[1];
    assert_eq!(nombre_m, "Mudo");
    assert_eq!(nombre_v, "Vivo");
    assert_eq!(total_m, 14, "7 consensos × 2 monedas");
    assert_eq!(total_v, 14);
    assert_eq!(nc_m, 0, "el voto mudo JAMÁS suma no-cero — esa es la muerte que el censo detecta");
    assert_eq!(nc_v, 14, "el voto vivo suma siempre");
}
