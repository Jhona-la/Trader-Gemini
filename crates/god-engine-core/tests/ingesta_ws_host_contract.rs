//! CL-51: el reinicio por reconexión del WS es del feed, no de la cuenta.

use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;

/// El reinicio por reconexión es del feed. Antes `reset_engines` también
/// hacía `risk_engine.reset(base_capital)`: con un máximo de 20 USD y la
/// cuenta en 15, tras la reconexión el veto medía la caída desde 13.
#[test]
fn cl51_una_reconexion_no_olvida_el_pico_del_drawdown() {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let mut core = GodEngineCore::new(arena);
    core.swing_nn = None;
    core.scalp_forest = None;
    core.risk_engine.peak_capital = 20.0;
    core.reset_engines();
    assert_eq!(core.risk_engine.peak_capital, 20.0);
}
