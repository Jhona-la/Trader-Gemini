//! Suite de pruebas de risk-engine (tarea 3 del integrador): espectro, regimen, Koopman, Hodge, RMT y caminos sin asignacion.
//! Un solo binario de prueba; cada modulo es el antiguo `tests/<modulo>.rs` sin cambios
//! salvo las rutas de `include_str!`. Solo entran archivos que no escriben estado global
//! del proceso (registro de simbolos, contadores de rechazo, variables de entorno).

mod f1_089_091_zero_alloc_hot_path_contract;
mod full_spectrum_contract;
mod hodge_zero_alloc_contract;
mod koopman_kelly_envelope_contract;
mod lg31_koopman_bifurcacion_contract;
mod regime_taxonomy_diagnostics;
mod spectral_regime_contract;
mod spectral_risk_contract;
mod zero_alloc_rmt_contract;
