//! Suite de pruebas de risk-engine (tarea 3 del integrador): admision, Kelly, sondas SPRT, evidencia, drawdown y contratos de fuente.
//! Un solo binario de prueba; cada modulo es el antiguo `tests/<modulo>.rs` sin cambios
//! salvo las rutas de `include_str!`. Solo entran archivos que no escriben estado global
//! del proceso (registro de simbolos, contadores de rechazo, variables de entorno).

mod cf2_estadistica_solucion_conocida;
mod cf2_riesgo_solucion_conocida;
mod cf3_registro_consejo;
mod cf7_sonda_p0_alto;
mod compounding_audit;
mod cramer_lundberg_exposure_monotonicity_contract;
mod empirical_bayes_prior_contract;
mod envio_contract;
mod f1_014_cramer_lundberg_real_exposure_contract;
mod f1_081_payoff_single_ema_and_sample_memory_contract;
mod f1_082_kelly_capital_scale_continuity_contract;
mod f1_083_orchestrator_continuous_pressure_contract;
mod f1_090_sonda_sprt_reachability_and_zero_alloc_contract;
mod f1_115_capital_regime_dashmap_cache_contract;
mod lg34_sonda_sprt_prior_real_contract;
mod nonfinite_admission_contract;
mod portfolio_admission_contract;
mod qsr_cl07_drawdown_consejo_contract;
mod r7_r2_rejection_names_ruin_lcb_contract;
mod sol_a2_margen_reducido_auditable;
mod sonda_sprt_adaptation_contract;
mod ville_evidence_contract;
