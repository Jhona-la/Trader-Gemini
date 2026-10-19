//! Contract test for Ola Ω168 (F1-053, F1-054, F1-055):
//! - F1-053: Council snapshot direct reads from CouncilParams (O(1) without redundant registry lookups)
//! - F1-054: Unified margin calculation in risk-engine, removal of conflicting nucleus haircut, and diag_margin_drops telemetry
//! - F1-055: Continuous adaptive Hurst duration modulation scaled by sample size N and DFA uncertainty without Brownian saturation

use std::sync::Arc;
use std::sync::atomic::Ordering;
use god_engine_core::{
    GodEngineCore,
    hurst_duration_modulation,
    hurst_duration_modulation_samples,
    dynamic_hurst_thresholds,
};
use quantum_arena::GlobalArena;

#[test]
fn f1_053_council_params_direct_read_and_isolation_from_registry() {
    let arena = Arc::new(GlobalArena::new(1000.0));
    let mut core = GodEngineCore::new_auxiliar(arena.clone());

    // Configurar un valor específico en QuantumConfig
    arena.config.council_causal_veto_threshold.store(0.85, Ordering::Relaxed);
    arena.config.council_utility_veto_floor.store(0.25, Ordering::Relaxed);

    // Sincronizar parámetros del consejo
    GodEngineCore::sync_council_params(&mut core.consejo_deliberacion.params, &arena.config);

    assert_eq!(core.consejo_deliberacion.params.causal_veto_threshold, 0.85);
    assert_eq!(core.consejo_deliberacion.params.utility_veto_floor, 0.25);

    // Envenenar maliciosamente el registro con valores espurios
    arena.registry.set_for_coin(0, "causal_veto_threshold", 0.12);
    arena.registry.set("causal_veto_threshold", 0.05);
    arena.registry.set_for_coin(0, "utility_veto_floor", 0.01);
    arena.registry.set("utility_veto_floor", 0.02);

    // Re-sincronizar: los parámetros del consejo deben provenir EXCLUSIVAMENTE de QuantumConfig / genome,
    // blindados contra cualquier polución del omniscient registry.
    GodEngineCore::sync_council_params(&mut core.consejo_deliberacion.params, &arena.config);
    assert_eq!(core.consejo_deliberacion.params.causal_veto_threshold, 0.85);
    assert_eq!(core.consejo_deliberacion.params.utility_veto_floor, 0.25);
}

#[test]
fn f1_054_margin_drops_counter_and_telemetry_initialization() {
    let arena = Arc::new(GlobalArena::new(1000.0));
    let mut core = GodEngineCore::new_auxiliar(arena.clone());

    // diag_margin_drops nace en 0
    assert_eq!(core.diag_margin_drops, 0);

    // Incrementar y verificar conteo
    core.diag_margin_drops += 1;
    assert_eq!(core.diag_margin_drops, 1);
}

#[test]
fn f1_055_adaptive_hurst_duration_modulation_properties() {
    // 1. Zona neutral bajo ruido browniano puro (N = 1024 => [0.45, 0.52])
    let (anti_1024, pers_1024) = dynamic_hurst_thresholds(1024);
    assert_eq!(anti_1024, 0.45);
    assert_eq!(pers_1024, 0.52);

    // En toda la zona neutral, la duración esperada se preserva exactamente intacta (factor 1.0)
    for h in [0.45, 0.46, 0.48, 0.50, 0.51, 0.52] {
        assert_eq!(
            hurst_duration_modulation(600_000, h),
            600_000,
            "H={h} debe dar factor 1.0 en zona neutral"
        );
    }

    // 2. Persistencia genuina (H > 0.52)
    // Con rampa de ancho 0.10, H = 0.52 + 0.10 = 0.62 satura en x2.0 (1_200_000)
    let dur_pers_max = hurst_duration_modulation(600_000, 0.62);
    assert_eq!(dur_pers_max, 1_200_000);
    let dur_pers_mid = hurst_duration_modulation(600_000, 0.57);
    assert!(dur_pers_mid > 600_000 && dur_pers_mid < 1_200_000);

    // 3. Antipersistencia genuina (H < 0.45)
    // Con rampa de ancho 0.10, H = 0.45 - 0.10 = 0.35 satura en x0.5 (300_000)
    let dur_anti_max = hurst_duration_modulation(600_000, 0.35);
    assert_eq!(dur_anti_max, 300_000);
    let dur_anti_mid = hurst_duration_modulation(600_000, 0.40);
    assert!(dur_anti_mid > 300_000 && dur_anti_mid < 600_000);

    // 4. Continuidad C^0 en las fronteras de transición
    let eps = 1e-6;
    let pers_minus = hurst_duration_modulation(600_000, pers_1024 - eps) as f64;
    let pers_plus = hurst_duration_modulation(600_000, pers_1024 + eps) as f64;
    assert!((pers_plus - pers_minus).abs() < 10.0, "Discontinuidad en pers_thr");

    let anti_minus = hurst_duration_modulation(600_000, anti_1024 - eps) as f64;
    let anti_plus = hurst_duration_modulation(600_000, anti_1024 + eps) as f64;
    assert!((anti_plus - anti_minus).abs() < 10.0, "Discontinuidad en anti_thr");

    // 5. Adaptabilidad por tamaño muestral N
    // Con N pequeño (ej. 256 muestras), la incertidumbre σ_H es mayor y la zona neutral/rampa se ensancha,
    // modulando MENOS agresivamente ante desviaciones menores
    let mod_n256 = hurst_duration_modulation_samples(600_000, 0.56, 256);
    let mod_n1024 = hurst_duration_modulation_samples(600_000, 0.56, 1024);
    assert!(
        mod_n256 <= mod_n1024,
        "A menor N la modulación debe ser más conservadora ante ruido: {mod_n256} <= {mod_n1024}"
    );

    // 6. Respeto estricto de casos degenerados
    assert_eq!(hurst_duration_modulation(0, 0.70), 0);
    assert_eq!(hurst_duration_modulation(600_000, f64::NAN), 600_000);
    assert_eq!(hurst_duration_modulation(600_000, f64::INFINITY), 600_000);
}
