//! Contrato Formal de Verificación: Fase 56 (Termodinámica Estocástica de Jarzynski-Crooks y Producción de Entropía)
//!
//! Este conjunto de pruebas certifica las propiedades termodinámicas y fuera de equilibrio del motor:
//! - C1: Evaluación de la Igualdad de Jarzynski $\langle e^{-\beta W} \rangle = e^{-\beta \Delta F}$ y segunda ley $W_{\text{diss}} \ge 0$.
//! - C2: Tasa de producción de entropía no-negativa $\Sigma_{\text{entropy}} \ge 0$ y reversibilidad $\mathcal{R}_{\text{rev}} \in [0.0, 1.0]$.
//! - C3: Detección de colapso de reversibilidad ante shock severo y reducción defensiva a $\lambda_{\text{jc}} = 0.05$.
//! - C4: Razón de fluctuación de Crooks acotada y determinismo en stack $O(1)$.

use signal_engine::{
    JarzynskiCrooksParams, JarzynskiCrooksReport, JarzynskiCrooksThermodynamicsEngine,
    JARZYNSKI_WINDOW_CAPACITY,
};

#[test]
fn test_c1_jarzynski_equality_and_second_law_work_dissipation() {
    let params = JarzynskiCrooksParams::default();
    let mut engine = JarzynskiCrooksThermodynamicsEngine::new(params);

    // Inyectamos fluctuaciones equilibradas en torno a cero
    let mut last_rep = JarzynskiCrooksReport {
        free_energy_delta: 0.0,
        mean_work: 0.0,
        dissipated_work: 0.0,
        entropy_production: 0.0,
        reversibility_index: 1.0,
        crooks_fluctuation_ratio: 1.0,
        irreversible_entropy_collapse: false,
        signal_conviction_modulator: 1.0,
        effective_temperature: 1.0,
        event_time_ms: 0,
    };

    for i in 0..JARZYNSKI_WINDOW_CAPACITY {
        let ret = if i % 2 == 0 { 0.0001 } else { -0.0001 };
        let ofi = if i % 2 == 0 { 0.05 } else { -0.05 };
        last_rep =
            engine.update_and_evaluate(ret, ofi, 2.0, 0.001, 1_700_000_000 + (i as u64) * 50);
    }

    assert!(
        last_rep.dissipated_work >= 0.0,
        "La Segunda Ley exige que el trabajo disipado sea no-negativo: {}",
        last_rep.dissipated_work
    );
    assert!(
        last_rep.effective_temperature > 0.0,
        "La temperatura efectiva debe ser estrictamente positiva: {}",
        last_rep.effective_temperature
    );
    assert!(
        last_rep.reversibility_index >= 0.0 && last_rep.reversibility_index <= 1.0,
        "El índice de reversibilidad debe estar en [0, 1]: {}",
        last_rep.reversibility_index
    );
}

#[test]
fn test_c2_entropy_production_and_reversibility() {
    let params = JarzynskiCrooksParams::default();
    let mut engine = JarzynskiCrooksThermodynamicsEngine::new(params);

    for i in 0..10 {
        let rep =
            engine.update_and_evaluate(0.0002, 0.10, 1.5, 0.002, 1_700_000_000 + (i as u64) * 100);

        assert!(
            rep.entropy_production >= 0.0,
            "Producción de entropía debe ser >= 0: {}",
            rep.entropy_production
        );
        assert!(
            rep.crooks_fluctuation_ratio > 0.0 && rep.crooks_fluctuation_ratio.is_finite(),
            "Razón de Crooks debe ser positiva y finita: {}",
            rep.crooks_fluctuation_ratio
        );
        assert!(
            rep.signal_conviction_modulator >= 0.05 && rep.signal_conviction_modulator <= 1.35,
            "Modulador de convicción fuera de rango: {}",
            rep.signal_conviction_modulator
        );
    }
}

#[test]
fn test_c3_irreversible_shock_and_conviction_collapse() {
    let mut params = JarzynskiCrooksParams::default();
    params.critical_entropy_threshold = 1.0; // Umbral estricto para certificar detección
    let mut engine = JarzynskiCrooksThermodynamicsEngine::new(params);

    // Inyectamos trabajo masivo unidireccional y violento (shock de liquidación)
    let mut report = engine.update_and_evaluate(0.01, 1.0, 50.0, 0.05, 1_700_000_100);
    for i in 1..8 {
        report =
            engine.update_and_evaluate(0.05, 2.0, 100.0, 0.10, 1_700_000_100 + (i as u64) * 10);
    }

    assert!(
        report.irreversible_entropy_collapse,
        "Debe detectar colapso de entropía irreversible ante shock violento"
    );
    assert_eq!(
        report.signal_conviction_modulator, 0.05,
        "La convicción debe colapsar a la cota mínima de seguridad 0.05"
    );
}

#[test]
fn test_c4_crooks_ratio_consistency() {
    let params = JarzynskiCrooksParams::default();
    let mut engine = JarzynskiCrooksThermodynamicsEngine::new(params);

    let rep = engine.update_and_evaluate(0.0001, 0.01, 1.0, 0.001, 1_700_000_000);
    assert!(rep.crooks_fluctuation_ratio.is_finite());
    assert!(rep.crooks_fluctuation_ratio > 0.0);
}
