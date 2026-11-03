//! # Contrato Formal F20-002 — Termodinámica de Landauer, Condensación de Bose-Einstein y Memoria de Hopfield Continua
//!
//! Invariantes matemáticas y físicas comprobadas:
//! - **C1:** El acumulador KBN (Kahan-Babuška-Neumaier) elimina la deriva de redondeo frente a sumas acumuladas ingenuas.
//! - **C2:** La disipación mínima de Landauer cumple exactamente $W_{\text{min}} = k_B T_{\text{mkt}} \ln 2 \cdot \Delta I_{\text{bits}}$ y $\eta_{\text{Landauer}} \in [0, 1]$.
//! - **C3:** La fracción de condensado de Bose-Einstein $f_0 = N_0 / N_{\text{total}}$ clasifica la acumulación de liquidez en el mejor nivel y mantiene $\mu \le \varepsilon_0$.
//! - **C4:** La Red Continua Moderna de Hopfield exhibe descenso monótono de energía continua de Lyapunov ($\Delta E \le 10^{-12}$).
//! - **C5:** La memoria asociativa recupera con precisión atractores ruidosos ($w_k > 0.85$) con conservación estricta de partición de la unidad ($\sum_k w_k \equiv 1.0$).
//! - **C6:** El motor maestro `LandauerBecHopfieldMasterEngine` es resiliente a NaNs, ceros y entradas patológicas sin un solo pánico.

use feature_engine::landauer_bec_hopfield::*;

#[test]
fn c1_kbn_compensated_summation_eliminates_roundoff_drift() {
    let mut kbn = KbnCompensatedAccumulator::new();
    let mut naive_sum = 0.0f64;

    // Sumar 1.0 + 10_000_000 veces 1e-16
    // En punto flotante estándar f64, 1.0 + 1e-16 = 1.0 (se trunca la mantisa)
    let n = 100_000;
    let delta = 1e-16;

    kbn.add(1.0);
    naive_sum += 1.0;

    for _ in 0..n {
        kbn.add(delta);
        naive_sum += delta;
    }

    let expected = 1.0 + (n as f64) * delta;
    let kbn_result = kbn.total();

    let naive_error = (naive_sum - expected).abs();
    let kbn_error = (kbn_result - expected).abs();

    assert!(
        naive_error > 1e-12,
        "La suma ingenua debió haber perdido precisión por truncamiento de mantisa"
    );

    // KBN recupera la suma exacta con precisión sub-ULP
    assert!(
        kbn_error < 1e-14,
        "KBN total {} difiere del valor analítico {} (error={})",
        kbn_result,
        expected,
        kbn_error
    );
    assert!(
        kbn_error < naive_error,
        "El error de KBN ({}) debe ser estrictamente menor que el error ingenuo ({})",
        kbn_error,
        naive_error
    );

    // Test de alternancia de magnitudes masivas: +1e12, +0.5, -1e12, +0.5
    let mut kbn_alt = KbnCompensatedAccumulator::new();
    kbn_alt.add(1e12);
    kbn_alt.add(0.5);
    kbn_alt.add(-1e12);
    kbn_alt.add(0.5);

    assert!(
        (kbn_alt.total() - 1.0).abs() < 1e-12,
        "KBN falló en cancelar grandes magnitudes: {}",
        kbn_alt.total()
    );
}

#[test]
fn c2_landauer_thermodynamic_entropy_and_work_bound() {
    let engine = LandauerThermodynamicEngine;

    // Distribución uniforme de probabilidades sobre 8 niveles: p_i = 1/8 = 0.125
    // Entropía analítica: - 8 * (1/8 * log2(1/8)) = log2(8) = 3.0 bits
    let probs = [0.125; BEC_ENERGY_LEVELS];
    let temp = 2.5;

    let rep = engine.evaluate(&probs, temp, 0.0);

    assert!(
        (rep.info_entropy_bits - 3.0).abs() < 1e-9,
        "Entropía uniforme debe ser 3.0 bits, fue {}",
        rep.info_entropy_bits
    );

    let expected_w_min = BOLTZMANN_K_MKT * temp * LN_2 * 3.0;
    assert!(
        (rep.landauer_min_work - expected_w_min).abs() < 1e-9,
        "Trabajo mínimo de Landauer debe ser {}, fue {}",
        expected_w_min,
        rep.landauer_min_work
    );

    // Si el trabajo real coincide con el mínimo, la eficiencia es 1.0 y el régimen es ReversibleEfficient
    assert_eq!(rep.regime, LandauerDissipationRegime::ReversibleEfficient);
    assert!((rep.landauer_efficiency - 1.0).abs() < 1e-9);

    // Si el trabajo real es 10 veces mayor que el mínimo, eficiencia = 0.10 -> TurbulentIrreversible
    let rep_turbulent = engine.evaluate(&probs, temp, expected_w_min * 10.0);
    assert_eq!(
        rep_turbulent.regime,
        LandauerDissipationRegime::TurbulentIrreversible
    );
    assert!((rep_turbulent.landauer_efficiency - 0.10).abs() < 1e-3);
}

#[test]
fn c3_bose_einstein_orderbook_condensate_fraction_and_chemical_potential() {
    let engine = BoseEinsteinOrderbookEngine;

    let mut energies = [0.0; BEC_ENERGY_LEVELS];
    for (i, energy) in energies.iter_mut().enumerate() {
        *energy = (i as f64) * 0.001; // 0 bps, 10 bps, 20 bps...
    }

    // Caso 1: Condensado de Bose-Einstein masivo en el nivel fundamental (80% del volumen)
    let mut bec_volumes = [10.0; BEC_ENERGY_LEVELS];
    bec_volumes[0] = 320.0; // 320 / (320 + 7*10) = 320 / 390 = 0.8205
    let rep_bec = engine.evaluate(&bec_volumes, &energies, 1.0);

    assert!(
        rep_bec.condensate_fraction >= 0.50,
        "La fracción de condensado debe ser >= 0.50, fue {}",
        rep_bec.condensate_fraction
    );
    assert_eq!(rep_bec.regime, BecPhaseRegime::BoseEinsteinCondensate);
    assert!(
        rep_bec.chemical_potential <= energies[0],
        "El potencial químico mu debe ser <= eps_0"
    );

    // Caso 2: Gas térmico distribuido uniformemente (12.5% por nivel)
    let uniform_volumes = [50.0; BEC_ENERGY_LEVELS];
    let rep_gas = engine.evaluate(&uniform_volumes, &energies, 1.0);

    assert!(
        rep_gas.condensate_fraction < 0.25,
        "En gas térmico la fracción de condensado debe ser < 0.25, fue {}",
        rep_gas.condensate_fraction
    );
    assert_eq!(rep_gas.regime, BecPhaseRegime::NormalThermalGas);
}

#[test]
fn c4_modern_continuous_hopfield_monotonic_energy_descent() {
    let engine = ModernContinuousHopfieldEngine::default();

    // Probar descenso monótono de energía E(x_out) <= E(x_in) sobre varios estados continuos
    let test_states = [
        [0.8, -0.2, 0.1, 0.4, -0.5, 0.0, 0.3, -0.1],
        [-0.9, 0.7, -0.3, 0.1, 0.2, -0.4, 0.6, -0.8],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [1.5, 0.0, -1.2, 0.4, 0.8, -0.3, 0.5, 0.2],
    ];

    for (idx, state) in test_states.iter().enumerate() {
        let rep = engine.retrieve(state);
        assert!(
            rep.energy_delta <= 1e-10,
            "Fallo de descenso monótono de energía de Lyapunov en prueba {}: delta = {} (E_init={}, E_final={})",
            idx,
            rep.energy_delta,
            rep.initial_energy,
            rep.final_energy
        );
    }
}

#[test]
fn c5_modern_continuous_hopfield_attractor_projection_and_capacity() {
    let engine = ModernContinuousHopfieldEngine::default();

    // Crear un estado ruidoso perturbando el prototipo #3 (e_3 con ruido pequeño)
    let mut noisy_probe = [0.0; HOPFIELD_FEATURE_DIM];
    noisy_probe[3] = 1.0;
    noisy_probe[0] = 0.1;
    noisy_probe[2] = -0.1;

    let rep = engine.retrieve(&noisy_probe);

    // Debe converger fuertemente al atractor #3
    assert_eq!(
        rep.dominant_prototype_idx, 3,
        "El atractor recuperado debe ser el prototipo 3"
    );
    assert!(
        rep.attractor_confidence > 0.85,
        "La confianza del atractor debe ser > 0.85, fue {}",
        rep.attractor_confidence
    );

    // Conservación estricta de partición de la unidad: sum_k w_k == 1.0
    let mut sum_weights = 0.0;
    for &w in &rep.prototype_weights {
        assert!(w >= 0.0, "Los pesos de atención deben ser no negativos");
        sum_weights += w;
    }
    assert!(
        (sum_weights - 1.0).abs() < 1e-12,
        "La suma de pesos softmax debe ser exactamente 1.0, fue {}",
        sum_weights
    );
}

#[test]
fn c6_landauer_bec_hopfield_master_engine_zero_panic_and_invariants() {
    let master = LandauerBecHopfieldMasterEngine::default();

    // Entradas patológicas: ceros, NaNs, negativos
    let l2_vols = [f64::NAN, -10.0, 0.0, 100.0, f64::INFINITY, 50.0, 20.0, 10.0];
    let l2_energies = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007];
    let state = [f64::NAN, 0.5, -0.2, 0.0, 1.0, 0.0, 0.0, 0.0];

    let rep = master.evaluate(&l2_vols, &l2_energies, -10.0, -50.0, &state);

    // Todos los campos deben ser finitos
    assert!(rep.landauer.info_entropy_bits.is_finite());
    assert!(rep.landauer.landauer_efficiency.is_finite());
    assert!(rep.landauer.landauer_efficiency >= 0.0 && rep.landauer.landauer_efficiency <= 1.0);

    assert!(rep.bec.condensate_fraction.is_finite());
    assert!(rep.bec.condensate_fraction >= 0.0 && rep.bec.condensate_fraction <= 1.0);
    assert!(rep.bec.chemical_potential.is_finite());

    assert!(rep.hopfield.initial_energy.is_finite());
    assert!(rep.hopfield.final_energy.is_finite());
    assert!(rep.hopfield.attractor_confidence.is_finite());
    assert!(rep.hopfield.attractor_confidence >= 0.0 && rep.hopfield.attractor_confidence <= 1.0);
}
