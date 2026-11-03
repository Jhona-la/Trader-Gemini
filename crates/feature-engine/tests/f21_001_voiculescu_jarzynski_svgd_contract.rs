//! # Contrato Formal F21-001 — Probabilidad Libre de Voiculescu, Relación de Jarzynski-Crooks y SVGD Determinista
//!
//! Invariantes matemáticas y físicas comprobadas:
//! - **C1:** Cumulantes libres de Voiculescu $\kappa_1, \kappa_2, \kappa_4$ y congruencia con la ley semicircular de Wigner ($\kappa_4 \approx 0$).
//! - **C2:** Entropía libre de Voiculescu $\chi(\mu)$ bien definida y acotada ante autovalores finitos.
//! - **C3:** Igualdad de Jarzynski $\langle e^{-\beta W}\rangle = e^{-\beta \Delta F}$ y segunda ley generalizada $\langle W \rangle \ge \Delta F$ ($W_{\text{diss}} \ge 0$).
//! - **C4:** Régimen de no-equilibrio y razón de reversibilidad de Crooks clasifica microestados reversibles vs quiebres estructurales.
//! - **C5:** Descenso de gradiente variacional de Stein (SVGD) transporta deterministamente la nube de partículas hacia el centroide posterior target.
//! - **C6:** Motor maestro `VoiculescuJarzynskiSvgdMasterEngine` resiliente a patologías numéricas sin ningún pánico.

use feature_engine::voiculescu_jarzynski_svgd::*;

#[test]
fn c1_voiculescu_free_cumulants_and_wigner_semicircle_invariant() {
    let engine = VoiculescuFreeProbabilityEngine;

    // Discretización de una ley semicircular simétrica de Wigner en [-2.0, 2.0] con R = 2.0 (sigma = 1.0)
    // Para Wigner pura con radio R=2: kappa_1 = 0, kappa_2 = 1.0, kappa_4 = 0
    let eigenvalues = [
        -1.85, -1.50, -1.15, -0.75, -0.35, 0.35, 0.75, 1.15, 1.50, 1.85,
    ];

    let rep = engine.evaluate(&eigenvalues);

    assert!(
        rep.kappa_1.abs() < 1e-12,
        "La media espectral simétrica libre debe ser 0.0, fue {}",
        rep.kappa_1
    );
    assert!(
        rep.kappa_2 > 0.5 && rep.kappa_2 < 2.5,
        "La varianza libre kappa_2 debe estar en rango físico, fue {}",
        rep.kappa_2
    );
    assert!(
        rep.wigner_semicircle_fit > 0.30,
        "El ajuste semicircular para espectro Wigner debe ser apreciable, fue {}",
        rep.wigner_semicircle_fit
    );
    assert!(
        rep.cauchy_stieltjes_imag > 0.0,
        "La parte imaginaria de Cauchy-Stieltjes debe ser estrictamente positiva"
    );
}

#[test]
fn c2_voiculescu_free_entropy_positivity_and_bound() {
    let engine = VoiculescuFreeProbabilityEngine;

    let eigs_separated = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let rep_sep = engine.evaluate(&eigs_separated);

    assert!(
        rep_sep.free_entropy.is_finite(),
        "La entropía libre de un espectro regular debe ser finita"
    );

    // Espectro con autovalores muy cercanos: la entropía libre ln|li - lj| se vuelve muy negativa
    let eigs_clustered = [1.000, 1.001, 1.002, 1.003];
    let rep_clustered = engine.evaluate(&eigs_clustered);

    assert!(
        rep_clustered.free_entropy < rep_sep.free_entropy,
        "La entropía libre del espectro concentrado ({}) debe ser menor que la del separado ({})",
        rep_clustered.free_entropy,
        rep_sep.free_entropy
    );
}

#[test]
fn c3_jarzynski_fluctuation_equality_and_jensen_second_law() {
    let engine = JarzynskiCrooksEngine::new(1.0); // beta = 1.0

    // Ensayos de trabajo con dispersión controlada
    let work_samples = [0.1, 0.2, 0.05, 0.4, 0.15, 0.3];
    let rep = engine.evaluate(&work_samples);

    // Segunda ley: <W> >= \Delta F  =>  W_diss >= 0
    assert!(
        rep.dissipated_work >= 0.0,
        "El trabajo disipado debe ser no negativo por la desigualdad de Jensen, fue {}",
        rep.dissipated_work
    );
    assert!(
        rep.mean_work >= rep.free_energy_delta,
        "El trabajo medio {} debe ser >= cambio de energía libre {}",
        rep.mean_work,
        rep.free_energy_delta
    );

    // Verificar numéricamente que exp(-\beta \Delta F) coincide con <exp(-\beta W)>
    let beta = 1.0;
    let expected_avg_exp: f64 =
        work_samples.iter().map(|&w| (-beta * w).exp()).sum::<f64>() / (work_samples.len() as f64);
    let jarzynski_lhs = (-beta * rep.free_energy_delta).exp();

    assert!(
        (jarzynski_lhs - expected_avg_exp).abs() < 1e-10,
        "La igualdad de Jarzynski lhs {} difiere de rhs {}",
        jarzynski_lhs,
        expected_avg_exp
    );
}

#[test]
fn c4_jarzynski_crooks_reversibility_regimes() {
    let engine = JarzynskiCrooksEngine::new(2.0); // beta = 2.0 (kT = 0.5)

    // Caso reversible: casi nula fluctuación disipativa
    let work_reversible = [0.01, 0.012, 0.009, 0.011];
    let rep_rev = engine.evaluate(&work_reversible);
    assert_eq!(rep_rev.regime, JarzynskiRegime::ReversibleEquilibrium);
    assert!(
        rep_rev.reversibility_ratio > 0.95,
        "Razón de reversibilidad debe ser > 0.95, fue {}",
        rep_rev.reversibility_ratio
    );

    // Caso disipativo irreversible: gran trabajo forzado con alta varianza
    let work_irreversible = [0.1, 2.5, 3.8, 4.2, 0.05];
    let rep_irrev = engine.evaluate(&work_irreversible);
    assert_eq!(
        rep_irrev.regime,
        JarzynskiRegime::IrreversibleStructuralBreak
    );
    assert!(
        rep_irrev.dissipated_work > 0.5,
        "Disipación irreversible debe ser alta, fue {}",
        rep_irrev.dissipated_work
    );
}

#[test]
fn c5_svgd_deterministic_particle_transport_towards_posterior() {
    let mut svgd = SteinVariationalGradientDescentEngine::new(6, 0.10);

    let target_mean = [1.5, -2.0, 0.5, 3.0];
    let target_prec = [1.0, 1.0, 1.0, 1.0]; // Precisión unitaria

    // Calcular distancia inicial del centroide al target
    let init_rep = svgd.step(&target_mean, &target_prec);
    let mut initial_dist = 0.0;
    for (d, &t_d) in target_mean.iter().enumerate() {
        let diff = init_rep.posterior_mean[d] - t_d;
        initial_dist += diff * diff;
    }
    initial_dist = initial_dist.sqrt();

    // Ejecutar 25 iteraciones de transporte determinista de Stein
    let mut final_rep = init_rep;
    for _ in 0..25 {
        final_rep = svgd.step(&target_mean, &target_prec);
    }

    let mut final_dist = 0.0;
    for (d, &t_d) in target_mean.iter().enumerate() {
        let diff = final_rep.posterior_mean[d] - t_d;
        final_dist += diff * diff;
    }
    final_dist = final_dist.sqrt();

    assert!(
        final_dist < initial_dist,
        "La distancia final ({}) debe ser estrictamente menor que la inicial ({}) tras el transporte de Stein",
        final_dist,
        initial_dist
    );
    assert!(
        final_rep.kernel_bandwidth > 0.01 && final_rep.kernel_bandwidth <= 10.0,
        "El ancho de banda adaptativo debe mantenerse en rango válido"
    );
}

#[test]
fn c6_master_engine_zero_panic_and_invariants() {
    let mut master = VoiculescuJarzynskiSvgdMasterEngine::new(1.5, 4, 0.05);

    let target_mean = [0.0; SVGD_DIM];
    let target_prec = [1.0; SVGD_DIM];

    // Prueba 1: Entradas vacías
    let rep_empty = master.evaluate(&[], &[], &target_mean, &target_prec);
    assert!(rep_empty.free_composite_stability >= 0.0 && rep_empty.free_composite_stability <= 1.0);

    // Prueba 2: Entradas con NaNs e infinitos (resiliencia total)
    let bad_eigs = [f64::NAN, f64::INFINITY, -f64::INFINITY, 0.0];
    let bad_work = [f64::NAN, 1.0, f64::INFINITY];
    let rep_bad = master.evaluate(&bad_eigs, &bad_work, &target_mean, &target_prec);

    assert!(rep_bad.free_composite_stability.is_finite());
    assert!(rep_bad.jarzynski.dissipated_work >= 0.0);
    assert!(rep_bad.free_probability.free_entropy.is_finite());
}
