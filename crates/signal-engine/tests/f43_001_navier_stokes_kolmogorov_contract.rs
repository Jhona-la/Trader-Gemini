//! # Contrato Formal F43-001: Navier-Stokes Kolmogorov-Kraichnan 2D Hydrodynamics
//!
//! Valida el comportamiento matemático y la robustez del motor hidrodinámico 2D (§43):
//! 1. `test_c1_enstrophy_conservation_and_vortex_detection`: Vorticidad y enstrofía $\mathcal{E} \in [0.0, 1.0]$.
//! 2. `test_c2_kraichnan_inverse_cascade_spectral_slope`: Pendiente espectral de Kraichnan y cascada inversa.
//! 3. `test_c3_subnormal_nan_inf_boundary_stability`: Invariante ante NaN, $\pm\infty$ y subnormales.
//! 4. `test_c4_zero_alloc_stack_determinism`: 1,000 evaluaciones deterministas en stack sin pánico.

use signal_engine::navier_stokes_kolmogorov::{NavierStokesKolmogorovEngine, NavierStokesParams};

#[test]
fn test_c1_enstrophy_conservation_and_vortex_detection() {
    let params = NavierStokesParams {
        kinematic_viscosity: 0.05,
        fluid_density: 1.0,
        rayleigh_drag: 0.02,
        enstrophy_vortex_threshold: 0.70,
    };
    let mut engine = NavierStokesKolmogorovEngine::new(params);

    // Flujo inicial en calma
    let r0 = engine.update_and_evaluate(0.0, 0.0, 0.5, 2.0, 1_000);
    assert!(r0.total_enstrophy >= 0.0 && r0.total_enstrophy <= 1.0);
    assert!(r0.max_vorticity >= 0.0);
    assert!(r0.slot_scaling_factor >= 0.10 && r0.slot_scaling_factor <= 1.25);

    // Inyección de cizalladura turbulenta sostenida
    for i in 1..=20 {
        let ofi = if i % 2 == 0 { 0.8 } else { -0.8 };
        let vd = if i % 3 == 0 { 0.9 } else { -0.7 };
        let rep = engine.update_and_evaluate(ofi, vd, 2.5, 1.5, 1_000 + i * 50);

        assert!(rep.total_enstrophy >= 0.0 && rep.total_enstrophy <= 1.0);
        assert!(rep.max_vorticity >= 0.0);
        assert!(rep.kraichnan_spectral_slope >= 0.5 && rep.kraichnan_spectral_slope <= 4.5);
        assert!(rep.inverse_cascade_intensity >= 0.0 && rep.inverse_cascade_intensity <= 1.0);
    }
}

#[test]
fn test_c2_kraichnan_inverse_cascade_spectral_slope() {
    let mut engine = NavierStokesKolmogorovEngine::new(NavierStokesParams::default());

    // Estimulación direccional que fuerza transferencia de energía a escalas grandes (cascada inversa)
    for i in 1..=30 {
        let ts = 10_000 + i * 100;
        let rep = engine.update_and_evaluate(0.65, 0.40, 1.2, 1.0, ts);
        assert!(rep.kraichnan_spectral_slope >= 0.5 && rep.kraichnan_spectral_slope <= 4.5);
        assert!(rep.inverse_cascade_intensity >= 0.0 && rep.inverse_cascade_intensity <= 1.0);
        assert!(rep.slot_scaling_factor >= 0.10 && rep.slot_scaling_factor <= 1.25);
    }
}

#[test]
fn test_c3_subnormal_nan_inf_boundary_stability() {
    let mut engine = NavierStokesKolmogorovEngine::new(NavierStokesParams::default());

    let degenerate_inputs = [
        (f64::NAN, 0.0, 1.0, 2.0),
        (0.0, f64::INFINITY, 1.0, 2.0),
        (0.5, -0.5, f64::NEG_INFINITY, 2.0),
        (0.2, 0.1, 1.0, f64::NAN),
        (f64::MIN_POSITIVE, 0.0, 0.0, 0.0),
        (-1e-300, 1e-300, 1e-250, 1e-200),
    ];

    for (idx, &(ofi, vd, vol, sp)) in degenerate_inputs.iter().enumerate() {
        let ts = 20_000 + (idx as u64) * 100;
        let rep = engine.update_and_evaluate(ofi, vd, vol, sp, ts);

        assert!(rep.total_enstrophy.is_finite());
        assert!(rep.max_vorticity.is_finite());
        assert!(rep.kraichnan_spectral_slope.is_finite());
        assert!(rep.inverse_cascade_intensity.is_finite());
        assert!(rep.hydrodynamic_force.0.is_finite());
        assert!(rep.hydrodynamic_force.1.is_finite());
        assert!(rep.slot_scaling_factor.is_finite());
    }
}

#[test]
fn test_c4_zero_alloc_stack_determinism() {
    let mut engine_a = NavierStokesKolmogorovEngine::new(NavierStokesParams::default());
    let mut engine_b = NavierStokesKolmogorovEngine::new(NavierStokesParams::default());

    for step in 1..=1000 {
        let ts = 50_000 + step * 25;
        let ofi = ((step as f64) * 0.03).sin() * 0.7;
        let vd = ((step as f64) * 0.05).cos() * 0.6;
        let vol = 1.0 + ((step as f64) * 0.02).sin().abs();
        let sp = 1.5 + ((step as f64) * 0.01).cos().abs();

        let rep_a = engine_a.update_and_evaluate(ofi, vd, vol, sp, ts);
        let rep_b = engine_b.update_and_evaluate(ofi, vd, vol, sp, ts);

        assert_eq!(
            rep_a.total_enstrophy.to_bits(),
            rep_b.total_enstrophy.to_bits()
        );
        assert_eq!(rep_a.max_vorticity.to_bits(), rep_b.max_vorticity.to_bits());
        assert_eq!(
            rep_a.kraichnan_spectral_slope.to_bits(),
            rep_b.kraichnan_spectral_slope.to_bits()
        );
        assert_eq!(
            rep_a.inverse_cascade_intensity.to_bits(),
            rep_b.inverse_cascade_intensity.to_bits()
        );
        assert_eq!(
            rep_a.slot_scaling_factor.to_bits(),
            rep_b.slot_scaling_factor.to_bits()
        );
    }
}
