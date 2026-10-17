//! Contrato de Certificación Ola Ω137:
//! Descomposición de Helmholtz-Hodge (L² = ∇φ ⊕ ∇×A ⊕ h) en Stack Zero-Alloc (<25 µs).

use dark_alpha_engine::helmholtz_hodge::{HelmholtzHodgeEngine, HelmholtzHodgeParams};
use std::time::Instant;

#[test]
fn test_helmholtz_laminar_flow_detection() {
    let mut engine = HelmholtzHodgeEngine::new(HelmholtzHodgeParams::default());

    // Simular flujo laminar puro: precio y OFI se desplazan monótonamente en la misma dirección (momentum continuo)
    for i in 0..25 {
        let p_vel = 2.5 + (i as f64) * 0.1;
        let ofi = 0.85;
        let decomp = engine.update(p_vel, ofi);

        if i > 15 {
            assert!(
                decomp.is_laminar_safe,
                "Flujo direccional monótono debe ser calificado como laminar seguro: {:?}",
                decomp
            );
            assert!(
                decomp.laminar_fraction > 0.30,
                "Fracción laminar debe ser sustancial: {}",
                decomp.laminar_fraction
            );
            assert!(
                decomp.conviction_factor >= 1.0,
                "Factor de convicción debe ser >= 1.0: {}",
                decomp.conviction_factor
            );
        }
    }
}

#[test]
fn test_helmholtz_vortical_spoofing_detection() {
    let mut engine = HelmholtzHodgeEngine::new(HelmholtzHodgeParams {
        enstrophy_threshold: 0.50,
        min_laminar_fraction: 0.40,
        ..Default::default()
    });

    // Simular spoofing/vorticidad rotacional: cancelaciones masivas y oscilaciones de libro sin avance de precio
    for i in 0..30 {
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let p_vel = 0.05 * sign; // Precio plano
        let ofi = 0.95 * sign;   // Libro oscilando agresivamente (wash/spoof)
        let decomp = engine.update(p_vel, ofi);

        if i > 20 {
            assert!(
                decomp.enstrophy > 0.10,
                "La enstrofía rotacional debe capturar la vorticidad de spoofing: {}",
                decomp.enstrophy
            );
        }
    }
}

#[test]
fn test_helmholtz_zero_alloc_latency_budget() {
    let mut engine = HelmholtzHodgeEngine::new(HelmholtzHodgeParams::default());

    // Pre-cargar historial
    for _i in 0..10 {
        engine.update(1.0, 0.5);
    }

    let iterations = 20_000;
    let start = Instant::now();
    for i in 0..iterations {
        let p = if i % 2 == 0 { 1.2 } else { -1.1 };
        let decomp = engine.update(p, 0.4);
        assert!(decomp.conviction_factor.is_finite());
    }
    let elapsed = start.elapsed();
    let per_update_ns = elapsed.as_nanos() as f64 / iterations as f64;

    println!("⚡ Latencia Descomposición Helmholtz-Hodge: {:.2} ns / actualización", per_update_ns);
    // Mandato estricto: ejecución en nanosegundos / microsegundos (<25 µs = 25,000 ns)
    assert!(
        per_update_ns < 25_000.0,
        "La descomposición debe ejecutarse en nanosegundos (<25 µs): {:.2} ns",
        per_update_ns
    );
}

#[test]
fn test_helmholtz_nan_inf_immunity() {
    let mut engine = HelmholtzHodgeEngine::new(HelmholtzHodgeParams::default());

    let decomp_nan = engine.update(f64::NAN, 0.5);
    assert!(decomp_nan.total_energy.is_finite());
    assert!(decomp_nan.conviction_factor.is_finite());

    let decomp_inf = engine.update(1.5, f64::INFINITY);
    assert!(decomp_inf.total_energy.is_finite());
    assert!(decomp_inf.enstrophy.is_finite());
}

#[test]
fn test_g8_omega_07_c1_continuity_and_scale_invariance() {
    let mut engine_small = HelmholtzHodgeEngine::new(HelmholtzHodgeParams::default());
    let mut engine_large = HelmholtzHodgeEngine::new(HelmholtzHodgeParams::default());

    for i in 0..24 {
        let base_vel = 2.0 + (i as f64) * 0.15;
        let ofi = 0.80;
        let d_small = engine_small.update(base_vel, ofi);
        let d_large = engine_large.update(base_vel * 50.0, ofi);
        if i >= 16 {
            assert!(
                (d_small.laminar_fraction - d_large.laminar_fraction).abs() < 0.08,
                "La fracción laminar debe ser invariante de escala tras normalización RMS: small={} vs large={}",
                d_small.laminar_fraction,
                d_large.laminar_fraction
            );
            assert!(
                d_small.conviction_factor >= 1.0 && d_large.conviction_factor >= 1.0,
                "Ambos regímenes laminares deben preservar convicción >= 1.0: small={} vs large={}",
                d_small.conviction_factor,
                d_large.conviction_factor
            );
        }
    }

    // Verificar ausencia del acantilado C⁰ de magnitud 0.90 en min_laminar_fraction = 0.35
    let mut e_below = HelmholtzHodgeEngine::new(HelmholtzHodgeParams {
        min_laminar_fraction: 0.35,
        ..Default::default()
    });
    let mut e_above = HelmholtzHodgeEngine::new(HelmholtzHodgeParams {
        min_laminar_fraction: 0.3499,
        ..Default::default()
    });
    for i in 0..16 {
        let p = if i % 3 == 0 { 1.0 } else { 0.4 };
        let o = if i % 2 == 0 { 0.6 } else { -0.2 };
        let db = e_below.update(p, o);
        let da = e_above.update(p, o);
        assert!(
            (db.conviction_factor - da.conviction_factor).abs() < 0.01,
            "Proximidad de min_laminar_fraction no debe causar salto C⁰: {} vs {}",
            db.conviction_factor,
            da.conviction_factor
        );
    }
}

