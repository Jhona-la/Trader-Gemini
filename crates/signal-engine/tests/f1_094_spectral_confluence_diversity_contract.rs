//! Contrato de Diversidad Espectral y Acoplamiento OBI/ML — Ola Ω197: F1-094
//!
//! # QUÉ
//! Certifica que `FlowExcitationConfluenceEngine::voto_espectral_con_obi_ml_wasserstein_y_fisica`:
//! 1. Incorpora Order Book Imbalance (OBI) y Lift Direccional de ML en el flujo base confluente.
//! 2. Erradica el comportamiento de clon monótono respecto a `HawkesBessel` y `FlowImpulse`.
//! 3. Muestra divergencia de signo (`sign(voto_confluence) != sign(voto_hawkes)`) en micro-escalas
//!    ante desbalances del libro y en macro-escalas ante predicción ML en contra del momentum.
//! 4. Conserva compatibilidad exacta con el camino legado cuando OBI y ML son neutros.
//! 5. Es inmune a entradas no finitas (NaN, ±∞) sin pánicos.
//!
//! # POR QUÉ (F1-094)
//! Con corrientes neutras, `voto_confluence` reducía a `tanh(x) * excit`, siendo idéntico en signo
//! y e-proceso de Ville a `HawkesBessel` y `FlowImpulse`. Esto triplicaba artificialmente el peso
//! de una misma hipótesis de momentum e ignoraba el OBI y la calibración ML en el consenso espectral.

use signal_engine::flow_excitation_confluence::FlowExcitationConfluenceEngine;
use signal_engine::hawkes_bessel::HawkesBesselEngine;
use signal_engine::flow_impulse::FlowImpulseEngine;

#[test]
fn f1_094_control_backward_compatibility_when_neutral() {
    let mut desplazamientos = [0.0f64; 32];
    for k in 0..32 {
        desplazamientos[k] = (k as f64 - 15.5) * 0.1;
    }

    let ve_legacy =
        FlowExcitationConfluenceEngine::voto_espectral_con_wasserstein_dirac_bsd_riemann_kahler_gauge_y_helmholtz(
            &desplazamientos,
            2.2,
            0.15,
            0.95,
            0.20,
            -0.10,
            1.05,
            0.10,
            1.02,
            -0.15,
            1.08,
            0.05,
            1.00,
        );

    let ve_con_obi_ml =
        FlowExcitationConfluenceEngine::voto_espectral_con_obi_ml_wasserstein_y_fisica(
            &desplazamientos,
            2.2,
            0.15,
            0.95,
            0.20,
            -0.10,
            1.05,
            0.10,
            1.02,
            -0.15,
            1.08,
            0.05,
            1.00,
            0.0,
            0.0,
        );

    for k in 0..32 {
        let diff = (ve_legacy.en_escala(k) - ve_con_obi_ml.en_escala(k)).abs();
        assert!(
            diff < 1e-12,
            "Scale {} mismatch: legacy={} vs new_with_neutral={}",
            k,
            ve_legacy.en_escala(k),
            ve_con_obi_ml.en_escala(k)
        );
    }
}

#[test]
fn f1_094_micro_scale_obi_domination_and_anti_clone_diversity() {
    let desplazamientos = [0.15f64; 32]; // Leve momentum alcista en todas las escalas
    let ratio_hawkes = 2.5; // Excitación activa

    let ve_hawkes = HawkesBesselEngine::voto_espectral(&desplazamientos, ratio_hawkes);
    let ve_impulse = FlowImpulseEngine::voto_espectral(&desplazamientos, ratio_hawkes);

    // Muro de venta masivo en el libro (OBI = -0.80)
    let ve_confluence =
        FlowExcitationConfluenceEngine::voto_espectral_con_obi_ml_wasserstein_y_fisica(
            &desplazamientos,
            ratio_hawkes,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            -0.80, // Fuerte desbalance vendedor
            0.0,
        );

    // En micro-escala k=0, Hawkes e Impulse votan alcistas siguiendo el momentum leve
    assert!(ve_hawkes.en_escala(0) > 0.0, "Hawkes debe votar alcista");
    assert!(ve_impulse.en_escala(0) > 0.0, "Impulse debe votar alcista");

    // Confluence en k=0 detecta la absorción/muro y vota BAJISTA (signo opuesto)
    assert!(
        ve_confluence.en_escala(0) < 0.0,
        "Confluence en k=0 debe votar bajista ante muro de venta (OBI=-0.80): voto={}",
        ve_confluence.en_escala(0)
    );
    assert_ne!(
        ve_confluence.en_escala(0).signum(),
        ve_hawkes.en_escala(0).signum(),
        "Confluence y Hawkes deben diferir en signo ante libro opuesto a momentum"
    );
}

#[test]
fn f1_094_macro_scale_ml_lift_domination_and_anti_clone_diversity() {
    let desplazamientos = [0.15f64; 32];
    let ratio_hawkes = 2.5;

    let ve_hawkes = HawkesBesselEngine::voto_espectral(&desplazamientos, ratio_hawkes);

    // Predicción ML con fuerte lift bajista en macro (ml_directional_lift = -1.5)
    let ve_confluence =
        FlowExcitationConfluenceEngine::voto_espectral_con_obi_ml_wasserstein_y_fisica(
            &desplazamientos,
            ratio_hawkes,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            -1.50, // Fuerte lift predictivo bajista
        );

    // En macro-escala k=30, Hawkes vota alcista
    assert!(ve_hawkes.en_escala(30) > 0.0);

    // Confluence en k=30 incorpora el lift ML y vota BAJISTA
    assert!(
        ve_confluence.en_escala(30) < 0.0,
        "Confluence en k=30 debe votar bajista guiado por ML lift (-1.50): voto={}",
        ve_confluence.en_escala(30)
    );
    assert_ne!(
        ve_confluence.en_escala(30).signum(),
        ve_hawkes.en_escala(30).signum(),
        "Confluence y Hawkes deben diferir en signo en macro ante ML adverso"
    );
}

#[test]
fn f1_094_anti_clone_diversity_fraction_across_spectrum() {
    let mut desplazamientos = [0.0f64; 32];
    for k in 0..32 {
        // Momentum que oscila suavemente en torno a +0.20
        desplazamientos[k] = 0.20 + 0.05 * (k as f64 * 0.5).sin();
    }
    let ratio_hawkes = 2.0;

    let ve_hawkes = HawkesBesselEngine::voto_espectral(&desplazamientos, ratio_hawkes);

    // Condición real de mercado: libro vendedor (OBI = -0.70) y señal ML corta (-1.20)
    let ve_confluence =
        FlowExcitationConfluenceEngine::voto_espectral_con_obi_ml_wasserstein_y_fisica(
            &desplazamientos,
            ratio_hawkes,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            -0.70,
            -1.20,
        );

    let mut divergencias = 0usize;
    for k in 0..32 {
        let s_conf = ve_confluence.en_escala(k);
        let s_hawk = ve_hawkes.en_escala(k);
        if s_conf * s_hawk < 0.0 {
            divergencias += 1;
        }
    }

    // Al menos 25% de las escalas deben diferir en signo (erradicación demostrada de clones)
    assert!(
        divergencias >= 8,
        "Divergencia de signo insuficiente entre Confluence y Hawkes: divergencias={}/32",
        divergencias
    );
}

#[test]
fn f1_094_non_finite_immunity() {
    let nan_displacements = [f64::NAN; 32];
    let ve_nan = FlowExcitationConfluenceEngine::voto_espectral_con_obi_ml_wasserstein_y_fisica(
        &nan_displacements,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        f64::NAN,
    );

    for k in 0..32 {
        assert_eq!(
            ve_nan.en_escala(k),
            0.0,
            "Non-finite inputs must fail closed to 0.0 at scale {}",
            k
        );
    }
}
