//! Algebraic contracts only: no active genome, market connection or promotion.
//!
//! LG-4 (Línea G, 2026-10-12): el dominio de la banda es la BANDA OPERABLE
//! [30 s, 12 h], no los extremos de la malla representacional (1 ns – 146
//! años). Los parámetros de los casos se eligieron para que sus raíces caigan
//! dentro de la banda operable.
use quantum_arena::genome::SuperGenotype;
use quantum_arena::temporal_spectrum::{HorizonCurve, TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS};

const FEE: f64 = 0.001;

fn genome(a: f64, b: f64) -> SuperGenotype {
    let mut g = SuperGenotype::new_baseline(0.0002, 0.0005);
    g.sl_horizon_curve = HorizonCurve { a, b };
    g
}

fn limits() -> (f64, f64) {
    (TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS)
}

fn close_log(actual: f64, expected: f64) {
    assert!(
        (actual.ln() - expected.ln()).abs() < 2e-12,
        "{actual} != {expected}"
    );
}

#[test]
fn decreasing_stop_has_an_upper_bound() {
    let g = genome(-5.0, -0.1);
    let root = ((SuperGenotype::min_viable_sl(FEE).ln() + 5.0) / -0.1).exp();
    let (lo, hi) = g.tradeable_band_ms(FEE).unwrap();
    assert_eq!(lo, limits().0);
    close_log(hi, root);
    assert!(hi < limits().1 && hi > limits().0);
}

#[test]
fn decreasing_stop_below_floor_everywhere_is_empty() {
    let g = genome(-10.5, -0.1);
    assert!(g.sl_horizon_curve.eval(limits().0) < SuperGenotype::min_viable_sl(FEE));
    assert_eq!(g.tradeable_band_ms(FEE), None);
    assert_eq!(g.min_tradeable_tau_ms(FEE), f64::INFINITY);
}

#[test]
fn increasing_stop_has_a_lower_bound() {
    let g = genome(-9.0, 0.2);
    let root = ((SuperGenotype::min_viable_sl(FEE).ln() + 9.0) / 0.2).exp();
    let (lo, hi) = g.tradeable_band_ms(FEE).unwrap();
    close_log(lo, root);
    assert_eq!(hi, limits().1);
    assert_eq!(g.min_tradeable_tau_ms(FEE), lo);
}

#[test]
fn flat_curves_are_all_or_nothing_including_equality() {
    let a = SuperGenotype::min_viable_sl(FEE).ln();
    for b in [0.0, -0.0] {
        assert_eq!(genome(a, b).tradeable_band_ms(FEE), Some(limits()));
        assert_eq!(genome(a + 0.1, b).tradeable_band_ms(FEE), Some(limits()));
        assert_eq!(genome(a - 0.1, b).tradeable_band_ms(FEE), None);
    }
}

#[test]
fn small_nonzero_slopes_are_not_replaced_by_flat_curves() {
    // Raíz en τ0 = 10⁶ ms (dentro de la banda operable). La raíz se obtiene
    // dividiendo una diferencia de orden b·ln τ0 por b: con |b| = 1e-9 el
    // redondeo de la resta deja un error de ~1e-6 en ln τ.
    let x0 = 1.0e6_f64.ln();
    for b in [-1e-9, 1e-9] {
        let a = SuperGenotype::min_viable_sl(FEE).ln() - b * x0;
        let (lo, hi) = genome(a, b).tradeable_band_ms(FEE).unwrap();
        if b > 0.0 {
            assert!((lo.ln() - x0).abs() < 1e-5, "lo={lo}");
            assert_eq!(hi, limits().1);
        } else {
            assert_eq!(lo, limits().0);
            assert!((hi.ln() - x0).abs() < 1e-5, "hi={hi}");
        }
    }
}

#[test]
fn nonfinite_coefficients_do_not_define_an_admissible_band() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for (a, b) in [(invalid, 0.0), (invalid, 0.1), (-4.0, invalid)] {
            let g = genome(a, b);
            assert_eq!(g.tradeable_band_ms(FEE), None, "a={a}, b={b}");
            assert_eq!(g.min_tradeable_tau_ms(FEE), f64::INFINITY);
        }
    }
}

#[test]
fn out_of_domain_roots_mean_full_or_empty_not_infinite_bands() {
    for (a, b, full) in [
        (-2.0, 0.01, true),
        (-2.0, -0.01, true),
        (-15.0, 0.01, false),
        (-15.0, -0.01, false),
    ] {
        let g = genome(a, b);
        assert_eq!(
            g.tradeable_band_ms(FEE),
            if full { Some(limits()) } else { None }
        );
        assert_eq!(
            g.min_tradeable_tau_ms(FEE),
            if full { limits().0 } else { f64::INFINITY }
        );
    }
}

#[test]
fn raising_the_cost_only_shrinks_the_band_for_either_slope() {
    // Raíces de FEE y de 2·FEE dentro de la banda operable para ambas pendientes.
    for (a, b) in [(-3.3, -0.2), (-8.9, 0.2)] {
        let g = genome(a, b);
        let cheap = g.tradeable_band_ms(FEE).unwrap();
        let expensive = g.tradeable_band_ms(2.0 * FEE).unwrap();
        assert!(expensive.0 >= cheap.0);
        assert!(expensive.1 <= cheap.1);
        assert!(expensive != cheap);
    }
}

#[test]
fn common_rescaling_of_stop_and_fee_preserves_the_band() {
    for (a, b) in [(-3.3, -0.2), (-8.9, 0.2)] {
        let original = genome(a, b).tradeable_band_ms(FEE).unwrap();
        for multiplier in [0.001_f64, 2.0, 1000.0] {
            let scaled = genome(a + multiplier.ln(), b)
                .tradeable_band_ms(FEE * multiplier)
                .unwrap();
            close_log(original.0, scaled.0);
            close_log(original.1, scaled.1);
        }
    }
}

#[test]
fn invalid_or_zero_fee_keeps_the_existing_reference_policy() {
    let g = genome(-6.0, -0.1);
    for fee in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        assert_eq!(
            g.tradeable_band_ms(fee),
            g.tradeable_band_ms(SuperGenotype::REFERENCE_ROUNDTRIP_FEE)
        );
    }
}

#[test]
fn band_membership_matches_the_defining_inequality_on_both_sides() {
    let (lo, hi) = limits();
    let floor = SuperGenotype::min_viable_sl(FEE).ln();
    for a in [-10.5, -9.0, -6.0, -4.0, -3.0] {
        for b in [-0.2, -0.1, 0.0, 0.1, 0.35] {
            let g = genome(a, b);
            let band = g.tradeable_band_ms(FEE);
            for i in 0..=512 {
                let tau = (lo.ln() + (hi.ln() - lo.ln()) * i as f64 / 512.0)
                    .exp()
                    .clamp(lo, hi);
                let margin = a + b * tau.ln() - floor;
                if margin.abs() < 1e-12 {
                    continue;
                } // Do not confuse root rounding with membership.
                let inside = band.is_some_and(|(lower, upper)| tau >= lower && tau <= upper);
                assert_eq!(
                    inside,
                    margin > 0.0,
                    "a={a}, b={b}, tau={tau}, band={band:?}"
                );
            }
        }
    }
}

/// LG-4 — la reparación RR no toca un genoma que ya cumple la invariante en
/// la banda operable. Antes, con pendiente de SL positiva, el borde inferior
/// de la banda era la raíz del piso de fricción en nanosegundos y la
/// reparación subía el intercepto del TP ×1,68 en todo el espectro al cargar
/// (curvas del genoma activo de demo/prod a la fecha del hallazgo).
#[test]
fn lg4_un_genoma_que_cumple_en_la_banda_operable_no_se_repara() {
    let mut g = SuperGenotype::new_baseline(0.0002, 0.0005);
    g.tp_horizon_curve = HorizonCurve { a: -4.9600, b: 0.09144 };
    g.sl_horizon_curve = HorizonCurve { a: -5.7194, b: 0.08644 };
    let fee = SuperGenotype::REFERENCE_ROUNDTRIP_FEE;
    let (lo, hi) = g.tradeable_band_ms(fee).expect("banda operable no vacía");
    assert!(lo >= TAU_ANCHOR_FAST_MS && hi <= TAU_ANCHOR_SLOW_MS, "({lo}, {hi})");
    for tau in [lo, hi] {
        let tp = g.tp_horizon_curve.eval(tau);
        let sl = g.sl_horizon_curve.eval(tau);
        assert!(
            tp >= sl * SuperGenotype::min_rr_for(SuperGenotype::WORST_TOLERATED_WR, fee, sl) * 1.10,
            "precondición: cumple con holgura en tau={tau}"
        );
    }
    let a_tp = g.tp_horizon_curve.a;
    let a_sl = g.sl_horizon_curve.a;
    g.enforce_curve_rr();
    assert!((g.tp_horizon_curve.a - a_tp).abs() < 1e-12, "TP reparado: {} → {}", a_tp, g.tp_horizon_curve.a);
    assert!((g.sl_horizon_curve.a - a_sl).abs() < 1e-12, "SL reparado: {} → {}", a_sl, g.sl_horizon_curve.a);
}
