//! ST audit: wrapper contracts and limitations, separate from algebraic unit tests.
use feature_engine::path_signatures::{
    firma_nivel2_incrementos, firma_ventana_logprecio, Signature2,
};

fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-12, "{a:.17e} != {b:.17e}");
}

fn same(a: Signature2, b: Signature2) {
    for (x, y) in a.to_features().into_iter().zip(b.to_features()) {
        close(x, y);
    }
}

#[test]
fn window_time_is_a_fraction_of_elapsed_duration() {
    let s = firma_ventana_logprecio(&[100.0, 101.0, 100.0], &[1000, 2000, 3000]).unwrap();
    close(s.level1[1], 1.0);
    close(s.level2[1][1], 0.5);
}

#[test]
fn moving_the_epoch_does_not_change_window_features() {
    let p = [100.0, 102.0, 99.0, 103.0];
    let a = firma_ventana_logprecio(&p, &[1000, 1100, 1900, 3000]).unwrap();
    let b = firma_ventana_logprecio(&p, &[1_001_000, 1_001_100, 1_001_900, 1_003_000]).unwrap();
    same(a, b);
}

#[test]
fn backward_internal_clock_is_rejected_without_sorting_prices() {
    assert!(firma_ventana_logprecio(&[100.0, 110.0, 105.0], &[1000, 900, 2000]).is_none());
}

#[test]
fn integer_elapsed_time_survives_absolute_epoch_above_f64_precision() {
    let t0 = 1_u64 << 53;
    let s = firma_ventana_logprecio(&[100.0, 101.0], &[t0, t0 + 1])
        .expect("one millisecond is nonzero before conversion to f64");
    close(s.level1[1], 1.0);
}

#[test]
fn adjacent_representable_prices_keep_their_log_return() {
    let p = 1e300_f64;
    let q = f64::from_bits(p.to_bits() + 1);
    let expected = ((q - p) / p).ln_1p();
    let actual = firma_ventana_logprecio(&[p, q], &[0, 1]).unwrap().level1[0];
    assert!(
        actual > 0.0,
        "representable positive return disappeared: {actual}"
    );
    assert!((actual - expected).abs() <= expected * 1e-12);
}

#[test]
fn equal_timestamps_preserve_event_order_when_total_span_is_positive() {
    let actual = firma_ventana_logprecio(&[1.0, 2.0, 4.0], &[0, 0, 10]).unwrap();
    let expected = firma_nivel2_incrementos(&[(2.0_f64.ln(), 0.0), (2.0_f64.ln(), 1.0)]);
    same(actual, expected);
}

#[test]
fn large_price_declines_do_not_lose_ratio_precision_near_minus_one() {
    // ln_1p((q-p)/p) is ill-conditioned when subtraction rounds towards -1.
    let expected = 1e-10_f64.ln();
    let actual = firma_ventana_logprecio(&[1.0, 1e-10], &[0, 1]).unwrap();
    close(actual.level1[0], expected);
}

#[test]
fn extreme_positive_prices_do_not_require_a_representable_price_ratio() {
    let tiny = f64::from_bits(1);
    for (a, b) in [(tiny, f64::MAX), (f64::MAX, tiny)] {
        let actual = firma_ventana_logprecio(&[a, b], &[0, 1]).unwrap();
        assert!(actual.to_features().into_iter().all(f64::is_finite));
        close(actual.level1[0], b.ln() - a.ln());
    }
}

#[test]
fn normalized_features_alone_do_not_identify_absolute_horizon() {
    // Diagnostic limitation, not evidence of a runtime integration.
    let p = [100.0, 102.0, 101.0];
    same(
        firma_ventana_logprecio(&p, &[0, 1, 2]).unwrap(),
        firma_ventana_logprecio(&p, &[0, 1_000_000, 2_000_000]).unwrap(),
    );
}

#[test]
fn level_two_collides_for_distinct_strictly_time_augmented_paths() {
    let oscillating = [(1.0, 0.25), (-1.0, 0.25), (-1.0, 0.25), (1.0, 0.25)];
    let flat = [(0.0, 1.0)];
    same(
        firma_nivel2_incrementos(&oscillating),
        firma_nivel2_incrementos(&flat),
    );
    // S^{x,x,t}=1/2 integral x(t)^2 dt distinguishes them at level three.
    let xxt = |incs: &[(f64, f64)]| {
        let mut x = 0.0;
        let mut total = 0.0;
        for &(dx, dt) in incs {
            total += 0.5 * dt * (x * x + x * dx + dx * dx / 3.0);
            x += dx;
        }
        total
    };
    close(xxt(&oscillating), 1.0 / 6.0);
    close(xxt(&flat), 0.0);
}

#[test]
fn unchecked_kernel_is_not_a_nonfinite_input_validator() {
    // This test records a PRECONDITION of the raw API, not a repaired behavior.
    for increments in [[(f64::MAX, 1.0)], [(f64::NAN, 1.0)]] {
        assert!(firma_nivel2_incrementos(&increments)
            .to_features()
            .into_iter()
            .any(|x| !x.is_finite()));
    }
}

#[test]
fn subdividing_each_segment_preserves_the_same_geometric_path() {
    let coarse = [(0.5, 0.25), (-0.25, 0.5), (0.125, 0.25)];
    let mut fine = Vec::new();
    for &(dx, dt) in &coarse {
        fine.extend([(dx / 4.0, dt / 4.0); 4]);
    }
    same(
        firma_nivel2_incrementos(&coarse),
        firma_nivel2_incrementos(&fine),
    );
}

#[test]
fn shuffle_identity_exposes_redundant_level_two_coordinates() {
    let s = firma_nivel2_incrementos(&[(0.4, 0.3), (-0.1, 0.2), (0.2, 0.5)]);
    for i in 0..2 {
        for j in 0..2 {
            close(s.level2[i][j] + s.level2[j][i], s.level1[i] * s.level1[j]);
        }
    }
}

/// GLM 162 — RECTIFICACIÓN del 161 + diseño del nulo unieje, blindado
/// algebraicamente. La identidad de permutación de PARES verdadera es:
/// S¹, S², S¹¹, S²² y la SUMA S¹²+S²¹ = S¹·S² se conservan EXACTO (función
/// de ΣΔ¹, ΣΔ², Σ(Δ¹)², Σ(Δ²)²); los cross INDIVIDUALES S¹² y S²¹ SÍ
/// cambian por ventana (~O(‖r‖·‖d‖)) — el "Δfirma≡0 EXACTO" del 161 fue
/// sobre-redondeo de MEDIAS ±1e-5, no identidad de los términos sueltos.
/// El nulo unieje (barajar SÓLO log-retornos, dt fijos) rompe además la
/// suma parcial E[Σ_{a<b} r_a d_b] bajo permutación ⇒ cross con sesgo
/// distinto. Contrato: (i) invariante exacta de los 5 términos bajo
/// permutación de pares; (ii) cross individuales se mueven bajo ambos
/// nulos; (iii) caso degenerado VERDADERO: retornos idénticos BAJO NULO
/// UNIEJE (dt no se tocan ⇒ firma idéntica).
#[test]
fn nulo_unieje_y_rectificacion_identidad_pares() {
    let incs: Vec<(f64, f64)> = vec![
        (0.30, 0.05), (-0.20, 0.10), (0.15, 0.05), (0.05, 0.30),
        (-0.10, 0.20), (0.25, 0.15), (-0.05, 0.10), (0.10, 0.05),
    ];
    let original = firma_nivel2_incrementos(&incs);

    // (i) Permutación de PARES (rotación fija de pares completos):
    // S¹, S², S¹¹, S²² y S¹²+S²¹ EXACTO.
    let mut pares_rot = incs.clone();
    pares_rot.rotate_left(3);
    let nulo_pares = firma_nivel2_incrementos(&pares_rot);
    close(original.level1[0], nulo_pares.level1[0]);
    close(original.level1[1], nulo_pares.level1[1]);
    close(original.level2[0][0], nulo_pares.level2[0][0]);
    close(original.level2[1][1], nulo_pares.level2[1][1]);
    close(
        original.level2[0][1] + original.level2[1][0],
        nulo_pares.level2[0][1] + nulo_pares.level2[1][0],
    );
    // Y la identidad estructural que la explica: S¹²+S²¹ = S¹·S² SIEMPRE.
    close(original.level2[0][1] + original.level2[1][0], original.level1[0] * original.level1[1]);
    // (ii) Los cross individuales se mueven bajo permutación de pares.
    assert!((original.level2[0][1] - nulo_pares.level2[0][1]).abs() > 1e-6);

    // Nulo UNIEJE: retornos rotados, dt en orden original.
    let mut rets: Vec<f64> = incs.iter().map(|(r, _)| *r).collect();
    rets.rotate_left(3);
    let unieje: Vec<(f64, f64)> =
        rets.iter().map(|r| (*r, 0.0)).collect();
    let mut mezclados = unieje;
    for (par, (_, dt)) in mezclados.iter_mut().zip(incs.iter()) {
        par.1 = *dt;
    }
    let nulo_unieje = firma_nivel2_incrementos(&mezclados);
    // Marginal del precio intacta ⇒ S¹ y S¹¹ EXACTO; dt intactos ⇒ S² y S²².
    close(original.level1[0], nulo_unieje.level1[0]);
    close(original.level2[0][0], nulo_unieje.level2[0][0]);
    close(original.level1[1], nulo_unieje.level1[1]);
    close(original.level2[1][1], nulo_unieje.level2[1][1]);
    // La suma sigue atada por la identidad S¹²+S²¹=S¹S²...
    close(
        nulo_unieje.level2[0][1] + nulo_unieje.level2[1][0],
        nulo_unieje.level1[0] * nulo_unieje.level1[1],
    );
    // ...pero el cross individual toma OTRO valor que el original.
    assert!(
        (original.level2[0][1] - nulo_unieje.level2[0][1]).abs() > 1e-6,
        "el cross S¹² cambia bajo nulo unieje: {} vs {}",
        original.level2[0][1],
        nulo_unieje.level2[0][1]
    );

    // (iii) Degeneración VERDADERA: retornos idénticos bajo nulo UNIEJE
    // (los dt no se tocan y todo retorno es igual ⇒ el camino es el mismo).
    let planos: Vec<(f64, f64)> = (0..6).map(|k| (0.10, 0.05 + 0.01 * k as f64)).collect();
    let mut planos_rets: Vec<f64> = planos.iter().map(|(r, _)| *r).collect();
    planos_rets.rotate_left(2); // rotar idénticos = no hacer nada
    let planos_unieje: Vec<(f64, f64)> = planos
        .iter()
        .zip(planos_rets.iter())
        .map(|((_, dt), r)| (*r, *dt))
        .collect();
    same(
        firma_nivel2_incrementos(&planos),
        firma_nivel2_incrementos(&planos_unieje),
    );
}
