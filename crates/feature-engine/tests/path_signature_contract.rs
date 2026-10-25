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
        (0.30, 0.05),
        (-0.20, 0.10),
        (0.15, 0.05),
        (0.05, 0.30),
        (-0.10, 0.20),
        (0.25, 0.15),
        (-0.05, 0.10),
        (0.10, 0.05),
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
    close(
        original.level2[0][1] + original.level2[1][0],
        original.level1[0] * original.level1[1],
    );
    // (ii) Los cross individuales se mueven bajo permutación de pares.
    assert!((original.level2[0][1] - nulo_pares.level2[0][1]).abs() > 1e-6);

    // Nulo UNIEJE: retornos rotados, dt en orden original.
    let mut rets: Vec<f64> = incs.iter().map(|(r, _)| *r).collect();
    rets.rotate_left(3);
    let unieje: Vec<(f64, f64)> = rets.iter().map(|r| (*r, 0.0)).collect();
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

/// GLM 163 — NULO DE MÁXIMA ENTROPÍA: momentos EXACTOS del nulo unieje.
/// Bajo permutación uniforme de los log-retornos x contra dt fijos y, el
/// intercambio del orden de sumatoria del término prefijo reduce cada
/// cross del nivel-2 a un ESTADÍSTICO LINEAL DE ASIGNACIÓN:
///   S¹²_nulo = Σ_a v_a·x_σ(a) con v_a = (Σ_{b>a} y_b) + ½·y_a
///   S²¹_nulo = Σ_a u_a·x_σ(a) con u_a = (Σ_{b<a} y_b) + ½·y_a
/// (pref^y es determinista: no se permuta). Los momentos de la permutación
/// sin reemplazo son elementales: E = x̄·Σw y Var = Sxx·Sww/(n−1).
/// Esto separa EXACTAMENTE la señal del ruido del nulo que en 162 era
/// inseparable con una sola permutación muestral.
fn pesos_nulo_unieje(y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    let mut v = Vec::with_capacity(n); // pesos de S¹² (sufijo de dt + ½·y)
    let mut u = Vec::with_capacity(n); // pesos de S²¹ (prefijo de dt + ½·y)
    let mut prefijo = 0.0;
    let mut sufijo: f64 = y.iter().sum();
    for &dy in y {
        sufijo -= dy;
        v.push(sufijo + 0.5 * dy);
        u.push(prefijo + 0.5 * dy);
        prefijo += dy;
    }
    (v, u)
}

/// (media, varianza) de Σ_a w_a·x_σ(a) bajo σ uniforme sobre las n!.
fn momentos_asignacion(x: &[f64], w: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let (sx, sw) = (x.iter().sum::<f64>(), w.iter().sum::<f64>());
    let sxx = x.iter().map(|&a| (a - sx / n) * (a - sx / n)).sum::<f64>();
    let sww = w.iter().map(|&a| (a - sw / n) * (a - sw / n)).sum::<f64>();
    (sx / n * sw, sxx * sww / (n - 1.0))
}

fn todas_las_permutaciones(x: &[f64]) -> Vec<Vec<f64>> {
    fn rec(x: &[f64], usados: &mut Vec<bool>, actual: &mut Vec<f64>, out: &mut Vec<Vec<f64>>) {
        if actual.len() == x.len() {
            out.push(actual.clone());
            return;
        }
        for i in 0..x.len() {
            if !usados[i] {
                usados[i] = true;
                actual.push(x[i]);
                rec(x, usados, actual, out);
                actual.pop();
                usados[i] = false;
            }
        }
    }
    let (mut usados, mut actual, mut out) = (vec![false; x.len()], Vec::new(), Vec::new());
    rec(x, &mut usados, &mut actual, &mut out);
    out
}

#[test]
fn momentos_exactos_del_nulo_unieje_vs_enumeracion() {
    // n=6 genérico (dt NO simétricos ⇒ v ≠ u ⇒ E[S¹²] ≠ E[S²¹]).
    let incs: Vec<(f64, f64)> = vec![
        (0.30, 0.05),
        (-0.20, 0.25),
        (0.15, 0.05),
        (0.05, 0.30),
        (-0.10, 0.20),
        (0.25, 0.15),
    ];
    let (x, y): (Vec<f64>, Vec<f64>) = incs.iter().copied().unzip();
    let (v, u) = pesos_nulo_unieje(&y);
    let (e12, var12) = momentos_asignacion(&x, &v);
    let (e21, var21) = momentos_asignacion(&x, &u);

    // Asimetría anclada: pesos distintos ⇒ esperanzas distintas en general.
    assert!(
        (e12 - e21).abs() > 1e-6,
        "v ≠ u debe separar E[S¹²] de E[S²¹]"
    );
    // Reflexión estructural: v_a + u_a = (sufijo+prefijo+y_a) = Σy ⇒
    // Sww idéntica ⇒ las DOS varianzas del nulo coinciden SIEMPRE; sólo
    // las medias separan S¹² de S²¹ bajo el nulo unieje.
    close(var12, var21);
    // Simetría anti-exacta de los z (medida 163: z(S²¹) ≡ −z(S¹²) en TODO
    // símbolo): con Σy=1, S²=1 y S¹=Σx=n·x̄, la identidad del kernel
    // S¹²+S²¹=S¹·S² fuerza E₁₂+E₂₁ = x̄·n = S¹ ⇒ el error de ambos cross
    // es opuesto y comparten varianza ⇒ z(S²¹) = −z(S¹²) IDENTICO. La
    // pareja no duplica información: es UN número con signo.
    close(e12 + e21, x.iter().sum::<f64>());

    // Enumeración bruta de las 720 permutaciones: momento exacto vs forma cerrada.
    let (mut s12, mut s12sq, mut s21, mut s21sq, mut m) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for perm in todas_las_permutaciones(&x) {
        let ventana: Vec<(f64, f64)> = perm.iter().zip(y.iter()).map(|(r, dy)| (*r, *dy)).collect();
        let s = firma_nivel2_incrementos(&ventana);
        s12 += s.level2[0][1];
        s12sq += s.level2[0][1] * s.level2[0][1];
        s21 += s.level2[1][0];
        s21sq += s.level2[1][0] * s.level2[1][0];
        m += 1.0;
    }
    let (m12, v12m) = (s12 / m, s12sq / m - (s12 / m) * (s12 / m));
    let (m21, v21m) = (s21 / m, s21sq / m - (s21 / m) * (s21 / m));
    close(e12, m12);
    close(e21, m21);
    assert!(
        (var12 - v12m).abs() <= 1e-12,
        "Var[S¹²] cerrada {var12:.17e} vs enumerada {v12m:.17e}"
    );
    assert!(
        (var21 - v21m).abs() <= 1e-12,
        "Var[S²¹] cerrada {var21:.17e} vs enumerada {v21m:.17e}"
    );

    // n=2 ancla manual: Var = (v₀−v₁)²(x₀−x₁)²/4.
    let (x2, y2) = (vec![0.2, -0.1], vec![0.7, 0.3]);
    let (v2, _) = pesos_nulo_unieje(&y2);
    let (_, var2) = momentos_asignacion(&x2, &v2);
    close(
        var2,
        (v2[0] - v2[1]) * (v2[0] - v2[1]) * (x2[0] - x2[1]) * (x2[0] - x2[1]) / 4.0,
    );

    // Degenerado VERDADERO: retornos constantes ⇒ Sxx=0 ⇒ Var=0 ⇒ z no medible.
    let xc = vec![0.10_f64; 6];
    let (ec, varc) = momentos_asignacion(&xc, &v);
    let var_enum: f64 = {
        let mut acc = 0.0;
        for perm in todas_las_permutaciones(&xc) {
            let ventana: Vec<(f64, f64)> =
                perm.iter().zip(y.iter()).map(|(r, dy)| (*r, *dy)).collect();
            let s = firma_nivel2_incrementos(&ventana).level2[0][1];
            acc += (s - ec) * (s - ec);
        }
        acc / 720.0
    };
    close(varc, 0.0);
    close(var_enum, 0.0);
}
