//! El Sinkhorn de `MongeKantorovichWassersteinEngine` da los mismos bits que la
//! versión de la Ola 8 (a8f5fa1f2).
//!
//! Esa versión llevaba los dos medios pasos de cada iteración escritos en
//! línea, y con opt-level=3 LLVM (rustc nightly-2026-06-30) no terminaba de
//! optimizar la función: el crate no compilaba y el CI de main caía por
//! tiempo. Ahora cada medio paso es `medio_paso_sinkhorn`, y el de g recorre la
//! matriz transpuesta en el mismo orden.
//!
//! `Ola8` es una copia de aquella versión con un solo cambio: cada medio paso
//! va en su propia función `#[inline(never)]`, para que esta prueba compile.
//! Sus medios pasos recorren `cost` como entonces (el de g, por columnas), con
//! un único error máximo para los dos, y hacen las mismas operaciones en el
//! mismo orden. Motor y copia corren en el mismo binario, así que comparten la
//! libm y la comparación bit a bit no depende del destino: `exp` y `ln` dan
//! bits distintos según la libm y, en UCRT, según la ruta FMA3.

use feature_engine::monge_kantorovich_wasserstein::{
    MongeKantorovichWassersteinEngine, WassersteinParams, WassersteinReport, WASSERSTEIN_NODES,
};

const N: usize = WASSERSTEIN_NODES;

/// La versión de la Ola 8 de `MongeKantorovichWassersteinEngine`.
struct Ola8 {
    params: WassersteinParams,
    cost_matrix_w1: [[f64; N]; N],
    cost_matrix_w2: [[f64; N]; N],
    reference_distribution: [f64; N],
    last_w1: f64,
    last_drift_velocity: f64,
    last_ts_ms: u64,
}

/// Medio paso de f de la Ola 8: por filas de `cost`.
#[inline(never)]
fn medio_paso_f(
    f: &mut [f64; N],
    g: &[f64; N],
    cost: &[[f64; N]; N],
    log_mu: &[f64; N],
    eps: f64,
    max_err: &mut f64,
) {
    for (i, fila) in cost.iter().enumerate() {
        let mut max_val = f64::NEG_INFINITY;
        for (gj, c) in g.iter().zip(fila.iter()) {
            let v = (gj - c) / eps;
            if v > max_val {
                max_val = v;
            }
        }

        let mut sum_exp = 0.0;
        for (gj, c) in g.iter().zip(fila.iter()) {
            let v = (gj - c) / eps;
            sum_exp += (v - max_val).exp();
        }

        let lse = max_val + sum_exp.max(1e-15).ln();
        let next_f = eps * (log_mu[i] - lse);
        let diff = (next_f - f[i]).abs();
        if diff > *max_err {
            *max_err = diff;
        }
        f[i] = next_f;
    }
}

/// Medio paso de g de la Ola 8: por columnas de `cost`, sin transponer.
#[inline(never)]
fn medio_paso_g(
    g: &mut [f64; N],
    f: &[f64; N],
    cost: &[[f64; N]; N],
    log_nu: &[f64; N],
    eps: f64,
    max_err: &mut f64,
) {
    for (j, (gj, &lnu)) in g.iter_mut().zip(log_nu.iter()).enumerate() {
        let mut max_val = f64::NEG_INFINITY;
        for (fi, fila) in f.iter().zip(cost.iter()) {
            let v = (fi - fila[j]) / eps;
            if v > max_val {
                max_val = v;
            }
        }

        let mut sum_exp = 0.0;
        for (fi, fila) in f.iter().zip(cost.iter()) {
            let v = (fi - fila[j]) / eps;
            sum_exp += (v - max_val).exp();
        }

        let lse = max_val + sum_exp.max(1e-15).ln();
        let next_g = eps * (lnu - lse);
        let diff = (next_g - *gj).abs();
        if diff > *max_err {
            *max_err = diff;
        }
        *gj = next_g;
    }
}

impl Ola8 {
    fn new(params: WassersteinParams) -> Self {
        let mut grid_coords = [0.0; N];
        let step = 2.0 / ((N - 1) as f64).max(1.0);
        for (i, coord) in grid_coords.iter_mut().enumerate() {
            *coord = -1.0 + (i as f64) * step;
        }

        let mut cost_matrix_w1 = [[0.0; N]; N];
        let mut cost_matrix_w2 = [[0.0; N]; N];
        for (i, &xi) in grid_coords.iter().enumerate() {
            for (j, &xj) in grid_coords.iter().enumerate() {
                let diff = (xi - xj).abs();
                cost_matrix_w1[i][j] = diff;
                cost_matrix_w2[i][j] = diff * diff;
            }
        }

        let mut reference_distribution = [0.0; N];
        let mut sum_ref = 0.0;
        let sigma_ref = 0.40;
        for (ref_slot, &x) in reference_distribution.iter_mut().zip(grid_coords.iter()) {
            let density = (-0.5 * (x / sigma_ref).powi(2)).exp();
            *ref_slot = density;
            sum_ref += density;
        }
        if sum_ref > 1e-12 {
            for ref_slot in reference_distribution.iter_mut() {
                *ref_slot /= sum_ref;
            }
        }

        Self {
            params,
            cost_matrix_w1,
            cost_matrix_w2,
            reference_distribution,
            last_w1: 0.0,
            last_drift_velocity: 0.0,
            last_ts_ms: 0,
        }
    }

    fn update_and_evaluate(
        &mut self,
        empirical_weights: &[f64; N],
        current_tau_ms: f64,
        ts_ms: u64,
    ) -> WassersteinReport {
        let mut mu = [0.0; N];
        let mut sum_mu = 0.0;
        for (slot, &w) in mu.iter_mut().zip(empirical_weights.iter()) {
            let val = if w.is_finite() && w > 0.0 { w } else { 0.0 };
            *slot = val;
            sum_mu += val;
        }
        if sum_mu <= 1e-12 {
            let uniform = 1.0 / (N as f64);
            for slot in mu.iter_mut() {
                *slot = uniform;
            }
        } else {
            for slot in mu.iter_mut() {
                *slot /= sum_mu;
            }
        }

        let nu = self.reference_distribution;

        let tau_scale_factor =
            1.0 + self.params.tau_cost_weight * (1.0 + current_tau_ms.max(1.0).ln());
        let mut cost = [[0.0; N]; N];
        for (fila, fila_w1) in cost.iter_mut().zip(self.cost_matrix_w1.iter()) {
            for (c, &w1) in fila.iter_mut().zip(fila_w1.iter()) {
                *c = w1 * tau_scale_factor;
            }
        }

        let eps = self.params.epsilon.max(1e-4);
        let mut f = [0.0; N];
        let mut g = [0.0; N];

        let mut log_mu = [0.0; N];
        let mut log_nu = [0.0; N];
        for (l, &m) in log_mu.iter_mut().zip(mu.iter()) {
            *l = (m.max(1e-15)).ln();
        }
        for (l, &v) in log_nu.iter_mut().zip(nu.iter()) {
            *l = (v.max(1e-15)).ln();
        }

        let mut iters_done = 0;
        for iter in 0..self.params.max_iterations {
            iters_done = iter + 1;
            let mut max_err = 0.0;
            medio_paso_f(&mut f, &g, &cost, &log_mu, eps, &mut max_err);
            medio_paso_g(&mut g, &f, &cost, &log_nu, eps, &mut max_err);
            if max_err < self.params.tolerance {
                break;
            }
        }

        let mut w1 = 0.0;
        let mut w2_sq = 0.0;
        let mut entropy = 0.0;
        for (i, fila) in cost.iter().enumerate() {
            for (j, &c) in fila.iter().enumerate() {
                let log_p = (f[i] + g[j] - c) / eps;
                let p = if log_p > -50.0 { log_p.exp() } else { 0.0 };
                if p > 0.0 {
                    w1 += p * c;
                    w2_sq += p * self.cost_matrix_w2[i][j] * tau_scale_factor;
                    entropy -= p * p.max(1e-15).ln();
                }
            }
        }

        let dt = if self.last_ts_ms > 0 && ts_ms > self.last_ts_ms {
            ((ts_ms - self.last_ts_ms) as f64) / 1000.0
        } else {
            0.10
        };
        let drift_velocity = if dt > 1e-4 {
            (w1 - self.last_w1) / dt
        } else {
            0.0
        };
        let drift_acceleration = if dt > 1e-4 {
            (drift_velocity - self.last_drift_velocity) / dt
        } else {
            0.0
        };
        let v_excess = (drift_velocity.abs() - self.params.drift_velocity_threshold).max(0.0);
        let liquidity_evaporation_risk = (1.0 - (-v_excess * 15.0).exp()).clamp(0.0, 1.0);

        self.last_w1 = w1;
        self.last_drift_velocity = drift_velocity;
        self.last_ts_ms = ts_ms;

        WassersteinReport {
            w1_distance: w1,
            w2_squared: w2_sq,
            drift_velocity,
            drift_acceleration,
            liquidity_evaporation_risk,
            transport_entropy: entropy,
            iterations_completed: iters_done,
        }
    }
}

/// Generador congruencial determinista: la secuencia de enteros es la misma
/// en cada ejecución y en cada máquina.
struct Lcg(u64);

impl Lcg {
    fn siguiente(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    fn unidad(&mut self) -> f64 {
        (self.siguiente() as f64) / ((1u64 << 53) as f64)
    }
}

/// Peso empírico de una celda: casi siempre finito y positivo, a veces cero,
/// NaN, ±∞, negativo o extremo, que el motor descarta o renormaliza.
fn peso(r: &mut Lcg) -> f64 {
    match r.siguiente() % 20 {
        0 => 0.0,
        1 => f64::NAN,
        2 => f64::INFINITY,
        3 => f64::NEG_INFINITY,
        4 => -r.unidad(),
        5 => 1e-300 * r.unidad(),
        6 => 1e300 * r.unidad(),
        _ => r.unidad(),
    }
}

/// Una llamada: pesos, τ (NaN, ±∞, nula, negativa o finita) y marca de tiempo
/// (repetida, que retrocede o que avanza).
fn entrada(r: &mut Lcg, ts: &mut u64) -> ([f64; N], f64) {
    let mut pesos = [0.0f64; N];
    let modo = r.siguiente() % 6;
    for celda in pesos.iter_mut() {
        *celda = match modo {
            0 => peso(r),
            1 => 0.0,
            2 => r.unidad(),
            3 => {
                if r.siguiente().is_multiple_of(4) {
                    r.unidad()
                } else {
                    0.0
                }
            }
            _ => (r.unidad() * 3.0).exp(),
        };
    }
    let tau = match r.siguiente() % 7 {
        0 => f64::NAN,
        1 => 0.0,
        2 => -5.0,
        3 => 1e12 * r.unidad(),
        4 => f64::INFINITY,
        5 => f64::NEG_INFINITY,
        _ => 1000.0 + 1e6 * r.unidad(),
    };
    *ts = match r.siguiente() % 6 {
        0 => *ts,
        1 => ts.saturating_sub(50),
        _ => *ts + 1 + (r.siguiente() % 5000),
    };
    (pesos, tau)
}

/// Corre motor y copia sobre la misma secuencia y exige los mismos bits en
/// cada campo de cada llamada. Devuelve las iteraciones de cada llamada.
fn comparar(params: WassersteinParams, semilla: u64, llamadas: usize) -> Vec<usize> {
    let mut motor = MongeKantorovichWassersteinEngine::new(params);
    let mut copia = Ola8::new(params);
    let mut r = Lcg(semilla);
    let mut ts: u64 = 1_700_000_000_000;
    let mut iteraciones = Vec::with_capacity(llamadas);
    for k in 0..llamadas {
        let (pesos, tau) = entrada(&mut r, &mut ts);
        let a = motor.update_and_evaluate(&pesos, tau, ts);
        let b = copia.update_and_evaluate(&pesos, tau, ts);
        let campos = [
            ("w1_distance", a.w1_distance, b.w1_distance),
            ("w2_squared", a.w2_squared, b.w2_squared),
            ("drift_velocity", a.drift_velocity, b.drift_velocity),
            (
                "drift_acceleration",
                a.drift_acceleration,
                b.drift_acceleration,
            ),
            (
                "liquidity_evaporation_risk",
                a.liquidity_evaporation_risk,
                b.liquidity_evaporation_risk,
            ),
            (
                "transport_entropy",
                a.transport_entropy,
                b.transport_entropy,
            ),
        ];
        for (nombre, x, y) in campos {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "llamada {k}, {nombre}: motor {x:e}, Ola 8 {y:e}"
            );
        }
        assert_eq!(
            a.iterations_completed, b.iterations_completed,
            "llamada {k}, iterations_completed"
        );
        iteraciones.push(a.iterations_completed);
    }
    iteraciones
}

#[test]
fn con_los_parametros_por_defecto_da_los_bits_de_la_ola_8() {
    let params = WassersteinParams::default();
    let iteraciones = comparar(params, 0x5EED_1234_ABCD_0042, 400);
    // La secuencia llega a parar por tolerancia y a agotar las iteraciones.
    assert!(iteraciones
        .iter()
        .any(|&n| n > 0 && n < params.max_iterations));
    assert!(iteraciones.contains(&params.max_iterations));
}

#[test]
fn agotando_las_iteraciones_da_los_bits_de_la_ola_8() {
    // Tolerancia cero: el error máximo nunca es menor, así que cada llamada
    // agota las iteraciones. ε por debajo del suelo de 1e-4.
    let params = WassersteinParams {
        epsilon: 1e-6,
        max_iterations: 200,
        tolerance: 0.0,
        ..WassersteinParams::default()
    };
    let iteraciones = comparar(params, 0x0BAD_C0DE_0000_0001, 120);
    assert!(iteraciones.iter().all(|&n| n == params.max_iterations));
}

#[test]
fn sin_iteraciones_da_los_bits_de_la_ola_8() {
    let params = WassersteinParams {
        max_iterations: 0,
        ..WassersteinParams::default()
    };
    let iteraciones = comparar(params, 0x1234_5678_9ABC_DEF0, 200);
    assert!(iteraciones.iter().all(|&n| n == 0));
}

#[test]
fn con_parametros_no_finitos_da_los_bits_de_la_ola_8() {
    // ε, tolerancia y pesos del coste NaN o ±∞: el motor no debe divergir de
    // la Ola 8 tampoco con parámetros corruptos.
    for (epsilon, tolerance, tau_cost_weight) in [
        (f64::NAN, f64::NAN, f64::NAN),
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY),
        (-1.0, 1e300, -1.0),
    ] {
        let params = WassersteinParams {
            epsilon,
            tolerance,
            tau_cost_weight,
            ..WassersteinParams::default()
        };
        comparar(params, 0xC0FF_EE00_1234_5678, 80);
    }
}
