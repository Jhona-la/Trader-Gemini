//! CL-78 (C-F3, F3.7-01): z de la base futuro-spot contra su propia media y
//! su propia dispersión, a la escala dominante de la moneda.
//!
//! Un z de reversión mide cuánto se aparta la base b = (F − S)/F de SU
//! equilibrio, en unidades de SU dispersión: z = (b − E[b])/σ_b. Antes el
//! núcleo publicaba como `vecm_zscore` el NIVEL de la base dividido por el ATR
//! del futuro, (F − S)/(F·máx(ATR%, 5 pb)). Una prima de financiación
//! constante (contango de 30 pb) daba z ≈ 2,4–2,9 en cada evento, y VECM
//! votaba reversión sin nada que revirtiera.
//!
//! Estimador por nodo. La base se observa en los eventos y se sostiene entre
//! ellos (muestreo y retención). Para cada nodo k de la malla espectral,
//! con τ_k = `SPECTRUM_SCALES_MS[k]`, tres promedios temporales con olvido
//! exponencial de constante τ_k:
//!
//! - S0 = ∫ e^{−(t−u)/τ_k} du,
//! - S1 = ∫ b(u)·e^{−(t−u)/τ_k} du,
//! - S2 = ∫ b(u)²·e^{−(t−u)/τ_k} du,
//!
//! con m = S1/S0 y σ² = S2/S0 − m². Al llegar una observación a t con el valor
//! anterior b0 sostenido desde t0, cada suma decae por e^{−Δt/τ_k} y suma
//! b0^p·τ_k·(1 − e^{−Δt/τ_k}), la integral exacta del tramo. Pesa el tiempo,
//! no los eventos: el mismo mercado da la misma media con cualquier densidad
//! de eventos (la lección de CL-70).
//!
//! Por qué por nodo y no una sola EWMA con la τ dominante vigente: la τ
//! dominante salta entre nodos (68,7 s ↔ 274,9 s en el fixture de la prueba),
//! y una EWMA cuya memoria cambia con cada evento recuerda la τ anterior: tras
//! subir de nodo se queda con la historia corta y no puede saber si ya hay
//! una τ de historia. Con τ_k fija cada nodo es exacto, y el calentamiento de
//! un nodo es S0 ≥ (1 − 1/e)·τ_k, es decir, una τ_k de historia observada.
//!
//! z a la τ pedida: interpolación lineal en ln τ entre los dos nodos que la
//! rodean (continua en τ); fuera de la malla [τ_17, τ_23] (17 s a 19,5 h), el
//! nodo extremo. Un nodo con peso no nulo sin calentar, o sin dispersión,
//! deja el z en `None` (abstención). La τ es la escala dominante de la
//! moneda, la misma que recibe la SDE OU de la base
//! (`StatArbEngine::update_with_clock`): la dislocación se mide contra la
//! base a la escala a la que opera la moneda.
//!
//! CL-87 (spot_z-1): la retención acredita como mucho `RETENCION_MAX_MS`
//! (30 s, la edad máxima del spot, CL-79). Más allá no hay datos: spot
//! caducado, entradas bloqueadas (kill-switch, pánico de latencia, deriva),
//! o una segunda observación muy lejos de la primera al arrancar. Las sumas
//! decaen por todo el tiempo transcurrido, pero sólo los primeros 30 s
//! suman la base retenida. Antes la base se acreditaba durante todo el
//! hueco: σ del nodo caía al piso y el z saturaba en ±3 al volver
//! (P(|z| > 1,5) ≈ 0,71 frente a 0,12). Tras un hueco largo frente a τ_k, el
//! nodo vuelve a calentar y mientras tanto se abstiene.
//!
//! El z del evento usa la observación nueva contra la media y la dispersión
//! del pasado, sin incluirla: es la innovación, causal. El piso de σ es la
//! resolución de la base, un tick del futuro relativo a su precio (CL-69):
//! por debajo no se distingue una dislocación de la cuantización del libro.

use quantum_arena::temporal_spectrum::SPECTRUM_SCALES_MS;

/// Primer y último nodo de la malla con estimador: rodean la banda operable
/// [30 s, 12 h] (τ_17 = 17,2 s, τ_23 = 19,5 h).
pub const NODO_MIN: usize = 17;
pub const NODO_MAX: usize = 23;
const NODOS: usize = NODO_MAX - NODO_MIN + 1;

/// CL-87: tiempo máximo que la base retenida cuenta como observada (ms). Es
/// la edad máxima del spot con la que el núcleo todavía usa la base
/// (SPOT_TTL_MS, CL-79).
pub const RETENCION_MAX_MS: f64 = 30_000.0;

/// Estado del z de la base de una moneda.
#[derive(Debug, Clone, Default)]
pub struct BaseZ {
    /// (S0, S1, S2) por nodo.
    sumas: [[f64; 3]; NODOS],
    /// Última observación: marca (ms) y base sostenida desde entonces.
    ultima: Option<(u64, f64)>,
}

impl BaseZ {
    pub fn new() -> Self {
        Self::default()
    }

    /// Observa la base `b` en `ts_ms` y devuelve su z a la escala `tau_ms`
    /// frente al pasado, o `None` si falta historia o dispersión a esa
    /// escala, o con entradas no válidas. `piso` es la resolución de la base
    /// (≥ 0). Una marca anterior a la última se descarta sin tocar el estado.
    pub fn observar(&mut self, ts_ms: u64, b: f64, tau_ms: f64, piso: f64) -> Option<f64> {
        if !b.is_finite() || !tau_ms.is_finite() || tau_ms <= 0.0 {
            return None;
        }
        match self.ultima {
            Some((t0, _)) if ts_ms < t0 => return None,
            Some((t0, b0)) if ts_ms > t0 => {
                let dt = (ts_ms - t0) as f64;
                // CL-87: la retención cuenta como mucho RETENCION_MAX_MS; el
                // resto del intervalo sólo hace decaer las sumas.
                let retenido = dt.min(RETENCION_MAX_MS);
                self.integrar(b0, retenido);
                if dt > retenido {
                    self.decaer(dt - retenido);
                }
            }
            _ => {}
        }
        let z = self.z_a(b, tau_ms, piso);
        self.ultima = Some((ts_ms, b));
        z
    }

    /// Integra la base `b0` sostenida durante `dt` ms: cada suma decae por
    /// e^{−Δt/τ_k} y suma b0^p·τ_k·(1 − e^{−Δt/τ_k}).
    fn integrar(&mut self, b0: f64, dt: f64) {
        // Malla 4^k: e^{−Δt/τ_{k−1}} = (e^{−Δt/τ_k})⁴. Una sola `exp_m1`, en
        // el nodo más lento; los demás por cuadrados, con
        // 1 − d⁴ = (1 − d)(1 + d)(1 + d²) para no perder precisión cuando
        // d ≈ 1.
        let em1 = (-dt / SPECTRUM_SCALES_MS[NODO_MAX]).exp_m1();
        let (mut d, mut m) = (1.0 + em1, -em1);
        for j in (0..NODOS).rev() {
            let peso = SPECTRUM_SCALES_MS[NODO_MIN + j] * m;
            let s = &mut self.sumas[j];
            s[0] = s[0] * d + peso;
            s[1] = s[1] * d + b0 * peso;
            s[2] = s[2] * d + b0 * b0 * peso;
            let d2 = d * d;
            m *= (1.0 + d) * (1.0 + d2);
            d = d2 * d2;
        }
    }

    /// CL-87: decaimiento sin datos durante `dt` ms.
    fn decaer(&mut self, dt: f64) {
        let mut d = (-dt / SPECTRUM_SCALES_MS[NODO_MAX]).exp();
        for j in (0..NODOS).rev() {
            for v in self.sumas[j].iter_mut() {
                *v *= d;
            }
            let d2 = d * d;
            d = d2 * d2;
        }
    }

    /// z de `b` a la escala `tau_ms`, interpolado en ln τ entre nodos.
    fn z_a(&self, b: f64, tau_ms: f64, piso: f64) -> Option<f64> {
        let piso = if piso.is_finite() { piso.max(0.0) } else { 0.0 };
        let (lo, hi) = (SPECTRUM_SCALES_MS[NODO_MIN], SPECTRUM_SCALES_MS[NODO_MAX]);
        let tau = tau_ms.clamp(lo, hi);
        let mut i = NODO_MIN;
        while i < NODO_MAX && SPECTRUM_SCALES_MS[i + 1] <= tau {
            i += 1;
        }
        if i == NODO_MAX {
            return self.z_nodo(i, b, piso);
        }
        let (t0, t1) = (SPECTRUM_SCALES_MS[i], SPECTRUM_SCALES_MS[i + 1]);
        let w = (tau / t0).ln() / (t1 / t0).ln();
        if w <= 0.0 {
            return self.z_nodo(i, b, piso);
        }
        let z0 = self.z_nodo(i, b, piso)?;
        let z1 = self.z_nodo(i + 1, b, piso)?;
        Some((1.0 - w) * z0 + w * z1)
    }

    /// z de `b` en el nodo `k` (con historia y dispersión).
    fn z_nodo(&self, k: usize, b: f64, piso: f64) -> Option<f64> {
        let [s0, s1, s2] = self.sumas[k - NODO_MIN];
        let tau_k = SPECTRUM_SCALES_MS[k];
        if s0 < (1.0 - (-1.0f64).exp()) * tau_k {
            return None;
        }
        let media = s1 / s0;
        let var = (s2 / s0 - media * media).max(0.0);
        let sigma = var.sqrt().max(piso);
        if sigma <= 0.0 {
            return None;
        }
        let z = (b - media) / sigma;
        z.is_finite().then_some(z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tau(k: usize) -> f64 {
        SPECTRUM_SCALES_MS[k]
    }

    /// Verdad: una base constante tiene z = 0 (con piso positivo). La
    /// media en coma flotante difiere de la constante en ~1e-19, que entre un
    /// piso de 1e-6 da ~1e-13: tolerancia 1e-9.
    #[test]
    fn base_constante_da_z_cero() {
        let mut e = BaseZ::new();
        let mut z = None;
        for k in 0..2_000u64 {
            z = e.observar(k * 1_000, 0.003, tau(18), 1e-6);
        }
        let z = z.expect("z con historia");
        assert!(z.abs() < 1e-9, "z = {z}");
    }

    /// Verdad: sin una τ de historia no hay z; con ella, sí. En el nodo 18
    /// (68,7 s), S0 = τ·(1 − e^{−t/τ}) llega a (1 − 1/e)·τ con t = τ.
    #[test]
    fn sin_historia_no_hay_z() {
        let mut e = BaseZ::new();
        let b = |k: u64| 0.003 + 1e-5 * (k % 2) as f64;
        for k in 0..68u64 {
            assert!(e.observar(k * 1_000, b(k), tau(18), 0.0).is_none(), "k = {k}");
        }
        let mut alguno = false;
        for k in 70..200u64 {
            alguno |= e.observar(k * 1_000, b(k), tau(18), 0.0).is_some();
        }
        assert!(alguno);
    }

    /// Verdad: la media pesa el tiempo, no los eventos. La base vale 0 en
    /// tramos de 10 s y 1 en tramos de 10 s; el tramo en 1 trae 100 eventos y
    /// el tramo en 0 trae uno. La media temporal es ½ (no ≈ 0,99) y un valor
    /// de ½ da z ≈ 0.
    #[test]
    fn la_media_pesa_el_tiempo_no_los_eventos() {
        let mut e = BaseZ::new();
        let mut t = 0u64;
        for _ in 0..400 {
            e.observar(t, 0.0, tau(20), 0.0);
            t += 10_000;
            for j in 0..100u64 {
                e.observar(t + j * 100, 1.0, tau(20), 0.0);
            }
            t += 10_000;
        }
        let z = e.observar(t, 0.5, tau(20), 0.0).expect("z con historia");
        assert!(z.abs() < 0.05, "z de ½ = {z}");
    }

    /// Verdad: una dislocación de k σ sobre una base estacionaria da z ≈ k.
    #[test]
    fn una_dislocacion_da_su_z() {
        let mut e = BaseZ::new();
        // Base alterna ±s alrededor de m, con tramos iguales: media m, σ = s.
        let (m, s) = (0.003, 1e-4);
        for k in 0..3_000u64 {
            let b = if k % 2 == 0 { m + s } else { m - s };
            e.observar(k * 1_000, b, tau(19), 0.0);
        }
        let z = e.observar(3_000_000, m + 3.0 * s, tau(19), 0.0).expect("z");
        assert!((z - 3.0).abs() < 0.01, "z = {z}");
    }

    /// Verdad: si la τ pedida salta entre nodos, el z no se pierde: con
    /// 15 min de historia, pedir τ_18 o τ_19 en eventos alternos da z siempre.
    #[test]
    fn la_tau_que_salta_no_borra_la_historia() {
        let mut e = BaseZ::new();
        for k in 0..900u64 {
            e.observar(k * 1_000, 0.003 + 1e-5 * ((k * 7) % 3) as f64, tau(18), 0.0);
        }
        for k in 900..1_000u64 {
            let t = if k % 2 == 0 { tau(19) } else { tau(18) };
            assert!(
                e.observar(k * 1_000, 0.003 + 1e-5 * ((k * 7) % 3) as f64, t, 0.0).is_some(),
                "k = {k}"
            );
        }
    }

    /// Verdad: entre dos nodos el z es la interpolación en ln τ de los z de
    /// los nodos (continuo en τ, sin saltos al cambiar de nodo).
    #[test]
    fn el_z_es_continuo_en_tau() {
        let mut e = BaseZ::new();
        for k in 0..4_000u64 {
            let b = 0.003 + 1e-5 * ((k * 13) % 7) as f64 + 1e-6 * (k / 500) as f64;
            e.observar(k * 1_000, b, tau(19), 0.0);
        }
        let b = 0.00305;
        let z0 = e.z_a(b, tau(19), 0.0).unwrap();
        let z1 = e.z_a(b, tau(20), 0.0).unwrap();
        let medio = (tau(19) * tau(20)).sqrt();
        let zm = e.z_a(b, medio, 0.0).unwrap();
        assert!((zm - 0.5 * (z0 + z1)).abs() < 1e-12, "{z0} {zm} {z1}");
        assert!((e.z_a(b, tau(20) * (1.0 - 1e-12), 0.0).unwrap() - z1).abs() < 1e-6);
    }

    /// Verdad: las sumas por cuadrados son las de la exponencial directa de
    /// cada nodo (error relativo < 1e-12), con pasos de 1 ms a 1 día. Con la
    /// retención acotada (CL-87): el valor sostenido suma durante
    /// c = mín(Δt, RETENCION_MAX_MS), y el resto del paso sólo decae, así que
    /// el peso es τ·e^{−(Δt−c)/τ}·(1 − e^{−c/τ}).
    #[test]
    fn cuadrados_igual_que_exp_directa() {
        let mut e = BaseZ::new();
        let mut referencia = [[0.0f64; 3]; NODOS];
        let mut t = 0u64;
        let mut b0 = 0.0;
        let pasos = [1u64, 7, 250, 1_000, 60_000, 3_600_000, 86_400_000];
        for k in 0..400usize {
            let dt = pasos[k % pasos.len()];
            let b = 0.003 + 1e-5 * ((k * 13) % 7) as f64;
            t += dt;
            e.observar(t, b, tau(18), 0.0);
            if k > 0 {
                for (j, s) in referencia.iter_mut().enumerate() {
                    let tk = tau(NODO_MIN + j);
                    let dec = (-(dt as f64) / tk).exp();
                    let c = (dt as f64).min(RETENCION_MAX_MS);
                    let p = -tk * (-((dt as f64) - c) / tk).exp() * (-c / tk).exp_m1();
                    s[0] = s[0] * dec + p;
                    s[1] = s[1] * dec + b0 * p;
                    s[2] = s[2] * dec + b0 * b0 * p;
                }
            }
            b0 = b;
        }
        for j in 0..NODOS {
            for p in 0..3 {
                let (x, r) = (e.sumas[j][p], referencia[j][p]);
                assert!((x - r).abs() <= 1e-12 * r.abs(), "nodo {j}, suma {p}: {x} frente a {r}");
            }
        }
    }

    /// Verdad: entradas no finitas, τ no positiva y marcas que retroceden
    /// no producen z ni corrompen el estado.
    #[test]
    fn entradas_invalidas() {
        let mut e = BaseZ::new();
        assert!(e.observar(1_000, f64::NAN, 1_000.0, 0.0).is_none());
        assert!(e.observar(1_000, 0.1, 0.0, 0.0).is_none());
        assert!(e.observar(1_000, 0.1, f64::NAN, 0.0).is_none());
        e.observar(10_000, 0.1, 1_000.0, 0.0);
        assert!(e.observar(5_000, 0.2, 1_000.0, 0.0).is_none());
        assert_eq!(e.ultima, Some((10_000, 0.1)));
    }
}
