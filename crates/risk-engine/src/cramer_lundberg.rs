//! P-A del consejo (Ola 23, Qoder 2026-10-01) — CRAMÉR–LUNDBERG sobre
//! siniestros MEDIDOS.
//!
//! ## Por qué
//!
//! El tope de ruina vigente (`ruin.rs`, streak-bound) modela la ruina por
//! RACHA de pérdidas Bernoulli sin memoria de MAGNITUD: dos sistemas con
//! la misma tasa de acierto pero colas distintas reciben el mismo tope.
//! El modelo actuarial clásico (Cramér 1930, Lundberg 1926) usa la
//! distribución completa de siniestros: si los retornos netos por trade
//! y_i tienen deriva positiva, existe R > 0 (coeficiente de ajuste) que
//! resuelve E[e^{−R·y}] = 1 y la probabilidad de que la SUMA ACUMULADA
//! S_n = Σy_i (retorno-fracción lineal, NO log-capital) caiga por debajo
//! de −m antes de recuperarse cumple la cota
//!
//!   ψ(m) ≤ e^{−R·m}.
//!
//! ## Qué mide este módulo (observación, no política)
//!
//! R sobre el anillo de CIERRES NETOS en unidades de retorno por NOCIONAL
//! (independiente del sizing: el consumidor que quiera la cota de capital
//! usa R_capital(f) = R_nocional / f). Sin deriva positiva en la muestra
//! (Σ y ≤ 0) no hay R: la cota no significa nada sin edge medido —
//! honestidad Fisher/D-754, no un número inventado.
//!
//! ## Contrato
//!
//! - `observar(y)`: y = retorno neto por trade (fracción del nocional),
//!   anillo de 256 cierres (≈ media robusta de régimen corto).
//! - `lundberg()`: raíz positiva de g(R) = (1/n)Σ e^{−R·y_i} − 1 por
//!   BISECCIÓN en (1e-9, 100] (Newton es infiable aquí: g' cambia de
//!   signo y crece a través de la raíz — ver comentario del método);
//!   g(0)=0 es la raíz trivial. Sin deriva, sin cruce en el rango o
//!   muestra < 30 cierres → `None`.
//! - `margen_de_cota(r, epsilon)`: caída acumulada mínima m (en unidades
//!   de retorno-fracción) tal que la cota promete ψ ≤ epsilon:
//!   m = ln(1/ε)/R (monótona en ambos argumentos).
//! - Falsación: bootstrap MC (semilla fija) verifica que la frecuencia
//!   empírica de mínimos de capital queda BAJO la cota; distribución a
//!   dos puntos con raíz analítica conocida verifica el solver.

const N_SINIESTROS: usize = 256;

#[derive(Debug, Clone)]
pub struct EstimadorSiniestros {
    anillo: [f64; N_SINIESTROS],
    head: usize,
    count: usize,
}

impl Default for EstimadorSiniestros {
    fn default() -> Self {
        Self::new()
    }
}

impl EstimadorSiniestros {
    pub fn new() -> Self {
        Self {
            anillo: [0.0; N_SINIESTROS],
            head: 0,
            count: 0,
        }
    }

    /// Registra el retorno neto por nocional de un cierre. Valores no
    /// finitos se descartan (sin inventar).
    pub fn observar(&mut self, y: f64) {
        if !y.is_finite() {
            return;
        }
        self.anillo[self.head] = y;
        self.head = (self.head + 1) % N_SINIESTROS;
        if self.count < N_SINIESTROS {
            self.count += 1;
        }
    }

    /// Coeficiente de ajuste R de Lundberg de la muestra, o `None` sin
    /// deriva positiva / sin convergencia / muestra insuficiente (< 30
    /// cierres — misma disciplina que MUESTRAS_MADURAS).
    pub fn lundberg(&self) -> Option<f64> {
        const MIN_MUESTRAS: usize = 30;
        if self.count < MIN_MUESTRAS {
            return None;
        }
        let n = self.count as f64;
        let mut media = 0.0;
        for i in 0..self.count {
            media += self.anillo[i];
        }
        media /= n;
        if !(media.is_finite() && media > 0.0) {
            return None; // sin deriva positiva no hay cota significativa
        }
        let mut var2 = 0.0;
        for i in 0..self.count {
            let d = self.anillo[i] - media;
            var2 += d * d;
        }
        let var2 = (var2 / n).max(1e-18);
        // BISECCIÓN sobre g(R) = (1/n)Σ e^{−R·y_i} − 1. Con deriva positiva
        // g(0⁺) < 0 (g'(0) = −media < 0); si g(hi) > 0 hay cambio de signo
        // y la raíz es única en el intervalo (g es convexa: Σ e^{−Ry} tiene
        // a lo sumo DOS cruces con 0, el trivial en R=0 y la raíz buscada).
        // Newton es INFIABLE aquí: g' cambia de signo (crece a través de la
        // raíz) y un arranque heavy-traffic puede caer en la zona de g' > 0.
        let g = |r: f64| -> f64 {
            let mut acc = 0.0;
            for i in 0..self.count {
                acc += (-r * self.anillo[i]).exp();
            }
            acc / n - 1.0
        };
        // F1-B1 / F1-B3: Techo adaptativo derivado de la aproximación de difusión
        // R ≈ 2μ/σ² usando var2. En micro-retornos (0.1%-0.5%) R puede ser 200-5000;
        // un hi fijo de 100 devolvía None falsamente a pesar de existir edge legítimo.
        let mut hi = (4.0 * media / var2).max(100.0).min(100_000.0);
        let mut g_hi = g(hi);
        if g_hi <= 0.0 {
            for _ in 0..8 {
                hi *= 2.0;
                if hi > 100_000.0 {
                    break;
                }
                g_hi = g(hi);
                if g_hi.is_finite() && g_hi > 0.0 {
                    break;
                }
            }
        }
        if !(g_hi.is_finite() && g_hi > 0.0) {
            return None; // sin cruce en el rango acotado
        }
        let mut lo = 1e-9_f64;
        if g(lo) > 0.0 {
            return None;
        }
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if !mid.is_finite() {
                return None;
            }
            if g(mid) > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
            if hi - lo <= 1e-12 {
                break;
            }
        }
        let r = 0.5 * (lo + hi);
        if r.is_finite() && r > 0.0 {
            Some(r)
        } else {
            None
        }
    }

    /// Error estándar asintótico del estimador M de R:
    /// SE(R̂) = sqrt( Σ (e^{-R̂ y_i} - 1)² ) / ( Σ y_i e^{-R̂ y_i} )
    pub fn standard_error_r(&self, r: f64) -> Option<f64> {
        if self.count < 30 || !r.is_finite() || r <= 0.0 {
            return None;
        }
        let mut sum_sq_residuals = 0.0f64;
        let mut denom = 0.0f64;
        for i in 0..self.count {
            let y = self.anillo[i];
            let exp_term = (-r * y).exp();
            let residual = exp_term - 1.0;
            sum_sq_residuals += residual * residual;
            denom += y * exp_term;
        }
        let denom_abs = denom.abs();
        if !denom_abs.is_finite() || denom_abs <= 1e-18 || !sum_sq_residuals.is_finite() {
            return None;
        }
        let se = sum_sq_residuals.sqrt() / denom_abs;
        if se.is_finite() && se >= 0.0 {
            Some(se)
        } else {
            None
        }
    }

    /// Cota inferior conservadora (LCB) de R: R_lcb = R̂ - z · SE(R̂).
    /// Resuelve R7-R2-A-3: bajo incertidumbre muestral o pocas observaciones,
    /// R̂ puntual puede sobreestimar el coeficiente y desproteger el margen de ruina.
    /// Retorna `None` si la cota inferior no garantiza un coeficiente estrictamente positivo.
    pub fn lundberg_lcb(&self, z: f64) -> Option<f64> {
        let r = self.lundberg()?;
        let z_eff = if z.is_finite() && z >= 0.0 { z } else { 1.645 };
        let se = self.standard_error_r(r)?;
        let r_lcb = r - z_eff * se;
        if r_lcb.is_finite() && r_lcb > 0.0 {
            Some(r_lcb)
        } else {
            None
        }
    }

    /// Margen log m mínimo para que la cota prometa ψ ≤ epsilon con
    /// coeficiente r: m = ln(1/ε)/r. `None` con ε fuera de (0,1) o r ≤ 0.
    pub fn margen_de_cota(r: f64, epsilon: f64) -> Option<f64> {
        if !(r.is_finite() && r > 0.0) || !(epsilon > 0.0 && epsilon < 1.0) {
            return None;
        }
        Some((1.0 / epsilon).ln() / r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(estado: &mut u64) -> f64 {
        *estado ^= *estado << 13;
        *estado ^= *estado >> 7;
        *estado ^= *estado << 17;
        ((*estado >> 11) as f64) / ((1u64 << 53) as f64)
    }

    #[test]
    fn qo_600_newton_recupera_la_raiz_analitica_de_dos_puntos() {
        // y = +1.5% (p=0.5) / −1% (p=0.5): la raíz de
        // 0.5·e^{−1.5R} + 0.5·e^{+1R} = 1 se obtiene por bisección y el
        // estimador (alimentado con la muestra EXACTA alternada) debe
        // coincidir.
        let mut est = EstimadorSiniestros::new();
        for i in 0..256 {
            est.observar(if i % 2 == 0 { 0.015 } else { -0.01 });
        }
        let r_est = est.lundberg().expect("muestra con deriva positiva");
        let g = |r: f64| 0.5 * (-0.015 * r).exp() + 0.5 * (0.01 * r).exp() - 1.0;
        let (mut lo, mut hi) = (0.0f64, 200.0f64);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if g(mid) > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let r_exacto = 0.5 * (lo + hi);
        assert!(
            (r_est - r_exacto).abs() < 1e-6,
            "newton={} biseccion={}",
            r_est,
            r_exacto
        );
        // Margen para prometer ψ ≤ 5%: m = ln(20)/R, monótono decreciente en R.
        let m = EstimadorSiniestros::margen_de_cota(r_est, 0.05).expect("cota");
        assert!((m - 20.0_f64.ln() / r_est).abs() < 1e-12);
        assert!(EstimadorSiniestros::margen_de_cota(r_est, 0.01).unwrap() > m);
        assert!(EstimadorSiniestros::margen_de_cota(-1.0, 0.05).is_none());
    }

    #[test]
    fn qo_600_sin_deriva_o_sin_muestras_no_hay_cota() {
        let mut perdedor = EstimadorSiniestros::new();
        for i in 0..100 {
            perdedor.observar(if i % 2 == 0 { -0.02 } else { 0.005 });
        }
        assert_eq!(perdedor.lundberg(), None, "sin edge la cota no significa nada");
        let mut frio = EstimadorSiniestros::new();
        for i in 0..29 {
            frio.observar(if i % 2 == 0 { 0.015 } else { -0.01 });
        }
        assert_eq!(frio.lundberg(), None, "29 cierres no opinan");
        let mut nan = EstimadorSiniestros::new();
        nan.observar(f64::NAN);
        assert_eq!(nan.count, 0, "no finito no entra al anillo");
    }

    #[test]
    fn qo_600_la_cota_dominar_la_frecuencia_empirica_bootstrap() {        // Falsación canónica: con la distribución de dos puntos, 20_000
        // caminatas de 400 trades; la frecuencia de caminatas cuyo mínimo
        // log cae más de m debe quedar bajo e^{−R·m} para todo m sondeado.
        // Anillo DETERMINISTA (128/128 exactos): la cota de Lundberg exige
        // el R de la distribución real — un R̂ muestreado con ruido puede
        // quedar por ENCIMA del verdadero y violar la cota (error de
        // estimación, no del teorema). En vivo, el uso conservador exige
        // descontar R̂ (LCB); aquí verificamos el teorema exacto.
        let mut est = EstimadorSiniestros::new();
        let mut lcg = 0xDEADBEEFCAFEBABEu64;
        let (g_val, p, x_val) = (0.015_f64, 0.5_f64, -0.01_f64);
        for i in 0..256 {
            est.observar(if i % 2 == 0 { g_val } else { x_val });
        }
        let r = est.lundberg().expect("cota estimable");
        let caminatas = 100_000usize;
        let trades = 400usize;
        let (mut cruces, mut total) = ([0usize; 4], 0usize);
        let margenes = [0.20_f64, 0.35, 0.50, 0.70];
        for _ in 0..caminatas {
            let mut log_cap = 0.0f64;
            let mut min_log = 0.0f64;
            for _ in 0..trades {
                log_cap += if xorshift(&mut lcg) < p { g_val } else { x_val };
                if log_cap < min_log {
                    min_log = log_cap;
                }
            }
            total += 1;
            for (k, &m) in margenes.iter().enumerate() {
                if min_log <= -m {
                    cruces[k] += 1;
                }
            }
        }
        for (k, &m) in margenes.iter().enumerate() {
            let empirica = cruces[k] as f64 / total as f64;
            let cota = (-r * m).exp();
            // Tolerancia 3σ: la frecuencia MC estima una probabilidad; el
            // teorema no puede violarse, la ESTIMACIÓN sí por ruido.
            let sigma = (cota * (1.0 - cota).max(0.0) / total as f64).sqrt().max(1e-12);
            assert!(
                empirica <= cota + 3.0 * sigma,
                "cota violada más allá de 3σ en m={}: emp={} > cota+3σ={} (R={})",
                m,
                empirica,
                cota + 3.0 * sigma,
                r
            );
        }
    }

    #[test]
    fn test_r7_r2_a3_lundberg_lcb_conservador() {
        let mut est = EstimadorSiniestros::new();
        for i in 0..128 {
            est.observar(if i % 2 == 0 { 0.015 } else { -0.01 });
        }
        let r_point = est.lundberg().expect("cota estimable");
        let se = est.standard_error_r(r_point).expect("error estándar calculable");
        assert!(se > 0.0 && se.is_finite(), "SE debe ser positivo finito: {se}");

        let r_lcb_95 = est.lundberg_lcb(1.645).expect("lcb debe existir");
        assert!(r_lcb_95 < r_point, "r_lcb={r_lcb_95} debe ser estrictamente menor que r_point={r_point}");
        assert!((r_lcb_95 - (r_point - 1.645 * se)).abs() < 1e-10);

        // Muestra ruidosa con drift apenas positivo debe dar None si z es alto
        let mut marginal = EstimadorSiniestros::new();
        for i in 0..60 {
            marginal.observar(if i % 2 == 0 { 0.01001 } else { -0.01 });
        }
        if let Some(_r_marg) = marginal.lundberg() {
            // Un z muy conservador (ej. 50.0) debe rechazar la cota (r_lcb <= 0 -> None)
            assert_eq!(marginal.lundberg_lcb(50.0), None);
        }
    }
}
