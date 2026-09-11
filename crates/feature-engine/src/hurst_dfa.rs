//! ESTIMADOR DE HURST MULTIESCALA REAL (D-615 / D-616 — DÉCIMA OLA).
//!
//! # Qué estaba mal
//!
//! `MultifractalSpectrumEngine` declaraba calcular el exponente de Hölder/Hurst
//! y alimentaba con él al consejo de seniors, al remapeo de horizonte, a la
//! fórmula de confianza, a la matriz de apalancamiento y a la ley fractal de
//! TP/SL. Tenía **tres fallos independientes**:
//!
//! **(1) No había multiescala.** El exponente de Hurst se define por cómo
//! escala una medida de fluctuación con la ventana de AGREGACIÓN τ:
//! `F(τ) ∝ τ^H`, luego `H = d[ln F(τ)] / d[ln τ]`. Estimarlo exige al menos dos
//! valores de τ y una regresión. La implementación anterior usaba
//! **exclusivamente retornos de un paso**: existía un único τ y la derivada
//! respecto a `ln τ` ni siquiera estaba definida.
//!
//! **(2) Lo que sí medía era curtosis.** Calculaba
//! `E[|r|] / (√E[r²] · √(2/π))`, que para una gaussiana vale **exactamente 1**
//! con independencia de la autocorrelación. Era ciego precisamente a lo que
//! Hurst mide.
//!
//! **(3) Dependía del número de muestras.** El denominador era `ln n` con `n`
//! el CONTADOR DE MUESTRAS, no una escala temporal, de modo que
//! `H(n) → 0,50` de forma determinista al crecer el proceso. El «régimen de
//! mercado» detectado era una función decreciente del tiempo transcurrido
//! desde el arranque.
//!
//! ## La consecuencia que explica el sesgo hacia el scalping
//!
//! Las tres «escalas» de `MultiScaleHurstConfluence` (10/25/50) diferían
//! **sólo en ese `ln n`**, con lo que la relación entre ellas era una constante
//! del sistema: `Δh_macro/Δh_micro = ln10/ln50 = 0,589`, siempre, para todo
//! mercado. Aplicando los umbrales de viabilidad:
//!
//! * `is_scalp_viable` exigía `|ln SR| > 0,230` → se cruza rutinariamente con
//!   la leptocurtosis normal de las criptomonedas;
//! * `is_swing_viable` exigía `|ln SR| > 0,587` → prácticamente inalcanzable.
//!
//! **El detector de régimen de horizonte largo estaba matemáticamente apagado.**
//! El sesgo hacia el scalping no era una decisión de diseño: era un artefacto
//! del denominador.
//!
//! # El estimador correcto
//!
//! DFA (*Detrended Fluctuation Analysis*) sobre agregaciones temporales REALES.
//! Para cada escala `s`, se divide el paseo integrado en ventanas de longitud
//! `s`, se elimina la tendencia lineal de cada una y se toma la fluctuación
//! cuadrática media. La pendiente de `ln F(s)` sobre `ln s` es `H`.
//!
//! Frente a R/S clásico, DFA es robusto a tendencias no estacionarias — que es
//! exactamente la condición de una serie de precios.
//!
//! ## Coste
//!
//! O(N) por escala en la actualización incremental, con el histórico acotado a
//! `MAX_HISTORY` retornos. Se recalcula cada `RECOMPUTE_EVERY` muestras, de
//! modo que el coste amortizado por tick es despreciable.

/// Escalas de agregación, en número de muestras. Log-espaciadas base 2 para
/// que la regresión sobre `ln s` tenga puntos uniformemente distribuidos.
/// Cubren de 4 a 256 muestras: una década y media de escala, suficiente para
/// una pendiente estable sin exigir un histórico enorme.
const DFA_SCALES: [usize; 7] = [4, 8, 16, 32, 64, 128, 256];

/// Histórico de retornos. Debe superar holgadamente la escala mayor para que
/// ésta tenga varias ventanas independientes.
const MAX_HISTORY: usize = 1024;

/// Cada cuántas muestras se recalcula la regresión completa.
const RECOMPUTE_EVERY: usize = 32;

/// Mínimo de muestras para que la estimación sea significativa: al menos
/// cuatro ventanas de la escala mayor.
const MIN_SAMPLES: usize = 512;

#[derive(Debug, Clone)]
pub struct HurstDfa {
    /// Retornos logarítmicos, buffer circular.
    returns: Vec<f64>,
    head: usize,
    filled: usize,
    last_price: f64,
    since_recompute: usize,
    /// Última estimación válida. 0,5 = difusión browniana (sin memoria).
    pub hurst: f64,
    /// Bondad del ajuste de la regresión log-log, en [0,1]. Un `r²` bajo
    /// significa que la serie NO sigue una ley de potencias y que el valor de
    /// `hurst` no debe usarse con convicción — información que el estimador
    /// anterior no podía siquiera expresar.
    pub r_squared: f64,
    /// `true` cuando hay muestras suficientes para una estimación con sentido.
    pub is_valid: bool,
}

impl Default for HurstDfa {
    fn default() -> Self {
        Self::new()
    }
}

impl HurstDfa {
    pub fn new() -> Self {
        Self {
            returns: vec![0.0; MAX_HISTORY],
            head: 0,
            filled: 0,
            last_price: 0.0,
            since_recompute: 0,
            hurst: 0.5,
            r_squared: 0.0,
            is_valid: false,
        }
    }

    /// Retornos acumulados en el histórico circular.
    #[inline]
    pub fn samples(&self) -> usize {
        self.filled
    }

    /// Alimenta un precio. Devuelve `(hurst, r_squared)`.
    pub fn update(&mut self, price: f64) -> (f64, f64) {
        if !price.is_finite() || price <= 0.0 {
            return (self.hurst, self.r_squared);
        }
        if self.last_price <= 0.0 {
            self.last_price = price;
            return (self.hurst, self.r_squared);
        }
        let r = (price / self.last_price).ln();
        self.last_price = price;
        if !r.is_finite() {
            return (self.hurst, self.r_squared);
        }

        self.returns[self.head] = r;
        self.head = (self.head + 1) % MAX_HISTORY;
        if self.filled < MAX_HISTORY {
            self.filled += 1;
        }
        self.since_recompute += 1;

        if self.filled >= MIN_SAMPLES && self.since_recompute >= RECOMPUTE_EVERY {
            self.since_recompute = 0;
            self.recompute();
        }
        (self.hurst, self.r_squared)
    }

    /// Retornos en orden cronológico (el más antiguo primero).
    fn ordered(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.filled);
        let start = if self.filled < MAX_HISTORY {
            0
        } else {
            self.head
        };
        for i in 0..self.filled {
            out.push(self.returns[(start + i) % MAX_HISTORY]);
        }
        out
    }

    /// Regresión de `ln F(s)` sobre `ln s`. La pendiente es H.
    fn recompute(&mut self) {
        let r = self.ordered();
        let n = r.len();
        if n < MIN_SAMPLES {
            return;
        }

        // Paseo integrado con la media eliminada: el objeto sobre el que DFA
        // mide fluctuación. Sin esta integración se estaría midiendo el ruido,
        // no su acumulación — que es donde vive la memoria del proceso.
        let mean = r.iter().sum::<f64>() / n as f64;
        let mut walk = Vec::with_capacity(n);
        let mut acc = 0.0;
        for &x in r.iter() {
            acc += x - mean;
            walk.push(acc);
        }

        let mut ln_s = Vec::with_capacity(DFA_SCALES.len());
        let mut ln_f = Vec::with_capacity(DFA_SCALES.len());

        for &s in DFA_SCALES.iter() {
            if s < 4 || n / s < 4 {
                continue; // menos de 4 ventanas: la escala no es estimable
            }
            let windows = n / s;
            let mut sum_sq = 0.0;
            for w in 0..windows {
                let seg = &walk[w * s..(w + 1) * s];
                // Ajuste lineal por mínimos cuadrados dentro de la ventana:
                // eliminar la tendencia local es lo que hace a DFA robusto
                // frente a la no estacionariedad de una serie de precios.
                let m = s as f64;
                let sum_x = (s * (s - 1)) as f64 / 2.0;
                let sum_xx = ((s - 1) * s * (2 * s - 1)) as f64 / 6.0;
                let mut sum_y = 0.0;
                let mut sum_xy = 0.0;
                for (i, &y) in seg.iter().enumerate() {
                    sum_y += y;
                    sum_xy += i as f64 * y;
                }
                let denom = m * sum_xx - sum_x * sum_x;
                let (a, b) = if denom.abs() > 1e-12 {
                    let slope = (m * sum_xy - sum_x * sum_y) / denom;
                    let intercept = (sum_y - slope * sum_x) / m;
                    (intercept, slope)
                } else {
                    (sum_y / m, 0.0)
                };
                let mut resid = 0.0;
                for (i, &y) in seg.iter().enumerate() {
                    let d = y - (a + b * i as f64);
                    resid += d * d;
                }
                sum_sq += resid / m;
            }
            let f_s = (sum_sq / windows as f64).sqrt();
            if f_s > 1e-15 && f_s.is_finite() {
                ln_s.push((s as f64).ln());
                ln_f.push(f_s.ln());
            }
        }

        if ln_s.len() < 3 {
            // Menos de tres puntos no define una pendiente con sentido: se
            // declara no válido en lugar de devolver un número inventado.
            self.is_valid = false;
            self.r_squared = 0.0;
            return;
        }

        let k = ln_s.len() as f64;
        let mx = ln_s.iter().sum::<f64>() / k;
        let my = ln_f.iter().sum::<f64>() / k;
        let mut sxy = 0.0;
        let mut sxx = 0.0;
        let mut syy = 0.0;
        for i in 0..ln_s.len() {
            let dx = ln_s[i] - mx;
            let dy = ln_f[i] - my;
            sxy += dx * dy;
            sxx += dx * dx;
            syy += dy * dy;
        }
        if sxx <= 1e-15 {
            self.is_valid = false;
            return;
        }
        let slope = sxy / sxx;
        // r²: cuánta de la variación de ln F(s) explica la ley de potencias.
        self.r_squared = if syy > 1e-15 {
            (sxy * sxy / (sxx * syy)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        // H físicamente admisible. [0,05, 0,95] cubre desde anti-persistencia
        // extrema hasta tendencia casi determinista.
        self.hurst = if slope.is_finite() {
            slope.clamp(0.05, 0.95)
        } else {
            0.5
        };
        self.is_valid = true;
    }

    /// Hurst sólo si el ajuste es fiable; en caso contrario, 0,5 (difusión sin
    /// memoria), que es la hipótesis nula honesta.
    ///
    /// El consumidor que quiera modular por convicción debe usar `r_squared`
    /// en lugar de tratar `hurst` como un número siempre significativo — que es
    /// lo que hacía el sistema con el estimador anterior.
    pub fn hurst_or_neutral(&self, min_r2: f64) -> f64 {
        if self.is_valid && self.r_squared >= min_r2 {
            self.hurst
        } else {
            0.5
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generador congruencial determinista: el test no puede depender del azar.
    struct Rng(u64);
    impl Rng {
        fn next_uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f64 / (u32::MAX as f64)) - 0.5
        }
        /// Normal aproximada por suma de uniformes (Irwin–Hall, k=12).
        fn next_normal(&mut self) -> f64 {
            (0..12).map(|_| self.next_uniform()).sum::<f64>()
        }
    }

    fn correr(precios: impl Iterator<Item = f64>) -> HurstDfa {
        let mut h = HurstDfa::new();
        for p in precios {
            h.update(p);
        }
        h
    }

    /// Un paseo aleatorio SIN memoria debe dar H ≈ 0,5. Es la calibración
    /// básica que el estimador anterior cumplía por construcción (siempre daba
    /// 0,5) y que éste debe cumplir por medición.
    #[test]
    fn paseo_aleatorio_da_hurst_medio() {
        let mut rng = Rng(42);
        let mut p = 60_000.0;
        let h = correr((0..4_000).map(|_| {
            p *= 1.0 + rng.next_normal() * 0.0008;
            p
        }));
        assert!(h.is_valid, "debe haber estimación válida con 4000 muestras");
        assert!(
            (h.hurst - 0.5).abs() < 0.14,
            "paseo aleatorio debe dar H ≈ 0,5, dio {} (r²={})",
            h.hurst,
            h.r_squared
        );
    }

    /// Serie PERSISTENTE (retornos autocorrelados positivamente) debe dar
    /// H > 0,5. El estimador anterior era estructuralmente incapaz de esto:
    /// era ciego a la autocorrelación.
    #[test]
    fn serie_persistente_da_hurst_alto() {
        let mut rng = Rng(7);
        let mut p = 60_000.0;
        let mut prev = 0.0;
        let h = correr((0..4_000).map(|_| {
            // AR(1) con phi alto: memoria larga positiva.
            let e = rng.next_normal() * 0.0008;
            prev = 0.85 * prev + e;
            p *= 1.0 + prev;
            p
        }));
        assert!(h.is_valid);
        assert!(
            h.hurst > 0.58,
            "serie persistente debe dar H > 0,58, dio {} (r²={})",
            h.hurst,
            h.r_squared
        );
    }

    /// Serie ANTI-PERSISTENTE (reversión a la media) debe dar H < 0,5.
    #[test]
    fn serie_antipersistente_da_hurst_bajo() {
        let mut rng = Rng(99);
        let mut p = 60_000.0;
        let mut prev = 0.0;
        let h = correr((0..4_000).map(|_| {
            // AR(1) con phi negativo: cada movimiento tiende a deshacerse.
            let e = rng.next_normal() * 0.0008;
            prev = -0.60 * prev + e;
            p *= 1.0 + prev;
            p
        }));
        assert!(h.is_valid);
        assert!(
            h.hurst < 0.42,
            "serie anti-persistente debe dar H < 0,42, dio {} (r²={})",
            h.hurst,
            h.r_squared
        );
    }

    /// D-615(3) — EL ESTIMADOR NO PUEDE DEPENDER DEL NÚMERO DE MUESTRAS.
    ///
    /// El anterior tenía `delta_h = ln(SR)/ln(n)` con `n` el contador de
    /// muestras, de modo que convergía deterministamente a 0,5 al correr el
    /// proceso: el «régimen» detectado era una función del tiempo transcurrido
    /// desde el arranque, no del mercado.
    #[test]
    fn d615_la_estimacion_no_deriva_con_el_numero_de_muestras() {
        let hacer = |n: usize| {
            let mut rng = Rng(2024);
            let mut p = 60_000.0;
            let mut prev = 0.0;
            correr((0..n).map(|_| {
                let e = rng.next_normal() * 0.0008;
                prev = 0.85 * prev + e;
                p *= 1.0 + prev;
                p
            }))
            .hurst
        };
        let h_corto = hacer(1_200);
        let h_largo = hacer(6_000);
        assert!(
            (h_corto - h_largo).abs() < 0.12,
            "la misma serie con más muestras no puede cambiar el régimen: \\
             {h_corto} vs {h_largo}"
        );
    }

    /// D-616 — LAS ESCALAS DEBEN SER TEMPORALES, NO TAMAÑOS DE MUESTRA.
    ///
    /// El estimador anterior tenía tres «escalas» que sólo diferían en el `ln n`
    /// del denominador, con lo que su cociente era una constante del sistema
    /// (0,589) para TODO mercado. Aquí una serie persistente y otra
    /// anti-persistente deben quedar a lados opuestos de 0,5: la escala mide
    /// estructura, no tamaño de ventana.
    #[test]
    fn d616_las_escalas_discriminan_estructura_real() {
        let mut rng = Rng(555);
        let mut p1 = 60_000.0;
        let mut prev1 = 0.0;
        let persistente = correr((0..4_000).map(|_| {
            let e = rng.next_normal() * 0.0008;
            prev1 = 0.85 * prev1 + e;
            p1 *= 1.0 + prev1;
            p1
        }));
        let mut rng2 = Rng(555);
        let mut p2 = 60_000.0;
        let mut prev2 = 0.0;
        let reversion = correr((0..4_000).map(|_| {
            let e = rng2.next_normal() * 0.0008;
            prev2 = -0.60 * prev2 + e;
            p2 *= 1.0 + prev2;
            p2
        }));
        assert!(
            persistente.hurst > reversion.hurst + 0.20,
            "el estimador debe separar persistencia de reversión: {} vs {}",
            persistente.hurst,
            reversion.hurst
        );
    }

    /// La bondad del ajuste debe ser alta cuando hay ley de potencias real.
    #[test]
    fn el_ajuste_log_log_es_bueno_en_series_con_escalamiento() {
        let mut rng = Rng(31337);
        let mut p = 60_000.0;
        let h = correr((0..4_000).map(|_| {
            p *= 1.0 + rng.next_normal() * 0.0008;
            p
        }));
        assert!(
            h.r_squared > 0.90,
            "un paseo aleatorio escala como una ley de potencias limpia: r²={}",
            h.r_squared
        );
    }

    /// Entradas corruptas no deben envenenar el estado.
    #[test]
    fn inmunidad_a_entradas_no_finitas() {
        let mut h = HurstDfa::new();
        for _ in 0..50 {
            h.update(f64::NAN);
            h.update(-1.0);
            h.update(0.0);
            h.update(f64::INFINITY);
        }
        assert_eq!(h.hurst, 0.5);
        assert!(!h.is_valid);
    }
}
