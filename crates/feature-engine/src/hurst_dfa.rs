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
//! **(2) Medía un descriptor marginal, no curtosis ni Hurst.** El cociente
//! `E[|r|] / (√E[r²] · √(2/π))` vale 1 en población para una gaussiana
//! centrada. No identifica la dependencia temporal; tampoco usa el cuarto
//! momento necesario para llamarlo curtosis.
//!
//! **(3) Dependía del número de muestras.** El denominador era `ln n` con `n`
//! el CONTADOR DE MUESTRAS: no efectuaba una regresión multiescala.
//! Si n creciera sin límite con descriptor
//! acotado se induciría convergencia a 0,50, pero el código actual limita
//! n por ventana: no se debe atribuirle una convergencia asintótica inexistente.
//!
//! ## La consecuencia que explica el sesgo hacia el scalping
//!
//! Las ventanas 10/25/50 combinaban el descriptor marginal con distintos
//! denominadores y umbrales bajo etiquetas de operación. El cociente de
//! desviaciones sería ln10/ln50 sólo si compartieran el MISMO descriptor;
//! sus muestras distintas impiden afirmar que sea constante en todo mercado.
//! Ninguna de esas etiquetas convierte el descriptor en exponente temporal.
//! La migración de consumidores sigue requiriendo versionar features y modelos.
//!
//! # Estimador implementado y límites de interpretación
//!
//! DFA (*Detrended Fluctuation Analysis*) sobre ventanas de MUESTRAS.
//! Para cada escala `s`, se divide el paseo integrado en ventanas de longitud
//! `s`, se elimina la tendencia lineal de cada una y se toma la fluctuación
//! cuadrática media. La pendiente de `ln F(s)` sobre `ln s` se interpreta
//! como H de los retornos sólo bajo un modelo estacionario de escalamiento.
//!
//! DFA1 elimina tendencias lineales del perfil integrado, no cualquier
//! no-estacionariedad de precios/retornos. Un R² alto no demuestra memoria
//! larga, causalidad, significación ni ausencia de sesgo de muestra finita.
//! Esta API no recibe timestamps: el caller debe justificar muestreo regular,
//! huecos y soporte temporal. No estima por sí sola un espectro 1 ns–100 años.
//!
//! ## Coste
//!
//! O(N·K) por recomputación, con asignaciones temporales y hasta K=7 escalas;
//! histórico acotado a `MAX_HISTORY`. Se recalcula cada `RECOMPUTE_EVERY`
//! muestras aceptadas. La latencia p99 debe medirse, no se presume despreciable.

/// Escalas de agregación, en número de muestras. Log-espaciadas base 2 para
/// que la regresión sobre `ln s` tenga puntos uniformemente distribuidos.
/// Cubren de 4 a 256 muestras. Es un soporte finito de estimación, no una
/// partición en motores de trading ni garantía de una pendiente estable.
const DFA_SCALES: [usize; 7] = [4, 8, 16, 32, 64, 128, 256];

/// Histórico de retornos. Debe superar holgadamente la escala mayor para que
/// ésta tenga varias ventanas independientes.
const MAX_HISTORY: usize = 1024;

/// Cada cuántas muestras se recalcula la regresión completa.
const RECOMPUTE_EVERY: usize = 32;

/// Warmup de política. A 512 retornos la mayor escala admisible es 128;
/// la escala 256 requiere las 1024 muestras (cuatro ventanas por escala).
const MIN_SAMPLES: usize = 512;

#[derive(Debug, Clone)]
pub struct HurstDfa {
    /// Retornos logarítmicos, buffer circular.
    returns: Vec<f64>,
    head: usize,
    filled: usize,
    last_price: f64,
    since_recompute: usize,
    /// Salida acotada de compatibilidad; 0,5 también es fallback sin evidencia,
    /// no prueba de difusión browniana. Consultar is_valid y raw_exponent.
    pub hurst: f64,
    /// Bondad descriptiva del ajuste log-log, en [0,1], no probabilidad de
    /// que exista una ley de potencias. R² alto no descarta crossovers o sesgo.
    pub r_squared: f64,
    /// Pendiente sin clipping, incluso fuera de (0,1); None si no hubo ajuste.
    /// Permite diagnosticar mala especificación sin ocultarla en una frontera.
    pub raw_exponent: Option<f64>,
    /// Número de escalas realmente utilizadas en el ajuste más reciente.
    pub scales_used: usize,
    /// Mayor escala utilizada, en muestras, NO milisegundos.
    pub max_scale: usize,
    /// Ajuste finito y pendiente dentro de (0,1), dominio del modelo de
    /// retornos estacionarios adoptado aquí. No es una prueba de estacionariedad.
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
            raw_exponent: None,
            scales_used: 0,
            max_scale: 0,
            is_valid: false,
        }
    }

    /// Retornos acumulados en el histórico circular.
    #[inline]
    pub fn samples(&self) -> usize {
        self.filled
    }

    /// Alimenta un precio. Devuelve `(hurst, r_squared)` del último ajuste.
    /// Ignorar un precio inválido no equivale a revalidar su frescura temporal.
    pub fn update(&mut self, price: f64) -> (f64, f64) {
        if !price.is_finite() || price <= 0.0 {
            return (self.hurst, self.r_squared);
        }
        if self.last_price <= 0.0 {
            self.last_price = price;
            return (self.hurst, self.r_squared);
        }
        // Within a factor of two, subtraction benefits from Sterbenz's range
        // and log1p retains nearby-price increments. Outside that range use
        // the ratio directly, avoiding cancellation of delta/p near -1.
        // If the ratio under/overflows, log-difference remains finite.
        let ratio = price / self.last_price;
        let r = if (0.5..=2.0).contains(&ratio) {
            ((price - self.last_price) / self.last_price).ln_1p()
        } else if ratio.is_finite() && ratio > 0.0 {
            ratio.ln()
        } else {
            price.ln() - self.last_price.ln()
        };
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

    /// Regresión de ln F(s) sobre ln s; interpretar la pendiente requiere modelo.
    fn recompute(&mut self) {
        self.is_valid = false;
        self.hurst = 0.5;
        self.r_squared = 0.0;
        self.raw_exponent = None;
        self.scales_used = 0;
        self.max_scale = 0;
        let r = self.ordered();
        let n = r.len();
        if n < MIN_SAMPLES {
            return;
        }

        // Paseo integrado con la media eliminada: el objeto sobre el que DFA
        // mide fluctuación. Sin esta integración se estaría midiendo el ruido,
        // no su acumulación — que es donde vive la memoria del proceso.
        // DFA is homogeneous: F(a*r)=|a|F(r). A common positive normalization
        // changes only the log-log intercept, not slope or R². This prevents
        // squared residual under/overflow and removes the absolute F>1e-15 gate.
        if r.iter().any(|x| !x.is_finite()) {
            return;
        }
        let amplitude = r.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
        if amplitude == 0.0 {
            return;
        }
        let mean = r.iter().map(|x| x / amplitude).sum::<f64>() / n as f64;
        let mut walk = Vec::with_capacity(n);
        let mut acc = 0.0;
        for &x in r.iter() {
            acc += x / amplitude - mean;
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
                // elimina sólo la componente lineal del perfil local; no
                // prueba robustez frente a cualquier no-estacionariedad.
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
            if f_s > 0.0 && f_s.is_finite() {
                ln_s.push((s as f64).ln());
                ln_f.push(f_s.ln());
                self.max_scale = s;
            }
        }

        self.scales_used = ln_s.len();
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
        if !sxx.is_finite() || sxx <= 0.0 || !syy.is_finite() || syy <= 0.0 {
            return;
        }
        let slope = sxy / sxx;
        if !slope.is_finite() {
            return;
        }
        self.raw_exponent = Some(slope);
        // r²: cuánta de la variación de ln F(s) explica la ley de potencias.
        let r2 = sxy / sxx.sqrt() / syy.sqrt();
        if !r2.is_finite() {
            return;
        }
        self.r_squared = (r2 * r2).clamp(0.0, 1.0);
        self.is_valid = slope > 0.0 && slope < 1.0;
        if self.is_valid {
            // Legacy output policy, not a physical law. Raw slope remains
            // available; an out-of-model slope is never certified by clipping.
            self.hurst = slope.clamp(0.05, 0.95);
        }
    }

    /// Salida bajo el dominio adoptado y un umbral descriptivo de R².
    /// En caso contrario 0,5 es fallback, no evidencia a favor de browniano.
    ///
    /// El consumidor que quiera modular por convicción debe usar `r_squared`
    /// en lugar de tratar `hurst` como un número siempre significativo — que es
    /// lo que hacía el sistema con el estimador anterior.
    pub fn hurst_or_neutral(&self, min_r2: f64) -> f64 {
        if min_r2.is_finite()
            && (0.0..=1.0).contains(&min_r2)
            && self.is_valid
            && self.r_squared.is_finite()
            && self.r_squared >= min_r2
        {
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
            // 53 random bits divided by 2^53: [0,1), then centered. The old
            // 31-bit numerator / 32-bit denominator produced only negatives.
            ((self.0 >> 11) as f64 / (1_u64 << 53) as f64) - 0.5
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
            // AR(1) con phi alto: correlación positiva de corto alcance;
            // la pendiente aparente en estas escalas no prueba memoria larga.
            let e = rng.next_normal() * 0.0008;
            prev = 0.85 * prev + e;
            p *= 1.0 + prev;
            p
        }));
        let raw = h.raw_exponent.expect("finite AR(1) scale fit");
        assert_eq!(h.is_valid, raw > 0.0 && raw < 1.0);
        assert!(
            raw > 0.58,
            "short-memory AR(1) must show an elevated finite-scale slope: {} (r²={})",
            raw,
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
            .raw_exponent
            .expect("finite AR(1) scale fit")
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
            persistente.raw_exponent.unwrap() > reversion.raw_exponent.unwrap() + 0.20,
            "finite-scale slopes must distinguish AR(1) structures: {:?} vs {:?}",
            persistente.raw_exponent,
            reversion.raw_exponent
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

    fn fit_returns(scale: f64) -> HurstDfa {
        let mut h = HurstDfa::new();
        let mut state = 7_u64;
        for r in &mut h.returns {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
            *r = scale * u;
        }
        h.filled = MAX_HISTORY;
        h.recompute();
        h
    }

    #[test]
    fn contract_dfa_is_invariant_to_return_amplitude() {
        let baseline = fit_returns(1.0);
        assert!(baseline.is_valid);
        for scale in [1e-200, 1e-16, 1e-8, 1e8, 1e200] {
            let h = fit_returns(scale);
            assert!(
                h.is_valid,
                "amplitude {scale} discarded an estimable series"
            );
            assert!((h.hurst - baseline.hurst).abs() < 1e-10);
            assert!((h.r_squared - baseline.r_squared).abs() < 1e-10);
        }
    }

    #[test]
    fn contract_finite_positive_prices_do_not_lose_extreme_log_returns() {
        let mut h = HurstDfa::new();
        h.update(f64::MIN_POSITIVE);
        h.update(f64::MAX);
        h.update(f64::MIN_POSITIVE);
        assert_eq!(
            h.samples(),
            2,
            "ratio overflow/underflow dropped valid observations"
        );
    }

    #[test]
    fn contract_nonstationary_return_trend_is_not_certified_as_stationary_hurst() {
        let mut h = HurstDfa::new();
        for (i, r) in h.returns.iter_mut().enumerate() {
            *r = i as f64;
        }
        h.filled = MAX_HISTORY;
        h.recompute();
        assert!(
            !h.is_valid,
            "out-of-model slope was clamped into a valid Hurst"
        );
        assert_eq!(h.hurst_or_neutral(0.85), 0.5);
    }

    #[test]
    fn contract_test_innovations_are_centered_and_two_sided() {
        let mut rng = Rng(42);
        let draws: Vec<f64> = (0..20_000).map(|_| rng.next_normal()).collect();
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let variance = draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / draws.len() as f64;
        assert!(
            draws.iter().any(|x| *x > 0.0),
            "all test innovations are negative"
        );
        assert!(mean.abs() < 0.05, "innovation mean={mean}");
        assert!(
            (variance - 1.0).abs() < 0.05,
            "innovation variance={variance}"
        );
    }

    #[test]
    fn contract_support_reports_actual_scales_and_clears_stale_estimates() {
        let mut h = fit_returns(1.0);
        assert_eq!((h.scales_used, h.max_scale), (7, 256));
        h.filled = MIN_SAMPLES;
        h.recompute();
        assert_eq!((h.scales_used, h.max_scale), (6, 128));
        h.returns.fill(2.0);
        h.recompute();
        assert!(!h.is_valid);
        assert_eq!((h.hurst, h.r_squared), (0.5, 0.0));
        assert_eq!(h.raw_exponent, None);
        assert_eq!((h.scales_used, h.max_scale), (0, 0));
    }

    #[test]
    fn contract_return_offset_preserves_detrended_slope() {
        let baseline = fit_returns(1.0);
        let mut shifted = baseline.clone();
        for x in &mut shifted.returns {
            *x += 10.0;
        }
        shifted.recompute();
        assert!(shifted.is_valid);
        assert!((shifted.hurst - baseline.hurst).abs() < 1e-10);
        assert!((shifted.r_squared - baseline.r_squared).abs() < 1e-10);
    }

    #[test]
    fn contract_invalid_quality_threshold_never_validates_a_fit() {
        let h = fit_returns(1.0);
        for bad in [f64::NAN, f64::INFINITY, -1.0, 2.0] {
            assert_eq!(h.hurst_or_neutral(bad), 0.5);
        }
    }

    #[test]
    fn contract_large_price_decline_preserves_log_return_accuracy() {
        let mut h = HurstDfa::new();
        h.update(1.0);
        h.update(1e-15);
        assert!(
            (h.returns[0] - 1e-15_f64.ln()).abs() < 1e-13,
            "observed={}, expected={}",
            h.returns[0],
            1e-15_f64.ln()
        );
    }
}
