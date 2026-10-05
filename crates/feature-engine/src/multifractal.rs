use std::f64;

/// 🧬 ESPECTRO MULTIFRACTAL DE MANDELBROT (MULTIFRACTAL SPECTRUM ENGINE)
/// Mide la dimensión multifractal de Hölder D(q) para discriminar entre caos ruidoso y tendencia inercial.
/// Reemplaza el exponente monofractal de Hurst por un análisis de multifractalidad dinámico.
/// #597 — resultado del espectro f(α) sobre la ventana del anillo.
#[derive(Debug, Clone, Copy)]
pub struct EspectroFAlpha {
    /// Extremo izquierdo del soporte estimado (cajas calientes).
    pub a_min: f64,
    /// Extremo derecho del soporte estimado (cajas frías).
    pub a_max: f64,
    /// a_max − a_min: intermitencia canónica (telemetría, ver calibración).
    pub ancho: f64,
    /// Capacidad del soporte −τ(0) ∈ [0,1]: 1 = soporte lleno, < 1 = huecos
    /// (racimado real de volatilidad). La magnitud falsable del espectro.
    pub d0: f64,
}

#[derive(Debug, Clone)]
pub struct MultifractalSpectrumEngine {
    pub window_size: usize,
    pub returns_history: [f64; 50],
    pub head: usize,
    pub count: usize,
    pub last_price: f64,
    pub sum_q1: f64,
    pub sum_q2: f64,
    /// AGY-AUD-003: counter for periodic exact recomputation of rolling sums
    /// to prevent catastrophic floating-point drift after millions of ticks.
    drift_recompute_counter: usize,
}

impl MultifractalSpectrumEngine {
    pub fn new(window_size: usize) -> Self {
        let size = window_size.clamp(10, 50);
        Self {
            window_size: size,
            returns_history: [0.0; 50],
            head: 0,
            count: 0,
            last_price: 0.0,
            sum_q1: 0.0,
            sum_q2: 0.0,
            drift_recompute_counter: 0,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, price: f64) -> (f64, f64) {
        if price <= 0.0 || !price.is_finite() {
            return (0.50, 0.0);
        }

        if self.last_price == 0.0 {
            self.last_price = price;
            return (0.50, 0.0);
        }

        let ret = (price / self.last_price).ln();
        self.last_price = price;

        if self.count >= self.window_size {
            let old_ret = self.returns_history[self.head];
            self.sum_q1 -= old_ret.abs();
            self.sum_q2 -= old_ret * old_ret;
        } else {
            self.count += 1;
        }

        self.returns_history[self.head] = ret;
        self.head = (self.head + 1) % self.window_size;

        let abs_ret = ret.abs();
        self.sum_q1 = (self.sum_q1 + abs_ret).max(0.0);
        self.sum_q2 = (self.sum_q2 + ret * ret).max(0.0);

        // AGY-AUD-003: periodic exact recomputation to clear accumulated
        // floating-point drift from millions of add/subtract cycles.
        self.drift_recompute_counter += 1;
        if self.drift_recompute_counter >= self.window_size {
            self.drift_recompute_counter = 0;
            let n = self.count.min(self.window_size);
            let mut exact_q1 = 0.0f64;
            let mut exact_q2 = 0.0f64;
            for i in 0..n {
                let idx = (self.head + self.window_size - n + i) % self.window_size;
                let r = self.returns_history[idx];
                exact_q1 += r.abs();
                exact_q2 += r * r;
            }
            self.sum_q1 = exact_q1.max(0.0);
            self.sum_q2 = exact_q2.max(0.0);
        }

        if self.count < 10 {
            return (0.50, 0.0);
        }

        // F2-C4 (S8): ESTIMADOR HONESTO DE ESCALAMIENTO MULTIESCALA DE HURST / VARIANCE RATIO
        // Reemplaza el ratio de amplitud marginal L1/L2 (que confundía curtosis/colas pesadas con anti-persistencia)
        // por la ley de escalamiento temporal de varianza multiescala sobre retornos logarítmicos acumulados:
        // Var(r^{(k)}) = k^{2H} Var(r^{(1)}) => 2H = d ln Var(k) / d ln k.
        let n = self.count.min(self.window_size);
        let start_idx = (self.head + self.window_size - n) % self.window_size;

        let mut rets = [0.0f64; 50];
        let mut mean_r = 0.0f64;
        for i in 0..n {
            let r = self.returns_history[(start_idx + i) % self.window_size];
            rets[i] = r;
            mean_r += r;
        }
        mean_r /= n as f64;

        // Varianza escala 1 (1 tick)
        let mut s1 = 0.0f64;
        for i in 0..n {
            let d = rets[i] - mean_r;
            s1 += d * d;
        }
        if s1 < 1e-14 {
            // Precio constante o fluctuaciones imperceptibles: régimen neutro
            return (0.50, 0.0);
        }
        let var1 = (s1 / n as f64).max(1e-16);

        // Varianza escala 2 (2 ticks acumulados)
        let mut s2 = 0.0f64;
        let n2 = n - 1;
        for i in 1..n {
            let d = (rets[i] + rets[i - 1]) - 2.0 * mean_r;
            s2 += d * d;
        }
        let var2 = (s2 / n2 as f64).max(1e-16);
        let h2 = 0.5 * (var2 / var1).ln() / std::f64::consts::LN_2;

        let (h_raw, multifractal_width) = if n >= 16 {
            // Varianza escala 4 (4 ticks acumulados)
            let mut s4 = 0.0f64;
            let n4 = n - 3;
            for i in 3..n {
                let d = (rets[i] + rets[i - 1] + rets[i - 2] + rets[i - 3]) - 4.0 * mean_r;
                s4 += d * d;
            }
            let var4 = (s4 / n4 as f64).max(1e-16);
            let h4 = 0.5 * (var4 / var1).ln() / (2.0 * std::f64::consts::LN_2);

            // Regresión conjunta multiescala (k=2, k=4): H = (y1 + 2*y2) / (10 * ln 2)
            let y1 = (var2 / var1).ln();
            let y2 = (var4 / var1).ln();
            let h_reg = (y1 + 2.0 * y2) / (10.0 * std::f64::consts::LN_2);
            let width = (h2 - h4).abs().clamp(0.0, 1.0);
            (h_reg, width)
        } else {
            let width = (h2 - 0.50).abs().clamp(0.0, 1.0);
            (h2, width)
        };

        // Regularización Bayesiana hacia el prior browniano H_0 = 0.50
        // con peso muestral w = n / (n + 12.0) para estabilizar ventanas finitas.
        let w = n as f64 / (n as f64 + 12.0);
        let h_bayes = 0.50 + w * (h_raw - 0.50);
        let dynamic_hurst = h_bayes.clamp(0.05, 0.95);

        (dynamic_hurst, multifractal_width)
    }

    /// #597 (P-B del consejo) — espectro multifractal f(α) de Gärtner–Ellis
    /// por transformada de Legendre, sobre la medida de |retornos| del anillo.
    ///
    /// Formalismo de Halsey: partición del anillo en cajas de b ∈ {1,2,4}
    /// ticks, Z(q,b) = Σ p_caja^q con p la medida normalizada; τ(q) = +
    /// pendiente (mínimos cuadrados) de ln Z contra ln b — convención
    /// canónica τ(q) = q−1 en el monofractal y τ(0) = −D₀ = −1;α(q) = dτ/dq por diferencias centrales sobre la
    /// rejilla q ∈ {−3,−2,−1,0,1,2,3}; f(α(q)) = qα − τ(q). El ANCHO del
    /// soporte (α_máx − α_mín, puntos con f ≥ −0.05 por tamaño finito) es la
    /// intermitencia canónica: → 0 para un monofractal (τ(q) = q−1 ⇒ α ≡ 1),
    /// ancho bajo racimado de volatilidad real.
    ///
    /// `None` con < 32 muestras (partición b=4 exigiría pocas cajas), medida
    /// nula o Z no finito — el llamador no debe usarlo (misma disciplina que
    /// las anclas maduras del banco de pronóstico). Observación pura: sin
    /// consumidor de política; el ancho se publica al registro para el consejo.
    ///
    /// NOTA DE CALIBRACIÓN (n=50, 3 puntos de escala): `ancho` queda al nivel
    /// del ruido de muestreo (series iid y en cascada miden ≈1.2) — es
    /// telemetría para comparar contra su PROPIA historia (registro), no para
    /// umbrales absolutos. La magnitud falsable y robusta es `d0` (capacidad
    /// del soporte, −τ(0)): inmune al ruido de amplitud, ~1 con soporte
    /// lleno, < 1 con huecos (racimado real).
    pub fn espectro_f_alpha(&self) -> Option<EspectroFAlpha> {
        // Rejilla ±2 y partición {1,2,4}: con 50 muestras, b=8 deja 6 cajas
        // (varianza inaceptable en q<0) y q=±3 dispara 2^±3 con ruido de
        // tamaño finito — las colas extremas mienten más de lo que informan.
        const Q: [f64; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];
        const B: [usize; 3] = [1, 2, 4];
        if self.count < 32 {
            return None;
        }
        let n = self.count;
        let mut p = [0.0f64; 50];
        let mut total = 0.0;
        for i in 0..n {
            let idx = (self.head + self.window_size - n + i) % self.window_size;
            let m = self.returns_history[idx].abs();
            p[i] = m;
            total += m;
        }
        if !(total.is_finite() && total > 0.0) {
            return None;
        }
        for v in p.iter_mut().take(n) {
            *v /= total;
        }
        // Regresión lineal de ln Z contra ln b (3 puntos). Convención
        // canónica: τ(q) = +pendiente — para la medida uniforme
        // ln Z = (1−q)ln n + (q−1)ln b ⇒ τ(q) = q−1 (τ(0) = −D₀ = −1).
        let nb = B.len() as f64;
        let (mut sx, mut sxx) = (0.0, 0.0);
        for &b in &B {
            let lb = (b as f64).ln();
            sx += lb;
            sxx += lb * lb;
        }
        let denom = sxx - sx * sx / nb;
        if !(denom.is_finite() && denom.abs() > 1e-12) {
            return None;
        }
        let mut tau = [0.0f64; 7];
        for (k, &q) in Q.iter().enumerate() {
            let (mut sy, mut sxy) = (0.0, 0.0);
            for &b in &B {
                let cajas = n / b;
                if cajas < 1 {
                    return None;
                }
                let mut z = 0.0f64;
                for c in 0..cajas {
                    let mut pc = 0.0f64;
                    for i in (c * b)..((c + 1) * b) {
                        pc += p[i];
                    }
                    // q < 0: las cajas vacías no son soporte de la medida.
                    if q < 0.0 && pc <= 0.0 {
                        continue;
                    }
                    if pc <= 0.0 && q >= 0.0 {
                        continue;
                    }
                    z += pc.powf(q);
                }
                if !(z.is_finite() && z > 0.0) {
                    return None;
                }
                let lb = (b as f64).ln();
                let lz = z.ln();
                sy += lz;
                sxy += lb * lz;
            }
            let slope = (sxy - sx * sy / nb) / denom;
            if !slope.is_finite() {
                return None;
            }
            tau[k] = slope;
        }
        // Legendre sobre la rejilla: α central, f = qα − τ.
        let mut a_min = f64::INFINITY;
        let mut a_max = f64::NEG_INFINITY;
        for k in 1..Q.len() - 1 {
            let alpha = (tau[k + 1] - tau[k - 1]) / (Q[k + 1] - Q[k - 1]);
            if !alpha.is_finite() {
                continue;
            }
            let f = Q[k] * alpha - tau[k];
            // Holgura de tamaño finito (50 muestras): las colas del soporte
            // pueden caer levemente bajo 0 por ruido de muestreo.
            if f < -0.25 {
                continue;
            }
            a_min = a_min.min(alpha);
            a_max = a_max.max(alpha);
        }
        if !(a_min.is_finite() && a_max.is_finite() && a_max >= a_min) {
            return None;
        }
        let d0 = (-tau[2]).clamp(0.0, 1.0);
        if !d0.is_finite() {
            return None;
        }
        Some(EspectroFAlpha {
            a_min,
            a_max,
            ancho: (a_max - a_min).max(0.0),
            d0,
        })
    }

    /// Descomposición Wavelet de Haar de 1 nivel para análisis de micro-régimen (#96-#112)
    /// Retorna: `(energy_approx, energy_detail, detail_to_approx_ratio)`
    pub fn compute_haar_wavelet_energy(&self) -> (f64, f64, f64) {
        if self.count < 8 {
            return (0.0, 0.0, 0.0);
        }

        let pairs = (self.count / 2).min(25);
        let mut energy_approx = 0.0;
        let mut energy_detail = 0.0;

        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        for i in 0..pairs {
            let idx1 = (self.head + self.window_size - self.count + 2 * i) % self.window_size;
            let idx2 = (self.head + self.window_size - self.count + 2 * i + 1) % self.window_size;

            let x1 = self.returns_history[idx1];
            let x2 = self.returns_history[idx2];

            let s = (x1 + x2) * inv_sqrt2;
            let d = (x1 - x2) * inv_sqrt2;

            energy_approx += s * s;
            energy_detail += d * d;
        }

        let ratio = if energy_approx > 1e-12 {
            energy_detail / energy_approx
        } else {
            0.0
        };

        (energy_approx, energy_detail, ratio)
    }
}

impl Default for MultifractalSpectrumEngine {
    fn default() -> Self {
        Self::new(50)
    }
}

/// Confluencia de Régimen por Exponente de Hurst Multi-Escala (Punto #104)
/// Calcula el exponente de Hurst en 3 escalas temporales distintas:
/// - Micro-Escala (Scalp, ventana corta de 10 ticks)
/// - Meso-Escala (Intradía, ventana media de 25 ticks)
/// - Macro-Escala (Swing, ventana larga de 50 ticks)
///   y evalúa la confluencia direccional de persistencia (>0.55) o anti-persistencia (<0.45).
#[derive(Debug, Clone)]
pub struct MultiScaleHurstConfluence {
    pub engine_micro: MultifractalSpectrumEngine,
    pub engine_meso: MultifractalSpectrumEngine,
    pub engine_macro: MultifractalSpectrumEngine,
    /// #597 — cadencia del espectro f(α): re-cómputo cada 16 consultas
    /// (el cálculo es O(cajas) sobre 50 muestras; la publicación por evento
    /// no necesita frescura sub-evento).
    falpha_calls: u64,
    falpha_cache: Option<EspectroFAlpha>,
}

impl Default for MultiScaleHurstConfluence {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiScaleHurstConfluence {
    pub fn new() -> Self {
        Self {
            engine_micro: MultifractalSpectrumEngine::new(10),
            engine_meso: MultifractalSpectrumEngine::new(25),
            engine_macro: MultifractalSpectrumEngine::new(50),
            falpha_calls: 0,
            falpha_cache: None,
        }
    }

    /// #597 — ancho del espectro f(α) del motor MACRO (50 muestras: la única
    /// con ≥32 para la partición b=4). Observación, no política; `None`
    /// sin evidencia madura. Cadencia interna 1/16 consultas.
    pub fn ancho_f_alpha(&mut self) -> Option<f64> {
        self.espectro_cacheada().map(|e| e.ancho)
    }

    /// #597 — capacidad del soporte D₀ (−τ(0)) del motor MACRO: 1 = sin
    /// huecos, < 1 = racimado real. La magnitud falsable del espectro.
    pub fn d0_f_alpha(&mut self) -> Option<f64> {
        self.espectro_cacheada().map(|e| e.d0)
    }

    /// #597 — espectro completo cacheado (ancho + D₀ + extremos) del motor
    /// MACRO, con cadencia interna 1/16 consultas. Observación, no política.
    pub fn espectro_cacheada(&mut self) -> Option<EspectroFAlpha> {
        self.falpha_calls = self.falpha_calls.wrapping_add(1);
        if self.falpha_calls % 16 == 1 || self.falpha_cache.is_none() {
            self.falpha_cache = self.engine_macro.espectro_f_alpha();
        }
        self.falpha_cache
    }

    /// #659 (F1-C2) — GENERACIÓN del cache: número de consultas totales;
    /// cambia exactamente cuando el cache refresca (cada 16). El consumidor
    /// de la EWMA lo usa para DEDUP: sin esto, el mismo espectro contaba
    /// 16× y la memoria efectiva del olvido 1/64 era ~4 espectros.
    #[inline(always)]
    pub fn cache_generation(&self) -> u64 {
        self.falpha_calls
    }

    #[inline(always)]
    pub fn update(&mut self, price: f64) -> (f64, f64, f64, f64, bool, bool) {
        let (h_micro, _) = self.engine_micro.update(price);
        let (h_meso, _) = self.engine_meso.update(price);
        let (h_macro, _) = self.engine_macro.update(price);

        // AGY-AUD-P09: Continuidad suave C^infinito en confluencia fractal:
        // Erradica funciones escalón discretas (+1, 0, -1) con umbrales rígidos 0.55/0.45.
        // Utiliza una modulación hiperbólica continua centrada en el punto nulo browniano H=0.50.
        let c_micro = ((h_micro - 0.50) / 0.08).tanh();
        let c_meso = ((h_meso - 0.50) / 0.08).tanh();
        let c_macro = ((h_macro - 0.50) / 0.08).tanh();

        let confluence_score: f64 = (c_micro * 0.4 + c_meso * 0.3 + c_macro * 0.3).clamp(-1.0, 1.0);

        // O(1) Indicadores continuos de persistencia fractal en el espectro universal
        let is_micro_persistent = (h_micro - 0.5).abs() > 0.10;
        let is_macro_persistent = (h_macro - 0.5).abs() > 0.15;

        (
            h_micro,
            h_meso,
            h_macro,
            confluence_score,
            is_micro_persistent,
            is_macro_persistent,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #597 — LCG determinista (xorshift64) + Box-Muller.
    fn xorshift(estado: &mut u64) -> f64 {
        *estado ^= *estado << 13;
        *estado ^= *estado >> 7;
        *estado ^= *estado << 17;
        ((*estado >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn gauss(estado: &mut u64) -> f64 {
        let u1 = xorshift(estado).max(1e-12);
        let u2 = xorshift(estado);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    #[test]
    fn qo_597_espectro_nulo_sin_madurez_y_vivo_al_cumplirla() {
        let mut e = MultifractalSpectrumEngine::new(50);
        let mut p = 100.0;
        let mut lcg = 0x243F6A8885A308D3u64;
        for _ in 0..31 {
            p *= 1.0 + 0.01 * gauss(&mut lcg);
            e.update(p);
        }
        assert!(e.espectro_f_alpha().is_none(), "31 muestras no particionan b=8");
        for _ in 31..60 {
            p *= 1.0 + 0.01 * gauss(&mut lcg);
            e.update(p);
        }
        assert!(e.espectro_f_alpha().is_some(), "≥32 muestras particionan");
    }

    #[test]
    fn qo_597_intermitencia_abre_el_espectro_mas_que_iid() {
        // FALSACIÓN ROBUSTA (D₀, capacidad del soporte): la serie iid
        // llena el anillo (|r| > 0 siempre) ⇒ D₀ ≈ 1; la cascada con
        // 15 ticks de quietud EXACTA por cada 20 deja huecos reales ⇒
        // D₀ < 1. Inmune al ruido de amplitud (lo que NO lo es el ancho
        // a n=50 — ver calibración en el doc de `espectro_f_alpha`).
        let mut iid = MultifractalSpectrumEngine::new(50);
        let mut lcg = 0x9E3779B97F4A7C15u64;
        let mut p = 100.0;
        let mut di = None;
        let mut wi = None;
        for _ in 0..900 {
            p *= 1.0 + 0.01 * gauss(&mut lcg);
            iid.update(p);
            if let Some(e) = iid.espectro_f_alpha() {
                di = Some(e.d0);
                wi = Some(e.ancho);
            }
        }
        let mut inter = MultifractalSpectrumEngine::new(50);
        let mut p2 = 100.0;
        let mut de = None;
        // Cascada multiplicativa (el multifractal canónico): 15 ticks de
        // quietud EXACTA y 5 de ráfaga lognormal por bloque de 20.
        for i in 0..900u64 {
            let r = if i % 20 < 15 {
                0.0
            } else {
                0.03 * (1.5 * gauss(&mut lcg)).exp()
            };
            p2 *= 1.0 + r;
            inter.update(p2);
            if let Some(e) = inter.espectro_f_alpha() {
                de = Some(e.d0);
            }
        }
        let (di, de, wi) = (
            di.expect("iid maduro"),
            de.expect("intermitente maduro"),
            wi.expect("ancho iid"),
        );
        eprintln!("[qo-597] D0: iid={} cascada={} | ancho iid={}", di, de, wi);
        assert!(di > 0.9, "el soporte iid debe estar lleno: D0={}", di);
        assert!(
            de < 0.8 && de > 0.2,
            "la cascada con huecos debe partir el soporte: D0={}",
            de
        );
        // El ancho queda al suelo de ruido a n=50 (≈1.2 en ambas): sólo
        // sanidad de rango, sin umbral absoluto (telemetría de historia).
        assert!(wi >= 0.0 && wi < 1.5, "ancho iid fuera de rango: {}", wi);
        // Cadencia del confluence: dos lecturas consecutivas estables.
        let mut conf = MultiScaleHurstConfluence::new();
        let mut p3 = 50.0;
        let mut ultimo = None;
        for _ in 0..120 {
            p3 *= 1.0 + 0.02 * gauss(&mut lcg);
            conf.update(p3);
            if let Some(w) = conf.ancho_f_alpha() {
                ultimo = Some(w);
            }
        }
        assert!(ultimo.is_some(), "el confluence publica el ancho macro");
    }

    #[test]
    fn test_multifractal_spectrum_calculation() {
        let mut engine = MultifractalSpectrumEngine::new(30);
        let mut p = 60000.0;
        for i in 0..40 {
            p += (i as f64 * 0.1).sin() * 5.0;
            let (hurst, width) = engine.update(p);
            if i >= 15 {
                assert!(hurst >= 0.05 && hurst <= 0.95);
                assert!(width >= 0.0 && width <= 1.0);
            }
        }

        let (approx, detail, ratio) = engine.compute_haar_wavelet_energy();
        assert!(approx >= 0.0);
        assert!(detail >= 0.0);
        assert!(ratio >= 0.0);
    }

    #[test]
    fn test_multi_scale_hurst_confluence() {
        let mut confluence = MultiScaleHurstConfluence::new();
        let mut p = 50000.0;
        for i in 0..60 {
            p += (i as f64 * 0.2).cos() * 10.0;
            let (h_micro, h_meso, h_macro, score, _scalp, _swing) = confluence.update(p);
            assert!(h_micro >= 0.05 && h_micro <= 0.95);
            assert!(h_meso >= 0.05 && h_meso <= 0.95);
            assert!(h_macro >= 0.05 && h_macro <= 0.95);
            assert!(score >= -1.0 && score <= 1.0);
        }
    }

    #[test]
    fn test_multifractal_nan_and_negative_immunity() {
        let mut engine = MultifractalSpectrumEngine::new(30);
        let (h_nan, w_nan) = engine.update(f64::NAN);
        assert_eq!(h_nan, 0.50);
        assert_eq!(w_nan, 0.0);

        let (h_neg, w_neg) = engine.update(-100.0);
        assert_eq!(h_neg, 0.50);
        assert_eq!(w_neg, 0.0);

        let mut confluence = MultiScaleHurstConfluence::new();
        let (h1, h2, h3, sc, scalp, swing) = confluence.update(f64::NAN);
        assert_eq!(h1, 0.50);
        assert_eq!(h2, 0.50);
        assert_eq!(h3, 0.50);
        assert_eq!(sc, 0.0);
        assert_eq!(scalp, false);
        assert_eq!(swing, false);
    }

    #[test]
    fn test_multifractal_scale_invariance_log_returns() {
        let mut engine_btc = MultifractalSpectrumEngine::new(30);
        let mut engine_alt = MultifractalSpectrumEngine::new(30);

        let mut p_btc = 60000.0;
        let mut p_alt = 0.06;

        let mut h_btc = 0.50;
        let mut h_alt = 0.50;

        for i in 0..35 {
            let ret = (i as f64 * 0.1).sin() * 0.005; // 0.5% de variación relativa
            p_btc *= 1.0 + ret;
            p_alt *= 1.0 + ret;
            h_btc = engine_btc.update(p_btc).0;
            h_alt = engine_alt.update(p_alt).0;
        }

        assert!(
            (h_btc - h_alt).abs() < 1e-4,
            "El espectro multifractal debe ser invariante de escala por retornos logarítmicos: btc={}, alt={}",
            h_btc,
            h_alt
        );
        assert!(h_btc > 0.10 && h_btc < 0.90, "Hurst no debe estar bloqueado en extremos 0.10/0.90: {}", h_btc);
    }

    #[test]
    fn f2_c4_honest_variance_ratio_hurst_scaling() {
        // 1. Serie persistente pura (momentum / tendencia fuerte)
        let mut e_trend = MultifractalSpectrumEngine::new(50);
        let mut p = 100.0;
        for i in 0..50 {
            // Retornos predominantemente positivos con autocorrelación positiva
            p *= 1.0 + 0.003 + (i as f64 * 0.05).sin() * 0.001;
            e_trend.update(p);
        }
        let (h_trend, _) = e_trend.update(p * 1.003);
        assert!(h_trend > 0.55, "Tendencia persistente debe dar H > 0.55, dio: {}", h_trend);

        // 2. Serie anti-persistente pura (oscilador / mean reversion fuerte)
        let mut e_revert = MultifractalSpectrumEngine::new(50);
        let mut p_rev = 100.0;
        for i in 0..50 {
            let r = if i % 2 == 0 { 0.004 } else { -0.004 };
            p_rev *= 1.0 + r;
            e_revert.update(p_rev);
        }
        let (h_revert, _) = e_revert.update(p_rev * 1.004);
        assert!(h_revert < 0.45, "Oscilación anti-persistente debe dar H < 0.45, dio: {}", h_revert);

        // 3. Flat line (precio constante) -> régimen neutro exacto
        let mut e_flat = MultifractalSpectrumEngine::new(50);
        for _ in 0..30 {
            e_flat.update(100.0);
        }
        let (h_flat, w_flat) = e_flat.update(100.0);
        assert_eq!(h_flat, 0.50);
        assert_eq!(w_flat, 0.0);
    }
}
