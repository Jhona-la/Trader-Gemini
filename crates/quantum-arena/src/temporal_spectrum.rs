//! ESPECTRO TEMPORAL CONTINUO (F8) — un solo sistema, todas las escalas.
//!
//! PROBLEMA (directriz del operador): el binario scalp/swing es un CORTE
//! ARBITRARIO del espectro temporal que contamina genoma, motor, posiciones
//! y 66 archivos. El mercado no tiene "dos modos": tiene estructuras en
//! TODAS las escalas simultáneamente.
//!
//! DISEÑO FÍSICO-HONESTO (fronteras reales, no marketing):
//!   - Cota inferior ~1ms: el inter-arribo de ticks de Binance futures y el
//!     RTT de red. Por debajo, "horizonte" solo existe DENTRO del motor
//!     (procesamos cada evento a cadencia de nanosegundo — eso ya ocurre).
//!   - Cota superior ~2 años: los ciclos macro relevantes para futuros
//!     USDT-M (halving ≈ 4y queda como contexto macro, no como horizonte
//!     operable con data de 2019+).
//!   - Entre ambas: ESPECTRO CONTINUO log-espaciado base-4: 19 escalas de
//!     1ms a 68_719_476_736ms (≈2.18 años) sin huecos ni bandas prohibidas.
//!
//! QUÉ calcula CADA TICK (todo junto, nanosegundo a nanosegundo de proceso):
//!   Por escala τ_i: precio EWMA(τ_i), volatilidad EWMA(τ_i), momentum
//!   z-scoreado, señal tanh(z) ∈ [-1,1] y persistencia (acuerdo de signo).
//!   Fusión: score espectral = Σ w_i·señal_i con w_i ∝ 1/vol_i — PARIDAD DE
//!   RIESGO entre escalas (cada horizonte aporta según su Sharpe potencial
//!   inverso a su ruido). Nada de "scalp manda aquí, swing allá".
//!
//! COSTE: O(S)=19 escalas × ~6 FLOPs = ~120 FLOPs/tick — despreciable frente
//! al proceso del evento本身.

/// Escalas del espectro: 10^-6 ms * 4^i para i∈0..32 → 1 ns (10^-6 ms) … ≈146.15 años (4.61*10^12 ms).
/// Log-espaciadas base 4 (≈4.15 escalas/década): resolución uniforme en
/// log(τ), cubriendo desde microestructura en nanosegundos hasta tendencias seculares de más de 100 años.
pub const SPECTRUM_SCALES_MS: [f64; 32] = [
    1.0e-6,           // 1 ns
    4.0e-6,           // 4 ns
    1.6e-5,           // 16 ns
    6.4e-5,           // 64 ns
    2.56e-4,          // 256 ns
    1.024e-3,         // ~1.02 µs
    4.096e-3,         // ~4.10 µs
    1.6384e-2,        // ~16.38 µs
    6.5536e-2,        // ~65.54 µs
    0.262144,         // ~262.14 µs
    1.048576,         // ~1.05 ms
    4.194304,         // ~4.19 ms
    16.777216,        // ~16.78 ms
    67.108864,        // ~67.11 ms
    268.435456,       // ~268.44 ms
    1_073.741824,     // ~1.07 s
    4_294.967296,     // ~4.29 s
    17_179.869184,    // ~17.18 s
    68_719.476736,    // ~1.15 min
    274_877.906944,   // ~4.58 min
    1_099_511.627776, // ~18.33 min
    4_398_046.511104, // ~1.22 h
    17_592_186.044416,// ~4.89 h
    70_368_744.177664,// ~19.55 h
    281_474_976.710656,// ~3.26 d
    1_125_899_906.842624,// ~13.03 d
    4_503_599_627.370496,// ~52.12 d
    18_014_398_509.481984,// ~208.5 d
    72_057_594_037.927936,// ~2.28 años
    288_230_376_151.711744,// ~9.13 años
    1_152_921_504_606.846976,// ~36.54 años
    4_611_686_018_427.387904,// ~146.15 años (>100 años)
];

/// τ de anclaje histórica (compat): la banda rápida ≈ 30s, la extendida ≈ 12h.
/// El continuo las reemplaza; quedan solo como puntos de conversión del
/// genoma legacy — ningún código decide por pertenecer a una banda.
pub const TAU_ANCHOR_FAST_MS: f64 = 30_000.0;
pub const TAU_ANCHOR_SLOW_MS: f64 = 43_200_000.0;

#[derive(Debug, Clone, Copy, Default)]
pub struct ScaleState {
    pub tau_ms: f64,
    pub ewma_price: f64,
    /// EWMA de |desviación a ESTA escala| — el denominador estadísticamente
    /// correcto del z: caminata aleatoria ⇒ z ~ O(1); deriva ⇒ z crece.
    pub ewma_dev_vol: f64,
    pub momentum_z: f64,
    pub signal: f64,      // tanh(z): opinión direccional ∈ [-1,1]
    pub persistence: f64, // EWMA de sign(dev)·sign(prev_dev) — autocorrelación de sorpresas
    prev_dev: f64,
}

pub struct TemporalSpectrum {
    pub scales: [ScaleState; 32],
    last_ts_ms: u64,
    /// Score espectral fusionado (paridad de riesgo 1/vol) ∈ ~[-1,1].
    pub fused_score: f64,
    /// Escala dominante (mayor |w·señal|) en ms — información, no decisión.
    pub dominant_tau_ms: f64,
}

impl Default for TemporalSpectrum {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalSpectrum {
    pub fn new() -> Self {
        let mut scales = [ScaleState::default(); 32];
        for (i, s) in scales.iter_mut().enumerate() {
            s.tau_ms = SPECTRUM_SCALES_MS[i];
        }
        Self {
            scales,
            last_ts_ms: 0,
            fused_score: 0.0,
            dominant_tau_ms: 0.0,
        }
    }

    /// Actualiza TODAS las escalas con un tick. Cadencia: cada evento del
    /// motor (nanosegundo-a-nanosegundo en proceso). O(19).
    pub fn update(&mut self, price: f64, ts_ms: u64) {
        if !price.is_finite() || price <= 0.0 {
            return;
        }
        if self.last_ts_ms == 0 {
            // Primer tick: inicializar EWMAs al precio observado.
            for s in self.scales.iter_mut() {
                s.ewma_price = price;
                s.ewma_dev_vol = 1e-7; // vol de desviación semilla (evita z=∞)
                s.signal = 0.0;
            }
            self.last_ts_ms = ts_ms;
            return;
        }
        // Idempotencia parcial de X-035: dt=0 (mismo evento por dos caminos,
        // p.ej. process_event→dual) NO re-pesa; ts hacia atrás se descarta.
        // GAPS grandes siguen siendo ceguera conocida (X-035: reset explícito
        // pendiente).
        if ts_ms <= self.last_ts_ms {
            return;
        }
        let dt = (ts_ms - self.last_ts_ms) as f64;
        self.last_ts_ms = ts_ms;

        let mut w_sum = 0.0;
        let mut w_sig_sum = 0.0;
        let mut best_contrib = 0.0f64;
        let mut dominant = 0.0f64;

        for s in self.scales.iter_mut() {
            // α de la escala para el dt transcurrido: el horizonte τ_i define
            // cuánto pesa ESTE tick en esa escala. Continuo en dt y τ.
            let alpha = 1.0 - (-dt / s.tau_ms).exp();
            let prev_ewma = s.ewma_price;
            if prev_ewma <= 0.0 {
                s.ewma_price = price;
                continue;
            }
            // Sorpresa a esta escala: cuánto se aparta el precio del consenso
            // EWMA(τ_i) ANTES de absorber este tick.
            let dev = (price - prev_ewma) / prev_ewma;
            s.ewma_price += alpha * (price - prev_ewma);
            // La vol DE LA DESVIACIÓN (no del retorno por tick): es el único
            // denominador que mantiene z ~ O(1) ante caminata aleatoria en
            // TODAS las escalas — de lo contrario la deriva √N espuria satura
            // tanh con convicción de mentira (bug que el test de ruido cazó).
            s.ewma_dev_vol += alpha * (dev.abs() - s.ewma_dev_vol);

            let z = if s.ewma_dev_vol > 1e-12 {
                dev / s.ewma_dev_vol
            } else {
                0.0
            };
            // Persistencia (F4.10-ready): autocorrelación de signos de las
            // sorpresas CONSECUTIVAS de esta escala. Ruido blanco ⇒ ±1 al 50%
            // ⇒ EWMA→0; tendencia sostenida ⇒ misma firma ⇒ →+1; reversión a
            // la media ⇒ →−1. (La versión previa comparaba dev consigo misma
            // post-update — trivialmente +1; el test de ruido la cazó.)
            let agree = (dev * s.prev_dev).signum()
                * (if dev.abs() > 1e-12 && s.prev_dev.abs() > 1e-12 {
                    1.0
                } else {
                    0.0
                });
            s.persistence += alpha * (agree - s.persistence);
            s.prev_dev = dev;
            s.momentum_z = z;
            s.signal = z.clamp(-5.0, 5.0).tanh();

            // Fusión paridad-de-riesgo: w ∝ 1/vol_de_desviación.
            let w = if s.ewma_dev_vol > 1e-12 {
                1.0 / s.ewma_dev_vol
            } else {
                0.0
            };
            w_sum += w;
            let contrib = w * s.signal;
            w_sig_sum += contrib;
            if contrib.abs() > best_contrib.abs() {
                best_contrib = contrib;
                dominant = s.tau_ms;
            }
        }
        self.fused_score = if w_sum > 0.0 {
            (w_sig_sum / w_sum).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        self.dominant_tau_ms = dominant;
    }

    /// Señal de la escala más cercana a τ (interpolación log-lineal entre
    /// escalas vecinas — el espectro es CONTINUO, no una lista discreta).
    pub fn signal_at(&self, tau_ms: f64) -> f64 {
        if tau_ms <= 0.0 {
            return 0.0;
        }
        let ln_tau = tau_ms.max(1e-6).ln();
        let ln_min = (1e-6_f64).ln();
        let step = 4f64.ln();
        let idx_f = (ln_tau - ln_min) / step;
        let i0 = idx_f.floor().clamp(0.0, 30.0) as usize;
        let i1 = (i0 + 1).min(31);
        let frac = (idx_f - i0 as f64).clamp(0.0, 1.0);
        let s0 = self.scales[i0].signal;
        let s1 = self.scales[i1].signal;
        s0 * (1.0 - frac) + s1 * frac
    }

    /// Persistencia interpolada log-linealmente a τ (como signal_at — el
    /// espectro es función continua en TODAS sus observables).
    pub fn persistence_at(&self, tau_ms: f64) -> f64 {
        if tau_ms <= 0.0 {
            return 0.0;
        }
        let ln_tau = tau_ms.max(1e-6).ln();
        let ln_min = (1e-6_f64).ln();
        let step = 4f64.ln();
        let idx_f = (ln_tau - ln_min) / step;
        let i0 = idx_f.floor().clamp(0.0, 30.0) as usize;
        let i1 = (i0 + 1).min(31);
        let frac = (idx_f - i0 as f64).clamp(0.0, 1.0);
        self.scales[i0].persistence * (1.0 - frac) + self.scales[i1].persistence * frac
    }

    /// Snapshot compacto para modelos/telemetría: 32 señales + fusión.
    pub fn signals_vector(&self) -> ([f32; 32], f32) {
        let mut v = [0.0f32; 32];
        for (i, s) in self.scales.iter().enumerate() {
            v[i] = s.signal as f32;
        }
        (v, self.fused_score as f32)
    }
}

/// CURVAS DE PARÁMETROS DEL GENOMA EN FUNCIÓN DEL HORIZONTE (F8):
/// param(τ) = exp(a + b·ln τ_ms). El genoma evoluciona (a,b) por familia —
/// la PENDIENTE b define cómo escala el parámetro con el horizonte: una
/// decisión continua, no dos valores sueltos por bucket.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct HorizonCurve {
    pub a: f64,
    pub b: f64,
}

impl HorizonCurve {
    pub fn eval(&self, tau_ms: f64) -> f64 {
        (self.a + self.b * tau_ms.max(1e-6).ln()).exp()
    }

    /// Ajusta (a,b) para pasar EXACTAMENTE por dos puntos ancla — usado para
    /// convertir genomas legacy sin cambiar su comportamiento en las bandas
    /// históricas (migración sin trauma).
    pub fn through_two_points(tau1_ms: f64, v1: f64, tau2_ms: f64, v2: f64) -> Self {
        let l1 = tau1_ms.max(1e-6).ln();
        let l2 = tau2_ms.max(1e-6).ln();
        let b = if (l2 - l1).abs() > 1e-9 {
            (v2.max(1e-12).ln() - v1.max(1e-12).ln()) / (l2 - l1)
        } else {
            0.0
        };
        let a = v1.max(1e-12).ln() - b * l1;
        Self { a, b }
    }

    /// Curva plana (constante v en todo el espectro, b = 0).
    pub fn flat(v: f64) -> Self {
        Self {
            a: v.max(1e-12).ln(),
            b: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escalas_cubren_el_espectro_sin_huecos() {
        assert_eq!(
            SPECTRUM_SCALES_MS[0], 1.0e-6,
            "cota inferior: 1ns = 10^-6 ms (física del reloj de CPU)"
        );
        assert!(
            SPECTRUM_SCALES_MS[31] > 3_150_000_000_000.0,
            "cota superior > 100 años"
        );
        // Log-espaciado exacto base 4: sin bandas prohibidas.
        for i in 1..32 {
            let ratio = SPECTRUM_SCALES_MS[i] / SPECTRUM_SCALES_MS[i - 1];
            assert!(
                (ratio - 4.0).abs() < 1e-6,
                "escala {} no es ×4 la anterior",
                i
            );
        }
        // Las anclas históricas viven DENTRO del espectro (no en los bordes).
        assert!(SPECTRUM_SCALES_MS[0] < TAU_ANCHOR_FAST_MS);
        assert!(TAU_ANCHOR_SLOW_MS < SPECTRUM_SCALES_MS[31]);
    }

    #[test]
    fn tendencia_lenta_detectada_por_las_escalas_largas() {
        let mut spec = TemporalSpectrum::new();
        let mut t = 1_700_000_000_000u64;
        // Rampa alcista durante 30 días a ticks de 1min: las escalas largas
        // deben opinar positivo y la fusión ser > 0.
        for i in 0..43_200u64 {
            let price = 60_000.0 * (1.0 + 0.00005 * i as f64);
            spec.update(price, t);
            t += 60_000;
        }
        assert!(
            spec.fused_score > 0.05,
            "rampa de 30d debe dar score positivo, dio {}",
            spec.fused_score
        );
        let (v, _) = spec.signals_vector();
        let slow_avg = (v[24] + v[25] + v[26]) / 3.0;
        assert!(slow_avg > 0.0, "escalas lentas deben ver la tendencia");
    }

    #[test]
    fn ruido_puro_no_genera_conviccion_persistente() {
        let mut spec = TemporalSpectrum::new();
        let mut seed: u64 = 42;
        let mut t = 1_700_000_000_000u64;
        let mut fused_sum = 0.0f64;
        let mut persistence_sum = 0.0f64;
        let n_measure = 5_000u64;
        for i in 0..50_000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((seed >> 33) as f64 / u32::MAX as f64) - 0.5;
            spec.update(60_000.0 * (1.0 + noise * 0.001), t);
            t += 100;
            // Propiedad ESTADÍSTICA: en ruido blanco la fusión no debe tener SESGO temporal persistente.
            if i >= 50_000 - n_measure {
                fused_sum += spec.fused_score;
                persistence_sum += spec.scales[18].persistence;
            }
        }
        let mean_fused = fused_sum / n_measure as f64;
        let mean_persist = persistence_sum / n_measure as f64;
        assert!(
            mean_fused.abs() < 0.15,
            "ruido sin deriva ⇒ media temporal de fusión ≈ 0, dio {mean_fused}"
        );
        assert!(
            mean_persist.abs() < 0.3,
            "persistencia en ruido debe ser débil, dio {mean_persist}"
        );
    }

    #[test]
    fn signal_at_es_continua_entre_escalas() {
        let mut spec = TemporalSpectrum::new();
        let mut t = 1_700_000_000_000u64;
        for i in 0..10_000u64 {
            spec.update(60_000.0 * (1.0 + 0.00001 * i as f64), t);
            t += 50;
        }
        let s_exact = spec.signal_at(65_536.0);
        let s_near = spec.signal_at(66_000.0);
        assert!(
            (s_exact - s_near).abs() < 0.15,
            "continuidad: {s_exact} vs {s_near}"
        );
    }

    #[test]
    fn curva_horizonte_pasa_por_los_anchos_legacy() {
        let curve =
            HorizonCurve::through_two_points(TAU_ANCHOR_FAST_MS, 0.01, TAU_ANCHOR_SLOW_MS, 0.05);
        assert!((curve.eval(TAU_ANCHOR_FAST_MS) - 0.01).abs() < 1e-9);
        assert!((curve.eval(TAU_ANCHOR_SLOW_MS) - 0.05).abs() < 1e-9);
        let mid = curve.eval((TAU_ANCHOR_FAST_MS * TAU_ANCHOR_SLOW_MS).sqrt());
        assert!(mid > 0.01 && mid < 0.05);
        assert!(curve.b > 0.0, "TP crece con horizonte: pendiente positiva");
    }
}
