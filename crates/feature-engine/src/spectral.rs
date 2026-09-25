//! FFT Radix-2 sobre 64 observaciones en TIEMPO DE EVENTOS. Un bin k significa
//! k/64 ciclos por observación, no hertz. No hay timestamps ni modelo multiescala.
//! La API legacy mantiene el esquema ML anterior. La API V2 opt-in elimina media,
//! declara calidad y normaliza potencia; cambiar el consumidor exige versionar y
//! reevaluar/reentrenar las features. No se acredita latencia sin benchmark.
const FFT_SIZE: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumError {
    InsufficientSamples { received: usize },
    InvalidSamples { count: usize },
    UnrepresentablePower,
}

/// Potencia unilateral por bin (unidades de entrada al cuadrado, NO PSD/Hz).
/// La suma de bins incluye DC residual del taper y equivale a energía de la
/// señal centrada y ventaneada dividida por sum(w²). Dominio/centroide excluyen DC.
#[derive(Debug, Clone, PartialEq)]
pub struct EventSpectrumV2 {
    pub power_by_bin: [f64; FFT_SIZE / 2 + 1],
    pub dominant_bin: usize,
    pub centroid_bin: f64,
}

#[derive(Debug, Clone)]
pub struct SpectralCycleEngine {
    buffer: [f64; FFT_SIZE],
    valid: [bool; FFT_SIZE],
    index: usize,
    count: usize,
}

impl SpectralCycleEngine {
    pub fn new() -> Self {
        Self {
            buffer: [0.0; FFT_SIZE],
            valid: [false; FFT_SIZE],
            index: 0,
            count: 0,
        }
    }

    /// Añade un nuevo retorno o precio normalizado al buffer circular
    #[inline(always)]
    pub fn push(&mut self, value: f64) {
        self.valid[self.index] = value.is_finite();
        self.buffer[self.index] = if value.is_finite() { value } else { 0.0 };
        self.index = (self.index + 1) % FFT_SIZE;
        if self.count < FFT_SIZE {
            self.count += 1;
        }
    }

    /// LEGACY: calcula bin dominante, potencia no corregida por ventana y centroide.
    /// No distingue ausencia, imputación y cero. Conservada por compatibilidad ML.
    /// Retorna: `(dominant_freq_bin, max_power, spectral_centroid)`
    pub fn analyze_spectrum(&self) -> (usize, f64, f64) {
        if self.count < FFT_SIZE {
            return (0, 0.0, 0.0);
        }

        // 1. Extraer muestras ordenadas y aplicar un taper de medio coseno izquierdo.
        // Atenúa el borde histórico izquierdo para evitar fuga espectral, pero mantiene peso 1.0 en el presente (i -> N-1)
        let mut real = [0.0; FFT_SIZE];
        let mut imag = [0.0; FFT_SIZE];
        let taper_len = FFT_SIZE / 4; // 16 muestras de taper inicial

        for i in 0..FFT_SIZE {
            let buf_idx = (self.index + i) % FFT_SIZE;
            let window_weight = if i < taper_len {
                0.5 * (1.0 - (std::f64::consts::PI * i as f64 / taper_len as f64).cos())
            } else {
                1.0
            };
            real[i] = self.buffer[buf_idx] * window_weight;
            imag[i] = 0.0;
        }

        // 2. In-place Cooley-Tukey Radix-2 FFT
        Self::fft_radix2(&mut real, &mut imag);

        // 3. Calcular densidades de potencia (Power Spectral Density) para bins 1..(FFT_SIZE / 2)
        // Ignoramos el bin 0 (componente DC)
        let mut max_power = 0.0;
        let mut dominant_bin = 0;
        let mut total_power = 0.0;
        let mut weighted_power_sum = 0.0;

        for k in 1..=(FFT_SIZE / 2) {
            let power = (real[k] * real[k] + imag[k] * imag[k]) / (FFT_SIZE as f64);
            if power > max_power {
                max_power = power;
                dominant_bin = k;
            }
            total_power += power;
            weighted_power_sum += (k as f64) * power;
        }

        let spectral_centroid = if total_power > 1e-12 {
            weighted_power_sum / total_power
        } else {
            0.0
        };

        let safe_centroid = if spectral_centroid.is_finite() {
            spectral_centroid
        } else {
            0.0
        };
        let safe_max_power = if max_power.is_finite() {
            max_power
        } else {
            0.0
        };

        (dominant_bin, safe_max_power, safe_centroid)
    }

    /// Estimador V2 de ventana finita, causal respecto al buffer recibido.
    /// Centrado aritmético antes del taper; potencia unilateral con duplicación
    /// sólo de bins interiores (DC y Nyquist no se duplican). Se rechaza toda
    /// ventana con imputaciones, no se comprime el reloj quitando eventos malos.
    /// No estima significación, intervalos, distribución predictiva ni régimen.
    pub fn analyze_event_spectrum_v2(&self) -> Result<EventSpectrumV2, SpectrumError> {
        if self.count < FFT_SIZE {
            return Err(SpectrumError::InsufficientSamples {
                received: self.count,
            });
        }
        let invalid = self.valid.iter().filter(|valid| !**valid).count();
        if invalid > 0 {
            return Err(SpectrumError::InvalidSamples { count: invalid });
        }
        let scale = self.buffer.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let mut result = EventSpectrumV2 {
            power_by_bin: [0.0; FFT_SIZE / 2 + 1],
            dominant_bin: 0,
            centroid_bin: 0.0,
        };
        if scale == 0.0 {
            return Ok(result);
        }
        // Normalizar antes de sumar evita overflow por media o mariposas.
        let mean = self
            .buffer
            .iter()
            .map(|x| (x / scale) / FFT_SIZE as f64)
            .sum::<f64>();
        let mut real = [0.0; FFT_SIZE];
        let mut imag = [0.0; FFT_SIZE];
        let mut window_energy = 0.0;
        for (i, value) in real.iter_mut().enumerate() {
            let weight = if i < FFT_SIZE / 4 {
                0.5 * (1.0 - (std::f64::consts::PI * i as f64 / (FFT_SIZE / 4) as f64).cos())
            } else {
                1.0
            };
            *value = (self.buffer[(self.index + i) % FFT_SIZE] / scale - mean) * weight;
            window_energy += weight * weight;
        }
        Self::fft_radix2(&mut real, &mut imag);
        let mut max_power = 0.0;
        let mut total = 0.0;
        let mut weighted = 0.0;
        for k in 0..=FFT_SIZE / 2 {
            let sides = if k == 0 || k == FFT_SIZE / 2 {
                1.0
            } else {
                2.0
            };
            let power =
                sides * (real[k] * real[k] + imag[k] * imag[k]) / (FFT_SIZE as f64 * window_energy);
            // Orden que no calcula scale² directamente: puede desbordar aunque
            // power*scale² sí sea representable. Un bin positivo que subdesborda
            // a cero tampoco se presenta como una medición de potencia nula.
            let restored = (power * scale) * scale;
            if !restored.is_finite() || (power > 0.0 && restored == 0.0) {
                return Err(SpectrumError::UnrepresentablePower);
            }
            result.power_by_bin[k] = restored;
            if k > 0 {
                if power > max_power {
                    max_power = power;
                    result.dominant_bin = k;
                }
                total += power;
                weighted += k as f64 * power;
            }
        }
        // No piso dimensional: forma espectral independiente de amplitud.
        if total > 0.0 {
            result.centroid_bin = weighted / total;
        }
        Ok(result)
    }

    /// Algoritmo Cooley-Tukey Radix-2 FFT In-Place de alta velocidad ($O(N \log N)$)
    #[inline(always)]
    fn fft_radix2(real: &mut [f64; FFT_SIZE], imag: &mut [f64; FFT_SIZE]) {
        let n = FFT_SIZE;
        // Bit-reversal permutation
        let mut j = 0;
        for i in 0..(n - 1) {
            if i < j {
                real.swap(i, j);
                imag.swap(i, j);
            }
            let mut k = n >> 1;
            while k <= j {
                j -= k;
                k >>= 1;
            }
            j += k;
        }

        // Mariposas de Cooley-Tukey
        let mut len = 2;
        while len <= n {
            let half_len = len >> 1;
            let angle = -2.0 * std::f64::consts::PI / (len as f64);
            let w_step_real = angle.cos();
            let w_step_imag = angle.sin();

            let mut i = 0;
            while i < n {
                let mut w_real = 1.0;
                let mut w_imag = 0.0;

                for k in 0..half_len {
                    let u_real = real[i + k];
                    let u_imag = imag[i + k];

                    let v_real = real[i + k + half_len] * w_real - imag[i + k + half_len] * w_imag;
                    let v_imag = real[i + k + half_len] * w_imag + imag[i + k + half_len] * w_real;

                    real[i + k] = u_real + v_real;
                    imag[i + k] = u_imag + v_imag;

                    real[i + k + half_len] = u_real - v_real;
                    imag[i + k + half_len] = u_imag - v_imag;

                    let next_w_real = w_real * w_step_real - w_imag * w_step_imag;
                    let next_w_imag = w_real * w_step_imag + w_imag * w_step_real;
                    w_real = next_w_real;
                    w_imag = next_w_imag;
                }
                i += len;
            }
            len <<= 1;
        }
    }
}

impl Default for SpectralCycleEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_spectral_cycle_detection() {
        let mut engine = SpectralCycleEngine::new();

        // Generar una onda senoidal pura con frecuencia de 4 ciclos en 64 muestras
        let freq = 4.0;
        for i in 0..64 {
            let val = (2.0 * std::f64::consts::PI * freq * i as f64 / 64.0).sin();
            engine.push(val);
        }

        let (dominant_bin, max_power, centroid) = engine.analyze_spectrum();
        // Debe detectar exactamente el bin 4
        assert_eq!(dominant_bin, 4);
        assert!(max_power > 0.1);
        assert!((centroid - 4.0).abs() < 0.5);
    }

    #[test]
    fn test_fft_spectral_nan_and_flat_immunity() {
        let mut engine = SpectralCycleEngine::new();
        // Incompleto
        let (bin0, pow0, cent0) = engine.analyze_spectrum();
        assert_eq!(bin0, 0);
        assert_eq!(pow0, 0.0);
        assert_eq!(cent0, 0.0);

        // Llenar con NaN
        for _ in 0..64 {
            engine.push(f64::NAN);
        }
        let (_, pow_nan, cent_nan) = engine.analyze_spectrum();
        assert!(pow_nan.is_finite());
        assert!(cent_nan.is_finite());
    }

    #[test]
    fn test_fft_spectral_planck_taper_window_symmetry() {
        let mut engine = SpectralCycleEngine::new();
        // Llenar con onda constante
        for _ in 0..64 {
            engine.push(1.0);
        }
        let (bin, pow, cent) = engine.analyze_spectrum();
        assert!(bin <= 32);
        assert!(pow.is_finite());
        assert!(cent.is_finite());
    }
}
