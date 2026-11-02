//! # Fase 17 — Bitmaps de Hardware `TZCNT`/`LZCNT` en 1 Ciclo (`BitmapTickScanner256`), Codificador ASCII por Tabla `LUT` sin `core::fmt` (`FastAsciiLutEncoder`) y Clase de Universalidad de Kardar-Parisi-Zhang (`KpzSurfaceGrowthEngine`)
//!
//! ## 👨‍🏫 MODO PROFESOR (QUÉ, POR QUÉ, PARA QUÉ, CÓMO, CUÁNDO, DÓNDE, QUIÉN)
//!
//! - **QUÉ:** Tres motores deterministas en pila (`O(1)` memoria, cero `unwrap`/`assert!`, 100% Rust nativo):
//!   1. `BitmapTickScanner256`: Mapa de bits jerárquico de 256 niveles de precio (`4 x u64` + máscara resumen de 4 bits) que localiza el siguiente Ask activo o el mejor Bid previo en **1 instrucción de hardware** (`TZCNT` / `LZCNT` / `POPCNT`) sin iterar sobre niveles vacíos.
//!   2. `FastAsciiLutEncoder`: Codificador numérico de enteros `u64` y precios decimales de punto fijo hacia buffers `[u8; 32]` en pila usando una tabla estática de pares de dígitos (`DIGIT_PAIRS_LUT` de 200 bytes en L1D), prescindiendo por completo de `core::fmt` y `format!`.
//!   3. `KpzSurfaceGrowthEngine`: Evolucionador de la interfaz de rugosidad del libro de órdenes bajo la ecuación no lineal de **Kardar-Parisi-Zhang (KPZ)** $\partial_t h = \nu \partial_x^2 h + \frac{\lambda}{2} (\partial_x h)^2 + \eta$ resuelta exactamente vía la **Transformación de Cole-Hopf** $Z(x,t) = \exp(\frac{\lambda}{2\nu} h(x,t))$, midiendo el ancho de rugosidad $W(L,t) \propto t^{1/3}$, la velocidad de avalancha lateral $\frac{\lambda}{2}\langle(\nabla h)^2\rangle$ y la asimetría de fluctuaciones tipo Tracy-Widom.
//! - **POR QUÉ:** En libros dispersos con huecos de liquidez, un bucle lineal desperdicia ciclos; `core::fmt` introduce indirecciones polimórficas al serializar órdenes FIX/REST; y el crecimiento de huecos en el libro obedece a la clase de universalidad KPZ ($\beta = 1/3, z = 3/2$) y no a la difusión gaussiana ($\beta = 1/4, z = 2$).
//! - **PARA QUÉ:** Saltar huecos del libro en 1 ciclo de reloj, serializar precios/cantidades en $< 8\text{ ns}$ en pila, y anticipar avalanchas laterales de barrido de liquidez (`KpzAvalancheBreakout`).
//! - **CÓMO:** Primitivas intrínsecas `trailing_zeros()`, `leading_zeros()` y `count_ones()`, indexación de 2 dígitos por división entera entre 100, e integración del calor estocástico en la variable de Cole-Hopf $Z_i$.
//! - **CUÁNDO:** En el barrido instantáneo de niveles L2, en la construcción de tramas de salida de órdenes y en cada evaluación de rugosidad superficial del libro.
//! - **DÓNDE:** `crates/feature-engine/src/tzcnt_lut_kpz.rs`.
//! - **QUIÉN:** `BitmapTickScanner256`, `FastAsciiLutEncoder`, `KpzSurfaceGrowthEngine`, `TzcntLutKpzMasterEngine`.

/// Dimensión espacial de la interfaz superficial KPZ en pila ($16$ sitios contiguos del libro).
pub const KPZ_SURFACE_SITES: usize = 16;

/// Exponente crítico universal de rugosidad espacial de Kardar-Parisi-Zhang en $1+1$ dimensiones: $\alpha = 1/2$.
pub const KPZ_ROUGHNESS_ALPHA: f64 = 0.5;

/// Exponente crítico universal de crecimiento temporal de Kardar-Parisi-Zhang en $1+1$ dimensiones: $\beta = 1/3$.
pub const KPZ_GROWTH_BETA: f64 = 1.0 / 3.0;

/// Exponente dinámico universal de Kardar-Parisi-Zhang: $z = \alpha / \beta = 3/2$ (superdifusivo frente a $z=2$ Gaussiano).
pub const KPZ_DYNAMIC_Z: f64 = 1.5;

/// Tabla estática de 200 bytes (`"00"` a `"99"`) residente en caché L1D para codificación ASCII de 2 dígitos por paso.
pub const DIGIT_PAIRS_LUT: &[u8; 200] = b"\
0001020304050607080910111213141516171819\
2021222324252627282930313233343536373839\
4041424344454647484950515253545556575859\
6061626364656667686970717273747576777879\
8081828384858687888990919293949596979899";

/// Clasificación del régimen superficial KPZ y densidad de niveles activos.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KpzBitmapRegime {
    /// Interfaz suave Edwards-Wilkinson ($\lambda_{\text{KPZ}} (\nabla h)^2$ bajo) y libro denso.
    SmoothEdwardsWilkinson,
    /// Crecimiento lateral no lineal KPZ dominante con avalancha direccional tipo Tracy-Widom.
    KpzAvalancheBreakout,
    /// Libro fracturado con huecos extensos de liquidez detectados por `TZCNT`/`LZCNT`.
    SparseLiquidityGapVeto,
}

/// Escáner jerárquico de 256 niveles de precio mediante instrucciones de bits `TZCNT`, `LZCNT` y `POPCNT`.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BitmapTickScanner256 {
    /// 4 palabras de 64 bits que cubren 256 ticks contiguos (`0..256`).
    pub words: [u64; 4],
    /// Máscara resumen de 4 bits: el bit `w` vale 1 si y sólo si `words[w] != 0`.
    pub summary_mask: u8,
}

impl BitmapTickScanner256 {
    /// Construye un escáner de 256 niveles inicialmente vacío.
    #[inline]
    pub const fn new() -> Self {
        Self {
            words: [0; 4],
            summary_mask: 0,
        }
    }

    /// Activa o desactiva en $\mathcal{O}(1)$ el bit de ocupación del nivel `tick_idx` (`0..256`).
    #[inline(always)]
    pub fn set_level(&mut self, tick_idx: usize, active: bool) -> bool {
        if tick_idx >= 256 {
            return false;
        }
        let w = tick_idx >> 6;
        let b = tick_idx & 63;
        let mask = 1_u64 << b;
        if active {
            self.words[w] |= mask;
            self.summary_mask |= 1_u8 << w;
        } else {
            self.words[w] &= !mask;
            if self.words[w] == 0 {
                self.summary_mask &= !(1_u8 << w);
            }
        }
        true
    }

    /// Cuenta en 4 instrucciones `POPCNT` el total de niveles con liquidez activa en los 256 ticks.
    #[inline(always)]
    pub const fn active_level_count(&self) -> u32 {
        self.words[0].count_ones()
            + self.words[1].count_ones()
            + self.words[2].count_ones()
            + self.words[3].count_ones()
    }

    /// Encuentra el primer nivel activo $\ge \text{start\_tick}$ (barrido Ask ascendente) usando `trailing_zeros` (`TZCNT`).
    #[inline(always)]
    pub const fn next_active_from(&self, start_tick: usize) -> Option<usize> {
        if start_tick >= 256 || self.summary_mask == 0 {
            return None;
        }
        let w0 = start_tick >> 6;
        let b0 = start_tick & 63;

        // 1. Inspeccionar la palabra inicial enmascarando los bits inferiores a b0
        let first_word = self.words[w0] & (!0_u64 << b0);
        if first_word != 0 {
            return Some((w0 << 6) + (first_word.trailing_zeros() as usize));
        }

        // 2. Saltar directamente a la siguiente palabra no vacía usando summary_mask
        let rem_summary = self.summary_mask & (!0_u8 << (w0 + 1));
        if rem_summary == 0 {
            return None;
        }
        let next_w = rem_summary.trailing_zeros() as usize;
        if next_w < 4 {
            let word = self.words[next_w];
            Some((next_w << 6) + (word.trailing_zeros() as usize))
        } else {
            None
        }
    }

    /// Encuentra el nivel activo más alto $\le \text{start\_tick}$ (barrido Bid descendente) usando `leading_zeros` (`LZCNT`).
    #[inline(always)]
    pub const fn prev_active_from(&self, start_tick: usize) -> Option<usize> {
        if self.summary_mask == 0 {
            return None;
        }
        let clamped = if start_tick >= 256 { 255 } else { start_tick };
        let w0 = clamped >> 6;
        let b0 = clamped & 63;

        // 1. Inspeccionar la palabra inicial conservando los bits 0..=b0
        let shift = 63 - b0;
        let first_word = self.words[w0] & (!0_u64 >> shift);
        if first_word != 0 {
            let highest_bit = 63 - (first_word.leading_zeros() as usize);
            return Some((w0 << 6) + highest_bit);
        }

        // 2. Buscar en las palabras inferiores usando summary_mask
        if w0 == 0 {
            return None;
        }
        let low_mask = (1_u8 << w0) - 1;
        let rem_summary = self.summary_mask & low_mask;
        if rem_summary == 0 {
            return None;
        }
        let prev_w = 7 - (rem_summary.leading_zeros() as usize);
        let word = self.words[prev_w];
        let highest_bit = 63 - (word.leading_zeros() as usize);
        Some((prev_w << 6) + highest_bit)
    }

    /// Calcula el spread en ticks entre el mejor Ask $\ge \text{mid\_tick}$ y el mejor Bid $\le \text{mid\_tick}$.
    #[inline]
    pub const fn spread_ticks_around(&self, mid_tick: usize) -> Option<usize> {
        match (
            self.prev_active_from(mid_tick),
            self.next_active_from(mid_tick),
        ) {
            (Some(bid_idx), Some(ask_idx)) if ask_idx >= bid_idx => Some(ask_idx - bid_idx),
            _ => None,
        }
    }
}

/// Codificador numérico ultrarrápido basado en `DIGIT_PAIRS_LUT` sin `core::fmt`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FastAsciiLutEncoder;

impl FastAsciiLutEncoder {
    /// Codifica un entero `u64` en base 10 dentro de `buf` de dos en dos dígitos usando `DIGIT_PAIRS_LUT`,
    /// devolviendo la longitud escrita (y alineando el resultado desde el índice `0` de `buf`).
    #[inline]
    pub fn encode_u64(mut value: u64, buf: &mut [u8; 32]) -> usize {
        if value == 0 {
            buf[0] = b'0';
            return 1;
        }
        let mut scratch = [0_u8; 24];
        let mut pos = 24_usize;

        while value >= 100 {
            let rem = ((value % 100) as usize) * 2;
            value /= 100;
            pos -= 2;
            scratch[pos] = DIGIT_PAIRS_LUT[rem];
            scratch[pos + 1] = DIGIT_PAIRS_LUT[rem + 1];
        }

        if value < 10 {
            pos -= 1;
            scratch[pos] = b'0' + (value as u8);
        } else {
            let rem = (value as usize) * 2;
            pos -= 2;
            scratch[pos] = DIGIT_PAIRS_LUT[rem];
            scratch[pos + 1] = DIGIT_PAIRS_LUT[rem + 1];
        }

        let len = 24 - pos;
        let mut i = 0;
        while i < len {
            buf[i] = scratch[pos + i];
            i += 1;
        }
        len
    }

    /// Codifica un precio/cantidad `f64` no negativo con `decimals` cifras decimales (`0..=8`) en `buf`
    /// sin usar `core::fmt`, devolviendo la cantidad de bytes ASCII escritos.
    #[inline]
    pub fn encode_fixed_f64(value: f64, decimals: u8, buf: &mut [u8; 32]) -> usize {
        let clean = if value.is_finite() && value >= 0.0 {
            value.min(1.0e11)
        } else {
            0.0
        };
        let dec = decimals.min(8);
        let scale: u64 = match dec {
            0 => 1,
            1 => 10,
            2 => 100,
            3 => 1_000,
            4 => 10_000,
            5 => 100_000,
            6 => 1_000_000,
            7 => 10_000_000,
            _ => 100_000_000,
        };

        let scaled = (clean * (scale as f64)).round() as u64;
        let int_part = scaled / scale;
        let frac_part = scaled % scale;

        let mut int_buf = [0_u8; 32];
        let int_len = Self::encode_u64(int_part, &mut int_buf);
        let mut idx = 0;
        while idx < int_len {
            buf[idx] = int_buf[idx];
            idx += 1;
        }

        if dec == 0 {
            return idx;
        }

        buf[idx] = b'.';
        idx += 1;

        // Codificar exactamente `dec` dígitos fraccionarios con relleno de ceros a la izquierda
        let mut rem_frac = frac_part;
        let mut frac_pos = idx + (dec as usize);
        let final_len = frac_pos;

        while frac_pos >= idx + 2 {
            let pair = ((rem_frac % 100) as usize) * 2;
            rem_frac /= 100;
            frac_pos -= 2;
            buf[frac_pos] = DIGIT_PAIRS_LUT[pair];
            buf[frac_pos + 1] = DIGIT_PAIRS_LUT[pair + 1];
        }
        if frac_pos > idx {
            buf[idx] = b'0' + ((rem_frac % 10) as u8);
        }

        final_len
    }
}

/// Diagnóstico físico de la interfaz superficial KPZ del libro de órdenes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KpzSurfaceReport {
    /// Perfil de altura evolucionado $h(x, t+\Delta t)$ mediante la transformada inversa de Cole-Hopf.
    pub evolved_heights: [f64; KPZ_SURFACE_SITES],
    /// Altura media de la interfaz $\bar{h}(t) = \frac{1}{L} \sum_x h(x,t)$.
    pub mean_height: f64,
    /// Ancho de rugosidad superficial $W(L, t) = \sqrt{\frac{1}{L} \sum_x (h(x,t) - \bar{h})^2}$.
    pub roughness_width_w: f64,
    /// Velocidad de crecimiento lateral no lineal $\frac{\lambda_{\text{KPZ}}}{2} \langle (\nabla h)^2 \rangle$.
    pub nonlinear_lateral_velocity: f64,
    /// Asimetría de fluctuaciones (momento estandarizado de 3er orden tipo Tracy-Widom).
    pub tracy_widom_skewness: f64,
    /// Razón de dominancia no lineal KPZ frente a difusión lineal de Edwards-Wilkinson.
    pub kpz_nonlinearity_ratio: f64,
}

/// Motor de evolución superficial de Kardar-Parisi-Zhang (KPZ) mediante la transformación de Cole-Hopf.
#[derive(Debug, Clone, Copy)]
pub struct KpzSurfaceGrowthEngine {
    /// Tensión superficial de alisamiento difusivo $\nu > 0$ (término de Edwards-Wilkinson $\nu \nabla^2 h$).
    pub surface_tension_nu: f64,
    /// Acoplamiento de crecimiento lateral no lineal $\lambda_{\text{KPZ}}$ ($\frac{\lambda}{2} (\nabla h)^2$).
    pub nonlinear_lambda: f64,
    /// Paso espacial entre sitios $\Delta x > 0$.
    pub grid_dx: f64,
}

impl Default for KpzSurfaceGrowthEngine {
    #[inline]
    fn default() -> Self {
        Self {
            surface_tension_nu: 0.50,
            nonlinear_lambda: 1.20,
            grid_dx: 1.0,
        }
    }
}

impl KpzSurfaceGrowthEngine {
    /// Evoluciona el perfil de alturas `heights` un paso `dt` bajo la ecuación de KPZ
    /// usando la transformación exacta de Cole-Hopf:
    /// $$Z_i(t) = \exp\left(\frac{\lambda}{2\nu} (h_i(t) - h_0)\right), \qquad \partial_t Z = \nu \nabla^2 Z + \frac{\lambda}{2\nu} Z \eta_i$$
    #[inline]
    pub fn step_cole_hopf(
        &self,
        heights: &[f64; KPZ_SURFACE_SITES],
        stochastic_forcing: &[f64; KPZ_SURFACE_SITES],
        dt: f64,
    ) -> KpzSurfaceReport {
        let nu = if self.surface_tension_nu.is_finite() && self.surface_tension_nu > 1e-4 {
            self.surface_tension_nu.min(50.0)
        } else {
            0.50
        };
        let lam = if self.nonlinear_lambda.is_finite() {
            self.nonlinear_lambda.clamp(-20.0, 20.0)
        } else {
            1.20
        };
        let dx = if self.grid_dx.is_finite() && self.grid_dx > 1e-4 {
            self.grid_dx
        } else {
            1.0
        };
        let h_dt = if dt.is_finite() && dt > 0.0 {
            dt.min(0.20)
        } else {
            0.01
        };

        let mut h = [0.0; KPZ_SURFACE_SITES];
        let mut h_sum = 0.0;
        let mut i = 0;
        while i < KPZ_SURFACE_SITES {
            let v = heights[i];
            h[i] = if v.is_finite() {
                v.clamp(-1e4, 1e4)
            } else {
                0.0
            };
            h_sum += h[i];
            i += 1;
        }
        let h_ref = h_sum / (KPZ_SURFACE_SITES as f64);

        // Si |lambda| es despreciable, evoluciona en el límite lineal exacto de Edwards-Wilkinson
        let alpha_ch = lam / (2.0 * nu);
        let inv_dx2 = 1.0 / (dx * dx);

        let mut evolved = [0.0; KPZ_SURFACE_SITES];
        if alpha_ch.abs() < 1e-5 {
            let mut k = 0;
            while k < KPZ_SURFACE_SITES {
                let left = h[(k + KPZ_SURFACE_SITES - 1) % KPZ_SURFACE_SITES];
                let right = h[(k + 1) % KPZ_SURFACE_SITES];
                let lap = (left - 2.0 * h[k] + right) * inv_dx2;
                let eta = if stochastic_forcing[k].is_finite() {
                    stochastic_forcing[k].clamp(-100.0, 100.0)
                } else {
                    0.0
                };
                evolved[k] = h[k] + h_dt * (nu * lap + eta);
                k += 1;
            }
        } else {
            // Transformada de Cole-Hopf centrada en h_ref para inmunidad total contra overflow exponencial
            let mut z = [1.0; KPZ_SURFACE_SITES];
            let mut k = 0;
            while k < KPZ_SURFACE_SITES {
                let arg = (alpha_ch * (h[k] - h_ref)).clamp(-40.0, 40.0);
                z[k] = arg.exp();
                k += 1;
            }

            let mut m = 0;
            while m < KPZ_SURFACE_SITES {
                let z_left = z[(m + KPZ_SURFACE_SITES - 1) % KPZ_SURFACE_SITES];
                let z_right = z[(m + 1) % KPZ_SURFACE_SITES];
                let lap_z = (z_left - 2.0 * z[m] + z_right) * inv_dx2;
                let eta = if stochastic_forcing[m].is_finite() {
                    stochastic_forcing[m].clamp(-50.0, 50.0)
                } else {
                    0.0
                };
                // Ecuación de calor estocástica multiplicativa en variable de Cole-Hopf
                let dz = h_dt * (nu * lap_z + alpha_ch * z[m] * eta);
                let z_next = (z[m] + dz).max(1e-18);
                evolved[m] = h_ref + z_next.ln() / alpha_ch;
                m += 1;
            }
        }

        // Estadísticos de rugosidad, velocidad lateral y asimetría Tracy-Widom
        let mut mean = 0.0;
        let mut grad_sq_sum = 0.0;
        let mut lap_abs_sum = 0.0;
        let mut idx = 0;
        while idx < KPZ_SURFACE_SITES {
            mean += evolved[idx];
            let next = evolved[(idx + 1) % KPZ_SURFACE_SITES];
            let prev = evolved[(idx + KPZ_SURFACE_SITES - 1) % KPZ_SURFACE_SITES];
            let grad = (next - prev) / (2.0 * dx);
            let lap = (next - 2.0 * evolved[idx] + prev) * inv_dx2;
            grad_sq_sum += grad * grad;
            lap_abs_sum += (nu * lap).abs();
            idx += 1;
        }
        mean /= KPZ_SURFACE_SITES as f64;

        let mut var_sum = 0.0;
        let mut third_moment_sum = 0.0;
        let mut j = 0;
        while j < KPZ_SURFACE_SITES {
            let d = evolved[j] - mean;
            let d2 = d * d;
            var_sum += d2;
            third_moment_sum += d2 * d;
            j += 1;
        }

        let variance = var_sum / (KPZ_SURFACE_SITES as f64);
        let width_w = variance.sqrt();
        let skewness = if width_w > 1e-9 {
            (third_moment_sum / (KPZ_SURFACE_SITES as f64)) / (width_w * width_w * width_w)
        } else {
            0.0
        };

        let mean_grad_sq = grad_sq_sum / (KPZ_SURFACE_SITES as f64);
        let lateral_vel = 0.5 * lam * mean_grad_sq;
        let mean_diffusive = lap_abs_sum / (KPZ_SURFACE_SITES as f64);
        let nl_ratio = lateral_vel.abs() / (lateral_vel.abs() + mean_diffusive + 1e-9);

        KpzSurfaceReport {
            evolved_heights: evolved,
            mean_height: if mean.is_finite() { mean } else { 0.0 },
            roughness_width_w: if width_w.is_finite() { width_w } else { 0.0 },
            nonlinear_lateral_velocity: if lateral_vel.is_finite() {
                lateral_vel
            } else {
                0.0
            },
            tracy_widom_skewness: if skewness.is_finite() {
                skewness.clamp(-10.0, 10.0)
            } else {
                0.0
            },
            kpz_nonlinearity_ratio: if nl_ratio.is_finite() {
                nl_ratio.clamp(0.0, 1.0)
            } else {
                0.0
            },
        }
    }
}

/// Reporte unificado de la Fase 17 (`BitmapTickScanner256` + `FastAsciiLutEncoder` + `KpzSurfaceGrowthEngine`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TzcntLutKpzReport {
    pub active_ticks: u32,
    pub spread_ticks: Option<usize>,
    pub kpz: KpzSurfaceReport,
    pub directional_avalanche_signal: f64,
    pub regime: KpzBitmapRegime,
}

/// Motor maestro de Fase 17 que integra el escáner bitmap `TZCNT`/`LZCNT` y el motor de universalidad KPZ.
#[derive(Debug, Clone, Copy, Default)]
pub struct TzcntLutKpzMasterEngine {
    pub kpz_engine: KpzSurfaceGrowthEngine,
}

impl TzcntLutKpzMasterEngine {
    /// Evalúa un ciclo completo del escáner bitmap de 256 niveles y la evolución KPZ de Cole-Hopf.
    #[inline]
    pub fn evaluate_cycle(
        &self,
        scanner: &BitmapTickScanner256,
        mid_tick: usize,
        heights: &[f64; KPZ_SURFACE_SITES],
        forcing: &[f64; KPZ_SURFACE_SITES],
        dt: f64,
    ) -> TzcntLutKpzReport {
        let active_ticks = scanner.active_level_count();
        let spread_ticks = scanner.spread_ticks_around(mid_tick);
        let kpz = self.kpz_engine.step_cole_hopf(heights, forcing, dt);

        let raw_sig = kpz.nonlinear_lateral_velocity.tanh();
        let directional_avalanche_signal = if raw_sig.is_finite() {
            raw_sig.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let regime = match spread_ticks {
            None => KpzBitmapRegime::SparseLiquidityGapVeto,
            Some(sp) if sp > 32 || active_ticks < 4 => KpzBitmapRegime::SparseLiquidityGapVeto,
            _ if kpz.kpz_nonlinearity_ratio > 0.55 => KpzBitmapRegime::KpzAvalancheBreakout,
            _ => KpzBitmapRegime::SmoothEdwardsWilkinson,
        };

        TzcntLutKpzReport {
            active_ticks,
            spread_ticks,
            kpz,
            directional_avalanche_signal,
            regime,
        }
    }
}
