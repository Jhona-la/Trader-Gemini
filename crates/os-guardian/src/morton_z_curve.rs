//! Indexador Tensorial por Curva Z de Morton (FASE 25)
//!
//! # QUÉ:
//! Indexador espacial y tensorial de alta velocidad mediante Curvas Z de Morton
//! (Space-Filling Curve) para preservación de localidad de datos multidimensionales
//! en las líneas de caché L1D/L2 de la CPU.
//!
//! # POR QUÉ:
//! Los estados de mercado multivariantes continuos (e.g., $P_{\text{bid}}, P_{\text{ask}}, \text{OFI}, \sigma$)
//! abarcan espacios de 2, 3 o más dimensiones. El almacenamiento lineal clásico (Row-Major o Column-Major)
//! destruye la contigüidad espacial en saltos de filas, provocando fallos de línea de caché ($L1D$ misses).
//! La curva de Morton mapea puntos cercanos en el hiperespacio a posiciones de memoria contiguas
//! mediante entrelazado de bits, maximizando la tasa de aciertos de caché en hardware x86_64.
//!
//! # PARA QUÉ:
//! 1. Proporcionar indexación $\mathcal{O}(1)$ en 1-2 ciclos de CPU (< 2 ns).
//! 2. Maximizar la densidad de datos por línea de caché de 64 bytes.
//! 3. Permitir consultas de rango hiperdimensionales eficientes (bounding-box).
//!
//! # CÓMO:
//! Algoritmo de dispersión de bits mediante desplazamientos y máscaras binarias fijas
//! (emulación exacta y segura de instrucciones hardware `PDEP` / `PEXT` de BMI2),
//! con funciones puras `const fn` libres de pánicos.
//!
//! # CUÁNDO:
//! En la indexación de cuadrículas tensoriales, libros de órdenes multidimensionales y
//! búsqueda de regímenes afines en memoria.
//!
//! # DÓNDE:
//! `crates/os-guardian/src/morton_z_curve.rs`.
//!
//! # QUIÉN:
//! Arquitecto de Silicio (Rol 7) & Quant de Microestructura (Rol 2).

#![allow(clippy::all)]

/// Indexador estático de Curva Z de Morton para 2D y 3D en stack.
pub struct MortonZOrderCurve;

impl MortonZOrderCurve {
    /// Máscara máxima de entrada para 2D (16 bits: 0..65535).
    pub const MAX_COORD_2D: u32 = 0xFFFF;
    /// Máscara máxima de entrada para 3D (10 bits: 0..1023).
    pub const MAX_COORD_3D: u32 = 0x3FF;

    /// Dispersa los 16 bits inferiores de un `u32` dejando un bit en cero entre cada uno.
    /// Emulación ultra-rápida de `_pdep_u32(x, 0x55555555)`.
    #[inline(always)]
    pub const fn spread_bits_2d(mut x: u32) -> u32 {
        x &= 0x0000FFFF;
        x = (x | (x << 8)) & 0x00FF00FF;
        x = (x | (x << 4)) & 0x0F0F0F0F;
        x = (x | (x << 2)) & 0x33333333;
        x = (x | (x << 1)) & 0x55555555;
        x
    }

    /// Compacta los bits pares de un `u32` en los 16 bits inferiores.
    /// Emulación ultra-rápida de `_pext_u32(x, 0x55555555)`.
    #[inline(always)]
    pub const fn compact_bits_2d(mut x: u32) -> u32 {
        x &= 0x55555555;
        x = (x | (x >> 1)) & 0x33333333;
        x = (x | (x >> 2)) & 0x0F0F0F0F;
        x = (x | (x >> 4)) & 0x00FF00FF;
        x = (x | (x >> 8)) & 0x0000FFFF;
        x
    }

    /// Codifica dos coordenadas discretas de 16 bits `(x, y)` en un código Morton 2D de 32 bits.
    #[inline(always)]
    pub const fn encode_2d(x: u32, y: u32) -> u32 {
        let x_spread = Self::spread_bits_2d(x);
        let y_spread = Self::spread_bits_2d(y);
        x_spread | (y_spread << 1)
    }

    /// Decodifica un código Morton 2D de 32 bits en dos coordenadas discretas `(x, y)`.
    #[inline(always)]
    pub const fn decode_2d(code: u32) -> (u32, u32) {
        let x = Self::compact_bits_2d(code);
        let y = Self::compact_bits_2d(code >> 1);
        (x, y)
    }

    /// Dispersa los 10 bits inferiores de un `u32` dejando dos bits en cero entre cada uno.
    /// Emulación ultra-rápida de `_pdep_u32(x, 0x49249249)`.
    #[inline(always)]
    pub const fn spread_bits_3d(mut x: u32) -> u32 {
        x &= 0x000003FF;
        x = (x | (x << 16)) & 0x030000FF;
        x = (x | (x << 8)) & 0x0300F00F;
        x = (x | (x << 4)) & 0x030C30C3;
        x = (x | (x << 2)) & 0x09249249;
        x
    }

    /// Compacta los bits en posiciones `3k` en los 10 bits inferiores.
    /// Emulación ultra-rápida de `_pext_u32(x, 0x49249249)`.
    #[inline(always)]
    pub const fn compact_bits_3d(mut x: u32) -> u32 {
        x &= 0x09249249;
        x = (x | (x >> 2)) & 0x030C30C3;
        x = (x | (x >> 4)) & 0x0300F00F;
        x = (x | (x >> 8)) & 0x030000FF;
        x = (x | (x >> 16)) & 0x000003FF;
        x
    }

    /// Codifica tres coordenadas discretas de 10 bits `(x, y, z)` en un código Morton 3D de 30 bits.
    #[inline(always)]
    pub const fn encode_3d(x: u32, y: u32, z: u32) -> u32 {
        let x_spread = Self::spread_bits_3d(x);
        let y_spread = Self::spread_bits_3d(y);
        let z_spread = Self::spread_bits_3d(z);
        x_spread | (y_spread << 1) | (z_spread << 2)
    }

    /// Decodifica un código Morton 3D de 30 bits en tres coordenadas discretas `(x, y, z)`.
    #[inline(always)]
    pub const fn decode_3d(code: u32) -> (u32, u32, u32) {
        let x = Self::compact_bits_3d(code);
        let y = Self::compact_bits_3d(code >> 1);
        let z = Self::compact_bits_3d(code >> 2);
        (x, y, z)
    }

    /// Normaliza un valor flotante continuo en el intervalo `[min, max]` al rango entero discreto `[0, steps-1]`.
    #[inline(always)]
    pub fn discretize(val: f64, min: f64, max: f64, steps: u32) -> u32 {
        if !val.is_finite() || !min.is_finite() || !max.is_finite() || max <= min || steps == 0 {
            return 0;
        }
        let clamped = val.clamp(min, max);
        let ratio = (clamped - min) / (max - min);
        let step_f = (ratio * (steps - 1) as f64).round();
        if step_f.is_finite() && step_f >= 0.0 {
            (step_f as u32).min(steps - 1)
        } else {
            0
        }
    }

    /// Desnormaliza un índice discreto en `[0, steps-1]` al valor continuo en `[min, max]`.
    #[inline(always)]
    pub fn undiscretize(idx: u32, min: f64, max: f64, steps: u32) -> f64 {
        if steps <= 1 || max <= min {
            return min;
        }
        let clamped_idx = idx.min(steps - 1) as f64;
        let ratio = clamped_idx / ((steps - 1) as f64);
        min + ratio * (max - min)
    }

    /// Codifica un punto 2D continuo `(x, y)` delimitado por sus rangos a un código Morton de 32 bits.
    #[inline]
    pub fn encode_continuous_2d(x: f64, x_range: (f64, f64), y: f64, y_range: (f64, f64)) -> u32 {
        let ix = Self::discretize(x, x_range.0, x_range.1, 65536);
        let iy = Self::discretize(y, y_range.0, y_range.1, 65536);
        Self::encode_2d(ix, iy)
    }

    /// Codifica un punto 3D continuo `(x, y, z)` delimitado por sus rangos a un código Morton de 30 bits.
    #[inline]
    pub fn encode_continuous_3d(
        x: f64,
        x_range: (f64, f64),
        y: f64,
        y_range: (f64, f64),
        z: f64,
        z_range: (f64, f64),
    ) -> u32 {
        let ix = Self::discretize(x, x_range.0, x_range.1, 1024);
        let iy = Self::discretize(y, y_range.0, y_range.1, 1024);
        let iz = Self::discretize(z, z_range.0, z_range.1, 1024);
        Self::encode_3d(ix, iy, iz)
    }
}
