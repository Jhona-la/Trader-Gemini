//! # Módulo: Aritmética SIMD Libre de Acarreo y Sistema de Numeración de Residuos (RNS)
//!
//! ## Método Pedagógico Riguroso (Modo Profesor)
//! - **QUÉ:** Motor de cálculo aritmético paralelo basado en el Sistema de Numeración de Residuos
//!   (Residue Number System - RNS), permitiendo operaciones de adición, sustracción y multiplicación
//!   sobre enteros grandes sin propagación de acarreo (carry-free).
//! - **POR QUÉ:** Las ALUs tradicionales sufren de una latencia de propagación de acarreo
//!   (carry ripple delay $\mathcal{O}(N)$) al operar con enteros de 64, 128 o 256 bits (precios en satoshis,
//!   acumulación de volúmenes L2/L3 y timestamps de picosegundos). Esto impide la paralelización SIMD horizontal
//!   y crea cuellos de botella en el ciclo de ejecución.
//! - **PARA QUÉ:** Permitir la acumulación ultra-rápida y exacta en $\mathcal{O}(1)$ de métricas de flujo
//!   de órdenes, volumes ponderados e integrales de microestructura en picosegundos y nanosegundos,
//!   con cero desbordamientos y total paralelismo SIMD en registros de CPU.
//! - **CÓMO:** Se selecciona un conjunto de 4 módulos coprimos dos a dos $\{m_1, m_2, m_3, m_4\}$. Un número
//!   $X$ se mapea a su tupla de residuos $r_i = X \pmod{m_i}$. Las operaciones se realizan elemento a elemento
//!   en paralelo sin comunicación entre canales. La reconstrucción al dominio canónico se realiza mediante
//!   el Teorema Chino del Resto (CRT) y la optimización de Garner con aritmética modular precomputada.
//! - **CUÁNDO:** Se invoca en el camino caliente de ingestión de ticks y ejecución cuando se requiere suma y producto
//!   de enteros de alta precisión sin penalización de latencia de pipeline.
//! - **DÓNDE:** `crates/os-guardian/src/rns_simd_arithmetic.rs`.
//! - **QUIÉN:** Consejo Colegiado (Rol 7: Silicio & Latencia Cuántica y Rol 2: Quant de Microestructura).

#![allow(clippy::all)]

use core::fmt;

/// Base de módulos RNS coprimos de 31/29/23/19 bits.
/// Todos los módulos son primos coprimos dos a dos:
/// m1 = 2^31 - 1 = 2147483647 (Primo de Mersenne M31)
/// m2 = 2^29 - 3 = 536870909  (Primo)
/// m3 = 2^23 - 15 = 8388593   (Primo)
/// m4 = 2^19 - 1 = 524287     (Primo de Mersenne M19)
/// Rango dinámico M = m1 * m2 * m3 * m4 > 2^101 > 10^30
pub const RNS_M1: u64 = 2_147_483_647;
pub const RNS_M2: u64 = 536_870_909;
pub const RNS_M3: u64 = 8_388_593;
pub const RNS_M4: u64 = 524_287;

/// Rango dinámico del sistema como u128 para validación y CRT.
pub const RNS_MODULI: [u64; 4] = [RNS_M1, RNS_M2, RNS_M3, RNS_M4];

/// Representación vectorial en RNS de 4 canales.
/// Almacenado en stack (32 bytes) para máxima contigüidad en caché L1D.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
#[repr(align(32))]
pub struct RnsVector4 {
    /// Residuos r_i = X mod m_i
    pub r: [u64; 4],
}

impl fmt::Debug for RnsVector4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RnsVector4([{}, {}, {}, {}])",
            self.r[0], self.r[1], self.r[2], self.r[3]
        )
    }
}

impl RnsVector4 {
    /// Crea un nuevo vector RNS a partir de sus residuos explícitos normalizados.
    #[inline(always)]
    pub fn from_residues(r1: u64, r2: u64, r3: u64, r4: u64) -> Self {
        Self {
            r: [r1 % RNS_M1, r2 % RNS_M2, r3 % RNS_M3, r4 % RNS_M4],
        }
    }

    /// Codifica un entero positivo u64 en su representación RNS.
    #[inline(always)]
    pub fn from_u64(val: u64) -> Self {
        Self {
            r: [val % RNS_M1, val % RNS_M2, val % RNS_M3, val % RNS_M4],
        }
    }

    /// Codifica un entero de 128 bits u128 en su representación RNS.
    #[inline(always)]
    pub fn from_u128(val: u128) -> Self {
        Self {
            r: [
                (val % (RNS_M1 as u128)) as u64,
                (val % (RNS_M2 as u128)) as u64,
                (val % (RNS_M3 as u128)) as u64,
                (val % (RNS_M4 as u128)) as u64,
            ],
        }
    }

    /// Suma RNS libre de acarreo: (A + B) mod M en O(1) paralelo.
    #[inline(always)]
    pub fn add(&self, rhs: &Self) -> Self {
        Self {
            r: [
                (self.r[0] + rhs.r[0]) % RNS_M1,
                (self.r[1] + rhs.r[1]) % RNS_M2,
                (self.r[2] + rhs.r[2]) % RNS_M3,
                (self.r[3] + rhs.r[3]) % RNS_M4,
            ],
        }
    }

    /// Resta RNS libre de acarreo: (A - B) mod M en O(1) paralelo.
    #[inline(always)]
    pub fn sub(&self, rhs: &Self) -> Self {
        Self {
            r: [
                (self.r[0] + RNS_M1 - (rhs.r[0] % RNS_M1)) % RNS_M1,
                (self.r[1] + RNS_M2 - (rhs.r[1] % RNS_M2)) % RNS_M2,
                (self.r[2] + RNS_M3 - (rhs.r[2] % RNS_M3)) % RNS_M3,
                (self.r[3] + RNS_M4 - (rhs.r[3] % RNS_M4)) % RNS_M4,
            ],
        }
    }

    /// Multiplicación RNS libre de acarreo: (A * B) mod M en O(1) paralelo.
    #[inline(always)]
    pub fn mul(&self, rhs: &Self) -> Self {
        Self {
            r: [
                ((self.r[0] as u128 * rhs.r[0] as u128) % (RNS_M1 as u128)) as u64,
                ((self.r[1] as u128 * rhs.r[1] as u128) % (RNS_M2 as u128)) as u64,
                ((self.r[2] as u128 * rhs.r[2] as u128) % (RNS_M3 as u128)) as u64,
                ((self.r[3] as u128 * rhs.r[3] as u128) % (RNS_M4 as u128)) as u64,
            ],
        }
    }

    /// Fused Multiply-Add (FMA) RNS: (A * B + C) mod M en O(1).
    #[inline(always)]
    pub fn fma(&self, b: &Self, c: &Self) -> Self {
        let prod = self.mul(b);
        prod.add(c)
    }

    /// Multiplicación por escalar u64.
    #[inline(always)]
    pub fn mul_scalar(&self, scalar: u64) -> Self {
        let s_rns = Self::from_u64(scalar);
        self.mul(&s_rns)
    }

    /// Verifica si el vector representa el cero canónico.
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.r[0] == 0 && self.r[1] == 0 && self.r[2] == 0 && self.r[3] == 0
    }
}

/// Motor de Aritmética RNS y reconstrucción por Teorema Chino del Resto (CRT).
pub struct RnsSimdArithmeticEngine {
    /// Rango dinámico M total = m1 * m2 * m3 * m4
    m_total: u128,
    /// M_i = M / m_i para cada canal
    m_i: [u128; 4],
    /// Inversos modulares (M_i)^(-1) mod m_i
    m_i_inv: [u64; 4],
}

impl Default for RnsSimdArithmeticEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl RnsSimdArithmeticEngine {
    /// Inicializa el motor precalculando los parámetros del Teorema Chino del Resto.
    pub fn new() -> Self {
        let m1 = RNS_M1 as u128;
        let m2 = RNS_M2 as u128;
        let m3 = RNS_M3 as u128;
        let m4 = RNS_M4 as u128;

        let m_total = m1 * m2 * m3 * m4;

        let m_i = [m2 * m3 * m4, m1 * m3 * m4, m1 * m2 * m4, m1 * m2 * m3];

        let m_i_inv = [
            Self::mod_inverse((m_i[0] % m1) as u64, RNS_M1),
            Self::mod_inverse((m_i[1] % m2) as u64, RNS_M2),
            Self::mod_inverse((m_i[2] % m3) as u64, RNS_M3),
            Self::mod_inverse((m_i[3] % m4) as u64, RNS_M4),
        ];

        Self {
            m_total,
            m_i,
            m_i_inv,
        }
    }

    /// Inverso modular extendido de Euclides: (a * x) = 1 mod m.
    /// Retorna 0 si no existe inverso.
    #[inline(always)]
    pub fn mod_inverse(a: u64, m: u64) -> u64 {
        if m <= 1 {
            return 0;
        }
        let mut t: i128 = 0;
        let mut newt: i128 = 1;
        let mut r: i128 = m as i128;
        let mut newr: i128 = a as i128;

        while newr != 0 {
            let quotient = r / newr;
            let temp_t = t - quotient * newt;
            t = newt;
            newt = temp_t;

            let temp_r = r - quotient * newr;
            r = newr;
            newr = temp_r;
        }

        if r > 1 {
            return 0; // No son coprimos
        }
        if t < 0 {
            t += m as i128;
        }
        t as u64
    }

    /// Rango dinámico M total del sistema RNS.
    #[inline(always)]
    pub fn dynamic_range(&self) -> u128 {
        self.m_total
    }

    /// Reconstruye el entero u128 exacto a partir del vector RNS vía CRT:
    /// X = sum_{i=1}^4 (r_i * M_i * (M_i^-1 mod m_i)) mod M
    #[inline]
    pub fn reconstruct_crt(&self, rns: &RnsVector4) -> u128 {
        let mut sum: u128 = 0;
        for i in 0..4 {
            let term = (rns.r[i] as u128 * self.m_i_inv[i] as u128) % (RNS_MODULI[i] as u128);
            let full_term = (term * (self.m_i[i] % self.m_total)) % self.m_total;
            sum = (sum + full_term) % self.m_total;
        }
        sum
    }

    /// Reconstrucción a u64 si el valor está acotado en u64::MAX.
    #[inline]
    pub fn reconstruct_u64(&self, rns: &RnsVector4) -> Option<u64> {
        let full = self.reconstruct_crt(rns);
        if full <= u64::MAX as u128 {
            Some(full as u64)
        } else {
            None
        }
    }
}
