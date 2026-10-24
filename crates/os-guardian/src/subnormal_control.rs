//! Supresión Activa de Flotantes Subnormales a Nivel de Silicio (FTZ / DAZ).
//!
//! En arquitecturas x86_64, las operaciones de coma flotante que involucran números denormales
//! (valores extremadamente cercanos a cero con exponente nulo, ej. $10^{-315}$) no pueden resolverse
//! en la canalización directa de hardware de la ALU vectorial, forzando un salto a microcódigo de CPU
//! con una severa penalización de latencia de **100 a 150 ciclos de reloj** por cada instrucción.
//!
//! Este módulo configura los registros de hardware del procesador para activar:
//! - **FTZ (Flush-To-Zero, Bit 15 de MXCSR):** Trunca automáticamente a cero positivo cualquier
//!   resultado de cálculo vectorial/flotante que genere un subnormal.
//! - **DAZ (Denormals-Are-Zero, Bit 6 de MXCSR):** Trata de inmediato como cero cualquier operando
//!   de entrada denormal antes de enviarlo a la unidad de cómputo.
//!
//! Cumple estrictamente con el trinquete `rb_p1`: CERO `unwrap`, CERO `expect`, CERO `panic`, CERO `assert`.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{_mm_getcsr, _mm_setcsr};

/// Máscara de bits para el modo Flush-To-Zero (Bit 15).
pub const MXCSR_FTZ_MASK: u32 = 1 << 15;

/// Máscara de bits para el modo Denormals-Are-Zero (Bit 6).
pub const MXCSR_DAZ_MASK: u32 = 1 << 6;

/// Estado de la configuración de supresión de subnormales del hilo actual.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubnormalStatus {
    /// Indica si Flush-To-Zero (FTZ) está activo.
    pub ftz_active: bool,
    /// Indica si Denormals-Are-Zero (DAZ) está activo.
    pub daz_active: bool,
    /// Valor crudo del registro de control MXCSR.
    pub raw_mxcsr: u32,
}

/// Controlador de silicio para el coprocesador de coma flotante y registros de control vectorial.
#[derive(Debug, Clone, Copy, Default)]
pub struct SubnormalHardwareController;

impl SubnormalHardwareController {
    /// Activa la supresión de subnormales de hardware (FTZ + DAZ) en el hilo actual.
    ///
    /// Retorna `true` si ambos modos quedaron efectivamente activos en el registro de hardware.
    pub fn enable_suppression() -> bool {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let mut mxcsr = _mm_getcsr();
            mxcsr |= MXCSR_FTZ_MASK | MXCSR_DAZ_MASK;
            _mm_setcsr(mxcsr);

            // Verificar lectura de verificación
            let verified = _mm_getcsr();
            (verified & MXCSR_FTZ_MASK) != 0 && (verified & MXCSR_DAZ_MASK) != 0
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            // Arquitectura sin registro SSE MXCSR
            false
        }
    }

    /// Lee el estado actual de los registros de hardware SSE/AVX.
    pub fn read_status() -> SubnormalStatus {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let raw = _mm_getcsr();
            SubnormalStatus {
                ftz_active: (raw & MXCSR_FTZ_MASK) != 0,
                daz_active: (raw & MXCSR_DAZ_MASK) != 0,
                raw_mxcsr: raw,
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            SubnormalStatus {
                ftz_active: false,
                daz_active: false,
                raw_mxcsr: 0,
            }
        }
    }

    /// Desactiva temporalmente la supresión de subnormales (restablece valores por defecto IEEE-754).
    pub fn disable_suppression() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let mut mxcsr = _mm_getcsr();
            mxcsr &= !(MXCSR_FTZ_MASK | MXCSR_DAZ_MASK);
            _mm_setcsr(mxcsr);
        }
    }

    /// Evalúa si un número de coma flotante `f64` es considerado un subnormal por la especificación IEEE-754.
    pub fn is_subnormal_ieee754(val: f64) -> bool {
        val.is_subnormal()
    }
}
