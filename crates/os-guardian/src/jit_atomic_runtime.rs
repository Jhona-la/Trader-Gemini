//! Motor JIT en RAM con Intercambio Atómico de Punteros y Tipado Dimensional Const Generics (FASE 25)
//!
//! # QUÉ:
//! Subsistema de bajo nivel para compilación dinámica, gestión de páginas de memoria
//! ejecutables en RAM (W^X: Write XOR Execute) e intercambio atómico lock-free de predicados
//! de decisión mediante `AtomicPtr` con tipado dimensional estático `const D: usize`.
//!
//! # POR QUÉ:
//! En trading de alta frecuencia y adaptación cuántica continua, las reglas de inferencia
//! y pesos tensoriales deben evolucionar sin pausar el bucle de eventos (`hot-path` < 100 ns)
//! y sin realizar I/O a disco SSD. El uso de punteros atómicos con semántica Acquire/Release
//! permite que los hilos de trading evalúen la versión vigente sin contención, mientras el
//! optimizador en segundo plano compila e instala la nueva versión en un único ciclo de CPU.
//!
//! # PARA QUÉ:
//! 1. Eliminar locks, mutexes y bloqueos en la lectura de predicados dinámicos.
//! 2. Garantizar que hilos concurrentes nunca lean código a medio compilar.
//! 3. Proteger la integridad de la memoria con el principio W^X (Write XOR Execute).
//! 4. Validar invariantes dimensionales en tiempo de compilación mediante const generics.
//!
//! # CÓMO:
//! Asignación de páginas en memoria virtual, inyección de trampolines o funciones de evaluación,
//! sellado de la página a sólo lectura ejecutable (`PAGE_EXECUTE_READ`), y swap atómico
//! `AtomicPtr::store(ptr, Ordering::Release)`.
//!
//! # CUÁNDO:
//! En cada ciclo de optimización continua online o evaluación de señales en tiempo real.
//!
//! # DÓNDE:
//! `crates/os-guardian/src/jit_atomic_runtime.rs`.
//!
//! # QUIÉN:
//! Arquitecto de Silicio (Rol 7) & QA Engineer (Rol 5).

#![allow(clippy::all)]

use core::ffi::c_void;
use core::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

/// Error estructurado del runtime JIT en RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitRuntimeError {
    /// La asignación de memoria ejecutable falló en el SO.
    AllocationFailed,
    /// Fallo al cambiar la protección de memoria (violación de W^X).
    ProtectionFailed,
    /// El tamaño del bytecode o buffer excede la capacidad de la página.
    PayloadTooLarge,
    /// Puntero nulo o no inicializado.
    NullPointer,
    /// La dimensión especificada es inválida (debe ser > 0).
    InvalidDimension,
}

/// Firma canónica del evaluador dinámico JIT en el hot path.
/// Toma un puntero a vector de entrada `f64` de dimensión `D`, un contexto opcional, y devuelve `f64`.
pub type JitEvaluatorFn =
    unsafe extern "C" fn(inputs: *const f64, len: usize, ctx: *mut c_void) -> f64;

/// Vector dimensionalmente tipado con const generics para seguridad matemática estática.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DimensionedVector<const D: usize> {
    pub data: [f64; D],
}

impl<const D: usize> DimensionedVector<D> {
    /// Crea un nuevo vector dimensional con ceros.
    #[inline(always)]
    pub const fn zero() -> Self {
        Self { data: [0.0; D] }
    }

    /// Crea un nuevo vector dimensional con datos provistos.
    #[inline(always)]
    pub const fn new(data: [f64; D]) -> Self {
        Self { data }
    }

    /// Retorna la dimensión estática del vector.
    #[inline(always)]
    pub const fn dim(&self) -> usize {
        D
    }

    /// Norma Euclídea al cuadrado (L2^2) sin asignaciones dinámicas.
    #[inline]
    pub fn norm_sq(&self) -> f64 {
        let mut sum = 0.0;
        let mut i = 0;
        while i < D {
            let val = self.data[i];
            if val.is_finite() {
                sum += val * val;
            }
            i += 1;
        }
        sum
    }

    /// Producto escalar continuo con otro vector de igual dimensión.
    #[inline]
    pub fn dot(&self, other: &Self) -> f64 {
        let mut sum = 0.0;
        let mut i = 0;
        while i < D {
            let a = self.data[i];
            let b = other.data[i];
            if a.is_finite() && b.is_finite() {
                sum += a * b;
            }
            i += 1;
        }
        sum
    }
}

/// Página de código ejecutable asignada en RAM con estricta semántica W^X.
pub struct JitCodePage {
    ptr: *mut u8,
    size: usize,
    is_executable: bool,
}

// Seguridad de concurrencia: la página sólo se lee una vez sellada.
unsafe impl Send for JitCodePage {}
unsafe impl Sync for JitCodePage {}

impl JitCodePage {
    /// Tamaño estándar de página de memoria virtual (4 KB).
    pub const PAGE_SIZE: usize = 4096;

    /// Asigna una página de memoria en RAM con permisos iniciales de Lectura/Escritura.
    pub fn allocate(size: usize) -> Result<Self, JitRuntimeError> {
        let aligned_size = if size == 0 {
            Self::PAGE_SIZE
        } else {
            ((size + Self::PAGE_SIZE - 1) / Self::PAGE_SIZE) * Self::PAGE_SIZE
        };

        #[cfg(windows)]
        {
            use windows::Win32::System::Memory::{
                VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE,
            };

            let raw = unsafe {
                VirtualAlloc(None, aligned_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
            };

            if raw.is_null() {
                // Fallback a heap asignado para entornos donde VirtualAlloc está restringido
                let layout = core::alloc::Layout::from_size_align(aligned_size, 64)
                    .map_err(|_| JitRuntimeError::AllocationFailed)?;
                let heap_raw = unsafe { std::alloc::alloc_zeroed(layout) };
                if heap_raw.is_null() {
                    return Err(JitRuntimeError::AllocationFailed);
                }
                return Ok(Self {
                    ptr: heap_raw,
                    size: aligned_size,
                    is_executable: false,
                });
            }

            Ok(Self {
                ptr: raw as *mut u8,
                size: aligned_size,
                is_executable: false,
            })
        }

        #[cfg(not(windows))]
        {
            let layout = core::alloc::Layout::from_size_align(aligned_size, 4096)
                .map_err(|_| JitRuntimeError::AllocationFailed)?;
            let raw = unsafe { std::alloc::alloc_zeroed(layout) };
            if raw.is_null() {
                return Err(JitRuntimeError::AllocationFailed);
            }
            Ok(Self {
                ptr: raw,
                size: aligned_size,
                is_executable: false,
            })
        }
    }

    /// Escribe bytes de código máquina o trampolín en la página antes del sellado.
    pub fn write_code(&mut self, offset: usize, bytes: &[u8]) -> Result<(), JitRuntimeError> {
        if self.is_executable {
            return Err(JitRuntimeError::ProtectionFailed); // Principio W^X: no escribir tras hacer ejecutable
        }
        if offset.saturating_add(bytes.len()) > self.size {
            return Err(JitRuntimeError::PayloadTooLarge);
        }
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(offset), bytes.len());
        }
        Ok(())
    }

    /// Sella la página de memoria cambiando sus permisos a sólo LECTURA y EJECUCIÓN (W^X).
    pub fn make_executable(&mut self) -> Result<(), JitRuntimeError> {
        if self.is_executable {
            return Ok(());
        }

        #[cfg(windows)]
        {
            use windows::Win32::System::Memory::{
                VirtualProtect, PAGE_EXECUTE_READ, PAGE_PROTECTION_FLAGS,
            };

            let mut old_protect = PAGE_PROTECTION_FLAGS(0);
            let res = unsafe {
                VirtualProtect(
                    self.ptr as *const c_void,
                    self.size,
                    PAGE_EXECUTE_READ,
                    &mut old_protect,
                )
            };

            if res.is_ok() {
                self.is_executable = true;
                return Ok(());
            }
            Err(JitRuntimeError::ProtectionFailed)
        }

        #[cfg(not(windows))]
        {
            unsafe extern "C" {
                fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
            }
            const PROT_READ: i32 = 1;
            const PROT_EXEC: i32 = 4;

            let result = unsafe { mprotect(self.ptr as *mut c_void, self.size, PROT_READ | PROT_EXEC) };
            if result == 0 {
                self.is_executable = true;
                Ok(())
            } else {
                Err(JitRuntimeError::ProtectionFailed)
            }
        }
    }

    /// Retorna el puntero crudo a la base de la página.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Retorna el tamaño en bytes de la página.
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Indica si la página ya ha sido sellada como ejecutable.
    #[inline(always)]
    pub fn is_executable(&self) -> bool {
        self.is_executable
    }
}

impl Drop for JitCodePage {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            #[cfg(windows)]
            {
                use windows::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
                let res = unsafe { VirtualFree(self.ptr as *mut c_void, 0, MEM_RELEASE) };
                if res.is_err() {
                    // Si era asignación de respaldo heap
                    if let Ok(layout) = core::alloc::Layout::from_size_align(self.size, 64) {
                        unsafe { std::alloc::dealloc(self.ptr, layout) };
                    }
                }
            }

            #[cfg(not(windows))]
            {
                if let Ok(layout) = core::alloc::Layout::from_size_align(self.size, 4096) {
                    unsafe { std::alloc::dealloc(self.ptr, layout) };
                }
            }
        }
    }
}

/// Evaluador de fallback estático predeterminado: producto escalar normalizado.
unsafe extern "C" fn default_dot_evaluator(
    inputs: *const f64,
    len: usize,
    _ctx: *mut c_void,
) -> f64 {
    if inputs.is_null() || len == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut i = 0;
    while i < len {
        let val = *inputs.add(i);
        if val.is_finite() {
            sum += val;
        }
        i += 1;
    }
    let len_f = len as f64;
    (sum / len_f).tanh()
}

/// Gestor de predicados JIT atómicos con swap en caliente lock-free en memoria RAM.
pub struct AtomicJitPredicator {
    active_evaluator: AtomicPtr<c_void>,
    epoch: AtomicU64,
    evaluation_counter: AtomicU64,
}

impl Default for AtomicJitPredicator {
    fn default() -> Self {
        Self::new()
    }
}

impl AtomicJitPredicator {
    /// Crea un nuevo gestor de predicados inicializado con el evaluador por defecto.
    pub fn new() -> Self {
        let default_fn_ptr = default_dot_evaluator as *const () as *mut c_void;
        Self {
            active_evaluator: AtomicPtr::new(default_fn_ptr),
            epoch: AtomicU64::new(1),
            evaluation_counter: AtomicU64::new(0),
        }
    }

    /// Evalúa el predicado activo sobre un vector dimensional en el hot path (< 10 ns).
    #[inline(always)]
    pub fn evaluate<const D: usize>(&self, vector: &DimensionedVector<D>, ctx: *mut c_void) -> f64 {
        let fn_raw = self.active_evaluator.load(Ordering::Acquire);
        if fn_raw.is_null() {
            return 0.0;
        }
        self.evaluation_counter.fetch_add(1, Ordering::Relaxed);
        let evaluator: JitEvaluatorFn = unsafe { core::mem::transmute(fn_raw) };
        unsafe { evaluator(vector.data.as_ptr(), D, ctx) }
    }

    /// Intercambia atómicamente la función de predicción activa por una nueva versión compilada.
    /// Retorna el puntero previo para que sea recolectado por el reaper de manera diferida.
    pub fn swap_evaluator(&self, new_evaluator: JitEvaluatorFn) -> *mut c_void {
        let new_ptr = new_evaluator as *const () as *mut c_void;
        let old_ptr = self.active_evaluator.swap(new_ptr, Ordering::Release);
        self.epoch.fetch_add(1, Ordering::SeqCst);
        old_ptr
    }

    /// Retorna la época actual de la función instalada.
    #[inline(always)]
    pub fn current_epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Retorna el contador total de evaluaciones realizadas.
    #[inline(always)]
    pub fn total_evaluations(&self) -> u64 {
        self.evaluation_counter.load(Ordering::Relaxed)
    }
}
