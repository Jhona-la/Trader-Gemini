//! # Fase 15.1 — Destrucción Diferida Lock-Free (`DeferredDropReaperRing`) y Despacho Polimórfico Estático Sin `dyn Trait` (`StaticSignalDispatch`)
//!
//! ## 👨‍🏫 MODO PROFESOR (QUÉ, POR QUÉ, PARA QUÉ, CÓMO, CUÁNDO, DÓNDE, QUIÉN)
//!
//! - **QUÉ:** Dos primitivas de silicio de ultra-baja latencia en Rust 100% nativo:
//!   1. `DeferredDropReaperRing<T, N>`: Cola circular SPSC wait-free con cursores atómicos aislados a 128 bytes donde el hilo del camino crítico retira estructuras (`retire_nonblocking`) en tiempo constante $\mathcal{O}(1)$ sin ejecutar `Drop::drop` en el hilo productor, delegando la destrucción determinista (`drop_in_place`) al hilo sepulturero fuera de línea (`reap_batch`).
//!   2. `StaticSignalDispatch<K>`: Tabla contigua de despacho polimórfico estático basada en uniones etiquetadas (`StaticSignalKernel`) alineadas a 64 bytes que reemplaza por completo el polimorfismo dinámico `&dyn Trait` (punteros gordos + saltos indirectos vtable).
//! - **POR QUÉ:** Invocar `drop()` en el hilo de trading o usar `&dyn Trait` introduce llamadas implícitas al asignador del sistema operativo (`HeapFree`/`munmap`) y fallos de predicción de saltos indirectos en el *Branch Target Buffer* (BTB) de la CPU.
//! - **PARA QUÉ:** Garantizar latencia determinista sub-nanosegundo en el camino crítico con cero invocaciones de `Drop` en caliente y cero indirecciones de tabla virtual (`vtable`).
//! - **CÓMO:** Almacenamiento en pila con `MaybeUninit<T>` protegido por cursores atómicos `Acquire`/`Release` alineados a 128B (`CacheAligned128`), y evaluación en bucle plano mediante un `enum` cerrado en registros SSE/AVX.
//! - **CUÁNDO:** Al descartar instantáneas de telemetría/estado en el bucle principal y al evaluar cadenas de filtros de señal en tiempo real.
//! - **DÓNDE:** `crates/os-guardian/src/deferred_drop_reaper.rs`.
//! - **QUIÉN:** `DeferredDropReaperRing`, `ReapBatchReport`, `StaticSignalKernel`, `StaticSignalDispatch`.

use crate::icache_sheaf_silicon::CacheAligned128;
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Reporte de recolección emitido por el hilo sepulturero tras drenar un lote de objetos retirados.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReapBatchReport {
    /// Cantidad de objetos destruidos (`drop_in_place`) en este lote.
    pub reaped_count: usize,
    /// Total acumulado de objetos retirados por el productor desde el inicio.
    pub total_retired: u64,
    /// Total acumulado de objetos destruidos por el sepulturero desde el inicio.
    pub total_reaped: u64,
    /// Objetos actualmente pendientes de destrucción en el anillo.
    pub pending_in_ring: usize,
}

/// Anillo SPSC lock-free/wait-free de destrucción diferida (*Deferred Drop Reaper*).
///
/// El hilo crítico transfiere la propiedad de `T` mediante `retire_nonblocking` sin ejecutar su destructor;
/// el hilo sepulturero invoca `reap_batch` fuera del camino crítico para ejecutar `drop_in_place`.
#[repr(C, align(128))]
pub struct DeferredDropReaperRing<T, const N: usize> {
    write_idx: CacheAligned128<AtomicUsize>,
    read_idx: CacheAligned128<AtomicUsize>,
    total_retired: CacheAligned128<AtomicU64>,
    total_reaped: CacheAligned128<AtomicU64>,
    slots: [UnsafeCell<MaybeUninit<T>>; N],
}

// SAFETY: `DeferredDropReaperRing` transfiere la propiedad de `T` entre el productor y el sepulturero
// mediante sincronización atómica `Acquire`/`Release` cuando `T: Send`.
unsafe impl<T: Send, const N: usize> Sync for DeferredDropReaperRing<T, N> {}
unsafe impl<T: Send, const N: usize> Send for DeferredDropReaperRing<T, N> {}

impl<T, const N: usize> Default for DeferredDropReaperRing<T, N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> DeferredDropReaperRing<T, N> {
    /// Construye un nuevo anillo de destrucción diferida en pila sin asignaciones en el heap.
    #[inline]
    pub const fn new() -> Self {
        Self {
            write_idx: CacheAligned128::new(AtomicUsize::new(0)),
            read_idx: CacheAligned128::new(AtomicUsize::new(0)),
            total_retired: CacheAligned128::new(AtomicU64::new(0)),
            total_reaped: CacheAligned128::new(AtomicU64::new(0)),
            slots: [const { UnsafeCell::new(MaybeUninit::uninit()) }; N],
        }
    }

    /// Capacidad máxima del anillo.
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Cantidad de objetos pendientes de destrucción en este instante.
    #[inline]
    pub fn pending_count(&self) -> usize {
        let w = self.write_idx.load(Ordering::Acquire);
        let r = self.read_idx.load(Ordering::Acquire);
        w.wrapping_sub(r)
    }

    /// Retira un objeto `item` desde el camino crítico en $\mathcal{O}(1)$ wait-free **sin ejecutar su `Drop`**.
    ///
    /// Si el anillo está lleno, devuelve `Err(item)` devolviendo la propiedad intacta al llamador.
    #[inline]
    pub fn retire_nonblocking(&self, item: T) -> Result<u64, T> {
        if N == 0 {
            return Err(item);
        }
        let w = self.write_idx.load(Ordering::Relaxed);
        let r = self.read_idx.load(Ordering::Acquire);
        if w.wrapping_sub(r) >= N {
            return Err(item);
        }
        let slot_idx = w % N;
        // SAFETY: El slot `slot_idx` está libre porque `w - r < N` y el productor es único.
        unsafe {
            (*self.slots[slot_idx].get()).write(item);
        }
        self.write_idx.store(w.wrapping_add(1), Ordering::Release);
        let seq = self.total_retired.fetch_add(1, Ordering::Relaxed) + 1;
        Ok(seq)
    }

    /// Ejecuta la destrucción diferida (`drop_in_place`) de hasta `max_batch` objetos en el hilo sepulturero.
    #[inline]
    pub fn reap_batch(&self, max_batch: usize) -> ReapBatchReport {
        if N == 0 {
            return ReapBatchReport {
                reaped_count: 0,
                total_retired: 0,
                total_reaped: 0,
                pending_in_ring: 0,
            };
        }
        let limit = max_batch.min(N);
        let mut r = self.read_idx.load(Ordering::Relaxed);
        let w = self.write_idx.load(Ordering::Acquire);
        let available = w.wrapping_sub(r);
        let to_reap = available.min(limit);

        let mut count = 0;
        while count < to_reap {
            let slot_idx = r % N;
            // SAFETY: `r < w` bajo sincronización `Acquire` garantiza que el slot fue inicializado por `retire_nonblocking`
            // y que aún no ha sido destruido.
            unsafe {
                let maybe_ref = &mut *self.slots[slot_idx].get();
                ptr::drop_in_place(maybe_ref.as_mut_ptr());
            }
            r = r.wrapping_add(1);
            count += 1;
        }

        if count > 0 {
            self.read_idx.store(r, Ordering::Release);
            self.total_reaped.fetch_add(count as u64, Ordering::Relaxed);
        }

        let total_ret = self.total_retired.load(Ordering::Relaxed);
        let total_rep = self.total_reaped.load(Ordering::Relaxed);
        ReapBatchReport {
            reaped_count: count,
            total_retired: total_ret,
            total_reaped: total_rep,
            pending_in_ring: w.wrapping_sub(r),
        }
    }

    /// Drena y destruye la totalidad de los objetos pendientes en el anillo.
    #[inline]
    pub fn reap_all(&self) -> ReapBatchReport {
        self.reap_batch(N)
    }
}

impl<T, const N: usize> Drop for DeferredDropReaperRing<T, N> {
    fn drop(&mut self) {
        let _ = self.reap_all();
    }
}

/// Kernel elemental de procesamiento de señal para despacho polimórfico estático libre de `dyn Trait`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StaticSignalKernel {
    /// Etapa identidad / inactiva.
    Identity,
    /// Transformación afín acotada $y = \text{clamp}(\alpha x + \beta, -L, L)$.
    AffineClamp { alpha: f64, beta: f64, limit: f64 },
    /// Compresor sigmoidal suave hiperbólico $y = g \cdot \tanh(s \cdot x)$.
    HyperbolicTanh { scale: f64, gain: f64 },
    /// Filtro de umbral suave (*Soft-Thresholding* LASSO / Wavelet) $y = \text{sign}(x) \max(0, |x| - \tau)$.
    SoftThreshold { threshold: f64, gain: f64 },
    /// Resonador cuadrático con preservación de signo $y = g \cdot x |x|$.
    SignedQuadratic { gain: f64 },
}

impl StaticSignalKernel {
    /// Evalúa el kernel en línea sin saltos indirectos vtable ni asignaciones dinámicas.
    #[inline(always)]
    pub fn evaluate(&self, x: f64) -> f64 {
        let clean_x = if x.is_finite() {
            x.clamp(-1e6, 1e6)
        } else {
            0.0
        };
        let out = match *self {
            Self::Identity => clean_x,
            Self::AffineClamp { alpha, beta, limit } => {
                let a = if alpha.is_finite() { alpha } else { 1.0 };
                let b = if beta.is_finite() { beta } else { 0.0 };
                let lim = if limit.is_finite() && limit > 0.0 {
                    limit
                } else {
                    1.0
                };
                clean_x.mul_add(a, b).clamp(-lim, lim)
            }
            Self::HyperbolicTanh { scale, gain } => {
                let s = if scale.is_finite() { scale } else { 1.0 };
                let g = if gain.is_finite() { gain } else { 1.0 };
                g * (s * clean_x).tanh()
            }
            Self::SoftThreshold { threshold, gain } => {
                let tau = if threshold.is_finite() && threshold > 0.0 {
                    threshold
                } else {
                    0.0
                };
                let g = if gain.is_finite() { gain } else { 1.0 };
                let abs_x = clean_x.abs();
                if abs_x <= tau {
                    0.0
                } else {
                    g * clean_x.signum() * (abs_x - tau)
                }
            }
            Self::SignedQuadratic { gain } => {
                let g = if gain.is_finite() { gain } else { 1.0 };
                g * clean_x * clean_x.abs()
            }
        };
        if out.is_finite() {
            out
        } else {
            0.0
        }
    }
}

/// Pipeline contiguo de despacho estático (`#[repr(C, align(64))]`) que ejecuta hasta `K` etapas
/// sin una sola indirección de tabla virtual (`dyn Trait`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticSignalDispatch<const K: usize> {
    pub stages: [StaticSignalKernel; K],
    pub weights: [f64; K],
    pub active_len: usize,
}

impl<const K: usize> Default for StaticSignalDispatch<K> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<const K: usize> StaticSignalDispatch<K> {
    /// Construye un pipeline estático inicializado con etapas `Identity`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            stages: [StaticSignalKernel::Identity; K],
            weights: [1.0; K],
            active_len: 0,
        }
    }

    /// Registra una nueva etapa estática en la posición siguiente disponible.
    #[inline]
    pub fn push_stage(&mut self, kernel: StaticSignalKernel, weight: f64) -> bool {
        if self.active_len >= K {
            return false;
        }
        self.stages[self.active_len] = kernel;
        self.weights[self.active_len] = if weight.is_finite() { weight } else { 0.0 };
        self.active_len += 1;
        true
    }

    /// Evalúa en serie (composición funcional $f_{K-1} \circ \dots \circ f_0(x)$) las etapas activas.
    #[inline]
    pub fn evaluate_cascade(&self, input: f64) -> f64 {
        let n = self.active_len.min(K);
        let mut acc = if input.is_finite() { input } else { 0.0 };
        let mut i = 0;
        while i < n {
            let w = self.weights[i];
            acc = w * self.stages[i].evaluate(acc);
            i += 1;
        }
        if acc.is_finite() {
            acc
        } else {
            0.0
        }
    }

    /// Evalúa en paralelo (combinación lineal $\sum_{i=0}^{n-1} w_i f_i(x)$) las etapas activas.
    #[inline]
    pub fn evaluate_parallel_ensemble(&self, input: f64) -> f64 {
        let n = self.active_len.min(K);
        let x = if input.is_finite() { input } else { 0.0 };
        let mut sum = 0.0;
        let mut i = 0;
        while i < n {
            sum = self.weights[i].mul_add(self.stages[i].evaluate(x), sum);
            i += 1;
        }
        if sum.is_finite() {
            sum
        } else {
            0.0
        }
    }
}
