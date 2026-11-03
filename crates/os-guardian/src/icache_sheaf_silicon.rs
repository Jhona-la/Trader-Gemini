//! # FASE 12: Aislamiento de Silicio a 128 Bytes & Precalentamiento Fantasma de I-Cache / BTB
//!
//! **QUÉ:** Primitivas de hardware para aislamiento de líneas de caché pares/impares
//! (`#[repr(C, align(128))]`), cola SPSC lock-free de Lamport con índices sombra
//! aislados a 128 bytes, y motor de precalentamiento en seco (*Phantom Branch Warmer*)
//! de la caché de instrucciones L1I y del *Branch Target Buffer* (BTB).
//!
//! **POR QUÉ:**
//! 1. Los procesadores modernos x86_64 (Intel Golden/Raptor Cove, AMD Zen 4/5) poseen un
//!    *Spatial Adjacent Cache-Line Prefetcher* en L2 que empareja líneas contiguas de 64 bytes
//!    en bloques físicos de 128 bytes. Alinear variables atómicas a solo 64 bytes sigue
//!    provocando *false sharing* e invalidaciones cruzadas MESI cuando dos hilos operan en
//!    las mitades par e impar del mismo par de 128 bytes.
//! 2. Cuando el flujo de mercado pasa varios milisegundos sin disparar órdenes reales, las
//!    instrucciones de evaluación de riesgo, serialización binaria y firma HMAC son desalojadas
//!    de L1I y sus predictores de salto son purgados del BTB. Al llegar un evento real, el CPU
//!    sufre penalizaciones de 15–20 ciclos por *branch misprediction* y fallos de L1I.
//!
//! **PARA QUÉ:** Garantizar latencia determinista en nanosegundos, eliminando el *false sharing*
//! del prefetcher espacial y manteniendo calientes las líneas de L1I y el BTB con cero efectos
//! secundarios en el mercado (*zero side effects*).

use crate::simd_crypto::sha256_hmac_stack;
use std::cell::Cell;
use std::hint::black_box;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Tamaño canónico del par de líneas de caché emparejadas por el Spatial Prefetcher (2 x 64B).
pub const SPATIAL_PREFETCH_LINE_BYTES: usize = 128;

/// Envoltorio alineado estrictamente a 128 bytes para neutralizar tanto el *false sharing*
/// directo de L1D (64B) como el *false sharing* inducido por el *Spatial Adjacent Prefetcher* (128B).
#[repr(C, align(128))]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CacheAligned128<T> {
    value: T,
}

impl<T> CacheAligned128<T> {
    /// Construye una instancia alineada a 128 bytes en memoria.
    #[inline(always)]
    pub const fn new(value: T) -> Self {
        Self { value }
    }

    /// Devuelve una referencia inmutable al valor interno.
    #[inline(always)]
    pub const fn get(&self) -> &T {
        &self.value
    }

    /// Devuelve una referencia mutable al valor interno.
    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.value
    }

    /// Consume el contenedor y extrae el valor subyacente.
    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.value
    }

    /// Devuelve la dirección física/virtual numérica del bloque en memoria.
    #[inline(always)]
    pub fn addr(&self) -> usize {
        self as *const Self as usize
    }

    /// Verifica en tiempo de ejecución que la dirección sea múltiplo exacto de 128 bytes.
    #[inline(always)]
    pub fn is_aligned_128(&self) -> bool {
        self.addr().is_multiple_of(SPATIAL_PREFETCH_LINE_BYTES)
    }
}

impl<T> Deref for CacheAligned128<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> DerefMut for CacheAligned128<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

/// Cola circular SPSC (Single-Producer Single-Consumer) de Lamport en stack con
/// aislamiento estricto de 128 bytes entre cursores atómicos y cursores sombra cacheados.
#[repr(C, align(128))]
pub struct LamportRing128<T: Copy + Default, const N: usize> {
    /// Secuencia monotónica de escritura (modificada por el productor, leída por el consumidor).
    write_seq: CacheAligned128<AtomicUsize>,
    /// Copia local de `read_seq` cacheada para el productor (evita lecturas atómicas cruzadas MESI).
    cached_read_seq: CacheAligned128<Cell<usize>>,
    /// Secuencia monotónica de lectura (modificada por el consumidor, leída por el productor).
    read_seq: CacheAligned128<AtomicUsize>,
    /// Copia local de `write_seq` cacheada para el consumidor (evita lecturas atómicas cruzadas MESI).
    cached_write_seq: CacheAligned128<Cell<usize>>,
    /// Almacenamiento contiguo pre-asignado en stack.
    buffer: CacheAligned128<[Cell<T>; N]>,
}

impl<T: Copy + Default, const N: usize> Default for LamportRing128<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Default, const N: usize> LamportRing128<T, N> {
    /// Inicializa la cola circular SPSC con capacidad fija `N` en stack.
    #[inline]
    pub fn new() -> Self {
        Self {
            write_seq: CacheAligned128::new(AtomicUsize::new(0)),
            cached_read_seq: CacheAligned128::new(Cell::new(0)),
            read_seq: CacheAligned128::new(AtomicUsize::new(0)),
            cached_write_seq: CacheAligned128::new(Cell::new(0)),
            buffer: CacheAligned128::new(std::array::from_fn(|_| Cell::new(T::default()))),
        }
    }

    /// Capacidad máxima de elementos almacenables en el anillo.
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Cantidad actual de elementos en tránsito dentro de la cola.
    #[inline(always)]
    pub fn len(&self) -> usize {
        let w = self.write_seq.load(Ordering::Acquire);
        let r = self.read_seq.load(Ordering::Acquire);
        w.wrapping_sub(r).min(N)
    }

    /// Indica si la cola carece de elementos pendientes.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Inserta un elemento en $O(1)$ consultando primero el índice sombra local antes de
    /// sincronizar por bus MESI. Devuelve `Err(item)` si la cola está llena o `N == 0`.
    #[inline(always)]
    pub fn try_push(&self, item: T) -> Result<(), T> {
        if N == 0 {
            return Err(item);
        }
        let w = self.write_seq.load(Ordering::Relaxed);
        let mut r_cached = self.cached_read_seq.get().get();

        if w.wrapping_sub(r_cached) >= N {
            r_cached = self.read_seq.load(Ordering::Acquire);
            self.cached_read_seq.get().set(r_cached);
            if w.wrapping_sub(r_cached) >= N {
                return Err(item);
            }
        }

        let slot_idx = w % N;
        if let Some(slot) = self.buffer.get().get(slot_idx) {
            slot.set(item);
            self.write_seq.store(w.wrapping_add(1), Ordering::Release);
            Ok(())
        } else {
            Err(item)
        }
    }

    /// Extrae el siguiente elemento en orden FIFO $O(1)$ consultando primero el cursor sombra local.
    #[inline(always)]
    pub fn try_pop(&self) -> Option<T> {
        if N == 0 {
            return None;
        }
        let r = self.read_seq.load(Ordering::Relaxed);
        let mut w_cached = self.cached_write_seq.get().get();

        if r == w_cached {
            w_cached = self.write_seq.load(Ordering::Acquire);
            self.cached_write_seq.get().set(w_cached);
            if r == w_cached {
                return None;
            }
        }

        let slot_idx = r % N;
        let item = self.buffer.get().get(slot_idx)?.get();
        self.read_seq.store(r.wrapping_add(1), Ordering::Release);
        Some(item)
    }

    /// Verifica formalmente que todos los bloques críticos de control estén alineados a 128 bytes
    /// y separados por al menos 128 bytes entre sí (cero *false sharing* espacial).
    #[inline]
    pub fn verify_layout_isolation(&self) -> bool {
        let a_w = self.write_seq.addr();
        let a_cr = self.cached_read_seq.addr();
        let a_r = self.read_seq.addr();
        let a_cw = self.cached_write_seq.addr();
        let a_buf = self.buffer.addr();

        let aligned = self.write_seq.is_aligned_128()
            && self.cached_read_seq.is_aligned_128()
            && self.read_seq.is_aligned_128()
            && self.cached_write_seq.is_aligned_128()
            && self.buffer.is_aligned_128();

        let separated = a_cr.abs_diff(a_w) >= SPATIAL_PREFETCH_LINE_BYTES
            && a_r.abs_diff(a_cr) >= SPATIAL_PREFETCH_LINE_BYTES
            && a_cw.abs_diff(a_r) >= SPATIAL_PREFETCH_LINE_BYTES
            && a_buf.abs_diff(a_cw) >= SPATIAL_PREFETCH_LINE_BYTES;

        aligned && separated
    }
}

/// Candidato sintético de orden para ejercitar el pipeline de decisión y firma en seco.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhantomOrderCandidate {
    /// Identificador numérico del activo.
    pub symbol_id: u32,
    /// Precio límite propuesto.
    pub price: f64,
    /// Cantidad propuesta en unidades del activo.
    pub quantity: f64,
    /// Apalancamiento efectivo solicitado.
    pub leverage: f64,
    /// Distancia fraccional al stop-loss.
    pub stop_distance_pct: f64,
    /// Tasa de margen de mantenimiento del símbolo.
    pub maintenance_margin_rate: f64,
    /// Valor esperado bruto estimado en fracción.
    pub expected_edge_pct: f64,
    /// Fricción roundtrip estimada (comisiones + slippage).
    pub roundtrip_friction_pct: f64,
    /// Marca de tiempo monotónica en nanosegundos.
    pub timestamp_ns: u64,
}

impl Default for PhantomOrderCandidate {
    fn default() -> Self {
        Self {
            symbol_id: 1,
            price: 65_000.0,
            quantity: 0.002,
            leverage: 5.0,
            stop_distance_pct: 0.015,
            maintenance_margin_rate: 0.004,
            expected_edge_pct: 0.0045,
            roundtrip_friction_pct: 0.0010,
            timestamp_ns: 1_700_000_000_000_000_000,
        }
    }
}

/// Reporte determinista del ciclo de precalentamiento fantasma de I-Cache y BTB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhantomWarmupReport {
    /// Número de iteraciones completas ejecutadas sobre el pipeline en seco.
    pub iterations_executed: u32,
    /// Número de ramas condicionales de validación y codificación tomadas.
    pub branches_exercised: u32,
    /// Huella truncada de 64 bits del último HMAC-SHA256 calculado en stack.
    pub last_digest_tag: u64,
    /// Contador estricto de efectos secundarios o envíos reales (siempre 0 en modo fantasma).
    pub side_effects_committed: u64,
}

/// Motor de precalentamiento de caché de instrucciones (L1I) y *Branch Target Buffer* (BTB).
/// Ejecuta las ramas calientes de validación matemática, empaquetado binario de 48 bytes y
/// autenticación HMAC-SHA256 en registros de pila, abortando en la compuerta final mediante
/// `black_box(false)` para impedir que el optimizador elimine el código y garantizar cero efectos externos.
#[repr(C, align(128))]
pub struct PhantomBranchWarmer {
    secret_key: [u8; 32],
    total_warmups: CacheAligned128<AtomicU64>,
    committed_side_effects: CacheAligned128<AtomicU64>,
}

impl Default for PhantomBranchWarmer {
    fn default() -> Self {
        Self::new([0x5A; 32])
    }
}

impl PhantomBranchWarmer {
    /// Construye un nuevo precalentador fantasma con una clave efímera en stack.
    #[inline]
    pub const fn new(secret_key: [u8; 32]) -> Self {
        Self {
            secret_key,
            total_warmups: CacheAligned128::new(AtomicU64::new(0)),
            committed_side_effects: CacheAligned128::new(AtomicU64::new(0)),
        }
    }

    /// Devuelve el total acumulado de pasadas de precalentamiento ejecutadas.
    #[inline(always)]
    pub fn total_warmups(&self) -> u64 {
        self.total_warmups.load(Ordering::Relaxed)
    }

    /// Devuelve el contador de efectos secundarios reales (invariante: 0).
    #[inline(always)]
    pub fn committed_side_effects(&self) -> u64 {
        self.committed_side_effects.load(Ordering::Acquire)
    }

    /// Codifica el candidato en un frame binario de 48 bytes en stack (formato SBE-like).
    #[inline(always)]
    pub fn encode_phantom_wire_frame(candidate: &PhantomOrderCandidate) -> [u8; 48] {
        let mut frame = [0u8; 48];
        frame[0..4].copy_from_slice(&candidate.symbol_id.to_le_bytes());
        frame[4..12].copy_from_slice(&candidate.price.to_bits().to_le_bytes());
        frame[12..20].copy_from_slice(&candidate.quantity.to_bits().to_le_bytes());
        frame[20..28].copy_from_slice(&candidate.leverage.to_bits().to_le_bytes());
        frame[28..36].copy_from_slice(&candidate.stop_distance_pct.to_bits().to_le_bytes());
        frame[36..44].copy_from_slice(&candidate.timestamp_ns.to_le_bytes());
        let notional_flag: u32 = if candidate.price * candidate.quantity >= 5.0 {
            0x0102_0304
        } else {
            0x0000_0000
        };
        frame[44..48].copy_from_slice(&notional_flag.to_le_bytes());
        frame
    }

    /// Ejecuta `iterations` pasadas de precalentamiento de L1I y BTB sobre el candidato base.
    #[inline(never)]
    pub fn warm_hot_path_branches(
        &self,
        base_candidate: &PhantomOrderCandidate,
        iterations: u32,
    ) -> PhantomWarmupReport {
        let bounded_iters = iterations.clamp(1, 256);
        let mut branches_exercised = 0u32;
        let mut last_digest_tag = 0u64;

        for step in 0..bounded_iters {
            let mut candidate = black_box(*base_candidate);
            candidate.timestamp_ns = candidate.timestamp_ns.wrapping_add(u64::from(step));

            // Rama 1: Validación de finitud IEEE-754 y positividad estricta
            let finite_ok = candidate.price.is_finite()
                && candidate.price > 0.0
                && candidate.quantity.is_finite()
                && candidate.quantity > 0.0
                && candidate.leverage.is_finite()
                && candidate.leverage >= 1.0;
            if finite_ok {
                branches_exercised = branches_exercised.saturating_add(1);
            }

            // Rama 2: Guarda exacta de distancia de liquidación aislada
            let liq_lhs = candidate.leverage
                * (candidate.stop_distance_pct * (1.0 + candidate.maintenance_margin_rate)
                    + candidate.maintenance_margin_rate);
            if liq_lhs.is_finite() && liq_lhs < 0.95 {
                branches_exercised = branches_exercised.saturating_add(1);
            }

            // Rama 3: Blindaje de viabilidad EV >= 3x fricción roundtrip
            let required_edge = 3.0 * candidate.roundtrip_friction_pct.max(1e-6);
            if candidate.expected_edge_pct >= required_edge {
                branches_exercised = branches_exercised.saturating_add(1);
            }

            // Rama 4: Empaquetado binario y firma HMAC-SHA256 en stack
            let frame = Self::encode_phantom_wire_frame(&candidate);
            let mac = sha256_hmac_stack(&self.secret_key, &frame);
            let mut tag_bytes = [0u8; 8];
            tag_bytes.copy_from_slice(&mac[0..8]);
            last_digest_tag ^= u64::from_le_bytes(tag_bytes);
            branches_exercised = branches_exercised.saturating_add(1);

            // Compuerta fantasma: black_box(false) impide la eliminación de código muerto por LLVM
            // mientras garantiza que jamás se ejecute el bloque de efecto secundario.
            let commit_gate = black_box(false);
            if commit_gate && last_digest_tag == u64::MAX {
                self.committed_side_effects.fetch_add(1, Ordering::SeqCst);
            }
        }

        self.total_warmups
            .fetch_add(u64::from(bounded_iters), Ordering::Relaxed);

        PhantomWarmupReport {
            iterations_executed: bounded_iters,
            branches_exercised,
            last_digest_tag: black_box(last_digest_tag),
            side_effects_committed: self.committed_side_effects(),
        }
    }
}
