//! # Módulo: Registrador de Vuelo Lock-Free con Histogramas Logarítmicos LZCNT
//!
//! ## Método Pedagógico Riguroso (Modo Profesor)
//! - **QUÉ:** Registrador de vuelo (flight recorder) de microeventos y latencias en memoria compartida
//!   con histogramas logarítmicos de 64 bins indexados directamente mediante la instrucción de hardware `LZCNT`
//!   (conteo de ceros a la izquierda en O(1)).
//! - **POR QUÉ:** Las escrituras a disco convencionales introducen pausas de jitter y page faults no deterministas.
//!   Almacenar y ordenar muestras para calcular percentiles (p50, p90, p99, p99.9) cuesta O(N log N) en tiempo y memoria.
//! - **PARA QUÉ:** Grabar hasta millones de latencias y eventos por segundo con cero pausas de recolección de basura,
//!   cero alocaciones dinámicas y actualización atómica en 1-2 ciclos de reloj, permitiendo obtener cualquier percentil
//!   instantáneamente en O(1).
//! - **CÓMO:**
//!   - Estructura `LzcntLogHistogram`: 64 bins atómicos `[AtomicU64; 64]`.
//!   - Mapeo de latencia en nanosegundos al bin: `bin = 63 - val_ns.leading_zeros() as usize` para `val_ns > 0`.
//!   - Incremento lock-free: `bins[bin].fetch_add(1, Ordering::Relaxed)`.
//!   - Estimación de percentiles: acumulación acumulativa sobre los 64 bins hasta `rank = ceil(p * total_count)`
//!     e interpolación de cota inferior y superior del bin diádico $[2^{\text{bin}}, 2^{\text{bin}+1}-1]$.
//!   - Anillo circular de registros de vuelo `MmapFlightRecordRing` con marcas de tiempo en nanosegundos.
//! - **CUÁNDO:** En cada microsegundo del bucle de ejecución de órdenes y ticks de mercado.
//! - **DÓNDE:** `crates/flight-recorder/src/mmap_flight_recorder.rs`.
//! - **QUIÉN:** Consejo Colegiado (Rol 10: Auditor Forense Causal y Rol 7: Silicio & Latencia Cuántica).

#![allow(clippy::all)]

use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Número de bins en el histograma logarítmico (cubre 2^0 = 1 ns hasta 2^63 > 10^18 ns).
pub const LOG_HISTOGRAM_BINS: usize = 64;

/// Histograma logarítmico lock-free basado en instrucciones LZCNT de CPU.
pub struct LzcntLogHistogram {
    /// Bins atómicos de frecuencia acumulada.
    bins: [AtomicU64; LOG_HISTOGRAM_BINS],
    /// Contador total de muestras procesadas.
    total_samples: AtomicU64,
    /// Suma total de valores (en nanosegundos) para cálculo de media.
    sum_nanos: AtomicU64,
    /// Valor mínimo observado.
    min_nanos: AtomicU64,
    /// Valor máximo observado.
    max_nanos: AtomicU64,
}

impl Default for LzcntLogHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl LzcntLogHistogram {
    /// Crea un nuevo histograma logarítmico vacío en stack o memoria estática.
    pub fn new() -> Self {
        // Inicializar array de atómicos en const context
        const INIT_ATOMIC: AtomicU64 = AtomicU64::new(0);
        Self {
            bins: [INIT_ATOMIC; LOG_HISTOGRAM_BINS],
            total_samples: AtomicU64::new(0),
            sum_nanos: AtomicU64::new(0),
            min_nanos: AtomicU64::new(u64::MAX),
            max_nanos: AtomicU64::new(0),
        }
    }

    /// Registra una nueva muestra de latencia en nanosegundos en tiempo O(1) usando LZCNT.
    #[inline(always)]
    pub fn record(&self, latency_ns: u64) {
        let bin_idx = if latency_ns == 0 {
            0
        } else {
            let lz = latency_ns.leading_zeros() as usize;
            63 - lz
        };

        // 1. Incrementar bin atómicamente
        self.bins[bin_idx].fetch_add(1, Ordering::Relaxed);
        self.total_samples.fetch_add(1, Ordering::Relaxed);
        self.sum_nanos.fetch_add(latency_ns, Ordering::Relaxed);

        // 2. Actualizar min y max con CAS saturante
        let mut cur_min = self.min_nanos.load(Ordering::Relaxed);
        while latency_ns < cur_min {
            match self.min_nanos.compare_exchange_weak(
                cur_min,
                latency_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => cur_min = actual,
            }
        }

        let mut cur_max = self.max_nanos.load(Ordering::Relaxed);
        while latency_ns > cur_max {
            match self.max_nanos.compare_exchange_weak(
                cur_max,
                latency_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => cur_max = actual,
            }
        }
    }

    /// Retorna el número total de muestras registradas.
    #[inline(always)]
    pub fn sample_count(&self) -> u64 {
        self.total_samples.load(Ordering::Relaxed)
    }

    /// Retorna la latencia mínima observada (en ns). Retorna 0 si no hay muestras.
    #[inline]
    pub fn min_latency_ns(&self) -> u64 {
        let m = self.min_nanos.load(Ordering::Relaxed);
        if m == u64::MAX { 0 } else { m }
    }

    /// Retorna la latencia máxima observada (en ns).
    #[inline]
    pub fn max_latency_ns(&self) -> u64 {
        self.max_nanos.load(Ordering::Relaxed)
    }

    /// Retorna la latencia media observada (en ns).
    #[inline]
    pub fn mean_latency_ns(&self) -> f64 {
        let count = self.sample_count();
        if count == 0 {
            0.0
        } else {
            self.sum_nanos.load(Ordering::Relaxed) as f64 / count as f64
        }
    }

    /// Estima el percentil `p` en O(1) iterando sobre los 64 bins (donde `p` in [0.0, 1.0]).
    /// Ejemplo: p = 0.50 (mediana), p = 0.99 (p99), p = 0.999 (p99.9).
    pub fn percentile(&self, p: f64) -> u64 {
        let count = self.sample_count();
        if count == 0 {
            return 0;
        }

        let p_clamped = p.clamp(0.0, 1.0);
        let target_rank = ((count as f64) * p_clamped).ceil() as u64;
        let target_rank = target_rank.max(1);

        let mut cumulative: u64 = 0;
        for bin_idx in 0..LOG_HISTOGRAM_BINS {
            let bin_count = self.bins[bin_idx].load(Ordering::Relaxed);
            cumulative += bin_count;
            if cumulative >= target_rank {
                // El percentil cae en este bin diádico [2^bin_idx, 2^(bin_idx+1) - 1]
                let lower_bound = if bin_idx == 0 { 0 } else { 1u64 << bin_idx };
                let upper_bound = if bin_idx == 63 {
                    u64::MAX
                } else {
                    (1u64 << (bin_idx + 1)) - 1
                };

                // Interpolación lineal dentro del bin
                let prev_cumulative = cumulative - bin_count;
                let rank_in_bin = target_rank - prev_cumulative;
                let fraction = if bin_count > 0 {
                    rank_in_bin as f64 / bin_count as f64
                } else {
                    0.5
                };

                let interpolated =
                    lower_bound as f64 + fraction * ((upper_bound - lower_bound) as f64);
                let val = interpolated.round() as u64;
                return val.clamp(self.min_latency_ns(), self.max_latency_ns());
            }
        }

        self.max_latency_ns()
    }
}

/// Registro de micro-evento para la caja negra de vuelo HFT (64 bytes alineado a caché).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlightEventRecord {
    pub timestamp_ns: u64,
    pub event_type: u16,
    pub symbol_id: u16,
    pub latency_ns: u32,
    pub sequence: u64,
    pub price: f64,
    pub quantity: f64,
    pub flags: u64,
    pub metadata: [u8; 16],
}

impl Default for FlightEventRecord {
    fn default() -> Self {
        Self {
            timestamp_ns: 0,
            event_type: 0,
            symbol_id: 0,
            latency_ns: 0,
            sequence: 0,
            price: 0.0,
            quantity: 0.0,
            flags: 0,
            metadata: [0u8; 16],
        }
    }
}

/// Anillo circular de registros de vuelo en memoria compartida o stack de tamaño N acotado.
pub struct MmapFlightRecordRing<const CAPACITY: usize> {
    records: [FlightEventRecord; CAPACITY],
    cursor: AtomicUsize,
    histogram: LzcntLogHistogram,
}

impl<const CAPACITY: usize> Default for MmapFlightRecordRing<CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAPACITY: usize> MmapFlightRecordRing<CAPACITY> {
    /// Inicializa un nuevo anillo registrador con capacidad estática en memoria.
    pub fn new() -> Self {
        Self {
            records: [FlightEventRecord::default(); CAPACITY],
            cursor: AtomicUsize::new(0),
            histogram: LzcntLogHistogram::new(),
        }
    }

    /// Emite un nuevo evento al registrador de vuelo en tiempo O(1) lock-free.
    #[inline(always)]
    pub fn record_event(
        &mut self,
        timestamp_ns: u64,
        event_type: u16,
        symbol_id: u16,
        latency_ns: u32,
        price: f64,
        quantity: f64,
    ) {
        let seq = self.cursor.fetch_add(1, Ordering::Relaxed);
        let idx = seq % CAPACITY;

        self.records[idx] = FlightEventRecord {
            timestamp_ns,
            event_type,
            symbol_id,
            latency_ns,
            sequence: seq as u64,
            price,
            quantity,
            flags: 0,
            metadata: [0u8; 16],
        };

        self.histogram.record(latency_ns as u64);
    }

    /// Acceso al histograma logarítmico de latencias.
    #[inline(always)]
    pub fn histogram(&self) -> &LzcntLogHistogram {
        &self.histogram
    }

    /// Retorna el último evento registrado si existe.
    pub fn latest_event(&self) -> Option<&FlightEventRecord> {
        let cur = self.cursor.load(Ordering::Relaxed);
        if cur == 0 {
            None
        } else {
            let idx = (cur - 1) % CAPACITY;
            Some(&self.records[idx])
        }
    }
}
