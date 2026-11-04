//! Streaming Visual Desacoplado vía Memory-Mapped Files (mmap lock-free) (FASE 5)
//!
//! Lector de ultra-baja latencia y cero contención que mapea en memoria
//! los archivos de telemetría de trading (`telemetry_mmap.bin` y `mmap_bus.bin`)
//! sin interferir ni bloquear el hot path de nanosegundos del motor.
//!
//! # QUÉ:
//! Lector de memoria compartida en modo sólo lectura (SHM read-only) con doble
//! lectura idéntica para consumir snapshots del estado unificado en streaming.
//!
//! # POR QUÉ:
//! El motor de trading no debe hacer operaciones de red lentas, serialización JSON
//! ni bloqueos de concurrencia dentro de su bucle de eventos. Mmap permite desacoplar
//! completamente la visualización del cálculo financiero.
//!
//! # PARA QUÉ:
//! 1. Alimentar la aplicación interactiva de terminal `trader_viz`.
//! 2. Servir la suite visual desacoplada en tiempo real.
//! 3. Rechazar snapshots cambiantes o inválidos sin inventar valores.
//!
//! # CÓMO:
//! Mapeo vía `memmap2::MmapOptions`, dos lecturas volátiles con comparación bit a
//! bit, campos requeridos finitos y `tensor_drift=NaN` como ausencia opcional.
//!
//! # DÓNDE:
//! En `crates/telemetry-server/src/mmap_viz_stream.rs`.
//!
//! # QUIÉN:
//! Ingeniero de Silicio HFT & DevOps / SRE (Fase 5).

use crate::causal_attribution::CausalMetaAnalyticsReport;
use crate::hyperdimensional_graph::HyperdimensionalGraphSnapshot;
use crate::mirofish_lattice::{MirofishLattice, TailRidge3D};
use memmap2::{Mmap, MmapMut, MmapOptions};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

/// Snapshot de telemetría de memoria compartida (SHM).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub tick_counter: u64,
    pub unified_capital: f64,
    pub pnl_realized: f64,
    pub global_leverage: f64,
    pub ai_ml_prob: f64,
    pub hurst_exponent: f64,
    pub memory_used_mb: f64,
    pub global_roi: f64,
    pub win_rate: f64,
    pub tensor_drift: f64,
}

/// Paquete completo de visualización desacoplada para la Suite Visual.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnifiedVizPacket {
    pub timestamp_ns: u64,
    pub tick_counter: u64,
    pub unified_capital: f64,
    pub pnl_realized: f64,
    pub global_leverage: f64,
    pub global_roi: f64,
    pub win_rate: f64,
    pub memory_used_mb: f64,
    pub lattice: Option<MirofishLattice>,
    pub tail_ridge: Option<TailRidge3D>,
    pub hypergraph: Option<HyperdimensionalGraphSnapshot>,
    pub causal_report: Option<CausalMetaAnalyticsReport>,
}

/// Lector desacoplado de telemetría por memoria compartida (mmap).
pub struct MmapVizStreamReader {
    mmap_path: PathBuf,
    mmap_handle: Option<Mmap>,
}

impl MmapVizStreamReader {
    /// Inicializa el lector sobre la ruta del snapshot de telemetría.
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let p = path.as_ref().to_path_buf();
        let mmap_handle = Self::try_open(&p);
        Self {
            mmap_path: p,
            mmap_handle,
        }
    }

    fn try_open(path: &Path) -> Option<Mmap> {
        let mut opts = OpenOptions::new();
        opts.read(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            opts.share_mode(3); // FILE_SHARE_READ | FILE_SHARE_WRITE
        }

        let file = opts.open(path).ok()?;
        let meta = file.metadata().ok()?;
        let expected_size = std::mem::size_of::<TelemetrySnapshot>() as u64;

        if meta.len() < expected_size {
            return None;
        }

        unsafe { MmapOptions::new().map(&file).ok() }
    }

    /// Lee dos veces el mismo snapshot y rechaza carreras o campos inválidos.
    pub fn read_snapshot(&mut self) -> Option<TelemetrySnapshot> {
        if self.mmap_handle.is_none() {
            self.mmap_handle = Self::try_open(&self.mmap_path);
        }

        let mmap = self.mmap_handle.as_ref()?;
        let expected_size = std::mem::size_of::<TelemetrySnapshot>();

        if mmap.len() < expected_size {
            return None;
        }

        let first: TelemetrySnapshot =
            unsafe { std::ptr::read_volatile(mmap.as_ptr() as *const TelemetrySnapshot) };
        let second: TelemetrySnapshot =
            unsafe { std::ptr::read_volatile(mmap.as_ptr() as *const TelemetrySnapshot) };

        if snapshot_bits(&first) != snapshot_bits(&second)
            || !first.unified_capital.is_finite()
            || first.unified_capital <= 0.0
            || !first.global_leverage.is_finite()
            || first.global_leverage <= 0.0
            || !first.pnl_realized.is_finite()
            || !first.ai_ml_prob.is_finite()
            || !first.hurst_exponent.is_finite()
            || !first.memory_used_mb.is_finite()
            || !first.global_roi.is_finite()
            || !first.win_rate.is_finite()
        {
            return None;
        }

        Some(first)
    }

    /// Ensambla un paquete sólo con evidencia disponible; no fabrica ausencias.
    pub fn build_unified_packet(
        &mut self,
        symbols: &[String],
        vn_entropies: &[f64],
        tsallis_entropies: &[f64],
        gauge_actions: &[f64],
        void_risks: &[f64],
        betti_1_counts: &[f64],
    ) -> Option<UnifiedVizPacket> {
        let snap = self.read_snapshot()?;
        let timestamp_ns = u64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?
                .as_nanos(),
        )
        .ok()?;
        let spot = snap.unified_capital;
        let lattice = snap.tensor_drift.is_finite().then(|| {
            MirofishLattice::compute(
                spot,
                0.45,
                snap.ai_ml_prob - 0.5,
                snap.tensor_drift,
                0.10,
                8,
                1.0,
            )
        });
        let tail_ridge = Some(TailRidge3D::generate(spot, 0.45, snap.hurst_exponent, 0.05));
        let hypergraph = Some(HyperdimensionalGraphSnapshot::build_from_state(
            symbols,
            vn_entropies,
            tsallis_entropies,
            gauge_actions,
            void_risks,
            betti_1_counts,
        )?);

        Some(UnifiedVizPacket {
            timestamp_ns,
            tick_counter: snap.tick_counter,
            unified_capital: snap.unified_capital,
            pnl_realized: snap.pnl_realized,
            global_leverage: snap.global_leverage,
            global_roi: snap.global_roi,
            win_rate: snap.win_rate,
            memory_used_mb: snap.memory_used_mb,
            lattice,
            tail_ridge,
            hypergraph,
            causal_report: None,
        })
    }
}

fn snapshot_bits(snapshot: &TelemetrySnapshot) -> [u64; 10] {
    [
        snapshot.tick_counter,
        snapshot.unified_capital.to_bits(),
        snapshot.pnl_realized.to_bits(),
        snapshot.global_leverage.to_bits(),
        snapshot.ai_ml_prob.to_bits(),
        snapshot.hurst_exponent.to_bits(),
        snapshot.memory_used_mb.to_bits(),
        snapshot.global_roi.to_bits(),
        snapshot.win_rate.to_bits(),
        snapshot.tensor_drift.to_bits(),
    ]
}

/// Escritor atómico de telemetría por memoria compartida (mmap lock-free).
///
/// Permite al motor de ejecución escribir snapshots de telemetría directamente
/// en el archivo mapeado sin locks y con visibilidad instantánea para lectores.
pub struct MmapVizStreamWriter {
    mmap_handle: Option<MmapMut>,
}

impl MmapVizStreamWriter {
    /// Inicializa o crea el archivo mapeado en memoria para escritura zero-copy.
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let p = path.as_ref().to_path_buf();
        let expected_size = std::mem::size_of::<TelemetrySnapshot>();
        let mut opts = OpenOptions::new();
        opts.read(true).write(true).create(true).truncate(false);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            opts.share_mode(3); // FILE_SHARE_READ | FILE_SHARE_WRITE
        }
        let file = opts.open(&p)?;
        if file.metadata()?.len() < expected_size as u64 {
            file.set_len(expected_size as u64)?;
        }
        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        Ok(Self {
            mmap_handle: Some(mmap),
        })
    }

    /// Escribe un snapshot y propaga tanto handle ausente como fallo de flush.
    pub fn write_snapshot(&mut self, snapshot: &TelemetrySnapshot) -> std::io::Result<()> {
        let mmap = self.mmap_handle.as_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "mmap mut handle not available",
            )
        })?;
        let bytes = unsafe {
            std::slice::from_raw_parts(
                snapshot as *const TelemetrySnapshot as *const u8,
                std::mem::size_of::<TelemetrySnapshot>(),
            )
        };
        mmap[..bytes.len()].copy_from_slice(bytes);
        mmap.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmap_viz_stream_packet_assembly() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_mmap_viz_stream_packet.bin");
        let snapshot = TelemetrySnapshot {
            tick_counter: 42,
            unified_capital: 150.0,
            pnl_realized: 25.0,
            global_leverage: 1.0,
            ai_ml_prob: 0.75,
            hurst_exponent: 0.60,
            memory_used_mb: 120.0,
            global_roi: 50.0,
            win_rate: 0.80,
            tensor_drift: 0.2,
        };
        let mut writer = MmapVizStreamWriter::new(&path).expect("writer");
        writer.write_snapshot(&snapshot).expect("write");
        let mut reader = MmapVizStreamReader::new(&path);
        let symbols = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()];
        let vn = vec![0.40, 0.60];
        let tsallis = vec![0.45, 0.65];
        let gauge = vec![0.5, 1.2];
        let void = vec![0.05, 0.15];
        let betti_1 = vec![0.0, 1.0];

        let packet = reader
            .build_unified_packet(&symbols, &vn, &tsallis, &gauge, &void, &betti_1)
            .expect("snapshot válido");

        assert_eq!(packet.hypergraph.as_ref().expect("graph").nodes.len(), 2);
        assert_eq!(packet.tail_ridge.as_ref().expect("ridge").points.len(), 32);
        assert!(packet.causal_report.is_none());
        assert!(packet.lattice.as_ref().expect("lattice").levels.len() >= 8);

        drop(reader);
        drop(writer);
        std::fs::remove_file(path).expect("cleanup");
    }
}
