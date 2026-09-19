//! High-performance memory-mapped binary tick replayer.
//!
//! R2.4 — Formato versionado: los archivos nuevos comienzan con un header
//! `magic (8 bytes "TGMTICK1")`. Un archivo con otro magic, otra versión o un
//! tamaño no múltiplo del registro se RECHAZA con error explícito — antes un
//! lector compilado contra otra versión de `BinTick` deserializaba basura en
//! silencio. Los archivos legacy sin header (todo el histórico actual) se
//! aceptan con la marca de compatibilidad y una única advertencia por archivo.

use quantum_arena::TickEvent;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BinTick {
    pub timestamp: u64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_qty: f64,
    pub ask_qty: f64,
}

/// Magic + versión del formato en un solo literal de 8 bytes.
///
/// D-721 (DÉCIMA OLA · auditoría integral): EL MAGIC DICE EL ORIGEN, NO SÓLO LA
/// VERSIÓN. La corrección de D-691 hizo que el forense distinguiera «datos
/// reales» de «velas expandidas» POR LA CABECERA, pero `parquet_to_bin` —el
/// generador sintético cuyo no-causalidad motivó D-691— escribía exactamente el
/// mismo literal que el tape de aggTrades reales. Hoy acierta por accidente
/// (los ficheros sintéticos en disco son legado sin cabecera); en cuanto se
/// regeneren —y el propio lector se lo pide al operador— el forense declararía
/// «versionado» un fichero sintético y toda la advertencia de D-691 se apagaría.
/// Ahora hay dos magics del mismo tamaño y los lectores aceptan ambos.
pub const TICK_MAGIC: &[u8; 8] = b"TGMTICK1";

/// Tape de aggTrades REALES del exchange (alias explícito de `TICK_MAGIC`, que
/// es el que ya escribe `binance_vision_sync`).
pub const TICK_MAGIC_REAL: &[u8; 8] = TICK_MAGIC;

/// Velas expandidas a subticks por `parquet_to_bin`: NO son ticks (D-691).
pub const TICK_MAGIC_SYNTH: &[u8; 8] = b"TGMSYNT1";

/// Origen declarado por la cabecera de un fichero de ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickOrigin {
    /// aggTrades reales del exchange.
    Real,
    /// Velas expandidas: microestructura fabricada (D-691).
    Sintetico,
    /// Legado sin cabecera: origen desconocido, se trata como sintético.
    LegadoSinCabecera,
}

impl TickOrigin {
    /// Lee el origen de los primeros bytes de un fichero de ticks.
    pub fn from_header(bytes: &[u8]) -> (TickOrigin, usize) {
        if bytes.len() >= 8 && &bytes[..8] == TICK_MAGIC_REAL {
            (TickOrigin::Real, 8)
        } else if bytes.len() >= 8 && &bytes[..8] == TICK_MAGIC_SYNTH {
            (TickOrigin::Sintetico, 8)
        } else {
            (TickOrigin::LegadoSinCabecera, 0)
        }
    }

    pub fn descripcion(self) -> &'static str {
        match self {
            TickOrigin::Real => "aggTrades REALES del exchange (TGMTRAW/TGMTICK1)",
            TickOrigin::Sintetico => "VELAS EXPANDIDAS por parquet_to_bin — microestructura fabricada (D-691)",
            TickOrigin::LegadoSinCabecera => "legado SIN cabecera: origen desconocido, se asume sintético (D-691)",
        }
    }

    /// ¿Sirve para dictar un veredicto sobre microestructura?
    pub fn es_real(self) -> bool {
        matches!(self, TickOrigin::Real)
    }
}

fn validate_alignment(len: usize, path: &Path) -> std::io::Result<()> {
    let tick_size = std::mem::size_of::<BinTick>();
    if len % tick_size != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "{}: tamaño {} no es múltiplo del registro de {} bytes — archivo corrupto o de otro formato",
                path.display(),
                len,
                tick_size
            ),
        ));
    }
    Ok(())
}

pub fn load_binary_ticks(path: &Path, coin_id: usize) -> std::io::Result<Vec<TickEvent>> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let tick_size = std::mem::size_of::<BinTick>();

    let (data_ptr, data_len, legacy): (usize, usize, bool) =
        if mmap.len() >= 8 && TickOrigin::from_header(&mmap[..8]).1 == 8 {
            // Formato versionado: header de 8 bytes + registros alineados.
            validate_alignment(mmap.len() - 8, path)?;
            (8, mmap.len() - 8, false)
        } else {
            // Legacy sin header (histórico pre-R2.4): compatible, advertido.
            validate_alignment(mmap.len(), path)?;
            (0, mmap.len(), true)
        };

    if legacy {
        eprintln!(
            "⚠️ [TICK-REPLAYER] {} en formato LEGACY sin header — regenerar con parquet_to_bin para versionado (R2.4).",
            path.display()
        );
    }

    let num_ticks = data_len / tick_size;
    // Safety: el slice está alineado a 8 (header de 8 bytes + registros
    // repr(C) de 40 bytes sobre un mmap alineado a página).
    let slice = unsafe {
        std::slice::from_raw_parts(mmap.as_ptr().add(data_ptr) as *const BinTick, num_ticks)
    };

    let mut ticks = Vec::with_capacity(num_ticks);
    for t in slice {
        ticks.push(TickEvent {
            coin_id,
            timestamp: t.timestamp,
            bid_price: if t.bid_price.is_finite() && t.bid_price > 0.0 {
                t.bid_price
            } else {
                0.0
            },
            ask_price: if t.ask_price.is_finite() && t.ask_price > 0.0 {
                t.ask_price
            } else {
                0.0
            },
            bid_qty: if t.bid_qty.is_finite() && t.bid_qty >= 0.0 {
                t.bid_qty
            } else {
                0.0
            },
            ask_qty: if t.ask_qty.is_finite() && t.ask_qty >= 0.0 {
                t.ask_qty
            } else {
                0.0
            },
        });
    }
    Ok(ticks)
}

pub fn load_multi_coin_binary_ticks(files: &[(&Path, usize)]) -> std::io::Result<Vec<TickEvent>> {
    let mut all_ticks = Vec::new();
    for (path, cid) in files {
        // R2.4: un archivo ilegible ABORTA la carga multi-moneda con su error
        // — antes `if let Ok` lo silenciaba y el símbolo desaparecía del
        // backtest sin rastro.
        all_ticks.append(&mut load_binary_ticks(path, *cid)?);
    }
    all_ticks.sort_by_key(|t| t.timestamp);
    Ok(all_ticks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_file(path: &Path, bytes: &[u8]) {
        std::fs::write(path, bytes).expect("write temp");
    }

    #[test]
    fn test_r24_versioned_roundtrip_and_legacy_compat() {
        let dir = std::env::temp_dir();
        let tick = BinTick {
            timestamp: 1_700_000_000_000,
            bid_price: 100.0,
            ask_price: 100.1,
            bid_qty: 1.0,
            ask_qty: 2.0,
        };
        let recs = unsafe { std::slice::from_raw_parts(&tick as *const BinTick as *const u8, 40) };

        // Formato versionado: magic + registro.
        let v_path = dir.join("tgm_tick_v1_test.bin");
        write_file(&v_path, TICK_MAGIC);
        write_file(&v_path, recs); // append via segunda escritura no vale: reescribir
        let mut full = TICK_MAGIC.to_vec();
        full.extend_from_slice(recs);
        write_file(&v_path, &full);
        let ticks = load_binary_ticks(&v_path, 3).expect("versioned load");
        assert_eq!(ticks.len(), 1);
        assert_eq!(ticks[0].coin_id, 3);
        assert_eq!(ticks[0].timestamp, tick.timestamp);
        assert!((ticks[0].ask_price - 100.1).abs() < 1e-9);

        // Legacy sin header: mismo registro, sin magic.
        let l_path = dir.join("tgm_tick_legacy_test.bin");
        write_file(&l_path, recs);
        let ticks = load_binary_ticks(&l_path, 0).expect("legacy load");
        assert_eq!(ticks.len(), 1);

        // Archivo corrupto (tamaño no múltiplo): rechazo ruidoso.
        let c_path = dir.join("tgm_tick_corrupt_test.bin");
        write_file(&c_path, &full[..full.len() - 7]);
        assert!(load_binary_ticks(&c_path, 0).is_err());

        let _ = std::fs::remove_file(&v_path);
        let _ = std::fs::remove_file(&l_path);
        let _ = std::fs::remove_file(&c_path);
    }
}
