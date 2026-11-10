//! Cinta real compartida por las certificaciones locales de RB-V1a.
//!
//! Formato: registros de 40 bytes (ts u64, bid, ask, bid_qty, ask_qty f64,
//! little-endian), con cabecera opcional `TGMTICK1`. Como `backtest_windows`,
//! se toman los últimos n registros y se descartan los inválidos; sólo se lee
//! esa cola del archivo (las cintas mensuales pesan más de 1 GB). La ruta llega
//! por `RB_V1A_CINTA` y n por `RB_V1A_TICKS` (200 000 si no se indica).

use backtest_engine::booktick_replay::{ReplayConfig, ReplayTick};
use quantum_arena::genome::SuperGenotype;
use std::io::{Read, Seek, SeekFrom};

const REGISTRO: u64 = 40;

fn f64_en(b: &[u8], desde: usize) -> Option<f64> {
    let bytes: [u8; 8] = b.get(desde..desde + 8)?.try_into().ok()?;
    Some(f64::from_le_bytes(bytes))
}

/// Los ticks válidos de los últimos `RB_V1A_TICKS` registros de `RB_V1A_CINTA`.
pub fn cargar_desde_entorno() -> Vec<ReplayTick> {
    let Ok(ruta) = std::env::var("RB_V1A_CINTA") else {
        panic!("define RB_V1A_CINTA con la ruta de una cinta *_ticks*.bin");
    };
    let max = std::env::var("RB_V1A_TICKS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(200_000);
    let mut archivo = match std::fs::File::open(&ruta) {
        Ok(a) => a,
        Err(e) => panic!("no se pudo abrir la cinta {ruta}: {e}"),
    };
    let largo = match archivo.metadata() {
        Ok(m) => m.len(),
        Err(e) => panic!("sin metadatos de la cinta {ruta}: {e}"),
    };
    let mut magia = [0u8; 8];
    let cabecera = if largo >= 8 && archivo.read_exact(&mut magia).is_ok() && magia == *b"TGMTICK1"
    {
        8
    } else {
        0
    };
    let total = (largo - cabecera) / REGISTRO;
    let n = total.min(max);
    if let Err(e) = archivo.seek(SeekFrom::Start(cabecera + (total - n) * REGISTRO)) {
        panic!("no se pudo posicionar la cinta {ruta}: {e}");
    }
    let mut cuerpo = vec![0u8; (n * REGISTRO) as usize];
    if let Err(e) = archivo.read_exact(&mut cuerpo) {
        panic!("no se pudo leer la cola de la cinta {ruta}: {e}");
    }
    let ticks: Vec<ReplayTick> = cuerpo
        .chunks_exact(REGISTRO as usize)
        .filter_map(|r| {
            let ts: [u8; 8] = r.get(0..8)?.try_into().ok()?;
            Some(ReplayTick {
                ts_ms: u64::from_le_bytes(ts),
                bid: f64_en(r, 8)?,
                ask: f64_en(r, 16)?,
                bid_qty: f64_en(r, 24)?,
                ask_qty: f64_en(r, 32)?,
            })
        })
        .filter(|t| t.bid > 0.0 && t.ask > 0.0 && t.bid <= t.ask && t.ts_ms > 0)
        .collect();
    assert!(
        ticks.len() > 1_000,
        "cinta demasiado corta: {} ticks",
        ticks.len()
    );
    ticks
}

/// Los parámetros de `backtest_windows`: capital de comparabilidad,
/// calentamiento y desplazamiento de ATR.
pub fn config_de_referencia() -> ReplayConfig {
    ReplayConfig {
        initial_capital: 1000.0,
        warmup_ticks: 500,
        trade_only: false,
        shift_atr_frac: 0.10,
    }
}

/// Genoma de referencia (no el activo del almacén): la certificación no
/// depende del entorno de quien la corre.
pub fn genoma_de_referencia() -> SuperGenotype {
    SuperGenotype::new_baseline(0.0002, 0.0005)
}
