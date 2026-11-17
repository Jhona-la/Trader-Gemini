//! Cinta real compartida por las certificaciones locales de RB-V1a.
//!
//! Formato: registros de 40 bytes (ts u64, bid, ask, bid_qty, ask_qty f64,
//! little-endian), con cabecera opcional `TGMTICK1`. Como `backtest_windows`,
//! se toman los últimos n registros y se descartan los inválidos; sólo se lee
//! esa cola del archivo (las cintas mensuales pesan más de 1 GB). La ruta llega
//! por `RB_V1A_CINTA` y n por `RB_V1A_TICKS` (200 000 si no se indica).

use backtest_engine::booktick_replay::{ReplayConfig, ReplayTick};
use quantum_arena::genome::SuperGenotype;

const REGISTRO: usize = 40;

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
    // TGB-E: cinta lógica (`.bin` mapeado, o `.bin.tgb` verificado entero si el `.bin` no existe). En `.bin`
    // sólo se tocan las páginas de la cola; en `.tgb` el SHA del original cubre la cinta entera.
    let tape = match tick_tape::logica::open_logical_tape(
        std::path::Path::new(&ruta),
        tick_tape::tgb::Presupuesto::sin_tope(),
    ) {
        Ok(t) => t,
        Err(e) => panic!("no se pudo abrir la cinta {ruta}: {e}"),
    };
    let bytes = tape.as_bytes();
    let largo = bytes.len() as u64;
    let cabecera = if bytes.len() >= 8 && bytes[..8] == *b"TGMTICK1" { 8 } else { 0 };
    let registro = REGISTRO as u64;
    let total = (largo - cabecera) / registro;
    let n = total.min(max);
    let desde = (cabecera + (total - n) * registro) as usize;
    let cuerpo = bytes[desde..desde + (n * registro) as usize].to_vec();
    // Se leyeron n registros completos: la cola sobrante está vacía.
    let (registros, _cola) = cuerpo.as_chunks::<REGISTRO>();
    let ticks: Vec<ReplayTick> = registros
        .iter()
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

/// Registra el símbolo de la cinta (el prefijo del nombre del archivo, como
/// `backtest_windows`) con su spec de `get_official_binance_spec` y como
/// universo. Sin símbolo, el núcleo no reconoce las comisiones de la moneda 0
/// (`sync_reality_fees_for_coin`) y bloquea toda entrada: el replay no
/// produciría ni una intención.
pub fn registrar_simbolo_de_la_cinta() -> String {
    let Ok(ruta) = std::env::var("RB_V1A_CINTA") else {
        panic!("define RB_V1A_CINTA con la ruta de una cinta *_ticks*.bin");
    };
    let Some(nombre) = std::path::Path::new(&ruta)
        .file_name()
        .and_then(|n| n.to_str())
    else {
        panic!("la ruta {ruta} no tiene nombre de archivo");
    };
    let Some(simbolo) = nombre.split('_').next().map(str::to_uppercase) else {
        panic!("no se deduce el símbolo de {nombre}");
    };
    quantum_arena::symbols::update_dynamic_universe(vec![simbolo.clone()]);
    quantum_arena::symbol_registry::update_registry(vec![
        quantum_arena::symbol_registry::get_official_binance_spec(&simbolo),
    ]);
    simbolo
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
