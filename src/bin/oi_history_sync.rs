//! OI HISTORY SYNC — histórico de open interest para el predictor {SYM}_OI.
//!
//! # Por qué existe
//! P-3c: el OI dejó de ser sólo ingesta (poller vivo por símbolo) para ser
//! PREDICCIÓN. Entrenar `--label oi` exige una serie histórica que el
//! endpoint sólo conserva ~30 días — este bin la congela a CSV antes de
//! que la ventana corra, y se re-ejecuta periódicamente para adelantarla.
//!
//! # Fuente
//! `/futures/data/openInterestHist?symbol=X&period=1h&limit=500` (MAINNET:
//! el histórico de testnet es vacío/incierto; el OI es la misma realidad
//! del contrato, el transporte es lo de menos). Paginación hacia atrás por
//! `startTime` hasta ~30 días o techo de 1000 filas.
//!
//! # Salida
//! `data/oihist/{SYM}.csv` con cabecera `ts,oi` (ts en ms) — el contrato
//! que `train_forest --label oi` consume con join as-of estricto.

use std::process::Command;

const MAINNET: &str = "https://fapi.binance.com";
const PERIOD_MS: u64 = 3_600_000;
const MAX_ROWS: usize = 1_000;
const UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

/// GET vía curl (stdout del proceso) — mismo transporte verificado de
/// macro_history_sync (el reqwest raíz muere contra los CDNs de Binance
/// desde esta red).
fn http_get(url: &str) -> Option<String> {
    let out = Command::new("curl")
        .args(["-s", "-m", "20", "-A", UA, url])
        .output()
        .ok()?;
    if !out.status.success() || out.stdout.is_empty() {
        return None;
    }
    String::from_utf8(out.stdout).ok()
}

/// Descarga el histórico de un símbolo (hacia atrás desde `end_ms`).
fn fetch_oi_history(symbol: &str, end_ms: u64) -> Option<Vec<(u64, f64)>> {
    let mut all: Vec<(u64, f64)> = Vec::new();
    let mut cursor_end = end_ms;
    let floor = end_ms.saturating_sub(31 * 24 * PERIOD_MS);
    loop {
        let url = format!(
            "{}/futures/data/openInterestHist?symbol={}&period=1h&limit=500&endTime={}",
            MAINNET, symbol, cursor_end
        );
        let body = http_get(&url)?;
        let json: serde_json::Value = serde_json::from_str(&body).ok()?;
        let arr = json.as_array()?;
        if arr.is_empty() {
            break;
        }
        let mut got = 0usize;
        for item in arr {
            let ts = item.get("timestamp").and_then(|v| v.as_i64()).unwrap_or(0) as u64;
            let oi = item
                .get("sumOpenInterest")
                .and_then(|v| v.as_str())
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0);
            if ts > 0 && oi.is_finite() && oi > 0.0 {
                all.push((ts, oi));
                got += 1;
            }
        }
        if got == 0 {
            break;
        }
        let oldest = all.last().map(|(ts, _)| *ts).unwrap_or(0);
        if oldest <= floor || all.len() >= MAX_ROWS {
            break;
        }
        cursor_end = oldest.saturating_sub(1);
    }
    all.sort_unstable_by_key(|(ts, _)| *ts);
    all.dedup_by(|a, b| a.0 == b.0);
    if all.len() >= 24 {
        Some(all)
    } else {
        None
    }
}

/// Escribe la serie FUSIONÁNDOla con la existente: el endpoint sólo conserva
/// ~30 días, así que cada ejecución añade las filas nuevas al archivo — el
/// histórico ACUMULA con los meses de re-ejecución (sin esto, la ventana
/// de entrenamiento del predictor {SYM}_OI jamás crecería).
fn write_series(symbol: &str, rows: &[(u64, f64)]) -> bool {
    let path = format!("data/oihist/{}.csv", symbol);
    // Cargar existente y fusionar por ts.
    let mut merged: Vec<(u64, f64)> = Vec::with_capacity(rows.len() + 1024);
    if let Ok(prev) = std::fs::read_to_string(&path) {
        for ln in prev.lines().skip(1) {
            let mut it = ln.split(',');
            let (Some(Ok(ts)), Some(Ok(oi))) = (
                it.next().map(|s| s.trim().parse::<u64>()),
                it.next().map(|s| s.trim().parse::<f64>()),
            ) else {
                continue;
            };
            if ts > 0 && oi.is_finite() && oi > 0.0 {
                merged.push((ts, oi));
            }
        }
    }
    merged.extend_from_slice(rows);
    merged.sort_unstable_by_key(|(ts, _)| *ts);
    merged.dedup_by(|a, b| a.0 == b.0);
    let mut out = String::with_capacity(merged.len() * 24);
    out.push_str("ts,oi\n");
    for (ts, oi) in &merged {
        out.push_str(&format!("{},{}\n", ts, oi));
    }
    std::fs::write(&path, out).is_ok()
}

fn main() {
    println!("============================================================");
    println!("👁️  OI HISTORY SYNC — /futures/data/openInterestHist (1h, ~30d)");
    println!("============================================================");
    std::fs::create_dir_all("data/oihist").unwrap();

    let args: Vec<String> = std::env::args().skip(1).collect();
    // Símbolos por argumento; si no, roster de modelos ∩ datos de ticks
    // (sólo se entrena lo que tiene AMBAS series).
    let symbols: Vec<String> = if args.is_empty() {
        let mut roster = std::collections::HashSet::new();
        if let Ok(entries) = std::fs::read_dir("models") {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if name.ends_with("_MOTOR.json") {
                    roster.insert(name.trim_end_matches("_MOTOR.json").to_string());
                }
            }
        }
        let mut with_data: Vec<String> = Vec::new();
        if let Ok(entries) = std::fs::read_dir("data") {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if let Some(sym) = name
                    .strip_suffix("_2026-08_REAL.bin")
                    .or_else(|| name.strip_suffix("_AUG_REAL.bin"))
                {
                    if roster.contains(sym) {
                        with_data.push(sym.to_string());
                    }
                }
            }
        }
        with_data.sort();
        with_data
    } else {
        args.iter().map(|s| s.to_uppercase()).collect()
    };

    if symbols.is_empty() {
        eprintln!("❌ sin símbolos: pasa símbolos como argumentos o asegura models/*_MOTOR.json + data/*_REAL.bin");
        std::process::exit(1);
    }

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let mut failures = 0usize;
    for sym in &symbols {
        println!("🚀 {} → data/oihist/{}.csv", sym, sym);
        match fetch_oi_history(sym, now_ms) {
            Some(rows) => {
                let span_days = (rows.last().unwrap().0 - rows.first().unwrap().0) as f64
                    / (24.0 * PERIOD_MS as f64);
                if write_series(sym, &rows) {
                    println!("   ✅ {} filas · span {:.1} días", rows.len(), span_days);
                } else {
                    failures += 1;
                }
            }
            None => {
                println!("   ❌ sin datos (<24 filas)");
                failures += 1;
            }
        }
    }
    if failures > 0 {
        println!("⚠️ {} símbolos sin serie", failures);
    }
}
