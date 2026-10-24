//! B3.4 — HISTORIA MACRO REAL (VIX, SP500, NASDAQ, DXY).
//!
//! Paridad de SERIES con el feed vivo (`omni_multiplexer.rs`):
//! - VIX/SP500/NASDAQ: Yahoo v8 chart (^VIX/^GSPC/^IXIC, cierre diario) —
//!   los MISMOS cierres que VIXCLS/SP500/NASDAQCOM de FRED. El endpoint
//!   v7 download de Yahoo murió y el reqwest raíz (0.11+rustls) recibe
//!   connection-reset del CDN de FRED (fingerprint TLS, verificado
//!   2026-09-15); el v8 chart responde y curl+UA de navegador pasa.
//! - DXY: Yahoo v8 chart DX-Y.NYB (ICE Dollar Index) desde B3.23 — la
//!   paridad exige la MISMA serie en el trainer y en el vivo, y ambos usan
//!   DX-Y.NYB (el DTWEXBGS de FRED quedó bloqueado desde esta red; su
//!   cliente `fred_daily` se retiró en RET-45). Si alguna de las cuatro
//!   series falla, el exit code es 1 — train_forest no entrena sin las
//!   cuatro reales.
//!
//! Uso: macro_history_sync
//! Salida: data/macro/{VIX,SP500,DXY,NASDAQ}.csv — date_ms,value ascendente.
//! Sin fallback sintético: un random-walk etiquetado como VIX contaminaría
//! el bloque macro del vector ML (lección R2.3).

use std::fs::File;
use std::io::Write;
use std::path::Path;

const UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

/// GET vía curl (stdout del proceso). Transporte verificado contra el CDN
/// de Yahoo desde esta red.
fn http_get(url: &str) -> Option<String> {
    let out = std::process::Command::new("curl")
        .args(["-s", "-m", "30", "-A", UA, url])
        .output()
        .ok()?;
    if !out.status.success() || out.stdout.is_empty() {
        return None;
    }
    String::from_utf8(out.stdout).ok()
}

fn write_series(tag: &str, rows: &[(u64, f64)]) -> bool {
    if rows.len() < 500 {
        println!("   ❌ {tag}: historia demasiado corta ({} filas)", rows.len());
        return false;
    }
    let mut rows = rows.to_vec();
    rows.sort_unstable_by_key(|r| r.0);
    rows.dedup_by_key(|r| r.0);
    let path = Path::new("data/macro").join(format!("{tag}.csv"));
    let Ok(mut f) = File::create(&path) else {
        println!("   ❌ {tag}: no pude crear {:?}", path);
        return false;
    };
    let _ = writeln!(f, "date_ms,value");
    for (ms, v) in &rows {
        let _ = writeln!(f, "{ms},{v}");
    }
    println!(
        "   ✅ {tag}: {} días · cobertura {} días",
        rows.len(),
        (rows[rows.len() - 1].0 - rows[0].0) / 86_400_000
    );
    true
}

/// Yahoo v8 chart: pares (ts_ms, close) del rango pedido.
fn yahoo_daily(symbol: &str, range: &str) -> Option<Vec<(u64, f64)>> {
    let url = format!(
        "https://query1.finance.yahoo.com/v8/finance/chart/{}?range={range}&interval=1d",
        symbol.replace('%', "%25")
    );
    let body = http_get(&url)?;
    let v: serde_json::Value = serde_json::from_str(&body).ok()?;
    let result = v.get("chart")?.get("result")?.get(0)?;
    let ts = result.get("timestamp")?.as_array()?;
    let closes = result
        .get("indicators")?
        .get("quote")?
        .get(0)?
        .get("close")?
        .as_array()?;
    let mut rows = Vec::with_capacity(ts.len());
    for (t, c) in ts.iter().zip(closes.iter()) {
        let (Some(sec), Some(close)) = (t.as_u64(), c.as_f64()) else {
            continue;
        };
        if close.is_finite() && close > 0.0 {
            rows.push((sec * 1000, close));
        }
    }
    Some(rows)
}

fn main() {
    println!("============================================================");
    println!("🌍 MACRO HISTORY SYNC — Yahoo v8 (VIX, SP500, NASDAQ y DXY)");
    println!("============================================================");
    if let Err(e) = std::fs::create_dir_all("data/macro") {
        eprintln!("🚨 no se pudo crear data/macro: {e}");
        std::process::exit(1);
    }

    let mut failures = 0usize;

    // Paridad de VALOR con el feed vivo: ^VIX/^GSPC/^IXIC cierran igual
    // que VIXCLS/SP500/NASDAQCOM — misma serie, distinto transporte.
    // B3.23 — DXY también por Yahoo (DX-Y.NYB, ICE Dollar Index): FRED
    // bloquea esta red desde hace días y la directriz es SOLUCIONAR — la
    // paridad exige MISMA SERIE en trainer y vivo, no una serie concreta:
    // ambos lados usan ahora DX-Y.NYB (~99-105) y la dim 46 despierta.
    for (yahoo_sym, tag) in [
        ("^VIX", "VIX"),
        ("^GSPC", "SP500"),
        ("^IXIC", "NASDAQ"),
        ("DX-Y.NYB", "DXY"),
    ] {
        println!("🚀 {yahoo_sym} → data/macro/{tag}.csv");
        match yahoo_daily(yahoo_sym, "10y") {
            Some(rows) => {
                if !write_series(tag, &rows) {
                    failures += 1;
                }
            }
            None => {
                println!("   ❌ {yahoo_sym}: sin datos");
                failures += 1;
            }
        }
    }

    if failures > 0 {
        eprintln!("🚨 {failures} serie(s) sin sincronizar — train_forest no entrenará sin las cuatro reales.");
        std::process::exit(1);
    }
    println!("✅ Historia macro real completa.");
}
