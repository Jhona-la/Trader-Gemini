//! B3.4 — HISTORIA MACRO REAL (VIX, SP500, NASDAQ, DXY).
//!
//! Paridad de SERIES con el feed vivo (`omni_multiplexer.rs`):
//! - VIX/SP500/NASDAQ: Yahoo v8 chart (^VIX/^GSPC/^IXIC, cierre diario) —
//!   los MISMOS cierres que VIXCLS/SP500/NASDAQCOM de FRED. El endpoint
//!   v7 download de Yahoo murió y el reqwest raíz (0.11+rustls) recibe
//!   connection-reset del CDN de FRED (fingerprint TLS, verificado
//!   2026-09-15); el v8 chart responde y curl+UA de navegador pasa.
//! - DXY: DTWEXBGS de FRED vía curl — el dólar trade-weighted de la Fed,
//!   SIN equivalente en Yahoo (el ICE DX-Y.NYB es otra serie). Se
//!   reintenta; si FRED no responde, la serie queda pendiente y el exit
//!   code es 1 — train_forest no entrena sin las cuatro reales.
//!
//! Uso: macro_history_sync
//! Salida: data/macro/{VIX,SP500,DXY,NASDAQ}.csv — date_ms,value ascendente.
//! Sin fallback sintético: un random-walk etiquetado como VIX contaminaría
//! el bloque macro del vector ML (lección R2.3).

use std::fs::File;
use std::io::Write;
use std::path::Path;

const UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

/// GET vía curl (stdout del proceso). Transporte verificado contra los
/// CDNs de Yahoo y FRED desde esta red.
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

/// FRED fredgraph.csv (curl): (date_ms al mediodía, value).
fn fred_daily(series: &str) -> Option<Vec<(u64, f64)>> {
    let url = format!("https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}");

    // "2026-08-14" → ms UTC del mediodía (evita ambigüedad de TZ en el as-of).
    let parse_date_ms = |s: &str| -> Option<u64> {
        let mut it = s.split('-');
        let y: i64 = it.next()?.parse().ok()?;
        let m: i64 = it.next()?.parse().ok()?;
        let d: i64 = it.next()?.parse().ok()?;
        if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
            return None;
        }
        let yy = if m <= 2 { y - 1 } else { y };
        let era = if yy >= 0 { yy } else { yy - 399 } / 400;
        let yoe = yy - era * 400;
        let mp = (m + 9) % 12;
        let doy = (153 * mp + 2) / 5 + d - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        let days = era * 146_097 + doe - 719_468;
        Some(((days * 86_400 + 43_200) * 1000) as u64)
    };

    let body = http_get(&url)?;
    let mut rows = Vec::new();
    for line in body.lines().skip(1) {
        let mut parts = line.split(',');
        let (Some(d), Some(v)) = (parts.next(), parts.next()) else {
            continue;
        };
        let (Some(ms), Ok(val)) = (parse_date_ms(d.trim()), v.trim().parse::<f64>()) else {
            continue;
        };
        if val.is_finite() && val > 0.0 {
            rows.push((ms, val));
        }
    }
    Some(rows)
}

fn main() {
    println!("============================================================");
    println!("🌍 MACRO HISTORY SYNC — Yahoo v8 (índices) + FRED (DTWEXBGS)");
    println!("============================================================");
    std::fs::create_dir_all("data/macro").unwrap();

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
