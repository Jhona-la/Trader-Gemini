//! MEDICIÓN DEL ENTRELAZAMIENTO TEMPORAL NETO SOBRE TAPE REAL (GLM 155).
//! El observable (§56, `feature_engine::entrelazamiento`) está certificado
//! con contratos sintéticos en CI; este vehículo lo mide sobre los 10 tapes
//! del 2026-09-14 para dar al consejo la evidencia con la que decidir su
//! consumo (voz/gate) — la misma trayectoria que Koopman 149→150→215.
//!
//! Qué se mide por símbolo × bloque × modo de binarización (signo y
//! mediana): S del estado empírico ψ=√p̂ sobre ventanas deslizantes de
//! L=8 retornos, S_nulo (permutación determinista: misma marginal, orden
//! destruido) y ΔS = S − S_nulo = entrelazamiento temporal genuino neto
//! del ruido de muestreo. Anclas certificadas: iid ⇒ ΔS≈0 (test en CI);
//! alternante ⇒ S=ln2 (test en CI).
//!
//! #[ignore]: investigación local con tapes REALES en data/ (24 GB).
//! Recibo persistente outputs/mps_entrelazamiento_*.log.

use feature_engine::entrelazamiento::{ModoBinarizacion, entrelazamiento_neto};

/// Tapes del censo: los 10 símbolos del 2026-09-14 (mismo período exacto
/// para comparación cruzada limpia).
const TAPES: [&str; 10] = [
    "ADAUSDT_2026-09-14_REAL.bin",
    "ATOMUSDT_2026-09-14_REAL.bin",
    "BNBUSDT_2026-09-14_REAL.bin",
    "BTCUSDT_2026-09-14_REAL.bin",
    "ICPUSDT_2026-09-14_REAL.bin",
    "NEARUSDT_2026-09-14_REAL.bin",
    "SOLUSDT_2026-09-14_REAL.bin",
    "XLMUSDT_2026-09-14_REAL.bin",
    "XMRUSDT_2026-09-14_REAL.bin",
    "XRPUSDT_2026-09-14_REAL.bin",
];

const L: usize = 8;
const BLOQUES: usize = 10;
const MAX_TICKS: usize = 2_000_000;

fn path_tape(nombre: &str) -> Option<std::path::PathBuf> {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    [
        // TG_DATA_ROOT es la CONVENCIÓN de DIRECTORIO de datos (lección 136):
        // el tape se resuelve DENTRO, no como ruta en sí — usarlo crudo
        // selecciona el directorio (exists() = true) y el open niega.
        std::env::var("TG_DATA_ROOT")
            .map(|root| std::path::PathBuf::from(root).join(nombre))
            .ok(),
        Some(manifest.join("../..").join("data").join(nombre)),
        Some(manifest.join("../../..").join("data").join(nombre)),
    ]
    .into_iter()
    .flatten()
    .find(|p| p.is_file())
}

fn cuantil(vals: &mut Vec<f64>, q: f64) -> f64 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ix = ((vals.len() as f64 - 1.0) * q).round() as usize;
    vals[ix.min(vals.len() - 1)]
}

/// epoch-ms → "YYYY-MM-DD HH:MM:SS" UTC (sin chrono — fechas civiles por
/// días desde 1970, algoritmo de Howard Hinnant).
fn utc_legible(ms: u64) -> String {
    let secs = (ms / 1000) as i64;
    let dias = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let z = dias + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}",
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    )
}

#[test]
#[ignore = "investigación local: requiere tapes REALES en data/ (24 GB)"]
fn mps_entrelazamiento_sobre_tapes_2026_09_14() {
    let mut lineas: Vec<String> = Vec::new();
    let mut w = |s: String| lineas.push(s);
    w("========== ENTRELAZAMIENTO TEMPORAL NETO SOBRE TAPE REAL (GLM 155) ==========".into());
    w(format!(
        "ventana L={L} sitios (2^{L}=256 configs) · stride 1 · {BLOQUES} bloques/símbolo · \
         modos signo y mediana · ΔS = S − S_nulo (nulo: permutación determinista)"
    ));
    w("anclas certificadas (CI): iid ⇒ ΔS≈0 · serie alternante ⇒ S=ln2".into());
    w(String::new());

    // Agregado cruzado para el resumen final.
    let mut deltas_signo: Vec<(String, Vec<f64>)> = Vec::new();
    let mut deltas_mediana: Vec<(String, Vec<f64>)> = Vec::new();

    for nombre_tape in TAPES {
        let Some(path) = path_tape(nombre_tape) else {
            w(format!("⚠️ tape ausente — skip: {nombre_tape}"));
            continue;
        };
        let raw = match backtest_engine::tick_replayer::load_binary_ticks_prefix(&path, 0, MAX_TICKS)
        {
            Ok(r) => r,
            Err(e) => {
                // Diagnóstico honesto: ¿falla la APERTURA o el MMAP? La
                // lectura plana clasifica la causa sin adivinar.
                let lectura_plana = std::fs::File::open(&path)
                    .and_then(|mut f| {
                        let mut b = [0u8; 8];
                        use std::io::Read;
                        f.read_exact(&mut b)?;
                        Ok(String::from_utf8_lossy(&b).into_owned())
                    })
                    .map_err(|e2| format!("{e2}"));
                w(format!(
                    "⚠️ tape ilegible — skip: {nombre_tape} ({e:?}) | path={} | lectura_plana={}",
                    path.display(),
                    lectura_plana.as_deref().unwrap_or("ERR"),
                ));
                continue;
            }
        };
        let n = raw.len();
        if n < BLOQUES * (L + 2) {
            w(format!("⚠️ tape corto ({n} ticks) — skip: {nombre_tape}"));
            continue;
        }
        // mid prices + timestamps (una pasada, sin clonar el tape).
        let mids: Vec<f64> = raw.iter().map(|t| 0.5 * (t.bid_price + t.ask_price)).collect();
        let ts: Vec<u64> = raw.iter().map(|t| t.timestamp).collect();
        drop(raw);

        let chunk = n.div_ceil(BLOQUES);
        w(format!(
            "── {nombre_tape} · {n} ticks · período {} → {}",
            utc_legible(ts[0]),
            utc_legible(ts[n - 1])
        ));
        let mut ds_simbolo: Vec<f64> = Vec::new();
        let mut dm_simbolo: Vec<f64> = Vec::new();
        for (b, bloque) in mids.chunks(chunk).take(BLOQUES).enumerate() {
            if bloque.len() < L + 2 {
                continue;
            }
            let rets: Vec<f64> = bloque.windows(2).map(|par| (par[1] / par[0]).ln()).collect();
            let i0 = b * chunk;
            let i1 = i0 + bloque.len() - 1;
            let mut fila = |modo: ModoBinarizacion, tag: &str| -> String {
                match entrelazamiento_neto(&rets, L, modo) {
                    Ok(m) => {
                        if modo == ModoBinarizacion::Signo {
                            ds_simbolo.push(m.delta);
                        } else {
                            dm_simbolo.push(m.delta);
                        }
                        format!(
                            "  bloque {b:2} [{} → {}] [{tag:6}] S={:.5} S_nulo={:.5} ΔS={:+.5} \
                             ventanas={} desc={}",
                            utc_legible(ts[i0.min(n - 1)]),
                            utc_legible(ts[i1.min(n - 1)]),
                            m.s,
                            m.s_nulo,
                            m.delta,
                            m.ventanas,
                            m.descartados
                        )
                    }
                    Err(e) => format!(
                        "  bloque {b:2} [{} → {}] [{tag:6}] AUSENCIA HONESTA: {e:?}",
                        utc_legible(ts[i0.min(n - 1)]),
                        utc_legible(ts[i1.min(n - 1)])
                    ),
                }
            };
            w(fila(ModoBinarizacion::Signo, "signo"));
            w(fila(ModoBinarizacion::Mediana, "mediana"));
        }
        if !ds_simbolo.is_empty() {
            let media_s = ds_simbolo.iter().sum::<f64>() / ds_simbolo.len() as f64;
            let media_m = dm_simbolo.iter().sum::<f64>() / dm_simbolo.len() as f64;
            w(format!("  ⇒ media ΔS: signo {media_s:+.5} · mediana {media_m:+.5}"));
            deltas_signo.push((nombre_tape.to_string(), ds_simbolo));
            deltas_mediana.push((nombre_tape.to_string(), dm_simbolo));
        }
        w(String::new());
    }

    // Resumen cruzado.
    w("──── RESUMEN CRUZADO (10 símbolos, mismo período 2026-09-14) ────".into());
    for (tag, datos) in [("signo", &deltas_signo), ("mediana", &deltas_mediana)] {
        let mut todos: Vec<f64> = datos.iter().flat_map(|(_, v)| v.iter().copied()).collect();
        if todos.is_empty() {
            w(format!("modo {tag}: sin mediciones"));
            continue;
        }
        let media = todos.iter().sum::<f64>() / todos.len() as f64;
        let p95 = cuantil(&mut todos, 0.95);
        let p50 = cuantil(&mut todos, 0.50);
        let p5 = cuantil(&mut todos, 0.05);
        let positivos = datos
            .iter()
            .flat_map(|(_, v)| v.iter())
            .filter(|d| **d > 0.0)
            .count();
        let total: usize = datos.iter().map(|(_, v)| v.len()).sum();
        let (m_max, s_max) = datos
            .iter()
            .map(|(s, v)| (v.iter().sum::<f64>() / v.len() as f64, s))
            .fold((f64::NEG_INFINITY, String::new()), |(am, asx), (m, s)| {
                if m > am {
                    (m, s.clone())
                } else {
                    (am, asx)
                }
            });
        w(format!(
            "modo {tag}: ΔS media {media:+.5} · p5 {p5:+.5} · p50 {p50:+.5} · p95 {p95:+.5} \
             · bloques ΔS>0: {positivos}/{total} · mayor media: {m_max:+.5} ({s_max})"
        ));
    }
    w("============================================================================".into());

    let reporte = lineas.join("\n");
    eprintln!("{reporte}");
    let salida = std::path::Path::new("outputs").join("mps_entrelazamiento_2026-09-14.log");
    if std::fs::create_dir_all("outputs").is_ok() {
        let _ = std::fs::write(&salida, format!("{reporte}\n"));
        eprintln!("recibo: {}", salida.display());
    }
}
