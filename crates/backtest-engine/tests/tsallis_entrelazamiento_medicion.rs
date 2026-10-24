//! BARRIDO DEL ENTRELAZAMIENTO DE TSALLIS S_q SOBRE TAPE REAL (GLM 158).
//! La generalización no-extensiva (mandato §10) del observable §56: el
//! MISMO espectro de Schmidt (una diagonalización), entropías con knob q.
//! Pregunta honesta del recibo (sin conclusión pre-escrita): ¿algún q≠1
//! AMPLIFICA la separación BTC-vs-resto (16× a q=1) o reordena símbolos?
//! Si q no añade nada, queda q=1 (von Neumann) como canónico.
//!
//! #[ignore]: investigación local con tapes REALES en data/.
//! Recibo persistente outputs/tsallis_entrelazamiento_2026-09-14.log.

use feature_engine::entrelazamiento::{ModoBinarizacion, entrelazamiento_neto_q};

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
const Q_GRID: [f64; 5] = [0.5, 0.8, 1.0, 1.2, 1.5];

fn path_tape(nombre: &str) -> Option<std::path::PathBuf> {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    [
        // TG_DATA_ROOT es DIRECTORIO (convención 136 / lección 155).
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

#[test]
#[ignore = "investigación local: requiere tapes REALES en data/ (24 GB)"]
fn tsallis_entrelazamiento_sobre_tapes_2026_09_14() {
    let mut lineas: Vec<String> = Vec::new();
    let mut w = |s: String| lineas.push(s);
    w("========== ENTRELAZAMIENTO TSALLIS S_q SOBRE TAPE REAL (GLM 158) ==========".into());
    w(format!(
        "ventana L={L} · {BLOQUES} bloques/símbolo · modo signo · ΔS_q = S_q − S_q,nulo \
         (nulo: permutación determinista) · q ∈ {Q_GRID:?}"
    ));
    w("anclas CI: q=1 ⇒ von Neumann EXACTO · gato ⇒ ln_q(2) · iid ⇒ ΔS_q≈0".into());
    w(String::new());

    // (símbolo → [ΔS_q promedio por q])
    let mut filas: Vec<(String, Vec<f64>)> = Vec::new();
    for nombre_tape in TAPES {
        let Some(path) = path_tape(nombre_tape) else {
            w(format!("⚠️ tape ausente — skip: {nombre_tape}"));
            continue;
        };
        let raw = match backtest_engine::tick_replayer::load_binary_ticks_prefix(&path, 0, MAX_TICKS)
        {
            Ok(r) => r,
            Err(e) => {
                w(format!("⚠️ tape ilegible — skip: {nombre_tape} ({e:?})"));
                continue;
            }
        };
        let n = raw.len();
        if n < BLOQUES * (L + 2) {
            continue;
        }
        let mids: Vec<f64> = raw.iter().map(|t| 0.5 * (t.bid_price + t.ask_price)).collect();
        drop(raw);
        let chunk = n.div_ceil(BLOQUES);

        // Acumular ΔS_q por q sobre los bloques.
        let mut sumas = vec![0.0f64; Q_GRID.len()];
        let mut cuentas = vec![0usize; Q_GRID.len()];
        for bloque in mids.chunks(chunk).take(BLOQUES) {
            if bloque.len() < L + 2 {
                continue;
            }
            let rets: Vec<f64> = bloque.windows(2).map(|par| (par[1] / par[0]).ln()).collect();
            for (iq, &q) in Q_GRID.iter().enumerate() {
                if let Ok(m) = entrelazamiento_neto_q(&rets, L, ModoBinarizacion::Signo, q) {
                    sumas[iq] += m.delta;
                    cuentas[iq] += 1;
                }
            }
        }
        let medias: Vec<f64> = sumas
            .iter()
            .zip(cuentas.iter())
            .map(|(s, c)| if *c > 0 { s / *c as f64 } else { f64::NAN })
            .collect();
        let simbolo = nombre_tape.trim_end_matches("_2026-09-14_REAL.bin");
        let campos: Vec<String> = Q_GRID
            .iter()
            .zip(medias.iter())
            .map(|(q, d)| format!("q={q:.1}: {d:+.4}"))
            .collect();
        w(format!("{simbolo:9} {}", campos.join("  ")));
        filas.push((simbolo.to_string(), medias));
    }

    // Resumen: dispersión transversal (max/min) y ranking por q.
    w(String::new());
    w("──── DISPERSIÓN TRANSVERSAL POR q (BTC vs resto) ────".into());
    for (iq, &q) in Q_GRID.iter().enumerate() {
        let valores: Vec<(String, f64)> = filas
            .iter()
            .filter(|(_, m)| m[iq].is_finite())
            .map(|(s, m)| (s.clone(), m[iq]))
            .collect();
        if valores.len() < 2 {
            continue;
        }
        let mut ordenadas = valores.clone();
        ordenadas.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let max = ordenadas.first().map(|(_, v)| *v).unwrap_or(f64::NAN);
        let min = ordenadas.last().map(|(_, v)| *v).unwrap_or(f64::NAN);
        let ratio = if min.abs() > 1e-9 { max / min } else { f64::INFINITY };
        let top3: Vec<String> = ordenadas
            .iter()
            .take(3)
            .map(|(s, v)| format!("{s} {v:+.4}"))
            .collect();
        w(format!(
            "q={q:.1}: max/min = {ratio:5.1}× · top3: {}",
            top3.join(", ")
        ));
    }
    w(String::new());
    w("PREGUNTA HONESTA: ¿algún q≠1 amplifica la separación o reordena?".into());
    w("(sin conclusión pre-escrita — los números de arriba deciden;".into());
    w("  si q≠1 no añade señal, q=1 queda como canónico y el knob se".into());
    w("  archiva para tapes con colas más extremas)".into());
    w("==========================================================================".into());

    let reporte = lineas.join("\n");
    eprintln!("{reporte}");
    let salida = std::path::Path::new("outputs").join("tsallis_entrelazamiento_2026-09-14.log");
    if std::fs::create_dir_all("outputs").is_ok() {
        let _ = std::fs::write(&salida, format!("{reporte}\n"));
        eprintln!("recibo: {}", salida.display());
    }
}
