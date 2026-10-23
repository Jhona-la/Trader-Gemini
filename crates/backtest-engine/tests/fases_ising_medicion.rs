//! MAPA DE FASES S↔g DEL ISING SOBRE LOS ΔS DEL MERCADO (GLM 156).
//! Calibración física del censo del 155: la curva S(g) del ground state
//! (DMRG del 154, mismo funcional — corte medio de ψ sobre 2^8) convierte
//! cada ΔS medido en una posición del diagrama de fases del Ising
//! transverso. Titular físico: la distancia logarítmica a criticidad
//! |log g − log g_max|. ΔS sobre el techo S_max ⇒ SUPERCRÍTICO en
//! entrelazamiento (ausencia honesta, sin rama inventada).
//!
//! #[ignore]: investigación local con tapes REALES en data/ + DMRG 40×.
//! Recibo persistente outputs/fases_ising_2026-09-14.log.

use feature_engine::entrelazamiento::{
    CurvaFases, EntrelazamientoError, ModoBinarizacion, entrelazamiento_neto,
};

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
/// Curva fina del vehículo: D=8 (el máximo certificado), 40 puntos
/// log-uniformes en [0.03, 33] (~5 décadas alrededor de g_c=1).
const CURVA_N: usize = 40;
const CURVA_SWEEPS: usize = 6;

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
#[ignore = "investigación local: tapes REALES en data/ + 40 corridas DMRG D=8"]
fn fases_ising_sobre_tapes_2026_09_14() {
    let mut lineas: Vec<String> = Vec::new();
    let mut w = |s: String| lineas.push(s);
    w("========== MAPA DE FASES S↔g DEL ISING (GLM 156) ==========".into());
    w(format!(
        "curva: L={L} D=8 sweeps={CURVA_SWEEPS} · {CURVA_N} puntos log-uniformes g∈[0.03, 33] · \
         ΔS del 155 (signo, bloques de ~{}k ticks)"
        , MAX_TICKS / BLOQUES / 1000
    ));

    // ── 1. Curva de fases del Ising (una construcción compartida).
    let curva = match CurvaFases::construir(L, 8, CURVA_SWEEPS, 0.03, 33.0, CURVA_N) {
        Ok(c) => c,
        Err(e) => {
            w(format!("AUSENCIA HONESTA al construir la curva: {e:?}"));
            return;
        }
    };
    w(format!(
        "S(g): S_max={:.5} en g_max={:.4} (región crítica g_c=1 del Ising)",
        curva.s_max, curva.g_max
    ));

    // ── 2. Por símbolo: ΔS por bloque → inversión → mapa de fases.
    let mut sobre_techo = 0usize;
    let mut invertibles = 0usize;
    let mut distancias: Vec<(String, f64, f64)> = Vec::new(); // (símbolo, distancia_log media, g medio)
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

        let mut linea_deltas = String::new();
        let mut dist_media: f64 = 0.0;
        let mut g_media: f64 = 0.0;
        let mut n_inv: usize = 0;
        for bloque in mids.chunks(chunk).take(BLOQUES) {
            if bloque.len() < L + 2 {
                continue;
            }
            let rets: Vec<f64> = bloque.windows(2).map(|par| (par[1] / par[0]).ln()).collect();
            let m = match entrelazamiento_neto(&rets, L, ModoBinarizacion::Signo) {
                Ok(m) => m,
                Err(e) => {
                    linea_deltas.push_str(&format!(" [AUS:{e:?}]"));
                    continue;
                }
            };
            match curva.invertir(m.delta) {
                Ok(inv) => {
                    // Titular: el cruce más cercano a criticidad g_c=1.
                    let (d, g) = inv
                        .distancias_log
                        .iter()
                        .zip(inv.cruces.iter())
                        .fold((f64::INFINITY, f64::NAN), |(md, mg), (d, g)| {
                            if *d < md { (*d, *g) } else { (md, mg) }
                        });
                    dist_media += d;
                    g_media += g;
                    n_inv += 1;
                    invertibles += 1;
                    linea_deltas.push_str(&format!(" [ΔS={:.4} g={:.3} d={:.2}]", m.delta, g, d));
                }
                Err(EntrelazamientoError::FueraDeCurva { s, s_max, .. }) => {
                    if s > s_max {
                        sobre_techo += 1;
                        linea_deltas.push_str(&format!(
                            " [ΔS={:.4} SOBRE-TECHO (>S_max={:.4}: más coherente que el gato)]",
                            m.delta, s_max
                        ));
                    } else {
                        linea_deltas.push_str(&format!(" [ΔS={:.4} sub-borde]", m.delta));
                    }
                }
                Err(e) => linea_deltas.push_str(&format!(" [ERR {e:?}]")),
            }
        }
        let simbolo = nombre_tape.trim_end_matches("_2026-09-14_REAL.bin");
        if n_inv > 0 {
            dist_media /= n_inv as f64;
            g_media /= n_inv as f64;
            distancias.push((simbolo.to_string(), dist_media, g_media));
        }
        w(format!("{simbolo:9}{linea_deltas}"));
    }

    // ── 3. Resumen + propuesta al consejo.
    w(String::new());
    w(format!(
        "invertibles: {invertibles} bloques · sobre-techo: {sobre_techo} bloques \
         (ΔS > S_max={:.5} = más coherentes que el GATO del Ising L=8)",
        curva.s_max
    ));
    w("lectura: ΔS ALTO ⇒ lado ORDENADO (g<1, gato coherente: tendencia); \
       ΔS BAJO ⇒ lado PARAMAGNÉTICO (g>1: ruido microestructural); \
       g_c=1 = punto crítico (transición de fase)."
        .into());
    distancias.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    w("posición en el diagrama (g medio por símbolo, ascendente = más paramagnético):".into());
    for (s, d, g) in &distancias {
        let lado = if *g < 1.0 { "ORDENADO " } else { "PARAMAGN." };
        w(format!("  {s:9} ḡ={g:.4} [{lado}] |log g−log g_c|={d:.3}"));
    }
    if let Some((mas_ordenado, _, g_min)) = distancias.first() {
        w(format!(
            "⇒ {mas_ordenado} opera MÁS CERCA del régimen ordenado/coherente (ḡ={g_min:.4})"
        ));
    }
    w(String::new());
    w("PROPUESTA AL CONSEJO (GLM 156 — el consumo NO está implementado):".into());
    w("  1. g(ΔS) por símbolo como FEATURE DE RÉGIMEN: lado ordenado (g<1) =".into());
    w("     estructura coherente de tendencia; lado paramagnético (g>1) =".into());
    w("     ruido dominante. Candidato a voz/gate del confluence con la".into());
    w("     calibración RMT de AGY como precedente de adopción.".into());
    w("  2. Distancia |log g − log g_c| = cuán lejos de la transición: los".into());
    w("     símbolos que CRUZAN g_c entre bloques marcan cambio de régimen".into());
    w("     — señal de re-pricing estructural (candidato a veto/atenúa).".into());
    w("  3. Consumo real SOLO tras visto bueno del consejo con este recibo.".into());
    w("============================================================================".into());

    let reporte = lineas.join("\n");
    eprintln!("{reporte}");
    let salida = std::path::Path::new("outputs").join("fases_ising_2026-09-14.log");
    if std::fs::create_dir_all("outputs").is_ok() {
        let _ = std::fs::write(&salida, format!("{reporte}\n"));
        eprintln!("recibo: {}", salida.display());
    }
}
