//! FIRMAS DE CAMINOS RUGOSOS SOBRE TAPE REAL (GLM 161). El módulo §10
//! (path_signatures, GLM XLVIII-B) está certificado (Chen, reverso,
//! invariancia a remuestreo) pero JAMÁS midió tape real — el estado de
//! entrelazamiento antes del 155. Este vehículo lo despierta con el
//! estándar de mi línea: firma nivel-2 por ventana (L=64 mids), NULO por
//! permutación determinista de incrementos, Δfirma_i = S_i − S_i,nulo
//! por cada uno de los 6 términos. Pregunta honesta: ¿qué términos llevan
//! señal neta real (los cross S^{ij} son la parte no-lineal genuina)?
//!
//! #[ignore]: investigación local con tapes REALES en data/.
//! Recibo persistente outputs/firmas_camino_2026-09-14.log.

use feature_engine::path_signatures::{Signature2, firma_nivel2_incrementos, firma_ventana_logprecio};

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

/// Ventana de firma: 64 mids (camino 2D = log-precio × fracción de ventana).
const VENTANA: usize = 64;
const MAX_TICKS: usize = 2_000_000;
/// Ventanas por símbolo (submuestreo del total disponible).
const N_VENTANAS: usize = 4_000;

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

/// Fisher–Yates determinista (misma semilla de mi línea — reproducible).
fn permutacion_determinista<T>(v: &mut [T]) {
    let mut semilla: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rng = move || {
        semilla ^= semilla >> 12;
        semilla ^= semilla << 25;
        semilla ^= semilla >> 27;
        semilla.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    for i in (1..v.len()).rev() {
        let j = (rng() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

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

/// Reconstruye los incrementos de una ventana igual que el motor y aplica
/// el nulo UNIEJE (GLM 162, corrección que exige el hallazgo del 161):
/// permuta SÓLO el vector de log-retornos contra los dt en su orden
/// original. Bajo este nulo ΣΔ¹, Σ(Δ¹)² — y por la identidad del 161
/// S¹ y S¹¹ — se conservan EXACTO (marginal del precio intacta); los
/// cross S¹²/S²¹ sí se rompen (ΣΔ¹Δ² cambia): son los ÚNICOS términos
/// que pueden llevar Δfirma ≠ 0. Permutar pares (el nulo del 161)
/// conservaba hasta los cross — el cero medido era álgebra, no mercado.
fn incrementos_ventana(mids: &[f64], ts_ms: &[u64]) -> Option<Vec<(f64, f64)>> {
    let span = ts_ms.last()?.checked_sub(*ts_ms.first()?)?;
    if span == 0 {
        return None;
    }
    let mut incs: Vec<(f64, f64)> = Vec::with_capacity(mids.len() - 1);
    for w in mids.windows(2).zip(ts_ms.windows(2)) {
        let (p_prev, p) = (w.0[0], w.0[1]);
        let (t_prev, t) = (w.1[0], w.1[1]);
        if p <= 0.0 || p_prev <= 0.0 {
            return None;
        }
        let log_return = (p / p_prev).ln();
        incs.push((log_return, (t - t_prev) as f64 / span as f64));
    }
    Some(incs)
}

/// Nulo unieje: baraja SOLO los log-retornos, dt quedan en orden original.
fn firma_nula_unieje(mids: &[f64], ts_ms: &[u64]) -> Option<Signature2> {
    let mut incs = incrementos_ventana(mids, ts_ms)?;
    let mut rets: Vec<f64> = incs.iter().map(|(r, _)| *r).collect();
    permutacion_determinista(&mut rets);
    for (par, r) in incs.iter_mut().zip(rets.iter()) {
        par.0 = *r;
    }
    let s = firma_nivel2_incrementos(&incs);
    s.to_features().iter().all(|x| x.is_finite()).then_some(s)
}

#[test]
#[ignore = "investigación local: requiere tapes REALES en data/ (24 GB)"]
fn firmas_camino_sobre_tapes_2026_09_14() {
    let mut lineas: Vec<String> = Vec::new();
    let mut w = |s: String| lineas.push(s);
    w("========== FIRMAS DE CAMINOS RUGOSOS NIVEL-2 SOBRE TAPE REAL (GLM 161) ==========".into());
    w(format!(
        "ventana {VENTANA} mids · hasta {N_VENTANAS} ventanas/símbolo · camino 2D = (log-retorno, fracción de ventana) · \
         Δfirma = S − S_nulo (nulo: permutación determinista de pares)"
    ));
    w("términos: S¹=log-acum · S²=tiempo-acum(≡1) · S¹¹,S²² · S¹²/S²¹ (cross = señal no-lineal)".into());
    w("ancla CI: línea recta ⇒ firma idéntica al remuestreo (invariancia)".into());

    // ── Ancla viva: ventana sintética en línea recta pura.
    let mut lin_p = Vec::new();
    let mut lin_t = Vec::new();
    for k in 0..VENTANA {
        lin_p.push(100.0 * (1.0 + 0.001 * k as f64));
        lin_t.push(1_700_000_000_000u64 + (k as u64) * 100);
    }
    let f_lin = firma_ventana_logprecio(&lin_p, &lin_t).expect("línea recta firme");
    // Remuestreo de la MISMA recta a la mitad de puntos ⇒ firma ≈ idéntica.
    let mut lin_p2 = Vec::new();
    let mut lin_t2 = Vec::new();
    for k in (0..VENTANA).step_by(2) {
        lin_p2.push(lin_p[k]);
        lin_t2.push(lin_t[k]);
    }
    let f_lin2 = firma_ventana_logprecio(&lin_p2, &lin_t2).expect("recta remuestreada");
    let d_ancla: f64 = f_lin
        .to_features()
        .iter()
        .zip(f_lin2.to_features().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    w(format!("ancla recta vs remuestreo: max|Δtérmino| = {d_ancla:.2e} (esperado ≈ 0)"));
    w(String::new());

    // ── Por símbolo. Además de la MEDIA del Δ (lo único que reportó el
    // 161 y que ocultó la señal por cancelación de signo), este ciclo
    // reporta mean|Δ| y std del Δ de los CROSS — la métrica honesta.
    struct Acum {
        delta_sum: [f64; 6],
        abs_sum: [f64; 6],
        sq_sum: [f64; 6],
        n: usize,
    }
    let mut tabla: Vec<(String, Acum)> = Vec::new();
    for nombre_tape in TAPES {
        let Some(path) = path_tape(nombre_tape) else {
            w(format!("⚠️ tape ausente — skip: {nombre_tape}"));
            continue;
        };
        let raw =
            match backtest_engine::tick_replayer::load_binary_ticks_prefix(&path, 0, MAX_TICKS) {
                Ok(r) => r,
                Err(e) => {
                    w(format!("⚠️ tape ilegible — skip: {nombre_tape} ({e:?})"));
                    continue;
                }
            };
        let n = raw.len();
        if n < VENTANA + 2 {
            continue;
        }
        let mids: Vec<f64> = raw.iter().map(|t| 0.5 * (t.bid_price + t.ask_price)).collect();
        let ts: Vec<u64> = raw.iter().map(|t| t.timestamp).collect();
        drop(raw);

        let paso = ((n - VENTANA) / N_VENTANAS).max(1);
        let mut acum = Acum { delta_sum: [0.0; 6], abs_sum: [0.0; 6], sq_sum: [0.0; 6], n: 0 };
        let mut ventanas_usadas = 0usize;
        for i in (0..=n - VENTANA).step_by(paso) {
            if ventanas_usadas >= N_VENTANAS {
                break;
            }
            let (p_v, t_v) = (&mids[i..i + VENTANA], &ts[i..i + VENTANA]);
            let Some(f) = firma_ventana_logprecio(p_v, t_v) else {
                continue;
            };
            let Some(f_nulo) = firma_nula_unieje(p_v, t_v) else {
                continue;
            };
            let (fr, fn_) = (f.to_features(), f_nulo.to_features());
            for k in 0..6 {
                let d = fr[k] - fn_[k];
                acum.delta_sum[k] += d;
                acum.abs_sum[k] += d.abs();
                acum.sq_sum[k] += d * d;
            }
            acum.n += 1;
            ventanas_usadas += 1;
        }
        let simbolo = nombre_tape.trim_end_matches("_2026-09-14_REAL.bin");
        if acum.n > 0 {
            // to_features orden: (S¹, S², S¹¹, S¹², S²¹, S²²) — los cross
            // son los índices 3 y 4 (la cabecera del 161 los rotuló mal).
            let medias: Vec<String> = acum
                .delta_sum
                .iter()
                .map(|d| format!("{:+.5}", d / acum.n as f64))
                .collect();
            let mean_abs_cross = acum.abs_sum[3] / acum.n as f64;
            let var = (acum.sq_sum[3] / acum.n as f64)
                - (acum.delta_sum[3] / acum.n as f64).powi(2);
            let std_cross = var.max(0.0).sqrt();
            w(format!(
                "{simbolo:9} [{} → {}] n={} Δmedia=({}) · |ΔS¹²|med={:.5} σ(ΔS¹²)={:.5}",
                utc_legible(ts[0]),
                utc_legible(ts[n - 1]),
                acum.n,
                medias.join(", "),
                mean_abs_cross,
                std_cross
            ));
            tabla.push((simbolo.to_string(), acum));
        }
    }

    // ── Resumen por término (media Y magnitud — la lección del 162).
    w(String::new());
    w("──── LECTURA POR TÉRMINO (media y mean|Δ| transversales) ────".into());
    // to_features orden: (S¹, S², S¹¹, S¹², S²¹, S²²).
    let nombres = ["S¹", "S²", "S¹¹", "S¹²", "S²¹", "S²²"];
    for k in 0..6 {
        let vals: Vec<f64> = tabla
            .iter()
            .map(|(_, a)| a.delta_sum[k] / a.n as f64)
            .collect();
        let abss: Vec<f64> = tabla
            .iter()
            .map(|(_, a)| a.abs_sum[k] / a.n as f64)
            .collect();
        if vals.is_empty() {
            continue;
        }
        let media = vals.iter().sum::<f64>() / vals.len() as f64;
        let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let abs_med = abss.iter().sum::<f64>() / abss.len() as f64;
        w(format!(
            "{:3} Δmedia {media:+.5} · [|Δ|]med {abs_med:.5} · min {min:+.5} · max {max:+.5} · spread {s:.5}",
            nombres[k],
            s = max - min
        ));
    }
    w(String::new());
    w("RECTIFICACIÓN DEL 161 (lección del contrato 162): la identidad de".into());
    w("permutación de pares cubre S¹,S²,S¹¹,S²² y la SUMA S¹²+S²¹=S¹S²;".into());
    w("los cross INDIVIDUALES cambian por ventana (~0.02 en sintético).".into());
    w("El 'Δfirma≡0 EXACTO' del 161 fue sobre-redondeo de MEDIAS ±1e-5:".into());
    w("la media con signo cancela; mean|Δ| y σ son la métrica honesta.".into());
    w("Bajo el nulo UNIEJE (retornos barajados, dt fijos): S¹,S²,S¹¹,S²²".into());
    w("conservados EXACTO (contrato CI) — TODO el Δ vive en los cross.".into());
    w("Consumo: propuesta al consejo SOLO si mean|ΔS¹²| separa símbolos".into());
    w("transversalmente más allá del ruido del nulo.".into());
    w("==========================================================================".into());

    let reporte = lineas.join("\n");
    eprintln!("{reporte}");
    let salida = std::path::Path::new("outputs").join("firmas_camino_2026-09-14.log");
    if std::fs::create_dir_all("outputs").is_ok() {
        let _ = std::fs::write(&salida, format!("{reporte}\n"));
        eprintln!("recibo: {}", salida.display());
    }
}
