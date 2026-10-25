//! FIRMAS DE CAMINOS RUGOSOS SOBRE TAPE REAL (GLM 161→163). El módulo §10
//! (path_signatures, GLM XLVIII-B) está certificado (Chen, reverso,
//! invariancia a remuestreo) pero JAMÁS midió tape real — el estado de
//! entrelazamiento antes del 155. Este vehículo lo despierta con el
//! estándar de mi línea: firma nivel-2 por ventana (L=64 mids), NULO por
//! permutación determinista de incrementos, Δfirma_i = S_i − S_i,nulo
//! por cada uno de los 6 términos. Pregunta honesta: ¿qué términos llevan
//! señal neta real (los cross S^{ij} son la parte no-lineal genuina)?
//!
//! GLM 163 — NULO DE MÁXIMA ENTROPÍA: bajo permutación uniforme de los
//! retornos contra dt fijos, cada cross es un estadístico LINEAL de
//! asignación ⇒ media y variancia CERRADAS (E = x̄·Σw, Var = Sxx·Sww/(n−1))
//! ⇒ z(S¹²), z(S²¹) EXACTOS por ventana (contrato CI:
//! momentos_exactos_del_nulo_unieje_vs_enumeracion, 720 perms). El 162
//! no pudo separar señal del ruido de UNA permutación muestral; el z la
//! separa analíticamente.
//!
//! #[ignore]: investigación local con tapes REALES en data/.
//! Recibo persistente outputs/firmas_camino_maxent_2026-09-14.log.

use feature_engine::path_signatures::{
    firma_nivel2_incrementos, firma_ventana_logprecio, Signature2,
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

/// GLM 163 — Momentos EXACTOS del nulo unieje (máxima entropía sobre las
/// n! asignaciones de retornos a dt). Reducción: S¹²_nulo = Σ_a v_a·x_σ(a)
/// con v_a = (Σ_{b>a} y_b) + ½·y_a; S²¹_nulo = Σ_a u_a·x_σ(a) con
/// u_a = (Σ_{b<a} y_b) + ½·y_a. E = x̄·Σw, Var = Sxx·Sww/(n−1) — la forma
/// cerrada está certificada por enumeración bruta en el contrato CI del
/// feature-engine (misma matemática, copia local de diseño experimental).
/// None = ventana degenerada (retornos constantes ⇒ Var=0 ⇒ z no medible).
fn momentos_nulo_unieje(incs: &[(f64, f64)]) -> Option<((f64, f64), (f64, f64))> {
    let n = incs.len();
    if n < 2 {
        return None;
    }
    let (x, y): (Vec<f64>, Vec<f64>) = incs.iter().copied().unzip();
    let nf = n as f64;
    let (sx, sy) = (x.iter().sum::<f64>(), y.iter().sum::<f64>());
    let x̄ = sx / nf;
    let sxx = x.iter().map(|&a| (a - x̄) * (a - x̄)).sum::<f64>();
    // Pesos de asignación: sufijo (S¹²) y prefijo (S²¹) de dt, ambos + ½·y.
    let mut v = Vec::with_capacity(n);
    let mut u = Vec::with_capacity(n);
    let (mut prefijo, mut sufijo) = (0.0f64, sy);
    for &dy in &y {
        sufijo -= dy;
        v.push(sufijo + 0.5 * dy);
        u.push(prefijo + 0.5 * dy);
        prefijo += dy;
    }
    let media_var = |w: &[f64]| -> Option<(f64, f64)> {
        let sw = w.iter().sum::<f64>();
        let w̄ = sw / nf;
        let sww = w.iter().map(|&a| (a - w̄) * (a - w̄)).sum::<f64>();
        let var = sxx * sww / (nf - 1.0);
        (var > 1e-24 && var.is_finite()).then(|| (x̄ * sw, var))
    };
    let m12 = media_var(&v)?;
    let m21 = media_var(&u)?;
    Some((m12, m21))
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
    w(
        "términos: S¹=log-acum · S²=tiempo-acum(≡1) · S¹¹,S²² · S¹²/S²¹ (cross = señal no-lineal)"
            .into(),
    );
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
    w(format!(
        "ancla recta vs remuestreo: max|Δtérmino| = {d_ancla:.2e} (esperado ≈ 0)"
    ));
    w(String::new());

    // ── Por símbolo. Además de la MEDIA del Δ (lo único que reportó el
    // 161 y que ocultó la señal por cancelación de signo), este ciclo
    // reporta mean|Δ| y std del Δ de los CROSS — la métrica honesta.
    struct Acum {
        delta_sum: [f64; 6],
        abs_sum: [f64; 6],
        sq_sum: [f64; 6],
        n: usize,
        // GLM 163 — z EXACTO contra el nulo de máxima entropía.
        z12_sum: f64,
        z12_sq: f64,
        z21_sum: f64,
        z21_sq: f64,
        nz: usize,
        z12_gt2: usize,
        z21_gt2: usize,
        degeneradas: usize,
    }
    impl Acum {
        fn nuevo() -> Self {
            Self {
                delta_sum: [0.0; 6],
                abs_sum: [0.0; 6],
                sq_sum: [0.0; 6],
                n: 0,
                z12_sum: 0.0,
                z12_sq: 0.0,
                z21_sum: 0.0,
                z21_sq: 0.0,
                nz: 0,
                z12_gt2: 0,
                z21_gt2: 0,
                degeneradas: 0,
            }
        }
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
        let mids: Vec<f64> = raw
            .iter()
            .map(|t| 0.5 * (t.bid_price + t.ask_price))
            .collect();
        let ts: Vec<u64> = raw.iter().map(|t| t.timestamp).collect();
        drop(raw);

        let paso = ((n - VENTANA) / N_VENTANAS).max(1);
        let mut acum = Acum::nuevo();
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

            // GLM 163 — z EXACTO del cross real contra los momentos
            // cerrados del nulo (máxima entropía sobre las n! asignaciones).
            match incrementos_ventana(p_v, t_v)
                .as_deref()
                .and_then(momentos_nulo_unieje)
            {
                None => acum.degeneradas += 1,
                Some(((e12, var12), (e21, var21))) => {
                    let (z12, z21) = ((fr[3] - e12) / var12.sqrt(), (fr[4] - e21) / var21.sqrt());
                    if z12.is_finite() && z21.is_finite() {
                        acum.z12_sum += z12;
                        acum.z12_sq += z12 * z12;
                        acum.z21_sum += z21;
                        acum.z21_sq += z21 * z21;
                        acum.z12_gt2 += (z12.abs() > 2.0) as usize;
                        acum.z21_gt2 += (z21.abs() > 2.0) as usize;
                        acum.nz += 1;
                    } else {
                        acum.degeneradas += 1;
                    }
                }
            }
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
            let var =
                (acum.sq_sum[3] / acum.n as f64) - (acum.delta_sum[3] / acum.n as f64).powi(2);
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
            if acum.nz > 1 {
                let (mz12, mz21) = (acum.z12_sum / acum.nz as f64, acum.z21_sum / acum.nz as f64);
                let se = 1.0 / (acum.nz as f64).sqrt();
                let sz12 = ((acum.z12_sq / acum.nz as f64 - mz12 * mz12).max(0.0)).sqrt();
                let sz21 = ((acum.z21_sq / acum.nz as f64 - mz21 * mz21).max(0.0)).sqrt();
                w(format!(
                    "{:>9} z(S¹²): media {mz12:+.4} σ {sz12:.3} |z|>2 {:>4.1}% · \
                     z(S²¹): media {mz21:+.4} σ {sz21:.3} |z|>2 {:>4.1}% · \
                     SE(media)={se:.4} · no-medibles {}",
                    "",
                    100.0 * acum.z12_gt2 as f64 / acum.nz as f64,
                    100.0 * acum.z21_gt2 as f64 / acum.nz as f64,
                    acum.degeneradas,
                ));
            }
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

    // ── GLM 163: resumen transversal de los z EXACTOS.
    let con_z: Vec<&(String, Acum)> = tabla.iter().filter(|(_, a)| a.nz > 1).collect();
    if !con_z.is_empty() {
        w(String::new());
        w(
            "──── NULO DE MÁXIMA ENTROPÍA (GLM 163): z = (S_real − E[S_nulo])/√Var_nulo ────"
                .into(),
        );
        w("E y Var CERRADAS: cross ≡ estadístico lineal de asignación;".into());
        w("E = x̄·Σw, Var = Sxx·Sww/(n−1) (contrato CI: 720 perms exactas).".into());
        w("Bajo H0 (acoplamiento precio-tiempo nulo): z ~ (0,1) por ventana.".into());
        for (k, etiqueta) in [(0usize, "z(S¹²)"), (1usize, "z(S²¹)")] {
            let medios: Vec<f64> = con_z
                .iter()
                .map(|(_, a)| {
                    if k == 0 {
                        a.z12_sum / a.nz as f64
                    } else {
                        a.z21_sum / a.nz as f64
                    }
                })
                .collect();
            let media = medios.iter().sum::<f64>() / medios.len() as f64;
            let max = medios.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min = medios.iter().cloned().fold(f64::INFINITY, f64::min);
            w(format!(
                "{etiqueta:7} media-z transversal {media:+.4} · min {min:+.4} · max {max:+.4} · spread {s:.4}",
                s = max - min
            ));
        }
        w(String::new());
        w("GATE (decisión honesta): propuesta de consumo al consejo SOLO si".into());
        w("algún símbolo mantiene |media-z| ≫ SE(≈0.016 con n=4000) COHERENTE".into());
        w("y la separación transversal entre símbolos es clara; si los z son".into());
        w("~(0,1) el acoplamiento precio-tiempo es estadísticamente nulo y la".into());
        w("línea se archiva con el álgebra completa (161 identidad + 162".into());
        w("rectificación + 163 momentos exactos) — cualquiera de los dos".into());
        w("cierres es limpio y medible.".into());
    }
    w("==========================================================================".into());

    let reporte = lineas.join("\n");
    eprintln!("{reporte}");
    let salida = std::path::Path::new("outputs").join("firmas_camino_maxent_2026-09-14.log");
    if std::fs::create_dir_all("outputs").is_ok() {
        let _ = std::fs::write(&salida, format!("{reporte}\n"));
        eprintln!("recibo: {}", salida.display());
    }
}
