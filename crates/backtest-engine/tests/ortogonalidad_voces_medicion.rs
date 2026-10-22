//! MEDICIÓN DE ORTOGONALIDAD ESPECTRAL DEL ENSAMBLE (GLM 146, 2026-10-21/22).
//!
//! El censo (GLM 129) midió QUIÉN habla; este vehículo mide si lo que dicen
//! es información INDEPENDIENTE. Sobre el tape REAL:
//!   1. corral de votos por fuente (voto medio CON SIGNO, `votos_por_fuente_snapshot`),
//!   2. co-observaciones por par ⇒ k-core de hablantes (grado ≥1 excluye mudos;
//!      pelado iterativo exige TODOS-CONTRA-TODOS ≥ MIN_CO_OBS),
//!   3. correlación por CASOS COMPLETOS del core (ticks donde TODO el core
//!      habló) — PSD por construcción; la variante por pares-deletion NO es
//!      PSD (subconjuntos distintos por par) y el eigensolver la rechaza,
//!   4. espectro Jacobi zero-alloc + borde Marchenko-Pastur + effective_bets
//!      ((Σ√λ)²/Σλ sobre modos > borde = apuestas efectivas independientes),
//!   5. caza-clones: max|corr| por fuente (los 0.85-0.97 son informativamente
//!      clones que la invariancia por fuente del 127 no detecta).
//!
//! pilar «Inferencia Causal y Reducción de Multicolinealidad Espectral».
//! Reutiliza effective_bets/mp_upper_edge/Jacobi de risk-engine (misma
//! matemática del riesgo de cartera aplicada a VOCES — cero duplicación).
//!
//! #[ignore]: investigación local con tape REAL. Recibo persistente en
//! outputs/ortogonalidad_voces_*.log para el informe integral.

use backtest_engine::booktick_replay::{ReplayConfig, ReplayTick, run_booktick_replay_with_observer};

/// Pares con menos co-observaciones que esto no se reportan como medidos.
const MIN_CO_OBS: u64 = 100;
/// Muestreo del observer: 1 de cada N ticks (500K muestras de 2M ticks).
const MUESTREO: usize = 4;
/// Slots del corral de votos (el orchestrator capea fuentes en 32).
const SLOTS: usize = 32;

#[test]
#[ignore = "investigación local: requiere tape REAL de 239 MB en data/"]
fn ortogonalidad_voces_rango_efectivo_btc_sep() {
    let modelos = god_engine_core::model_reload::preload_models_from_disk();
    println!("[LG-23] modelos cargados de disco: {modelos}");
    let nombre_tape = "BTCUSDT_2026-09-14_REAL.bin";
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let candidatos = [
        std::env::var("TG_DATA_ROOT").map(std::path::PathBuf::from).ok(),
        Some(manifest.join("../..").join("data").join(nombre_tape)),
        Some(manifest.join("../../..").join("data").join(nombre_tape)),
    ];
    let Some(path) = candidatos.into_iter().flatten().find(|p| p.exists()) else {
        eprintln!("[ortogonalidad] tape ausente — skip");
        return;
    };
    let raw = backtest_engine::tick_replayer::load_binary_ticks(&path, 0)
        .expect("tape legible");
    let n = raw.len().min(2_000_000);
    let ticks: Vec<ReplayTick> = raw[..n]
        .iter()
        .map(|t| ReplayTick {
            ts_ms: t.timestamp,
            bid: t.bid_price,
            ask: t.ask_price,
            bid_qty: t.bid_qty,
            ask_qty: t.ask_qty,
        })
        .collect();
    drop(raw);

    backtest_engine::asegurar_spec_nativo("BTCUSDT");
    let genome = quantum_arena::genome_store::GenomeEnvelope::load_active()
        .map(|e| e.genome)
        .unwrap_or_else(|| quantum_arena::genome::SuperGenotype::new_baseline(0.0002, 0.0005));
    let cfg = ReplayConfig {
        initial_capital: 1000.0,
        warmup_ticks: 200,
        trade_only: false,
        shift_atr_frac: 0.10,
    };

    // Corral: nombres (orden de primera aparición, estable por registro),
    // pares Welford (histograma/k-core) y BUFFER de instantes muestreados
    // ([f64; SLOTS] con NaN donde la fuente no habló — jamás 0 imputado)
    // para la correlación por CASOS COMPLETOS del core.
    let mut nombres: Vec<&'static str> = Vec::new();
    let mut pares: Vec<((usize, usize), (u64, f64, f64, f64, f64, f64))> = Vec::new();
    let mut buffer: Vec<[f64; SLOTS]> = Vec::new();
    let mut muestras: u64 = 0;

    let _stats = run_booktick_replay_with_observer(
        &ticks,
        &genome,
        None,
        &cfg,
        |i, core| {
            if i % MUESTREO != 0 {
                return;
            }
            let votos = core.tensor_orchestrator.votos_por_fuente_snapshot();
            if votos.is_empty() {
                return;
            }
            for (nombre, _) in &votos {
                if !nombres.iter().any(|n| n == nombre) {
                    nombres.push(nombre);
                }
            }
            let mut instante = [f64::NAN; SLOTS];
            let mut hablantes: Vec<(usize, f64)> = Vec::new();
            for (nombre, v) in &votos {
                let Some(ix) = nombres.iter().position(|n| n == nombre) else {
                    continue;
                };
                let v = *v;
                if ix < SLOTS && v.is_finite() {
                    instante[ix] = v;
                    if v.abs() > 1e-9 {
                        hablantes.push((ix, v));
                    }
                }
            }
            buffer.push(instantante_segura(instante));
            muestras += 1;
            for a in 0..hablantes.len() {
                for b in (a + 1)..hablantes.len() {
                    let (i_, x) = hablantes[a];
                    let (j_, y) = hablantes[b];
                    let (lo, hi, xv, yv) = if i_ < j_ { (i_, j_, x, y) } else { (j_, i_, y, x) };
                    match pares.iter_mut().find(|((pi, pj), _)| *pi == lo && *pj == hi) {
                        Some((_, acc)) => {
                            acc.0 += 1;
                            acc.1 += xv;
                            acc.2 += yv;
                            acc.3 += xv * yv;
                            acc.4 += xv * xv;
                            acc.5 += yv * yv;
                        }
                        None => pares.push(((lo, hi), (1, xv, yv, xv * yv, xv * xv, yv * yv))),
                    }
                }
            }
        },
    );

    // ── k-core: grado ≥1 (excluye mudos) + pelado todos-contra-todos.
    let k = nombres.len();
    let tiene_par_fuerte = |a: usize, b: usize| -> bool {
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        pares
            .iter()
            .any(|((pi, pj), acc)| *pi == lo && *pj == hi && acc.0 >= MIN_CO_OBS)
    };
    let mut vivos: Vec<usize> = (0..k)
        .filter(|&f| (0..k).any(|g| g != f && tiene_par_fuerte(f, g)))
        .collect();
    loop {
        let debiles: Vec<usize> = vivos
            .iter()
            .filter(|&&f| vivos.iter().any(|&g| g != f && !tiene_par_fuerte(f, g)))
            .copied()
            .collect();
        if debiles.is_empty() || vivos.is_empty() {
            break;
        }
        for f in debiles {
            vivos.retain(|&x| x != f);
        }
    }
    let core: Vec<usize> = vivos;
    let m = core.len();

    let mut lineas: Vec<String> = Vec::new();
    let mut w = |s: String| lineas.push(s);

    // Histograma de co-observaciones: la ORTOGONALIDAD TEMPORAL es parte del
    // hallazgo — voces que disparan en instantes disjuntos no comparten
    // estadística conjunta estimable.
    {
        let mut co_obs: Vec<u64> = pares.iter().map(|(_, a)| a.0).collect();
        co_obs.sort_unstable_by(|a, b| b.cmp(a));
        w("--- co-observaciones por par (top 15) — ortogonalidad TEMPORAL ---".into());
        let mut filas: Vec<(u64, String, String)> = pares
            .iter()
            .map(|((i, j), a)| (a.0, nombres[*i].to_string(), nombres[*j].to_string()))
            .collect();
        filas.sort_by(|x, y| y.0.cmp(&x.0));
        for (n, a, b) in filas.iter().take(15) {
            w(format!("  {:>28} × {:<28} n={n}", a, b));
        }
        let p50 = co_obs.get(co_obs.len() / 2).copied().unwrap_or(0);
        w(format!(
            "pares con datos: {} | p50 co-obs: {} | pares ≥{MIN_CO_OBS}: {}",
            co_obs.len(),
            p50,
            co_obs.iter().filter(|&&n| n >= MIN_CO_OBS).count()
        ));
    }

    w("=============== ORTOGONALIDAD DE VOCES (GLM 146) ===============".into());
    w(format!(
        "tape: {nombre_tape} | ticks: {} | muestras: {} | fuentes: {} | pares observados: {}",
        ticks.len(),
        muestras,
        k,
        pares.len()
    ));
    w(format!(
        "core bien-observado (pares ≥{MIN_CO_OBS}, todos-contra-todos): {m} de {k} fuentes"
    ));

    if m >= 2 {
        // ── CASOS COMPLETOS: instantes donde TODO el core habló.
        let completos: Vec<&[f64; SLOTS]> = buffer
            .iter()
            .filter(|inst| core.iter().all(|&c| inst[c].is_finite() && inst[c].abs() > 1e-9))
            .collect();
        let t = completos.len();
        w(format!(
            "casos completos del core (todas las fuentes hablaron): {t} de {} muestras",
            buffer.len()
        ));
        if t > (m as usize) {
            // Correlación por casos completos (PSD por construcción).
            let mut suma = vec![0.0f64; m];
            let mut suma2 = vec![0.0f64; m];
            let mut sxy = vec![vec![0.0f64; m]; m];
            for inst in &completos {
                for (a, &ca) in core.iter().enumerate() {
                    let x = inst[ca];
                    suma[a] += x;
                    suma2[a] += x * x;
                    for (b, &cb) in core.iter().enumerate().skip(a + 1) {
                        sxy[a][b] += x * inst[cb];
                    }
                }
            }
            let tf = t as f64;
            let mut corr = vec![vec![0.0f64; m]; m];
            for (a, &ca) in core.iter().enumerate() {
                corr[a][a] = 1.0;
                let vx = (suma2[a] - suma[a] * suma[a] / tf) / (tf - 1.0);
                for (b, &cb) in core.iter().enumerate().skip(a + 1) {
                    let vy = (suma2[b] - suma[b] * suma[b] / tf) / (tf - 1.0);
                    let cov = (sxy[a][b] - suma[a] * suma[b] / tf) / (tf - 1.0);
                    let r = if vx > 1e-18 && vy > 1e-18 {
                        (cov / (vx.sqrt() * vy.sqrt())).clamp(-1.0, 1.0)
                    } else {
                        f64::NAN
                    };
                    corr[a][b] = r;
                    corr[b][a] = r;
                }
            }
            if corr.iter().any(|f| f.iter().any(|v| !v.is_finite())) {
                w("⚠️ fuentes sin varianza en casos completos — espectro NO calculado".into());
            } else {
                let mut flat: Vec<f64> = Vec::with_capacity(m * m);
                for fila in &corr {
                    flat.extend_from_slice(fila);
                }
                let mut espectro = vec![0.0f64; m];
                if let Some(nm) = risk_engine::random_matrix::full_spectrum_flat_zero_alloc(
                    &flat, m, &mut espectro,
                ) {
                    w(format!("espectro propio casos-completos (descendente, {nm}):"));
                    for (ix, lambda) in espectro.iter().take(nm).enumerate() {
                        w(format!("  λ{:>2} = {:.6}", ix + 1, lambda));
                    }
                    if let Some(edge) = risk_engine::random_matrix::mp_upper_edge(m, t) {
                        let sobre = espectro.iter().take(nm).filter(|&&l| l > edge).count();
                        w(format!(
                            "borde Marchenko-Pastur (N={m}, T={t}): {edge:.6} ⇒ modos de SEÑAL: {sobre}"
                        ));
                    } else {
                        w("borde MP no aplicable (T ≤ N)".into());
                    }
                    if let Some(bets) = risk_engine::random_matrix::effective_bets(&corr, t) {
                        w(format!(
                            "★ APUESTAS EFECTIVAS INDEPENDIENTES: {bets:.3} de {m} fuentes"
                        ));
                    } else {
                        w("★ effective_bets: sin modos sobre el borde MP — TODO ruido muestral".into());
                    }
                    let suma_l: f64 = espectro.iter().take(nm).sum();
                    let suma_l2: f64 = espectro.iter().take(nm).map(|l| l * l).sum();
                    if suma_l > 0.0 && suma_l2 > 0.0 {
                        w(format!(
                            "participación espectral CRUDA ((Σλ)²/Σλ²): {:.3}",
                            suma_l * suma_l / suma_l2
                        ));
                    }
                } else {
                    w("⚠️ eigensolver rechazó la matriz (¿no PSD?)" .into());
                }
            }
        } else {
            w("⚠️ casos completos ≤ tamaño del core — estadística insuficiente".into());
        }

        // Caza-clones: max|corr| por fuente dentro del core (por pares, la
        // tabla de clones es válida par a par aunque la matriz global no sea
        // PSD — cada par usa SOLO sus co-observaciones).
        w("--- max |corr| por fuente (core, por pares) ---".into());
        let mut filas: Vec<(f64, String)> = Vec::new();
        for (a, &ca) in core.iter().enumerate() {
            let mut maxr = 0.0f64;
            for (b, &cb) in core.iter().enumerate() {
                if a == b {
                    continue;
                }
                let (lo, hi) = if ca < cb { (ca, cb) } else { (cb, ca) };
                if let Some((_, acc)) = pares.iter().find(|((pi, pj), _)| *pi == lo && *pj == hi) {
                    let (nn, sx, sy, sxy, sx2, sy2) = *acc;
                    let nf = nn as f64;
                    let cov = (sxy - sx * sy / nf) / (nf - 1.0);
                    let vx = (sx2 - sx * sx / nf) / (nf - 1.0);
                    let vy = (sy2 - sy * sy / nf) / (nf - 1.0);
                    if vx > 1e-18 && vy > 1e-18 {
                        let r = (cov / (vx.sqrt() * vy.sqrt())).abs();
                        if r > maxr {
                            maxr = r;
                        }
                    }
                }
            }
            filas.push((maxr, nombres[ca].to_string()));
        }
        filas.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
        for (maxr, nombre) in filas {
            w(format!("  {nombre:<30} max|corr| {maxr:.4}"));
        }
    } else {
        w("⚠️ core bien-observado < 2 fuentes".into());
    }

    let reporte = lineas.join("\n");
    eprintln!("{reporte}");

    // Recibo persistente para el informe integral (emisor: átomo siguiente).
    let salida = std::path::Path::new("outputs").join(format!(
        "ortogonalidad_voces_{}.log",
        nombre_tape.trim_end_matches(".bin")
    ));
    if std::fs::create_dir_all("outputs").is_ok() {
        let _ = std::fs::write(&salida, format!("{reporte}\n"));
        eprintln!("recibo: {}", salida.display());
    }
}

/// El corral por instante preserva NaN en no-hablantes (ausencia honesta).
#[inline(always)]
fn instantante_segura(v: [f64; SLOTS]) -> [f64; SLOTS] {
    v
}
