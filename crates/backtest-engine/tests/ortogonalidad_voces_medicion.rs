//! MEDICIÓN DE ORTOGONALIDAD ESPECTRAL DEL ENSAMBLE (GLM 146, 2026-10-21).
//!
//! El censo (GLM 129) midió QUIÉN habla; este vehículo mide si lo que dicen
//! es información INDEPENDIENTE. Sobre el tape REAL se reconstruye la matriz
//! de correlación de votos firmados por fuente (Welford online por pares,
//! sólo ticks donde AMBAS fuentes opinan — jamás se imputan ceros a mudos),
//! y sobre ella:
//!   - espectro propio completo (Jacobi zero-alloc de risk-engine),
//!   - EDGE Marchenko-Pastur: modos bajo el borde = ruido muestral,
//!   - effective_bets ((Σ√λ)²/Σλ sobre modos > edge): apuestas efectivas
//!     INDEPENDIENTES del ensamble — 19 fuentes pueden ser 3 apuestas,
//!   - por fuente: max|corr| con las demás (cazador de clones informativos
//!     que la invariancia por fuente del 127 no detecta).
//!
//! pilar «Inferencia Causal y Reducción de Multicolinealidad Espectral» del
//! mandato del operador. Reutiliza effective_bets/mp_upper_edge de
//! risk-engine (misma matemática del riesgo de cartera, aplicada a voces).
//!
//! #[ignore]: investigación local con tape REAL (walk-up compartido con
//! censo_mudos_medicion: TG_DATA_ROOT / workspace / raíz). Recibo persistente
//! en outputs/ortogonalidad_voces_*.log para el informe integral.

use backtest_engine::booktick_replay::{ReplayConfig, ReplayTick, run_booktick_replay_with_observer};

/// Pares con menos co-observaciones que esto no se reportan como medidos
/// (CL-15: ausencia explícita, jamás corr=0 inventado).
const MIN_CO_OBS: u64 = 500;
/// Muestreo del observer: 1 de cada N ticks (la correlación no necesita 2M).
const MUESTREO: usize = 4;

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

    // Estado del observador: nombres (orden de primera aparición, estable por
    // registro) y acumuladores Welford por par (sólo co-observaciones).
    let mut nombres: Vec<&'static str> = Vec::new();
    // Por par (i,j) i<j: (n, Σx, Σy, Σxy, Σx², Σy²).
    let mut pares: Vec<((usize, usize), (u64, f64, f64, f64, f64, f64))> = Vec::new();
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
            // Indexar por nombre (estable entre llamadas: mismo registro).
            for (nombre, _) in &votos {
                if !nombres.iter().any(|n| n == nombre) {
                    nombres.push(nombre);
                }
            }
            let idx = |nombre: &str| nombres.iter().position(|n| *n == nombre);
            // Pares que hablan en ESTE tick (voto con signo, |v|>1e-9).
            let hablantes: Vec<(usize, f64)> = votos
                .iter()
                .filter_map(|(nombre, v)| {
                    let v = *v;
                    (v.is_finite() && v.abs() > 1e-9)
                        .then(|| idx(nombre).map(|ix| (ix, v)))
                        .flatten()
                })
                .collect();
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

    // ── Construcción honesta de la matriz: subconjunto CLIQUE bien observado
    // (todas las parejas ≥ MIN_CO_OBS). Fuentes con pares débiles quedan
    // FUERA del espectro y se reportan como no medibles juntas (CL-15).
    let k = nombres.len();
    let mut bien: Vec<usize> = Vec::new();
    'fuente: for f in 0..k {
        for &g in &bien {
            let (lo, hi) = if f < g { (f, g) } else { (g, f) };
            let ok = pares
                .iter()
                .any(|((pi, pj), acc)| *pi == lo && *pj == hi && acc.0 >= MIN_CO_OBS);
            if !ok {
                continue 'fuente;
            }
        }
        bien.push(f);
    }
    let m = bien.len();

    let mut lineas: Vec<String> = Vec::new();
    let mut w = |s: String| lineas.push(s);
    w("=============== ORTOGONALIDAD DE VOCES (GLM 146) ===============".into());
    w(format!(
        "tape: {nombre_tape} | ticks: {} | muestras: {} | fuentes: {} | pares observados: {}",
        ticks.len(),
        muestras,
        k,
        pares.len()
    ));
    w(format!(
        "subconjunto bien-observado (pares ≥{MIN_CO_OBS}): {m} de {k} fuentes"
    ));

    if m >= 2 {
        // Matriz de correlación del clique (unidades en diagonal).
        let mut corr = vec![vec![0.0f64; m]; m];
        let mut t_min = u64::MAX;
        for (a, &fa) in bien.iter().enumerate() {
            corr[a][a] = 1.0;
            for (b, &fb) in bien.iter().enumerate().skip(a + 1) {
                let (lo, hi) = if fa < fb { (fa, fb) } else { (fb, fa) };
                let Some((_, acc)) = pares
                    .iter()
                    .find(|((pi, pj), _)| *pi == lo && *pj == hi)
                else {
                    continue; // imposible dentro del clique; defensivo
                };
                let (n, sx, sy, sxy, sx2, sy2) = *acc;
                t_min = t_min.min(n);
                let nf = n as f64;
                let cov = (sxy - sx * sy / nf) / (nf - 1.0);
                let vx = (sx2 - sx * sx / nf) / (nf - 1.0);
                let vy = (sy2 - sy * sy / nf) / (nf - 1.0);
                let r = if vx > 1e-18 && vy > 1e-18 {
                    (cov / (vx.sqrt() * vy.sqrt())).clamp(-1.0, 1.0)
                } else {
                    f64::NAN // sin varianza ⇒ no medible, no cero
                };
                corr[a][b] = r;
                corr[b][a] = r;
            }
        }
        if corr.iter().any(|fila| fila.iter().any(|v| !v.is_finite())) {
            w("⚠️ pares sin varianza dentro del clique — espectro NO calculado".into());
        } else {
            // Espectro completo (Jacobi zero-alloc, m ≤ 19 ≤ 32).
            let mut flat = Vec::with_capacity(m * m);
            for fila in &corr {
                flat.extend_from_slice(fila);
            }
            let mut espectro = vec![0.0f64; m];
            if let Some(nm) = risk_engine::random_matrix::full_spectrum_flat_zero_alloc(
                &flat, m, &mut espectro,
            ) {
                w(format!("espectro propio (descendente, {nm}):"));
                for (ix, lambda) in espectro.iter().take(nm).enumerate() {
                    w(format!("  λ{:>2} = {:.6}", ix + 1, lambda));
                }
                if let Some(edge) = risk_engine::random_matrix::mp_upper_edge(m, t_min as usize) {
                    let sobre = espectro.iter().take(nm).filter(|&&l| l > edge).count();
                    w(format!(
                        "borde Marchenko-Pastur (N={m}, T={}): {edge:.6} ⇒ modos de SEÑAL: {sobre}",
                        t_min
                    ));
                } else {
                    w("borde MP no aplicable (T ≤ N)".into());
                }
                if let Some(bets) = risk_engine::random_matrix::effective_bets(&corr, t_min as usize) {
                    w(format!(
                        "★ APUESTAS EFECTIVAS INDEPENDIENTES: {bets:.3} de {m} fuentes (participación sobre modos > borde MP)"
                    ));
                } else {
                    w("★ effective_bets: sin modos sobre el borde — TODO el ensamble es ruido muestral bajo MP".into());
                }
                let suma: f64 = espectro.iter().take(nm).sum();
                let suma2: f64 = espectro.iter().take(nm).map(|l| l * l).sum();
                if suma > 0.0 && suma2 > 0.0 {
                    w(format!(
                        "participación espectral CRUDA ((Σλ)²/Σλ², sin filtro MP): {:.3}",
                        suma * suma / suma2
                    ));
                }
            } else {
                w("⚠️ eigensolver no convergió".into());
            }
        }
        // Caza-clones: max|corr| por fuente dentro del clique.
        w("--- max |corr| por fuente (clique) ---".into());
        let mut filas: Vec<(f64, String)> = Vec::new();
        for (a, &fa) in bien.iter().enumerate() {
            let mut maxr = 0.0f64;
            let mut peor = "";
            for (b, &fb) in bien.iter().enumerate() {
                if a == b {
                    continue;
                }
                let r = corr[a][b].abs();
                if r > maxr {
                    maxr = r;
                    peor = nombres[fb];
                }
            }
            filas.push((
                maxr,
                format!("  {:<28} max|corr| {:.4} vs {}", nombres[fa], maxr, peor),
            ));
        }
        filas.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, linea) in filas {
            w(linea);
        }
    } else {
        w("⚠️ clique bien-observado < 2 fuentes — subir MIN_CO_OBS no, conseguir tapes más largos sí".into());
    }

    let reporte = lineas.join("\n");
    eprintln!("{reporte}");

    // Recibo persistente para el informe integral (emisor de ortogonalidad:
    // átomo siguiente). outputs/ existe por convención del repo.
    let salida = std::path::Path::new("outputs").join(format!(
        "ortogonalidad_voces_{}.log",
        nombre_tape.trim_end_matches(".bin")
    ));
    if std::fs::create_dir_all("outputs").is_ok() {
        let _ = std::fs::write(&salida, format!("{reporte}\n"));
        eprintln!("recibo: {}", salida.display());
    }
}
