//! MEDICIÓN DE LOS OBSERVABLES KOOPMAN EN VIVO (GLM 150, continuación del
//! 149). El motor (§57) está cableado en `StatefulEngine` (análisis %64);
//! este vehículo lo mide sobre el tape REAL para dar al consejo la
//! evidencia con la que decidir su consumo (voz 20 / gates / horizonte de
//! predictibilidad):
//!   - distribución de |λ| dominante (¿el retorno BTC es amortiguado <1 o
//!     expansivo >1 en cada ventana?),
//!   - ángulo dominante vs el bin dominante del FFT (¿la frecuencia oculta
//!     de Koopman coincide con el espectro?),
//!   - ventanas con bifurcación (máx|λ| ≥ 1 — régimen amplificador).
//!
//! #[ignore]: investigación local con tape REAL (walk-up compartido).
//! Recibo persistente outputs/koopman_observables_*.log.

use backtest_engine::booktick_replay::{ReplayConfig, ReplayTick, run_booktick_replay_with_observer};

/// Muestreo del observer: 1 de cada N ticks.
const MUESTREO: usize = 4;

#[test]
#[ignore = "investigación local: requiere tape REAL de 239 MB en data/"]
fn koopman_observables_sobre_tape_btc_sep() {
    let modelos = god_engine_core::model_reload::preload_models_from_disk();
    println!("[LG-23] modelos cargados de disco: {modelos}");
    let nombre_tape = "BTCUSDT_2026-09-14_REAL.bin";
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    // TG_DATA_ROOT es la CONVENCIÓN de DIRECTORIO (lección 136; bug latente
    // hallado en GLM 155): el tape se resuelve DENTRO con join(nombre) y el
    // candidato exige is_file() — el env crudo pasa exists() como directorio
    // y el open niega con un Access Denied críptico.
    let candidatos = [
        std::env::var("TG_DATA_ROOT")
            .map(|root| std::path::PathBuf::from(root).join(nombre_tape))
            .ok(),
        Some(manifest.join("../..").join("data").join(nombre_tape)),
        Some(manifest.join("../../..").join("data").join(nombre_tape)),
    ];
    let Some(path) = candidatos.into_iter().flatten().find(|p| p.is_file()) else {
        eprintln!("[koopman] tape ausente — skip");
        return;
    };
    let raw = backtest_engine::tick_replayer::load_binary_ticks_prefix(&path, 0, 2_000_000)
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

    // Acumuladores de los observables PUBLICADOS (análisis cada 64 ticks).
    let mut modulos: Vec<f32> = Vec::new();
    let mut angulos: Vec<f32> = Vec::new();
    let mut bifurcaciones: Vec<f32> = Vec::new();
    let mut fft_bins: Vec<f32> = Vec::new();
    let mut parejas: usize = 0;
    let mut acuerdos_05bin: usize = 0;

    let _stats = run_booktick_replay_with_observer(
        &ticks,
        &genome,
        None,
        &cfg,
        |i, core| {
            if i % MUESTREO != 0 {
                return;
            }
            let Some(fe) = core.feature_engines.first() else {
                return;
            };
            let m = fe.koopman_modulo;
            if !m.is_finite() {
                return; // warmup o degenerado — ausencia honesta
            }
            modulos.push(m);
            let a = fe.koopman_angulo;
            if a.is_finite() {
                angulos.push(a);
            }
            let b = fe.koopman_bifurcacion;
            if b.is_finite() {
                bifurcaciones.push(b);
            }
            // Comparativa de frecuencias: ángulo Koopman (rad/muestra de la
            // ventana de retornos del tick) vs bin dominante del FFT-64.
            // El bin del FFT cubre 2π/64 rad por bin; sólo comparables tras
            // normalizar ambos al mismo reloj (ambos consumen norm_return
            // por tick) — ángulo_en_bins = angulo / (2π/64).
            let bin_kpm = a as f64 / (2.0 * std::f64::consts::PI / 64.0);
            let bin_fft = fe.spectral_bin as f64;
            if bin_fft > 0.0 {
                fft_bins.push(bin_fft as f32);
                parejas += 1;
                if (bin_kpm - bin_fft).abs() <= 0.5 {
                    acuerdos_05bin += 1;
                }
            }
        },
    );

    let mut lineas: Vec<String> = Vec::new();
    let mut w = |s: String| lineas.push(s);
    w("================ OBSERVABLES KOOPMAN EN VIVO (GLM 150) ================".into());
    w(format!(
        "tape: {nombre_tape} | ticks: {} | muestras con |λ| finito: {}",
        ticks.len(),
        modulos.len()
    ));
    if modulos.is_empty() {
        w("⚠️ ningún observable publicado — revisar wiring %64".into());
    } else {
        let mut ms = modulos.clone();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cuantil = |q: f32| -> f32 {
            let ix = ((ms.len() as f32 - 1.0) * q).round() as usize;
            ms[ix.min(ms.len() - 1)]
        };
        let media = ms.iter().sum::<f32>() / ms.len() as f32;
        w(format!(
            "|λ| dominante: p05 {:.4} · p25 {:.4} · p50 {:.4} · p75 {:.4} · p95 {:.4} · media {:.4}",
            cuantil(0.05),
            cuantil(0.25),
            cuantil(0.50),
            cuantil(0.75),
            cuantil(0.95),
            media
        ));
        w(format!(
            "⇒ el retorno BTC en este tape es {} (mediana |λ| {:.4} {} 1.0)",
            if cuantil(0.50) < 1.0 { "AMORTIGUADO" } else { "EXPANSIVO" },
            cuantil(0.50),
            if cuantil(0.50) < 1.0 { "<" } else { "≥" }
        ));
        if !bifurcaciones.is_empty() {
            let sobre1 = bifurcaciones.iter().filter(|&&b| b >= 1.0).count();
            let pct = 100.0 * sobre1 as f64 / bifurcaciones.len() as f64;
            w(format!(
                "ventanas con BIFURCACIÓN (máx|λ|≥1): {sobre1} de {} ({pct:.2}%) — régimen amplificador",
                bifurcaciones.len()
            ));
        }
        if !angulos.is_empty() {
            let mut as_ = angulos.clone();
            as_.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let ix = (as_.len() as f32 - 1.0) * 0.5;
            w(format!(
                "ángulo dominante (rad/muestra): mediana {:.4} ⇒ periodo ≈ {:.1} muestras",
                as_[ix.round() as usize],
                2.0 * std::f64::consts::PI / as_[ix.round() as usize].max(1e-6) as f64
            ));
        }
        if parejas > 0 {
            let pct = 100.0 * acuerdos_05bin as f64 / parejas as f64;
            w(format!(
                "frecuencia Koopman vs bin FFT-64: {acuerdos_05bin}/{parejas} acuerdos ±0.5 bin ({pct:.1}%)"
            ));
        }
    }
    w("======================================================================".into());
    let reporte = lineas.join("\n");
    eprintln!("{reporte}");

    let salida = std::path::Path::new("outputs").join(format!(
        "koopman_observables_{}.log",
        nombre_tape.trim_end_matches(".bin")
    ));
    if std::fs::create_dir_all("outputs").is_ok() {
        let _ = std::fs::write(&salida, format!("{reporte}\n"));
        eprintln!("recibo: {}", salida.display());
    }
}
