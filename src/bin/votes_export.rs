//! Export de VOTOS de investigación (LXXXII — ADR-0010, L2 fase 1).
//!
//! Pregunta servible: ¿una agregación APRENDIDA de los votos espectrales
//! supera a la modulación fija del consenso fuera de muestra? Este bin
//! produce el DATASET: conduce el GodEngineCore REAL (mismo replay de
//! backtest) sobre un tape y, en cada punto de la rejilla, vuelca la
//! telemetría de votos del registro por moneda (sombra_*_consenso,
//! consenso_espectral_*, régimen p∈Δ³) + el retorno forward CRUDO como
//! fuente de etiqueta (el umbral de decisivas se aplica en la fase 2,
//! no se hornea aquí).
//!
//! Honestidad del dataset: las FEATURES son estrictamente point-in-time
//! (registro leído ANTES del evento que cruza el punto de rejilla);
//! r_fwd_{h} es la ÚNICA columna con información futura y existe sólo
//! como fuente de etiqueta (documentado en el manifiesto).
//!
//! Uso: votes_export <SIMBOLO> --in <tape> --out <jsonl>
//!      [--stride-ms 15000] [--horizon-ms 300000] [--max-rows 200000]
//! Genoma: el ACTIVO (config_dir/genotypes/active_genome.json); sin él,
//! SuperGenotype::default() y el manifiesto lo declara.

use backtest_engine::booktick_replay::{
    run_booktick_replay_with_observer, ReplayConfig, ReplayTick,
};
use quantum_arena::genome::SuperGenotype;
use quantum_arena::genome_store::GenomeEnvelope;
use serde_json::json;
use std::io::Write;

/// Registro TGMTICK1: header 8B + 40B (ts u64 ms + bid + ask + bq + aq).
struct Tick {
    ts_ms: u64,
    bid: f64,
    ask: f64,
}

fn cargar_tape(ruta: &str) -> Result<Vec<Tick>, String> {
    let bytes = std::fs::read(ruta).map_err(|e| format!("leer {ruta}: {e}"))?;
    const REG: usize = 40;
    if bytes.len() <= 8 || &bytes[..8] != b"TGMTICK1" || (bytes.len() - 8) % REG != 0 {
        return Err("tape no TGMTICK1 válido".into());
    }
    let n = (bytes.len() - 8) / REG;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let o = 8 + i * REG;
        let mut b8 = [0u8; 8];
        b8.copy_from_slice(&bytes[o..o + 8]);
        let ts_ms = u64::from_le_bytes(b8);
        b8.copy_from_slice(&bytes[o + 8..o + 16]);
        let bid = f64::from_le_bytes(b8);
        b8.copy_from_slice(&bytes[o + 16..o + 24]);
        let ask = f64::from_le_bytes(b8);
        if !bid.is_finite() || !ask.is_finite() || bid <= 0.0 || ask <= 0.0 {
            return Err(format!("tick {i}: precios inválidos"));
        }
        if out.last().is_some_and(|t: &Tick| ts_ms < t.ts_ms) {
            return Err("tape desordenado".into());
        }
        out.push(Tick { ts_ms, bid, ask });
    }
    Ok(out)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut simbolo = String::new();
    let mut input = String::new();
    let mut output = String::new();
    let mut stride_ms: u64 = 15_000;
    let mut horizon_ms: u64 = 300_000;
    let mut max_rows: usize = 200_000;
    // LXXXII fase 2: etiqueta al horizonte = consenso_espectral_tau de CADA
    // muestra (la escala propia del voto), no un horizonte fijo.
    let mut tau_matched = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--in" => input = args.get(i + 1).expect("--in <ruta>").clone(),
            "--out" => output = args.get(i + 1).expect("--out <ruta>").clone(),
            "--stride-ms" => stride_ms = args.get(i + 1).expect("--stride-ms N").parse().unwrap(),
            "--horizon-ms" => horizon_ms = args.get(i + 1).expect("--horizon-ms N").parse().unwrap(),
            "--max-rows" => max_rows = args.get(i + 1).expect("--max-rows N").parse().unwrap(),
            "--tau-matched" => {
                tau_matched = true;
                i += 1;
                continue;
            }
            other if !other.starts_with("--") => {
                simbolo = other.to_string();
                i += 1;
                continue;
            }
            other => panic!("argumento desconocido {other}"),
        }
        i += 2;
    }
    assert!(!simbolo.is_empty() && !input.is_empty() && !output.is_empty(),
        "uso: votes_export <SIMBOLO> --in <tape> --out <jsonl> [--stride-ms] [--horizon-ms] [--max-rows]");

    let ticks = cargar_tape(&input).expect("tape");
    println!("✓ {simbolo}: {} ticks de {}", ticks.len(), input);

    // Genoma activo si existe (los votos dependen de parámetros evolucionados)
    let (genome, genoma_fuente) = match GenomeEnvelope::load_active() {
        Some(env) => (env.genome, format!("activo g{}", env.generation)),
        None => (SuperGenotype::default(), "default (SIN activo)".to_string()),
    };
    println!("🧬 genoma: {genoma_fuente}");

    let replay_ticks: Vec<ReplayTick> = ticks
        .iter()
        .map(|t| ReplayTick {
            ts_ms: t.ts_ms,
            bid: t.bid,
            ask: t.ask,
            bid_qty: 1.0,
            ask_qty: 1.0,
        })
        .collect();
    let cfg = ReplayConfig::default();

    // Puntos de rejilla + r_fwd por punto (fuente de etiqueta, fase 2 aplica umbral)
    let t0 = ticks.first().unwrap().ts_ms;
    let tf = ticks.last().unwrap().ts_ms;
    let mut grid: Vec<u64> = Vec::new();
    let mut g = ((t0 + stride_ms - 1) / stride_ms) * stride_ms;
    let margen = if tau_matched { 43_200_000 } else { horizon_ms };
    while g <= tf.saturating_sub(margen) && grid.len() < max_rows {
        grid.push(g);
        g += stride_ms;
    }
    println!("rejilla: {} puntos (stride {stride_ms}ms, horizonte {horizon_ms}ms)", grid.len());

    let mid_en = |ts: u64| -> Option<f64> {
        // último tick con ts' <= t (búsqueda binaria derecha)
        let idx = ticks.partition_point(|t| t.ts_ms <= ts);
        idx.checked_sub(1).map(|p| (ticks[p].bid + ticks[p].ask) * 0.5)
    };

    let file = std::fs::File::create(&output).expect("crear salida");
    let mut out = std::io::BufWriter::new(file);
    let manifiesto = json!({
        "kind": "votes_dataset", "schema": "tgm.l2_votes.v1", "research_only": true,
        "symbol": simbolo, "input": input, "stride_ms": stride_ms,
        "horizon_ms": horizon_ms, "tau_matched": tau_matched, "genome": genoma_fuente,
        "features_point_in_time": true,
        "r_fwd_es_fuente_de_etiqueta": "el umbral de decisivas se aplica en fase 2",
        "cols": ["ts","sombra_osc","sombra_res","sombra_coax","sombra_trend","sombra_entropia",
                 "consenso_dom","consenso_media","consenso_tau",
                 "p_range","p_bull","p_crash","p_chaos","mid","r_fwd"]
    });
    serde_json::to_writer(&mut out, &manifiesto).unwrap();
    out.write_all(b"\n").unwrap();

    let mut fila = 0usize;
    let mut proximo_grid = 0usize; // índice en grid
    {
        let grid_ref = &grid;
        let filas_ref = &mut fila;
        let pg_ref = &mut proximo_grid;
        let stats = run_booktick_replay_with_observer(
            &replay_ticks, &genome, None, &cfg,
            |idx, core| {
                // estado ANTES del evento idx: si el tick por llegar cruza el
                // punto de rejilla, el registro contiene el estado en T.
                while *pg_ref < grid_ref.len()
                    && replay_ticks[idx].ts_ms >= grid_ref[*pg_ref]
                {
                    let t_grid = grid_ref[*pg_ref];
                    let reg = &core.arena.registry;
                    let v = |k: &str| reg.get_for_coin_or(0, k, f64::NAN);
                    let mid = mid_en(t_grid).unwrap_or(f64::NAN);
                    // fase 2: horizonte de etiqueta = tau del consenso de ESTA
                    // fila (ms, clamp a [1s, 12h]); sin tau finita se cae a
                    // horizon_ms y la fila queda marcada tau_ms=fija.
                    let tau_ms = if tau_matched {
                        let t = v("consenso_espectral_tau");
                        if t.is_finite() && t >= 1.0 {
                            (t as u64).clamp(1_000, 43_200_000)
                        } else {
                            horizon_ms
                        }
                    } else {
                        horizon_ms
                    };
                    let fwd = mid_en(t_grid + tau_ms).unwrap_or(f64::NAN);
                    let r_fwd = if mid.is_finite() && fwd.is_finite() && mid > 0.0 {
                        fwd / mid - 1.0
                    } else {
                        f64::NAN
                    };
                    let fila_json = json!({
                        "ts": t_grid,
                        "sombra_osc": v("sombra_osc_consenso"),
                        "sombra_res": v("sombra_res_consenso"),
                        "sombra_coax": v("sombra_coax_consenso"),
                        "sombra_trend": v("sombra_trend_consenso"),
                        "sombra_entropia": v("sombra_entropia_consenso"),
                        "consenso_dom": v("consenso_espectral_dominante"),
                        "consenso_media": v("consenso_espectral_media"),
                        "consenso_tau": v("consenso_espectral_tau"),
                        "p_range": core.arena.regime_p_range.load(std::sync::atomic::Ordering::Relaxed),
                        "p_bull": core.arena.regime_p_bull.load(std::sync::atomic::Ordering::Relaxed),
                        "p_crash": core.arena.regime_p_crash.load(std::sync::atomic::Ordering::Relaxed),
                        "p_chaos": core.arena.regime_p_chaos.load(std::sync::atomic::Ordering::Relaxed),
                        "mid": mid,
                        "tau_ms": tau_ms,
                        "r_fwd": r_fwd,
                    });
                    serde_json::to_writer(&mut out, &fila_json).unwrap();
                    out.write_all(b"\n").unwrap();
                    *filas_ref += 1;
                    *pg_ref += 1;
                }
            },
        );
        let _ = stats;
    }
    out.flush().unwrap();
    println!("✅ {output}: {fila} filas de votos (rejilla consumida: {proximo_grid}/{})", grid.len());
}
