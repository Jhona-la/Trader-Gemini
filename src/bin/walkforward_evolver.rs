//! EVOLUCIONADOR WALK-FORWARD SOBRE EL INSTRUMENTO DE REFERENCIA (D-686).
//!
//! # Por qué existe
//!
//! La verificación forense de la Décima Ola demostró que el genoma de
//! producción se seleccionó contra un motor con errores que se compensaban:
//! corregido el motor, sus genes quedaron calibrados para una física que ya no
//! existe. Los evolucionadores disponibles no servían para re-evolucionarlo:
//!
//! * `evolver` recorre `process_tick` (no el camino `process_event` de
//!   producción), inicializa islas «scalp» y «swing», usa su propia aptitud y
//!   puede promover un genoma que pierde.
//! * `evolution` evalúa con `booktick_replay`, que modela el deslizamiento de
//!   otra forma que el forense, y muta sólo quince genes, varios de ellos vistas
//!   que `derive_anchors_from_curves` sobrescribe.
//!
//! # Qué hace
//!
//! * **Evaluador único:** cada candidato se evalúa ejecutando el binario
//!   `audit_forensic_backtest` —el instrumento con el que se midió todo— sobre
//!   una ventana de ticks, con el genoma pasado por fichero y fijado para que el
//!   almacén no lo sustituya. Varias evaluaciones corren en paralelo.
//! * **Genoma completo:** la variación es `mutate_cmaes`, que recorre el vector
//!   de 144 genes y re-deriva vistas y curvas.
//! * **Aptitud única:** `evolution_engine::fitness::compute` (crecimiento
//!   logarítmico penalizado por drawdown y por degradación fuera de muestra).
//! * **Walk-forward:** la selección usa sólo la partición de entrenamiento; los
//!   finalistas y el genoma semilla se evalúan después en la de validación, que
//!   ninguna decisión anterior ha visto.
//! * **Puerta de promoción:** validación con PnL neto positivo, al menos
//!   `WF_MIN_TRADES` operaciones y aptitud combinada superior a la de la semilla
//!   medida del mismo modo. Sin puerta superada no se promueve nada.
//!
//! # Uso
//!
//! ```text
//! cargo build --release --bin audit_forensic_backtest --bin walkforward_evolver
//! target/release/walkforward_evolver
//! ```
//!
//! Variables: `WF_GENERATIONS` (8), `WF_POPULATION` (16), `WF_ELITE` (4),
//! `WF_PARALLEL` (8), `WF_TRAIN_FRACTION` (0,6), `WF_TOP_K` (4),
//! `WF_MIN_TRADES` (15, el mínimo muestral de la regla X-014), `WF_SEED_ENV`
//! (prod), `WF_DATA`, `WF_FORENSIC_BIN`, `WF_PROMOTE` (0/1: promueve al entorno
//! backtest) y `WF_PROMOTE_ENVS` (p. ej. `demo,prod`: cruces adicionales, que
//! exigen `TG_GENOME_PROMOTE_ARMED=1` y `TG_GENOME_PROMOTE_OPERATOR`).

use evolution_engine::fitness::{compute, FitnessInputs, INVIABLE};
use quantum_arena::genome::SuperGenotype;
use quantum_arena::genome_store::GenomeEnvelope;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Tamaño de `BinTick` en disco: `u64` + 4 × `f64`.
const BIN_TICK_BYTES: u64 = 40;

#[derive(Clone, Debug, Default)]
struct Eval {
    ok: bool,
    initial_capital: f64,
    final_capital: f64,
    net_roi: f64,
    trades: u32,
    net_wins: u32,
    max_drawdown: f64,
    tp: u32,
    sl: u32,
    trail: u32,
    zombie: u32,
    toxic: u32,
}

impl Eval {
    fn net_pnl(&self) -> f64 {
        self.final_capital - self.initial_capital
    }

    fn line(&self) -> String {
        if !self.ok {
            return "evaluación fallida".to_string();
        }
        format!(
            "ROI {:+6.2} % · {:>3} ops · acierto {:>4.1} % · DD {:>5.2} % · TP {} SL {} TRAIL {} ZOMBIE {}",
            self.net_roi * 100.0,
            self.trades,
            if self.trades > 0 { self.net_wins as f64 / self.trades as f64 * 100.0 } else { 0.0 },
            self.max_drawdown * 100.0,
            self.tp,
            self.sl,
            self.trail,
            self.zombie
        )
    }
}

fn env_num<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

/// Ejecuta el forense sobre `[start, start + len)` con el genoma dado.
fn evaluate(bin: &Path, data: &str, genome: &SuperGenotype, start: u64, len: u64, tag: &str, dir: &Path) -> Eval {
    let path = dir.join(format!("{tag}.json"));
    let json = match serde_json::to_string(genome) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("⚠️ {tag}: no se pudo serializar el genoma: {e}");
            return Eval::default();
        }
    };
    if let Err(e) = std::fs::write(&path, json) {
        eprintln!("⚠️ {tag}: no se pudo escribir {}: {e}", path.display());
        return Eval::default();
    }
    let output = Command::new(bin)
        .env("FORENSIC_GENOME_PATH", &path)
        .env("FORENSIC_START_TICK", start.to_string())
        .env("MAX_TICKS", len.to_string())
        .env("FORENSIC_DATA_PATH", data)
        .env("TG_GENOME_ENV", "backtest")
        .output();
    let _ = std::fs::remove_file(&path);
    let output = match output {
        Ok(o) => o,
        Err(e) => {
            eprintln!("⚠️ {tag}: no se pudo lanzar {}: {e}", bin.display());
            return Eval::default();
        }
    };
    let stdout = String::from_utf8_lossy(&output.stdout);
    let Some(line) = stdout.lines().rev().find(|l| l.starts_with("FORENSIC_JSON ")) else {
        eprintln!(
            "⚠️ {tag}: la corrida no emitió FORENSIC_JSON (código {:?})",
            output.status.code()
        );
        return Eval::default();
    };
    let value: serde_json::Value = match serde_json::from_str(&line["FORENSIC_JSON ".len()..]) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("⚠️ {tag}: FORENSIC_JSON ilegible: {e}");
            return Eval::default();
        }
    };
    let num = |k: &str| value.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0);
    Eval {
        ok: true,
        initial_capital: num("initial_capital"),
        final_capital: num("final_capital"),
        net_roi: num("net_roi"),
        trades: num("trades") as u32,
        net_wins: num("net_wins") as u32,
        max_drawdown: num("max_drawdown"),
        tp: num("tp") as u32,
        sl: num("sl") as u32,
        trail: num("trail") as u32,
        zombie: num("zombie") as u32,
        toxic: num("toxic") as u32,
    }
}

/// Aptitud única. Con `oos`, la degradación fuera de muestra amplifica el
/// castigo y atenúa el premio.
fn fitness(train: &Eval, min_trades: u32, oos: Option<&Eval>) -> f64 {
    if !train.ok {
        return INVIABLE;
    }
    let (oos_start, oos_end) = match oos {
        Some(o) if o.ok => (o.initial_capital, o.final_capital),
        Some(_) => return INVIABLE,
        None => (train.initial_capital, train.initial_capital),
    };
    compute(&FitnessInputs {
        initial_capital: train.initial_capital,
        final_capital: train.final_capital,
        max_drawdown_pct: train.max_drawdown,
        total_trades: train.trades,
        min_trades_required: min_trades,
        oos_start_capital: oos_start,
        oos_end_capital: oos_end,
    })
}

fn fmt_fit(f: f64) -> String {
    if f.is_finite() {
        format!("{f:+.4}")
    } else {
        "inviable".to_string()
    }
}

fn main() {
    let generations: usize = env_num("WF_GENERATIONS", 8usize).max(1);
    let population: usize = env_num("WF_POPULATION", 16usize).max(2);
    let elite: usize = env_num("WF_ELITE", 4usize).clamp(1, population - 1);
    let parallel: usize = env_num("WF_PARALLEL", 8usize).max(1);
    let train_fraction: f64 = env_num("WF_TRAIN_FRACTION", 0.6f64).clamp(0.2, 0.9);
    let top_k: usize = env_num("WF_TOP_K", 4usize).max(1);
    let min_trades: u32 = env_num("WF_MIN_TRADES", 15u32);
    let data = std::env::var("WF_DATA").unwrap_or_else(|_| "data/BTCUSDT_ticks.bin".to_string());
    let bin = PathBuf::from(std::env::var("WF_FORENSIC_BIN").unwrap_or_else(|_| {
        format!("target/release/audit_forensic_backtest{}", std::env::consts::EXE_SUFFIX)
    }));

    let total_ticks = std::fs::metadata(&data).map(|m| m.len() / BIN_TICK_BYTES).unwrap_or(0);
    if total_ticks < 10_000 {
        eprintln!("❌ {data}: {total_ticks} ticks; se necesitan al menos 10 000.");
        std::process::exit(1);
    }
    if !bin.exists() {
        eprintln!(
            "❌ No existe {}. Compílalo con: cargo build --release --bin audit_forensic_backtest",
            bin.display()
        );
        std::process::exit(1);
    }
    let train_len = (total_ticks as f64 * train_fraction) as u64;
    let val_start = train_len;
    let val_len = total_ticks - train_len;

    let dir = std::env::temp_dir().join(format!("tg_walkforward_{}", std::process::id()));
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("❌ No se pudo crear {}: {e}", dir.display());
        std::process::exit(1);
    }
    let pool = match rayon::ThreadPoolBuilder::new().num_threads(parallel).build() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("❌ No se pudo crear el pool de evaluación: {e}");
            std::process::exit(1);
        }
    };

    let seed_env = std::env::var("WF_SEED_ENV").unwrap_or_else(|_| "prod".to_string());
    std::env::set_var("TG_GENOME_ENV", &seed_env);
    let seed = SuperGenotype::load_or_default();

    println!("════════════════════════════════════════════════════════════════════");
    println!("🧬 EVOLUCIONADOR WALK-FORWARD · evaluador: {}", bin.display());
    println!(
        "   datos {data} · {total_ticks} ticks · entrenamiento [0, {train_len}) · validación [{val_start}, {total_ticks})"
    );
    println!(
        "   {generations} generaciones · población {population} · élite {elite} · {parallel} en paralelo · semilla: entorno {seed_env}"
    );
    println!("════════════════════════════════════════════════════════════════════");

    let mut candidates: Vec<SuperGenotype> = vec![seed.clone(), SuperGenotype::new_baseline(0.0002, 0.0005)];
    while candidates.len() < population {
        candidates.push(seed.mutate_cmaes(0.20));
    }

    let mut ranked: Vec<(SuperGenotype, Eval, f64)> = Vec::new();
    let mut seed_train = Eval::default();

    for gen in 1..=generations {
        let progress = if generations > 1 { (gen - 1) as f64 / (generations - 1) as f64 } else { 1.0 };
        let rate = 0.20 + (0.04 - 0.20) * progress;
        let t0 = std::time::Instant::now();
        let batch: Vec<(usize, SuperGenotype)> = candidates.iter().cloned().enumerate().collect();
        let results: Vec<(SuperGenotype, Eval)> = pool.install(|| {
            batch
                .par_iter()
                .map(|(i, g)| {
                    let e = evaluate(&bin, &data, g, 0, train_len, &format!("g{gen}_c{i}"), &dir);
                    (g.clone(), e)
                })
                .collect()
        });
        if gen == 1 {
            seed_train = results[0].1.clone();
        }
        for (g, e) in results {
            let f = fitness(&e, min_trades, None);
            ranked.push((g, e, f));
        }
        ranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(elite.max(top_k));

        let best = &ranked[0];
        println!(
            "G{gen}/{generations} · mutación {rate:.3} · {:.0} s · mejor aptitud {} · {}",
            t0.elapsed().as_secs_f64(),
            fmt_fit(best.2),
            best.1.line()
        );
        if gen == generations {
            break;
        }
        let parents = elite.min(ranked.len());
        candidates = (0..population).map(|i| ranked[i % parents].0.mutate_cmaes(rate)).collect();
    }

    // ── Validación: finalistas y semilla sobre datos que la selección no vio ──
    let finalists: Vec<(SuperGenotype, Eval, f64)> = ranked.into_iter().take(top_k).collect();
    let mut to_validate: Vec<SuperGenotype> = finalists.iter().map(|f| f.0.clone()).collect();
    to_validate.push(seed.clone());
    let validation: Vec<Eval> = pool.install(|| {
        to_validate
            .par_iter()
            .enumerate()
            .map(|(i, g)| evaluate(&bin, &data, g, val_start, val_len, &format!("val_{i}"), &dir))
            .collect()
    });
    let seed_val = validation.last().cloned().unwrap_or_default();
    let seed_combined = fitness(&seed_train, min_trades, Some(&seed_val));

    println!("────────────────────────────────────────────────────────────────────");
    println!("VALIDACIÓN FUERA DE MUESTRA");
    println!("  semilla   · entrenamiento: {}", seed_train.line());
    println!("            · validación:    {}", seed_val.line());
    println!("            · aptitud combinada {}", fmt_fit(seed_combined));

    let mut champion: Option<(usize, f64)> = None;
    for (i, (_, train, _)) in finalists.iter().enumerate() {
        let val = &validation[i];
        let combined = fitness(train, min_trades, Some(val));
        println!("  finalista {i} · entrenamiento: {}", train.line());
        println!("            · validación:    {}", val.line());
        println!("            · aptitud combinada {}", fmt_fit(combined));
        if combined.is_finite() && champion.map_or(true, |(_, c)| combined > c) {
            champion = Some((i, combined));
        }
    }

    let decision = match champion {
        None => Err("ningún finalista es viable en validación".to_string()),
        Some((i, combined)) => {
            let val = &validation[i];
            if !val.ok || val.net_pnl() <= 0.0 {
                Err(format!("el mejor finalista pierde en validación ({:+.4} $)", val.net_pnl()))
            } else if val.trades < min_trades {
                Err(format!("el mejor finalista opera {} veces en validación (< {min_trades})", val.trades))
            } else if seed_combined.is_finite() && combined <= seed_combined {
                Err(format!(
                    "el mejor finalista no supera a la semilla ({} ≤ {})",
                    fmt_fit(combined),
                    fmt_fit(seed_combined)
                ))
            } else {
                Ok((i, combined))
            }
        }
    };

    println!("────────────────────────────────────────────────────────────────────");
    match decision {
        Err(motivo) => {
            println!("🛡️ NO SE PROMUEVE: {motivo}.");
        }
        Ok((i, combined)) => {
            let champ = &finalists[i];
            let val = &validation[i];
            let reason = format!(
                "walk-forward: entrenamiento ROI {:+.2} % ({} ops, DD {:.2} %) · validación ROI {:+.2} % ({} ops, DD {:.2} %) · aptitud {:+.4} frente a semilla {}",
                champ.1.net_roi * 100.0,
                champ.1.trades,
                champ.1.max_drawdown * 100.0,
                val.net_roi * 100.0,
                val.trades,
                val.max_drawdown * 100.0,
                combined,
                fmt_fit(seed_combined)
            );
            println!("✅ PUERTA SUPERADA · finalista {i} · {reason}");
            if env_num("WF_PROMOTE", 0u8) == 1 {
                std::env::set_var("TG_GENOME_ENV", "backtest");
                match GenomeEnvelope::promote(champ.0.clone(), "walkforward_evolver", &reason) {
                    Ok(env) => println!("🧬 Promovido a backtest: generación {}.", env.generation),
                    Err(e) => {
                        println!("❌ El almacén rechazó la promoción a backtest: {e}");
                        let _ = std::fs::remove_dir_all(&dir);
                        return;
                    }
                }
                let targets = std::env::var("WF_PROMOTE_ENVS").unwrap_or_default();
                for target in targets.split(',').map(str::trim).filter(|t| !t.is_empty()) {
                    std::env::set_var("TG_GENOME_ENV", target);
                    match GenomeEnvelope::promote_across_env("backtest", &reason) {
                        Ok(env) => println!("🧬 Promovido a {target}: generación {}.", env.generation),
                        Err(e) => println!("❌ Promoción a {target} rechazada: {e}"),
                    }
                }
            } else {
                println!("ℹ️ WF_PROMOTE no vale 1: el campeón no se escribe en el almacén.");
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
