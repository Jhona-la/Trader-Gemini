//! T-2 — COBERTURA DE LAS PUERTAS DE DECISIÓN (QS-T2).
//!
//! # Por qué existe
//!
//! El fixture del T-1 es una tendencia determinista fuerte con 1 000 USD. Seis
//! cambios seguidos de puertas de decisión (C-22, D2, D3, QS-K, #711, #712;
//! ledger §7.4) dejaron sus 145 huellas por gen idénticas bit a bit:
//! - en tendencia todos los asientos direccionales coinciden y el consejo
//!   aprueba siempre;
//! - con 1 000 USD la rama micro (D2, D3, suelo de viabilidad) no se ejecuta;
//! - las ramas de rango del núcleo nunca llegan a sus bordes.
//!
//! El T-1 certifica que no hay regresión de parámetros y dimensionado, pero no
//! ve las puertas. Este test usa otro régimen para que sí las vea.
//!
//! # Fixture
//!
//! - Serie lateral con regímenes alternos: rango con reversión a la media,
//!   tramo alcista, rango, tramo bajista. En rango la vela previa cambia de
//!   signo a menudo, y con ella el OBI sintético del runner (±0,6): el libro
//!   contradice a las señales lentas y el consejo vota dividido.
//! - Cuenta de 13 USD: régimen micro, nocional mínimo, D2/D3 y viabilidad.
//! - Mismo predictor sintético direccional que el T-1 (base 0,5).
//!
//! # Qué mide
//!
//! Lo mismo que el T-1 (perturbación de cada gen a su extremo más lejano, en
//! paralelo, con una huella por gen) sobre este fixture. Al no ser un
//! trinquete todavía, sólo exige que el fixture tenga actividad; la lista y
//! las huellas se comparan entre árboles para ver si un cambio de puerta las
//! mueve (ledger §7.4).
//!
//! Uso: `cargo test --release -p backtest-engine --test t2_cobertura_puertas
//! -- --include-ignored --nocapture` (con `T1_THREADS` para fijar los hilos).

use backtest_engine::{STATS_LEN, run_backtest_native};
use quantum_arena::genome::SuperGenotype;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "support/t1_measurement.rs"]
mod measurement;
use measurement::{changed_slots, diagnostic_stats_line, difiere, furthest_endpoint};

/// Capital de la cuenta del dueño: régimen micro.
const CAPITAL_MICRO: f64 = 13.0;

/// Serie lateral determinista con regímenes alternos (QS-T2).
///
/// Cuatro tramos de igual longitud:
/// 1. rango: Ornstein-Uhlenbeck en log-precio alrededor de 60 000;
/// 2. tendencia alcista suave con el mismo ruido;
/// 3. rango alrededor del nivel alcanzado;
/// 4. tendencia bajista suave.
///
/// El ruido por vela (σ ≈ 0,25 %) y el rango intravela se eligen para que la
/// σ(τ) del fixture pague la fricción del roundtrip a τ de minutos, como en
/// el T-1 (si no, todo sale por el suelo de TP/SL y no se mide nada).
pub fn serie_lateral(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut closes = Vec::with_capacity(n);
    let mut highs = Vec::with_capacity(n);
    let mut lows = Vec::with_capacity(n);
    let mut vols = Vec::with_capacity(n);
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut ln_p = 60_000.0f64.ln();
    let mut ancla = ln_p;
    let tramo = (n / 4).max(1);
    for i in 0..n {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Uniforme centrada en (−0,5, 0,5): σ = 1/√12 ≈ 0,289.
        let u = ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
        let ruido = u * 0.0087; // σ ≈ 0,25 % por vela
        let fase = (i / tramo).min(3);
        let deriva = match fase {
            1 => 0.0006,
            3 => -0.0006,
            _ => 0.0,
        };
        if fase == 0 || fase == 2 {
            // Reversión a la media en log-precio (semivida ≈ 20 velas).
            if i % tramo == 0 {
                ancla = ln_p;
            }
            ln_p += -0.035 * (ln_p - ancla) + ruido;
        } else {
            ln_p += deriva + ruido;
        }
        let p = ln_p.exp();
        closes.push(p);
        highs.push(p * (1.0 + 0.0015 + u.abs() * 0.0020));
        lows.push(p * (1.0 - 0.0015 - u.abs() * 0.0020));
        vols.push(900.0 + u.abs() * 2_000.0);
    }
    (closes, highs, lows, vols)
}

fn cargar_predictor() {
    let confident = god_engine_core::ml_inference::NanoForestData {
        children_left: vec![1, -1, -1],
        children_right: vec![2, -1, -1],
        feature: vec![0, -1, -1],
        threshold: vec![0.0, 0.0, 0.0],
        value: vec![0.0, -3.0, 3.0],
        tree_offsets: vec![0, 3],
        init_score: 0.0,
    };
    let forest = god_engine_core::ml_inference::NanoForest::from_data(confident)
        .expect("forest sintético del T-2 fuera de contrato");
    god_engine_core::ml_inference::NanoForest::store_global("BTCUSDT_MOTOR", forest);
}

fn evaluar(
    cfg: &SuperGenotype,
    serie: &(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>),
) -> [f64; STATS_LEN] {
    let mut pnl = vec![0.0f64; serie.0.len().max(1)];
    let mut stats = [0.0f64; STATS_LEN];
    run_backtest_native(
        &serie.0,
        &serie.1,
        &serie.2,
        &serie.3,
        cfg,
        &mut pnl,
        &mut stats,
        "BTCUSDT",
        CAPITAL_MICRO,
    );
    stats
}

fn hilos() -> usize {
    std::env::var("T1_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&h| h >= 1)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

fn huella(stats: &[f64; STATS_LEN]) -> u64 {
    stats.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, x| {
        x.to_bits()
            .to_le_bytes()
            .iter()
            .fold(h, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3))
    })
}

/// Una sola evaluación con el genoma base: si el fixture no opera, el T-2 no
/// mide nada. Imprime las estadísticas y el reparto de rechazos del riesgo.
#[test]
fn t2_diag_actividad_del_fixture_lateral_micro() {
    cargar_predictor();
    let datos = serie_lateral(3_000);
    let base = SuperGenotype::new_baseline(0.0002, 0.0005);
    let stats = evaluar(&base, &datos);
    println!("[T2-STATS] gene=base huella={:016x}", huella(&stats));
    println!("[T2-DIAG] {}", diagnostic_stats_line(&stats));
    println!("[T2-DIAG] risk-rejects: {}", risk_engine::reject_report());
    assert!(
        stats[1] >= 10.0,
        "el fixture lateral micro casi no opera ({} cierres): no mediría las puertas",
        stats[1]
    );
}

/// Sensibilidad gen a gen sobre el fixture lateral micro (sin trinquete
/// todavía: se publica la lista y las huellas para compararlas entre árboles).
#[test]
#[ignore = "medición larga (≈ 16 min con 4 núcleos): cargo test ... -- --ignored --nocapture"]
fn t2_cobertura_de_puertas_lateral_micro() {
    cargar_predictor();
    let datos = serie_lateral(3_000);
    let base = SuperGenotype::new_baseline(0.0002, 0.0005);
    let lo = SuperGenotype::get_lower_bounds();
    let hi = SuperGenotype::get_upper_bounds();
    let base_vec = base.to_vector();
    let n = base_vec.len();

    let stats_base = evaluar(&base, &datos);
    println!("[T2-STATS] gene=base huella={:016x}", huella(&stats_base));
    println!("[T2-DIAG] base: {}", diagnostic_stats_line(&stats_base));

    let mut candidatos: Vec<(usize, SuperGenotype)> = Vec::with_capacity(n);
    for g in 0..n {
        let mut v = base_vec.clone();
        v[g] = furthest_endpoint(base_vec[g], lo[g], hi[g]);
        if (v[g] - base_vec[g]).abs() < 1e-12 {
            continue;
        }
        let mutado = SuperGenotype::from_vector(&v);
        let realized = mutado.to_vector();
        println!(
            "[T2-PROBE] gene={g} requested={} realized={} changed_slots={:?}",
            v[g],
            realized[g],
            changed_slots(&base_vec, &realized)
        );
        candidatos.push((g, mutado));
    }

    let h = hilos().min(candidatos.len().max(1));
    println!("[T-2] {} evaluaciones en {h} hilos", candidatos.len());
    let siguiente = AtomicUsize::new(0);
    let resultados: Vec<OnceLock<[f64; STATS_LEN]>> =
        (0..candidatos.len()).map(|_| OnceLock::new()).collect();
    std::thread::scope(|s| {
        for _ in 0..h {
            std::thread::Builder::new()
                .stack_size(64 << 20)
                .spawn_scoped(s, || loop {
                    let i = siguiente.fetch_add(1, Ordering::Relaxed);
                    let Some((_, cfg)) = candidatos.get(i) else { break };
                    let _ = resultados[i].set(evaluar(cfg, &datos));
                })
                .expect("no se pudo lanzar un hilo del T-2");
        }
    });

    let mut sensibles: Vec<usize> = Vec::new();
    for (i, (g, _)) in candidatos.iter().enumerate() {
        let stats = resultados[i].get().expect("evaluación del T-2 ausente");
        println!("[T2-STATS] gene={g} huella={:016x}", huella(stats));
        if difiere(&stats_base, stats) {
            sensibles.push(*g);
        }
    }
    println!(
        "\n[T-2] COBERTURA DE PUERTAS (lateral, 13 USD): {}/{n} genes sensibles ({:.1} %)\n[T-2] Sensibles: {:?}\n",
        sensibles.len(),
        100.0 * sensibles.len() as f64 / n as f64,
        sensibles
    );
    assert!(stats_base[1] >= 10.0, "el fixture base casi no opera");
}
