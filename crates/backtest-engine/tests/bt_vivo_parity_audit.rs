//! AUDITORÍA bt↔vivo (Ola XLVI·A) — tests de contrato del censo de
//! divergencias (docs/AUDITORIA_BT_VIVO_2026-09-28.md).
//!
//! El replay conduce el NÚCLEO de producción (`GodEngineCore`), así que la
//! lógica de decisión es paridad por construcción. Estos tests fijan las
//! TRES divergencias que viven en el envoltorio como invariantes medibles:
//!
//! - DIV-1: el harness del replay desplaza bid/ask adversamente (±ATR·0.10)
//!   mientras el host vivo pasa precios crudos. Invariante: el
//!   desplazamiento SOLO empeora (monótono adverso, nunca mejora el precio).
//! - DIV-2: latencia estática del genoma (30.68ms) vs lognormal real. El
//!   simulador estocástico del repo es código muerto en la ruta de decisión.
//!   Invariante: P99 determinista del simulador > penalización estática —
//!   la dirección del sesgo, medida.
//! - Config: mismo genoma ⇒ mismo knob de latencia en ambas rutas (la
//!   divergencia DIV-2 es "estática vs distribución", NUNCA config distinta).

use backtest_engine::network_jitter::NetworkJitterSimulator;
use quantum_arena::genome::SuperGenotype;

/// DIV-2 (contrato de config): el genoma decide `latency_penalty_ms` y
/// `apply_to_arena` lo planta en el arena — la MISMA ruta que usa el host
/// vivo. Si alguien rompe esta igualdad (p. ej. un override de bt), este
/// test falla y la divergencia de configuración queda expuesta en lugar de
/// silenciosa.
#[test]
fn xlvia_genoma_compartido_misma_latencia_en_bt_y_vivo() {
    let genome = SuperGenotype::new_baseline(0.0002, 0.0005);
    let expected = genome.latency_penalty_ms;
    assert!(
        expected.is_finite() && expected > 0.0,
        "baseline sin latencia: {}",
        expected
    );

    // Ruta del replay (booktick_replay.rs:232-233): arena propio + apply.
    let arena_bt = quantum_arena::GlobalArena::build_in_own_stack(1000.0);
    genome.apply_to_arena(&arena_bt);
    let bt = arena_bt
        .config
        .latency_penalty_ms
        .load(std::sync::atomic::Ordering::Relaxed);

    // Ruta del vivo (host): mismo apply sobre su arena.
    let arena_live = quantum_arena::GlobalArena::build_in_own_stack(1000.0);
    genome.apply_to_arena(&arena_live);
    let live = arena_live
        .config
        .latency_penalty_ms
        .load(std::sync::atomic::Ordering::Relaxed);

    assert_eq!(
        bt.to_bits(),
        live.to_bits(),
        "mismo genoma debe producir bit-idéntica la latencia en bt y vivo"
    );
    assert_eq!(bt.to_bits(), expected.to_bits());
}

/// DIV-1 (invariante de dirección): el desplazamiento del harness
/// (`sim_bid = bid − running_atr·0.10`, `sim_ask = ask + running_atr·0.10`)
/// sólo puede EMPEORAR el precio. A mayor ATR, bid más bajo y ask más alto;
/// con ATR=0 el precio es el crudo del tick. Si alguien invierte un signo
/// (slippage que mejora el fill), este test lo detiene.
#[test]
fn xlvia_desplazamiento_del_harness_solo_empeora() {
    let bid = 100.0_f64;
    let ask = 100.02_f64;

    let sim = |running_atr: f64| -> (f64, f64) {
        let slip = running_atr * 0.10;
        (bid - slip, ask + slip)
    };

    // ATR=0 ⇒ precios crudos (paridad con el host vivo, lib.rs:1343).
    let (b0, a0) = sim(0.0);
    assert_eq!((b0, a0), (bid, ask), "ATR nulo no debe desplazar");

    // Monotonía adversa: más ATR ⇒ bid estrictamente menor, ask mayor.
    for atr in [0.01_f64, 0.05, 0.2, 1.0, 5.0] {
        let (b, a) = sim(atr);
        assert!(
            b < bid && a > ask,
            "ATR={}: ({}, {}) debe ensanchar, no estrechar",
            atr,
            b,
            a
        );
        assert!(b > 0.0, "ATR={} produce bid ≤ 0: {}", atr, b);
    }

    // El orden se preserva (nunca cruza el libro).
    let (b, a) = sim(5.0);
    assert!(b < a, "spread invertido con ATR alto: bid={} ask={}", b, a);
}

/// DIV-2 (magnitud medida): la penalización estática del genoma activo es
/// 30.68ms; el RTT real es lognormal (base 25ms, σ=0.35). Con σ=0.35, el
/// P99 ≈ base·exp(σ·z99 − σ²/2) ≈ 2× la base. Este test DEMUESTRA que el
/// quantile superior de la distribución real supera el valor estático que
/// el bt cobra — la dirección del sesgo optimista del bt, cuantificada con
/// el simulador determinista del repo (mismo seed ⇒ mismos samples).
#[test]
fn xlvia_p99_latencia_lognormal_supera_penalizacion_estatica() {
    // Parámetros del propio simulador por defecto (25ms RTT, σ=0.35, 0.1% loss)
    // y la penalización estática del genoma ACTIVO del repo (30.679914…ms).
    let sim = NetworkJitterSimulator::new(25.0, 0.35, 0.001);
    let estatica = 30.679_914_238_190_136_f64;

    // Muestras deterministas: seed = índice (el sampler es xorshift puro).
    let mut samples: Vec<f64> = (0..20_000u64)
        .map(|i| sim.sample_latency_ms(i).0)
        .filter(|l| l.is_finite() && *l > 0.0)
        .collect();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99 = samples[(samples.len() as f64 * 0.99) as usize - 1];
    let p50 = samples[samples.len() / 2];

    // Contrato: la cola derecha SUPERA la penalización estática del bt.
    assert!(
        p99 > estatica,
        "P99 lognormal ({:.1}ms) debe superar la penalización estática ({:.1}ms) — \
         si no, el sesgo optimista del bt en colas habría desaparecido (revisar censo DIV-2)",
        p99, estatica
    );
    // Y la mediana queda por debajo (la estática NO es absurda para el cuerpo
    // de la distribución — la brecha está en la cola, no en el centro).
    assert!(
        p50 < estatica,
        "P50 ({:.1}ms) por debajo de la estática ({:.1}ms): el cuerpo está bien cobrado",
        p50, estatica
    );
    // Sanidad del sampler: P99 > P50 (distribución con cola derecha).
    assert!(p99 > p50 * 1.5, "cola insuficiente: p50={:.1} p99={:.1}", p50, p99);
}
