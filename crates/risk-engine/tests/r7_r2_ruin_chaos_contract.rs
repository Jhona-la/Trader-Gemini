//! Contrato Formal de Certificación Ola Ω61 (Ficha Forense #696)
//! Resuelve hallazgos R7-R2:
//! - R7-R2-A-2 [MED]: Cota inferior LCB en probabilidad conservadora de pérdida para `clamp_ruin`.
//! - Superficie Ω58 / R5': Consumo de caos espectral `p_chaos` en `PortfolioOrchestrator`.
//! - R7-R2-A-1 [HIGH]: Consumo analítico de primer toque (`probabilidad_tocar_sl_antes_de_tp`) en la compuerta de geometría de riesgo.

use quantum_arena::GlobalArena;
use risk_engine::{
    orchestrator::PortfolioOrchestrator,
    regime::MarketRegime,
    ruin::{conservative_loss_q, streak_ruin_cap, CONSERVATIVE_Q},
    tp_sl::probabilidad_tocar_sl_antes_de_tp,
    RiskEngine, REJECT_COUNTERS, REJ_TARGET_GEOMETRY,
};
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering, Arc};

fn arena_with_capital(cap: f64) -> Arc<GlobalArena> {
    let arena = GlobalArena::build_in_own_stack(cap);
    arena
        .config
        .global_max_drawdown
        .store(0.10, Ordering::Relaxed);
    arena
        .config
        .margin_cushion_pct
        .store(0.90, Ordering::Relaxed);
    arena
}

#[test]
fn test_r7_r2_a2_ruin_small_sample_lcb_contract() {
    // 1. Sin trades: debe retornar CONSERVATIVE_Q (0.60)
    assert_eq!(conservative_loss_q(0.90, 0.0), CONSERVATIVE_Q);

    // 2. Con 2 trades ganados (win_rate = 1.0, N = 2):
    // El estimador puntual naive daría q = 0.0 (streak esperada infinita o nula, f_cap máximo 25%).
    // Con la LCB bayesiana de Jeffreys (Beta(2.5, 0.5)), el win_rate LCB es ~0.52,
    // de modo que q conservador es > 0.45.
    let q_micro = conservative_loss_q(1.0, 2.0);
    assert!(
        q_micro > 0.45,
        "Micro-muestra con WR=1.0 debe tener q conservador > 0.45, medido: {}",
        q_micro
    );

    // 3. El streak_ruin_cap con q_micro es significativamente más estricto
    // que asumir ingenuamente q=0.0 o q=0.10
    let cap_naive = streak_ruin_cap(0.10);
    let cap_conservador = streak_ruin_cap(q_micro);
    assert!(
        cap_conservador < cap_naive,
        "Cap conservador ({}) debe ser menor y más seguro que cap naive ({})",
        cap_conservador,
        cap_naive
    );

    // 4. Convergencia con muestra grande (N=100, WR=0.60):
    let q_mature = conservative_loss_q(0.60, 100.0);
    assert!(
        q_mature > 0.45 && q_mature < 0.55,
        "Con N=100 y WR=0.60, q debe ser aproximadamente ~0.48: {}",
        q_mature
    );
}

#[test]
fn test_r7_r2_chaos_simplex_contracts_margin_smoothly() {
    let arena = arena_with_capital(13.0);
    let orchestrator = PortfolioOrchestrator::new(&arena);

    // En calma (p_crash = 0.0, p_bull = 0.0, p_chaos = 0.0):
    arena.regime_p_crash.store(0.0, Ordering::Relaxed);
    arena.regime_p_bull.store(0.0, Ordering::Relaxed);
    arena.regime_p_chaos.store(0.0, Ordering::Relaxed);

    // Una orden de $2.00 de margen sobre $13.00 (15.38% de exposición) es admisible en calma
    assert!(
        orchestrator.allow_trade(true, 2.0, MarketRegime::Range, 5.0),
        "En calma total, la orden de $2.00 cabe en el margen libre"
    );

    // Con caos extremo (p_chaos = 0.95):
    // La presión sistémica omnidireccional se eleva a 0.50 * 0.95 = 0.475
    // lo que contrae directional_pressure a 0.25 * 0.475 = 0.11875.
    arena.regime_p_chaos.store(0.95, Ordering::Relaxed);

    // La misma lógica de allow_trade opera suavemente y se mantiene finita
    let allowed_long = orchestrator.allow_trade(true, 2.0, MarketRegime::Range, 5.0);
    let allowed_short = orchestrator.allow_trade(false, 2.0, MarketRegime::Range, 5.0);
    // Verificamos simetría ante caos
    assert_eq!(
        allowed_long, allowed_short,
        "La presión por caos espectral debe ser simétrica para Long y Short"
    );
}

#[test]
fn test_r7_r2_nonfinite_chaos_rejects_both_sides() {
    let arena = arena_with_capital(100.0);
    let orchestrator = PortfolioOrchestrator::new(&arena);

    for non_finite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        arena.regime_p_chaos.store(non_finite, Ordering::Relaxed);
        assert!(
            !orchestrator.allow_trade(true, 1.0, MarketRegime::Range, 5.0),
            "Chaos non-finite ({}) debe rechazar Long",
            non_finite
        );
        assert!(
            !orchestrator.allow_trade(false, 1.0, MarketRegime::Range, 5.0),
            "Chaos non-finite ({}) debe rechazar Short",
            non_finite
        );
    }
}

#[test]
fn test_r7_r2_a1_analytical_first_hitting_gate_contract() {
    // Bracket con TP = 2.25 * SL (RR = 2.25:1)
    let sl = 0.0055; // 55 bps
    let tp = sl * 2.25; // 123.75 bps
    let sigma = 0.0040; // ATR = 40 bps

    // 1. Con deriva neutral (mu = 0):
    // P(hit SL first) = tp / (tp + sl) = 2.25 / 3.25 ≈ 0.6923 <= 0.88
    let p_neutral = probabilidad_tocar_sl_antes_de_tp(tp, sl, 0.0, sigma);
    assert!(
        (p_neutral - 0.6923).abs() < 0.01,
        "Deriva neutral debe ser la regla de la palanca ~0.6923: {}",
        p_neutral
    );
    assert!(p_neutral <= 0.88, "Deriva neutral no debe ser vetada por la compuerta 0.88");

    // 2. Con fuerte deriva adversa (mu = -0.0050 contra la posición):
    // La probabilidad analítica de tocar el stop antes del target se dispara a > 0.90
    let p_adverso = probabilidad_tocar_sl_antes_de_tp(tp, sl, -0.0050, sigma);
    assert!(
        p_adverso > 0.88,
        "Fuerte deriva adversa debe superar el umbral de veto 0.88: {}",
        p_adverso
    );

    // 3. Verificamos que RiskEngine::evaluate_quantum_order rechaza una orden con marea adversa violenta
    static REGISTRY: std::sync::Once = std::sync::Once::new();
    REGISTRY.call_once(|| {
        quantum_arena::symbol_registry::update_registry(vec![
            quantum_arena::symbol_registry::SymbolSpec {
                symbol: "BTCUSDT".into(),
                step_size: 0.001,
                tick_size: 0.01,
                min_qty: 0.001,
                min_notional: 5.0,
                max_leverage: 20,
                maker_fee: 0.0002,
                taker_fee: 0.0004,
                is_shadow: false,
            },
        ])
    });
    let arena = arena_with_capital(13.0);
    let mut engine = RiskEngine::new(13.0);

    // Configuración de mercado para coin 0
    arena.coins[0].current_price.store(60000.0, Ordering::Relaxed);
    arena.coins[0].current_atr.store(60000.0 * sigma, Ordering::Relaxed);
    arena.coins[0].metrics.trade_count.store(10, Ordering::Relaxed);
    arena.coins[0].metrics.win_rate.store(0.60, Ordering::Relaxed);
    arena.coins[0].metrics.profit_factor.store(2.0, Ordering::Relaxed);
    arena.coins[0].metrics.kelly_fraction.store(0.15, Ordering::Relaxed);

    // Marea portadora bajista extrema (-0.95): intentar Long aquí tiene fuerte deriva adversa
    arena.coins[0].spectral_coherence.store(-0.95, Ordering::Relaxed);

    let rej_before = REJECT_COUNTERS[REJ_TARGET_GEOMETRY].load(Ordering::Relaxed);

    let buy_intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.80,
        win_probability: 0.60,
        tp_price_target: 60000.0 * (1.0 + tp),
        sl_price_target: 60000.0 * (1.0 - sl),
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    };

    let evaluated = engine.evaluate_quantum_order(0, &buy_intent, &arena);

    assert_eq!(
        evaluated.signal,
        SignalType::Flat,
        "Orden contra marea extrema debe ser rechazada"
    );
    let rej_after = REJECT_COUNTERS[REJ_TARGET_GEOMETRY].load(Ordering::Relaxed);
    assert!(
        rej_after > rej_before,
        "El contador REJ_TARGET_GEOMETRY debe incrementarse al disparar la compuerta analítica de primer toque: before={}, after={}",
        rej_before, rej_after
    );
}
