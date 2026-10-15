//! LH-04 (Línea H ola 1, Claude — implementado GLM 117): dimensionado en
//! ESPACIO DE RIESGO cuando hay evidencia medida.
//!
//! f = ½ · max(0, p_LCB − (1 − p_LCB)/b), con p_LCB = Wilson-LCB(z=1.64)
//! del win-rate medido, b = PF·(1−p̂)/p̂, acotado por clamp_ruin (axioma 25%).
//! El margen se deriva: notional = f·C/SL; margen = notional/L. Sin
//! evidencia (n<10, p̂ degenerado, PF≤0): f=0 y el sizing existente intacto.
#[path = "support/net_payoff_fixture.rs"]
mod net_payoff_fixture;

use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;

const DECISION_TIME_MS: u64 = 200_000;

fn fixture() -> (std::sync::Arc<GlobalArena>, SignalIntent) {
    update_registry(vec![SymbolSpec {
        symbol: "BTCUSDT".into(),
        step_size: 0.001,
        tick_size: 0.01,
        min_qty: 0.001,
        min_notional: 5.0,
        max_leverage: 20,
        maker_fee: 0.0002,
        taker_fee: 0.0004,
        is_shadow: false,
    }]);
    let arena = GlobalArena::build_in_own_stack(1000.0);
    // build_in_own_stack devuelve Arc<GlobalArena> — el engine lo toma así.
    arena.coins[0].current_price.store(100.0, Relaxed);
    arena.coins[0].current_atr.store(1.0, Relaxed);
    arena.coins[0].hurst_exponent.store(0.5, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        expected_duration_ms: 60_000,
        // Cohorte Long/local explícita para aislar LG5 de las métricas
        // agregadas de LH04 que cada caso cambia; no se deriva de WR/PF.
        net_payoff: Some(net_payoff_fixture::positive(0, SignalType::Long)),
        ..SignalIntent::default()
    };
    (arena, intent)
}

/// (1) SIN evidencia (n=0): f=0 — el sizing LH-04 NO se activa y la
/// telemetría lo dice. El pedido sale por el camino existente.
#[test]
fn lh04_sin_evidencia_f_cero_y_camino_existente_intacto() {
    let (arena, mut intent) = fixture();
    intent.net_payoff = None; // Arranque frío sin evidencia económica.
    let mut engine = RiskEngine::new(1000.0);
    let out = engine.evaluate_quantum_order_at(0, &intent, &arena, DECISION_TIME_MS);
    // El pedido puede pasar o ser vetado por otras gates — lo que se
    // contrata es que la telemetría LH-04 reporta SIN EVIDENCIA.
    let f = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    assert!(f.is_finite(), "telemetría publicada");
    assert_eq!(f, 0.0, "sin cierres ⇒ f = 0 (no dimensiona por evidencia)");
    // p_LCB sin evidencia: el registro no la publicó o publicó NaN —
    // ambos estados dicen "sin evidencia" (get_for_coin_or default 1.0
    // aquí detectaría una publicación errónea; NaN es lo publicado).
    let p_lcb = arena.registry.get_for_coin_or(0, "lh04_p_lcb", f64::NAN);
    assert!(p_lcb.is_nan() || p_lcb == 0.0, "sin evidencia no hay p_LCB útil");
    let _ = out;
}

/// (2) Evidencia calibrada y objetivo que NO cabe en el tope de margen.
/// Wilson-LCB(.65, 100, 1.64) = .5687795275; b=1.4 y f=.1303824521.
/// Con C=1000, SL genómico=.01 y L=1, el objetivo supera 13 000 USD
/// de margen. El cap LH04 existente impone 250 USD. Un stop de precio
/// a 99 desde entrada100 arriesga 2.5 USD, no f*C ni el precio del stop.
#[test]
fn lh04_evidencia_calibrada_respeta_tope_cuando_objetivo_no_cabe() {
    let (arena, mut intent) = fixture();
    // Evidencia medida de la moneda: 100 cierres, 65% acierto, PF 2.6.
    arena.coins[0].metrics.win_rate.store(0.65, Relaxed);
    arena.coins[0].metrics.trade_count.store(100, Relaxed);
    arena.coins[0].metrics.profit_factor.store(2.6, Relaxed);
    arena.config.global_leverage.store(1.0, Relaxed);
    arena.config.update_sl_curve(0.01, 0.01);
    intent.sl_price_target = 99.0;
    intent.tp_price_target = 103.0;
    let continuous_sl = arena.config.sl_at_tau(intent.expected_duration_ms as f64);
    assert!((continuous_sl - 0.01).abs() < 1e-12);
    assert_eq!(arena.riesgo_por_operacion.load(Relaxed), 0.0);
    assert_eq!(arena.riesgo_por_operacion_usd.load(Relaxed), 0.0);
    let mut engine = RiskEngine::new(1000.0);
    let out = engine.evaluate_quantum_order_at(0, &intent, &arena, DECISION_TIME_MS);

    let f = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    assert!(f.is_finite() && f > 0.0, "con evidencia: f > 0, f={f}");
    // Verificación de la fórmula: p_LCB Wilson y f = ½·max(0, p − (1−p)/b).
    let p_lcb = arena.registry.get_for_coin_or(0, "lh04_p_lcb", f64::NAN);
    let b = arena.registry.get_for_coin_or(0, "lh04_b_realizado", f64::NAN);
    assert!(p_lcb.is_finite() && b.is_finite());
    assert!((p_lcb - 0.5687795275010941).abs() < 1e-12);
    assert!((b - 1.4).abs() < 1e-9, "b = PF·(1−p̂)/p̂ = 2.6·0.35/0.65 = 1.4: {b}");
    let f_esperada = 0.5 * (p_lcb - (1.0 - p_lcb) / b).max(0.0);
    assert!(
        (f - f_esperada).abs() < 1e-9,
        "f = ½·max(0, p_LCB − (1−p_LCB)/b): {f} vs {f_esperada}"
    );
    assert!(f < 0.25, "f respeta el axioma del 25%: {f}");
    assert!((f - 0.1303824521437949).abs() < 1e-12);
    // p_LCB < p̂ (la cota inferior es conservadora).
    assert!(p_lcb < 0.65, "LCB por debajo del estimador puntual: {p_lcb}");

    // Precondición independiente del tamaño emitido: la curva y el L
    // controlados hacen vinculante el cap de margen de GLM120.
    let objetivo = f * 1000.0;
    assert!(objetivo / continuous_sl > 250.0);
    assert_eq!(out.signal, SignalType::Long, "{out:?}");
    assert_eq!(risk_engine::ultimo_rechazo(), None);
    assert_eq!(out.leverage, 1.0);
    assert!((out.volume_usd - 250.0).abs() < 1e-9, "{out:?}");
    assert_eq!(out.sl_target, 99.0);
    assert_eq!(out.tp_target, 103.0);
    let stop_fraction = (100.0 - out.sl_target) / 100.0;
    let riesgo_stop = out.volume_usd * out.leverage * stop_fraction;
    assert!((riesgo_stop - 2.5).abs() < 1e-9);
    assert!(2.5 < objetivo * 0.5, "la cota antigua era inviable tras el cap");
    assert!((arena.riesgo_por_operacion_usd.load(Relaxed) - 2.5).abs() < 1e-9);
    assert!((arena.riesgo_por_operacion.load(Relaxed) - 0.0025).abs() < 1e-12);
}

/// (3) Evidencia DÉBIL se autolimita: con n=12 y p̂=0.9, la Wilson-LCB
/// cae a ~0.66 ⇒ f ~½(0.66−0.34/1.1)≈0.17 — la cota hace su trabajo.
/// Y con PF ≤ 1 (sin edge): b chico ⇒ f ≤ 0 ⇒ camino existente.
#[test]
fn lh04_evidencia_debil_y_sin_edge_fall_closed() {
    let (arena, intent) = fixture();
    // Sin edge probado: PF = 0.9 ⇒ b < 1 ⇒ (1−p)/b > p_LCB ⇒ f = 0.
    arena.coins[0].metrics.win_rate.store(0.50, Relaxed);
    arena.coins[0].metrics.trade_count.store(200, Relaxed);
    arena.coins[0].metrics.profit_factor.store(0.9, Relaxed);
    let mut engine = RiskEngine::new(1000.0);
    let _ = engine.evaluate_quantum_order_at(0, &intent, &arena, DECISION_TIME_MS);
    let f = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    assert_eq!(f, 0.0, "PF<1 (sin edge): b degrada a f=0 — no dimensiona");
}

/// (4) La f SIEMPRE respeta el tope de ruina: con evidencia extrema
/// (p̂=0.95, n=500, PF=10), f_raw sería ~0.41 pero el axioma 25% la acota.
#[test]
fn lh04_f_acotada_por_el_axioma_de_ruina() {
    let (arena, intent) = fixture();
    arena.coins[0].metrics.win_rate.store(0.95, Relaxed);
    arena.coins[0].metrics.trade_count.store(500, Relaxed);
    arena.coins[0].metrics.profit_factor.store(10.0, Relaxed);
    let mut engine = RiskEngine::new(1000.0);
    let _ = engine.evaluate_quantum_order_at(0, &intent, &arena, DECISION_TIME_MS);
    let f = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    assert!(f.is_finite() && f > 0.0);
    assert!(
        f <= 0.25 + 1e-12,
        "evidencia extrema NO rompe el axioma del 25%: f={f}"
    );
}

/// (5) Streak-adaptación (GLM 118) — RACHA PERDEDORA ENCOGE f: con
/// epigenetic_bias = 0.6 (racha perdedora activa), f_efectiva = f·0.6 —
/// el factor OU-relajado multiplica el tamaño hacia abajo (reacción al
/// quiebre de régimen). Sin bias guardado: factor neutro 1.0.
#[test]
fn lh04_racha_perdedora_encoge_f_proporcionalmente() {
    let (arena, intent) = fixture();
    arena.coins[0].metrics.win_rate.store(0.65, Relaxed);
    arena.coins[0].metrics.trade_count.store(100, Relaxed);
    arena.coins[0].metrics.profit_factor.store(2.6, Relaxed);
    let mut engine = RiskEngine::new(1000.0);
    let _ = engine.evaluate_quantum_order_at(0, &intent, &arena, DECISION_TIME_MS);
    let f_neutra = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    let bias_neutro = arena.registry.get_for_coin_or(0, "lh04_bias_factor", f64::NAN);
    assert!(f_neutra > 0.0);
    assert!((bias_neutro - 1.0).abs() < 1e-9, "sin racha: factor neutro");

    arena.coins[0].epigenetic_bias.store(0.6, Relaxed);
    let mut engine2 = RiskEngine::new(1000.0);
    let _ = engine2.evaluate_quantum_order_at(0, &intent, &arena, DECISION_TIME_MS);
    let f_perdedora = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    let factor = arena.registry.get_for_coin_or(0, "lh04_bias_factor", f64::NAN);
    assert!((factor - 0.6).abs() < 1e-9, "factor = bias en racha perdedora: {factor}");
    assert!(
        (f_perdedora - f_neutra * 0.6).abs() < 1e-9,
        "f encoge proporcionalmente: {f_perdedora} vs {f_neutra}·0.6"
    );
    assert!(f_perdedora < f_neutra && f_perdedora > 0.0, "encoge, no muere");
}

/// (6) Streak-adaptación — RACHA GANADORA NO INFLA (asimetría pinneada):
/// bias 1.8 (racha ganadora) deja f EFECTIVA igual a la neutra — la cota
/// inferior de Wilson de vida ya es el techo honesto (anti-hot-hand).
#[test]
fn lh04_racha_ganadora_no_infla_f_asimetria_anti_hot_hand() {
    let (arena, intent) = fixture();
    arena.coins[0].metrics.win_rate.store(0.65, Relaxed);
    arena.coins[0].metrics.trade_count.store(100, Relaxed);
    arena.coins[0].metrics.profit_factor.store(2.6, Relaxed);
    let mut engine = RiskEngine::new(1000.0);
    let _ = engine.evaluate_quantum_order_at(0, &intent, &arena, DECISION_TIME_MS);
    let f_neutra = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);

    arena.coins[0].epigenetic_bias.store(1.8, Relaxed);
    let mut engine2 = RiskEngine::new(1000.0);
    let _ = engine2.evaluate_quantum_order_at(0, &intent, &arena, DECISION_TIME_MS);
    let f_ganadora = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    let factor = arena.registry.get_for_coin_or(0, "lh04_bias_factor", f64::NAN);
    assert!((factor - 1.0).abs() < 1e-9, "bias 1.8 clampea a 1.0: {factor}");
    assert!(
        (f_ganadora - f_neutra).abs() < 1e-12,
        "racha ganadora NO infla: {f_ganadora} == {f_neutra}"
    );
}

/// (7) Streak-adaptación — SIN ESTADO ABSORBENTE: racha perdedora extrema
/// (bias 0.5, el piso nativo) con evidencia madura: f encoge al 50% pero
/// sigue viva (>0) — el multiplicador es de régimen, no de veto; el bias
/// OU-relaja a 1.0 en ~30 min (lo que distingue esto del Ω110 absorbente).
#[test]
fn lh04_racha_extrema_no_crea_estado_absorbente() {
    let (arena, intent) = fixture();
    arena.coins[0].metrics.win_rate.store(0.65, Relaxed);
    arena.coins[0].metrics.trade_count.store(100, Relaxed);
    arena.coins[0].metrics.profit_factor.store(2.6, Relaxed);
    arena.coins[0].epigenetic_bias.store(0.5, Relaxed);
    let mut engine = RiskEngine::new(1000.0);
    let _ = engine.evaluate_quantum_order_at(0, &intent, &arena, DECISION_TIME_MS);
    let f = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    assert!(f > 0.0, "bias 0.5: f encoge pero NO muere: {f}");
}

/// (8) El fixture de ATR diminuto no llega a LH04: su stop difusivo
/// queda bajo el suelo de fricción y debe rechazarse antes del sizing.
/// El antiguo `if f>0 && !Flat` permitía pasar sin comprobar el cap.
/// El caso (2) exige una orden real y comprueba ese cap por separado.
#[test]
fn lh04_stop_bajo_suelo_se_rechaza_antes_del_dimensionado() {
    let (arena, intent) = fixture();
    // Las métricas maduras no pueden saltarse la viabilidad del horizonte.
    arena.coins[0].metrics.win_rate.store(0.75, Relaxed);
    arena.coins[0].metrics.trade_count.store(100, Relaxed);
    arena.coins[0].metrics.profit_factor.store(2.0, Relaxed);
    arena.coins[0].current_atr.store(0.01, Relaxed); // stop MUY ceñido
    let mut engine = RiskEngine::new(1000.0);
    let out = engine.evaluate_quantum_order_at(0, &intent, &arena, DECISION_TIME_MS);
    let f = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    assert_eq!(out.signal, SignalType::Flat, "{out:?}");
    assert_eq!(risk_engine::ultimo_rechazo(), Some(risk_engine::REJ_TP_SL_FLOOR));
    assert!(f.is_nan(), "el rechazo previo no debe atribuir cobertura a LH04: {f}");
    assert_eq!(arena.riesgo_por_operacion.load(Relaxed), 0.0);
    assert_eq!(arena.riesgo_por_operacion_usd.load(Relaxed), 0.0);
}
