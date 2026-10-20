//! Contrato Formal de Verificación F1-081:
//! Erradicación de la Doble EMA(0.05) en Payoff Ratio de Kelly y Cota de Memoria Muestral Efectiva.
//!
//! # QUÉ:
//! Erradicación de la composición espuria EMA∘EMA en el payoff ratio $b = \text{avg\_win} / \text{avg\_loss}$
//! que duplicaba el retardo de respuesta (40 cierres en lugar de 20) al calcular una EMA(0.05) en los llamadores
//! (`god_engine.rs`, `booktick_replay.rs`, `backtest-engine`, `online_daemon.rs`) y re-filtrarla dentro de
//! `RiskEnvelope::record_trade`. Alimentación con el PnL neto individual mediante `record_net_pnl`, y
//! acotamiento del tamaño de muestra efectivo en `conservative_payoff_ratio` por la memoria de la ventana
//! ($N_{\text{eff}} = \min(n, 2/\alpha - 1) = \min(n, 39.0)$).
//!
//! # POR QUÉ:
//! Cuando las condiciones de mercado empeoran (ensanchamiento de spread, deslizamiento o cambio de régimen),
//! un filtro EMA∘EMA tardaba 40-60 trades en reaccionar (alcanzando solo el 26% del escalón a los 20 trades),
//! provocando que el dimensionamiento de Kelly apostara hasta ~1.6x más de lo debido cuando la ventaja real
//! ya era nula o negativa. Además, usar el $n$ acumulativo sin cota eliminaba el descuento de incertidumbre
//! tras cientos de trades, desconociendo la varianza muestral finita de la ventana de 39 trades.
//!
//! # PARA QUÉ:
//! Proteger el capital ($13 USD) de sobre-apalancamiento ante deterioro súbito del payoff, garantizar
//! respuesta ágil (64.15% del escalón en 20 trades), preservar la memoria honesta del estimador y mantener
//! paridad estricta entre producción, backtest y evolución.
//!
//! # QUIÉN:
//! Consejo de 10 Roles Senior / Quant Sr. — Ola Ω185 (Cierre F1-081).

use risk_engine::kelly_envelope::RiskEnvelope;

/// Contrato 1: Respuesta al Escalón de la EMA Simple (64.15% en 20 cierres) vs Doble EMA (26.42%).
#[test]
fn f1_081_contrato_1_respuesta_al_escalon_ema_simple_vs_doble() {
    let mut env = RiskEnvelope::new();

    // 1. Inicializar con trades base de 10.0 USD
    for _ in 0..50 {
        env.record_net_pnl(10.0);
    }
    assert!((env.avg_win - 10.0).abs() < 1e-3);

    // 2. Escalón súbito a trades de 20.0 USD durante 20 trades
    for _ in 0..20 {
        env.record_net_pnl(20.0);
    }

    // Tras 20 trades con alpha = 0.05, el valor esperado analítico es:
    // 10.0 + (20.0 - 10.0) * (1 - (0.95)^20) = 10.0 + 10.0 * (1 - 0.3584859) = 16.4151
    let expected_single_ema = 10.0 + 10.0 * (1.0 - (0.95_f64).powi(20));
    assert!(
        (env.avg_win - expected_single_ema).abs() < 1e-3,
        "La EMA simple debe responder al 64.15% del escalón tras 20 trades (esperado: {expected_single_ema:.4}, obtenido: {:.4})",
        env.avg_win
    );

    // En contraste, una doble EMA (EMA∘EMA) se habría quedado en ~12.64 (solo 26.4% de respuesta)
    let double_ema_val = 12.64;
    assert!(
        env.avg_win > double_ema_val + 2.0,
        "La respuesta de la EMA simple es significativamente más rápida que la doble EMA"
    );
}

/// Contrato 2: Paridad Bit a Bit entre `record_trade` y `record_net_pnl`.
#[test]
fn f1_081_contrato_2_paridad_record_net_pnl() {
    let mut env1 = RiskEnvelope::new();
    let mut env2 = RiskEnvelope::new();

    let sequence = [15.5, -8.2, 22.0, -12.4, 5.0, -3.1, 18.9, -9.5];

    for &pnl in &sequence {
        env1.record_net_pnl(pnl);
        let won = pnl > 0.0;
        let abs_pnl = pnl.abs();
        env2.record_trade(won, abs_pnl, abs_pnl);
    }

    assert_eq!(env1.avg_win, env2.avg_win);
    assert_eq!(env1.avg_loss, env2.avg_loss);
    assert_eq!(env1.payoff_ratio, env2.payoff_ratio);
    assert_eq!(env1.posterior.alpha, env2.posterior.alpha);
    assert_eq!(env1.posterior.beta, env2.posterior.beta);
}

/// Contrato 3: Memoria Muestral Efectiva Acotada ($N_{\text{eff}} \le 39.0$).
#[test]
fn f1_081_contrato_3_memoria_muestral_efectiva_acotada() {
    let mut env = RiskEnvelope::new();

    // Alimentar con 1,000 trades para inflar el conteo de Bernoulli
    for i in 0..1000 {
        if i % 2 == 0 {
            env.record_net_pnl(30.0);
        } else {
            env.record_net_pnl(-10.0);
        }
    }

    assert!(env.posterior.n() >= 1000.0);

    // Con n >= 1000, el payoff ratio es ~3.0
    let raw_b = env.payoff_ratio;
    assert!(raw_b > 2.5);

    // Con N_eff acotado a 39.0, el descuento con z = 1.64 DEBE ser sustancial:
    // discount = exp(-1.64 / sqrt(39.0)) = exp(-1.64 / 6.245) = exp(-0.2626) = ~0.769
    // b_lcb = 1.0 + (b - 1.0) * 0.769
    let lcb_b = env.conservative_payoff_ratio(1.64);

    let excess = raw_b - 1.0;
    let expected_discount = (-1.64_f64 / (39.0_f64).sqrt()).exp();
    let expected_lcb = 1.0 + excess * expected_discount;

    assert!(
        (lcb_b - expected_lcb).abs() < 1e-4,
        "lcb_b ({lcb_b}) debe usar N_eff = 39.0 exacto (esperado: {expected_lcb})"
    );

    // Si hubiera usado n = 1000, el descuento habría sido exp(-1.64 / sqrt(1000)) = exp(-0.0518) = ~0.949 (casi nulo)
    let naive_lcb = 1.0 + excess * (-1.64_f64 / (1000.0_f64).sqrt()).exp();
    assert!(
        lcb_b < naive_lcb - 0.25,
        "El descuento con N_eff = 39.0 debe ser genuinamente más conservador que con n no acotado"
    );
}

/// Contrato 4: Protección de Capital ante Deterioro del Payoff Ratio.
#[test]
fn f1_081_contrato_4_proteccion_ante_deterioro_de_payoff() {
    let mut env = RiskEnvelope::new();

    // 1. Fase de alto payoff (win = 30, loss = 10, b = 3.0, WR = 55%)
    for i in 0..60 {
        if i % 20 < 11 {
            env.record_net_pnl(30.0);
        } else {
            env.record_net_pnl(-10.0);
        }
    }
    let f_inicial = env.risk_fraction(1.0, 10.0);
    assert!(f_inicial > 0.05, "Debe tener asignación positiva con b=3.0");

    // 2. Deterioro súbito a payoff adverso (win = 5, loss = 15, b = 0.33, mismo WR = 55%)
    // Con b adverso y p = 0.55: Kelly teórico = p - q/b = 0.55 - 0.45/0.33 < 0 (VENTAJA NEGATIVA)
    for i in 0..60 {
        if i % 20 < 11 {
            env.record_net_pnl(5.0);
        } else {
            env.record_net_pnl(-15.0);
        }
    }

    let f_tras_deterioro = env.risk_fraction(1.0, 10.0);
    assert_eq!(
        f_tras_deterioro, 0.0,
        "Tras trades con payoff adverso continuo, la fracción de riesgo debe ser CERO exacto (obtenido: {f_tras_deterioro})"
    );
}

/// Contrato 5: Inmunidad IEEE-754 ante Valores No Finitos.
#[test]
fn f1_081_contrato_5_inmunidad_ieee754() {
    let mut env = RiskEnvelope::new();

    for bad_val in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        env.record_net_pnl(bad_val);
        assert_eq!(env.avg_win, 0.0);
        assert_eq!(env.avg_loss, 0.0);
        assert_eq!(env.payoff_ratio, 1.0);
    }
}
