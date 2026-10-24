//! F1-088 (Ola Ω192, AGY / Quant Sr.): Contrato bilateral del Invariante Físico de
//! Liquidación en Futuros Perpetuos (`L · (SL + f + mmr) < 1.0`) y Preservación del Freno
//! de Volatilidad (CL-2 / D-746) en los Rescates de Nocional Mínimo.
//!
//! # Qué estaba mal (F1-088)
//! 1. Los dos rescates de nocional mínimo (:1637-1654 y :1736-1748) sustituían o elevaban
//!    el apalancamiento decidido por la matriz cuántica (`QuantumLeverageMatrix`),
//!    la cual aplica el freno de volatilidad CL-2/D-746 (`vol_brake = 1 / (1 + amenaza)`).
//!    Si una cuenta con capital holgado ($100 USD) pedía un margen inicial pequeño ($0.50 USD)
//!    con apalancamiento L=2 (freno activo), el rescate 1 calculaba `candidate_leverage = 5 / 0.50 = 10 -> 11x`,
//!    quintuplicando el riesgo en dólares al stop y anulando el freno de volatilidad, a pesar de
//!    que la cuenta podía pagar perfectamente el margen de $2.55 USD a L=2.
//! 2. No existía el invariante de liquidación `L · (SL + friccion + mmr) < 1.0`. Si un rescate o la
//!    matriz fijaban un apalancamiento donde la distancia al stop superaba la distancia a la liquidación
//!    del exchange (`1/L - mmr`), la posición sería liquidada por Binance ANTES de poder ejecutar el Stop Loss.
//!    El operador perdería el 100% del colateral más tasas de liquidación, mientras el sistema modelaba
//!    un SL ficticio.
//!
//! # Corrección F1-088
//! 1. Se define la distancia total al stop: `dist_stop_full = expected_loss + friccion_total + mmr_spec`.
//! 2. El techo físico de liquidación es `max_safe_lev_liq = floor((1.0 - 1e-4) / dist_stop_full)`.
//! 3. El apalancamiento inicial de la matriz se acota por `max_safe_lev_liq` y se preserva en `matrix_leverage`.
//! 4. Rescate 1: si `required_margin_at_matrix_lev <= safe_limit`, se mantiene `dynamic_leverage = matrix_leverage`
//!    y se eleva el margen, preservando el freno de volatilidad. Solo si excede `safe_limit` (régimen micro)
//!    se eleva el apalancamiento, estrictamente acotado por `max_safe_lev_liq`.
//! 5. Rescate 2 y comprobación terminal: aplican `max_safe_lev_liq` y el invariante terminal
//!    `dynamic_leverage * dist_stop_full < 1.0`, rechazando con `REJ_TARGET_GEOMETRY` si se viola.

use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;

fn fixture_mercado(capital: f64, max_leverage: u32) -> (std::sync::Arc<GlobalArena>, SignalIntent) {
    update_registry(vec![SymbolSpec {
        symbol: "BTCUSDT".into(),
        step_size: 0.001,
        tick_size: 0.01,
        min_qty: 0.001,
        min_notional: 5.0,
        max_leverage,
        maker_fee: 0.0002,
        taker_fee: 0.0004,
        is_shadow: false,
    }]);
    let arena = GlobalArena::build_in_own_stack(capital);
    arena.coins[0].current_price.store(100.0, Relaxed);
    arena.coins[0].current_atr.store(1.0, Relaxed);
    arena.coins[0].hurst_exponent.store(0.5, Relaxed);
    arena.config.global_leverage.store(max_leverage as f64, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    arena.config.margin_cushion_pct.store(0.80, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.85,
        win_probability: 0.80,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    };
    (arena, intent)
}

/// (1) Invariante de Liquidación: Una orden con SL amplio no puede recibir un apalancamiento
/// que coloque la liquidación antes del stop (`L · (SL + f + mmr) < 1.0`).
#[test]
fn test_f1_088_invariante_liquidacion_impide_leverage_mas_alla_de_barrera_sl() {
    let (arena, mut intent) = fixture_mercado(100.0, 50);
    // Configurar SL explícito en 8.0% de distancia (precio 100.0 -> SL en 92.0, TP en 120.0).
    intent.sl_price_target = 92.0;
    intent.tp_price_target = 120.0;

    // Con SL = 8.0%, fricción ~0.12%, MMR = 1.0% (0.01):
    // dist_stop_full = 0.08 + 0.0012 + 0.01 = 0.0912.
    // L_max_seguro = floor(0.9999 / 0.0912) = 10.
    // A L=11, liquidación = 1/11 - 0.01 = 8.09% < 8.12% (liquidaría antes del SL!).
    let mut engine = RiskEngine::new(100.0);
    let out = engine.evaluate_quantum_order(0, &intent, &arena);

    if out.signal != SignalType::Flat {
        assert!(
            out.leverage <= 10.0,
            "El apalancamiento ({}) excede el techo físico seguro de liquidación (10x) para un SL de 8%",
            out.leverage
        );
        let dist_stop = 0.08 + 0.0012 + 0.01;
        assert!(
            out.leverage * dist_stop < 1.0,
            "Invariante violado: L ({}) * dist_stop ({}) >= 1.0",
            out.leverage,
            dist_stop
        );
    }
}

/// (2) Preservación del freno de volatilidad en cuenta estándar ($100 USD):
/// Si el apalancamiento de la matriz es bajo (ej. L=2 o L=3 debido a volatilidad),
/// el rescate de nocional mínimo eleva el margen para alcanzar los $5.0 USD,
/// en lugar de inflar el apalancamiento a 10x-11x destruyendo el freno.
#[test]
fn test_f1_088_preservacion_freno_volatilidad_en_rescate_cuenta_estandar() {
    let (arena, intent) = fixture_mercado(100.0, 20);
    // Historial con edge moderado pero alta volatilidad ATR para activar vol_brake en matriz
    arena.coins[0].metrics.trade_count.store(50, Relaxed);
    arena.coins[0].metrics.profit_factor.store(1.5, Relaxed);
    arena.coins[0].metrics.win_rate.store(0.60, Relaxed);
    arena.coins[0].current_atr.store(4.0, Relaxed); // ATR alto para freno

    let mut engine = RiskEngine::new(100.0);
    let out = engine.evaluate_quantum_order(0, &intent, &arena);

    assert_eq!(out.signal, SignalType::Long, "La orden debe ser admitida");
    // El nocional debe ser >= $5.0 USD
    assert!(
        out.volume_usd * out.leverage >= 5.0,
        "Nocional {} < 5.0 USD",
        out.volume_usd * out.leverage
    );
    // En cuenta de $100 USD, el apalancamiento NO debe dispararse a 10x-15x por el rescate;
    // debe preservar la prudencia de la matriz
    assert!(
        out.leverage <= 5.0,
        "El rescate infló el apalancamiento a {}x, destruyendo el freno de volatilidad",
        out.leverage
    );
    // Y el margen debe haberse elevado para respaldar el nocional a bajo apalancamiento
    assert!(
        out.volume_usd >= 1.0,
        "El margen ({}) debió elevarse para satisfacer el nocional a bajo apalancamiento",
        out.volume_usd
    );
}

/// (3) Rescate en Micro-cuenta ($13 USD): eleva apalancamiento de forma acotada
/// para permitir viabilidad del nocional sin superar el techo de liquidación ni el cap micro.
#[test]
fn test_f1_088_micro_cuenta_rescate_viable_acotado_por_liquidacion() {
    let (arena, intent) = fixture_mercado(13.0, 20);
    // Sin historial de trades previo: régimen de exploración pura en micro-cuenta
    arena.coins[0].metrics.trade_count.store(0, Relaxed);

    let mut engine = RiskEngine::new(13.0);
    let out = engine.evaluate_quantum_order(0, &intent, &arena);

    assert_eq!(out.signal, SignalType::Long, "La orden micro debe ser admitida");
    assert!(
        out.volume_usd * out.leverage >= 5.0,
        "Nocional {} < 5.0 USD",
        out.volume_usd * out.leverage
    );
    // Margen no debe exceder el cap seguro micro (~$3.25 USD)
    assert!(
        out.volume_usd <= 3.30,
        "Margen {} excede el cap micro",
        out.volume_usd
    );
    // Invariante de liquidación preservado
    let sl = (100.0 - out.sl_target).abs() / 100.0;
    let dist_full = sl + 0.0012 + 0.01;
    assert!(
        out.leverage * dist_full < 1.0,
        "Liquidación violada en micro-cuenta: L={} * dist={}",
        out.leverage,
        dist_full
    );
}

/// (4) Rechazo Fail-Closed ante Geometría Imposible (SL que supera la distancia máxima).
#[test]
fn test_f1_088_rechazo_fail_closed_geometria_imposible() {
    let (arena, mut intent) = fixture_mercado(100.0, 20);
    // SL que colocaría la distancia al stop en 99.5% (dist_stop_full > 1.0)
    intent.sl_price_target = 0.5; // Caída de 99.5%
    intent.tp_price_target = 150.0;

    let mut engine = RiskEngine::new(100.0);
    let out = engine.evaluate_quantum_order(0, &intent, &arena);

    assert_eq!(
        out.signal,
        SignalType::Flat,
        "Debe rechazar órdenes con SL geométricamente imposible"
    );
}
