//! WB-5 (especificación GLM-APP v3, §5.4) — `kelly_con_encogimiento`: la
//! fracción con p_LCB de Wilson, el pago desde la cota del PF ya aplicada y el
//! encogimiento S(n), con `clamp_ruin` como último eslabón. EN SOMBRA: estas
//! pruebas fijan la función; que ninguna orden la lea lo fijan
//! `god-engine-core/tests/wb5_sombra_cable_contract.rs` y la traza de identidad
//! del replay (`backtest-engine/tests/wb45_sombra_identidad_ordenes.rs`).
//!
//! 5.4.6 (`b_neto` con fricción) queda fuera: su paso (WB-5c) espera la
//! derivación TP/SL v2.2 (orden de implementación, paso 5).

use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::evidence::{profit_factor_lcb, wilson_lcb, Z_LCB};
use risk_engine::kelly::{
    encogimiento, kelly_con_encogimiento, kelly_encogido_con_pago, pago_desde_pf,
    KAPPA_BERNOULLI_MAX,
};
use risk_engine::ruin::{clamp_ruin, conservative_loss_q};
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;

/// 5.4.1 — n = 1, p̂ = 1: la cota de Wilson cae a 0,271 y f* < 0 con b = 1,4,
/// así que f = 0 por el RETORNO TEMPRANO, aunque S(1) valga 0,598 (no ≈ 0).
#[test]
fn encogimiento_y_lcb_cero_con_n_chico() {
    let a = kelly_encogido_con_pago(1.0, 1.0, 1.4, 1.64, 0.5);
    assert!((a.p_lcb - 0.271).abs() < 5e-4, "p_LCB(1,1) = {}", a.p_lcb);
    assert!(a.f_estrella < 0.0, "f*(0,271; 1,4) = {}", a.f_estrella);
    assert_eq!(a.f_ajustada, 0.0, "sin edge, f = 0 exacto");
    // S(1) = 1/(1 + 2,6896·κ) con κ = 1/4: el valor real, no forzado a 0.
    let s1 = 1.0 / (1.0 + 1.64 * 1.64 * KAPPA_BERNOULLI_MAX);
    assert_eq!(a.s, s1);
    assert!((a.s - 0.598).abs() < 5e-4, "S(1) = {}", a.s);
    assert_eq!(KAPPA_BERNOULLI_MAX, 0.25, "κ = máx p(1−p)");
    // El umbral exacto de pago es (1 − p_LCB)/p_LCB = 2,6896 (la
    // especificación lo redondea a 2,69): por debajo, f = 0 con n = 1; por
    // encima (control), la fórmula sí apuesta.
    let umbral = (1.0 - a.p_lcb) / a.p_lcb;
    assert!(umbral > 2.689 && umbral < 2.690, "umbral {umbral}");
    for k in 1..=268 {
        let b = f64::from(k) / 100.0;
        assert_eq!(kelly_encogido_con_pago(1.0, 1.0, b, 1.64, 0.5).f_ajustada, 0.0, "b = {b}");
    }
    assert!(kelly_encogido_con_pago(1.0, 1.0, 2.70, 1.64, 0.5).f_ajustada > 0.0);
}

/// 5.4.2 — n grande: S → 1 y f → clamp_ruin(f*(p_LCB, b)); la diferencia
/// contra el Kelly con p̂ crudo es exactamente el margen de la cota.
#[test]
fn convergencia_con_n_grande() {
    let (p, b, z) = (0.55, 1.4, 1.64);
    let n = 1e12;
    let q = conservative_loss_q(p, n);
    let a = kelly_encogido_con_pago(p, n, b, z, q);
    assert!((a.s - 1.0).abs() < 1e-11, "S(n grande) = {}", a.s);
    assert!((a.p_lcb - p).abs() < 1e-5, "p_LCB → p̂: {}", a.p_lcb);
    assert_eq!(a.f_ajustada, clamp_ruin(a.f_estrella * a.s, q));
    let kelly_crudo = p - (1.0 - p) / b;
    let margen = (p - a.p_lcb) * (1.0 + 1.0 / b);
    assert!(((kelly_crudo - a.f_estrella) - margen).abs() < 1e-12);
    // Y S crece con n hacia 1, sin pasarse.
    let mut previo = 0.0;
    for n in [1.0, 2.0, 5.0, 10.0, 100.0, 1e4, 1e8] {
        let s = encogimiento(n, z);
        assert!(s > previo && s < 1.0, "S({n}) = {s}");
        previo = s;
    }
}

fn arena_lh04() -> (std::sync::Arc<GlobalArena>, SignalIntent) {
    std::env::set_var("TG_GENOME_ENV", "backtest");
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
        ..SignalIntent::default()
    };
    (arena, intent)
}

/// 5.4.3 — UNA sola fuente de p_LCB: la que publica LH-04 en la ruta viva
/// (`lh04_p_lcb`) es, bit a bit, la de la ruta nueva con la misma entrada. Y
/// el pago de la ruta nueva sale de la cota del PF que llegó, sin otro LCB.
#[test]
fn lcb_unificado_una_sola_fuente() {
    let (arena, intent) = arena_lh04();
    let mut engine = RiskEngine::new(1000.0);
    for (wr, n, pf) in [(0.75, 6usize, 8.0), (0.70, 20, 2.5), (0.55, 100, 1.4), (0.62, 1, 3.0)] {
        arena.coins[0].metrics.win_rate.store(wr, Relaxed);
        arena.coins[0].metrics.trade_count.store(n, Relaxed);
        arena.coins[0].metrics.profit_factor.store(pf, Relaxed);
        arena.registry.set_for_coin(0, "lh04_p_lcb", f64::NAN);
        let _ = engine.evaluate_quantum_order(0, &intent, &arena);
        let lh04 = arena.registry.get_for_coin_or(0, "lh04_p_lcb", f64::NAN);
        assert!(lh04.is_finite(), "LH-04 publicó su cota para wr {wr} n {n}");
        let z = arena.registry.get_value_or("lh04_z_lcb", Z_LCB);
        let nueva = kelly_con_encogimiento(wr, n as f64, pf, z, 0.5);
        assert_eq!(nueva.p_lcb.to_bits(), lh04.to_bits(), "wr {wr} n {n}: {} vs {lh04}", nueva.p_lcb);
        assert_eq!(nueva.p_lcb.to_bits(), wilson_lcb(wr, n as f64, z).to_bits());
    }
    // Sin doble LCB: el pago es el de la cota del PF tal cual llegó.
    let pf_para_kelly = profit_factor_lcb(30.0, 10.0, 40.0);
    let a = kelly_con_encogimiento(0.6, 40.0, pf_para_kelly, Z_LCB, 0.5);
    assert_eq!(a.b_lcb.to_bits(), pago_desde_pf(pf_para_kelly, 0.6).to_bits());
    let doble = pago_desde_pf(profit_factor_lcb(pf_para_kelly * 10.0, 10.0, 40.0), 0.6);
    assert_ne!(a.b_lcb, doble, "control: un segundo LCB daría otro pago");
}

/// 5.4.4 — el tope de ruina nunca se supera, y un f* negativo nunca llega a
/// un clamp que lo eleve.
#[test]
fn tope_ruina_nunca_superado() {
    for &(p, n, b) in &[(0.99, 1e9, 1e6), (0.95, 1e6, 50.0), (0.9, 1e4, 9.0)] {
        let q = conservative_loss_q(p, n);
        let a = kelly_encogido_con_pago(p, n, b, Z_LCB, q);
        assert!(a.f_estrella * a.s > clamp_ruin(1.0, q), "el caso empuja por encima del tope");
        assert_eq!(a.f_ajustada, clamp_ruin(1.0, q), "f se corta en el tope de ruina");
        assert!(a.f_ajustada <= 0.25);
    }
    for &(p, n, b) in &[(0.3, 50.0, 1.0), (0.5, 1e6, 0.9), (0.0, 10.0, 2.0)] {
        let a = kelly_encogido_con_pago(p, n, b, Z_LCB, 0.5);
        assert!(!(a.f_estrella > 0.0));
        assert_eq!(a.f_ajustada, 0.0, "f* ≤ 0 ⇒ 0, nunca elevado");
    }
    // Entradas fuera de dominio: sin apuesta.
    for a in [
        kelly_con_encogimiento(f64::NAN, 10.0, 2.0, Z_LCB, 0.5),
        kelly_con_encogimiento(0.6, 0.0, 2.0, Z_LCB, 0.5),
        kelly_con_encogimiento(0.6, 10.0, 0.0, Z_LCB, 0.5),
        kelly_con_encogimiento(1.0, 10.0, 2.0, Z_LCB, 0.5),
        kelly_con_encogimiento(0.6, 10.0, f64::INFINITY, Z_LCB, 0.5),
    ] {
        assert_eq!(a.f_ajustada, 0.0, "{a:?}");
    }
}

/// 5.4.5 — la ruta nueva (kelly.rs) no lleva ninguno de los literales de los
/// rieles del legado; el legado vive intacto en kelly_legado.rs.
#[test]
fn sin_clamps_literales_en_ruta_nueva() {
    let nueva = include_str!("../src/kelly.rs");
    for literal in ["0.20", "0.75", "0.15", "0.65", "0.40", "0.95", "0.25"] {
        assert!(!nueva.contains(literal), "kelly.rs contiene el literal {literal}");
    }
    assert!(nueva.contains("pub fn kelly_con_encogimiento("));
    assert!(nueva.contains("pub use crate::kelly_legado::{calculate_kelly_fraction, continuous_capital_scale};"));
    let legado = include_str!("../src/kelly_legado.rs");
    assert!(legado.contains("pub fn calculate_kelly_fraction("));
    assert!(legado.contains("clamp(0.20, 0.75)"), "control: el escaneo encuentra los literales donde están");
}

/// La reexportación es la MISMA función del legado (no una copia).
#[test]
fn el_legado_reexportado_es_el_mismo() {
    let via_kelly: fn(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) -> f64 =
        risk_engine::kelly::calculate_kelly_fraction;
    let via_legado: fn(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) -> f64 =
        risk_engine::kelly_legado::calculate_kelly_fraction;
    for args in [
        (0.55, 1.3, 13_000.0, 13.0, 1.5, 1.0, 0.01, 0.5, 0.35, 1.0),
        (0.4, 0.8, 13.0, 13.0, 1.5, 1.0, 0.01, 0.5, 0.35, 0.5),
    ] {
        let a = via_kelly(args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, args.9);
        let b = via_legado(args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, args.9);
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

/// La cota única es, bit a bit, el cálculo en línea que LH-04 tenía en C3
/// (copiado aquí literal) sobre una rejilla de p, n y z: el cambio de LH-04 a
/// `wilson_lcb` no mueve ningún tamaño.
#[test]
fn wilson_lcb_es_bit_a_bit_el_calculo_de_lh04_en_c3() {
    let en_linea_c3 = |lh04_wr: f64, n: f64, z_lcb_lh04: f64| {
        let z2 = z_lcb_lh04 * z_lcb_lh04;
        let denom = 1.0 + z2 / n;
        let centro = (lh04_wr + z2 / (2.0 * n)) / denom;
        let radio =
            z_lcb_lh04 * ((lh04_wr * (1.0 - lh04_wr) / n + z2 / (4.0 * n * n)).sqrt() / denom);
        (centro - radio).clamp(0.0, 1.0)
    };
    let mut casos = 0u32;
    for kp in 1..200u32 {
        let p = f64::from(kp) / 200.0;
        for n in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 31.0, 100.0, 1e3, 1e6] {
            for z in [1.28, 1.64, Z_LCB, 1.959_963_984_540_054, 2.33] {
                assert_eq!(wilson_lcb(p, n, z).to_bits(), en_linea_c3(p, n, z).to_bits(), "p {p} n {n} z {z}");
                casos += 1;
            }
        }
    }
    assert_eq!(casos, 199 * 10 * 5);
}