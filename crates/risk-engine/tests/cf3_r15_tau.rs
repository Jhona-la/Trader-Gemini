//! C-F3 / F3.2-03 (Línea C): R-15 — con brackets brownianos la probabilidad
//! de primer toque no debe depender de τ.
//!
//! Teoría que se prueba:
//! - Para un movimiento browniano con deriva μ y volatilidad σ, la
//!   probabilidad de tocar −sl antes que +tp es
//!   P = (e^{θ·tp} − 1)/(e^{θ·(tp+sl)} − 1), con θ = 2μ/σ²
//!   (`tp_sl::probabilidad_tocar_sl_antes_de_tp`).
//! - R-15 (ledger de decisión viva, declarado RESUELTO) escala la deriva
//!   espectral por √(30 s/τ) para «eliminar el castigo artificial a escalas
//!   largas». Con brackets ∝ √τ (la ley de la geometría con H = 0,5),
//!   θ·tp ∝ √(30 s/τ)·√τ es constante: P no depende de τ. Ésa es la promesa
//!   que se comprueba.
//! - El código recorta √(30 s/τ) a [0,15; 1] (risk-engine/src/lib.rs, bloque
//!   R-15). El recorte se activa con τ > 30 s/0,15² = 1 333 s; por encima
//!   vuelve la acumulación ∝ √τ: P = 0,883 a 4 h y 0,953 a 12 h, frente al
//!   umbral de veto 0,88.
//!
//! Solución conocida: precio 60 000, ATR 0,4 %, coherencia espectral −0,3,
//! largo; sl = 0,5·ATR·√(τ/60 s), tp = 2,25·sl, pasados como objetivos
//! explícitos (FMT-211 los reutiliza tal cual) con expected_duration_ms = τ.
//! Con esos datos θ·tp = −150·2,25·0,002·√0,5 = −0,4773 y
//! θ·(tp + sl) = −0,6894 sea cual sea τ ≤ 1 333 s, así que
//! P* = (e^{−0,4773} − 1)/(e^{−0,6894} − 1) ≈ 0,762.
//!
//! El valor servido se lee de la telemetría `p_hit_sl_first`, que el
//! risk-engine escribe justo antes del veto (lectura estricta por moneda: si
//! la orden no llega a R-15 no hay clave y la precondición falla).
//!
//! Convención: una prueba que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto. Para
//! correrla: `cargo test -p risk-engine --test cf3_r15_tau -- --include-ignored`.

use quantum_arena::GlobalArena;
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Once};

const PRECIO: f64 = 60_000.0;
const ATR_PCT: f64 = 0.004;
const COHERENCIA: f64 = -0.3;

fn arena_btc(cap: f64) -> Arc<GlobalArena> {
    static REG: Once = Once::new();
    REG.call_once(|| {
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
    let arena = GlobalArena::build_in_own_stack(cap);
    arena.config.global_max_drawdown.store(0.10, Relaxed);
    arena.config.margin_cushion_pct.store(0.90, Relaxed);
    let c = &arena.coins[0];
    c.current_price.store(PRECIO, Relaxed);
    c.current_atr.store(PRECIO * ATR_PCT, Relaxed);
    c.metrics.trade_count.store(10, Relaxed);
    c.metrics.win_rate.store(0.60, Relaxed);
    c.metrics.profit_factor.store(2.0, Relaxed);
    c.metrics.kelly_fraction.store(0.15, Relaxed);
    c.spectral_coherence.store(COHERENCIA, Relaxed);
    arena
}

/// P servida por el risk-engine para un largo a τ con brackets brownianos.
fn p_primer_toque(tau_ms: u64) -> f64 {
    let arena = arena_btc(1_000.0);
    let mut risk = RiskEngine::new(1_000.0);
    let sl = 0.5 * ATR_PCT * (tau_ms as f64 / 60_000.0).sqrt();
    let tp = 2.25 * sl;
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.80,
        win_probability: 0.60,
        tp_price_target: PRECIO * (1.0 + tp),
        sl_price_target: PRECIO * (1.0 - sl),
        expected_duration_ms: tau_ms,
        ..SignalIntent::default()
    };
    let _ = risk.evaluate_quantum_order(0, &intent, &arena);
    let p = arena.registry.get_for_coin_strict(0, "p_hit_sl_first");
    let p = p.unwrap_or_else(|| {
        panic!(
            "precondición: la orden a τ = {tau_ms} ms debe llegar a R-15 ({})",
            risk_engine::reject_report()
        )
    });
    assert!(
        p.is_finite(),
        "p_hit_sl_first no finita a τ = {tau_ms} ms: {p}"
    );
    p
}

/// Solución cerrada P* para el fixture con τ ≤ 1 333 s (θ·tp constante).
fn p_estrella() -> f64 {
    let theta = 2.0 * (COHERENCIA * ATR_PCT) / (ATR_PCT * ATR_PCT); // −150
    let tp_por_raiz = 2.25 * 0.5 * ATR_PCT * (30.0_f64 / 60.0).sqrt();
    let sl_por_raiz = 0.5 * ATR_PCT * (30.0_f64 / 60.0).sqrt();
    (theta * tp_por_raiz).exp_m1() / (theta * (tp_por_raiz + sl_por_raiz)).exp_m1()
}

/// Esperado: P idéntica a la de 300 s en 1 200 s, 4 h y 12 h. Tolerancia
/// 1e-9: es aritmética en coma flotante sobre la misma fórmula cerrada (los
/// errores de redondeo son O(1e-15)); la base se contrasta con P* a 1e-9.
/// Hoy 1 200 s pasa (por debajo del recorte) y 4 h da 0,883: falla por el
/// recorte a 0,15 de √(30 s/τ).
#[test]
#[ignore = "OPEN F3.2-03 (Línea C, compuerta R-15 del risk-engine; C-F4 LC-F4-04; avisar a AGY, Ω73): tau_ratio = √(30 s/τ) se recorta a 0,15, así que con τ > 1 333 s la deriva vuelve a acumularse ∝ √τ y R-15 veta las τ largas (P = 0,883 a 4 h, 0,953 a 12 h)"]
fn cf3_r15_primer_toque_invariante_en_tau_con_brackets_brownianos() {
    let estrella = p_estrella();
    let base = p_primer_toque(300_000);
    println!("observado: P*(cerrada) = {estrella:.6}; P(300 s) = {base:.6}");
    assert!(
        (base - estrella).abs() < 1e-9,
        "solución conocida a 300 s: servida {base} frente a {estrella}"
    );
    let taus = [1_200_000u64, 14_400_000, 43_200_000];
    let ps: Vec<f64> = taus.iter().map(|&t| p_primer_toque(t)).collect();
    for (t, p) in taus.iter().zip(&ps) {
        println!("observado: τ = {t} ms: P = {p:.6} (Δ = {:+.6})", p - base);
    }
    for (t, p) in taus.iter().zip(&ps) {
        assert!(
            (p - base).abs() < 1e-9,
            "τ = {t} ms: P = {p} frente a {base} a 300 s"
        );
    }
}
