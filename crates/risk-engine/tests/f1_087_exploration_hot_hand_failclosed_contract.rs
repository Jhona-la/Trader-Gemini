//! F1-087 (Ola Ω191, AGY / Quant Sr.): Contrato bilateral de preservación de
//! exploración en sub-Kelly, erradicación de inflación hot-hand en camino legado
//! y cierre fail-closed a sonda mínima para activos sin edge estadístico probado en LH-04.
//!
//! # Qué estaba mal
//! 1. En `crates/risk-engine/src/lib.rs:481`, `raw_kelly.clamp(clamp_min, clamp_max)`
//!    re-clampeaba el valor de Kelly a `clamp_min`. Sin embargo, cuando `profit_factor <= 1.0`,
//!    `calculate_kelly_fraction` (en `kelly.rs:118-124`) genera exploración fraccional en
//!    `[0, 0.25 · clamp_min]`. El re-clamp inflaba el riesgo entre 4x y 15x en activos sin edge,
//!    destruyendo la exploración ultra-conservadora.
//! 2. En `crates/risk-engine/src/lib.rs:702`, el sesgo epigenético se clampeaba en `[0.50, 2.00]`.
//!    En rachas ganadoras (`epi_bias > 1.0`), multiplicaba la confianza por hasta 2.0x,
//!    incurriendo en la falacia del "hot-hand". Esto contradecía la doctrina de LH-04 (:1522-1524),
//!    donde las rachas ganadoras NO inflan (`min(bias, 1.0)`).
//! 3. En LH-04 (:1457-1586), cuando una moneda tenía evidencia madura (`n >= muestras_min_lh04`)
//!    pero la cota de Wilson determinaba ausencia de edge (`lh04_f == 0.0` o `PF <= 1.0`),
//!    el bloque `if lh04_f_efectiva > 0.0` se saltaba silenciosamente. El margen quedaba
//!    en el valor legado de `raw_exposure`, cayendo en un fail-open que arriesgaba capital
//!    pleno en activos perdedores en vez de acotar a sonda mínima de muestreo.
//!
//! # Corrección F1-087
//! 1. `raw_kelly.min(clamp_max)`: preserva la exploración fraccional `[0, 0.25 · clamp_min]`
//!    sin forzar el piso `clamp_min` en regímenes de exploración.
//! 2. `epi_bias.clamp(0.50, 1.00)` en el camino legado: encoge en racha perdedora (quiebre de régimen),
//!    pero erradica la inflación ficticia de hot-hand en racha ganadora.
//! 3. `lh04_sin_edge`: detecta evidencia madura sin edge (`n >= min` y `f == 0.0 || PF <= 1.0`)
//!    y fija `final_margin = 0.0`, asegurando que el rescate posterior dimensione estrictamente
//!    la orden mínima de adquisición de información (`es_sonda || lh04_sin_edge`).

use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;

fn fixture_mercado(max_leverage: u32) -> (std::sync::Arc<GlobalArena>, SignalIntent) {
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
    let arena = GlobalArena::build_in_own_stack(1000.0);
    arena.coins[0].current_price.store(100.0, Relaxed);
    arena.coins[0].current_atr.store(1.0, Relaxed);
    arena.coins[0].hurst_exponent.store(0.5, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    arena.config.margin_cushion_pct.store(0.80, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.8,
        win_probability: 0.70,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    };
    (arena, intent)
}

/// (1) Preservación de la exploración fraccional de Kelly cuando PF <= 1.0 (sub-Kelly).
/// El riesgo asignado en el camino legado debe respetar `f <= 0.25 · clamp_min`
/// y NO ser inflado artificialmente a `clamp_min`.
#[test]
fn test_f1_087_exploracion_sub_kelly_preservada_sin_reinflar_a_clamp_min() {
    let (arena, intent) = fixture_mercado(5);
    // Pocos trades (n = 10 < 30 min para LH-04), PF <= 1.0 (0.80) y WR = 0.40.
    // kelly.rs calcula: pf_ramp = (0.8 - 0.5)/0.5 = 0.60; wr_ramp = (0.4/0.05).min(1.0) = 1.0.
    // exploration = clamp_min(0.01) * 0.25 * 0.60 * 1.0 = 0.0015.
    arena.coins[0].metrics.win_rate.store(0.40, Relaxed);
    arena.coins[0].metrics.trade_count.store(10, Relaxed);
    arena.coins[0].metrics.profit_factor.store(0.80, Relaxed);

    // Con capital = 2000.0 y conf = 0.8:
    // Con el fix F1-087: raw_kelly = 0.0015 => raw_exposure = 0.8 * 0.0015 * 2000 = 2.40 USD.
    // Al ser < min_notional (~5.10 USD con L=1), se rescata al mínimo ejecutable (~5.10255 USD).
    // Con el bug anterior: raw_kelly se re-clampeaba a clamp_min = 0.01 => raw_exposure = 0.8 * 0.01 * 2000 = 16.00 USD!
    let mut engine = RiskEngine::new(2000.0);
    let out = engine.evaluate_quantum_order(0, &intent, &arena);

    assert_ne!(out.signal, SignalType::Flat, "La orden de exploración debe ser emitida");
    assert!(
        out.volume_usd < 10.0,
        "El margen de exploración ({}) no debe inflarse hacia el clamp_min completo (16.0 USD)",
        out.volume_usd
    );
    // Verificar que se situó en el mínimo indispensable de la sonda
    assert!(
        out.volume_usd <= 5.5,
        "El margen debe acotarse al rescate mínimo de notional (~5.10 USD), obtenido: {}",
        out.volume_usd
    );
}

/// (2) Asimetría anti-hot-hand en el camino legado: una racha ganadora (`epi_bias = 2.0`)
/// no debe inflar la confianza ni duplicar el tamaño de posición (techo 1.00).
#[test]
fn test_f1_087_asimetria_anti_hot_hand_racha_ganadora_no_infla_legado() {
    let (arena, intent) = fixture_mercado(5);
    // Establecer racha ganadora extrema: sesgo epigenético = 2.0
    arena.coins[0].epigenetic_bias.store(2.0, Relaxed);
    arena.coins[0].last_close_ts.store(0, Relaxed);
    arena.coins[0].last_tick_timestamp_ms.store(10_000, Relaxed);

    // Trade count = 0 => arranque en frío (kelly_cold = 0.02)
    arena.coins[0].metrics.trade_count.store(0, Relaxed);

    let mut engine = RiskEngine::new(1000.0);
    let out_hot = engine.evaluate_quantum_order(0, &intent, &arena);

    // Ahora simular sesgo neutral = 1.0
    arena.coins[0].epigenetic_bias.store(1.0, Relaxed);
    let out_neutral = engine.evaluate_quantum_order(0, &intent, &arena);

    assert_ne!(out_hot.signal, SignalType::Flat);
    assert_ne!(out_neutral.signal, SignalType::Flat);

    // Con el techo anti-hot-hand en 1.00, la orden con bias = 2.0 no debe superar a la de bias = 1.0
    let diff = (out_hot.volume_usd - out_neutral.volume_usd).abs();
    assert!(
        diff < 1e-4,
        "Racha ganadora infló el margen: out_hot={} vs out_neutral={}",
        out_hot.volume_usd,
        out_neutral.volume_usd
    );
}

/// (3) Cierre fail-closed en LH-04: cuando un activo tiene evidencia madura (n >= 30)
/// pero la estadística demuestra ausencia de edge (f == 0.0 debido a PF <= 1.0 o WR bajo),
/// el sistema NO debe caer en fail-open al Kelly legado, sino acotarse estrictamente
/// al nocional mínimo de sonda para adquisición de información de mercado.
#[test]
fn test_f1_087_lh04_sin_edge_fail_closed_a_sonda_minima() {
    let (arena, intent) = fixture_mercado(5);
    // Evidencia madura: 100 operaciones, pero sin edge (WR = 0.35, PF = 0.85)
    arena.coins[0].metrics.win_rate.store(0.35, Relaxed);
    arena.coins[0].metrics.trade_count.store(100, Relaxed);
    arena.coins[0].metrics.profit_factor.store(0.85, Relaxed);

    // Con capital = 2000.0:
    // Sin el fix F1-087, el sistema caía en fail-open al Kelly legado, emitiendo una orden
    // de margen de $16.0 a $40.0 USD sobre un activo con edge negativo probado.
    // Con F1-087, `lh04_sin_edge` fija `final_margin = 0.0`, y el probe clamp downstream
    // acota la orden estrictamente al mínimo viable de sonda (~5.10 USD para L=1).
    let mut engine = RiskEngine::new(2000.0);
    let out = engine.evaluate_quantum_order(0, &intent, &arena);

    assert_ne!(out.signal, SignalType::Flat, "Debe emitirse como sonda de información");

    assert!(
        out.volume_usd <= 5.5,
        "La orden en moneda sin edge demostrado ({}) debe acotarse a sonda mínima (~5.10 USD)",
        out.volume_usd
    );

    // Telemetría LH-04 debe confirmar riesgo objetivo nulo (f_riesgo == 0)
    let f_objetivo = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    assert_eq!(
        f_objetivo, 0.0,
        "f_objetivo en LH-04 sin edge debe ser 0.0, obtenido: {f_objetivo}"
    );
}
