//! C-F3 / F3.3-08 (Línea C, bloque C-F4): la puntuación cruda no debe
//! dimensionar por su cuenta.
//!
//! Teoría que se prueba: el tamaño de una orden es una función de la
//! probabilidad de ganar (y de la geometría y el riesgo), no de la escala en
//! que una rama expresa su convicción. Con la MISMA probabilidad calibrada
//! (`SignalIntent::win_probability`), dos confianzas crudas distintas deben
//! dar el mismo nocional.
//!
//! Hoy la confianza cruda entra en el tamaño por tres vías y la calibrada por
//! una (HALLAZGOS_C-F3 §F3.3-08):
//! - `effective_confidence = conf·epi_bias` multiplica el margen;
//! - `1 + (conf − min_conf)⁺` multiplica el Kelly micro (sólo con capital
//!   micro, aquí 1 000 USD queda fuera);
//! - la convicción del apalancamiento `0,40 + 0,60·conf` y su techo por
//!   confianza (leverage_matrix.rs);
//! - la calibrada sólo pesa un 30 % en el prob_win del Kelly del
//!   apalancamiento.
//! El hallazgo estima ≈ 1,19 × 1,10 ≈ 1,31 veces más nocional con 0,95 que
//! con 0,80 (cota: el apalancamiento se cuantiza en enteros y sus techos
//! pueden enmascarar parte).
//!
//! Control (pedido por el escéptico de completitud): la prueba imprime margen
//! y apalancamiento por separado para que, si la cuantización entera del
//! apalancamiento enmascara su parte, se vea qué factor cambia.
//!
//! Fixture copiado de tests/leverage_admission_contract.rs, a capital 1 000
//! para quedar fuera del régimen micro.
//!
//! Convención: una prueba que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto.

use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Once};

fn arena(capital: f64) -> Arc<GlobalArena> {
    static REGISTRY: Once = Once::new();
    REGISTRY.call_once(|| {
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
        }])
    });
    let a = GlobalArena::build_in_own_stack(capital);
    a.coins[0].current_price.store(100.0, Relaxed);
    a.coins[0].current_atr.store(1.0, Relaxed);
    a.coins[0].hurst_exponent.store(0.5, Relaxed);
    a.config.global_leverage.store(20.0, Relaxed);
    a.config.global_max_drawdown.store(0.2, Relaxed);
    a.config.kelly_clamp_min.store(0.01, Relaxed);
    a.config.kelly_clamp_max.store(0.25, Relaxed);
    a.config.latency_penalty_ms.store(0.0, Relaxed);
    a.config.live_taker_fee.store(0.0004, Relaxed);
    a.config.base_slippage_floor.store(0.00001, Relaxed);
    a.config.min_confidence_btc.store(0.70, Relaxed);
    a
}

/// (margen, apalancamiento) de un largo con confianza cruda `conf` y la
/// misma probabilidad calibrada 0,9.
fn orden(conf: f64) -> (f64, f64) {
    let a = arena(1000.0);
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: conf,
        win_probability: 0.9, // misma probabilidad calibrada
        expected_duration_ms: 60_000,
        ..Default::default()
    };
    let o = RiskEngine::new(1000.0).evaluate_quantum_order(0, &intent, &a);
    assert_eq!(
        o.signal,
        SignalType::Long,
        "{}",
        risk_engine::reject_report()
    );
    (o.volume_usd, o.leverage)
}

/// Esperado: nocional (margen·apalancamiento) igual para conf 0,80 y 0,95,
/// con tolerancia relativa 1e-6 (las dos órdenes recorren la misma
/// aritmética; cualquier diferencia mayor sólo puede venir de la confianza).
/// Medido hoy: margen ×1,1875 (= 0,95/0,80, el factor conf·epi_bias) y
/// apalancamiento 3 en las dos (la cuantización entera enmascara el factor
/// 0,40 + 0,60·conf en este fixture): nocional ×1,1875, cota inferior del
/// ≈ 1,31 del hallazgo.
#[test]
#[ignore = "OPEN F3.3-08 (Línea C, C-F4 riesgo): la confianza cruda multiplica el margen (conf·epi_bias, ×1,1875 medido de 0,80 a 0,95) además de la convicción del apalancamiento (0,40 + 0,60·conf, aquí enmascarada por la cuantización entera); con la misma p calibrada, conf 0,95 da más nocional que 0,80"]
fn cf3_la_puntuacion_cruda_no_dimensiona_por_su_cuenta() {
    let (m_bajo, l_bajo) = orden(0.80);
    let (m_alto, l_alto) = orden(0.95);
    let (bajo, alto) = (m_bajo * l_bajo, m_alto * l_alto);
    println!(
        "observado: conf 0,80: margen {m_bajo:.6}, L {l_bajo}, nocional {bajo:.6}; \
         conf 0,95: margen {m_alto:.6}, L {l_alto}, nocional {alto:.6}; \
         razón margen {:.4}, razón L {:.4}, razón nocional {:.4}",
        m_alto / m_bajo,
        l_alto / l_bajo,
        alto / bajo
    );
    assert!(
        (alto / bajo - 1.0).abs() < 1e-6,
        "misma p calibrada, nocional {bajo} frente a {alto}"
    );
}
