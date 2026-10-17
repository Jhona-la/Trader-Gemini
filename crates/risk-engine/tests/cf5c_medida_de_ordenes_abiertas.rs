//! C-F5c · R2 y B3 de la revisión posterior del PR #62 (CL-131): lo medido de
//! las órdenes (riesgo al stop y geometría en stops) sale de las órdenes que
//! se ABREN, una vez por operación.
//!
//! Teoría. Los dos frenos leen medias móviles de las órdenes: el riesgo al
//! stop (`riesgo_por_operacion`, en fracción y en USD) y la geometría
//! `w = (TP − f)/SL`, `l = (SL + f)/SL` en stops. Son la muestra de las
//! operaciones que hace la cuenta: con ellas se deriva el umbral K del freno
//! duro y la escala del proceso e. Hasta ahora se actualizaban al final de
//! `evaluate_single_intent`, antes de los vetos del núcleo (consejo, ML-GATE,
//! margen, cantidad) y sin enfriamiento: la misma intención vetada se volvía a
//! medir en cada tick, y la «memoria de 20 órdenes» eran unos 20 ticks de
//! intenciones de cualquier moneda, sobre todo de las que no tienen modelo.
//!
//! Ahora el riesgo deja la medida de la orden validada pendiente, y sólo
//! entra en las medias cuando el núcleo confirma que la abrió
//! (`RiskEngine::registrar_orden_abierta`).
//!
//! Fixture: la cuenta sana del contrato C07 (13 USD, stop y TP explícitos de
//! 1 y 2 USD a 100), con capital que baja de una orden a la siguiente.

use quantum_arena::GlobalArena;
use risk_engine::RiskEngine;
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Mutex, Once};

static SERIE: Mutex<()> = Mutex::new(());

fn cuenta(capital: f64) -> (Arc<GlobalArena>, SignalIntent) {
    static REGISTRO: Once = Once::new();
    REGISTRO.call_once(|| {
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
        ]);
    });
    let arena = GlobalArena::build_in_own_stack(capital);
    let coin = &arena.coins[0];
    coin.current_price.store(100.0, Relaxed);
    coin.current_atr.store(1.0, Relaxed);
    coin.hurst_exponent.store(0.5, Relaxed);
    coin.metrics.trade_count.store(100, Relaxed);
    coin.metrics.profit_factor.store(2.0, Relaxed);
    coin.metrics.win_rate.store(0.75, Relaxed);
    coin.metrics.kelly_fraction.store(0.15, Relaxed);
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    arena.config.global_max_drawdown.store(0.2, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    arena.config.min_confidence_btc.store(0.62, Relaxed);
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        win_probability: 0.9,
        expected_duration_ms: 60_000,
        tp_price_target: 102.0,
        sl_price_target: 99.0,
        ..SignalIntent::default()
    };
    (arena, intent)
}

fn medido(a: &GlobalArena) -> [f64; 4] {
    [
        a.riesgo_por_operacion.load(Relaxed),
        a.riesgo_por_operacion_usd.load(Relaxed),
        a.ganancia_en_stops.load(Relaxed),
        a.perdida_en_stops.load(Relaxed),
    ]
}

/// Esperado: una orden validada que el núcleo no abre (la vetan el consejo o
/// el ML-GATE, o le falta margen), evaluada en 20 ticks, no mueve nada de lo
/// medido. Antes la medían las 20 veces.
#[test]
fn cl131_una_orden_evaluada_que_no_se_abre_no_mueve_lo_medido() {
    let _serie = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, intent) = cuenta(13.0);
    let mut riesgo = RiskEngine::new(13.0);
    for _ in 0..20 {
        let orden = riesgo.evaluate_quantum_order(0, &intent, &arena);
        assert_eq!(orden.signal, SignalType::Long, "{}", risk_engine::reject_report());
    }
    let m = medido(&arena);
    println!("observado: tras 20 evaluaciones sin abrir, lo medido es {m:?}");
    assert_eq!(m, [0.0; 4], "una orden que no se abre no es una operación de la cuenta");
}

/// Esperado (B3): con órdenes reales que se abren y capital que baja, lo
/// medido es la media de esas órdenes, cada una una vez:
/// - la primera fija la media (riesgo al stop = nocional × SL; w y l con la
///   fricción de la orden);
/// - confirmar dos veces la misma apertura no la cuenta dos veces;
/// - la apertura de otra moneda no consume la medida pendiente de ésta.
#[test]
fn cl131_cada_orden_abierta_entra_una_vez_en_lo_medido() {
    let _serie = SERIE.lock().unwrap_or_else(|p| p.into_inner());
    let (arena, intent) = cuenta(13.0);
    let mut riesgo = RiskEngine::new(13.0);

    let orden = riesgo.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(orden.signal, SignalType::Long, "{}", risk_engine::reject_report());
    assert!(riesgo.registrar_orden_abierta(&arena, 0), "la orden validada queda pendiente");
    let nocional = orden.volume_usd * orden.leverage;
    let sl = (100.0 - orden.sl_target) / 100.0;
    let [r, r_usd, w, l] = medido(&arena);
    println!("observado: nocional {nocional:.4} USD, riesgo {r:.6}, {r_usd:.6} USD, w {w:.4}, l {l:.4}");
    assert!((r_usd - nocional * sl).abs() < 1e-12, "riesgo al stop en USD = nocional × SL");
    assert!((r - nocional * sl / 13.0).abs() < 1e-12, "y en fracción del capital al dimensionar");
    assert!(l > 1.0 && w > 0.0 && w < 2.0, "geometría en stops con fricción: w {w}, l {l}");
    let f = (l - 1.0) * sl;
    assert!(((w * sl + f) - 0.02).abs() < 1e-12, "w·SL + f es la distancia al TP");

    assert!(!riesgo.registrar_orden_abierta(&arena, 0), "la misma apertura no se cuenta dos veces");
    assert_eq!(medido(&arena), [r, r_usd, w, l]);

    // La medida pendiente es de la moneda 0: abrir la 1 no la consume.
    let _ = riesgo.evaluate_quantum_order(0, &intent, &arena);
    assert!(!riesgo.registrar_orden_abierta(&arena, 1), "la pendiente es de la moneda 0");
    assert_eq!(medido(&arena), [r, r_usd, w, l]);

    // El capital baja: la orden siguiente se mide con su capital.
    arena.unified_capital.store(11.0, Relaxed);
    let orden2 = riesgo.evaluate_quantum_order(0, &intent, &arena);
    assert_eq!(orden2.signal, SignalType::Long, "{}", risk_engine::reject_report());
    assert!(riesgo.registrar_orden_abierta(&arena, 0));
    let r2 = orden2.volume_usd * orden2.leverage * sl / 11.0;
    let esperado = r + (r2 - r) / (risk_engine::drawdown::TRADE_HORIZON / 10.0);
    let [r_b, ..] = medido(&arena);
    assert!((r_b - esperado).abs() < 1e-12, "EWMA de memoria 20: {r_b} frente a {esperado}");
}
