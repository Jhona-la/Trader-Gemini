//! CL-161: toda orden la dimensiona LH-04 o es la orden mínima del símbolo.
//!
//! F1-087 cerró el fail-open del Kelly legado para la moneda sin edge, pero
//! quedaban dos caminos en que el margen legado (`raw_exposure`, Kelly en
//! espacio de margen) seguía fijando el tamaño de la orden tras la sonda:
//! un profit factor no finito y un `lh04_muestras_min` del registro por
//! encima de n. Ahora la evidencia insuficiente va a la orden mínima y las
//! entradas corruptas se rechazan (REJ_INVALID_INPUT, como FMT-212 y
//! F1-089). El bias epigenético NaN ya se rechazaba en la cadena legada
//! (`raw_exposure` no finito, risk-engine lib.rs, `let raw_exposure`): su
//! prueba es una guarda para cuando esa cadena salga y hoy no puede fallar
//! aunque falte la comprobación de CL-161.
//!
//! CL-165: la validez se comprueba antes de toda clase y sobre el bias
//! CRUDO. El factor `bias.clamp(0, 1)` llevaba +∞ a 1 (LH-04 dimensionaba
//! con la f entera) y un bias ≤ 0 a 0 (la orden mínima); un PF negativo iba a
//! la orden mínima por `lh04_sin_edge`, y la sonda salía con cualquier
//! entrada. Un bias finito y positivo fuera de la banda del escritor
//! ([0,5; 2]) no es corrupto: sólo encoge f o la deja igual.
//!
//! CL-167: el bias se lee y se valida en toda clase. Con f = 0 (la sonda,
//! la moneda sin edge, n < mínimo) CL-165 no lo leía y un bias ±∞ o ≤ 0
//! salía a la orden mínima.
//!
//! CL-165b: PF = 0 tampoco es corrupto. Es «sólo pérdidas», lo admite la
//! restauración del estado aprendido y los contratos del techo de comisión y
//! del bias (veto_logic_contracts.rs) lo usan para apagar LH-04: sigue en la
//! orden mínima.

use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::{RiskEngine, ValidatedOrder, REJECT_COUNTERS, REJ_INVALID_INPUT};
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Mutex;

/// Los contadores de rechazo son globales del proceso: las pruebas de este
/// archivo se serializan para leer su propio incremento.
static SERIE: Mutex<()> = Mutex::new(());

fn fixture() -> (std::sync::Arc<GlobalArena>, SignalIntent) {
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

/// Moneda tras la sonda (n = 20 > `CIERRES_DE_SONDA`) con acierto y pago dados.
fn evidencia(arena: &GlobalArena, wr: f64, n: usize, pf: f64) {
    arena.coins[0].metrics.win_rate.store(wr, Relaxed);
    arena.coins[0].metrics.trade_count.store(n, Relaxed);
    arena.coins[0].metrics.profit_factor.store(pf, Relaxed);
}

fn evaluar(arena: &GlobalArena, intent: &SignalIntent) -> (ValidatedOrder, u64) {
    let antes = REJECT_COUNTERS[REJ_INVALID_INPUT].load(Relaxed);
    let mut engine = RiskEngine::new(1000.0);
    let orden = engine.evaluate_quantum_order(0, intent, arena);
    (orden, REJECT_COUNTERS[REJ_INVALID_INPUT].load(Relaxed) - antes)
}

fn nocional(o: &ValidatedOrder) -> f64 {
    o.volume_usd * o.leverage
}

/// La orden mínima de referencia: moneda madura sin edge (PF ≤ 1, F1-087).
fn orden_minima_de_referencia() -> ValidatedOrder {
    let (arena, intent) = fixture();
    evidencia(&arena, 0.65, 20, 0.9);
    let (orden, _) = evaluar(&arena, &intent);
    assert!(orden.volume_usd > 0.0, "premisa: la moneda sin edge sale a la orden mínima");
    orden
}

#[test]
fn cl161_profit_factor_no_finito_tras_la_sonda_se_rechaza() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, intent) = fixture();
    evidencia(&arena, 0.65, 20, f64::NAN);
    let (orden, rechazos) = evaluar(&arena, &intent);
    assert_eq!(orden.volume_usd, 0.0, "sin PF medible no se dimensiona con el Kelly legado");
    assert_eq!(rechazos, 1, "el rechazo es REJ_INVALID_INPUT");
}

#[test]
fn cl161_bias_epigenetico_nan_se_rechaza() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, intent) = fixture();
    evidencia(&arena, 0.65, 100, 2.6);
    // Sin cierre registrado el bias activo es el almacenado tal cual. Guarda:
    // ya pasaba antes de CL-161 (lo rechazaba `raw_exposure`).
    arena.coins[0].epigenetic_bias.store(f64::NAN, Relaxed);
    arena.coins[0].last_close_ts.store(0, Relaxed);
    let (orden, rechazos) = evaluar(&arena, &intent);
    assert_eq!(orden.volume_usd, 0.0);
    assert_eq!(rechazos, 1, "el rechazo es REJ_INVALID_INPUT");
}

#[test]
fn cl161_muestras_minimas_sobre_n_dan_la_orden_minima() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let referencia = orden_minima_de_referencia();
    let (arena, intent) = fixture();
    arena.registry.set("lh04_muestras_min", 50.0);
    evidencia(&arena, 0.65, 20, 2.6);
    let (orden, rechazos) = evaluar(&arena, &intent);
    assert_eq!(rechazos, 0);
    assert!(orden.volume_usd > 0.0, "la evidencia insuficiente no se veta: es la orden mínima");
    let f = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    assert_eq!(f, 0.0, "premisa: LH-04 no dimensiona con n < muestras mínimas");
    assert_eq!(
        nocional(&orden).to_bits(),
        nocional(&referencia).to_bits(),
        "el nocional es el de la orden mínima, no el del margen legado"
    );
}

#[test]
fn cl161_la_orden_madura_con_edge_sigue_en_lh04() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let referencia = orden_minima_de_referencia();
    let (arena, intent) = fixture();
    evidencia(&arena, 0.65, 100, 2.6);
    let (orden, rechazos) = evaluar(&arena, &intent);
    assert_eq!(rechazos, 0);
    let f = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    assert!(f > 0.1, "control: LH-04 dimensiona (f ≈ 0,133), obtenido {f}");
    assert!(
        nocional(&orden) > 10.0 * nocional(&referencia),
        "control: la orden de LH-04 no se recorta a la mínima"
    );
}

/// Bias epigenético almacenado tal cual (sin cierre registrado el bias activo
/// es el almacenado).
fn bias_crudo(arena: &GlobalArena, bias: f64) {
    arena.coins[0].epigenetic_bias.store(bias, Relaxed);
    arena.coins[0].last_close_ts.store(0, Relaxed);
}

#[test]
fn cl165_bias_infinito_se_rechaza() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, intent) = fixture();
    evidencia(&arena, 0.65, 100, 2.6);
    bias_crudo(&arena, f64::INFINITY);
    let (orden, rechazos) = evaluar(&arena, &intent);
    assert_eq!(orden.volume_usd, 0.0, "un bias +∞ no dimensiona con la f entera");
    assert_eq!(rechazos, 1, "el rechazo es REJ_INVALID_INPUT");
}

#[test]
fn cl165_bias_no_positivo_se_rechaza() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    for bias in [0.0, -1.0, f64::NEG_INFINITY] {
        let (arena, intent) = fixture();
        evidencia(&arena, 0.65, 100, 2.6);
        bias_crudo(&arena, bias);
        let (orden, rechazos) = evaluar(&arena, &intent);
        assert_eq!(orden.volume_usd, 0.0, "bias {bias}: no es la orden mínima");
        assert_eq!(rechazos, 1, "bias {bias}: el rechazo es REJ_INVALID_INPUT");
    }
}

#[test]
fn cl165_profit_factor_negativo_se_rechaza() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    for pf in [-1.0, -1e-9] {
        let (arena, intent) = fixture();
        evidencia(&arena, 0.65, 20, pf);
        let (orden, rechazos) = evaluar(&arena, &intent);
        assert_eq!(orden.volume_usd, 0.0, "PF {pf}: no es una moneda sin edge, es una entrada corrupta");
        assert_eq!(rechazos, 1, "PF {pf}: el rechazo es REJ_INVALID_INPUT");
    }
}

#[test]
fn cl165b_profit_factor_cero_da_la_orden_minima() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let referencia = orden_minima_de_referencia();
    for n in [20, 100] {
        let (arena, intent) = fixture();
        evidencia(&arena, 0.65, n, 0.0);
        let (orden, rechazos) = evaluar(&arena, &intent);
        assert_eq!(rechazos, 0, "n = {n}: PF 0 (sólo pérdidas) no es una entrada corrupta");
        assert_eq!(
            nocional(&orden).to_bits(),
            nocional(&referencia).to_bits(),
            "n = {n}: PF 0 es una moneda sin edge y sale a la orden mínima"
        );
    }
}

#[test]
fn cl167_un_bias_corrupto_se_rechaza_tambien_con_f_cero() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    // (win rate, n, PF): la sonda sin cierres, la moneda sin edge y la que no
    // reúne las muestras mínimas. En las tres LH-04 no dimensiona (f = 0).
    let clases = [(0.65, 0, 2.6), (0.65, 20, 0.9), (0.65, 20, 2.6)];
    for (wr, n, pf) in clases {
        for bias in [f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
            let (arena, intent) = fixture();
            if n == 20 && pf > 1.0 {
                arena.registry.set("lh04_muestras_min", 50.0);
            }
            evidencia(&arena, wr, n, pf);
            bias_crudo(&arena, bias);
            let (orden, rechazos) = evaluar(&arena, &intent);
            assert_eq!(orden.volume_usd, 0.0, "n {n}, PF {pf}, bias {bias}: no sale a la orden mínima");
            assert_eq!(rechazos, 1, "n {n}, PF {pf}, bias {bias}: el rechazo es REJ_INVALID_INPUT");
        }
    }
}

#[test]
fn cl167_un_bias_valido_no_cambia_la_orden_minima() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let referencia = orden_minima_de_referencia();
    for bias in [0.3, 0.5, 2.0, 5.0] {
        let (arena, intent) = fixture();
        evidencia(&arena, 0.65, 20, 0.9);
        bias_crudo(&arena, bias);
        let (orden, rechazos) = evaluar(&arena, &intent);
        assert_eq!(rechazos, 0, "control: bias {bias} finito y positivo");
        assert_eq!(
            nocional(&orden).to_bits(),
            nocional(&referencia).to_bits(),
            "bias {bias}: la orden mínima no depende del bias"
        );
    }
}

#[test]
fn cl165_la_sonda_con_entradas_corruptas_se_rechaza() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, intent) = fixture();
    evidencia(&arena, 0.65, 0, -1.0);
    assert!(
        risk_engine::sonda::en_fase_de_sonda(&arena, 0),
        "premisa: sin cierres la moneda está en la sonda"
    );
    let (orden, rechazos) = evaluar(&arena, &intent);
    assert_eq!(orden.volume_usd, 0.0, "la sonda tampoco sale con un PF corrupto");
    assert_eq!(rechazos, 1, "el rechazo es REJ_INVALID_INPUT");
}

#[test]
fn cl165_bias_finito_positivo_fuera_de_banda_sigue_en_lh04() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, intent) = fixture();
    evidencia(&arena, 0.65, 100, 2.6);
    let (libre, _) = evaluar(&arena, &intent);
    let f_libre = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    for (bias, factor) in [(0.3, 0.3), (5.0, 1.0)] {
        let (arena, intent) = fixture();
        evidencia(&arena, 0.65, 100, 2.6);
        bias_crudo(&arena, bias);
        let (orden, rechazos) = evaluar(&arena, &intent);
        assert_eq!(rechazos, 0, "control: un bias {bias} finito y positivo no es corrupto");
        let f = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
        assert!((f - factor * f_libre).abs() < 1e-12, "bias {bias}: f = {f}, esperado {}", factor * f_libre);
        assert!(nocional(&orden) > 0.0 && nocional(&libre) > 0.0);
    }
}
