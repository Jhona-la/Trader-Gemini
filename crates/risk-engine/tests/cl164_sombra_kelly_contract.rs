//! CL-164: sombra del Kelly legado frente a LH-04.
//!
//! Regla del dueño: una teoría no se retira sin compararla. El Kelly legado
//! (kelly.rs: régimen de capital λ(r), banda S-1, amortiguador de Koopman y
//! PF_LCB dentro de b) ya no dimensiona ninguna orden, pero antes de borrar
//! esa cadena se mide qué habría ganado. Cada orden validada lleva sus dos
//! apuestas (fracción del capital al stop) y su pérdida al stop con la
//! fricción; al cerrar, el libro suma ln(1 + f·R) por regla (LH-04, legado
//! y el mínimo de las dos) con R = retorno neto / (SL + c). Es el criterio
//! de Kelly medido fuera de muestra. No dimensiona nada.
//!
//! CL-166: una regla arruinada (1 + f·R ≤ 0) tiene log-riqueza −∞; el libro
//! la marca y deja de sumarle cierres. El libro suma además R y R² en
//! unidades de stop, y `informe_sombra_kelly` lo publica (el forense lo
//! imprime como `SOMBRA_KELLY`).
//!
//! CL-168: en el informe, el `logw` de una regla arruinada sale como null
//! (JSON no tiene −∞) y cualquier número no finito también; antes salía la
//! suma congelada hasta la ruina (0 si se arruinaba en el primer cierre), que
//! la hacía comparable con las vivas.

use quantum_arena::{
    symbol_registry::{update_registry, SymbolSpec},
    GlobalArena,
};
use risk_engine::{MedidaOrden, RiskEngine};
use signal_engine::{SignalIntent, SignalType};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Mutex;

/// El registro de símbolos es global del proceso.
static SERIE: Mutex<()> = Mutex::new(());

/// La orden madura de `lh04_sizing_riesgo_contract` (n = 100, p̂ = 0,65,
/// PF = 2,6, f ≈ 0,13) y la medida que deja pendiente.
fn orden_medida() -> (std::sync::Arc<GlobalArena>, MedidaOrden) {
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
    arena.coins[0].metrics.win_rate.store(0.65, Relaxed);
    arena.coins[0].metrics.trade_count.store(100, Relaxed);
    arena.coins[0].metrics.profit_factor.store(2.6, Relaxed);
    let intent = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    };
    let mut engine = RiskEngine::new(1000.0);
    let orden = engine.evaluate_quantum_order(0, &intent, &arena);
    assert!(orden.volume_usd > 0.0, "premisa: LH-04 valida la orden");
    let medida = engine
        .tomar_medida_validada(0)
        .unwrap_or_else(|| panic!("la orden validada deja su medida"));
    (arena, medida)
}

fn clave(arena: &GlobalArena, nombre: &str) -> Option<f64> {
    arena.registry.get_for_coin_strict(0, nombre)
}

#[test]
fn cl164_una_perdida_suma_el_log_de_cada_regla_y_el_minimo_pierde_menos() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, medida) = orden_medida();
    RiskEngine::registrar_sombra_kelly(&arena, &medida, -0.002);
    let lh04 = clave(&arena, "sombra_kelly_logw_lh04").unwrap_or(f64::NAN);
    let legado = clave(&arena, "sombra_kelly_logw_legado").unwrap_or(f64::NAN);
    let minimo = clave(&arena, "sombra_kelly_logw_min").unwrap_or(f64::NAN);
    assert!(lh04 < 0.0 && legado < 0.0, "una pérdida resta en las dos: {lh04} {legado}");
    assert_ne!(lh04.to_bits(), legado.to_bits(), "premisa: las dos apuestas difieren");
    assert_eq!(minimo.to_bits(), lh04.max(legado).to_bits(), "con R < 0 la apuesta menor pierde menos");
    assert_eq!(clave(&arena, "sombra_kelly_n"), Some(1.0));
}

#[test]
fn cl164_una_ganancia_suma_menos_en_el_minimo() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, medida) = orden_medida();
    RiskEngine::registrar_sombra_kelly(&arena, &medida, 0.004);
    let lh04 = clave(&arena, "sombra_kelly_logw_lh04").unwrap_or(f64::NAN);
    let legado = clave(&arena, "sombra_kelly_logw_legado").unwrap_or(f64::NAN);
    let minimo = clave(&arena, "sombra_kelly_logw_min").unwrap_or(f64::NAN);
    assert!(lh04 > 0.0 && legado > 0.0);
    assert_eq!(minimo.to_bits(), lh04.min(legado).to_bits());
}

#[test]
fn cl164_un_hueco_mas_alla_del_stop_cuenta_ruina_sin_nan() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, medida) = orden_medida();
    let f_lh04 = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    assert!(f_lh04 > 0.1, "premisa: f de LH-04 ≈ 0,13, obtenido {f_lh04}");
    // Perder todo el nocional es R = −1/(SL + c), decenas de stops: con
    // f ≈ 0,13 el capital de la regla LH-04 no sobrevive.
    RiskEngine::registrar_sombra_kelly(&arena, &medida, -1.0);
    assert_eq!(clave(&arena, "sombra_kelly_ruina_lh04"), Some(1.0));
    assert_eq!(clave(&arena, "sombra_kelly_logw_lh04"), None, "la ruina no suma un log infinito");
    for nombre in [
        "sombra_kelly_logw_legado",
        "sombra_kelly_logw_min",
        "sombra_kelly_ruina_legado",
        "sombra_kelly_ruina_min",
        "sombra_kelly_n",
    ] {
        if let Some(v) = clave(&arena, nombre) {
            assert!(v.is_finite(), "{nombre} = {v}");
        }
    }
    assert_eq!(clave(&arena, "sombra_kelly_n"), Some(1.0));
}

#[test]
fn cl164_un_retorno_no_finito_no_se_registra() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, medida) = orden_medida();
    RiskEngine::registrar_sombra_kelly(&arena, &medida, f64::NAN);
    RiskEngine::registrar_sombra_kelly(&arena, &medida, f64::INFINITY);
    assert_eq!(clave(&arena, "sombra_kelly_n"), None);
    assert_eq!(clave(&arena, "sombra_kelly_logw_lh04"), None);
}

/// R de un cierre leído del log de LH-04: ln(1 + f·R) con f = `lh04_f_riesgo`.
fn r_de_lh04(arena: &GlobalArena, logw: f64) -> f64 {
    let f = arena.registry.get_for_coin_or(0, "lh04_f_riesgo", f64::NAN);
    logw.exp_m1() / f
}

#[test]
fn cl166_la_regla_arruinada_se_congela() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, medida) = orden_medida();
    RiskEngine::registrar_sombra_kelly(&arena, &medida, -1.0);
    assert_eq!(clave(&arena, "sombra_kelly_ruina_lh04"), Some(1.0), "premisa: el hueco arruina a LH-04");
    let tras_hueco: Vec<_> = ["lh04", "legado", "min"]
        .iter()
        .map(|r| {
            (
                *r,
                clave(&arena, &format!("sombra_kelly_ruina_{r}")) == Some(1.0),
                clave(&arena, &format!("sombra_kelly_logw_{r}")),
            )
        })
        .collect();
    RiskEngine::registrar_sombra_kelly(&arena, &medida, 0.01);
    for (regla, arruinada, logw) in tras_hueco {
        let ahora = clave(&arena, &format!("sombra_kelly_logw_{regla}"));
        if arruinada {
            assert_eq!(
                ahora.map(f64::to_bits),
                logw.map(f64::to_bits),
                "{regla}: una regla arruinada no compone después"
            );
            assert_eq!(clave(&arena, &format!("sombra_kelly_ruina_{regla}")), Some(1.0));
        } else {
            assert_ne!(ahora.map(f64::to_bits), logw.map(f64::to_bits), "{regla}: la viva sigue sumando");
        }
    }
    assert_eq!(clave(&arena, "sombra_kelly_n"), Some(2.0), "el cierre cuenta aunque las reglas se arruinen");
}

#[test]
fn cl166_el_libro_suma_r_en_unidades_de_stop() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, medida) = orden_medida();
    let sl = arena.registry.get_for_coin_or(0, "lh04_stop_real", f64::NAN);
    assert!(sl > 0.0, "premisa: la orden publica su stop real");
    // Tocar el stop sin fricción en la salida: R = −SL/(SL + c), algo más que −1.
    // RV8-T: c es la fricción de ida y vuelta del fixture (taker 4 pb, suelo
    // de deslizamiento 0,1 pb, ATR 1 %, latencia 0), la misma función que usa
    // el riesgo. Un divisor sin c daría R = −1 y la prueba lo ve.
    let c = risk_engine::tp_sl::roundtrip_friction(0.0004, 0.00001, 0.01, 0.0);
    assert!(c > 0.0008, "premisa: la fricción cuenta al menos las dos comisiones: {c}");
    RiskEngine::registrar_sombra_kelly(&arena, &medida, -sl);
    let r1 = r_de_lh04(&arena, clave(&arena, "sombra_kelly_logw_lh04").unwrap_or(f64::NAN));
    let esperado = -sl / (sl + c);
    assert!(
        (r1 - esperado).abs() < 1e-9,
        "un stop es R = −SL/(SL + c) = {esperado}, no −1: {r1}"
    );
    let suma_r = clave(&arena, "sombra_kelly_suma_r").unwrap_or(f64::NAN);
    assert!((suma_r - r1).abs() < 1e-9 * r1.abs(), "suma de R {suma_r} frente a R del log {r1}");
    RiskEngine::registrar_sombra_kelly(&arena, &medida, 2.0 * sl);
    let suma_r = clave(&arena, "sombra_kelly_suma_r").unwrap_or(f64::NAN);
    let suma_r2 = clave(&arena, "sombra_kelly_suma_r2").unwrap_or(f64::NAN);
    assert!((suma_r - (r1 - 2.0 * r1)).abs() < 1e-9, "R suma: {suma_r}");
    assert!((suma_r2 - 5.0 * r1 * r1).abs() < 1e-9, "R² suma: {suma_r2}");
}

#[test]
fn cl166_el_informe_publica_el_libro() {
    let _serie = SERIE.lock().unwrap_or_else(|e| e.into_inner());
    let (arena, medida) = orden_medida();
    assert_eq!(RiskEngine::informe_sombra_kelly(&arena, 0), None, "sin cierres no hay informe");
    let sl = arena.registry.get_for_coin_or(0, "lh04_stop_real", f64::NAN);
    RiskEngine::registrar_sombra_kelly(&arena, &medida, -sl);
    RiskEngine::registrar_sombra_kelly(&arena, &medida, -1.0);
    let linea = RiskEngine::informe_sombra_kelly(&arena, 0).unwrap_or_default();
    println!("observado: {linea}");
    assert!(linea.starts_with("{\"coin\":0,\"n\":2,"), "{linea}");
    assert!(linea.contains("\"ruina_lh04\":true"), "el hueco arruina a LH-04: {linea}");
    // CL-168: la log-riqueza de una regla arruinada es −∞; el informe la
    // publica como null, no como la suma congelada hasta la ruina.
    assert!(linea.contains("\"logw_lh04\":null"), "{linea}");
    let vivas: Vec<&str> = ["legado", "min"]
        .into_iter()
        .filter(|r| clave(&arena, &format!("sombra_kelly_ruina_{r}")).unwrap_or(0.0) == 0.0)
        .collect();
    for regla in vivas {
        let logw = clave(&arena, &format!("sombra_kelly_logw_{regla}")).unwrap_or(0.0);
        assert!(linea.contains(&format!("\"logw_{regla}\":{logw:.6}")), "{regla} viva: {linea}");
    }
    let r_media = clave(&arena, "sombra_kelly_suma_r").unwrap_or(f64::NAN) / 2.0;
    assert!(linea.contains(&format!("\"r_media\":{r_media:.6}")), "{linea}");
    assert!(linea.ends_with('}') && !linea.contains("NaN") && !linea.contains("inf"), "{linea}");
}
