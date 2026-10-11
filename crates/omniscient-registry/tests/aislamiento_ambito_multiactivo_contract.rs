//! Contrato de Aislamiento de Ámbito Multi-Activo y Rigor Dimensional (LC-F1-06 / D-219 / F-002)
//!
//! Este contrato verifica que:
//! 1. `get_scoped_parameter_strict` y `get_scoped_val_strict_or` aíslan estrictamente los
//!    parámetros y métricas de cada activo por símbolo y por coin_id sin caer al parámetro global.
//! 2. `get_for_coin_strict` y `get_for_coin_strict_or` no heredan valores globales de otra moneda.
//! 3. Las métricas de un activo (ej. BTC) nunca contaminan las consultas de otro activo (ej. ETH),
//!    preservando el principio fundamental de que "desconocido es desconocido, no el valor de otro".

use omniscient_registry::OmniscientRegistry;

#[test]
fn test_aislamiento_estricto_sin_caida_al_global() {
    let registry = OmniscientRegistry::new();

    // 1. Escribir parámetro global "mid_price" = 50_000.0 (contaminante potencial)
    registry.set("mid_price", 50_000.0);

    // Escribir métrica escopada para BTCUSDT (coin 0)
    registry.set_scoped("BTCUSDT", "mid_price", 60_000.0);
    registry.set_for_coin(0, "mid_price", 60_000.0);

    // 2. Consulta para BTCUSDT (coin 0): debe obtener 60_000.0
    let btc_val = registry.get_scoped_val_strict_or(Some("BTCUSDT"), Some(0), "mid_price", f64::NAN);
    assert!(
        (btc_val - 60_000.0).abs() < 1e-6,
        "BTCUSDT debe leer su propio valor escopado, obtenido: {btc_val}"
    );

    // 3. Consulta estricta para ETHUSDT (coin 1) que NO tiene valor registrado:
    // Con get_scoped_val_or (antiguo con fallback), caería a 50_000.0 (global contaminado)
    let eth_val_con_fallback = registry.get_scoped_val_or(Some("ETHUSDT"), Some(1), "mid_price", 0.0);
    assert!(
        (eth_val_con_fallback - 50_000.0).abs() < 1e-6,
        "get_scoped_val_or con fallback hereda globalmente (demuestra el comportamiento heredado)"
    );

    // Con get_scoped_val_strict_or, NUNCA cae al global: devuelve default (NaN)
    let eth_val_estricto = registry.get_scoped_val_strict_or(Some("ETHUSDT"), Some(1), "mid_price", f64::NAN);
    assert!(
        eth_val_estricto.is_nan(),
        "ETHUSDT debe devolver NaN en lectura estricta, jamás heredar el global: obtenido: {eth_val_estricto}"
    );

    // 4. Parámetro estricto get_scoped_parameter_strict
    let btc_param = registry.get_scoped_parameter_strict(Some("BTCUSDT"), Some(0), "mid_price", "test");
    assert!(btc_param.is_some(), "BTCUSDT debe resolver su parámetro escopado");
    assert!((btc_param.unwrap().get_value() - 60_000.0).abs() < 1e-6);

    let eth_param = registry.get_scoped_parameter_strict(Some("ETHUSDT"), Some(1), "mid_price", "test");
    assert!(
        eth_param.is_none(),
        "ETHUSDT no tiene parámetro registrado: debe ser None, nunca el global"
    );
}

#[test]
fn test_aislamiento_por_coin_id_estricto() {
    let registry = OmniscientRegistry::new();

    // Escribir solo en global y en coin 0
    registry.set("ml_model_base", 0.30);
    registry.set_for_coin(0, "ml_model_base", 0.30);

    // Lectura estricta para coin 0
    let coin0_val = registry.get_for_coin_strict_or(0, "ml_model_base", f64::NAN);
    assert!((coin0_val - 0.30).abs() < 1e-6);

    // Lectura estricta para coin 1 (sin registro para coin 1):
    // Debe devolver NaN, no caer a 0.30 (que es el valor de coin 0 en el global)
    let coin1_val = registry.get_for_coin_strict_or(1, "ml_model_base", f64::NAN);
    assert!(
        coin1_val.is_nan(),
        "Coin 1 debe devolver NaN (desconocido), no heredar 0.30 de coin 0: obtenido: {coin1_val}"
    );
}

#[test]
fn test_resolucion_precedencia_simbolo_sobre_coin_id() {
    let registry = OmniscientRegistry::new();

    // Símbolo específico tiene precedencia sobre coin_id
    registry.set_for_coin(0, "atr_1s", 10.0);
    registry.set_scoped("BTCUSDT", "atr_1s", 15.0);

    let val = registry.get_scoped_val_strict_or(Some("BTCUSDT"), Some(0), "atr_1s", 0.0);
    assert_eq!(val, 15.0, "El ámbito por símbolo debe preceder al de coin_id");

    // Si falta por símbolo, recurre a coin_id
    let val_cid = registry.get_scoped_val_strict_or(Some("OTHER"), Some(0), "atr_1s", 0.0);
    assert_eq!(val_cid, 10.0, "Si no hay clave por símbolo, recurre a coin_id");
}
