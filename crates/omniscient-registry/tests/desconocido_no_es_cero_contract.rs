use omniscient_registry::{OmniscientRegistry, Parameter, ParameterKind};

/// CONTRATO FORMAL C-F1 / OLA Ω82: «DESCONOCIDO NO ES CERO» & AISLAMIENTO MULTI-ACTIVO
///
/// Verifica formalmente:
/// 1. Desconocido ≠ 0: Valores `f64::NAN` y `±Inf` no se clompan silenciosamente a 0.0.
/// 2. `get_value_fast` preserva `NaN` para que los decisores puedan detectar falta de evidencia.
/// 3. `get_value_or` ante `NaN` devuelve el `default` seguro, nunca un 0.0 fabricado.
/// 4. `timestamp` atómico en `Parameter`: se actualiza en cada `set_value` / `set`.
/// 5. Aislamiento multi-activo: `get_for_coin_strict` y `get_for_coin_strict_or` garantizan
///    que una moneda nunca lea el valor de otra moneda si la suya está indefinida. Lo mismo por
///    símbolo: `get_scoped_strict` devuelve el dato crudo (NaN incluido, para que el decisor vea
///    la falta de evidencia), y `get_scoped_val_strict(_or)` no cae ni a otra moneda ni a la
///    clave global, y trata un NaN escrito como ausente.
#[test]
fn test_desconocido_no_es_cero_preservacion_nan() {
    let registry = OmniscientRegistry::new();

    // 1. Registro explícito de parámetro con NaN inicial
    let p_nan = Parameter::new("volatilidad_implicita", ParameterKind::Adaptive, f64::NAN, "test");
    assert!(p_nan.get_value().is_nan(), "El parámetro nuevo debe preservar NaN como desconocido");
    assert!(registry.register(p_nan).is_ok());

    // 2. Lectura fast preserva NaN
    let val_fast = registry.get_value_fast("volatilidad_implicita");
    assert!(val_fast.is_some(), "Debe existir en el mapa");
    assert!(val_fast.unwrap().is_nan(), "get_value_fast debe devolver Some(NaN), no 0.0");

    // 3. Lectura con fallback: get_value_or ante NaN debe devolver el default provisto, NUNCA 0.0
    let val_or = registry.get_value_or("volatilidad_implicita", 0.025);
    assert_eq!(val_or, 0.025, "get_value_or debe caer al default cuando el valor es NaN");

    // 4. Actualización a NaN no destruye a 0.0
    registry.set("spread_dinamico", 0.001);
    assert_eq!(registry.get_value_or("spread_dinamico", 0.05), 0.001);
    registry.set("spread_dinamico", f64::NAN);
    assert!(registry.get_value_fast("spread_dinamico").unwrap().is_nan());
    assert_eq!(registry.get_value_or("spread_dinamico", 0.05), 0.05);
}

#[test]
fn test_timestamp_atomico_frescura_en_actualizacion() {
    let registry = OmniscientRegistry::new();
    let t0 = chrono::Utc::now().timestamp_millis();
    registry.set("reynolds_l2", 1.5);

    let p = registry.get("reynolds_l2", "engine").expect("parametro existente");
    let ts_inicial = p.get_timestamp();
    assert!(ts_inicial >= t0, "Timestamp inicial debe ser >= t0");

    // Dormir breve intervalo para verificar avance temporal
    std::thread::sleep(std::time::Duration::from_millis(5));

    registry.set("reynolds_l2", 2.8);
    let ts_actualizado = p.get_timestamp();
    assert!(
        ts_actualizado >= ts_inicial,
        "Timestamp debe avanzar tras cada set_value (frescura viva garantizada)"
    );
    assert_eq!(p.get_value(), 2.8);
}

#[test]
fn test_aislamiento_estricto_entre_activos() {
    let registry = OmniscientRegistry::new();

    // Moneda 0 (BTC) publica su OFI
    registry.set_for_coin(0, "order_flow_imbalance", 0.85);

    // Moneda 1 (ETH) no ha publicado nada
    // 1. Moneda 0 lee su dato estricto
    assert_eq!(registry.get_for_coin_strict(0, "order_flow_imbalance"), Some(0.85));
    assert_eq!(registry.get_for_coin_strict_or(0, "order_flow_imbalance", 0.0), 0.85);

    // 2. Moneda 1 bajo consulta estricta devuelve None (aislamiento perfecto)
    assert_eq!(registry.get_for_coin_strict(1, "order_flow_imbalance"), None);
    assert_eq!(registry.get_for_coin_strict_or(1, "order_flow_imbalance", 0.0), 0.0);

    // 3. get_for_coin_or tradicional con default: Moneda 1 no ve el 0.85 de Moneda 0
    assert_eq!(registry.get_for_coin_or(1, "order_flow_imbalance", -0.1), -0.1);

    // 4. Idéntico comportamiento por símbolo (scoped)
    registry.set_scoped("BTCUSDT", "liquidity_density", 450.0);
    assert_eq!(registry.get_scoped_strict("BTCUSDT", "liquidity_density"), Some(450.0));
    assert_eq!(registry.get_scoped_strict("ETHUSDT", "liquidity_density"), None);
    assert_eq!(registry.get_scoped_val_strict_or(Some("ETHUSDT"), None, "liquidity_density", 10.0), 10.0);
    assert_eq!(registry.get_scoped_value_or("ETHUSDT", "liquidity_density", 10.0), 10.0);

    // 5. Con la clave global escrita, la lectura no estricta cae a ella y la estricta no.
    registry.set("liquidity_density", 99.0);
    assert_eq!(registry.get_scoped_value_or("ETHUSDT", "liquidity_density", 10.0), 99.0);
    assert_eq!(registry.get_scoped_val_strict_or(Some("ETHUSDT"), None, "liquidity_density", 10.0), 10.0);
    assert_eq!(registry.get_scoped_val_strict(Some("ETHUSDT"), None, "liquidity_density"), None);
    assert_eq!(registry.get_scoped_val_strict_or(Some("BTCUSDT"), None, "liquidity_density", 10.0), 450.0);

    // 6. Un NaN escrito por símbolo: la lectura cruda lo conserva y la estricta lo trata como ausente.
    registry.set_scoped("BTCUSDT", "liquidez_desconocida", f64::NAN);
    assert!(registry.get_scoped_strict("BTCUSDT", "liquidez_desconocida").is_some_and(f64::is_nan));
    assert_eq!(registry.get_scoped_val_strict(Some("BTCUSDT"), None, "liquidez_desconocida"), None);
    assert_eq!(registry.get_scoped_val_strict_or(Some("BTCUSDT"), None, "liquidez_desconocida", 10.0), 10.0);
}
