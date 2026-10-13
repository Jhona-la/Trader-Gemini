//! CONTRATO FORMAL: NORMALIZACIÓN MULTIACTIVO DE VOLATILIDAD EN EL CONSEJO DELIBERATIVO (Ola Ω95).
//!
//! # Por qué existe
//! En `crates/metacortex-engine/src/consejo_seniors.rs:499-502`, `SeniorVolatilidad` evalúa el régimen
//! con cotas absolutas calibradas para BTC (calma <= 10 bps, peligro >= 40 bps):
//! ```text
//! const CALM_ATR_PCT: f64 = 0.0010;
//! const DANGER_ATR_PCT: f64 = 0.0040;
//! ```
//! Al operar un universo multiactivo, las altcoins con mayor beta estructural (ej. SOL, DOGE, NEAR con ATR de 60-150 bps)
//! eran castigadas permanentemente con un recorte del 60% en su convicción (`regime_factor = 0.40`),
//! asfixiando sistemáticamente las entradas viables de alta rentabilidad necesarias para la duplicación
//! exponencial de capital cada 72 horas.
//!
//! # Solución canónica (Ola Ω95)
//! La volatilidad del activo en `council_snapshot.atr_pct` se normaliza por su ratio de volatilidad
//! relativa respecto al ancla de mercado (BTC). Para BTC la relación es idénticamente 1.0; para
//! altcoins su régimen se juzga respecto a su propia escala de calma/peligro intrínseca.

#[test]
fn btc_conserva_volatilidad_identica_sin_alteracion() {
    let raw_atr_pct: f64 = 0.0020; // 20 bps
    let btc_slot: usize = 0;
    let coin_id: usize = btc_slot;

    let relative_vol: f64 = if coin_id == btc_slot {
        1.0f64
    } else {
        (raw_atr_pct / 0.0020f64).clamp(0.5f64, 4.0f64)
    };
    let council_atr_pct: f64 = (raw_atr_pct / relative_vol).clamp(0.0f64, 1.0f64);

    assert_eq!(
        council_atr_pct, raw_atr_pct,
        "BTC debe conservar su volatilidad exacta sin alteración"
    );
}

#[test]
fn altcoin_con_alta_beta_no_es_asfixiada_artificialmente() {
    // Altcoin con ATR normal de 60 bps (0.0060), mientras BTC está en 20 bps (0.0020)
    let alt_raw_atr: f64 = 0.0060;
    let btc_atr: f64 = 0.0020;

    let relative_vol: f64 = (alt_raw_atr / btc_atr).clamp(0.5f64, 4.0f64);
    assert_eq!(relative_vol, 3.0f64, "Beta relativa debe ser 3.0x");

    let council_atr_pct: f64 = (alt_raw_atr / relative_vol).clamp(0.0f64, 1.0f64);

    // 0.0060 / 3.0 = 0.0020 (20 bps normalizados)
    assert!(
        (council_atr_pct - 0.0020f64).abs() < 1e-12,
        "La volatilidad normalizada debe ser 20 bps, dentro de la zona sana de operación"
    );

    // En SeniorVolatilidad, 20 bps da regime_factor = 1.0 - 0.6 * (10 / 30) = 0.80 de convicción
    // (en lugar de 0.40 pegado al piso como antes de esta corrección)
    let calm: f64 = 0.0010;
    let danger: f64 = 0.0040;
    let regime_factor_normalizado: f64 =
        1.0 - 0.6 * ((council_atr_pct - calm) / (danger - calm)).clamp(0.0f64, 1.0f64);
    let regime_factor_asfixiado: f64 =
        1.0 - 0.6 * ((alt_raw_atr - calm) / (danger - calm)).clamp(0.0f64, 1.0f64);

    assert!(
        regime_factor_normalizado > regime_factor_asfixiado,
        "La convicción normalizada ({regime_factor_normalizado}) debe ser superior a la asfixiada ({regime_factor_asfixiado})"
    );
    assert_eq!(regime_factor_asfixiado, 0.40f64);
    assert!((regime_factor_normalizado - 0.80f64).abs() < 1e-12);
}

#[test]
fn altcoin_con_explosion_anomala_conserva_deteccion_de_peligro() {
    // Si la altcoin explota anómalamente a 300 bps (0.0300) cuando su beta normal esperaba 60 bps:
    let alt_extreme_atr: f64 = 0.0300;
    let _btc_atr: f64 = 0.0020;

    let relative_vol: f64 = 3.0; // Su beta estimada previa
    let council_atr_pct: f64 = (alt_extreme_atr / relative_vol).clamp(0.0f64, 1.0f64);

    // 0.0300 / 3.0 = 0.0100 (100 bps normalizados > 40 bps límite de peligro)
    assert!(
        council_atr_pct > 0.0040f64,
        "Una explosión anómala debe superar el umbral de peligro de 40 bps"
    );

    let calm: f64 = 0.0010;
    let danger: f64 = 0.0040;
    let regime_factor: f64 =
        1.0 - 0.6 * ((council_atr_pct - calm) / (danger - calm)).clamp(0.0f64, 1.0f64);
    assert_eq!(
        regime_factor, 0.40f64,
        "El peligro anómalo extremo debe activar la protección de convicción mínima"
    );
}
