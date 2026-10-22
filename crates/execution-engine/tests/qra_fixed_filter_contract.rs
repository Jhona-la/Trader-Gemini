//! QRA-01: oráculos de unidades enteras y fracciones binarias, independientes
//! del redondeo f64 del cuantizador. No certifica envío de órdenes ni latencia.

use execution_engine::fixed_filter::{count_decimals, BinanceFixedFilter, FilterRejectReason};

// Lee los dígitos que recibiría el exchange; no vuelve a cuantizar con f64.
fn ascii_ratio(text: &str) -> (i128, i128) {
    let mut numerator = 0_i128;
    let mut denominator = 1_i128;
    let mut fraction = false;
    for byte in text.bytes() {
        if byte == b'.' {
            assert!(!fraction);
            fraction = true;
        } else {
            assert!(byte.is_ascii_digit());
            numerator = numerator * 10 + i128::from(byte - b'0');
            if fraction {
                denominator *= 10;
            }
        }
    }
    (numerator, denominator)
}

#[test]
fn qra_quantity_floor_matches_integer_oracle_at_large_indices() -> Result<(), FilterRejectReason> {
    // 1/128 y los medios pasos son exactamente representables en binario.
    let filter =
        BinanceFixedFilter::with_limits(1.0, 0.0078125, 1.0, 10.0, 0.0078125, 10_000_000.0, 0.0);
    for index in [1_i128, 2, 123, 1_000_000, 1_000_000_000] {
        let raw_numerator = 2 * index + 1;
        let raw = raw_numerator as f64 / 256.0;
        let order = filter.validate_and_quantize(1.0, raw, false)?;
        assert_eq!(order.qty, index as f64 / 128.0);
        assert!(order.qty <= raw);
        let (wire, denominator) = ascii_ratio(order.qty_str.as_str());
        assert_eq!(wire * 128, index * denominator);
        assert!(wire * 256 <= raw_numerator * denominator);
    }
    Ok(())
}

#[test]
fn qra_one_float_below_grid_point_cannot_round_up() -> Result<(), FilterRejectReason> {
    let filter =
        BinanceFixedFilter::with_limits(1.0, 0.0078125, 1.0, 10.0, 0.0078125, 10_000_000.0, 0.0);
    for index in [2_u64, 128, 1_000_000, 1_000_000_000] {
        let grid = index as f64 / 128.0;
        let raw = f64::from_bits(grid.to_bits() - 1);
        let order = filter.validate_and_quantize(1.0, raw, false)?;
        assert_eq!(order.qty, (index - 1) as f64 / 128.0);
        assert!(order.qty <= raw);
        let (wire, denominator) = ascii_ratio(order.qty_str.as_str());
        assert_eq!(wire * 128, i128::from(index - 1) * denominator);
    }
    Ok(())
}

#[test]
fn qra_fractional_step_above_one_preserves_wire_value() -> Result<(), FilterRejectReason> {
    assert_eq!(count_decimals(2.5), 1);
    assert_eq!(count_decimals(12.25), 2);
    let filter = BinanceFixedFilter::with_limits(2.5, 2.5, 2.5, 100.0, 2.5, 100.0, 0.0);
    let buy = filter.validate_and_quantize(12.25, 7.75, false)?;
    let sell = filter.validate_and_quantize(12.25, 7.75, true)?;
    assert_eq!(buy.price_str.as_str(), "10.0");
    assert_eq!(sell.price_str.as_str(), "12.5");
    assert_eq!(buy.qty_str.as_str(), "7.5");
    let (wire, denominator) = ascii_ratio(buy.qty_str.as_str());
    assert_eq!(wire * 2, 15 * denominator);
    assert!(wire * 4 <= 31 * denominator); // 7.5 <= 7.75, con enteros.
    Ok(())
}

#[test]
fn qra_grid_is_anchored_at_exchange_minimum() -> Result<(), FilterRejectReason> {
    // El mínimo no tiene por qué ser múltiplo del paso.
    let filter = BinanceFixedFilter::with_limits(0.1, 0.1, 0.15, 10.0, 0.15, 10.0, 0.0);
    let buy = filter.validate_and_quantize(0.20, 0.26, false)?;
    let sell = filter.validate_and_quantize(0.20, 0.26, true)?;
    assert_eq!(buy.price_str.as_str(), "0.15");
    assert_eq!(sell.price_str.as_str(), "0.25");
    assert_eq!(buy.qty_str.as_str(), "0.25");
    for text in [
        buy.price_str.as_str(),
        sell.price_str.as_str(),
        buy.qty_str.as_str(),
    ] {
        let (units, denominator) = ascii_ratio(text);
        assert_eq!((units * 100 - 15 * denominator) % (10 * denominator), 0);
    }
    Ok(())
}

#[test]
fn qra_out_of_range_input_is_rejected_without_clamping() {
    let filter = BinanceFixedFilter::with_limits(1.0, 1.0, 1.0, 10.0, 1.0, 10.0, 0.0);
    assert!(matches!(
        filter.validate_and_quantize(0.5, 2.0, false),
        Err(FilterRejectReason::PriceBelowMin { .. })
    ));
    assert!(matches!(
        filter.validate_and_quantize(10.5, 2.0, true),
        Err(FilterRejectReason::PriceAboveMax { .. })
    ));
    assert!(matches!(
        filter.validate_and_quantize(2.0, 10.5, false),
        Err(FilterRejectReason::QtyAboveMax { .. })
    ));
    let non_aligned_max = BinanceFixedFilter::with_limits(2.0, 1.0, 1.0, 10.0, 1.0, 10.0, 0.0);
    assert!(matches!(
        non_aligned_max.validate_and_quantize(9.9, 2.0, true),
        Err(FilterRejectReason::PriceAboveMax { current, max }) if current == 11.0 && max == 10.0
    ));
}

#[test]
fn qra_corrupt_configuration_is_not_repaired_with_defaults() {
    for invalid in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
        let filter = BinanceFixedFilter::with_limits(0.1, 0.1, invalid, 100.0, 0.1, 100.0, 0.0);
        assert!(!filter.is_valid_config());
        assert!(matches!(
            filter.validate_and_quantize(1.0, 1.0, false),
            Err(FilterRejectReason::InvalidFilterConfig)
        ));
        if invalid.is_nan() {
            assert!(filter.min_price.is_nan());
        } else {
            assert_eq!(filter.min_price, invalid);
        }
    }
    for invalid in [f64::NAN, f64::INFINITY, -1.0] {
        let filter = BinanceFixedFilter::with_limits(0.1, 0.1, 0.1, 100.0, 0.1, 100.0, invalid);
        assert!(!filter.is_valid_config());
    }
    // Un mínimo nocional cero válido se conserva como cero, no se inventan $5.
    let no_minimum = BinanceFixedFilter::with_limits(0.1, 0.1, 0.1, 100.0, 0.1, 100.0, 0.0);
    assert_eq!(no_minimum.min_notional, 0.0);
    assert!(no_minimum.validate_and_quantize(0.1, 0.1, false).is_ok());
}

#[test]
fn qra_precision_and_reciprocal_corruption_fail_closed() {
    let mut filter = BinanceFixedFilter::new(0.1, 0.001, 0.0);
    filter.qty_decimals = u8::MAX;
    assert!(!filter.is_valid_config());
    assert!(filter.format_qty(1.0).is_empty());
    filter = BinanceFixedFilter::new(0.1, 0.001, 0.0);
    filter.inv_step = 2.0;
    assert!(!filter.is_valid_config());
    let tiny = BinanceFixedFilter::new(1.0e-320, 0.001, 0.0);
    assert!(!tiny.is_valid_config());
    assert!(matches!(
        tiny.validate_and_quantize(1.0, 1.0, false),
        Err(FilterRejectReason::InvalidFilterConfig)
    ));
}

#[test]
fn qra_nonfinite_and_unrepresentable_values_are_not_serialized() {
    let filter = BinanceFixedFilter::with_limits(1.0, 1.0, 1.0, 1.0e100, 1.0, 1.0e100, 0.0);
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            filter.validate_and_quantize(1.0, invalid, false),
            Err(FilterRejectReason::NonFinite)
        ));
        assert!(filter.format_qty(invalid).is_empty());
    }
    assert!(matches!(
        filter.validate_and_quantize(1.0, 1.0e80, false),
        Err(FilterRejectReason::Unrepresentable)
    ));
    assert!(filter.format_qty(f64::MAX).is_empty());
}

#[test]
fn qra_public_formatter_preserves_decimal_rounding_for_noisy_floats() {
    let filter = BinanceFixedFilter::new(0.01, 0.001, 0.0);
    assert_eq!(
        filter.format_price(3500.25 + 3.0 * 0.01).as_str(),
        "3500.28"
    );
    assert_eq!(filter.format_price(3500.254).as_str(), "3500.25");
    assert_eq!(filter.format_qty(1.2344).as_str(), "1.234");
}

#[test]
fn qra_decimal_minimum_above_binary_raw_is_rejected_conservatively() {
    let filter = BinanceFixedFilter::with_limits(0.01, 0.01, 0.29, 10.0, 0.29, 10.0, 0.0);
    assert!(matches!(
        filter.validate_and_quantize(0.29, 0.29, false),
        Err(FilterRejectReason::Unrepresentable)
    ));
}

#[test]
fn qra_wire_decimal_respects_exact_ieee754_raw_bound() -> Result<(), FilterRejectReason> {
    let filter = BinanceFixedFilter::with_limits(0.1, 0.1, 0.1, 10.0, 0.1, 10.0, 0.0);
    let buy = filter.validate_and_quantize(0.3, 0.3, false)?;
    let sell = filter.validate_and_quantize(0.3, 0.3, true)?;
    // Valor racional exacto de f64::from_bits(0x3fd3333333333333), no decimal 3/10.
    let raw_numerator = 5_404_319_552_844_595_i128;
    let raw_denominator = 18_014_398_509_481_984_i128;
    assert_eq!(0.3_f64.to_bits(), 0x3fd3_3333_3333_3333);
    assert_eq!(buy.qty_str.as_str(), "0.2");
    assert_eq!(buy.price_str.as_str(), "0.2");
    assert_eq!(sell.price_str.as_str(), "0.3");
    for text in [
        buy.qty_str.as_str(),
        sell.qty_str.as_str(),
        buy.price_str.as_str(),
    ] {
        let (wire, denominator) = ascii_ratio(text);
        assert!(wire * raw_denominator <= raw_numerator * denominator);
    }
    let (wire, denominator) = ascii_ratio(sell.price_str.as_str());
    assert!(wire * raw_denominator >= raw_numerator * denominator);
    Ok(())
}
