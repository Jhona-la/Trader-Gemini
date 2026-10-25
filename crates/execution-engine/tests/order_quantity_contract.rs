//! F325 (censo C-F7b, lote 34; RET-34b): contrato de la cuantización de la
//! cantidad de una orden, migrado de quantum-arena/tests/order_quantity_contract.rs
//! (FMT/Codex XVIII) y de las dos regresiones FMT-175 de
//! universe_identity_diagnostics.rs al cuantizador canónico
//! `BinanceFixedFilter::validate_and_quantize`, al retirar
//! `SymbolSpec::validate_order`, que sólo llamaban las pruebas.
//!
//! Viejo frente a nuevo, caso por caso:
//! - Mismo veredicto: entradas no finitas, paso o mínimos inválidos, menos de
//!   un lote, retículo minQty + k·paso, minNotional DESPUÉS de redondear y el
//!   borde binario 0,3 → 0,2 con paso 0,1. Límite de 2^53: mismo veredicto con
//!   paso entero (el caso probado); con paso fraccionario el canónico es más
//!   estricto, porque cuenta unidades decimales (raw·10^d) y no lotes.
//! - Más estricto: minQty = 0 es configuración inválida (validate_order la
//!   aceptaba y caía en «sin lote positivo»).
//! - Más exacto: la cantidad es el decimal del retículo (0,075), no la suma
//!   f64 minQty + k·paso (0,07500000000000001), y viaja como cadena exacta.
//! - Distinto (RV6): el canónico redondea el precio al tick (abajo en compras,
//!   arriba en ventas) ANTES de juzgar minNotional; validate_order juzgaba
//!   al precio crudo (qty 4,15, precio 1,2199, paso 0,1: el viejo daba 4,1 y
//!   5,0016 USD, el canónico juzga 1,219·4,1 = 4,9979 y rechaza).
//!   `from_symbol_spec` fija min_price = tick y techos max_price = max_qty
//!   = 1e7, que el viejo no tenía, y `is_valid_config` rechaza pasos o
//!   mínimos sin decimal exacto de ≤ 10 cifras (validate_order aceptaba
//!   cualquier paso finito positivo).
use execution_engine::fixed_filter::{BinanceFixedFilter, FilterRejectReason, QuantizedOrder};
use quantum_arena::symbol_registry::SymbolSpec;

fn spec(step: f64) -> SymbolSpec {
    SymbolSpec {
        symbol: "AUDITUSDT".into(),
        step_size: step,
        tick_size: 0.01,
        min_qty: step,
        min_notional: 0.0,
        max_leverage: 1,
        maker_fee: 0.0,
        taker_fee: 0.0,
        is_shadow: false,
    }
}

fn filtro(s: &SymbolSpec) -> BinanceFixedFilter {
    BinanceFixedFilter::from_symbol_spec(s)
}

fn cantidad(s: &SymbolSpec, raw_qty: f64) -> Result<f64, FilterRejectReason> {
    filtro(s)
        .validate_and_quantize(100.0, raw_qty, false)
        .map(|o| o.qty)
}

/// La orden cuantizada a precio 100, o el fallo del test con su causa.
fn orden(f: &BinanceFixedFilter, raw_qty: f64, contexto: &str) -> Option<QuantizedOrder> {
    let r = f.validate_and_quantize(100.0, raw_qty, false);
    assert!(r.is_ok(), "{contexto}: {r:?}");
    r.ok()
}

/// Sin los techos de compatibilidad de 1e7 de `from_symbol_spec`: así se
/// ejercita el límite de representación y no maxQty/maxPrice.
fn sin_techos(step: f64) -> BinanceFixedFilter {
    BinanceFixedFilter::with_limits(0.01, step, 0.01, f64::MAX, step, f64::MAX, 0.0)
}

#[test]
fn nonfinite_order_inputs_are_errors() {
    let f = filtro(&spec(0.05));
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
        assert!(f.validate_and_quantize(100.0, x, false).is_err(), "quantity={x}");
        assert!(f.validate_and_quantize(x, 1.0, false).is_err(), "price={x}");
    }
}

#[test]
fn invalid_step_is_not_replaced_with_one() {
    for x in [f64::NAN, f64::INFINITY, 0.0, -0.01] {
        assert_eq!(
            cantidad(&spec(x), 10.0),
            Err(FilterRejectReason::InvalidFilterConfig),
            "step={x}"
        );
    }
}

#[test]
fn invalid_minima_are_errors() {
    for x in [f64::NAN, f64::INFINITY, -1.0] {
        let mut s = spec(0.05);
        s.min_qty = x;
        assert_eq!(cantidad(&s, 10.0), Err(FilterRejectReason::InvalidFilterConfig));
        s.min_qty = 0.05;
        s.min_notional = x;
        assert_eq!(cantidad(&s, 10.0), Err(FilterRejectReason::InvalidFilterConfig));
    }
}

#[test]
fn zero_min_qty_is_invalid_configuration() {
    // validate_order aceptaba minQty = 0 y, con 0,01 a paso 0,05, devolvía
    // «sin lote positivo»; el canónico rechaza la configuración entera.
    let mut s = spec(0.05);
    s.min_qty = 0.0;
    assert_eq!(cantidad(&s, 0.01), Err(FilterRejectReason::InvalidFilterConfig));
}

#[test]
fn non_power_of_ten_step_does_not_increase_quantity() {
    assert_eq!(cantidad(&spec(0.05), 0.06), Ok(0.05));
}

#[test]
fn non_power_of_ten_step_keeps_integer_lot_count() {
    let Some(o) = orden(&filtro(&spec(0.025)), 0.09, "3 lotes") else {
        return;
    };
    assert_eq!(o.qty_str.as_str(), "0.075");
    assert_eq!(o.qty, 0.075);
    assert!(o.qty <= 0.09);
}

#[test]
fn below_one_lot_and_overflow_are_errors() {
    assert!(matches!(
        cantidad(&spec(0.05), 0.01),
        Err(FilterRejectReason::QtyBelowMin { .. })
    ));
    assert!(sin_techos(1.0)
        .validate_and_quantize(f64::MAX, 2.0, false)
        .is_err());
    assert!(filtro(&spec(f64::MIN_POSITIVE))
        .validate_and_quantize(1.0, f64::MAX, false)
        .is_err());
}

#[test]
fn lot_count_must_have_unit_resolution() {
    let f = sin_techos(1.0);
    assert_eq!(
        f.validate_and_quantize(100.0, 2_f64.powi(53), false)
            .map(|o| o.qty),
        Err(FilterRejectReason::Unrepresentable)
    );
    let ultimo = 2_f64.powi(53) - 1.0;
    assert_eq!(
        f.validate_and_quantize(100.0, ultimo, false).map(|o| o.qty),
        Ok(ultimo)
    );
}

#[test]
fn conservative_projection_and_minimum_notional_hold() {
    for step in [0.025_f64, 0.05, 0.1, 0.125, 1.0, 2.5] {
        let f = filtro(&spec(step));
        let decimales = usize::from(f.qty_decimals);
        for n in 1..1000_u32 {
            let raw = (f64::from(n) + 0.375) * step;
            let Some(o) = orden(&f, raw, &format!("step={step} n={n}")) else {
                return;
            };
            // n lotes exactos: el decimal del retículo, no la suma f64.
            let esperado = format!("{:.*}", decimales, f64::from(n) * step);
            assert_eq!(o.qty_str.as_str(), esperado, "step={step} n={n}");
            assert_eq!(
                o.qty,
                esperado.parse::<f64>().unwrap_or(f64::NAN),
                "step={step} n={n}"
            );
            assert!(o.qty.is_finite() && o.qty > 0.0 && o.qty <= raw);
        }
    }
    let mut s = spec(0.05);
    s.min_notional = 10.0;
    assert!(matches!(
        cantidad(&s, 0.06),
        Err(FilterRejectReason::NotionalBelowMin { .. })
    ));
    assert_eq!(cantidad(&s, 0.125), Ok(0.1));
}

#[test]
fn minimum_quantity_is_the_lattice_origin() {
    let mut s = spec(0.025);
    s.min_qty = 0.01;
    let Some(o) = orden(&filtro(&s), 0.062, "minQty + 2 pasos") else {
        return;
    };
    assert_eq!(o.qty_str.as_str(), "0.060");
    assert_eq!(o.qty, 0.06);
    assert!(matches!(
        cantidad(&s, 0.009),
        Err(FilterRejectReason::QtyBelowMin { .. })
    ));
}

#[test]
fn binary_boundary_may_underfill_but_never_upsizes() {
    let q = cantidad(&spec(0.1), 0.3);
    assert!(q.is_ok(), "2 lotes: {q:?}");
    let q = q.unwrap_or(f64::NAN);
    assert!(q <= 0.3);
    // Límite conocido de toda API f64 (fixed_filter.rs:21-26): 0,3 llega como
    // 0,29999…; sólo una cantidad decimal de origen lo evitaría.
    assert_eq!(q, 0.2);
}

/// FMT-175 (antes en quantum-arena/tests/universe_identity_diagnostics.rs):
/// el spec del diagnóstico de universo, con minNotional 0,01.
fn spec_universo() -> SymbolSpec {
    SymbolSpec {
        symbol: "XVIIAAAUSDT".into(),
        step_size: 0.05,
        tick_size: 0.01,
        min_qty: 0.05,
        min_notional: 0.01,
        max_leverage: 20,
        maker_fee: 0.0002,
        taker_fee: 0.0004,
        is_shadow: false,
    }
}

#[test]
fn regression_order_validation_rejects_nonfinite_numbers() {
    let s = spec_universo();
    assert_eq!(
        filtro(&s)
            .validate_and_quantize(f64::NAN, 1.0, false)
            .map(|o| o.qty),
        Err(FilterRejectReason::NonFinite)
    );
    assert_eq!(cantidad(&s, f64::NAN), Err(FilterRejectReason::NonFinite));
}

#[test]
fn regression_rounding_preserves_lot_count_and_never_increases_quantity() {
    let mut s = spec_universo();
    // XVIII retiró el redondeo por número de decimales, que daba 0,1.
    assert_eq!(cantidad(&s, 0.06), Ok(0.05));
    s.step_size = 0.025;
    s.min_qty = 0.025;
    let Some(o) = orden(&filtro(&s), 0.09, "3 lotes") else {
        return;
    };
    assert_eq!(o.qty_str.as_str(), "0.075");
    assert_eq!(o.qty, 0.075);
}
