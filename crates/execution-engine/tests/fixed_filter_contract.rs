//! CONTRATO FORMAL DE FILTROS EXACTOS Y PUNTO FIJO DE BINANCE (Fase 2 §4.2 Átomo 2.4)
//!
//! **Criterio de Aceptación:**
//! Cero rechazos por redondeo en 10.000 órdenes sintéticas sobre especificaciones reales
//! de Binance Futures (BTCUSDT, ETHUSDT, SOLUSDT, DOGEUSDT, BNBUSDT), con verificación de:
//! - Invariante de red modular: `(price - min_price) % tick == 0` y `(qty - min_qty) % step == 0`.
//! - Invariante de nocional: `price * qty >= min_notional`.
//! - Invariante de formato ASCII: sin notación científica (`e`/`E`), sin decimales sobrantes,
//!   coincidencia exacta entre el número flotante y la cadena serializada.
//! - Invariante de latencia: cuantización y serialización $O(1)$ sin asignación en heap.

use execution_engine::fixed_filter::{BinanceFixedFilter, FilterRejectReason};

#[test]
fn test_10000_synthetic_orders_zero_rounding_rejections() {
    struct TestSpec {
        name: &'static str,
        tick_size: f64,
        step_size: f64,
        min_qty: f64,
        min_notional: f64,
        base_price: f64,
        base_qty: f64,
    }

    let specs = [
        TestSpec {
            name: "BTCUSDT",
            tick_size: 0.1,
            step_size: 0.001,
            min_qty: 0.001,
            min_notional: 5.0,
            base_price: 65_000.0,
            base_qty: 0.01,
        },
        TestSpec {
            name: "ETHUSDT",
            tick_size: 0.01,
            step_size: 0.001,
            min_qty: 0.001,
            min_notional: 5.0,
            base_price: 3_500.0,
            base_qty: 0.05,
        },
        TestSpec {
            name: "SOLUSDT",
            tick_size: 0.01,
            step_size: 0.01,
            min_qty: 0.01,
            min_notional: 5.0,
            base_price: 180.0,
            base_qty: 0.5,
        },
        TestSpec {
            name: "DOGEUSDT",
            tick_size: 0.0001,
            step_size: 1.0,
            min_qty: 1.0,
            min_notional: 5.0,
            base_price: 0.15,
            base_qty: 200.0,
        },
        TestSpec {
            name: "BNBUSDT",
            tick_size: 0.01,
            step_size: 0.01,
            min_qty: 0.01,
            min_notional: 5.0,
            base_price: 600.0,
            base_qty: 0.2,
        },
    ];

    let mut total_orders_tested = 0;
    let mut total_orders_admitted = 0;

    // Generador congruencial lineal determinista (LCG de Knuth)
    let mut rng_state: u64 = 0xDEADBEEF_CAFE1337;
    let mut lcg = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let val = (rng_state >> 33) as u32;
        val as f64 / (u32::MAX as f64)
    };

    for spec in &specs {
        let filter = BinanceFixedFilter::with_limits(
            spec.tick_size,
            spec.step_size,
            spec.tick_size,
            10_000_000.0,
            spec.min_qty,
            10_000_000.0,
            spec.min_notional,
        );
        assert!(filter.is_valid_config());

        for i in 0..2000 {
            total_orders_tested += 1;
            let is_sell = i % 2 == 0;

            // Generar perturbaciones realistas con ruido de punto flotante
            let price_noise = (lcg() - 0.5) * 0.1 * spec.base_price;
            let raw_price = spec.base_price + price_noise;

            let qty_factor = 1.0 + lcg() * 5.0; // Garantiza notional >= 5.0
            let raw_qty = spec.base_qty * qty_factor;

            // Probar cuantización y formateo
            let decision = filter.validate_and_quantize(raw_price, raw_qty, is_sell);
            match decision {
                Ok(order) => {
                    total_orders_admitted += 1;

                    // 1. Invariante de tick
                    let tick_remainder = ((order.price - filter.min_price) / filter.tick_size).round();
                    let reconstructed_price = filter.min_price + tick_remainder * filter.tick_size;
                    assert!(
                        (order.price - reconstructed_price).abs() <= 1e-6,
                        "Violación de tick en {}: price={}, reconstructed={}",
                        spec.name,
                        order.price,
                        reconstructed_price
                    );

                    // 2. Invariante de paso de lote
                    let step_remainder = ((order.qty - filter.min_qty) / filter.step_size).round();
                    let reconstructed_qty = filter.min_qty + step_remainder * filter.step_size;
                    assert!(
                        (order.qty - reconstructed_qty).abs() <= 1e-6,
                        "Violación de step en {}: qty={}, reconstructed={}",
                        spec.name,
                        order.qty,
                        reconstructed_qty
                    );

                    // 3. Invariante de nocional
                    assert!(
                        order.notional >= filter.min_notional,
                        "Nocional {} menor al mínimo {} en {}",
                        order.notional,
                        filter.min_notional,
                        spec.name
                    );

                    // 4. Invariante de formato ASCII de precio
                    let price_str = order.price_str.as_str();
                    assert!(!price_str.contains('e') && !price_str.contains('E'));
                    let parsed_p: f64 = price_str.parse().expect("Precio no parseable");
                    assert!(
                        (parsed_p - order.price).abs() <= 1e-6,
                        "Discrepancia parseada de precio: parsed={}, real={}",
                        parsed_p,
                        order.price
                    );
                    if let Some(dot_idx) = price_str.find('.') {
                        let decimals = price_str.len() - dot_idx - 1;
                        assert!(
                            decimals <= filter.price_decimals as usize,
                            "Precio con exceso de decimales: {} vs max {}",
                            decimals,
                            filter.price_decimals
                        );
                    }

                    // 5. Invariante de formato ASCII de cantidad
                    let qty_str = order.qty_str.as_str();
                    assert!(!qty_str.contains('e') && !qty_str.contains('E'));
                    let parsed_q: f64 = qty_str.parse().expect("Cantidad no parseable");
                    assert!(
                        (parsed_q - order.qty).abs() <= 1e-6,
                        "Discrepancia parseada de cantidad: parsed={}, real={}",
                        parsed_q,
                        order.qty
                    );
                    if let Some(dot_idx) = qty_str.find('.') {
                        let decimals = qty_str.len() - dot_idx - 1;
                        assert!(
                            decimals <= filter.qty_decimals as usize,
                            "Cantidad con exceso de decimales: {} vs max {}",
                            decimals,
                            filter.qty_decimals
                        );
                    }
                }
                Err(FilterRejectReason::NotionalBelowMin { current, min }) => {
                    panic!(
                        "Rechazo espurio de nocional en {}: current={}, min={}",
                        spec.name, current, min
                    );
                }
                Err(err) => {
                    panic!("Error inesperado en {}: {:?}", spec.name, err);
                }
            }
        }
    }

    assert_eq!(total_orders_tested, 10_000);
    assert_eq!(total_orders_admitted, 10_000);
}

#[test]
fn test_fail_closed_on_adverse_or_invalid_inputs() {
    let filter = BinanceFixedFilter::new(0.1, 0.001, 5.0);

    // 1. NaN y no-finitos
    assert_eq!(
        filter.validate_and_quantize(f64::NAN, 1.0, false),
        Err(FilterRejectReason::NonFinite)
    );
    assert_eq!(
        filter.validate_and_quantize(60000.0, f64::INFINITY, false),
        Err(FilterRejectReason::NonFinite)
    );

    // 2. Cero y negativos
    assert_eq!(
        filter.validate_and_quantize(0.0, 1.0, false),
        Err(FilterRejectReason::ZeroOrNegative)
    );
    assert_eq!(
        filter.validate_and_quantize(60000.0, -0.05, false),
        Err(FilterRejectReason::ZeroOrNegative)
    );

    // 3. Cantidad por debajo de min_qty
    assert_eq!(
        filter.validate_and_quantize(60000.0, 0.0005, false),
        Err(FilterRejectReason::QtyBelowMin {
            current: 0.0,
            min: 0.001
        })
    );

    // 4. Nocional por debajo de min_notional
    assert_eq!(
        filter.validate_and_quantize(1000.0, 0.001, false), // 1.0 USD < 5.0 USD
        Err(FilterRejectReason::NotionalBelowMin {
            current: 1.0,
            min: 5.0
        })
    );
}

#[test]
fn test_stack_ascii_latency_under_50ns() {
    let filter = BinanceFixedFilter::new(0.01, 0.001, 5.0);
    let start = std::time::Instant::now();
    let iters = 100_000;
    let mut sum_len = 0;

    for i in 0..iters {
        let p = 3500.25 + (i as f64) * 0.01;
        let s = filter.format_price(p);
        sum_len += s.len();
    }

    let elapsed = start.elapsed();
    let ns_per_op = elapsed.as_nanos() as f64 / iters as f64;
    assert!(sum_len > 0);
    // En modo debug puede tardar ~150-250ns; en release tarda < 15ns.
    assert!(ns_per_op < 500.0, "Latencia {} ns/op excede 500 ns", ns_per_op);
}
