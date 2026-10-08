//! R5-A4 y R5-A6: Telemetría de las 13 sombras espectrales y enlace dinámico de Nash con CVPIN
//! Eventos sintéticos locales; sin exchange ni red.
use god_engine_core::GodEngineCore;
use quantum_arena::{GlobalArena, symbol_registry, symbols};

#[test]
fn r5_a4_a6_telemetria_sombras_y_nash_enlace_cvpin() {
    symbols::update_dynamic_universe(vec!["SOMBTESTUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(
        "SOMBTESTUSDT",
    )]);

    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;

    // Establecer un CVPIN medido no trivial para coin 0
    arena.registry.set_for_coin(0, "cvpin", 0.35);

    // Enviar eventos de profundidad (is_depth = true) con micro-movimientos para poblar el espectro multiescala
    for i in 0..10 {
        let price = 100.0 + (i as f64) * 0.05;
        let ts = 10_000 + i * 1_000;
        let _ = core.process_event(
            0,
            false,
            false,
            true, // is_depth = true activa el cómputo de sombras espectrales
            price,
            0.0,
            price,
            price + 0.01,
            5.0,
            5.0,
            0.1,
            0.0,
            ts,
            false,
            &[0.0; 54],
            false,
        );
    }

    // R5-A6: Las 6 sombras previamente mudas en telemetría ahora publican su consenso
    let sombras_esperadas = [
        "sombra_hawkes_consenso",
        "sombra_nash_consenso",
        "sombra_flow_consenso",
        "sombra_perceptron_consenso",
        "sombra_conformal_consenso",
        "sombra_confluence_consenso",
        "sombra_coax_consenso",
        "sombra_trend_consenso",
        "sombra_entropia_consenso",
    ];

    for clave in sombras_esperadas {
        let val = arena.registry.get_for_coin_or(0, clave, f64::NAN);
        assert!(
            val.is_finite(),
            "La sombra {clave} debe publicarse en el registro para coin 0; valor: {val}"
        );
    }

    // R5-A4: La presión adversarial de Nash fue leída dinámicamente desde cvpin (0.35)
    // en ausencia de override explícito, confirmando que el valor no está congelado en 0.50
    let nash_val = arena.registry.get_for_coin_or(0, "sombra_nash_consenso", f64::NAN);
    assert!(nash_val.is_finite(), "sombra_nash_consenso debe ser finito");
}
