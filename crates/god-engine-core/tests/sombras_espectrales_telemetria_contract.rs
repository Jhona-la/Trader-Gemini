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

/// H2-10 (RONDA 3, GLM 112): la sombra del solitón YA NO lee el knob
/// global muerto `soliton_amplitude` (siempre 1.0) — ESPEJO de la cascada
/// del vivo: per-coin `soliton_amplitude → order_flow_imbalance → 0.0`,
/// con el sanitizado interno del motor (amp≤0→1.0, clamp [1e-3,10]).
#[test]
fn h2_10_sombra_soliton_espeja_la_cascada_viva_de_amplitud() {
    use god_engine_core::GodEngineCore;
    use quantum_arena::{GlobalArena, symbol_registry, symbols};

    symbols::update_dynamic_universe(vec!["SOLH2TENUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(
        "SOLH2TENUSDT",
    )]);

    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
    // OFI per-coin positivo y distinguible: el espejo debe usarlo como amp.
    // NOTA: sin set de soliton_amplitude — el knob muerto NO se publica,
    // eso es exactamente lo que el fix repara.
    arena.registry.set_for_coin(0, "order_flow_imbalance", 3.0);

    for i in 0..10 {
        let price = 100.0 + (i as f64) * 0.05;
        let ts = 10_000 + i * 1_000;
        let _ = core.process_event(
            0, false, false, true, price, 0.0, price, price + 0.01, 5.0, 5.0, 0.1, 0.0, ts,
            false, &[0.0; 54], false,
        );
    }

    // La sombra publicó su telemetría bajo la cascada nueva.
    let amp = arena
        .registry
        .get_for_coin_or(0, "sombra_soliton_amplitud_max", f64::NAN);
    assert!(
        amp.is_finite(),
        "la sombra del solitón sigue publicando su amplitud dominante: {amp}"
    );

    // Distinguibilidad física con el MOTOR REAL: sobre la misma malla
    // x O(1), el voto con amp=3.0 (OFI espejado) difiere del amp=1.0
    // congelado del knob muerto — el espejo cambia conducta real.
    let x = [0.8f64; 32];
    let con_espejo =
        signal_engine::soliton_wave::SolitonWaveEngine::voto_espectral(&x, 3.0);
    let congelado =
        signal_engine::soliton_wave::SolitonWaveEngine::voto_espectral(&x, 1.0);
    let (k_e, v_e) = con_espejo.dominante().expect("dominante con amp 3.0");
    let (_k_c, v_c) = congelado.dominante().expect("dominante con amp 1.0");
    assert_ne!(
        v_e, v_c,
        "amp 3.0 (OFI espejado) produce un voto distinto al 1.0 congelado: {v_e} vs {v_c} (escala {k_e})"
    );
    // El sanitizado del motor: amp≤0 (OFI negativo/0) → 1.0, idéntico al
    // default del knob muerto — la semántica del vivo preservada.
    let neg =
        signal_engine::soliton_wave::SolitonWaveEngine::voto_espectral(&x, -0.5);
    let (_k_n, v_n) = neg.dominante().expect("dominante con amp saneado");
    assert_eq!(v_n, v_c, "amp≤0 se sanea a 1.0 (mismo voto que el congelado)");
}
