use god_engine_core::reality_physics::RealityPhysics;

/// CERRADO por D-753 (fusión PR #5): la física unificada
/// (`tp_sl::latency_slippage_pct`) sanea latencia negativa/NaN a 0 ANTES de
/// la raíz, de modo que sqrt(-1) ya no existe y el piso .max() ya no puede
/// ocultar una latencia inválita. Contrato de regresión: inválida == válida.
#[test]
fn negative_latency_is_sanitized_to_zero_slippage() {
    let p = RealityPhysics::default();
    let valid = p
        .calculate_market_entry(100.0, true, 1e6, 0.001, 0.0001, 0.0)
        .0;
    let invalid = p
        .calculate_market_entry(100.0, true, 1e6, 0.001, 0.0001, -1.0)
        .0;
    assert_eq!(invalid, valid);
    let nan = p
        .calculate_market_entry(100.0, true, 1e6, 0.001, 0.0001, f64::NAN)
        .0;
    assert_eq!(nan, valid);
}

/// TRIAJE B (GLM 108) — DRENADO: contrato de SALIDA fail-closed. Un precio
/// base finito pero extremo (f64::MAX) desbordaba la multiplicación y
/// producía un fill a `inf` que escapaba del guard del caller (sólo cubría
/// <= 0.0, con qty = nominal/inf = 0). La heurística promete fill válido o
/// (0,0): salida no finita = no fill. Paridad entrada/salida.
#[test]
fn finite_extreme_base_price_fails_closed_instead_of_infinite_fill() {
    let p = RealityPhysics::default();
    assert_eq!(
        p.calculate_market_entry(f64::MAX, true, 1e6, 0.001, 0.0001, 15.0),
        (0.0, 0.0),
        "entrada long: precio extremo desborda hacia arriba ⇒ no fill"
    );
    assert_eq!(
        p.calculate_exit(f64::MAX, false, 1e6, false, 0.001, 0.0001, 15.0),
        (0.0, 0.0),
        "salida short: multiplica hacia arriba — mismo contrato que la entrada"
    );
    // El caso REAL del wire sigue intacto: precio normal ⇒ fill válido.
    let (precio, fee) = p.calculate_market_entry(60000.0, true, 100.0, 0.002, 0.0005, 15.0);
    assert!(precio.is_finite() && precio > 60000.0 && fee > 0.0);
}

/// TRIAJE B (GLM 108) — CONTRATO DELIBERADO (muerta-por-contrato CL-14):
/// `calculate_maker_entry` es superficie de diagnóstico que el guard CL-14
/// (lib.rs) PROHÍBE llamar desde el core — el host siempre envía MARKET y
/// simular entradas pasivas ahorraría al backtest lo que el vivo paga. Sin
/// datos de cola no existe probabilidad de fill que modelar: la estimación
/// incondicional (precio base + fee maker) ES el contrato.
#[test]
fn maker_estimate_is_a_dead_diagnostic_surface_by_contract() {
    let p = RealityPhysics::default();
    assert_eq!(p.calculate_maker_entry(100.0, true, 1000.0), (100.0, 0.2));
    assert_eq!(p.calculate_maker_entry(100.0, false, 1000.0), (100.0, 0.2));
}

/// TRIAJE B (GLM 108) — DRENADO (higiene D-747): el campo struct
/// `latency_penalty_ms` fue REMOVIDO — jamás se leía y honrarlo crearía
/// una SEGUNDA fuente de latencia (violación de la fuente única D-747:
/// la latencia vive en arena.config y entra por el ARGUMENTO explícito,
/// muestreada lognormal). El contrato ahora es de COMPILACIÓN: sin campo,
/// dos instancias no pueden divergir.
#[test]
fn latency_enters_only_by_argument_d747_single_source() {
    let a = RealityPhysics::default();
    let b = RealityPhysics::default();
    // Distinto argumento ⇒ distinta física (el argumento SÍ manda).
    let corto = a.calculate_market_entry(100.0, true, 1e6, 0.001, 0.0001, 1.0);
    let largo = b.calculate_market_entry(100.0, true, 1e6, 0.001, 0.0001, 1000.0);
    assert_ne!(corto, largo, "la latencia del argumento sí cambia la física");
}
