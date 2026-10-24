//! RET-13a (CL-157) — FIX #1454 en el OmniStrategyEngine VIVO.
//!
//! La guarda sólo existía en la copia muerta de la raíz
//! (`src/features/omni_strategies.rs:52-54` y `:174-179`, retirada en
//! RET-13b). En main 781e5bd3 el motor vivo aceptaba cualquier entrada: un
//! precio NaN dejaba `last_price = NaN` y el z de Bollinger (rasgo 3), entre
//! otros, salía NaN; un precio ≤ 0 o un high/low no finito movía el estado.
//! Y con precios finitos, un salto de escala (1e300 → 1) sacaba ±inf al
//! convertir a f32.
//!
//! En vivo es inerte: `StatefulEngine::try_process_tick` y
//! `try_process_kline` rechazan esas entradas antes de llamar a `update`, y
//! con precios reales ningún cociente desborda f32.

use feature_engine::OmniStrategyEngine;

fn bits(f: [f32; 22]) -> [u32; 22] {
    f.map(f32::to_bits)
}

fn calentado() -> OmniStrategyEngine {
    let mut e = OmniStrategyEngine::new();
    for i in 0..40u32 {
        let p = 100.0 + f64::from(i % 7) * 0.25 - f64::from(i % 3) * 0.1;
        e.update(p, p + 0.2, p - 0.2);
    }
    e
}

#[test]
fn entrada_no_finita_o_precio_no_positivo_no_toca_el_estado() {
    let base = calentado();
    let antes = base.extract_features();
    assert!(
        antes.iter().all(|f| f.is_finite()),
        "calentamiento no finito: {antes:?}"
    );
    let casos = [
        (f64::NAN, 100.5, 99.5),
        (f64::INFINITY, 100.5, 99.5),
        (f64::NEG_INFINITY, 100.5, 99.5),
        (-1.0, 100.5, 99.5),
        (0.0, 100.5, 99.5),
        (100.3, f64::NAN, 99.5),
        (100.3, 100.5, f64::INFINITY),
    ];
    for (precio, alto, bajo) in casos {
        let mut e = base.clone();
        e.update(precio, alto, bajo);
        let despues = e.extract_features();
        assert!(
            despues.iter().all(|f| f.is_finite()),
            "update({precio}, {alto}, {bajo}) dejó rasgos no finitos: {despues:?}"
        );
        assert_eq!(
            bits(despues),
            bits(antes),
            "update({precio}, {alto}, {bajo}) cambió el estado"
        );
    }
}

#[test]
fn rasgos_finitos_aunque_un_cociente_desborde_f32() {
    // Precios finitos y positivos: tras el salto, las EWMA del MACD guardan
    // ≈ 1e300 y (macd_line / precio) no cabe en f32. En main el rasgo 1 sale
    // −inf (y el 8, el 9 y el 10, ±inf).
    let mut e = OmniStrategyEngine::new();
    e.update(1e300, 1e300, 1e300);
    e.update(1.0, 1.0, 1.0);
    let f = e.extract_features();
    assert!(f.iter().all(|x| x.is_finite()), "rasgos no finitos: {f:?}");
}
