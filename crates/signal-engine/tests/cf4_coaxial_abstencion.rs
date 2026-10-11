//! CL-66 (C-F4, LC-F3-02): el coaxial se abstiene si le falta cualquiera de
//! sus tres escalas de ATR, o si no es un valor positivo y finito.
//!
//! Antes, sin `atr_1s` leía `atr_pct` (una fracción, no un ATR en precio),
//! sin `atr_5s` o `atr_1m` fabricaba ×√5 o ×√60, y los no finitos caían a
//! literales en USD (0,001, 0,005 y 0,020). Medido con el código anterior
//! (CL-66, rojo): sin `atr_1s` y con `atr_5s` = 0,005, `atr_1m` = 0,05 y
//! `atr_pct` = 0,001, el respaldo daba un voto de 0,681; con `atr_1s` = NaN,
//! el literal 0,001 daba el mismo 0,681.
//!
//! Verdad: voto 0 exacto en las 7 combinaciones con alguna escala ausente y
//! con valores no finitos o no positivos; voto > 0 con las tres presentes y
//! compresión real (control).

use omniscient_registry::OmniscientRegistry;
use signal_engine::coaxial_breakout::CoaxialBreakoutEngine;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

const CLAVES: [&str; 3] = ["atr_1s", "atr_5s", "atr_1m"];
/// Compresión real 1 s ≪ 5 s ≪ 1 min, la del test unitario del módulo.
const VALORES: [f64; 3] = [0.0005, 0.005, 0.050];

fn voto(presentes: [Option<f64>; 3]) -> f64 {
    let registry = Arc::new(OmniscientRegistry::new());
    for (clave, v) in CLAVES.iter().zip(presentes) {
        if let Some(v) = v {
            registry.set(clave, v);
        }
    }
    registry.set("atr_pct", 0.001);
    registry.set("order_flow_direction", 1.0);
    let mut e = CoaxialBreakoutEngine::new();
    e.init(registry).expect("init");
    e.evaluate()
}

#[test]
fn cl66_control_tres_escalas_con_compresion_votan() {
    let v = voto([Some(VALORES[0]), Some(VALORES[1]), Some(VALORES[2])]);
    assert!(v > 0.0 && v.is_finite(), "control: voto {v}");
}

#[test]
fn cl66_cualquier_escala_ausente_abstiene() {
    for mascara in 0u8..7 {
        let presentes = [0, 1, 2].map(|k| (mascara >> k & 1 == 1).then_some(VALORES[k]));
        let v = voto(presentes);
        assert_eq!(v, 0.0, "presentes {presentes:?}: voto {v}");
    }
}

#[test]
fn cl66_escala_no_finita_o_no_positiva_abstiene() {
    for malo in [f64::NAN, f64::INFINITY, 0.0, -0.01] {
        for k in 0..3 {
            let mut presentes = VALORES.map(Some);
            presentes[k] = Some(malo);
            let v = voto(presentes);
            assert_eq!(v, 0.0, "{} = {malo}: voto {v}", CLAVES[k]);
        }
    }
}
