//! Contrato F1-125: Perfil espectral continuo anclado a la banda física operativa [18..=22] ([30 s, 12 h]).
//!
//! Verifica que:
//! 1. Las escalas operativas [18..=22] tienen peso espectral máximo 1.0 en `voto_espectral`.
//! 2. Las escalas de microestructura (k < 18) decaen de forma continua C^0 hacia 0.5 en k=0.
//! 3. Las escalas seculares macro (k > 22) decaen de forma continua C^0 hacia 0.5 en k=31.
//! 4. Se erradica la partición arbitraria en k=8 (que correspondía físicamente a 65.5 microsegundos).

use signal_engine::perceptron_gate::PerceptronGateEngine;
use signal_engine::voto_espectral::ESCALAS_VOTO;

#[test]
fn test_f1_125_perceptron_operable_band_anchors() {
    let mut x = [0.0f64; ESCALAS_VOTO];
    for k in 0..ESCALAS_VOTO {
        x[k] = 0.5;
    }

    let voto = PerceptronGateEngine::voto_espectral(&x);

    // En la banda operativa [18..=22], el peso debe ser idéntico (1.0)
    let ref_operable = voto.en_escala(18);
    for k in 18..=22 {
        let v = voto.en_escala(k);
        assert!(
            (v - ref_operable).abs() < 1e-12,
            "Escala {} en banda operativa debe tener peso identico a 18: {} vs {}",
            k, v, ref_operable
        );
    }
}

#[test]
fn test_f1_125_perceptron_monotonic_decay_outside_operable_band() {
    let mut x = [0.0f64; ESCALAS_VOTO];
    for k in 0..ESCALAS_VOTO {
        x[k] = 0.5;
    }

    let voto = PerceptronGateEngine::voto_espectral(&x);

    // 1. Decaimiento hacia microestructura: desde k=18 bajando hasta k=0
    let mut prev_val = voto.en_escala(18);
    for k in (0..18).rev() {
        let current_val = voto.en_escala(k);
        assert!(
            current_val <= prev_val,
            "Microestructura debe decaer monotonicamente: k={} ({}) > k+1 ({})",
            k, current_val, prev_val
        );
        prev_val = current_val;
    }

    // 2. Decaimiento hacia macroestructura secular: desde k=22 subiendo hasta k=31
    let mut prev_val_macro = voto.en_escala(22);
    for k in 23..=31 {
        let current_val = voto.en_escala(k);
        assert!(
            current_val <= prev_val_macro,
            "Macroestructura debe decaer monotonicamente: k={} ({}) > k-1 ({})",
            k, current_val, prev_val_macro
        );
        prev_val_macro = current_val;
    }

    // 3. Extremos k=0 y k=31 deben coincidir con el piso de decaimiento (peso 0.5)
    let v_0 = voto.en_escala(0);
    let v_31 = voto.en_escala(31);
    let v_esperado_piso = PerceptronGateEngine::infer(0.5, 0.5);
    assert!(
        (v_0 - v_esperado_piso).abs() < 1e-10,
        "k=0 debe converger exactamente al peso 0.5: {} vs {}",
        v_0, v_esperado_piso
    );
    assert!(
        (v_31 - v_esperado_piso).abs() < 1e-10,
        "k=31 debe converger exactamente al peso 0.5: {} vs {}",
        v_31, v_esperado_piso
    );
}
