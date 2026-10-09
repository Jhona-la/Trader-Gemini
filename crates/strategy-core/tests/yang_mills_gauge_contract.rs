//! CONTRATO FORMAL DE TEORÍA DE GAUGE YANG-MILLS & CURVATURA DE ARBITRAJE (OLA Ω36)
//!
//! Verifica matemáticamente los axiomas invariantes de la teoría de campos gauge
//! aplicada a redes multiactivo continuas en Trader Gemini.

use strategy_core::yang_mills_gauge::YangMillsGaugeEngine;

#[test]
fn yang_mills_contrato_triada_libre_de_arbitraje() {
    // Si los precios están en perfecto equilibrio triangular: P_A = 100, P_B = 50, P_C = 25 (ratios 2, 2, 0.25)
    // Con ln(P_A) - ln(P_B) = ln(2), ln(P_B) - ln(P_C) = ln(2), ln(P_C) - ln(P_A) = -ln(4) = -2 ln(2)
    // El bucle de Wilson F_{ABC} debe ser idénticamente cero (curvatura plana).
    let mut engine = YangMillsGaugeEngine::new(3);
    let prices = [100.0, 50.0, 25.0];

    // Con betas iniciales 1.0 (arbitraje logarítmico triangular directo)
    // F_{012} = ln(100/50) + ln(50/25) + ln(25/100) = ln(2) + ln(2) - ln(4) = 0.0
    let (action, currents) = engine.update_and_calculate_curvature(&prices);

    assert!(
        action.abs() < 1e-12,
        "La densidad de acción de Yang-Mills debe ser cero en equilibrio libre de arbitraje: got {}",
        action
    );
    for (idx, &c) in currents[..3].iter().enumerate() {
        assert!(
            c.abs() < 1e-12,
            "La corriente de gauge en el activo {} debe ser 0 en equilibrio: got {}",
            idx,
            c
        );
    }
}

#[test]
fn yang_mills_contrato_antisimetria_y_permutaciones() {
    let mut engine = YangMillsGaugeEngine::new(4);
    let prices = [105.0, 50.0, 25.0, 10.0];
    let _ = engine.update_and_calculate_curvature(&prices);

    // Permutaciones cíclicas conservan la holonomía: F_{012} == F_{120} == F_{201}
    let f_012 = engine.loop_holonomy(0, 1, 2);
    let f_120 = engine.loop_holonomy(1, 2, 0);
    let f_201 = engine.loop_holonomy(2, 0, 1);

    assert!(
        (f_012 - f_120).abs() < 1e-12,
        "Permutación cíclica rota: F_012 ({f_012}) != F_120 ({f_120})"
    );
    assert!(
        (f_012 - f_201).abs() < 1e-12,
        "Permutación cíclica rota: F_012 ({f_012}) != F_201 ({f_201})"
    );

    // Inversión de orientación invierte el signo: F_{021} == -F_{012}
    let f_021 = engine.loop_holonomy(0, 2, 1);
    assert!(
        (f_012 + f_021).abs() < 1e-12,
        "Antisimetría de orientación de Wilson rota: F_012 ({f_012}) != -F_021 ({f_021})"
    );
}

#[test]
fn yang_mills_contrato_corriente_restauradora_tras_dislocacion() {
    // Red con elasticidades reales (ej. BTC=0, ETH=1, SOL=2 con betas 1.25, 0.90, 1.0)
    let mut engine = YangMillsGaugeEngine::new(3)
        .with_beta(0, 1, 1.25)
        .with_beta(1, 2, 0.90)
        .with_beta(2, 0, 1.0);

    // Dislocar Activo 0 hacia arriba (110.0 en lugar del equilibrio 100.0)
    let prices_dislocated = [110.0, 50.0, 25.0];
    let (action, currents) = engine.update_and_calculate_curvature(&prices_dislocated);

    // La acción de Yang-Mills debe ser estrictamente positiva (tensión de curvatura)
    assert!(
        action > 1e-4,
        "Una dislocación debe generar acción de Yang-Mills estrictamente positiva: got {}",
        action
    );

    // La corriente del activo 0 debe ser no nula y finita
    assert!(currents[0].is_finite() && currents[0].abs() > 1e-4);
}

#[test]
fn yang_mills_contrato_inmunidad_a_nan_y_precios_no_positivos() {
    let mut engine = YangMillsGaugeEngine::new(3);
    let (action_nan, currents_nan) = engine.update_and_calculate_curvature(&[f64::NAN, 50.0, 25.0]);
    assert_eq!(action_nan, 0.0);
    assert_eq!(currents_nan[0], 0.0);

    let (action_neg, currents_neg) = engine.update_and_calculate_curvature(&[-10.0, 50.0, 25.0]);
    assert_eq!(action_neg, 0.0);
    assert_eq!(currents_neg[0], 0.0);

    let (action_zero, _) = engine.update_and_calculate_curvature(&[0.0, 50.0, 25.0]);
    assert_eq!(action_zero, 0.0);
}

#[test]
fn yang_mills_contrato_retornos_dinamicos_y_estabilidad_lms() {
    let mut engine = YangMillsGaugeEngine::new(3)
        .with_beta(0, 1, 1.25)
        .with_beta(1, 2, 0.90)
        .with_beta(2, 0, 1.0);

    // Tick 1: Inicialización con precios base (count = 0 -> 1)
    let (action_1, _) = engine.update_and_calculate_curvature(&[100.0, 50.0, 25.0]);
    assert_eq!(engine.count, 1);
    assert!(action_1 >= 0.0);

    // Tick 2: Retornos de innovación. Activo 0 sube +2%, Activo 1 sube +1%, Activo 2 no se mueve (0%)
    // Con r_2 = 0, el par (0, 2) no debe sufrir deriva espuria en beta_matrix[0][2]
    let beta_02_inicial = engine.beta_matrix[0][2];
    let (action_2, currents_2) = engine.update_and_calculate_curvature(&[102.0, 50.5, 25.0]);
    assert_eq!(engine.count, 2);

    // La curvatura no debe colapsar a cero trivialmente: se evalúa sobre retornos dinámicos
    assert!(
        action_2.is_finite() && action_2 > 0.0,
        "La acción de Yang-Mills con innovaciones asimétricas debe ser estrictamente > 0, got {}",
        action_2
    );
    assert!(
        currents_2[0].is_finite() && currents_2[0].abs() > 1e-6,
        "La corriente gauge del activo 0 debe ser no-nula tras innovación"
    );

    // R6-C6: Dado que el Activo 2 tuvo retorno r_2 = 0, el paso de LMS sobre beta_02 es exactamente 0
    assert_eq!(
        engine.beta_matrix[0][2], beta_02_inicial,
        "El coeficiente beta hacia un activo que no cotizó (r=0) no debe mutar por gradiente espurio"
    );

    // Tick 3: Choque opuesto en Activo 1 mientras 0 y 2 se mantienen constantes
    let (action_3, currents_3) = engine.update_and_calculate_curvature(&[102.0, 49.0, 25.0]);
    assert!(action_3.is_finite() && action_3 > 0.0);
    assert!(currents_3[1].is_finite() && currents_3[1].abs() > 1e-6);
}
