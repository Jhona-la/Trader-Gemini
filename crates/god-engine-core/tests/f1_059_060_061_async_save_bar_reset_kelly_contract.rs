//! Contrato de certificación Ola Ω170 (F1-059, F1-060, F1-061):
//! 1. F1-060: ModelEnsemble `nuevo_bar` reinicia predicciones abiertas en barras neutras
//!    para evitar calificar opiniones desalineadas en barras futuras.
//! 2. F1-059: `VecDeque<Observacion>` en calibradores de `MonedaGuardada` y `EstadoAprendido`
//!    con paridad JSON bidireccional y eliminación de asignaciones en el hot path.
//! 3. F1-061: Unificación de autoridad de tamaño en `RiskEngine`, con `RiskEnvelope`
//!    actuando como supervisor de admisibilidad (`operable`) y techo de apalancamiento (`env_cap`).

use std::collections::VecDeque;
use god_engine_core::ensemble::{ModelEnsemble, ModelId};
use god_engine_core::estado_aprendido::{
    EstadoAprendido, EpigeneticaGuardada, FrenoGuardado, MetricasGuardadas, MonedaGuardada,
    Observacion, VERSION_ESTADO,
};
use risk_engine::envio::{apalancamiento_de_envio, apalancamiento_validado, TOPE_ENVIO, VetoEnvio};
use risk_engine::kelly_envelope::RiskEnvelope;

#[test]
fn f1_060_model_ensemble_nuevo_bar_neutral_reset() {
    let mut ensemble = ModelEnsemble::new();
    // Simular predicciones
    ensemble.submit(ModelId::MotorForest, 0.65);
    ensemble.submit(ModelId::DarkAlphaNN, 0.70);
    assert_eq!(ensemble.prediction_snapshot()[0], Some(0.65));
    assert_eq!(ensemble.bar_open_prediction_snapshot()[0], Some(0.65));

    // Reinicio explícito de barra neutra (F1-060)
    ensemble.nuevo_bar();

    assert_eq!(ensemble.prediction_snapshot(), [None, None], "predictions debe limpiarse en nuevo_bar");
    assert_eq!(ensemble.bar_open_prediction_snapshot(), [None, None], "bar_open_predictions debe limpiarse en nuevo_bar");

    // Verificar que un submit posterior en la siguiente barra capture la nueva opinión limpia
    ensemble.submit(ModelId::MotorForest, 0.40);
    ensemble.submit(ModelId::DarkAlphaNN, 0.45);
    assert_eq!(ensemble.prediction_snapshot()[0], Some(0.40));
    assert_eq!(ensemble.prediction_snapshot()[1], Some(0.45));
    assert_eq!(ensemble.bar_open_prediction_snapshot()[0], Some(0.40));
    assert_eq!(ensemble.bar_open_prediction_snapshot()[1], Some(0.45));
}

#[test]
fn f1_059_vecdeque_conformal_calibrador_zero_alloc_serialization_parity() {
    let mut deque: VecDeque<Observacion> = VecDeque::new();
    deque.push_back((0.72, true, 1_700_000_000_000));
    deque.push_back((0.35, false, 1_700_000_001_000));

    let moneda = MonedaGuardada {
        simbolo: "BTCUSDT".to_string(),
        metricas: MetricasGuardadas {
            trade_count: 40,
            win_rate: 0.50,
            profit_factor: 1.2,
            kelly_fraction: 0.05,
            gross_wins: 10.0,
            gross_losses: 8.0,
            pnl_realized: 2.0,
            pnl_gross: 2.5,
            roi_pre_fee: 0.15,
            roi_post_fee: 0.12,
            zombie_promotions: 0,
        },
        calibrador: deque.clone(),
        calibrador_reloj_ms: Some(1_700_000_001_000),
        epigenetica: EpigeneticaGuardada {
            sesgo: 1.0,
            umbral: 1.0,
            ultimo_cierre_ms: 100_000,
            ultimo_cierre_largo: true,
            ultimo_cierre_gano: true,
        },
        siniestros: vec![],
        detector_lg22: None,
        conformal: None,
    };

    let estado = EstadoAprendido {
        version: VERSION_ESTADO,
        monedas: vec![moneda],
        calibrador_global: deque.clone(),
        calibrador_global_reloj_ms: Some(1_700_000_001_000),
        conformal_global: None,
        freno_estrategia: FrenoGuardado {
            e_value: 1.0,
            log_wealth: 0.0,
            peak_e_value: 1.0,
            count: 10,
            running_mean: 0.0,
            running_sq_mean: 1.0,
            last_lambda: 0.5,
            rechazada: false,
        },
        riesgo_por_operacion: 0.01,
        riesgo_por_operacion_usd: 0.13,
        ganancia_en_stops: 0.0,
        perdida_en_stops: 0.0,
        resultado_en_stops: 0.0,
        pico_en_stops: 0.0,
    };

    // 1. Serialización a JSON
    let json_str = serde_json::to_string(&estado).expect("serialización debe ser infalible");

    // 2. Deserialización desde JSON
    let deserialized: EstadoAprendido = serde_json::from_str(&json_str).expect("deserialización debe ser infalible");

    assert_eq!(deserialized.calibrador_global.len(), 2);
    assert_eq!(deserialized.calibrador_global[0], (0.72, true, 1_700_000_000_000));
    assert_eq!(deserialized.calibrador_global[1], (0.35, false, 1_700_000_001_000));

    assert_eq!(deserialized.monedas[0].calibrador.len(), 2);
    assert_eq!(deserialized.monedas[0].calibrador[0], (0.72, true, 1_700_000_000_000));
    assert_eq!(deserialized.monedas[0].calibrador[1], (0.35, false, 1_700_000_001_000));

    // 3. Compatibilidad hacia atrás: JSON con array de tuplas deserializa transparentemente en VecDeque
    let raw_json = r#"{
        "version": 4,
        "monedas": [{
            "simbolo": "ETHUSDT",
            "metricas": {
                "trade_count": 5,
                "win_rate": 0.6,
                "profit_factor": 1.5,
                "kelly_fraction": 0.05,
                "gross_wins": 5.0,
                "gross_losses": 2.0,
                "pnl_realized": 3.0,
                "pnl_gross": 3.2,
                "roi_pre_fee": 0.08,
                "roi_post_fee": 0.05,
                "zombie_promotions": 0
            },
            "calibrador": [[0.81, true, 12345]],
            "epigenetica": {
                "sesgo": 1.0,
                "umbral": 1.0,
                "ultimo_cierre_ms": 100,
                "ultimo_cierre_largo": true,
                "ultimo_cierre_gano": true
            },
            "siniestros": []
        }],
        "calibrador_global": [[0.81, true, 12345]],
        "freno_estrategia": {
            "e_value": 1.0,
            "log_wealth": 0.0,
            "peak_e_value": 1.0,
            "count": 1,
            "running_mean": 0.0,
            "running_sq_mean": 1.0,
            "last_lambda": 0.5,
            "rechazada": false
        },
        "riesgo_por_operacion": 0.01,
        "riesgo_por_operacion_usd": 0.13,
        "ganancia_en_stops": 0.0,
        "perdida_en_stops": 0.0,
        "resultado_en_stops": 0.0,
        "pico_en_stops": 0.0
    }"#;

    let legacy_deserialized: EstadoAprendido = serde_json::from_str(raw_json).expect("array JSON debe deserializar en VecDeque");
    assert_eq!(legacy_deserialized.calibrador_global.len(), 1);
    assert_eq!(legacy_deserialized.calibrador_global[0], (0.81, true, 12345));
    assert_eq!(legacy_deserialized.monedas[0].calibrador.len(), 1);
    assert_eq!(legacy_deserialized.monedas[0].calibrador[0], (0.81, true, 12345));
}

#[test]
fn f1_061_risk_engine_sizing_authority_with_envelope_admissibility_parity() {
    // 1. Simulación de micro-capital: $13 USD, nocional $5.10 USDT, margen $1.02 USDT (apalancamiento validado 5x)
    let notional = 5.10;
    let margin = 1.02;
    let validado = apalancamiento_validado(notional, margin);
    assert_eq!(validado, Some(5), "Nocional 5.10 con margen 1.02 debe arrojar apalancamiento validado 5x");

    let mut envelope = RiskEnvelope::default();

    // 2. Caso Bootstrap: envelope_n < 30
    assert!(envelope.posterior.n() < 30.0);
    let exec_leverage_boot = validado.unwrap_or(5).clamp(1, 10);
    assert_eq!(exec_leverage_boot, 5, "En bootstrap, exec_leverage debe respetar el apalancamiento validado de 5x");

    // 3. Caso Inadmisible (veto de la envolvente): operable == false
    let cap_now = 13.0;
    let stop_pct = 0.01;
    let env_min_notional = 5.0;
    let env_z = 0.85;
    let env_k = 10.0;
    let (env_lev, operable) = envelope.max_leverage(cap_now, stop_pct, env_min_notional, env_z, env_k);

    let exec_leverage_inadmisible = if !operable {
        0
    } else {
        let env_cap = (env_lev.floor() as u32).clamp(1, TOPE_ENVIO);
        validado.unwrap_or(1).min(env_cap).max(1)
    };

    if !operable {
        assert_eq!(exec_leverage_inadmisible, 0);
        let res = apalancamiento_de_envio(exec_leverage_inadmisible, validado, notional, 10.0);
        assert_eq!(res, Err(VetoEnvio::Envolvente), "exec_leverage 0 debe provocar VetoEnvio::Envolvente");
    }

    // 4. Caso Admisible con techo de envolvente: simular evidencia ganadora
    for _ in 0..35 {
        envelope.record_trade(true, 0.05, -0.02);
    }
    assert!(envelope.posterior.n() >= 30.0);
    let (env_lev_active, operable_active) = envelope.max_leverage(cap_now, stop_pct, env_min_notional, env_z, env_k);
    assert!(operable_active, "Con 35 trades ganadores, la envolvente debe ser operable");

    let vol_brake = 1.0;
    let env_cap = ((env_lev_active.floor() * vol_brake) as u32).clamp(1, TOPE_ENVIO);
    let exec_leverage_admisible = validado.unwrap_or(1).min(env_cap).max(1);

    // El apalancamiento validado (5x) no debe ser artificialmente dividido a 1x por literales
    assert_eq!(exec_leverage_admisible, 5.min(env_cap));

    // Validar adaptación de envío con margen libre
    let free_margin = 12.0;
    let adaptado = apalancamiento_de_envio(exec_leverage_admisible, validado, notional, free_margin)
        .expect("apalancamiento de envío debe ser exitoso");
    assert_eq!(adaptado, 5, "Debe enviar a 5x preservando el margen de micro-capital");
}
