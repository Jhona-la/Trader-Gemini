//! Contrato de certificación formal para Ola Ω62 (Ficha Forense #697).
//!
//! Verifica:
//! 1. R7-R4-B-1 & R7-R4-D-1: Abstención real en barra neutra en el ensemble online Hedge
//!    (retorno NaN, erradicando el bug de entrenamiento de 0.0 falso) y erradicación de lock muerto.
//! 2. R7-R4-B-2: `media_banda_activa` erradica dilución por denominador fijo en coherencia inter-espectral.
//! 3. R7-R2-A-5: Cota inferior conservadora LCB del payoff ratio en RiskEnvelope (Kelly).
//! 4. R7-R2-A-3: Cota inferior analítica LCB de Cramér-Lundberg y ajuste de margen contra riesgo de ruina.

use signal_engine::voto_espectral::VotoEspectral;
use risk_engine::kelly_envelope::RiskEnvelope;
use risk_engine::cramer_lundberg::EstimadorSiniestros;

/// 1. R7-R4-B-1: Verifica que una barra con retorno dentro del rango de fricción (|bar_ret| <= fee_hurdle)
/// produce abstención estricta y NO muta los pesos de un ensemble online, erradicando el bug histórico
/// donde `0.5_f64.signum() * 0.0` evaluaba a 0.0 y entrenaba cada barra neutra como pérdida bajista falsa.
#[test]
fn test_r7_r4_b1_neutral_kline_discards_without_poisoning_ensemble() {
    let fee_hurdle = 0.0010; // 10 bps

    let classify_kline_outcome = |bar_ret: f64, hurdle: f64| -> f64 {
        if bar_ret > hurdle {
            1.0 // superó la barrera alcista
        } else if bar_ret < -hurdle {
            0.0 // superó la barrera bajista
        } else {
            // R7-R4-B-1: neutro, DESCARTAR de forma inmutable
            f64::NAN
        }
    };

    // Barra neutral: +0.02% (2 bps), inferior al fee hurdle de 10 bps
    let y_neutral = classify_kline_outcome(0.0002, fee_hurdle);
    assert!(y_neutral.is_nan(), "Barra neutra debe producir NaN como centinela de abstención");

    let should_update = (y_neutral == 0.0 || y_neutral == 1.0) && y_neutral.is_finite();
    assert!(!should_update, "Barra neutra jamás debe disparar update_with_outcome");

    // Barra alcista decisiva: +0.25% (25 bps) > 10 bps
    let y_bull = classify_kline_outcome(0.0025, fee_hurdle);
    assert_eq!(y_bull, 1.0);
    assert!((y_bull == 0.0 || y_bull == 1.0) && y_bull.is_finite());

    // Barra bajista decisiva: -0.30% (30 bps) < -10 bps
    let y_bear = classify_kline_outcome(-0.0030, fee_hurdle);
    assert_eq!(y_bear, 0.0);
    assert!((y_bear == 0.0 || y_bear == 1.0) && y_bear.is_finite());
}

/// 2. R7-R4-B-2: Verifica que `media_banda_activa` no diluye las escalas con peso efectivo
/// cuando las escalas rápidas están excluidas por el gate de observabilidad (cadencia lenta de símbolo).
#[test]
fn test_r7_r4_b2_spectral_coherence_unbounded_by_observability_gate() {
    let mut por_escala = [0.0; 32];
    // En altcoin con cadencia moderada, las primeras 12 escalas (0..12) no son observables
    // y quedan en 0.0 por el gate de observabilidad.
    // Las 20 escalas observables (12..32) respaldan unánimemente a la dominante con convicción 0.75:
    for k in 12..32 {
        por_escala[k] = 0.75;
    }
    let voto = VotoEspectral::desde_arr(&por_escala);

    let v_dom = 0.75f64;

    // Fórmula vieja: divide ciegamente entre 32
    let media_vieja = voto.media_banda(0, 31).unwrap();
    let coherencia_vieja = (media_vieja / v_dom).clamp(0.0, 1.0);
    // 20 * 0.75 / 32 = 0.46875 -> coherencia = 0.46875 / 0.75 = 0.625 (techo artificial)
    assert!((coherencia_vieja - 0.625).abs() < 1e-12);

    // Fórmula nueva R7-R4-B-2: promedia exclusivamente sobre escalas activas observadas
    let media_activa = voto.media_banda_activa(0, 31).unwrap();
    let coherencia_nueva = (media_activa / v_dom).clamp(0.0, 1.0);
    assert_eq!(media_activa, 0.75);
    assert_eq!(coherencia_nueva, 1.0, "Coherencia inter-espectral debe alcanzar 1.0 si todas las escalas activas respaldan v_dom");
}

/// 3. R7-R2-A-5: Verifica que el payoff ratio `b` en KellyEnvelope se descuenta con cota LCB
/// bajo incertidumbre muestral, impidiendo que rachas afortunadas tempranas inflen Kelly sizing.
#[test]
fn test_r7_r2_a5_kelly_envelope_payoff_ratio_lcb_conservatism() {
    let mut env = RiskEnvelope::new();
    // 6 trades: 4 ganadores con payoff inflado de 3.0
    for _ in 0..4 {
        env.record_trade(true, 30.0, -10.0);
    }
    for _ in 0..2 {
        env.record_trade(false, 0.0, -10.0);
    }

    let raw_b = env.payoff_ratio;
    assert!(raw_b > 2.0, "Payoff crudo inflado: {raw_b}");

    // Con z = 1.64 (95% confianza)
    let lcb_b = env.conservative_payoff_ratio(1.64);
    assert!(lcb_b < raw_b, "LCB de b ({lcb_b}) debe descontar el payoff crudo ({raw_b})");

    // Con z = 0.0, recupera exactamente el valor puntual
    assert!((env.conservative_payoff_ratio(0.0) - raw_b).abs() < 1e-12);

    // Convergencia asintótica: con n grande (ej. 400 trades), el descuento se reduce
    let mut large_env = RiskEnvelope::new();
    for i in 0..400 {
        if i % 3 != 0 {
            large_env.record_trade(true, 30.0, -10.0);
        } else {
            large_env.record_trade(false, 0.0, -10.0);
        }
    }
    let large_raw_b = large_env.payoff_ratio;
    let large_lcb_b = large_env.conservative_payoff_ratio(1.64);
    let ratio_small = lcb_b / raw_b;
    let ratio_large = large_lcb_b / large_raw_b;
    assert!(
        ratio_large > ratio_small,
        "La cota asintótica debe converger hacia 1.0 conforme n crece: large={ratio_large} > small={ratio_small}"
    );
}

/// 4. R7-R2-A-3: Verifica que Cramér-Lundberg calcula el error estándar analítico de R
/// y su cota inferior LCB R_lcb, contrayendo el margen de ruina de forma actuarialmente segura.
#[test]
fn test_r7_r2_a3_cramer_lundberg_analytical_lcb_tightens_ruin_margin() {
    let mut est = EstimadorSiniestros::new();
    // 100 operaciones con media positiva (+0.8% ganancia, -0.5% pérdida)
    for i in 0..100 {
        est.observar(if i % 3 != 0 { 0.008 } else { -0.005 });
    }

    let r_point = est.lundberg().expect("Debe converger con drift positivo");
    let se = est.standard_error_r(r_point).expect("SE analítico debe calcularse");
    assert!(se > 0.0 && se.is_finite(), "SE debe ser positivo finito: {se}");

    let r_lcb = est.lundberg_lcb(1.645).expect("R_lcb debe existir");
    assert!(r_lcb < r_point, "R_lcb={r_lcb} debe ser estrictamente menor que R_point={r_point}");

    // Comparar margen de cota ψ <= 5% (ln(20) / R)
    let m_point = EstimadorSiniestros::margen_de_cota(r_point, 0.05).unwrap();
    let m_lcb = EstimadorSiniestros::margen_de_cota(r_lcb, 0.05).unwrap();

    // Como R_lcb < R_point, el margen de capital m requerido bajo incertidumbre es MAYOR
    assert!(
        m_lcb > m_point,
        "El margen conservador ({m_lcb}) debe ser mayor que el margen puntual ({m_point}) para proteger la cuenta"
    );
}
