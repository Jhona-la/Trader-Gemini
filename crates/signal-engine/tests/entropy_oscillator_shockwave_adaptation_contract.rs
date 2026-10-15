//! 🧪 CONTRATO DE ADAPTACIÓN ESPECTRAL: ENTROPÍA NO-EXTENSIVA, OSCILADOR CUÁNTICO Y ONDA DE CHOQUE
//!
//! Ola Ω123: Verificación rigurosa de desacoplamiento paramétrico, continuidad C¹ y fail-closed
//! para RenyiTsallisEntropyEngine, QuantumOscillatorEngine, y SupersonicShockwaveEngine.

use signal_engine::{
    QuantumOscillatorEngine, QuantumOscillatorParams,
    RenyiTsallisEntropyEngine, RenyiTsallisEntropyParams,
    SupersonicShockwaveEngine, SupersonicShockwaveParams,
};

#[test]
fn test_renyi_tsallis_params_validation_and_defaults() {
    let default_params = RenyiTsallisEntropyParams::default();
    assert_eq!(default_params.q_tsallis, 1.5);
    assert_eq!(default_params.alpha_renyi, 2.0);
    assert_eq!(default_params.entropy_threshold_low, 0.40);
    assert_eq!(default_params.entropy_threshold_high, 0.60);
    assert!(default_params.validate().is_ok());

    // Validar rechazo de valores no-finitos o fuera de rango
    let invalid_q = RenyiTsallisEntropyParams {
        q_tsallis: f64::NAN,
        ..default_params
    };
    assert!(invalid_q.validate().is_err());

    let inverted_entropy = RenyiTsallisEntropyParams {
        entropy_threshold_low: 0.70,
        entropy_threshold_high: 0.30,
        ..default_params
    };
    assert!(inverted_entropy.validate().is_err());

    let inverted_obi = RenyiTsallisEntropyParams {
        obi_threshold_low: 0.50,
        obi_threshold_high: 0.20,
        ..default_params
    };
    assert!(inverted_obi.validate().is_err());
}

#[test]
fn test_renyi_tsallis_engine_adaptation_and_smoothness() {
    let custom_params = RenyiTsallisEntropyParams {
        q_tsallis: 2.0,
        alpha_renyi: 1.5,
        entropy_threshold_low: 0.30,
        entropy_threshold_high: 0.70,
        obi_threshold_low: 0.10,
        obi_threshold_high: 0.40,
    };

    let mut engine = RenyiTsallisEntropyEngine::new(1.5, 2.0).with_params(custom_params);
    assert_eq!(engine.params, custom_params);
    assert_eq!(engine.q_tsallis, 2.0);
    assert_eq!(engine.alpha_renyi, 1.5);

    // Dynamic update
    let updated_params = RenyiTsallisEntropyParams {
        q_tsallis: 0.8,
        alpha_renyi: 3.0,
        ..custom_params
    };
    engine.update_params(updated_params);
    assert_eq!(engine.params, updated_params);
    assert_eq!(engine.q_tsallis, 0.8);
    assert_eq!(engine.alpha_renyi, 3.0);

    // Vote with params: verificar rampa C1 suave y monotonía respecto a OBI
    let vote_neutral = RenyiTsallisEntropyEngine::vote_with_params(0.2, 0.0, &updated_params);
    assert_eq!(vote_neutral, 0.0);

    let vote_low_obi = RenyiTsallisEntropyEngine::vote_with_params(0.2, 0.05, &updated_params);
    assert_eq!(vote_low_obi, 0.0); // Debajo de obi_threshold_low (0.10)

    let vote_mid_obi = RenyiTsallisEntropyEngine::vote_with_params(0.2, 0.25, &updated_params);
    let vote_high_obi = RenyiTsallisEntropyEngine::vote_with_params(0.2, 0.50, &updated_params);
    assert!(vote_mid_obi > 0.0);
    assert!(vote_high_obi > vote_mid_obi);
    assert!(vote_high_obi <= 1.0);

    // Fail-closed en NaN
    assert_eq!(RenyiTsallisEntropyEngine::vote_with_params(f64::NAN, 0.5, &updated_params), 0.0);
    assert_eq!(RenyiTsallisEntropyEngine::vote_with_params(0.2, f64::NAN, &updated_params), 0.0);
}

#[test]
fn test_quantum_oscillator_params_validation_and_force() {
    let default_params = QuantumOscillatorParams::default();
    assert_eq!(default_params.k_spring, 1.0);
    assert_eq!(default_params.lambda_anharmonic, 0.1);
    assert_eq!(default_params.alpha_confinement, 0.5);
    assert!(default_params.validate().is_ok());

    let invalid_k = QuantumOscillatorParams {
        k_spring: -1.0,
        ..default_params
    };
    assert!(invalid_k.validate().is_err());

    let custom_params = QuantumOscillatorParams {
        k_spring: 2.5,
        lambda_anharmonic: 0.25,
        alpha_confinement: 0.8,
    };
    let engine = QuantumOscillatorEngine::new().with_params(custom_params);
    assert_eq!(engine.params, custom_params);

    // Restoring force: F(x) = -(k*x + 4*lambda*x^3)
    let pos = 1.0;
    let expected_force = -(2.5 * 1.0 + 4.0 * 0.25 * 1.0); // -(2.5 + 1.0) = -3.5
    let force = QuantumOscillatorEngine::compute_quantum_restoring_force(
        pos,
        custom_params.k_spring,
        custom_params.lambda_anharmonic,
    );
    assert!((force - expected_force).abs() < 1e-9);

    // Spectral vote responds across 32 scales with confinement
    let mut desplazamientos = [0.0; 32];
    desplazamientos[0] = 0.5;
    desplazamientos[15] = -1.2;
    desplazamientos[31] = f64::NAN;

    let voto = QuantumOscillatorEngine::voto_espectral_with_params(&desplazamientos, &custom_params);
    let scale_0 = voto.en_escala(0);
    let scale_15 = voto.en_escala(15);
    let scale_31 = voto.en_escala(31);

    assert!(scale_0 < 0.0); // Force opposes positive displacement
    assert!(scale_15 > 0.0); // Force opposes negative displacement
    assert_eq!(scale_31, 0.0); // Fail-closed on NaN
}

#[test]
fn test_supersonic_shockwave_params_validation_and_mach_jump() {
    let default_params = SupersonicShockwaveParams::default();
    assert_eq!(default_params.sonic_speed_z, 1.0);
    assert_eq!(default_params.mach_threshold, 1.0);
    assert_eq!(default_params.signature_scale, 2.0);
    assert!(default_params.validate().is_ok());

    let invalid_mach = SupersonicShockwaveParams {
        mach_threshold: 0.0,
        ..default_params
    };
    assert!(invalid_mach.validate().is_err());

    let custom_params = SupersonicShockwaveParams {
        sonic_speed_z: 1.5,
        mach_threshold: 1.2,
        signature_scale: 1.8,
        min_speed_of_sound: 1e-5,
    };
    let mut engine = SupersonicShockwaveEngine::new().with_params(custom_params);
    assert_eq!(engine.params, custom_params);

    let updated_params = SupersonicShockwaveParams {
        mach_threshold: 1.4,
        ..custom_params
    };
    engine.update_params(updated_params);
    assert_eq!(engine.params.mach_threshold, 1.4);

    // Mach number computation
    let mach = SupersonicShockwaveEngine::compute_mach_number(2.8, 1.4);
    assert!((mach - 2.0).abs() < 1e-9);

    // Shockwave jump: threshold = 1.4 -> mach 1.3 is subsonic (no jump), mach 2.0 is supersonic
    let jump_subsonic = SupersonicShockwaveEngine::compute_shockwave_jump_with_threshold(1.3, 1.4);
    assert_eq!(jump_subsonic, 0.0);

    let jump_supersonic = SupersonicShockwaveEngine::compute_shockwave_jump_with_threshold(2.0, 1.4);
    assert!(jump_supersonic > 0.0 && jump_supersonic <= 1.0);

    // Spectral vote with params
    let mut desplazamientos = [0.0; 32];
    desplazamientos[5] = 3.0; // Significant supersonic displacement
    desplazamientos[10] = -3.0;
    desplazamientos[20] = f64::NAN;

    let voto = SupersonicShockwaveEngine::voto_espectral_with_params(&desplazamientos, &updated_params);
    let scale_5 = voto.en_escala(5);
    let scale_10 = voto.en_escala(10);
    let scale_20 = voto.en_escala(20);

    assert!(scale_5 > 0.0);
    assert!(scale_10 < 0.0);
    assert!((scale_5 + scale_10).abs() < 1e-9); // Antisymmetric signed signature
    assert_eq!(scale_20, 0.0); // Fail-closed on NaN
}
