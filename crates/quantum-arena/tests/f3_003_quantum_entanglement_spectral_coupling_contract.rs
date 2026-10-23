//! CONTRATO FORMAL F3.3: Acoplamiento Cuántico Tensor MPS al Universo Espectral & Darwin Online
//! (Ola Omega220 — PR #112).
//!
//! Certifica:
//! 1. Amplificación coherente en fase ordenada ($g < 1.0$, régimen BTC tipo GATO).
//! 2. Atenuación continua monótona en fase paramagnética ($g > 1.0$, régimen ICP ruido disipativo).
//! 3. Continuidad $C^0$ exacta en la frontera crítica $g_c = 1.0$.
//! 4. Amortiguamiento Hermite $C^1$ de susceptibilidad cuántica en la ventana $|g - 1.0| \le \delta_{\text{crit}}$.
//! 5. Detección unívoca de cruce de punto crítico (evento de cambio de régimen macroscópico).
//! 6. Inversión honesta y determinista de $\Delta S$ sobre `CurvaFases` de DMRG (GLM 156) sin pánicos.
//! 7. Inmunidad matemática estricta IEEE 754 (NaN, Inf, negativos, cero).
//! 8. Integración simétrica y continua con `SpectralRegimeField` y respeto estricto de pisos de margen.

use feature_engine::entrelazamiento::CurvaFases;
use quantum_arena::quantum_phase::{
    calcular_factor_cuantico, clasificar_fase, evaluar_fase_cuantica, FaseCuantica,
    QuantumCouplingParams, QuantumPhaseState,
};
use quantum_arena::spectral_regime::{SpectralRegimeField, SpectralRegimeParams};

#[test]
fn test_f3_003_ordered_phase_amplification() {
    let params = QuantumCouplingParams::default();

    // BTC medido empíricamente por GLM 156: g = 0.850
    let g_btc = 0.850;
    let factor_btc = calcular_factor_cuantico(g_btc, &params);
    assert!(
        factor_btc > 1.0,
        "En fase ordenada g < 1.0 el factor debe amplificar (> 1.0): {}",
        factor_btc
    );
    // Para g = 0.850, dist_crit = 0.15 > delta_crit (0.05), phi_crit = 1.0.
    // f_base = 1.0 + (1.0 - 0.85) = 1.15
    assert!(
        (factor_btc - 1.15).abs() < 1e-5,
        "factor esperado 1.15, obtenido: {}",
        factor_btc
    );
    assert_eq!(clasificar_fase(g_btc, params.delta_crit), FaseCuantica::Ordenada);

    // Estado profundamente ordenado g = 0.10 (cercano al GATO puro)
    let g_deep = 0.10;
    let factor_deep = calcular_factor_cuantico(g_deep, &params);
    assert!(
        (factor_deep - 1.90).abs() < 1e-5,
        "factor profundo esperado 1.90, obtenido: {}",
        factor_deep
    );
    assert!(factor_deep <= params.max_factor_cap);
}

#[test]
fn test_f3_003_paramagnetic_phase_attenuation() {
    let params = QuantumCouplingParams::default();

    // ICP medido empíricamente por GLM 156: g = 4.095 (paramagnético profundo)
    let g_icp = 4.095;
    let factor_icp = calcular_factor_cuantico(g_icp, &params);
    assert!(
        factor_icp < 1.0,
        "En fase paramagnética g > 1.0 el factor debe atenuar (< 1.0): {}",
        factor_icp
    );
    let expected_icp = 1.0 / 4.095;
    assert!(
        (factor_icp - expected_icp).abs() < 1e-5,
        "factor ICP esperado {}, obtenido: {}",
        expected_icp,
        factor_icp
    );
    assert_eq!(clasificar_fase(g_icp, params.delta_crit), FaseCuantica::Paramagnetica);

    // Monotonía decreciente estricta en el régimen paramagnético antes del piso de protección
    let mut prev_f = factor_icp;
    for &g in &[5.0, 10.0, 18.0] {
        let f = calcular_factor_cuantico(g, &params);
        assert!(
            f < prev_f,
            "Monotonía violada: f({}) = {} no es menor que f previo {}",
            g,
            f,
            prev_f
        );
        assert!(f >= params.min_factor_floor);
        prev_f = f;
    }

    // Para valores extremos (g >= 20.0 con min_factor_floor = 0.05), se respeta el piso sin perforarlo
    let f_ext = calcular_factor_cuantico(50.0, &params);
    assert_eq!(
        f_ext, params.min_factor_floor,
        "Para g=50.0 el factor debe alcanzar exactamente el piso de protección {}",
        params.min_factor_floor
    );
}

#[test]
fn test_f3_003_continuity_at_criticality() {
    let mut params = QuantumCouplingParams::default();
    params.gamma_crit = 0.0; // Desactiva amortiguamiento para testear continuidad pura de f_base

    // Límite por izquierda y por derecha de g = 1.0
    for eps in [1e-3, 1e-4, 1e-5, 1e-6] {
        let f_left = calcular_factor_cuantico(1.0 - eps, &params);
        let f_right = calcular_factor_cuantico(1.0 + eps, &params);
        let diff = (f_left - f_right).abs();
        assert!(
            diff < 5.0 * eps,
            "Discontinuidad detectada en eps={}: f_left={}, f_right={}, diff={}",
            eps,
            f_left,
            f_right,
            diff
        );
    }
}

#[test]
fn test_f3_003_critical_hermite_damping() {
    let params = QuantumCouplingParams {
        w_quantum: 1.0,
        delta_crit: 0.05,
        gamma_crit: 0.20, // 20% de atenuación en criticidad
        min_factor_floor: 0.05,
        max_factor_cap: 2.0,
    };

    // Exactamente en el punto crítico g = 1.0:
    // u = 0, H(0) = 0, phi = 1.0 - 0.20 = 0.80.
    // f_base(1.0) = 1.0 -> f = 0.80.
    let f_crit = calcular_factor_cuantico(1.0, &params);
    assert!(
        (f_crit - 0.80).abs() < 1e-5,
        "f en punto crítico exacto esperado 0.80, obtenido: {}",
        f_crit
    );
    assert_eq!(clasificar_fase(1.0, params.delta_crit), FaseCuantica::Critica);

    // En la frontera del radio crítico g = 1.05 (dist_crit = 0.05 = delta_crit):
    // u = 1.0, H(1) = 1.0, phi = 1.0.
    // f_base(1.05) = 1.0 / 1.05 ≈ 0.95238
    let f_border = calcular_factor_cuantico(1.05, &params);
    let expected_border = 1.0 / 1.05;
    assert!(
        (f_border - expected_border).abs() < 1e-4,
        "f en frontera crítica esperado {}, obtenido: {}",
        expected_border,
        f_border
    );
}

#[test]
fn test_f3_003_crossing_detection() {
    let params = QuantumCouplingParams::default();

    // Curva DMRG simplificada para prueba
    let curva = CurvaFases::construir(6, 4, 3, 0.05, 20.0, 8).expect("curva valida");

    // Transición de Ordenado (g=0.80) a Desordenado (g=1.25): hubo cruce de g_c = 1.0
    let st_trans = evaluar_fase_cuantica(0.30, &curva, Some(0.80), &params);
    assert!(
        st_trans.g > 1.0,
        "g esperado en lado paramagnético, obtenido: {}",
        st_trans.g
    );
    assert!(
        st_trans.es_cruce_critico,
        "Debe detectar cruce crítico al pasar de 0.80 a {}",
        st_trans.g
    );

    // Dos observaciones consecutivas en el mismo lado (sin cruce)
    let st_no_trans = evaluar_fase_cuantica(0.30, &curva, Some(st_trans.g), &params);
    assert!(
        !st_no_trans.es_cruce_critico,
        "No debe detectar cruce si permanece en el mismo lado"
    );
}

#[test]
fn test_f3_003_inversion_from_curva_fases_dmrg() {
    let params = QuantumCouplingParams::default();
    let curva = CurvaFases::construir(6, 4, 4, 0.05, 20.0, 10).expect("curva valida");

    // Inversión normal dentro de la curva
    let s_mid = (curva.s_max + curva.puntos.iter().map(|p| p.s).fold(f64::INFINITY, f64::min)) * 0.5;
    let st = evaluar_fase_cuantica(s_mid, &curva, None, &params);
    assert!(st.g.is_finite() && st.g > 0.0);
    assert!(st.factor_cuantico > 0.0);
    assert!(st.distancia_criticidad_log.is_finite());

    // Hiper-coherencia: Delta S superior al máximo del GATO en L sitios
    let s_hyper = curva.s_max + 0.5;
    let st_hyper = evaluar_fase_cuantica(s_hyper, &curva, None, &params);
    assert_eq!(
        st_hyper.g, curva.puntos[0].g,
        "Hiper-coherencia debe anclarse honestamente en g_min"
    );
    assert_eq!(st_hyper.fase, FaseCuantica::Ordenada);

    // Ruido extremo: Delta S por debajo del piso
    let s_noise = -0.5;
    let st_noise = evaluar_fase_cuantica(s_noise, &curva, None, &params);
    assert_eq!(
        st_noise.g,
        curva.puntos.last().unwrap().g,
        "Ruido sub-piso debe anclarse honestamente en g_max"
    );
    assert_eq!(st_noise.fase, FaseCuantica::Paramagnetica);
}

#[test]
fn test_f3_003_ieee754_immunity() {
    let params = QuantumCouplingParams::default();

    // Valores no finitos o patológicos en calcular_factor_cuantico
    assert_eq!(calcular_factor_cuantico(f64::NAN, &params), 1.0);
    assert_eq!(calcular_factor_cuantico(f64::INFINITY, &params), 1.0);
    assert_eq!(calcular_factor_cuantico(f64::NEG_INFINITY, &params), 1.0);
    assert_eq!(calcular_factor_cuantico(-5.0, &params), 1.0);
    assert_eq!(calcular_factor_cuantico(0.0, &params), 1.0);

    // Valores no finitos en evaluar_fase_cuantica
    let curva = CurvaFases::construir(6, 4, 3, 0.05, 20.0, 8).expect("curva valida");
    let st_nan = evaluar_fase_cuantica(f64::NAN, &curva, None, &params);
    assert_eq!(st_nan, QuantumPhaseState::default());
}

#[test]
fn test_f3_003_spectral_regime_integration() {
    let mut spec = quantum_arena::temporal_spectrum::TemporalSpectrum::new();
    for t in 0..50u64 {
        spec.update(100.0 + (t as f64) * 0.1, 1_000 + t * 1_000);
    }

    let params = SpectralRegimeParams::default();
    let campo = SpectralRegimeField::from_spectrum_with_params(&spec, None, 60_000.0, &params);

    // Estado ordenado (e.g. BTC g=0.85): debe amplificar margen efectivo
    let st_ordenada = QuantumPhaseState {
        g: 0.85,
        delta_s: 0.50,
        fase: FaseCuantica::Ordenada,
        factor_cuantico: 1.15,
        distancia_criticidad_log: 0.85f64.ln().abs(),
        es_cruce_critico: false,
    };
    let campo_mod_ord = campo.with_quantum_phase(&st_ordenada);
    assert_eq!(campo_mod_ord.quantum_phase_factor, 1.15);
    assert_eq!(campo_mod_ord.quantum_coupling_g, 0.85);

    let margin_long_base = campo.margin_multiplier(true);
    let margin_long_ord = campo_mod_ord.effective_quantum_margin_multiplier(true, &params);
    assert!(
        margin_long_ord >= margin_long_base,
        "En fase ordenada el margen efectivo ({}) debe ser >= base ({})",
        margin_long_ord,
        margin_long_base
    );

    // Estado paramagnético (e.g. ICP g=4.095): debe contraer suavemente el margen
    let st_paramag = QuantumPhaseState {
        g: 4.095,
        delta_s: 0.05,
        fase: FaseCuantica::Paramagnetica,
        factor_cuantico: 1.0 / 4.095,
        distancia_criticidad_log: 4.095f64.ln().abs(),
        es_cruce_critico: false,
    };
    let campo_mod_para = campo.with_quantum_phase(&st_paramag);
    let margin_long_para = campo_mod_para.effective_quantum_margin_multiplier(true, &params);
    assert!(
        margin_long_para < margin_long_base,
        "En fase paramagnética el margen efectivo ({}) debe contraerse respecto a base ({})",
        margin_long_para,
        margin_long_base
    );
    assert!(
        margin_long_para >= params.min_margin_floor,
        "Nunca debe perforar el piso mínimo de margen: {}",
        margin_long_para
    );
}
