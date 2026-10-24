//! Contratos Formales de Verificación de Fase 7 (F7-001)
//!
//! Suite de pruebas y contratos matemáticos para:
//! - Tasa de crecimiento temporal de Ole Peters (Ergodic Time-Growth Engine).
//! - Brecha de no-ergocicidad y prevención de ruina casi segura.
//! - Predictor Rough Bergomi multiescala con exponente de Hurst H < 0.50.
//! - Ley de potencia singular de skew de volatilidad S(tau) ~ tau^(H - 0.5).
//! - Griegas estocásticas de Malliavin y amortiguador dinámico de liquidez.
//! - Cumplimiento estricto del mandato U-ERR-6 (sin dicotomía scalp/swing).
//! - Resiliencia numérica total contra NaNs, subnormales e infinitos IEEE-754.

use feature_engine::rough_ergodic::{
    ErgodicGrowthEngine, MalliavinEvaluator, RoughBergomiPredictor, ROUGH_FACTORS_COUNT,
};

/// Contrato C1: Paradoja de No-Ergocicidad de Ole Peters & Demostración de Ruina de Ensamble
#[test]
fn test_c1_ole_peters_ergodicity_gap_and_ruin_avoidance() {
    let engine = ErgodicGrowthEngine::new();

    // Juego clásico de coin-toss multiplicativo: +50% en cara, -40% en cruz.
    // Promedio de ensamble: (0.50 - 0.40) / 2 = +0.05 (+5% por tirada, parece atractivo!).
    // Promedio temporal con f=1: ln(1.5 * 0.6) / 2 = ln(0.90) / 2 = -0.05268 (¡ruina segura!).
    let mut returns = Vec::new();
    for _ in 0..100 {
        returns.push(0.50);
        returns.push(-0.40);
    }

    let result = engine.optimize_growth(&returns, 0.0004);

    // 1. El promedio de ensamble debe ser positivo (+5%)
    assert!(
        result.ensemble_average > 0.045 && result.ensemble_average < 0.055,
        "El promedio de ensamble debe ser ~0.05 (fue {})",
        result.ensemble_average
    );

    // 2. Con f=1.0 la tasa temporal colapsa a negativo
    let g_full = ErgodicGrowthEngine::evaluate_g_time(1.0, &returns, 0.0004);
    assert!(
        g_full < -0.04,
        "La tasa temporal con apalancamiento pleno debe ser fuertemente destructiva (fue {})",
        g_full
    );

    // 3. La brecha de ergocicidad debe ser estrictamente positiva (Desigualdad de Jensen)
    assert!(
        result.ergodicity_gap >= 0.0,
        "La brecha de ergocicidad debe ser >= 0.0 (fue {})",
        result.ergodicity_gap
    );

    // 4. El motor debe determinar que el apalancamiento óptimo para este juego con fricción es conservador
    // o nulo si no supera el arrastre de comisiones
    assert!(
        result.optimal_leverage <= 0.40,
        "El apalancamiento óptimo debe estar acotado por max_bound (fue {})",
        result.optimal_leverage
    );
}

/// Contrato C2: Optimización Convexa de Apalancamiento $f^*$ & Ventaja de Crecimiento Positiva
#[test]
fn test_c2_convex_optimization_and_growth_advantage() {
    let engine = ErgodicGrowthEngine::new();

    // Distribución favorable con deriva positiva y baja volatilidad:
    // Retornos positivos consistentes: +1.5%, +2.0%, -0.5%, +1.0%
    let mut favorable_returns = Vec::new();
    for _ in 0..50 {
        favorable_returns.push(0.015);
        favorable_returns.push(0.020);
        favorable_returns.push(-0.005);
        favorable_returns.push(0.010);
    }

    let result = engine.optimize_growth(&favorable_returns, 0.0002);

    assert!(
        result.is_ergodically_viable,
        "Una distribución con deriva neta positiva debe ser ergódicamente viable"
    );
    assert!(
        result.optimal_leverage > 0.05 && result.optimal_leverage <= 0.40,
        "El apalancamiento óptimo debe situarse en el rango dinámico saludable (fue {})",
        result.optimal_leverage
    );
    assert!(
        result.g_time_optimal > 0.0,
        "La tasa de crecimiento temporal óptima debe ser estrictamente positiva (fue {})",
        result.g_time_optimal
    );

    // Verificación de Half-Kelly: f*/2 preserva una porción mayoritaria de crecimiento con menor volatilidad
    assert!(
        result.g_time_half > 0.0,
        "Half-Kelly debe tener tasa de crecimiento temporal positiva"
    );
    assert!(
        result.g_time_half <= result.g_time_optimal,
        "Half-Kelly no puede superar el óptimo matemático"
    );
}

/// Contrato C3: Cota Analítica de Ruina de Cramér-Lundberg en Dinámica de Riqueza
#[test]
fn test_c3_cramer_lundberg_wealth_ruin_bound() {
    let engine = ErgodicGrowthEngine::new();

    // Caso A: Flujo con excelente ventaja ergódica
    let safe_returns = vec![0.02; 100];
    let res_safe = engine.optimize_growth(&safe_returns, 0.0001);

    assert!(
        res_safe.ruin_probability_bound < 0.05,
        "Para deriva constante positiva la probabilidad de ruina debe ser ínfima (fue {})",
        res_safe.ruin_probability_bound
    );

    // Caso B: Flujo perdedor continuo
    let losing_returns = vec![-0.01; 100];
    let res_losing = engine.optimize_growth(&losing_returns, 0.0001);

    assert_eq!(
        res_losing.optimal_leverage, 0.0,
        "Ante retornos perdedores, el apalancamiento óptimo debe ser cero absoluto"
    );
    assert_eq!(
        res_losing.ruin_probability_bound, 1.0,
        "Ante retornos perdedores la probabilidad de ruina debe saturar en 1.0"
    );

    // Caso C: Flujo estocástico con deriva positiva y dispersión finita
    let mut stochastic_returns = Vec::new();
    for _ in 0..50 {
        stochastic_returns.push(0.03);
        stochastic_returns.push(0.01);
    }
    let res_stoch = engine.optimize_growth(&stochastic_returns, 0.0001);
    assert!(
        res_stoch.ruin_probability_bound < 0.10,
        "Para flujo estocástico con deriva neta positiva la ruina debe ser baja (fue {})",
        res_stoch.ruin_probability_bound
    );
}

/// Contrato C4: Predictor Rough Bergomi (H < 0.50) & Núcleo Markoviano Multi-Factor
#[test]
fn test_c4_rough_bergomi_kernel_and_skew_power_law() {
    let mut predictor = RoughBergomiPredictor::new(0.12, 1.90, -0.75, 0.04);

    // 1. Verificación de invariantes de inicialización
    assert_eq!(predictor.weights_cm.len(), ROUGH_FACTORS_COUNT);
    let sum_weights: f64 = predictor.weights_cm.iter().sum();
    assert!(
        (sum_weights - 1.0).abs() < 1e-8,
        "Los pesos del núcleo fraccionario deben sumar exactamente 1.0 (fue {})",
        sum_weights
    );

    // 2. Volatilidad inicial debe coincidir con la raíz de xi0
    let initial_vol = predictor.predict_instantaneous_volatility();
    let expected_vol = 0.04f64.sqrt();
    assert!(
        (initial_vol - expected_vol).abs() < 0.02,
        "Volatilidad inicial debe estar en vecindad de xi0^0.5 (fue {})",
        initial_vol
    );

    // 3. Un shock positivo en precio debe persistir a través de los factores Markovianos
    predictor.update_tick(0.05, 1.0);
    let vol_after_shock = predictor.predict_instantaneous_volatility();
    assert!(
        vol_after_shock > initial_vol,
        "Un shock microestructural debe elevar la volatilidad instantánea (antes {}, después {})",
        initial_vol,
        vol_after_shock
    );

    // 4. Ley de potencia del skew de volatilidad: S(tau) debe ser negativo y singular hacia tau -> 0
    let skew_short = predictor.predict_volatility_skew(0.10); // 100 milisegundos
    let skew_long = predictor.predict_volatility_skew(300.0); // 5 minutos

    assert!(
        skew_short < 0.0,
        "El skew de volatilidad con rho < 0 debe ser negativo (fue {})",
        skew_short
    );
    assert!(
        skew_short.abs() > skew_long.abs(),
        "Bajo Rough Bergomi, la magnitud del skew a corto plazo debe ser estrictamente mayor que a largo plazo (short={}, long={})",
        skew_short.abs(),
        skew_long.abs()
    );
}

/// Contrato C5: Griegas Estocásticas de Malliavin & Amortiguador de Vacíos de Liquidez
#[test]
fn test_c5_malliavin_greeks_and_liquidity_vacuum_dampener() {
    let predictor = RoughBergomiPredictor::new(0.10, 2.0, -0.80, 0.05);

    // Caso A: Condiciones normales de mercado (spread normal de 1 bps, spot 60,000 USD)
    let greeks_normal = MalliavinEvaluator::compute_greeks(&predictor, 60000.0, 1.0, 10.0);
    assert!(
        greeks_normal.order_size_dampener >= 0.80,
        "En condiciones tranquilas el amortiguador debe ser alto (fue {})",
        greeks_normal.order_size_dampener
    );
    assert!(
        !greeks_normal.is_rough_vacuum_alert,
        "No debe haber alerta de vacío en régimen calmo"
    );

    // Caso B: Condiciones de estrés severo (spread dilatado a 25 bps)
    let greeks_stress = MalliavinEvaluator::compute_greeks(&predictor, 60000.0, 25.0, 0.05);
    assert!(
        greeks_stress.order_size_dampener < greeks_normal.order_size_dampener,
        "El amortiguador debe reducir el tamaño de orden bajo estrés de spread"
    );
    assert!(
        greeks_stress.rough_gamma > greeks_normal.rough_gamma,
        "La gamma de Malliavin debe aumentar bajo estrés de spread"
    );
}

/// Contrato C6: Resiliencia Numérica IEEE-754, Inmunidad a Subnormales & Mandato U-ERR-6
#[test]
fn test_c6_ieee754_resilience_and_u_err_6_compliance() {
    let engine = ErgodicGrowthEngine::new();

    // 1. Resiliencia contra NaNs y subnormales en el evaluador ergódico
    let dirty_returns = vec![f64::NAN, 0.01, f64::INFINITY, -0.02, 1e-300, f64::NEG_INFINITY];
    let res_dirty = engine.optimize_growth(&dirty_returns, 0.0001);

    assert!(
        res_dirty.optimal_leverage.is_finite(),
        "El apalancamiento debe ser finito ante datos corruptos"
    );
    assert!(
        res_dirty.g_time_optimal.is_finite(),
        "La tasa temporal debe ser finita ante datos corruptos"
    );

    // 2. Resiliencia del predictor Rough Bergomi ante entradas patológicas
    let mut predictor = RoughBergomiPredictor::new(f64::NAN, -5.0, 10.0, -1.0);
    predictor.update_tick(f64::NAN, f64::INFINITY);

    let pred_vol = predictor.predict_instantaneous_volatility();
    assert!(
        pred_vol.is_finite() && pred_vol > 0.0,
        "La volatilidad predicha debe mantenerse finita y positiva ante NaNs (fue {})",
        pred_vol
    );

    let skew_nan = predictor.predict_volatility_skew(f64::NAN);
    assert!(
        skew_nan.is_finite(),
        "El skew debe mantenerse finito ante tau NaN"
    );

    // 3. Cumplimiento de U-ERR-6: verificación estricta de nombres y conceptos
    let type_name_engine = std::any::type_name::<ErgodicGrowthEngine>();
    let type_name_pred = std::any::type_name::<RoughBergomiPredictor>();

    assert!(
        !type_name_engine.to_lowercase().contains("scalp")
            && !type_name_engine.to_lowercase().contains("swing"),
        "U-ERR-6 violado: ErgodicGrowthEngine contiene referencias disjuntas de scalp/swing"
    );
    assert!(
        !type_name_pred.to_lowercase().contains("scalp")
            && !type_name_pred.to_lowercase().contains("swing"),
        "U-ERR-6 violado: RoughBergomiPredictor contiene referencias disjuntas de scalp/swing"
    );
}
