//! CONTRATO FORMAL DE UNIFICACIÓN ESPECTRAL DEL UNIVERSO DINÁMICO (Ola Ω215)
//!
//! QUÉ: Verificación rigurosa de invariantes matemáticas del ranking multivariante espectral.
//! POR QUÉ: Erradicar la divergencia histórica entre `data-pipeline` y `execution-engine`.
//! PARA QUÉ: Asegurar selección reproducible, determinista y segura para cuentas micro ($13 USD).
//! CÓMO: 5 contratos formales cubriendo equivalencia de scoring, sanitización y resistencia a fricción.

use quantum_arena::symbols::{
    calculate_spectral_universe_score, is_sane_binance_symbol, CANONICAL_STABLECOINS,
    CANONICAL_TESTNET_BLACKLIST,
};

#[test]
fn c1_sane_symbol_strict_invariants() {
    // 1. Todas las stablecoins canónicas deben ser rechazadas sistemáticamente
    for stable in CANONICAL_STABLECOINS {
        assert!(
            !is_sane_binance_symbol(stable, false),
            "Stablecoin {} debe ser rechazada en prod",
            stable
        );
        assert!(
            !is_sane_binance_symbol(stable, true),
            "Stablecoin {} debe ser rechazada en testnet",
            stable
        );
    }

    // 2. Blacklist de testnet debe ser rechazada en testnet
    for blacklist_sym in CANONICAL_TESTNET_BLACKLIST {
        assert!(
            !is_sane_binance_symbol(blacklist_sym, true),
            "Símbolo en blacklist {} debe ser rechazado en testnet",
            blacklist_sym
        );
    }

    // 3. Mojibake y bases triviales < 3 chars
    let invalid_symbols = [
        "牛来USDT",
        "4USDT",
        "AUSDT",
        "BT",
        "BTC",
        "BTCUS",
        "BTC_USDT",
        "BTC-USDT",
        "btc_usdt",
    ];
    for inv in &invalid_symbols {
        assert!(
            !is_sane_binance_symbol(inv, false),
            "Símbolo inválido {} debe ser rechazado",
            inv
        );
    }

    // 4. Activos legítimos deben ser admitidos
    let legitimate = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "1000PEPEUSDT"];
    for leg in &legitimate {
        assert!(
            is_sane_binance_symbol(leg, false),
            "Activo legítimo {} debe ser aceptado",
            leg
        );
    }
}

#[test]
fn c2_spectral_scoring_unimodal_volatility_curvature() {
    let vol = 10_000_000.0; // $10M volumen

    // Calcular puntajes a lo largo del espectro de volatilidad diaria
    let s_1 = calculate_spectral_universe_score(vol, 1.0, None, None);
    let s_5 = calculate_spectral_universe_score(vol, 5.0, None, None);
    let s_15 = calculate_spectral_universe_score(vol, 15.0, None, None); // Pico óptimo
    let s_30 = calculate_spectral_universe_score(vol, 30.0, None, None);
    let s_70 = calculate_spectral_universe_score(vol, 70.0, None, None); // Extremo / manipulación

    // Verificar curvatura unimodal: s(1) < s(5) < s(15) y s(15) > s(30) > s(70)
    assert!(s_1 < s_5, "s(1%) < s(5%)");
    assert!(s_5 < s_15, "s(5%) < s(15%)");
    assert!(s_15 > s_30, "s(15%) debe ser mayor que s(30%)");
    assert!(s_30 > s_70, "s(30%) debe ser mayor que s(70%)");

    // Verificar valor analítico en el óptimo x = 15:
    // f(15) = 15 * exp(-1) = 15 / e ≈ 5.51819
    let expected_vola = 15.0 / std::f64::consts::E;
    let expected_score = (1.0 + vol).ln() * expected_vola;
    assert!(
        (s_15 - expected_score).abs() < 1e-9,
        "Puntaje en el óptimo debe coincidir exactamente con la solución analítica: {} vs {}",
        s_15,
        expected_score
    );
}

#[test]
fn c3_spectral_scoring_spread_friction_penalty() {
    let vol = 20_000_000.0;
    let pct = 12.0;

    // Spread nulo (0 bps)
    let s_zero_spread = calculate_spectral_universe_score(vol, pct, Some(0.0), None);
    // Spread moderado (5 bps)
    let s_5bps = calculate_spectral_universe_score(vol, pct, Some(5.0), None);
    // Spread penalizante (10 bps = exp(-1) de atenuación)
    let s_10bps = calculate_spectral_universe_score(vol, pct, Some(10.0), None);

    assert!(s_zero_spread > s_5bps, "0 bps debe puntuar mejor que 5 bps");
    assert!(s_5bps > s_10bps, "5 bps debe puntuar mejor que 10 bps");

    let expected_ratio = (-1.0_f64).exp();
    let actual_ratio = s_10bps / s_zero_spread;
    assert!(
        (actual_ratio - expected_ratio).abs() < 1e-9,
        "La atenuación a 10 bps debe ser exactamente exp(-1): {} vs {}",
        actual_ratio,
        expected_ratio
    );
}

#[test]
fn c4_spectral_scoring_fractal_memory_alignment() {
    let vol = 15_000_000.0;
    let pct = 10.0;

    // Ruido blanco puro H = 0.50
    let s_noise = calculate_spectral_universe_score(vol, pct, None, Some(0.50));
    // Persistencia fuerte H = 0.80
    let s_trend = calculate_spectral_universe_score(vol, pct, None, Some(0.80));
    // Antipersistencia fuerte H = 0.20
    let s_mean_rev = calculate_spectral_universe_score(vol, pct, None, Some(0.20));

    // Ambos regímenes con memoria superan al ruido browniano
    assert!(s_trend > s_noise, "Persistencia fractal H=0.80 supera al ruido H=0.50");
    assert!(s_mean_rev > s_noise, "Antipersistencia H=0.20 supera al ruido H=0.50");

    // Simetría de ventaja informacional respecto a H = 0.5
    assert!(
        (s_trend - s_mean_rev).abs() < 1e-9,
        "Simetría fractal |0.8 - 0.5| == |0.2 - 0.5|"
    );
}

#[test]
fn c5_spectral_scoring_nan_and_negative_immunity() {
    let test_cases = [
        (f64::NAN, 15.0, None, None),
        (10_000.0, f64::NAN, None, None),
        (10_000.0, 15.0, Some(f64::NAN), None),
        (10_000.0, 15.0, None, Some(f64::NAN)),
        (-1000.0, 15.0, None, None),
        (10_000.0, -5.0, None, None),
        (f64::INFINITY, 15.0, None, None),
        (10_000.0, f64::INFINITY, None, None),
    ];

    for (v, p, s, h) in test_cases {
        let score = calculate_spectral_universe_score(v, p, s, h);
        assert!(
            score.is_finite() && score >= 0.0,
            "El score debe ser estrictamente finito y no-negativo ante entradas adversas: got {}",
            score
        );
    }
}
