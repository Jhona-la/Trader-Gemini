//! CONTRATO FORMAL: DESACOPLAMIENTO MULTIACTIVO DE VOLATILIDAD EN EL CONSEJO DELIBERATIVO (F1-008, Ola Ω151).
//!
//! # Por qué existe (F1-008)
//! Anteriormente, `lib.rs:9021` calculaba `relative_vol = (raw / btc).clamp(0.5, 4.0)`
//! y `council_atr_pct = raw / relative_vol`, lo cual para toda altcoin con volatilidad entre
//! 0.5x y 4.0x la de BTC resultaba en una identidad cancelante:
//! `raw / (raw / btc) == btc_atr_pct`.
//! Esto hacía que `SeniorVolatilidad` y `SeniorEjecucion` evaluaran a las 29 altcoins
//! con la volatilidad de BTC en vez de con la propia, distorsionando convicciones y tolerancias a slippage.
//!
//! # Solución Canónica (Ola Ω151)
//! 1. El payload del Consejo recibe directamente el ATR propio del activo (`raw_atr_pct`),
//!    sin identidades cancelantes ni barridos de strings O(N) por candidato.
//! 2. `SeniorVolatilidad` evalúa el régimen mediante `evaluate_volatility_regime_spectral(raw_atr_pct, s, ...)`,
//!    donde las bandas de calma/peligro se expanden suavemente a lo largo del espectro continuo
//!    temporal τ (s ∈ [0, 1]), permitiendo que activos con mayor volatilidad natural u horizontes
//!    más lentos operen saludablemente mientras se preserva la protección ante explosiones anómalas.
//! 3. `SeniorEjecucion` calibra la tolerancia a slippage con el ATR real del activo (`(atr_bps / 10.0).sqrt()`).

use metacortex_engine::consejo_seniors::{
    evaluate_volatility_regime_spectral, SeniorAgent, SeniorOpinion, SeniorRole,
    SeniorVolatilidad, MarketSnapshotPayload, CALM_ATR_PCT, DANGER_ATR_PCT,
};

fn make_payload(atr_pct: f64, tau_ms: f64, dir: f64) -> MarketSnapshotPayload {
    MarketSnapshotPayload {
        atr_pct,
        dominant_tau_ms: tau_ms,
        intended_direction: dir,
        fast_slippage_floor: 35.0,
        slow_slippage_span: 65.0,
        ..MarketSnapshotPayload::default()
    }
}

#[test]
fn contract_f1_008_btc_operates_in_healthy_regime() {
    let senior = SeniorVolatilidad;
    // BTC con 20 bps a escala intermedia (tau = 15 min = 900_000 ms)
    let payload = make_payload(0.0020, 900_000.0, 1.0);
    let op: SeniorOpinion = senior.evaluate(&payload, 0.65);

    assert_eq!(op.role, SeniorRole::Volatilidad);
    assert_eq!(op.signal_direction, 1.0);
    assert!(!op.is_veto);
    // Convicción sana entre 0.60 y 1.0 (calma otorga convicción plena 1.0)
    assert!(
        op.confidence >= 0.60 && op.confidence <= 1.0,
        "BTC a 20 bps debe tener convicción sana, obtenido: {}",
        op.confidence
    );
}

#[test]
fn contract_f1_008_altcoin_high_natural_atr_not_choked_at_slower_scales() {
    let senior = SeniorVolatilidad;
    // Altcoin con ATR propio de 60 bps (0.0060) en horizonte swing/meso (tau = 4 horas = 14_400_000 ms)
    let payload = make_payload(0.0060, 14_400_000.0, 1.0);
    let op = senior.evaluate(&payload, 0.65);

    assert_eq!(op.signal_direction, 1.0);
    assert!(!op.is_veto);
    // Antes de F1-008, con bandas fijas de 10..40 bps, 60 bps quedaba asfixiado en 0.40 pegado al piso.
    // Con expansión espectral continua a tau=4h, las bandas se expanden y la convicción es > 0.50.
    assert!(
        op.confidence > 0.50,
        "Altcoin a 60 bps en tau=4h no debe estar pegada al piso de 0.40, obtenido: {}",
        op.confidence
    );
}

#[test]
fn contract_f1_008_anomalous_volatility_explosion_triggers_minimum_confidence() {
    let senior = SeniorVolatilidad;
    // Explosión extrema anómala a 350 bps (0.0350)
    let payload = make_payload(0.0350, 60_000.0, 1.0);
    let op = senior.evaluate(&payload, 0.65);

    assert_eq!(op.signal_direction, 1.0);
    assert!(!op.is_veto);
    assert!(
        (op.confidence - 0.40).abs() < 1e-6,
        "Explosión anómala extrema debe activar convicción mínima de 0.40, obtenido: {}",
        op.confidence
    );
}

#[test]
fn contract_f1_008_spectral_expansion_monotonic_across_tau() {
    // Para el mismo ATR (30 bps), a mayor escala temporal tau, las bandas se expanden
    // y la convicción por volatilidad es monotónicamente no-decreciente.
    let atr = 0.0030;
    let _tau_rapida = 30_000.0f64;    // 30 s (s = 0.0)
    let tau_media = 3_600_000.0f64;   // 1 h
    let _tau_lenta = 43_200_000.0f64;  // 12 h (s = 1.0)

    let s_rapida = 0.0;
    let s_media = (tau_media.ln() - 30_000.0f64.ln()) / (43_200_000.0f64.ln() - 30_000.0f64.ln());
    let s_lenta = 1.0;

    let f_rapida = evaluate_volatility_regime_spectral(atr, s_rapida, CALM_ATR_PCT, DANGER_ATR_PCT, 2.5);
    let f_media = evaluate_volatility_regime_spectral(atr, s_media, CALM_ATR_PCT, DANGER_ATR_PCT, 2.5);
    let f_lenta = evaluate_volatility_regime_spectral(atr, s_lenta, CALM_ATR_PCT, DANGER_ATR_PCT, 2.5);

    assert!(f_media >= f_rapida, "Convicción a 1h ({f_media}) debe ser >= a 30s ({f_rapida})");
    assert!(f_lenta >= f_media, "Convicción a 12h ({f_lenta}) debe ser >= a 1h ({f_media})");
}
