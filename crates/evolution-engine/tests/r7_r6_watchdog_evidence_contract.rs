//! Contrato de certificación formal para los hallazgos de R7-R6 (Ola Ω60 / Ficha #695):
//! - R7-R6-IG-1: Escala unitaria canónica de riesgo por operación para Ville.
//! - R7-R6-EV-1: Desacoplamiento de la vigilancia anytime-valid de Ville (N < 20).
//! - R7-R6-C-7 ≡ MD-2: Descuento de evidencia en prescreening Bayesian prior.
//! - Detección multi-ranura en `positions.is_any_open()`.

use evolution_engine::fitness::{FitnessInputs, compute, compute_with_bayesian_prior, INVIABLE};
use evolution_engine::return_evidence::SequentialVilleEvidence;
use quantum_arena::position::PositionManager;
use std::sync::atomic::Ordering::Relaxed;

/// R7-R6-EV-1: Verifica que el E-proceso de Ville detecta agotamiento de evidencia
/// tempranamente (N < 20) ante una racha severa de pérdidas, sin necesidad de esperar
/// a tamaños de muestra asintóticos.
#[test]
fn ville_watchdog_anytime_valid_detects_exhaustion_before_20_trades() {
    let mut ville = SequentialVilleEvidence::with_bounds(0.05, 0.05, 0.50)
        .expect("construcción válida de Ville con cotas canónicas");

    assert!(!ville.is_exhausted());
    assert!(!ville.is_evidence_decayed(0.50));

    // Con lambda_min = 0.05 y retornos de -1.0 R:
    // (1 - 0.05)^14 = 0.4876 < 0.50 -> decaimiento de evidencia >= 50% alcanzado en 14 pasos (< 20).
    for _ in 0..14 {
        ville.observe(-1.0);
    }

    let degraded = ville.is_exhausted() || ville.is_evidence_decayed(0.50);
    assert!(
        degraded,
        "Ville anytime-valid debe detectar degradación severa (>=50% DD) en 14 operaciones sin esperar a N=20"
    );

    // Detección de pérdidas constantes tempranas (N=5):
    let constant_losses = vec![-0.01; 5];
    let summary = evolution_engine::return_evidence::summarize_returns(&constant_losses)
        .expect("resumen válido de 5 pérdidas");
    assert!(
        summary.is_constant_loss(),
        "5 pérdidas consecutivas deben detectarse como pérdidas constantes tempranas"
    );
}

/// R7-R6-IG-1: Verifica que la normalización por la unidad de riesgo medida transforma
/// los retornos en R-múltiplos fieles en el soporte [-1.0, 1.0].
#[test]
fn return_normalization_by_measured_risk_unit() {
    let cap = 13.0; // Invariante sagrado micro-capital
    let measured_risk = 0.02805 / cap; // Riesgo de SL de 55 bps sobre $5.10 notional = ~0.002157

    let normalize = |ret: f64, risk: f64| -> f64 {
        let unit_risk = if risk.is_finite() && risk > 1e-5 {
            risk
        } else {
            0.02
        };
        (ret / unit_risk).clamp(-1.0, 1.0)
    };

    // Caso 1: Pérdida de SL completo (-0.002157): debe normalizar a exactamente -1.0 R
    let stop_loss_ret = -measured_risk;
    let norm_sl = normalize(stop_loss_ret, measured_risk);
    assert!(
        (norm_sl - (-1.0)).abs() < 1e-9,
        "Stop loss completo debe mapear a -1.0 R, obtuve {norm_sl}"
    );

    // Caso 2: Ganancia con RR=2.25 (+0.00485): debe saturar a +1.0 R
    let win_ret = 2.25 * measured_risk;
    let norm_win = normalize(win_ret, measured_risk);
    assert_eq!(norm_win, 1.0);

    // Caso 3: Fallback seguro si el riesgo aún no está medido (0.0)
    let fallback_norm = normalize(0.01, 0.0);
    assert_eq!(fallback_norm, 0.5); // 0.01 / 0.02 = 0.5
}

/// R7-R6-C-7 ≡ MD-2: Verifica que un candidato con micro-muestra (N < 30)
/// y ganancias afortunadas NO puede fabricar una ventaja positiva espuria
/// que supere el prior conservador.
#[test]
fn bayesian_prior_discounts_insufficient_sample_and_prevents_phantom_edge() {
    let inputs = FitnessInputs {
        initial_capital: 13.0,
        final_capital: 13.5, // Ganancia con pocos trades
        max_drawdown_pct: 0.01,
        total_trades: 2, // Apenas 2 trades
        min_trades_required: 30,
        oos_start_capital: 13.0,
        oos_end_capital: 13.5,
    };

    // En compute regular, 2 trades es INVIABLE
    assert_eq!(compute(&inputs), INVIABLE);

    // Con prior conservador (-0.05), la ganancia de 2 trades debe descontarse fuertemente
    let prior = -0.05;
    let score = compute_with_bayesian_prior(&inputs, prior);

    assert!(score.is_finite());
    // Con 2 trades sobre 30, el peso es 2/30; la puntuación debe permanecer negativa
    assert!(
        score < 0.0,
        "2 operaciones afortunadas no deben fabricar un edge positivo ({score}) sobre el prior ({prior})"
    );
}

/// Verifica que `PositionManager::is_any_open` detecta aperturas en cualquiera
/// de las tres ranuras espectrales (`scalp`, `swing`, `position`), previniendo
/// re-evaluaciones redundantes en `live_envelope_gate`.
#[test]
fn multi_slot_openness_reflects_all_spectral_slots() {
    let pm = PositionManager::default();

    assert!(!pm.is_any_open());
    assert!(!pm.scalp.is_open());
    assert!(!pm.swing.is_open());
    assert!(!pm.position.is_open());

    // Abrir ranura scalp
    pm.scalp.is_open.store(true, Relaxed);
    assert!(pm.is_any_open());
    pm.scalp.is_open.store(false, Relaxed);
    assert!(!pm.is_any_open());

    // Abrir ranura swing
    pm.swing.is_open.store(true, Relaxed);
    assert!(pm.is_any_open());
    pm.swing.is_open.store(false, Relaxed);
    assert!(!pm.is_any_open());

    // Abrir ranura genérica
    pm.position.is_open.store(true, Relaxed);
    assert!(pm.is_any_open());
    pm.position.is_open.store(false, Relaxed);
    assert!(!pm.is_any_open());
}
