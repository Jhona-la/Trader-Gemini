//! F1-079 (Ola Ω183): Contrato de cuantiles adaptativos con olvido temporal en ventana acotada (RingQuantile).
//!
//! Verifica:
//! 1. Olvido estricto de regímenes pasados: tras N=256 muestras del nuevo régimen, la ventana no retiene
//!    residuos del régimen anterior (solución al congelamiento por edad de sesión C-F2.2 / F1-079).
//! 2. Compuerta de madurez estadística (MUESTRAS_MADURAS_CUANTIL = 30): bajo 30 muestras se respeta el fallback
//!    frío derivado de la curva del genoma / arena.config sin inventar umbrales vivos prematuros.
//! 3. Adaptabilidad en micro-volatilidad: erradicación de pisos arbitrarios hardcodeados (0,02 / 0,10),
//!    permitiendo que activos de bajo OBI/OFI operen con percentiles empíricos fieles.
//! 4. Inmunidad a NaN, +Inf, -Inf.
//! 5. Cumplimiento de Send + Sync (thread-safety estática).

use quantum_arena::adaptive_quantiles::{
    AdaptiveQuantileEngine, RingQuantile, CAPACIDAD_VENTANA_CUANTIL, MUESTRAS_MADURAS_CUANTIL,
};

#[derive(Clone, Copy)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    // Box-Muller para generar normales
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn f1_079_thread_safety_send_sync() {
    assert_send_sync::<RingQuantile<256>>();
    assert_send_sync::<AdaptiveQuantileEngine>();
}

#[test]
fn f1_079_regime_shift_complete_forgetting_within_256_samples() {
    let mut rng = SimpleRng::new(12345);
    let mut engine = AdaptiveQuantileEngine::new();

    // 1. Régimen de baja volatilidad: sigma = 1.0 (|N(0, 1)|, p80 analítico ≈ 1.2816)
    for _ in 0..1000 {
        let ofi = rng.normal();
        let obi = rng.normal();
        let score = rng.normal();
        engine.update(ofi, obi, score, 0.01);
    }

    assert!(engine.obi_p80.is_initialized());
    assert_eq!(engine.obi_p80.window_len(), CAPACIDAD_VENTANA_CUANTIL);
    let val_regimen_1 = engine.dynamic_obi_threshold();
    assert!(
        (val_regimen_1 - 1.28).abs() < 0.25,
        "P80 régimen 1 debe estar cerca de 1.28: {}",
        val_regimen_1
    );

    // 2. Transición violenta de régimen: sigma se multiplica por 5.0 (|N(0, 25)|, p80 analítico ≈ 6.408)
    for _ in 0..CAPACIDAD_VENTANA_CUANTIL {
        let ofi = rng.normal() * 5.0;
        let obi = rng.normal() * 5.0;
        let score = rng.normal() * 5.0;
        engine.update(ofi, obi, score, 0.05);
    }

    // Tras exactamente 256 muestras del nuevo régimen, la ventana contiene 100% datos nuevos
    let val_regimen_2 = engine.dynamic_obi_threshold();
    assert!(
        (val_regimen_2 - 6.40).abs() < 1.20,
        "P80 régimen 2 debe adaptarse velozmente cerca de 6.40 tras 256 pasos: {}",
        val_regimen_2
    );

    // 3. Retorno a régimen tranquilo: sigma = 0.5 (|N(0, 0.25)|, p80 analítico ≈ 0.64)
    for _ in 0..CAPACIDAD_VENTANA_CUANTIL {
        let ofi = rng.normal() * 0.5;
        let obi = rng.normal() * 0.5;
        let score = rng.normal() * 0.5;
        engine.update(ofi, obi, score, 0.005);
    }

    let val_regimen_3 = engine.dynamic_obi_threshold();
    assert!(
        (val_regimen_3 - 0.64).abs() < 0.15,
        "P80 régimen 3 debe descender y olvidar el régimen violento en 256 pasos: {}",
        val_regimen_3
    );
}

#[test]
fn f1_079_maturity_gate_and_genome_fallback() {
    let mut engine = AdaptiveQuantileEngine::with_fallbacks(0.24, 0.36, 0.48);

    // Antes de 30 muestras: compuerta cerrada, responde fallback
    for i in 0..(MUESTRAS_MADURAS_CUANTIL - 1) {
        assert!(
            !engine.obi_p80.is_initialized(),
            "Muestra {i} no debe estar madura"
        );
        assert_eq!(engine.dynamic_ofi_threshold(), 0.24);
        assert_eq!(engine.dynamic_obi_threshold(), 0.36);
        assert_eq!(engine.dynamic_holistic_threshold(), 0.48);

        // Fallback dinámico contextual de la curva del genoma obi_threshold_at_tau(tau)
        assert_eq!(engine.dynamic_obi_threshold_con_fallback(0.65), 0.65);
        assert_eq!(engine.dynamic_ofi_threshold_con_fallback(0.18), 0.18);
        assert_eq!(engine.dynamic_holistic_threshold_con_fallback(0.55), 0.55);

        engine.update(1.0, 1.0, 1.0, 0.01);
    }

    // Muestra 30 (índice 29 procesado, entra muestra 30)
    engine.update(1.0, 1.0, 1.0, 0.01);
    assert!(
        engine.obi_p80.is_initialized(),
        "30 muestras deben activar madurez"
    );

    // Ahora dynamic_*_threshold_con_fallback devuelve el percentil empírico medido
    assert_eq!(engine.dynamic_obi_threshold_con_fallback(0.99), 1.0);
    assert_eq!(engine.dynamic_ofi_threshold_con_fallback(0.99), 1.0);
    assert_eq!(engine.dynamic_holistic_threshold_con_fallback(0.99), 1.0);
}

#[test]
fn f1_079_nan_inf_immunity() {
    let mut engine = AdaptiveQuantileEngine::new();
    for _ in 0..50 {
        engine.update(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN);
    }
    assert_eq!(engine.obi_p80.count(), 0);
    assert_eq!(engine.obi_p80.window_len(), 0);
    assert!(!engine.obi_p80.is_initialized());
    assert_eq!(engine.dynamic_obi_threshold(), 0.15);

    // Mezcla de datos válidos con veneno
    for i in 1..=40 {
        engine.update(i as f64 * 0.05, i as f64 * 0.05, i as f64 * 0.05, 0.01);
        engine.update(f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    assert_eq!(engine.obi_p80.count(), 40);
    assert_eq!(engine.obi_p80.window_len(), 40);
    assert!(engine.obi_p80.is_initialized());
    assert!(engine.dynamic_obi_threshold().is_finite());
}

#[test]
fn f1_079_zero_hardcoded_floor_in_micro_volatility() {
    let mut engine = AdaptiveQuantileEngine::new();

    // Activo o régimen con muy bajo OBI/OFI (e.g. 0.005)
    for _ in 0..100 {
        engine.update(0.004, 0.005, 0.006, 0.001);
    }

    // Antes el código forzaba .max(0.02) y .max(0.10) impidiendo adaptación en libros finos
    let medido_obi = engine.dynamic_obi_threshold();
    let medido_score = engine.dynamic_holistic_threshold();

    assert!(
        medido_obi < 0.015,
        "Piso empírico medido de OBI no debe estar inflado artificialmente a 0.02: {medido_obi}"
    );
    assert!(
        medido_score < 0.05,
        "Piso empírico medido de score no debe estar inflado artificialmente a 0.10: {medido_score}"
    );
}
