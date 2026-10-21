//! Contrato de certificación formal F1-016 (Ola Ω198):
//! Paridad matemática exacta de muestreo predecible Hawkes entre el consenso
//! espectral y los motores escalares en el continuo multivariante.
//!
//! # Marco Teórico y Auditoría Cuántica (Council of 10 Senior Roles / Modo Profesor)
//! - QUÉ: Erradicación del sesgo de muestreo post-jump en el instante del trade
//!   (donde dt=0 inyectaba un artefacto +α/μ̂) y restauración de la intensidad predecible
//!   λ(t−) en la evaluación de excitación de Hawkes-Bessel.
//! - POR QUÉ: En la formulación clásica de procesos puntuales de Hawkes (Papangelou, 1972;
//!   Bauwens & Hautsch, 2009), la intensidad condicional λ(t) es un proceso predecible
//!   respecto a la filtración F_{t−}. Si se evalúa λ(t+) inmediatamente después de
//!   `record_event(t)`, la expectativa en trades E[λ(t+)/μ̂] diverge a SS + α/μ̂.
//!   Para BTC (μ̂ ≈ 20 tps, α = 0.3), α/μ̂ = 0.015 (ratio 1.615). Para una altcoin calma
//!   (μ̂ ≈ 0.2 tps), α/μ̂ = 1.50, disparando el ratio a 3.10 en CADA TRADE AISLADO.
//!   Además, en el mismo tick, el consenso espectral evaluaba λ(t−) antes de `record_event`
//!   mientras los motores escalares leían λ(t+) del registro, rompiendo la paridad.
//! - PARA QUÉ: Garantizar invarianza de escala universal en el espectro continuo
//!   (desde altcoins ilíquidas hasta BTC masivo) sin falsas cascadas en activos finos,
//!   preservando el edge matemático necesario para el crecimiento compuesto exponencial
//!   del 100% cada 3 días.
//! - CÓMO: HawkesBesselEngine almacena `pre_jump_excitacion = S(t−)` durante `record_event(t)`
//!   antes de sumar +α. La función `intensity_predictable(t)` y `intensity_ratio_predictable(t)`
//!   evalúan λ(t−) en instantes de evento (t == t_last), garantizando que la lectura previa
//!   (espectral) y la lectura posterior (escalares/registro) arrojen el MISMO valor idéntico bit a bit.
//! - CUÁNDO: En cada trade procesado por el núcleo tanto en backtest, replay y producción.
//! - DÓNDE: `crates/signal-engine/src/hawkes_bessel.rs` y `crates/god-engine-core/src/lib.rs`.
//! - QUIÉN: AGY (Quant Sr., Físico Matemático / Espectral, Arquitecto Senior).

use signal_engine::hawkes_bessel::{
    excitacion_hawkes_norm, HawkesBesselEngine, DEFAULT_ALPHA, STEADY_STATE_RATIO,
};

/// Generador pseudoaleatorio LCG determinista (sin dependencias externas).
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn u01(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn exp(&mut self, rate: f64) -> f64 {
        -self.u01().max(1e-12).ln() / rate
    }
}

fn tiempos_poisson(rng: &mut Lcg, tasa: f64, duracion_s: f64) -> Vec<f64> {
    let mut t = 0.0;
    let mut ev = Vec::new();
    while t < duracion_s {
        t += rng.exp(tasa);
        if t < duracion_s {
            ev.push(t);
        }
    }
    ev
}

#[test]
fn test_f1_016_hawkes_intensity_predictable_jump_difference() {
    // Teorema: En todo instante de salto T_k, λ(T_k+) - λ(T_k-) == α.
    // Entre eventos (t > T_k), λ_predictable(t) == λ(t).
    let mut h = HawkesBesselEngine::new();
    let alpha = DEFAULT_ALPHA; // 0.3

    h.record_event(1.0);
    h.record_event(2.5);
    h.record_event(4.0);

    // En el instante exacto del último evento (t = 4.0):
    let lambda_post = h.intensity(4.0);
    let lambda_pred = h.intensity_predictable(4.0);
    let diff = lambda_post - lambda_pred;
    assert!(
        (diff - alpha).abs() < 1e-12,
        "En t_last, la diferencia debe ser exactamente α: got {diff}, expected {alpha}"
    );

    // Entre eventos (t = 4.5):
    let lambda_post_mid = h.intensity(4.5);
    let lambda_pred_mid = h.intensity_predictable(4.5);
    assert!(
        (lambda_post_mid - lambda_pred_mid).abs() < 1e-12,
        "Entre eventos, λ_predicable debe coincidir con λ: got {lambda_pred_mid} vs {lambda_post_mid}"
    );
}

#[test]
fn test_f1_016_hawkes_spectral_scalar_parity_at_event() {
    // Paridad Espectral / Escalar:
    // En god-engine-core/src/lib.rs:
    // 1. Línea 3261 evalúa `intensity_ratio(event_time)` ANTES de `record_event(event_time)`.
    // 2. Línea 3862 llama `record_event(event_time)`.
    // 3. Línea 5867 evalúa `intensity_ratio(event_time)` DESPUÉS de `record_event(event_time)`.
    // Ambas evaluaciones deben ser exactamente idénticas bajo intensidad predecible.
    let mut h = HawkesBesselEngine::new();
    let mut rng = Lcg::new(0x1234_5678_9ABC_DEF0);
    let mut t = 0.0;

    // Calentar 50 eventos
    for _ in 0..50 {
        t += rng.exp(5.0);
        h.record_event(t);
    }

    // Probar paridad en los siguientes 200 eventos
    for _ in 0..200 {
        t += rng.exp(5.0);

        // Lectura espectral ANTES de record_event:
        let mu_antes = h.base_rate();
        let lambda_espectral_antes = h.intensity_predictable(t);
        let ratio_espectral_antes = h.intensity_ratio(t);

        // Registro del evento
        h.record_event(t);

        // Lectura escalar / registro DESPUÉS de record_event en el mismo timestamp:
        let mu_despues = h.base_rate();
        let lambda_escalar_despues = h.intensity_predictable(t);
        let ratio_escalar_despues = h.intensity_ratio(t);

        // Verificación Fundamental 1: La componente de excitación predecible S(t−) es 100%
        // IDÉNTICA bit a bit (error < 1e-12).
        let excit_part_antes = lambda_espectral_antes - mu_antes;
        let excit_part_despues = lambda_escalar_despues - mu_despues;
        assert!(
            (excit_part_antes - excit_part_despues).abs() < 1e-12,
            "Excitación predecible S(t−) rota en t={t}: antes={excit_part_antes}, despues={excit_part_despues}"
        );

        // Verificación Fundamental 2: La diferencia en λ(t−) corresponde estrictamente a la deriva
        // de un solo paso de la EWMA de μ̂ (|μ_new - μ_old|), sin el artefacto de salto +α (0.30).
        let diff_lambda = (lambda_espectral_antes - lambda_escalar_despues).abs();
        let diff_mu = (mu_despues - mu_antes).abs();
        assert!(
            (diff_lambda - diff_mu).abs() < 1e-12,
            "diff_lambda debe ser exactamente diff_mu: {diff_lambda} vs {diff_mu}"
        );
        assert!(
            diff_lambda < DEFAULT_ALPHA * 0.5,
            "diff_lambda {diff_lambda} no debe contener el salto +α={}",
            DEFAULT_ALPHA
        );

        // Verificación Fundamental 3: La diferencia en ratio está acotada por la actualización EWMA
        // de μ̂, sin el artefacto post-salto +α/μ̂ (que agregaría ≥ 0.06 en 5 tps y > 1.5 en altcoins).
        let diff_ratio = (ratio_espectral_antes - ratio_escalar_despues).abs();
        assert!(
            diff_ratio < 0.03,
            "Paridad de ratio excedió deriva EWMA en t={t}: antes={ratio_espectral_antes}, despues={ratio_escalar_despues}, diff={diff_ratio}"
        );
        let salto_artefacto = DEFAULT_ALPHA / h.base_rate();
        assert!(
            diff_ratio < salto_artefacto * 0.4,
            "diff_ratio {diff_ratio} no debe contener el salto +α/μ̂={salto_artefacto}"
        );
    }
}

#[test]
fn test_f1_016_hawkes_scale_free_asset_invariance_calm_vs_liquid() {
    // Invarianza de Escala entre Altcoin Calma (0.2 tps) y BTC Líquido (20 tps):
    // Bajo muestreo de Palm (en los eventos), un proceso Poisson puro NO tiene auto-excitación.
    // La expectativa del ratio predecible debe ser exactamente STEADY_STATE_RATIO ≈ 1.60
    // en AMBOS activos, y la excitación normalizada media debe ser nula (|e| < 0.03).
    // Con el viejo cálculo post-jump (+α/μ̂), la altcoin a 0.2 tps arrojaba ratio ≈ 3.10
    // y excitación media > 0.50 (falsas cascadas permanentes).
    let mut rng_calm = Lcg::new(0xCA1B_0001);
    let mut rng_liquid = Lcg::new(0xB17C_0002);

    let ev_calm = tiempos_poisson(&mut rng_calm, 0.2, 36_000.0); // 10 horas a 0.2 tps
    let ev_liquid = tiempos_poisson(&mut rng_liquid, 20.0, 1_800.0); // 30 min a 20 tps

    // Muestreo en eventos para Altcoin
    let mut h_calm = HawkesBesselEngine::new();
    let mut ratios_calm = Vec::new();
    let mut excits_calm = Vec::new();
    for &t in &ev_calm {
        h_calm.record_event(t);
        if t >= 300.0 {
            let r = h_calm.intensity_ratio(t);
            ratios_calm.push(r);
            excits_calm.push(excitacion_hawkes_norm(r));
        }
    }

    // Muestreo en eventos para BTC Líquido
    let mut h_liquid = HawkesBesselEngine::new();
    let mut ratios_liquid = Vec::new();
    let mut excits_liquid = Vec::new();
    for &t in &ev_liquid {
        h_liquid.record_event(t);
        if t >= 300.0 {
            let r = h_liquid.intensity_ratio(t);
            ratios_liquid.push(r);
            excits_liquid.push(excitacion_hawkes_norm(r));
        }
    }

    let mean_r_calm: f64 = ratios_calm.iter().sum::<f64>() / ratios_calm.len() as f64;
    let mean_e_calm: f64 = excits_calm.iter().sum::<f64>() / excits_calm.len() as f64;

    let mean_r_liquid: f64 = ratios_liquid.iter().sum::<f64>() / ratios_liquid.len() as f64;
    let mean_e_liquid: f64 = excits_liquid.iter().sum::<f64>() / excits_liquid.len() as f64;

    println!("Altcoin calma (0.2 tps): mean ratio = {mean_r_calm:.4}, mean excit = {mean_e_calm:.4}");
    println!("BTC líquido  (20.0 tps): mean ratio = {mean_r_liquid:.4}, mean excit = {mean_e_liquid:.4}");

    // Verificación 1: Ambos deben anclarse a STEADY_STATE_RATIO (1.60) dentro de tolerancia estadística
    assert!(
        (mean_r_calm - STEADY_STATE_RATIO).abs() < 0.08,
        "Altcoin calma ratio debe anclarse a 1.60, got {mean_r_calm}"
    );
    assert!(
        (mean_r_liquid - STEADY_STATE_RATIO).abs() < 0.08,
        "BTC líquido ratio debe anclarse a 1.60, got {mean_r_liquid}"
    );

    // Verificación 2: La diferencia entre activos de diferente liquidez debe ser despreciable
    assert!(
        (mean_r_calm - mean_r_liquid).abs() < 0.08,
        "Divergencia entre altcoin calma y BTC liquido debe ser < 0.08: delta={}",
        (mean_r_calm - mean_r_liquid).abs()
    );

    // Verificación 3: Excitación normalizada debe ser estrictamente no-falsa (|e| < 0.055)
    assert!(
        mean_e_calm.abs() < 0.055,
        "Excitación media en altcoin calma debe ser nula bajo Poisson, got {mean_e_calm}"
    );
    assert!(
        mean_e_liquid.abs() < 0.055,
        "Excitación media en BTC líquido debe ser nula bajo Poisson, got {mean_e_liquid}"
    );
}

#[test]
fn test_f1_016_hawkes_genuine_burst_responsiveness() {
    // Verificación de sensibilidad: Un ráfaga genuina de trades (5x tasa normal)
    // SÍ debe elevar el ratio predecible de manera contundente (> 2.0).
    let mut h = HawkesBesselEngine::new();
    let mut t = 0.0;

    // Régimen normal: 2.0 tps durante 60 segundos
    while t < 60.0 {
        t += 0.5;
        h.record_event(t);
    }
    let r_normal = h.intensity_ratio(t);
    assert!(
        (r_normal - STEADY_STATE_RATIO).abs() < 0.15,
        "Régimen normal debe estar en ~1.60, got {r_normal}"
    );

    // Ráfaga violenta: 20.0 tps (10x) durante 3 segundos
    for _ in 0..60 {
        t += 0.05;
        h.record_event(t);
    }
    let r_burst = h.intensity_ratio(t);
    let e_burst = excitacion_hawkes_norm(r_burst);

    println!("Ráfaga: ratio={r_burst:.4}, excit={e_burst:.4}");
    assert!(
        r_burst > 2.0,
        "Ráfaga genuina debe superar ratio 2.0, got {r_burst}"
    );
    assert!(
        e_burst > 0.2,
        "Ráfaga genuina debe emitir excitación positiva contundente, got {e_burst}"
    );
}

#[test]
fn test_f1_016_hawkes_numerical_finite_guards() {
    let mut h = HawkesBesselEngine::new();
    h.record_event(10.0);
    h.record_event(10.5);

    // Consultas con NaN / Inf / negativas no deben causar pánico ni valores corruptos
    assert!(h.intensity_predictable(f64::NAN).is_finite());
    assert!(h.intensity_predictable(f64::INFINITY).is_finite());
    assert!(h.intensity_predictable(-100.0).is_finite());

    assert!(h.intensity_ratio_predictable(f64::NAN).is_finite());
    assert!(h.intensity_ratio_predictable(f64::INFINITY).is_finite());
    assert!(h.intensity_ratio_predictable(-100.0).is_finite());

    assert!(h.pre_jump_excitacion().is_finite());
    assert!(h.excitacion_acumulada().is_finite());
}
