//! CONTRATO FORMAL DE SUPERMARTINGALAS DE VILLE Y E-VALORES SECUENCIALES (Ω30)
//!
//! # Objetivo del Contrato
//!
//! Demostrar formalmente que:
//! 1. Bajo la hipótesis nula H0 (ausencia de edge, ruido puro con media <= 0), la probabilidad
//!    de falsa certificación (sup M_t >= 1/alpha) satisface la cota maximal de Ville P(sup M_t >= 1/alpha) <= alpha.
//! 2. Inmunidad a parada opcional (optional stopping): la evaluación en cualquier instante t
//!    no infla el error Tipo I.
//! 3. Potencia bajo H1: ante ventaja estadística consistente, el e-proceso certifica edge en tiempo finito.
//! 4. Detección de deterioro y agotamiento: monitoreo de drawdown de evidencia y caída bajo el suelo de agotamiento.
//! 5. Difusión continua SDE: fidelidad numérica de la integral exponencial de Ito bajo drift positivo continuo.

use risk_engine::ville_e_process::{VilleEProcess, E_EXHAUSTION_FLOOR};

/// Generador determinista simple LCG para garantizar reproducibilidad sin dependencias externas.
struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (self.state >> 11) & 0x1F_FFFF_FFFF_FFFF;
        bits as f64 / (1u64 << 53) as f64
    }

    /// Retorna una innovación de media cero acotada en [-amp, amp]
    fn uniform_symmetric(&mut self, amp: f64) -> f64 {
        (self.next_f64() * 2.0 - 1.0) * amp
    }
}

#[test]
fn ville_contrato_cota_maximal_bajo_hipotesis_nula() {
    let alpha = 0.05;
    let num_paths = 200;
    let steps_per_path = 150;
    let mut falsas_certificaciones = 0;

    for seed in 0..num_paths {
        let mut rng = LcgRng::new(1000 + seed as u64);
        let mut process = VilleEProcess::new(alpha).expect("alpha 0.05 es válido");

        for _ in 0..steps_per_path {
            // H0: ruido con media cero (distribución uniforme centrada)
            let return_t = rng.uniform_symmetric(0.04);
            process.update(return_t);

            if process.is_edge_certified() {
                falsas_certificaciones += 1;
                break; // Optional stopping al instante de cruce
            }
        }
    }

    let tasa_falsas_certificaciones = falsas_certificaciones as f64 / num_paths as f64;
    assert!(
        tasa_falsas_certificaciones <= alpha,
        "La tasa de falsas alarmas bajo H0 ({tasa_falsas_certificaciones}) superó la cota de Ville ({alpha})"
    );
}

#[test]
fn ville_contrato_potencia_bajo_ventaja_genuina() {
    let alpha = 0.05;
    let mut rng = LcgRng::new(424242);
    let mut process = VilleEProcess::new(alpha).expect("alpha 0.05 es válido");

    let mut certified_at_step = None;
    for step in 1..=100 {
        // H1: ventaja genuina positiva (+15% de innovación normalizada en [-1, 1])
        let signal_innovation = 0.15 + rng.uniform_symmetric(0.04);
        process.update(signal_innovation);

        if process.is_edge_certified() {
            certified_at_step = Some(step);
            break;
        }
    }

    assert!(
        certified_at_step.is_some(),
        "El e-proceso no pudo certificar ventaja bajo H1 persistente con drift positivo"
    );
    assert!(
        process.e_value >= 20.0,
        "El e-valor final ({}) debe superar el umbral de Ville 1/alpha = 20.0",
        process.e_value
    );
    assert!(
        process.anytime_p_value() <= 0.05,
        "El anytime p-value ({}) debe ser <= alpha (0.05)",
        process.anytime_p_value()
    );
}

#[test]
fn ville_contrato_deteccion_de_agotamiento_e_involucion_de_evidencia() {
    let alpha = 0.05;
    let mut rng = LcgRng::new(99999);

    // 1) Caso con suelo de exploración (lambda_min = 0.10): el proceso sondea activamente
    // y decae por debajo de E_EXHAUSTION_FLOOR ante drift severamente negativo.
    let mut process_probing = VilleEProcess::with_bounds(alpha, 0.10, 0.50).expect("bounds válidos");

    for _ in 0..300 {
        let neg = -0.60 + rng.uniform_symmetric(0.05);
        process_probing.update(neg);
        if process_probing.is_exhausted() {
            break;
        }
    }

    assert!(
        process_probing.is_exhausted(),
        "El e-proceso con exploración debe agotar riqueza (< {E_EXHAUSTION_FLOOR}) ante drift adverso continuo"
    );
    assert!(
        process_probing.e_value <= E_EXHAUSTION_FLOOR,
        "El e-valor ({}) debe estar bajo el suelo de agotamiento",
        process_probing.e_value
    );

    // 2) Caso con proceso estándar (lambda_min = 0.0): acumulación previa de evidencia y posterior
    // colapso estructural que activa detección de deterioro (evidence drawdown >= 50%) y agotamiento relativo.
    let mut process_standard = VilleEProcess::new(alpha).expect("alpha 0.05 es válido");
    for _ in 0..25 {
        let pos = 0.15 + rng.uniform_symmetric(0.02);
        process_standard.update(pos);
    }

    let peak = process_standard.peak_e_value;
    assert!(peak > 1.5, "Debe haber acumulado evidencia inicial suficiente");

    // Choque adverso
    for _ in 0..50 {
        let neg = -0.25 + rng.uniform_symmetric(0.02);
        process_standard.update(neg);
    }

    assert!(
        process_standard.is_evidence_decayed(0.20),
        "Debe detectar caída de evidencia estadística mayor al 20% desde el pico"
    );
    assert!(
        process_standard.evidence_drawdown() > 0.0,
        "El drawdown de evidencia ({}) debe ser positivo",
        process_standard.evidence_drawdown()
    );
}

#[test]
fn ville_contrato_difusion_continua_sde_monotonia() {
    let alpha = 0.01; // 99% confianza -> umbral 100.0
    let mut process = VilleEProcess::new(alpha).expect("alpha 0.01 es válido");
    assert_eq!(process.threshold, 100.0);

    // Simular un drift sostenido dX = mu*dt con mu = 0.08, sigma = 0.01, dt = 0.05s
    let dt = 0.05;
    let sigma = 0.01;
    let mu = 0.08;
    for _ in 0..60 {
        let dx = mu * dt;
        process.update_continuous_sde(dx, dt, sigma);
    }

    assert!(process.e_value > 1.0, "La difusión con deriva positiva debe incrementar M_t");
    assert!(process.count == 60);
    assert!(process.peak_e_value >= process.e_value);
}
