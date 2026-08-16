/// 🧠 ALGORITMO #96: MOTOR DE COMPUERTA PERCEPTRÓN HEBBIANA ADAPTATIVA (PERCEPTRON GATE ENGINE)
/// Perceptrón binario ultra-rápido de ciclo único de CPU con regla de actualización Hebbiana adaptativa,
/// regulando el paso atómico de señales de compra/venta en función de los PnL recientes.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct PerceptronGateEngine;

impl PerceptronGateEngine {
    /// Inferencia del tensor sin actualizar el peso (para el Engine Core)
    #[inline(always)]
    pub fn infer(signal_score: f64, weight: f64) -> f64 {
        let activation = signal_score * weight;
        ((activation - 0.5) * 5.0).tanh().max(0.0).min(1.0)
    }

    /// Aprendizaje Hebbiano Adaptativo V2 (Fase 21)
    /// Incorpora la varianza/volatilidad para escalar la tasa de aprendizaje.
    #[inline(always)]
    pub fn update_weight(weight: &mut f64, recent_pnl: f64, std_dev: f64) {
        // Tasa de aprendizaje base (0.05) modulada inversamente por el riesgo (std_dev).
        // A mayor volatilidad/incertidumbre (alto std_dev), menor es el paso de mutación.
        let learning_rate = 0.05 / (1.0 + std_dev * 100.0);

        if recent_pnl > 0.0 {
            *weight = (*weight + learning_rate).min(2.0);
        } else if recent_pnl < 0.0 {
            *weight = (*weight - learning_rate).max(0.1);
        }
    }
}
