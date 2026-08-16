/// 🧬 ALGORITMO #103: REBALANCEADOR DE CAPITAL EPIGENÉTICO PARA EL TOP 10 (EPIGENETIC CAPITAL ALLOCATOR ENGINE)
/// Modula las máscaras de metilación epigenéticas para canalizar dinámicamente el margen ($50.00capital base)
/// exclusivamente hacia las 10 mejores oportunidades con mayor payoff esperado.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct EpigeneticCapitalAllocEngine;

impl EpigeneticCapitalAllocEngine {
    /// Asigna las fracciones óptimas de capital para los Top 10 activos bajo expresión epigenética en O(1)
    #[inline(always)]
    pub fn allocate_epigenetic_top10_margin(top10_scores: &[f64; 10], methylation_tensor: &[f64; 10]) -> [f64; 10] {
        let mut weights = [0.0f64; 10];
        let mut total_w = 0.0;
        for i in 0..10 {
            // Elimina if/else: Utiliza la expresión epigenética (0.0 a 1.0) como multiplicador proporcional
            let epigenetic_multiplier = 0.5 + (methylation_tensor[i] * 1.0);
            weights[i] = top10_scores[i].max(0.01) * epigenetic_multiplier;
            total_w += weights[i];
        }
        let inv_total = if total_w > 0.0 { 1.0 / total_w } else { 0.0 };
        for i in 0..10 {
            weights[i] *= inv_total;
        }
        weights
    }

    /// SISTEMA SUPREMO: Genera el Tensor de Plasticidad (0.5 a 1.5) para acelerar el PPO
    /// en los activos que el modelo de Staking Epigenético considera más valiosos.
    #[inline(always)]
    pub fn calculate_plasticity_tensor(methylation_tensor: &[f64; 10]) -> [f64; 10] {
        let mut plasticity = [1.0f64; 10];
        for i in 0..10 {
            // Si la metilación es alta (1.0), el PPO aprenderá a 1.5x (Mayor plasticidad).
            // Si es baja (0.0), el PPO aprenderá a 0.5x (Menor plasticidad).
            plasticity[i] = 0.5 + (methylation_tensor[i] * 1.0);
        }
        plasticity
    }
}
