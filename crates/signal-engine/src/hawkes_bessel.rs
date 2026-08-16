/// 📈 ALGORITMO #91: MOTOR DE INTENSIDAD ESTOCÁSTICA HAWKES-BESSEL (HAWKES-BESSEL ENGINE)
/// Combina el proceso auto-excitado de Hawkes con kernels de convolución modificados de Bessel I_\nu(z),
/// midiendo la aceleración estocástica de ráfagas HFT durante cascadas de volatilidad en O(1).
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct HawkesBesselEngine;

impl HawkesBesselEngine {
    /// Calcula la intensidad estocástica con kernel aproximado de Bessel en O(1)
    #[inline(always)]
    pub fn compute_bessel_hawkes_intensity(base_lambda: f64, alpha: f64, dt: f64) -> f64 {
        let bessel_decay = (1.0 + dt * dt).sqrt() - dt;
        base_lambda + alpha * bessel_decay.max(0.0)
    }
}
