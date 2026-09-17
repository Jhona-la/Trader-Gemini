/// 🌐 SIMULADOR ESTOCÁSTICO DE LATENCIA Y JITTER DE RED (REALISTIC NETWORK JITTER SIMULATOR)
/// Modela la distribución de retardo RTT (Lognormal/Gamma) y slippage adverso por retraso de ejecución (#290-#295).

#[derive(Debug, Clone)]
pub struct NetworkJitterSimulator {
    pub base_latency_ms: f64,
    pub sigma_jitter: f64,
    pub packet_loss_prob: f64,
}

impl NetworkJitterSimulator {
    pub fn new(base_latency_ms: f64, sigma_jitter: f64, packet_loss_prob: f64) -> Self {
        Self {
            base_latency_ms: if base_latency_ms.is_finite() && base_latency_ms >= 1.0 {
                base_latency_ms
            } else {
                25.0 // 25ms RTT promedio a Tokyo/AWS
            },
            sigma_jitter: if sigma_jitter.is_finite() && sigma_jitter >= 0.0 {
                sigma_jitter
            } else {
                0.35 // Dispersión lognormal típica
            },
            packet_loss_prob: if packet_loss_prob.is_finite() && packet_loss_prob >= 0.0 {
                packet_loss_prob.clamp(0.0, 0.05)
            } else {
                0.001 // 0.1% pérdida de paquetes
            },
        }
    }

    /// Genera la latencia simulada en milisegundos para una solicitud usando Box-Muller para distribución Lognormal
    pub fn sample_latency_ms(&self, seed: u64) -> (f64, bool) {
        let mut rng_state = seed ^ 0x517CC1B727220A95;

        // Pseudo-random u1, u2
        rng_state = rng_state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z1 = rng_state;
        z1 = (z1 ^ (z1 >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z1 = (z1 ^ (z1 >> 27)).wrapping_mul(0x94D049BB133111EB);
        z1 = z1 ^ (z1 >> 31);
        let u1 = ((z1 as f64) / (u64::MAX as f64)).clamp(1e-6, 1.0 - 1e-6);

        rng_state = rng_state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z2 = rng_state;
        z2 = (z2 ^ (z2 >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z2 = (z2 ^ (z2 >> 27)).wrapping_mul(0x94D049BB133111EB);
        z2 = z2 ^ (z2 >> 31);
        let u2 = ((z2 as f64) / (u64::MAX as f64)).clamp(1e-6, 1.0 - 1e-6);

        // Chequeo de pérdida de paquete
        let is_dropped = u1 < self.packet_loss_prob;

        // Variable estándar normal N(0, 1) vía Box-Muller
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        // FIX #1484: Lognormal: $L = L_0 \cdot \exp(\sigma z - \sigma^2 / 2)$ con clamping de exponente
        let exponent = (self.sigma_jitter * z - 0.5 * self.sigma_jitter * self.sigma_jitter)
            .clamp(-50.0, 50.0);
        let latency = self.base_latency_ms * exponent.exp();

        (latency.clamp(2.0, 500.0), is_dropped)
    }

    /// Calcula el desplazamiento de precio adverso (slippage por latencia) dado el precio actual, la volatilidad y la latencia
    pub fn calculate_latency_slippage(
        &self,
        current_price: f64,
        volatility_per_sec: f64,
        latency_ms: f64,
        is_long: bool,
    ) -> f64 {
        if current_price <= 0.0 || !current_price.is_finite() {
            return 0.0;
        }

        if !volatility_per_sec.is_finite() || !latency_ms.is_finite() {
            return current_price;
        }

        let delta_t_sec = (latency_ms / 1000.0).max(0.001);
        let raw_drift = volatility_per_sec * delta_t_sec.sqrt();
        let drift_pct = if raw_drift.is_finite() {
            raw_drift.clamp(0.0, 0.02)
        } else {
            0.0
        };

        if is_long {
            // Comprador sufre precio más alto
            current_price * (1.0 + drift_pct)
        } else {
            // Vendedor sufre precio más bajo
            current_price * (1.0 - drift_pct)
        }
    }
}

impl Default for NetworkJitterSimulator {
    fn default() -> Self {
        Self::new(25.0, 0.35, 0.001)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_jitter_sampling_and_slippage() {
        let sim = NetworkJitterSimulator::new(20.0, 0.3, 0.0);

        let mut latencies = Vec::new();
        for i in 0..100 {
            let (lat, dropped) = sim.sample_latency_ms(1000 + i);
            assert!(!dropped);
            assert!(lat >= 2.0 && lat <= 500.0);
            latencies.push(lat);
        }

        // Media de latencias debe estar cerca de 20ms
        let avg_lat: f64 = latencies.iter().sum::<f64>() / (latencies.len() as f64);
        assert!((avg_lat - 20.0).abs() < 10.0);

        // Verificar slippage adverso
        let price = 50000.0;
        let vol_sec = 0.001; // 0.1% por segundo
        let exec_long = sim.calculate_latency_slippage(price, vol_sec, 25.0, true);
        let exec_short = sim.calculate_latency_slippage(price, vol_sec, 25.0, false);

        assert!(exec_long >= price);
        assert!(exec_short <= price);
    }

    #[test]
    fn test_network_jitter_nan_immunity() {
        let sim = NetworkJitterSimulator::new(f64::NAN, f64::INFINITY, -1.0);
        assert_eq!(sim.base_latency_ms, 25.0);
        assert_eq!(sim.sigma_jitter, 0.35);

        let (lat, _) = sim.sample_latency_ms(42);
        assert!(lat.is_finite() && lat >= 2.0);

        let slip_nan = sim.calculate_latency_slippage(f64::NAN, 0.001, 25.0, true);
        assert_eq!(slip_nan, 0.0);

        let slip_normal = sim.calculate_latency_slippage(100.0, f64::NAN, 25.0, true);
        assert_eq!(slip_normal, 100.0);
    }
}
