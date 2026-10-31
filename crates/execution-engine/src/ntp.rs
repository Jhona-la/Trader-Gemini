//! NTP Synchronizer for live timestamp drift correction against Binance Futures.
//!
//! Implementación con Filtro de Kalman 1D y rechazo de outliers para eliminar
//! las oscilaciones y falsos picos de latencia en Windows.

use crate::client::BinanceClient;
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Filtro de Kalman 1D para estimar de forma continua y suave el offset de reloj
/// entre el host local y los servidores de Binance Futures.
///
/// Estado $x$: Offset en milisegundos (`server_time - local_time`).
/// Varianza $P$: Incertidumbre de la estimación.
/// Ruido de proceso $Q$: Deriva temporal típica del oscilador de cristal en Windows (~0.5 ms² por ciclo).
/// Ruido de medición $R$: Incertidumbre del viaje de red RTT. $R = \max((RTT/2)^2, R_{\min})$.
///
/// Rechazo de outliers: Si $RTT > 350\text{ ms}$ y ya existe un prior convergente, la muestra
/// se descarta para evitar que una congestión de red transitoria contamine el reloj de trading.
#[derive(Debug, Clone, Copy)]
pub struct NtpKalmanFilter {
    pub offset_estimate: f64,
    pub error_covariance: f64,
    pub initialized: bool,
    pub process_noise_q: f64,
    pub sample_count: u64,
}

impl Default for NtpKalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl NtpKalmanFilter {
    pub const DEFAULT_Q: f64 = 0.5; // (ms)² por ciclo de 15s
    pub const MIN_MEASUREMENT_VARIANCE: f64 = 4.0; // Mínimo 2ms de incertidumbre (2² = 4)
    pub const OUTLIER_RTT_THRESHOLD_MS: u64 = 350; // Descarte de picos de congestión

    pub fn new() -> Self {
        Self {
            offset_estimate: 0.0,
            error_covariance: 10_000.0,
            initialized: false,
            process_noise_q: Self::DEFAULT_Q,
            sample_count: 0,
        }
    }

    /// I-5: como `update` pero con el offset crudo medido contra la
    /// PROYECCIÓN MONÓTONA (no contra la pared): el `raw_offset` lo calcula
    /// el llamador con `correccion_vs_proyeccion`. La puerta de outliers por
    /// RTT (>350 ms) se conserva.
    pub fn update_contra_proyeccion(&mut self, rtt_ms: u64, raw_offset: f64) -> Option<i64> {
        if !raw_offset.is_finite() {
            return None;
        }
        if self.initialized && rtt_ms > Self::OUTLIER_RTT_THRESHOLD_MS {
            return Some(self.offset_estimate.round() as i64);
        }
        let rtt_half = (rtt_ms as f64) / 2.0;
        if !self.initialized {
            self.offset_estimate = raw_offset;
            self.error_covariance = rtt_half.max(2.0).powi(2);
            self.initialized = true;
            self.sample_count = 1;
            return Some(raw_offset.round() as i64);
        }
        let p_prior = (self.error_covariance + self.process_noise_q).clamp(1e-4, 1e6);
        let r_meas = rtt_half.powi(2).max(Self::MIN_MEASUREMENT_VARIANCE);
        let k_gain = p_prior / (p_prior + r_meas);
        self.offset_estimate += k_gain * (raw_offset - self.offset_estimate);
        self.error_covariance = ((1.0 - k_gain) * p_prior).clamp(1e-4, 1e6);
        self.sample_count += 1;
        Some(self.offset_estimate.round() as i64)
    }

    /// Actualiza el filtro con una medición de sincronización NTP.
    /// Retorna el nuevo offset filtrado en milisegundos si la medición es válida.
    pub fn update(&mut self, t0: u64, t1: u64, server_time: u64) -> Option<i64> {
        let rtt = t1.saturating_sub(t0);

        // Descarte de outliers si ya estamos inicializados
        if self.initialized && rtt > Self::OUTLIER_RTT_THRESHOLD_MS {
            return Some(self.offset_estimate.round() as i64);
        }

        let rtt_half = (rtt as f64) / 2.0;
        let local_estimate = (t0 as f64) + rtt_half;
        let raw_offset = (server_time as f64) - local_estimate;

        if !raw_offset.is_finite() {
            return None;
        }

        if !self.initialized {
            self.offset_estimate = raw_offset;
            self.error_covariance = rtt_half.max(2.0).powi(2);
            self.initialized = true;
            self.sample_count = 1;
            return Some(raw_offset.round() as i64);
        }

        // 1. Predicción: P = P + Q
        let p_prior = (self.error_covariance + self.process_noise_q).clamp(1e-4, 1e6);

        // 2. Varianza de medición proporcional al RTT: R = max((RTT/2)², R_min)
        let r_meas = rtt_half.powi(2).max(Self::MIN_MEASUREMENT_VARIANCE);

        // 3. Ganancia de Kalman: K = P / (P + R)
        let k_gain = p_prior / (p_prior + r_meas);

        // 4. Actualización del estado e incertidumbre
        let innovation = raw_offset - self.offset_estimate;
        self.offset_estimate += k_gain * innovation;
        self.error_covariance = ((1.0 - k_gain) * p_prior).clamp(1e-4, 1e6);
        self.sample_count += 1;

        Some(self.offset_estimate.round() as i64)
    }

    #[inline(always)]
    pub fn current_offset(&self) -> Option<i64> {
        if self.initialized && self.offset_estimate.is_finite() {
            Some(self.offset_estimate.round() as i64)
        } else {
            None
        }
    }
}

/// I-5 (Línea I): proyección monótona de la hora del servidor. El ancla es
/// una estimación de servidor (`srv_ms`) tomada en un instante monótono
/// (`Instant`); a partir de ahí la proyección avanza con el reloj monótono
/// del proceso y NUNCA vuelve a leer el reloj de pared: una corrección de
/// w32time a mitad de sesión es estructuralmente invisible.
pub fn proyeccion_srv(ancla_srv_ms: i64, transcurrido: Duration) -> i64 {
    ancla_srv_ms.saturating_add(i64::try_from(transcurrido.as_millis()).unwrap_or(i64::MAX))
}

/// I-5: corrección medida contra la proyección monótona (no contra la pared).
/// Es lo único que puede entrar en la edad del dato como error de reloj:
/// deriva del oscilador y jitter, jamás un salto de w32time.
pub fn correccion_vs_proyeccion(server_ms: i64, proyeccion_ms: i64) -> i64 {
    server_ms.saturating_sub(proyeccion_ms)
}

pub async fn start_ntp_synchronizer(client: Arc<BinanceClient>, arena: Arc<GlobalArena>) {
    let mut interval = tokio::time::interval(Duration::from_secs(15));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut kalman = NtpKalmanFilter::new();
    // I-5: segundo filtro para la corrección contra la proyección monótona.
    // Estado: corrección (ms) = servidor − proyección; deriva del oscilador.
    let mut kalman_corr = NtpKalmanFilter::new();
    // I-5: ancla de la proyección. Se fija UNA vez, con la primera muestra
    // aceptada (época del arranque); después el reloj de pared no vuelve a
    // leerse para la edad del dato.
    let mut ancla: Option<(i64, Instant)> = None;

    loop {
        interval.tick().await;

        let t0 = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(d) => d.as_millis() as u64,
            Err(_) => continue,
        };

        match client.fetch_server_time().await {
            Ok(server_time) => {
                let t1 = match SystemTime::now().duration_since(UNIX_EPOCH) {
                    Ok(d) => d.as_millis() as u64,
                    Err(_) => continue,
                };

                if let Some(filtered_offset) = kalman.update(t0, t1, server_time) {
                    arena
                        .server_time_offset_ms
                        .store(filtered_offset, Ordering::Relaxed);

                    // I-5: corrección contra la proyección monótona.
                    let rtt = t1.saturating_sub(t0);
                    match ancla {
                        None => {
                            // Primera muestra aceptada: fijar el ancla con la
                            // estimación de servidor de ESTE instante (pared
                            // local + offset filtrado — misma época del
                            // arranque, antes de cualquier corrección).
                            let local_mid = t0 as i64 + (rtt as i64) / 2;
                            ancla = Some((local_mid + filtered_offset, Instant::now()));
                        }
                        Some((ancla_srv, ancla_mono)) => {
                            let proyeccion = proyeccion_srv(ancla_srv, ancla_mono.elapsed());
                            let raw =
                                correccion_vs_proyeccion(server_time as i64, proyeccion) as f64;
                            if raw.is_finite() {
                                if let Some(corr) =
                                    kalman_corr.update_contra_proyeccion(rtt, raw)
                                {
                                    arena
                                        .server_time_correction_ms
                                        .store(corr, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("⚠️ [NTP-SYNC] Server time fetch failed: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntp_kalman_initialization() {
        let mut kf = NtpKalmanFilter::new();
        assert!(!kf.initialized);
        assert_eq!(kf.current_offset(), None);

        // t0 = 1000, t1 = 1040 (RTT = 40ms, rtt_half = 20ms)
        // local_estimate = 1020
        // server_time = 1050 => raw_offset = +30ms
        let offset = kf.update(1000, 1040, 1050);
        assert_eq!(offset, Some(30));
        assert!(kf.initialized);
        assert_eq!(kf.sample_count, 1);
        assert_eq!(kf.current_offset(), Some(30));
    }

    #[test]
    fn test_ntp_kalman_smooth_filtering() {
        let mut kf = NtpKalmanFilter::new();
        // Inicializar con 50ms
        kf.update(1000, 1020, 1060); // t0=1000, t1=1020, rtt_half=10, server=1060 => offset = 50ms

        // Enviar mediciones con ruido simétrico de +- 6ms
        let offsets = [54, 46, 52, 48, 55, 45, 51, 49];
        for (i, &noisy_off) in offsets.iter().enumerate() {
            let t0 = 2000 + i as u64 * 1000;
            let t1 = t0 + 20; // RTT = 20ms
            let server_time = t0 + 10 + noisy_off as u64;
            let filtered = kf.update(t0, t1, server_time).expect("Valid update");
            // La estimación debe mantenerse suave alrededor de 50ms (+- 2ms)
            assert!(
                (filtered - 50).abs() <= 3,
                "Offset filtered {} divagó más de 3ms de 50",
                filtered
            );
        }
    }

    #[test]
    fn test_ntp_kalman_outlier_rejection() {
        let mut kf = NtpKalmanFilter::new();
        // Inicializar en 100ms con RTT sano de 20ms
        kf.update(1000, 1020, 1110);

        // Inyectar un spike de congestión severa: RTT = 600ms (t1 = t0 + 600)
        // server_time reporta un salto aparente de 400ms
        let outlier_offset = kf.update(2000, 2600, 2500);

        // El filtro debió rechazar la medición y mantener el estado previo (100ms)
        assert_eq!(outlier_offset, Some(100));
        assert_eq!(kf.current_offset(), Some(100));
    }
}
