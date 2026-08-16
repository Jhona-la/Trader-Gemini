use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TrajectoryDivergenceReason {
    VolumeStarvation,            // Volumen insuficiente en el tiempo transcurrido
    MomentumReversal,            // Inversión rápida de tendencia en contra
    TimeExhaustionWithoutProgress,// El tiempo pasó sin generar avance en el precio
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TrajectoryStatus {
    Aligned { coherence_score: f64 },
    Divergent { reason: TrajectoryDivergenceReason, score: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveTrajectoryTrack {
    pub symbol_id: usize,
    pub is_long: bool,
    pub is_scalp: bool,
    pub entry_price: f64,
    pub entry_time_ms: u64,
    pub expected_magnitude: f64,   // Delta precio % esperado
    pub expected_volume_usd: f64,  // Volumencapital base esperado en la ola
    pub expected_duration_ms: u64, // Duración esperada en ms
    
    // Métricas en tiempo real de seguimiento
    pub accumulated_volume_usd: f64,
    pub max_favorable_price: f64,
    pub max_adverse_price: f64,
    pub last_update_ms: u64,
}

impl ActiveTrajectoryTrack {
    pub fn new(
        symbol_id: usize,
        is_long: bool,
        is_scalp: bool,
        entry_price: f64,
        entry_time_ms: u64,
        expected_magnitude: f64,
        expected_volume_usd: f64,
        expected_duration_ms: u64,
    ) -> Self {
        Self {
            symbol_id,
            is_long,
            is_scalp,
            entry_price,
            entry_time_ms,
            expected_magnitude,
            expected_volume_usd,
            expected_duration_ms,
            accumulated_volume_usd: 0.0,
            max_favorable_price: entry_price,
            max_adverse_price: entry_price,
            last_update_ms: entry_time_ms,
        }
    }
}

pub struct TrajectoryAuditor {
    pub scalp_tracks: Vec<Option<ActiveTrajectoryTrack>>,
    pub swing_tracks: Vec<Option<ActiveTrajectoryTrack>>,
    pub num_coins: usize,
}

impl TrajectoryAuditor {
    pub fn new(num_coins: usize) -> Self {
        let mut scalp_tracks = Vec::with_capacity(num_coins);
        let mut swing_tracks = Vec::with_capacity(num_coins);
        for _ in 0..num_coins {
            scalp_tracks.push(None);
            swing_tracks.push(None);
        }
        Self {
            scalp_tracks,
            swing_tracks,
            num_coins,
        }
    }

    /// Registra el inicio de una nueva hipótesis de trayectoria al abrir posición
    pub fn record_entry(
        &mut self,
        symbol_id: usize,
        is_long: bool,
        is_scalp: bool,
        entry_price: f64,
        entry_time_ms: u64,
        expected_magnitude: f64,
        expected_volume_usd: f64,
        expected_duration_ms: u64,
    ) {
        if symbol_id >= self.num_coins {
            return;
        }
        let track = ActiveTrajectoryTrack::new(
            symbol_id,
            is_long,
            is_scalp,
            entry_price,
            entry_time_ms,
            expected_magnitude,
            expected_volume_usd,
            expected_duration_ms,
        );
        if is_scalp {
            self.scalp_tracks[symbol_id] = Some(track);
        } else {
            self.swing_tracks[symbol_id] = Some(track);
        }
    }

    /// Evalúa en nanosegundos la coherencia tick-a-tick entre la predicción teórica y la realidad del mercado
    pub fn evaluate_tick(
        &mut self,
        symbol_id: usize,
        is_scalp: bool,
        current_price: f64,
        tick_volume_usd: f64,
        current_time_ms: u64,
    ) -> TrajectoryStatus {
        if symbol_id >= self.num_coins {
            return TrajectoryStatus::Aligned { coherence_score: 1.0 };
        }

        let track_slot = if is_scalp {
            &mut self.scalp_tracks[symbol_id]
        } else {
            &mut self.swing_tracks[symbol_id]
        };

        let track = match track_slot {
            Some(t) => t,
            None => return TrajectoryStatus::Aligned { coherence_score: 1.0 },
        };

        // Actualizar métricas acumuladas
        track.accumulated_volume_usd += tick_volume_usd;
        track.last_update_ms = current_time_ms;

        if track.is_long {
            if current_price > track.max_favorable_price {
                track.max_favorable_price = current_price;
            }
            if current_price < track.max_adverse_price {
                track.max_adverse_price = current_price;
            }
        } else {
            if current_price < track.max_favorable_price {
                track.max_favorable_price = current_price;
            }
            if current_price > track.max_adverse_price {
                track.max_adverse_price = current_price;
            }
        }

        let elapsed_ms = current_time_ms.saturating_sub(track.entry_time_ms);
        let duration_ratio = (elapsed_ms as f64 / track.expected_duration_ms.max(1) as f64).clamp(0.0, 3.0);

        // 1. Rendimiento del Precio (% de avance hacia la magnitud esperada)
        let price_delta_pct = if track.is_long {
            (current_price - track.entry_price) / track.entry_price
        } else {
            (track.entry_price - current_price) / track.entry_price
        };

        // Check 1: Inversión brusca de momentum (Momentum Reversal)
        // Permite respiración al precio (magnitud * 1.50 o max -0.60%)
        let adverse_threshold = (-0.0060f64).max(-track.expected_magnitude * 1.50);
        if price_delta_pct < adverse_threshold {
            return TrajectoryStatus::Divergent {
                reason: TrajectoryDivergenceReason::MomentumReversal,
                score: price_delta_pct,
            };
        }

        // Check 2: Inanición de Volumen (Volume Starvation)
        // Si ha transcurrido más del 70% del tiempo esperado pero el volumen acumulado es < 1% del esperado
        if duration_ratio > 0.70 && track.expected_volume_usd > 5_000.0 {
            let volume_ratio = track.accumulated_volume_usd / track.expected_volume_usd;
            if volume_ratio < 0.01 && price_delta_pct < 0.0 {
                return TrajectoryStatus::Divergent {
                    reason: TrajectoryDivergenceReason::VolumeStarvation,
                    score: volume_ratio,
                };
            }
        }

        // Check 3: Agotamiento de Tiempo sin avance (Time Exhaustion Without Progress)
        // Si transcurrió > 400% del tiempo esperado Y el PnL es inferior a -0.30%
        if duration_ratio > 4.0 && price_delta_pct < -0.0030 {
            return TrajectoryStatus::Divergent {
                reason: TrajectoryDivergenceReason::TimeExhaustionWithoutProgress,
                score: duration_ratio,
            };
        }

        // Calcular Coherence Score general [0.0, 1.0]
        let mag_score = (price_delta_pct / track.expected_magnitude.max(0.0001)).clamp(-1.0, 1.0);
        let vol_score = (track.accumulated_volume_usd / track.expected_volume_usd.max(1.0)).clamp(0.0, 1.0);
        let coherence = (0.50 * mag_score + 0.30 * vol_score + 0.20 * (1.0 - duration_ratio.min(1.0))).clamp(0.0, 1.0);

        TrajectoryStatus::Aligned { coherence_score: coherence }
    }

    /// Limpia la traza al cerrar posición y devuelve el TCE (Trajectory Calibration Error)
    pub fn record_exit(&mut self, symbol_id: usize, is_scalp: bool, exit_price: f64, exit_time_ms: u64) -> Option<f64> {
        if symbol_id >= self.num_coins {
            return None;
        }
        let track_slot = if is_scalp {
            &mut self.scalp_tracks[symbol_id]
        } else {
            &mut self.swing_tracks[symbol_id]
        };

        let track = track_slot.take()?;
        let actual_mag = if track.is_long {
            (exit_price - track.entry_price) / track.entry_price
        } else {
            (track.entry_price - exit_price) / track.entry_price
        };

        // Trajectory Calibration Error: diferencia normalizada entre magnitud esperada y real
        let mag_error = (track.expected_magnitude - actual_mag).abs();
        let actual_duration = exit_time_ms.saturating_sub(track.entry_time_ms) as f64;
        let duration_error = (track.expected_duration_ms as f64 - actual_duration).abs() / track.expected_duration_ms.max(1) as f64;

        let tce = (0.70 * mag_error + 0.30 * duration_error).clamp(0.0, 1.0);
        Some(tce)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trajectory_aligned_and_exit() {
        let mut auditor = TrajectoryAuditor::new(5);
        let now_ms = 1_000_000;
        
        auditor.record_entry(0, true, true, 100.0, now_ms, 0.01, 50_000.0, 30_000);
        
        let status = auditor.evaluate_tick(0, true, 100.5, 20_000.0, now_ms + 5_000);
        match status {
            TrajectoryStatus::Aligned { coherence_score } => {
                assert!(coherence_score > 0.0);
            }
            _ => panic!("Expected trajectory aligned"),
        }

        let tce = auditor.record_exit(0, true, 101.0, now_ms + 25_000);
        assert!(tce.is_some());
    }

    #[test]
    fn test_trajectory_momentum_reversal_early_cut() {
        let mut auditor = TrajectoryAuditor::new(5);
        let now_ms = 1_000_000;
        
        auditor.record_entry(1, true, true, 100.0, now_ms, 0.01, 50_000.0, 30_000);
        
        // Caída brusca del precio del -0.7% (< -0.6% threshold)
        let status = auditor.evaluate_tick(1, true, 99.3, 5_000.0, now_ms + 2_000);
        match status {
            TrajectoryStatus::Divergent { reason, .. } => {
                assert_eq!(reason, TrajectoryDivergenceReason::MomentumReversal);
            }
            _ => panic!("Expected momentum reversal divergence"),
        }
    }

    #[test]
    fn test_trajectory_volume_starvation_early_cut() {
        let mut auditor = TrajectoryAuditor::new(5);
        let now_ms = 1_000_000;
        
        auditor.record_entry(2, true, true, 100.0, now_ms, 0.01, 100_000.0, 30_000);
        
        // Ha pasado el 80% del tiempo (24,000ms) pero solo ha habido $500 de volumen (< 1% de $100k) y el precio cayó (-0.01%)
        let status = auditor.evaluate_tick(2, true, 99.99, 500.0, now_ms + 24_000);
        match status {
            TrajectoryStatus::Divergent { reason, .. } => {
                assert_eq!(reason, TrajectoryDivergenceReason::VolumeStarvation);
            }
            _ => panic!("Expected volume starvation divergence"),
        }
    }
}
