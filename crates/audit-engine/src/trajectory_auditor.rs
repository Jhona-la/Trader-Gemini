use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TrajectoryDivergenceReason {
    VolumeStarvation,              // Volumen insuficiente en el tiempo transcurrido
    MomentumReversal,              // Inversión rápida de tendencia en contra
    TimeExhaustionWithoutProgress, // El tiempo pasó sin generar avance en el precio
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TrajectoryStatus {
    Aligned {
        coherence_score: f64,
    },
    Divergent {
        reason: TrajectoryDivergenceReason,
        score: f64,
    },
}

/// Hipótesis viva de trayectoria de UNA posición.
///
/// # U-ERR-3 (ERRADICACIÓN DEL BINARIO DE HORIZONTE)
///
/// Esta estructura llevaba un campo `is_scalp` que seleccionaba una de dos
/// tablas de tracks (`scalp_tracks` / `swing_tracks`). El único productor vivo
/// registraba SIEMPRE con `is_scalp = true`, de modo que `swing_tracks`
/// jamás contuvo nada: el auditor consultaba las dos ranuras y media
/// herramienta estaba permanentemente vacía. Peor: el consumidor evaluaba el
/// tick dos veces (una por ranura) y emitía la misma alerta etiquetada como
/// «scalp» y como «swing».
///
/// Una posición tiene UNA trayectoria. El horizonte de esa trayectoria no es
/// una etiqueta binaria: es `expected_duration_ms`, el τ esperado que el
/// genoma fijó al abrir, y contra ese τ se miden todas las divergencias.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveTrajectoryTrack {
    pub symbol_id: usize,
    pub is_long: bool,
    pub entry_price: f64,
    pub entry_time_ms: u64,
    pub expected_magnitude: f64, // Delta precio % esperado
    pub expected_volume_usd: f64, // Volumen/capital base esperado en la ola
    /// τ esperado de la posición en ms: el horizonte continuo de ESTA
    /// trayectoria. Todas las fracciones de tiempo se miden contra él.
    pub expected_duration_ms: u64,

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
        entry_price: f64,
        entry_time_ms: u64,
        expected_magnitude: f64,
        expected_volume_usd: f64,
        expected_duration_ms: u64,
    ) -> Self {
        // FIX #723: Sanitizar magnitud y volumen esperados
        let safe_magnitude = if expected_magnitude.is_finite() && expected_magnitude > 0.0 {
            expected_magnitude
        } else {
            0.01
        };
        let safe_volume = if expected_volume_usd.is_finite() && expected_volume_usd >= 0.0 {
            expected_volume_usd
        } else {
            0.0
        };
        let safe_entry = if entry_price.is_finite() && entry_price > 0.0 {
            entry_price
        } else {
            1.0
        };
        Self {
            symbol_id,
            is_long,
            entry_price: safe_entry,
            entry_time_ms,
            expected_magnitude: safe_magnitude,
            expected_volume_usd: safe_volume,
            expected_duration_ms,
            accumulated_volume_usd: 0.0,
            max_favorable_price: safe_entry,
            max_adverse_price: safe_entry,
            last_update_ms: entry_time_ms,
        }
    }
}

/// Auditor de coherencia entre la tesis de entrada y la realidad tick a tick.
///
/// U-ERR-3: UNA trayectoria por posición y por moneda. Ver
/// [`ActiveTrajectoryTrack`] para el defecto que esto corrige.
pub struct TrajectoryAuditor {
    pub tracks: Vec<Option<ActiveTrajectoryTrack>>,
    pub num_coins: usize,
}

impl TrajectoryAuditor {
    pub fn new(num_coins: usize) -> Self {
        let mut tracks = Vec::with_capacity(num_coins);
        for _ in 0..num_coins {
            tracks.push(None);
        }
        Self { tracks, num_coins }
    }

    /// Registra el inicio de una nueva hipótesis de trayectoria al abrir posición
    #[allow(clippy::too_many_arguments)]
    pub fn record_entry(
        &mut self,
        symbol_id: usize,
        is_long: bool,
        entry_price: f64,
        entry_time_ms: u64,
        expected_magnitude: f64,
        expected_volume_usd: f64,
        expected_duration_ms: u64,
    ) {
        if symbol_id >= self.num_coins {
            return;
        }
        self.tracks[symbol_id] = Some(ActiveTrajectoryTrack::new(
            symbol_id,
            is_long,
            entry_price,
            entry_time_ms,
            expected_magnitude,
            expected_volume_usd,
            expected_duration_ms,
        ));
    }

    /// Evalúa la coherencia tick-a-tick entre la predicción teórica y la realidad del mercado.
    /// Diagnóstico heurístico por evento. No hay garantía de latencia nanosegundo
    /// ni calibración probabilística de sus pesos/umbrales. No ordena un cierre.
    pub fn evaluate_tick(
        &mut self,
        symbol_id: usize,
        current_price: f64,
        tick_volume_usd: f64,
        current_time_ms: u64,
    ) -> TrajectoryStatus {
        if symbol_id >= self.num_coins {
            return TrajectoryStatus::Aligned {
                coherence_score: 1.0,
            };
        }

        // FIX #1449: Inmunidad ante NaNs o precios no positivos.
        if !current_price.is_finite() || current_price <= 0.0 {
            return TrajectoryStatus::Aligned {
                coherence_score: 1.0,
            };
        }

        let track = match &mut self.tracks[symbol_id] {
            Some(t) => t,
            None => {
                return TrajectoryStatus::Aligned {
                    coherence_score: 1.0,
                }
            }
        };

        // Actualizar métricas acumuladas
        let safe_vol = if tick_volume_usd.is_finite() && tick_volume_usd >= 0.0 {
            tick_volume_usd
        } else {
            0.0
        };
        track.accumulated_volume_usd += safe_vol;
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
        // U-ERR-4: fracción del τ esperado de ESTA posición consumida hasta
        // ahora. Antes se recortaba a 3,0 aquí y más abajo se comparaba con
        // 4,0: la comprobación de agotamiento temporal era INALCANZABLE por
        // construcción — `TimeExhaustionWithoutProgress` no podía emitirse
        // nunca. La fracción no se recorta; donde hace falta un valor acotado
        // (la puntuación de tiempo) se acota en el punto de uso.
        let duration_ratio = elapsed_ms as f64 / track.expected_duration_ms.max(1) as f64;

        if track.entry_price <= 0.0 || !track.entry_price.is_finite() {
            return TrajectoryStatus::Aligned {
                coherence_score: 1.0,
            };
        }

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
        // Si ha transcurrido más del 70% del τ esperado Y el volumen acumulado es < 1% del esperado con delta adverso
        if duration_ratio > 0.70 && track.expected_volume_usd > 1_000.0 {
            let volume_ratio = track.accumulated_volume_usd / track.expected_volume_usd;
            if volume_ratio < 0.01 && price_delta_pct < 0.0 {
                return TrajectoryStatus::Divergent {
                    reason: TrajectoryDivergenceReason::VolumeStarvation,
                    score: volume_ratio,
                };
            }
        }

        // Check 3: Agotamiento de Tiempo sin avance (Time Exhaustion Without Progress)
        // Si transcurrió > 4·τ esperado Y el PnL es inferior a -0.30%
        if duration_ratio > 4.0 && price_delta_pct < -0.0030 {
            return TrajectoryStatus::Divergent {
                reason: TrajectoryDivergenceReason::TimeExhaustionWithoutProgress,
                score: duration_ratio,
            };
        }

        // Calcular Coherence Score general [0.0, 1.0]
        // FIX #590 & #619: Normalizar mag_score en [0.0, 1.0] y neutralizar vol_score si expected_volume_usd es nulo
        let raw_mag = (price_delta_pct / track.expected_magnitude.max(0.0001)).clamp(-1.0, 1.0);
        let mag_score = 0.50 + 0.50 * raw_mag;
        let vol_score = if track.expected_volume_usd > 0.0 {
            (track.accumulated_volume_usd / track.expected_volume_usd).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let time_score = (1.0 - duration_ratio.min(1.0)).max(0.0);
        let coherence = (0.50 * mag_score + 0.30 * vol_score + 0.20 * time_score).clamp(0.0, 1.0);

        TrajectoryStatus::Aligned {
            coherence_score: coherence,
        }
    }

    /// Limpia la traza al cerrar posición y devuelve el TCE (Trajectory Calibration Error)
    pub fn record_exit(
        &mut self,
        symbol_id: usize,
        exit_price: f64,
        exit_time_ms: u64,
    ) -> Option<f64> {
        if symbol_id >= self.num_coins {
            return None;
        }

        let track = self.tracks[symbol_id].take()?;
        if !exit_price.is_finite() || exit_price <= 0.0 {
            return Some(1.0);
        }
        let actual_mag = if track.is_long {
            (exit_price - track.entry_price) / track.entry_price
        } else {
            (track.entry_price - exit_price) / track.entry_price
        };

        // Trajectory Calibration Error: diferencia normalizada entre magnitud esperada y real
        let mag_error = (track.expected_magnitude - actual_mag).abs();
        let actual_duration = exit_time_ms.saturating_sub(track.entry_time_ms) as f64;
        let duration_error = (track.expected_duration_ms as f64 - actual_duration).abs()
            / track.expected_duration_ms.max(1) as f64;

        let tce = (0.70 * mag_error + 0.30 * duration_error).clamp(0.0, 1.0);
        Some(tce)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trayectoria_alineada_y_cierre() {
        let mut auditor = TrajectoryAuditor::new(5);
        let now_ms = 1_000_000;

        auditor.record_entry(0, true, 100.0, now_ms, 0.01, 50_000.0, 30_000);

        let status = auditor.evaluate_tick(0, 100.5, 20_000.0, now_ms + 5_000);
        match status {
            TrajectoryStatus::Aligned { coherence_score } => {
                assert!(coherence_score > 0.0);
            }
            _ => panic!("Se esperaba trayectoria alineada"),
        }

        let tce = auditor.record_exit(0, 101.0, now_ms + 25_000);
        assert!(tce.is_some());
    }

    #[test]
    fn corte_temprano_por_inversion_de_momentum() {
        let mut auditor = TrajectoryAuditor::new(5);
        let now_ms = 1_000_000;

        auditor.record_entry(1, true, 100.0, now_ms, 0.01, 50_000.0, 30_000);

        // Caída brusca del precio del -0.7% (< -0.6% threshold)
        let status = auditor.evaluate_tick(1, 99.3, 5_000.0, now_ms + 2_000);
        match status {
            TrajectoryStatus::Divergent { reason, .. } => {
                assert_eq!(reason, TrajectoryDivergenceReason::MomentumReversal);
            }
            _ => panic!("Se esperaba divergencia por inversión de momentum"),
        }
    }

    #[test]
    fn corte_temprano_por_inanicion_de_volumen() {
        let mut auditor = TrajectoryAuditor::new(5);
        let now_ms = 1_000_000;

        auditor.record_entry(2, true, 100.0, now_ms, 0.01, 100_000.0, 30_000);

        // Ha pasado el 80% del τ (24.000 ms) pero sólo ha habido $500 de volumen
        // (< 1% de $100k) y el precio cayó (-0,01%).
        let status = auditor.evaluate_tick(2, 99.99, 500.0, now_ms + 24_000);
        match status {
            TrajectoryStatus::Divergent { reason, .. } => {
                assert_eq!(reason, TrajectoryDivergenceReason::VolumeStarvation);
            }
            _ => panic!("Se esperaba divergencia por inanición de volumen"),
        }
    }

    #[test]
    fn inmunidad_a_nan_y_simbolo_no_seguido() {
        let mut auditor = TrajectoryAuditor::new(5);
        let status_untracked = auditor.evaluate_tick(0, 100.0, 1000.0, 1000);
        match status_untracked {
            TrajectoryStatus::Aligned { coherence_score } => assert_eq!(coherence_score, 1.0),
            _ => panic!("Un símbolo sin traza debe quedar alineado por defecto"),
        }

        auditor.record_entry(0, true, f64::NAN, 1000, f64::NAN, f64::NAN, 10000);
        let status_nan = auditor.evaluate_tick(0, f64::NAN, f64::NAN, 2000);
        match status_nan {
            TrajectoryStatus::Aligned { coherence_score } => assert!(coherence_score >= 0.0),
            _ => {}
        }
    }

    /// U-ERR-3 — UNA TRAYECTORIA POR POSICIÓN.
    ///
    /// Falla con el código viejo: allí `record_exit` se consultaba en dos
    /// ranuras y la segunda (`swing`) devolvía siempre `None` porque nadie
    /// escribía en ella; el consumidor evaluaba además cada tick dos veces.
    /// Aquí se fija la invariante: una entrada crea UNA traza, un cierre la
    /// consume, y un segundo cierre sobre la misma posición no devuelve nada
    /// porque no hay una segunda ranura que rascar.
    #[test]
    fn u_err_3_una_entrada_produce_una_sola_traza() {
        let mut auditor = TrajectoryAuditor::new(3);
        let t0 = 1_000_000;

        auditor.record_entry(0, true, 100.0, t0, 0.01, 50_000.0, 30_000);
        assert_eq!(
            auditor.tracks.iter().filter(|t| t.is_some()).count(),
            1,
            "una entrada debe producir exactamente una traza viva"
        );

        let primero = auditor.record_exit(0, 101.0, t0 + 25_000);
        assert!(primero.is_some(), "el cierre debe devolver el TCE");

        let segundo = auditor.record_exit(0, 101.0, t0 + 25_000);
        assert!(
            segundo.is_none(),
            "no existe una segunda ranura: el cierre ya consumió la traza"
        );
    }

    /// U-ERR-4 — EL AGOTAMIENTO TEMPORAL ES ALCANZABLE.
    ///
    /// Falla con el código viejo: `duration_ratio` se recortaba a 3,0 y luego
    /// se comparaba con `> 4.0`, así que `TimeExhaustionWithoutProgress` era
    /// una rama muerta que no podía emitirse jamás. Con 5·τ transcurridos y
    /// PnL por debajo de −0,30% (pero por encima del umbral de inversión de
    /// momentum, para aislar esta comprobación) la divergencia debe emitirse.
    #[test]
    fn u_err_4_agotamiento_temporal_se_emite_pasado_cuatro_tau() {
        let mut auditor = TrajectoryAuditor::new(3);
        let t0 = 1_000_000u64;
        let tau_ms = 30_000u64;

        // expected_magnitude alto ⇒ el umbral de MomentumReversal queda en
        // −0,60% (el max(-0.0060, -mag*1.5) lo fija la rama de −0,60%),
        // de modo que un −0,40% NO dispara la inversión de momentum y deja
        // ver la comprobación de agotamiento.
        auditor.record_entry(0, true, 100.0, t0, 0.05, 0.0, tau_ms);

        // 5·τ transcurridos, precio a −0,40%.
        let status = auditor.evaluate_tick(0, 99.6, 1_000.0, t0 + 5 * tau_ms);
        match status {
            TrajectoryStatus::Divergent { reason, score } => {
                assert_eq!(reason, TrajectoryDivergenceReason::TimeExhaustionWithoutProgress);
                assert!(score > 4.0, "la fracción de τ no debe venir recortada: {score}");
            }
            other => panic!("se esperaba agotamiento temporal, llegó {other:?}"),
        }
    }
}
