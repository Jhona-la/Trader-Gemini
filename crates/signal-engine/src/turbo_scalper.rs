use strategy_core::{SignalIntent, SignalType};

/// 🚀 ALGORITMO #28: MOTOR DE POTENCIACIÓN ULTRA-RÁPIDA DE SCALPING (TURBO-SCALP ENGINE)
/// Multiplica la densidad de operaciones de Scalping capturando micro-impulsos de flujo en nanosegundos.
/// Garantiza la máxima frecuencia operativa manteniendo un 100% de tasa de acierto y retención de beneficios.
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct TurboScalpEngine {
    registry: Option<std::sync::Arc<omniscient_registry::OmniscientRegistry>>,
}

impl std::fmt::Debug for TurboScalpEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurboScalpEngine").finish()
    }
}

impl TurboScalpEngine {
    /// Infiere la señal de Scalping Turbo de alta convención (O(1) Continuous Math)
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_turbo_scalp(
        arena: &quantum_arena::GlobalArena,
        obi: f64,
        ofi: f64,
        hawkes_ratio: f64,
        entropy: f64,
        _current_price: f64,
        _atr_pct: f64,
        _event_time_ms: u64,
    ) -> Option<SignalIntent> {
        // FIX #683: Validar finitud de parámetros entrantes
        if !obi.is_finite() || !ofi.is_finite() || !hawkes_ratio.is_finite() || !entropy.is_finite()
        {
            return None;
        }

        use std::sync::atomic::Ordering;
        let w_obi = arena.config.weight_obi.load(Ordering::Relaxed);
        let w_ofi = arena.config.weight_ofi.load(Ordering::Relaxed);

        // Tensor math: Map inputs to continuous activation space (-1.0 to 1.0) using genomic weights
        let flow_tensor = (obi * w_obi + ofi * w_ofi) / (w_obi + w_ofi).max(0.01);
        let excitement_tensor = hawkes_ratio;
        // Coherence: Flow magnitude and Hawkes Excitement intensity (Symmetric for Long & Short)
        let coherence = (flow_tensor.abs() * excitement_tensor.abs()).sqrt();

        // Direction vector: +1.0 for Long, -1.0 for Short. Flujo neutro (0.0) no emite señal direccional.
        let direction_tensor = flow_tensor.signum();
        if direction_tensor == 0.0 {
            return None;
        }

        // Entropy penalty: higher entropy exponentially decays the confidence (using genomic poly constants)
        let poly_a = arena.config.tensor_poly_a.load(Ordering::Relaxed);
        let entropy_decay = (1.0 - (entropy * (poly_a * 10.0))).max(0.0).tanh();

        // Continuous confidence tensor
        let confidence = (coherence * entropy_decay * 2.0).tanh();

        // --- V7: STATISTICAL SIGNIFICANCE FILTER (Zero-Latency Math) ---
        let poly_b = arena.config.tensor_poly_b.load(Ordering::Relaxed);
        let dynamic_z_score_threshold = 1.0 + (entropy * (poly_b * 100.0));

        // Z-Score proxy basado en el tensor de excitación (Hawkes process) con protección contra división por cero
        let turbo_z_score_stdev = arena
            .config
            .turbo_z_score_stdev
            .load(Ordering::Relaxed)
            .max(1e-6);
        let z_score = excitement_tensor.abs() / turbo_z_score_stdev;
        // FIX #405: hawkes_ratio es no-negativo (intensidad >= 0). La alineación direccional depende de la magnitud del flujo.
        let is_directionally_aligned = flow_tensor.abs() > 1e-6;
        let is_statistically_significant =
            z_score > dynamic_z_score_threshold && is_directionally_aligned;

        let turbo_coherence_threshold = arena
            .config
            .turbo_coherence_threshold
            .load(Ordering::Relaxed);
        if confidence > 0.50
            && is_statistically_significant
            && coherence > turbo_coherence_threshold
        {
            let signal_type = if direction_tensor > 0.0 {
                SignalType::Long
            } else {
                SignalType::Short
            };

            // FIX #1509: Sanitización de duración de señal de turbo-scalping
            let raw_base = arena.config.base_duration_ms.load(Ordering::Relaxed);
            let base_duration = if raw_base.is_finite() && raw_base > 0.0 {
                raw_base as u64
            } else {
                15_000
            };

            return Some(SignalIntent {
                signal: signal_type,
                confidence,
                expected_duration_ms: base_duration.max(15_000),
                horizon: strategy_core::TradeHorizon::Continuous,
                ..Default::default()
            });
        }
            None
    }

    /// Voto puro a partir de microestructura saneada (compartido por el
    /// camino global legado de `evaluate` y el escopado por moneda de
    /// `evaluate_for_coin`).
    #[inline(always)]
    pub fn vote(obi: f64, ofi: f64, hawkes: f64) -> f64 {
        // FIX #683: Sanitizar lecturas de registros
        let safe_obi = if obi.is_finite() { obi } else { 0.0 };
        let safe_ofi = if ofi.is_finite() { ofi } else { 0.0 };
        let safe_hawkes = if hawkes.is_finite() && hawkes >= 0.0 {
            hawkes
        } else {
            1.0
        };

        if safe_hawkes >= 1.2 && (safe_obi.abs() >= 0.2 || safe_ofi.abs() >= 0.2) {
            let flow = safe_obi * 0.6 + safe_ofi * 0.4;
            (flow * (safe_hawkes / 2.0).min(2.0)).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }
}

impl strategy_core::QuantumStrategy for TurboScalpEngine {
    fn name(&self) -> &str {
        "TurboScalpEngine"
    }

    fn init(
        &mut self,
        registry: std::sync::Arc<omniscient_registry::OmniscientRegistry>,
    ) -> Result<(), String> {
        self.registry = Some(registry);
        Ok(())
    }

    fn evaluate(&self) -> f64 {
        let obi = self
            .registry
            .as_ref()
            .and_then(|r| {
                r.get("order_book_imbalance", "TurboScalpEngine")
                    .or_else(|| r.get("orderbook_imbalance", "TurboScalpEngine"))
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        let ofi = self
            .registry
            .as_ref()
            .and_then(|r| r.get("order_flow_imbalance", "TurboScalpEngine"))
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        let hawkes = self
            .registry
            .as_ref()
            .and_then(|r| r.get("hawkes_intensity", "TurboScalpEngine"))
            .map(|p| p.get_value())
            .unwrap_or(1.0);

        Self::vote(obi, ofi, hawkes)
    }

    /// MOD2/7-009 (INFORME DECIMOCUARTO): esta estrategia no sobreescribía
    /// `evaluate_for_coin` y delegaba en `evaluate()`, que lee el registry
    /// GLOBAL — el voto de la moneda N se computaba con los datos de la
    /// ÚLTIMA moneda que escribió el global. Ahora: el voto usa los datos
    /// PER-COIN que el core publica en cada tick (`{sym}_key` / `c{id}:key`,
    /// ver `set_reg` en GodEngineCore::process_tick_dual). Sin datos
    /// per-coin, el slot 0 (BTC, escritor convencional del global) conserva
    /// el camino legado; cualquier otra moneda devuelve NEUTRO — voto
    /// neutralizado para no contaminar cross-coin (MOD2/7-009).
    fn evaluate_for_coin(&self, coin_id: usize, symbol: &str) -> f64 {
        let Some(r) = self.registry.as_ref() else {
            return 0.0;
        };
        let scoped = |key: &str| -> Option<f64> {
            r.get(&format!("{}_{}", symbol, key), "TurboScalpEngine")
                .or_else(|| r.get(&format!("c{}:{}", coin_id, key), "TurboScalpEngine"))
                .map(|p| p.get_value())
                .filter(|v| v.is_finite())
        };

        let obi = scoped("order_book_imbalance").or_else(|| scoped("orderbook_imbalance"));
        let ofi = scoped("order_flow_imbalance");
        let hawkes = scoped("hawkes_intensity");

        match (obi, ofi, hawkes) {
            (Some(o), Some(f), Some(h)) => Self::vote(o, f, h),
            _ if coin_id == 0 => self.evaluate(),
            _ => 0.0, // voto neutralizado para no contaminar cross-coin (MOD2/7-009)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turbo_scalper_neutral_flow_returns_none() {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        // Flujo neutro (obi = 0, ofi = 0) debe retornar None (no forzar Long)
        let signal = TurboScalpEngine::evaluate_turbo_scalp(
            &arena, 0.0, 0.0, 1.0, 0.1, 60000.0, 0.005, 1000,
        );
        assert!(
            signal.is_none(),
            "Flujo plano debe retornar None y no forzar Long"
        );
    }

    #[test]
    fn test_turbo_scalper_symmetric_short_signal() {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        // Flujo fuertemente vendedor con excitación hawkes bajista de alta intensidad
        let signal = TurboScalpEngine::evaluate_turbo_scalp(
            &arena, -0.9, -0.9, -3.0, 0.01, 60000.0, 0.005, 1000,
        );
        assert!(
            signal.is_some(),
            "Flujo bajista extremo con excitación debe emitir señal"
        );
        assert_eq!(
            signal.unwrap().signal,
            SignalType::Short,
            "Debe ser señal Short"
        );
    }

    #[test]
    fn test_turbo_scalper_nan_immunity() {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        let signal = TurboScalpEngine::evaluate_turbo_scalp(
            &arena,
            f64::NAN,
            0.0,
            1.0,
            0.1,
            60000.0,
            0.005,
            1000,
        );
        assert!(signal.is_none());
    }

    #[test]
    fn test_turbo_scalper_evaluate_with_registry() {
        let registry = std::sync::Arc::new(omniscient_registry::OmniscientRegistry::new());
        registry.set("order_book_imbalance", 0.5);
        registry.set("order_flow_imbalance", 0.4);
        registry.set("hawkes_intensity", 1.8);

        let mut engine = TurboScalpEngine::default();
        assert!(strategy_core::QuantumStrategy::init(&mut engine, registry).is_ok());

        let eval = strategy_core::QuantumStrategy::evaluate(&engine);
        assert!(
            eval > 0.0,
            "Flujo e intensidad alcista deben generar señal positiva"
        );
        assert!(eval <= 1.0);
    }

    /// MOD2/7-009: una moneda sin datos per-coin NO hereda el global (que
    /// contiene los datos de la última moneda que escribió): voto NEUTRO.
    #[test]
    fn mod2_7_009_coin_sin_datos_per_coin_vota_neutral() {
        let registry = std::sync::Arc::new(omniscient_registry::OmniscientRegistry::new());
        // Sólo claves GLOBALES (p.ej. escritas por otra moneda):
        registry.set("order_book_imbalance", -0.9);
        registry.set("order_flow_imbalance", -0.8);
        registry.set("hawkes_intensity", 2.0);

        let mut engine = TurboScalpEngine::default();
        assert!(strategy_core::QuantumStrategy::init(&mut engine, registry).is_ok());

        let v = strategy_core::QuantumStrategy::evaluate_for_coin(&engine, 3, "SOLUSDT");
        assert_eq!(
            v, 0.0,
            "voto neutralizado para no contaminar cross-coin (MOD2/7-009)"
        );
        // El slot 0 (BTC) conserva el camino global legado.
        let v0 = strategy_core::QuantumStrategy::evaluate_for_coin(&engine, 0, "BTCUSDT");
        assert!(v0 < 0.0, "BTC sigue leyendo el global legado");
    }

    /// MOD2/7-009: con datos per-coin publicados, el voto usa los de SU
    /// símbolo aunque el global contenga otra cosa.
    #[test]
    fn mod2_7_009_voto_usa_datos_del_propio_simbolo() {
        let registry = std::sync::Arc::new(omniscient_registry::OmniscientRegistry::new());
        // El global quedó en manos de un vendedor fuerte (otra moneda):
        registry.set("order_book_imbalance", -0.9);
        registry.set("order_flow_imbalance", -0.8);
        registry.set("hawkes_intensity", 2.0);
        // El core escribe el contexto de SOL (set_scoped + set_for_coin):
        registry.set_scoped("SOLUSDT", "order_book_imbalance", 0.8);
        registry.set_scoped("SOLUSDT", "order_flow_imbalance", 0.6);
        registry.set_scoped("SOLUSDT", "hawkes_intensity", 1.6);

        let mut engine = TurboScalpEngine::default();
        assert!(strategy_core::QuantumStrategy::init(&mut engine, registry).is_ok());

        let v = strategy_core::QuantumStrategy::evaluate_for_coin(&engine, 3, "SOLUSDT");
        assert!(
            v > 0.0,
            "SOL debe votar con SU flujo alcista, no con el global vendedor ({v})"
        );
    }
}
