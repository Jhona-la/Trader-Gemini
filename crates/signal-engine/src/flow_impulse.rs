use strategy_core::{SignalIntent, SignalType};

/// MOTOR DE IMPULSO DE FLUJO.
///
/// Convierte microestructura cruda —desequilibrio del libro (OBI),
/// desequilibrio de flujo de órdenes (OFI), intensidad de auto-excitación de
/// Hawkes y entropía— en una opinión direccional continua con su confianza.
///
/// # U-ERR-1 (ERRADICACIÓN DEL BINARIO DE HORIZONTE)
///
/// Se llamaba `TurboScalpEngine` (`turbo_scalper.rs`) y su documentación
/// prometía «máxima frecuencia operativa manteniendo un 100% de tasa de
/// acierto»: una etiqueta de banda de horizonte más una afirmación de
/// rendimiento que ninguna medición forense respalda. Ninguna de las dos
/// describía la función real del motor, que no contiene ninguna escala
/// temporal en su regla: pondera flujo por pesos genómicos, penaliza por
/// entropía y exige significación estadística. El horizonte que emite es
/// `TradeHorizon::Continuous`. Nombre y documentación ahora dicen qué mide.
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct FlowImpulseEngine {
    registry: Option<std::sync::Arc<omniscient_registry::OmniscientRegistry>>,
}

impl std::fmt::Debug for FlowImpulseEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlowImpulseEngine").finish()
    }
}

impl FlowImpulseEngine {

    /// #649 (Ola 49) — VOTO ESPECTRAL del impulso de flujo, kernel-honesto:
    /// `voto_k = tanh(2·x(τ_k)) · excitacion_hawkes_norm(ratio)`. El tensor
    /// de flujo x(τ_k) YA es la agregación del flujo en la banda τ_k — la
    /// per-escala vive en él, no en un kernel. La excitación Hawkes (exceso
    /// real λ/μ̂ sobre SS) modula la convicción GLOBAL: un impulso de flujo
    /// en régimen normal no convence; en cascada, sí.
    ///
    /// Lo que ELIMINA (auditor A): el kernel e^{−β·τ_s} con τ en segundos
    /// mataba el voto en [30 s, 12 h] (e^{−15}…) y se contaba DOS veces
    /// (dentro de excitacion_k y otra vez en confianza ⇒ e^{−2βτ}); la
    /// malla hardcodeada 1e-6·4^k ms ni siquiera coincidía con la del
    /// espectro. El escalón de sign() se vuelve tanh (continuidad).
    /// Observacional: el voto vivo queda bit a bit (T-1 cero).
    pub fn voto_espectral(
        desplazamientos: &[f64; 32],
        ratio_lambda_mu: f64,
    ) -> crate::voto_espectral::VotoEspectral {
        let excitacion = crate::hawkes_bessel::excitacion_hawkes_norm(
            ratio_lambda_mu,
        );
        let mut por_escala = [0.0f64; 32];
        for k in 0..32 {
            let x = desplazamientos[k];
            if !x.is_finite() {
                continue;
            }
            // Respuesta AGUDA del tensor de flujo: el impulso es la
            // característica más nítida del flujo (pendiente ×2 del tanh).
            por_escala[k] = ((x.clamp(-10.0, 10.0) * 2.0).tanh() * excitacion).clamp(-1.0, 1.0);
        }
        crate::voto_espectral::VotoEspectral::desde_arr(&por_escala)
    }

    /// Infiere la intención direccional a partir del impulso de flujo (O(1)).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_flow_impulse(
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
        const SS_RATIO: f64 = crate::hawkes_bessel::STEADY_STATE_RATIO;
        let flow_tensor = (obi * w_obi + ofi * w_ofi) / (w_obi + w_ofi).max(0.01);
        // #663 (G2-2): el tensor de excitación es el EXCESO λ/μ̂ sobre
        // el estado estacionario (misma moneda de la casa que #649):
        // 0 en régimen normal (abstención), >0 sólo en cascada, y la
        // calma se recorta a 0 (no excita). Antes el ratio CRUDO
        // (SS=1.6) hacía que coherence y z_score no valieran 0 jamás —
        // el motor vivía encendido en régimen normal.
        let excitement_tensor =
            crate::hawkes_bessel::excitacion_hawkes_norm(hawkes_ratio)
                .max(0.0);
        // Coherence: Flow magnitude and Hawkes Excitement intensity (Symmetric for Long & Short)
        let coherence = (flow_tensor.abs() * excitement_tensor).sqrt();

        // Direction vector: continua C¹ (tanh) en lugar de signum —
        // #663 (G2-2): el escalón ±1 violaba el invariante 8. Flujo
        // neutro (|flow| ≤ 1e-6) no emite señal direccional.
        if flow_tensor.abs() <= 1e-6 {
            return None;
        }
        let direction_tensor = (flow_tensor / 1e-3).tanh();

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
        // Z-gate (#663 G2-2): sobre el EXCESO bruto (ratio−SS)/SS — el σ
        // del genoma (turbo_z_score_stdev) estaba calibrado en escala de
        // ratio; expresado en unidades de exceso se divide por SS=1.6.
        // En régimen normal (ratio≈SS) el exceso es 0 ⇒ z=0 ⇒ abstención
        // (antes el ratio crudo daba z≈0.64 de línea base y el motor
        // disparaba en cualquier pico normal por encima de ratio 2.7).
        let exceso_bruto = if hawkes_ratio.is_finite() && hawkes_ratio > 0.0 {
            (hawkes_ratio - SS_RATIO) / SS_RATIO
        } else {
            0.0
        };
        let z_score = exceso_bruto.max(0.0) / (turbo_z_score_stdev / SS_RATIO);
        let is_statistically_significant =
            z_score > dynamic_z_score_threshold;

        let turbo_coherence_threshold = arena
            .config
            .turbo_coherence_threshold
            .load(Ordering::Relaxed);
        // AGY-AUD-002: confidence gates signal CONTINUOUSLY, not as a binary
        // switch at 0.50. The downstream TensorVoteOrchestrator and council
        // apply genómic thresholds — this layer should not collapse a continuous
        // tensor to a boolean veto. Statistical significance and coherence
        // still gate the signal; confidence scales the intent's strength.
        if is_statistically_significant && coherence > turbo_coherence_threshold
        {
            let signal_type = if direction_tensor > 0.0 {
                SignalType::Long
            } else {
                SignalType::Short
            };

            // FIX #1509: Sanitización de la duración esperada de la señal.
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

        // #657 (F2-A2) — ERRADICACIÓN sombra/vivo: la MISMA moneda del
        // voto espectral #649 — excitación = exceso λ/μ̂ sobre
        // STEADY_STATE_RATIO. El umbral 1.2 fijo era TAUTOLÓGICO (1.2 <
        // SS ⇒ gate abierto en régimen normal) y los cortes 0.2 eran
        // escalones C⁰. La calma se abstiene (excit ≥ 0 — la firma
        // negativa se unifica con hawkes en la ola F2-A4); el flujo
        // entra continuo con la ganancia canónica del motor.
        const GANANCIA_FLUJO: f64 = 0.8;
        let excit = crate::hawkes_bessel::excitacion_hawkes_norm(safe_hawkes);
        if excit <= 0.0 {
            return 0.0;
        }
        let flow = (safe_obi + safe_ofi).clamp(-3.0, 3.0);
        (flow * GANANCIA_FLUJO).tanh() * excit
    }
}

impl strategy_core::QuantumStrategy for FlowImpulseEngine {
    fn name(&self) -> &str {
        "FlowImpulseEngine"
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
                r.get("order_book_imbalance", "FlowImpulseEngine")
                    .or_else(|| r.get("orderbook_imbalance", "FlowImpulseEngine"))
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        let ofi = self
            .registry
            .as_ref()
            .and_then(|r| r.get("order_flow_imbalance", "FlowImpulseEngine"))
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        let hawkes = self
            .registry
            .as_ref()
            .and_then(|r| r.get("hawkes_intensity", "FlowImpulseEngine"))
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
            r.get(&format!("{}_{}", symbol, key), "FlowImpulseEngine")
                .or_else(|| r.get(&format!("c{}:{}", coin_id, key), "FlowImpulseEngine"))
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
    fn flujo_neutro_no_emite_intencion() {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        // Flujo neutro (obi = 0, ofi = 0) debe retornar None (no forzar Long)
        let signal =
            FlowImpulseEngine::evaluate_flow_impulse(&arena, 0.0, 0.0, 1.0, 0.1, 60000.0, 0.005, 1000);
        assert!(
            signal.is_none(),
            "Flujo plano debe retornar None y no forzar Long"
        );
    }

    #[test]
    fn impulso_bajista_es_simetrico() {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        // #663: ratio FÍSICO de cascada (+6.4 ⇒ exceso 3σ sobre SS=1.6).
        // El −3.0 viejo era una intensidad negativa (no física) que sólo
        // pasaba el z-gate por leer el ratio CRUDO sin cero en SS.
        let signal = FlowImpulseEngine::evaluate_flow_impulse(
            &arena, -0.9, -0.9, 6.4, 0.01, 60000.0, 0.005, 1000,
        );
        assert!(
            signal.is_some(),
            "Flujo bajista extremo con cascada debe emitir señal"
        );
        assert_eq!(
            signal.unwrap().signal,
            SignalType::Short,
            "Debe ser señal Short"
        );
    }

    /// #663 (G2-2): el camino VIVO del flow_impulse se abstiene en
    /// régimen normal y en calma — la excitación es el EXCESO sobre SS,
    /// no el ratio crudo. Antes ratio=SS daba z≈0.64 de línea base.
    #[test]
    fn qo_663_regimen_normal_y_calma_abstienen_flow_impulse() {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        // Régimen normal (λ/μ̂ = SS = 1.6): exceso 0 ⇒ None.
        assert!(FlowImpulseEngine::evaluate_flow_impulse(
            &arena, 0.8, 0.8, 1.6, 0.1, 60000.0, 0.005, 1000,
        )
        .is_none());
        // Calma profunda (λ/μ̂ = 0.5): exceso negativo recortado ⇒ None.
        assert!(FlowImpulseEngine::evaluate_flow_impulse(
            &arena, 0.8, 0.8, 0.5, 0.1, 60000.0, 0.005, 1000,
        )
        .is_none());
        // Cascada clara (λ/μ̂ = 6.4 ⇒ exceso 3, z = 3/1.5625 = 1.92):
        // debe emitir.
        assert!(FlowImpulseEngine::evaluate_flow_impulse(
            &arena, 0.8, 0.8, 6.4, 0.01, 60000.0, 0.005, 1000,
        )
        .is_some());
    }

    #[test]
    fn inmunidad_a_nan() {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        let signal = FlowImpulseEngine::evaluate_flow_impulse(
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
    fn evalua_desde_el_registro() {
        let registry = std::sync::Arc::new(omniscient_registry::OmniscientRegistry::new());
        registry.set("order_book_imbalance", 0.5);
        registry.set("order_flow_imbalance", 0.4);
        registry.set("hawkes_intensity", 1.8);

        let mut engine = FlowImpulseEngine::default();
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

        let mut engine = FlowImpulseEngine::default();
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
        registry.set("hawkes_intensity", 1.6);
        // El core escribe el contexto de SOL (set_scoped + set_for_coin).
        // #657 (F2-A2): hawkes escopado en CASCADA (3.0) — el global queda
        // en SS (si leyera el global, se abstendría).
        registry.set_scoped("SOLUSDT", "order_book_imbalance", 0.8);
        registry.set_scoped("SOLUSDT", "order_flow_imbalance", 0.6);
        registry.set_scoped("SOLUSDT", "hawkes_intensity", 3.0);

        let mut engine = FlowImpulseEngine::default();
        assert!(strategy_core::QuantumStrategy::init(&mut engine, registry).is_ok());

        let v = strategy_core::QuantumStrategy::evaluate_for_coin(&engine, 3, "SOLUSDT");
        assert!(
            v > 0.0,
            "SOL debe votar con SU flujo alcista, no con el global vendedor ({v})"
        );
    }

    /// U-ERR-1: la identidad que el motor publica al registro no contiene
    /// etiquetas de banda de horizonte. Falla con el código viejo, cuyo
    /// `name()` devolvía «TurboScalpEngine».
    #[test]
    fn u_err_1_identidad_sin_etiqueta_de_banda() {
        let engine = FlowImpulseEngine::default();
        let n = strategy_core::QuantumStrategy::name(&engine).to_ascii_lowercase();
        assert!(
            !n.contains("scalp") && !n.contains("swing") && !n.contains("turbo"),
            "identidad con etiqueta de banda/marketing: {n}"
        );
    }
}

#[cfg(test)]
mod qo_619_tests {
    use super::*;
    use crate::voto_espectral::ESCALAS_VOTO;
    use crate::hawkes_bessel::STEADY_STATE_RATIO;

    /// #649 — el impulso convive con la excitación GLOBAL: régimen normal
    /// ⇒ abstención total; cascada ⇒ tanh(2x) en TODAS las escalas (el
    /// kernel viejo mataba la banda operable y contaba el kernel dos veces).
    #[test]
    fn qo_649_impulso_kernel_honesto() {
        let mut x = [0.0; ESCALAS_VOTO];
        for (k, v) in x.iter_mut().enumerate() {
            *v = 0.4 * (k as f64 - 15.5).signum();
        }
        // Estado estacionario ⇒ abstención total.
        let frio = FlowImpulseEngine::voto_espectral(&x, STEADY_STATE_RATIO);
        assert_eq!(frio.dominante(), None);
        // Cascada 3x: tanh(0.8) * tanh(1.4/1.6) en TODAS las escalas.
        let cascada = FlowImpulseEngine::voto_espectral(&x, 3.0);
        let esperado = (0.8f64).tanh() * ((3.0 - STEADY_STATE_RATIO) / STEADY_STATE_RATIO).tanh();
        for k in 0..ESCALAS_VOTO {
            let dir = if k as f64 - 15.5 > 0.0 { 1.0 } else { -1.0 };
            assert!(
                (cascada.en_escala(k) - esperado * dir).abs() < 1e-12,
                "escala {k}: {}",
                cascada.en_escala(k)
            );
        }
        // Antisimetría estricta.
        let mut x_neg = x;
        for v in &mut x_neg {
            *v = -*v;
        }
        let neg = FlowImpulseEngine::voto_espectral(&x_neg, 3.0);
        for k in 0..ESCALAS_VOTO {
            assert!((cascada.en_escala(k) + neg.en_escala(k)).abs() < 1e-12);
        }
        // Respuesta MAS AGUDA que el portador (pendiente x2 del tanh).
        let pico = FlowImpulseEngine::voto_espectral(&x, 3.0).en_escala(20).abs();
        let portador =
            crate::hawkes_bessel::HawkesBesselEngine::voto_espectral(&x, 3.0).en_escala(20).abs();
        assert!(pico > portador, "impulso {pico} vs portador {portador}");
    }
}

#[cfg(test)]
mod qo_657_tests {
    use super::*;
    use crate::hawkes_bessel::STEADY_STATE_RATIO;

    /// #657 (F2-A2): el umbral del vivo es el ESTADO ESTACIONARIO —
    /// régimen normal se abstiene aunque el flujo sea fuerte (antes el
    /// gate 1.2 < SS era tautológico); la cascada vota continuo.
    #[test]
    fn qo_657_flow_impulse_vote_umbral_ss() {
        assert_eq!(
            FlowImpulseEngine::vote(0.8, 0.5, STEADY_STATE_RATIO),
            0.0,
            "régimen normal se abstiene"
        );
        assert_eq!(
            FlowImpulseEngine::vote(0.8, 0.5, 1.2),
            0.0,
            "calma (1.2 < SS) no vota — antes era el gate tautológico"
        );
        let cascada = FlowImpulseEngine::vote(0.8, 0.5, 4.0);
        assert!(cascada > 0.3, "cascada vota alto, votó {cascada}");
        // Continuidad en el umbral: sin escalón al cruzar SS.
        let a = FlowImpulseEngine::vote(0.8, 0.5, STEADY_STATE_RATIO + 1e-9);
        assert!(a.abs() < 1e-6, "transición continua en SS, votó {a}");
        // Flujo continuo: 0.2 ya no es escalón.
        let debil = FlowImpulseEngine::vote(0.1, 0.05, 4.0);
        assert!(debil > 0.0 && debil < cascada, "flujo débil vota menos");
    }
}
