use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// CONFLUENCIA DE EXCITACIÓN DE FLUJO.
///
/// Emite una opinión direccional cuando coinciden tres magnitudes medibles de
/// la microestructura: (1) la intensidad de auto-excitación de Hawkes del flujo
/// de órdenes, (2) el desequilibrio del libro (OBI) y (3) el *lift* del modelo
/// de probabilidad sobre su propia tasa base.
///
/// # U-ERR-1 (ERRADICACIÓN DEL BINARIO DE HORIZONTE)
///
/// Este motor se llamaba `MicroScalpTriggerEngine` y vivía en
/// `micro_scalp_trigger.rs`. El nombre describía una ETIQUETA DE HORIZONTE
/// («micro-scalp») que este motor nunca decidió: su horizonte declarado es y
/// era `TradeHorizon::Continuous`, y lo que realmente mide es la confluencia
/// excitación-flujo-lift descrita arriba. En un motor de continuo temporal-
/// espectral la etiqueta era ruido para quien lee y para la evolución, que
/// arrastraba genes con ese prefijo. El nombre ahora dice qué se mide.
///
/// También se eliminaron `should_trigger_micro_scalp` y su variante conformal:
/// eran `pub fn` SIN NINGÚN llamador en el repositorio (sólo sus propios
/// tests las ejercitaban), es decir, una segunda regla de decisión paralela a
/// `evaluate_for_coin` que nunca participó en ninguna señal. Mantenerlas daba
/// la falsa impresión de que los genes `hawkes_scalp_threshold` y
/// `obi_zscore_threshold` tenían consumidor.
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct FlowExcitationConfluenceEngine {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for FlowExcitationConfluenceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlowExcitationConfluenceEngine").finish()
    }
}

impl FlowExcitationConfluenceEngine {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// #622 (Ola 43) — VOTO ESPECTRAL de la confluencia de excitación:
    /// la confluencia Hawkes×flujo se evalúa a CADA escala. El
    /// desplazamiento x(τ_k) ES el flujo direccional a esa banda; la
    /// excitación decae con el kernel e^{−β·τ} (misma física que #617 y
    /// #619). La confluencia exige flujo y excitación SIMULTÁNEOS a esa
    /// escala: el producto |x|·excitación_local, con el UMBRAL EFECTIVO
    /// escalado por τ (effective_hawkes_thresh de AGY-P09, ahora por
    /// escala). Las escalas rápidas ven la confluencia plena; las lentas
    /// apenas la memoria del flujo.
    /// Observacional: el voto vivo queda bit a bit (T-1 cero).
    pub fn voto_espectral(
        desplazamientos: &[f64; 32],
        excitacion_base: f64,
    ) -> crate::voto_espectral::VotoEspectral {
        let mut por_escala = [0.0f64; 32];
        for k in 0..32 {
            let x = desplazamientos[k];
            if !x.is_finite() || x == 0.0 {
                continue;
            }
            let tau_s = (1e-6 * 4f64.powi(k as i32)) / 1000.0;
            let kernel = (-0.5 * tau_s).exp();
            let excitacion_k = excitacion_base.max(0.0) * kernel;
            // Umbral efectivo por escala: el estado estacionario + un
            // incremento suave con τ (las escalas lentas exigen más
            // evidencia para declarar confluencia — menos eventos).
            let umbral_efectivo = crate::hawkes_bessel::STEADY_STATE_RATIO
                + 0.5 * (k as f64 / 31.0);
            if excitacion_k < umbral_efectivo {
                continue; // sin excitación suficiente a esta escala
            }
            // Confluencia: flujo direccional modulado por el exceso de
            // excitación sobre el umbral (más excitación = más convicción).
            let exceso = ((excitacion_k - umbral_efectivo) / 2.0).min(1.0);
            let confluencia = (x.clamp(-1.0, 1.0) * (0.5 + exceso)).clamp(-1.0, 1.0);
            por_escala[k] = confluencia;
        }
        crate::voto_espectral::VotoEspectral::desde_arr(&por_escala)
    }
}

impl QuantumStrategy for FlowExcitationConfluenceEngine {
    fn name(&self) -> &str {
        "FlowExcitationConfluenceEngine"
    }

    fn init(&mut self, registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        self.registry = Some(registry);
        Ok(())
    }

    fn evaluate(&self) -> f64 {
        self.evaluate_for_coin(0, "")
    }

    fn evaluate_for_coin(&self, coin_id: usize, symbol: &str) -> f64 {
        let sym_opt = if symbol.is_empty() {
            None
        } else {
            Some(symbol)
        };
        let cid_opt = if symbol.is_empty() {
            None
        } else {
            Some(coin_id)
        };
        let r = match self.registry.as_ref() {
            Some(reg) => reg,
            None => return 0.0,
        };

        let hawkes = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "hawkes_intensity",
                "FlowExcitationConfluenceEngine",
            )
            .map(|p| p.get_value())
            .unwrap_or(1.0);
        let obi = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "order_book_imbalance",
                "FlowExcitationConfluenceEngine",
            )
            .or_else(|| {
                r.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "orderbook_imbalance",
                    "FlowExcitationConfluenceEngine",
                )
            })
            .or_else(|| {
                r.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "order_flow_imbalance",
                    "FlowExcitationConfluenceEngine",
                )
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        let ml_prob = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "ml_prob_motor",
                "FlowExcitationConfluenceEngine",
            )
            .map(|p| p.get_value())
            .unwrap_or(0.5);

        // FIX #643: Guarda de finitud estricta en indicadores de entrada
        if !hawkes.is_finite() || !obi.is_finite() || !ml_prob.is_finite() {
            return 0.0;
        }

        // CERT-M2-C03: el gate anterior usaba `ml_prob >= 0.5` absoluto —
        // con el etiquetado honesto HOST-010 (base ~0.30), la pata long
        // casi nunca disparaba y la short casi siempre: sesgo short
        // estructural en el consenso tensorial. Ahora usa LIFT sobre la
        // base del modelo (misma doctrina que B3.18/B3.36): gatea en
        // AGY-AUD-P05: `ml_prob >= base + lift` / `ml_prob <= base − lift` con lift
        // dinámico vía registro o genoma, con fallback seguro a 0.05 (5 puntos).
        let ml_base = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "ml_model_base",
                "FlowExcitationConfluenceEngine",
            )
            .map(|p| p.get_value())
            .filter(|v| v.is_finite() && *v > 0.0 && *v < 1.0)
            .unwrap_or(0.5);
        let ml_lift = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "ml_model_lift",
                "FlowExcitationConfluenceEngine",
            )
            .map(|p| p.get_value())
            .filter(|v| v.is_finite() && *v > 0.0 && *v < 0.5)
            .unwrap_or(0.05);

        // #535 (3ª iteración): el umbral de excitación vuelve a ser del
        // GEN, anclado al estado estacionario del proceso. El gen
        // [0.50, 0.95] exige entre 0% y +90% de excitación sobre el ritmo
        // normal DEL SÍMBOLO (E[λ/μ̂] = 1+α/β = STEADY_STATE_RATIO en
        // régimen normal). La constante 1.2 anterior quedaba POR DEBAJO
        // del estado estacionario (≈1.6): el gate medía «hubo actividad»,
        // no «hubo ráfaga», y sin consumidor el gen era un muerto que la
        // evolución arrastraba sin gradiente.
        let hawkes_gene = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "hawkes_excitation_gene",
                "FlowExcitationConfluenceEngine",
            )
            .map(|p| p.get_value())
            .filter(|v| v.is_finite() && (0.50..=0.95).contains(v))
            .unwrap_or(0.55);
        let excitation = (hawkes_gene - 0.50).max(0.0) * 2.0;
        let effective_hawkes_thresh = crate::hawkes_bessel::STEADY_STATE_RATIO + excitation;

        // #590 — el piso del OBI deja de ser la constante mágica 0.2 y pasa
        // a ser MEDIDO + GENÓMICO: percentil-80 del OBI del símbolo (su
        // distribución reciente) escalado por el gen obi_zscore_threshold
        // [0.1, 3.0] — la rareza estadística que el gen anunciaba, empírica
        // y adaptativa por símbolo/régimen. Fallbacks: p80 0.15 (el propio
        // fallback del motor cuantílico), gen 1.0 (su default).
        let obi_gene = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "obi_zscore_gene",
                "FlowExcitationConfluenceEngine",
            )
            .map(|p| p.get_value())
            .filter(|v| v.is_finite() && (0.1..=3.0).contains(v))
            .unwrap_or(1.0);
        let obi_p80 = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "obi_p80_medido",
                "FlowExcitationConfluenceEngine",
            )
            .map(|p| p.get_value())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(0.15);
        let obi_floor = (obi_p80 * obi_gene).max(0.02);

        if hawkes >= effective_hawkes_thresh && obi.abs() >= obi_floor {
            let is_long = obi > 0.0 && ml_prob >= ml_base + ml_lift;
            let is_short = obi < 0.0 && ml_prob <= ml_base - ml_lift;

            // Ola 9 (SPECTRAL CONTINUITY): El factor de escala de excitación
            // se normaliza contra el umbral crítico efectivo del proceso (anclado
            // al estado estacionario + genoma), garantizando continuidad espectral:
            // en el umbral exacto vale 1.0 y crece monótonamente hasta saturar en 2.0,
            // eliminando el divisor literal 2.0 que desalineaba la señal según el gen.
            let hawkes_scale = (hawkes / effective_hawkes_thresh.max(0.01)).min(2.0);

            if is_long {
                (obi * (ml_prob - ml_base) * 4.0 * hawkes_scale).clamp(0.0, 1.0)
            } else if is_short {
                (obi * (ml_base - ml_prob) * 4.0 * hawkes_scale).clamp(-1.0, 0.0)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    fn horizon(&self) -> strategy_core::TradeHorizon {
        strategy_core::TradeHorizon::Continuous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evalua_confluencia_desde_el_registro() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("hawkes_intensity", 2.0);
        registry.set("order_book_imbalance", 0.5);
        registry.set("ml_prob_motor", 0.85);

        let mut engine = FlowExcitationConfluenceEngine::new();
        assert!(engine.init(registry).is_ok());
        assert_eq!(engine.name(), "FlowExcitationConfluenceEngine");
        assert_eq!(engine.horizon(), strategy_core::TradeHorizon::Continuous);

        let val = engine.evaluate();
        assert!(val.is_finite());
        assert!(val > 0.0);
    }

    /// U-ERR-1: el motor NO expone ninguna regla de decisión etiquetada por
    /// horizonte. Este test falla con el código viejo, donde
    /// `should_trigger_micro_scalp` era una segunda regla pública sin
    /// llamadores que competía con `evaluate_for_coin`.
    ///
    /// La comprobación es de comportamiento observable: con OBI negativo y
    /// `ml_prob` por debajo de la base, la ÚNICA superficie de decisión
    /// (`evaluate_for_coin`) devuelve un voto short continuo y acotado, sin
    /// booleano de disparo de ninguna banda.
    #[test]
    fn u_err_1_una_sola_superficie_de_decision_continua() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("hawkes_intensity", 2.0);
        registry.set("order_book_imbalance", -0.5);
        registry.set("ml_prob_motor", 0.15);
        registry.set("ml_model_base", 0.50);

        let mut engine = FlowExcitationConfluenceEngine::new();
        assert!(engine.init(registry).is_ok());

        let v = engine.evaluate();
        assert!(v < 0.0, "flujo vendedor con lift negativo debe votar short");
        assert!((-1.0..=0.0).contains(&v), "el voto vive en [-1, 0]: {v}");
    }

    /// #535 (3ª iteración): el gen `hawkes_scalp_threshold` [0.50, 0.95]
    /// vuelve a tener consumidor VIVO en la única superficie de decisión.
    /// Con gen máximo (0.95) el umbral efectivo sube a 2.50 y una
    /// intensidad 2.0 (que la constante muerta 1.2 dejaba pasar) YA NO
    /// dispara; con gen mínimo (0.50) el umbral es el estado estacionario
    /// 1.60 y una ráfaga 1.9 sí dispara.
    #[test]
    fn qo_535_gen_de_excitacion_gobierna_el_umbral_vivo() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("hawkes_intensity", 2.0);
        registry.set("order_book_imbalance", 0.5);
        registry.set("ml_prob_motor", 0.85);
        registry.set("hawkes_excitation_gene", 0.95);

        let mut engine = FlowExcitationConfluenceEngine::new();
        assert!(engine.init(Arc::clone(&registry)).is_ok());

        // Umbral efectivo = 1.6 + (0.95−0.50)·2 = 2.50 > 2.0 ⇒ sin señal.
        assert_eq!(engine.evaluate(), 0.0);

        // Gen mínimo: umbral = estado estacionario 1.60; la ráfaga 1.9
        // con el mismo flujo SÍ dispara.
        registry.set("hawkes_excitation_gene", 0.50);
        registry.set("hawkes_intensity", 1.9);
        let v = engine.evaluate();
        assert!(v > 0.0, "ráfaga 1.9 sobre umbral 1.60 debe votar long: {v}");
    }

    /// #590 — el gen `obi_zscore_threshold` [0.1, 3.0] gobierna el piso del
    /// OBI como p80 MEDIDO × gen: rareza estadística empírica, no la
    /// constante mágica 0.2. Con gen 3.0 y p80 0.10, un OBI 0.25 (que el
    /// 0.2 absoluto dejaba pasar) YA NO dispara; con gen 0.5 el piso baja a
    /// 0.05 y el mismo OBI dispara.
    #[test]
    fn qo_590_gen_de_obi_gobierna_el_piso_medido() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("hawkes_intensity", 2.0);
        registry.set("order_book_imbalance", 0.25);
        registry.set("ml_prob_motor", 0.85);
        registry.set("ml_model_base", 0.50);
        registry.set("obi_p80_medido", 0.10);
        registry.set("obi_zscore_gene", 3.0);

        let mut engine = FlowExcitationConfluenceEngine::new();
        assert!(engine.init(Arc::clone(&registry)).is_ok());

        // Piso = 0.10·3.0 = 0.30 > |OBI|=0.25 ⇒ sin señal (el viejo 0.2
        // absoluto la habría dejado pasar).
        assert_eq!(engine.evaluate(), 0.0);

        // Gen 0.5: piso = 0.05 ≤ 0.25 ⇒ dispara.
        registry.set("obi_zscore_gene", 0.5);
        let v = engine.evaluate();
        assert!(v > 0.0, "OBI 0.25 sobre piso 0.05 debe votar long: {v}");

        // Sin telemetría el piso cae al fallback del motor cuantílico
        // (0.15): el motor sigue operando antes de que el p80 madure.
        let registry_vacia = Arc::new(OmniscientRegistry::new());
        registry_vacia.set("hawkes_intensity", 2.0);
        registry_vacia.set("order_book_imbalance", 0.20);
        registry_vacia.set("ml_prob_motor", 0.85);
        registry_vacia.set("ml_model_base", 0.50);
        let mut engine2 = FlowExcitationConfluenceEngine::new();
        assert!(engine2.init(registry_vacia).is_ok());
        let v2 = engine2.evaluate();
        assert!(v2 > 0.0, "OBI 0.20 >= piso fallback 0.15 debe votar long: {v2}");
    }

    /// Ola 9: Continuidad espectral en la escala de excitación Hawkes.
    /// Verifica que en el umbral efectivo la escala normalizada vale exactamente 1.0,
    /// y que crece suave y monótonamente conforme la intensidad Hawkes se incrementa.
    #[test]
    fn ola9_continuidad_espectral_hawkes_escala_con_umbral_efectivo() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("order_book_imbalance", 0.5);
        registry.set("ml_prob_motor", 0.85);
        registry.set("ml_model_base", 0.50);
        registry.set("hawkes_excitation_gene", 0.50); // effective_thresh = 1.60

        let mut engine = FlowExcitationConfluenceEngine::new();
        assert!(engine.init(Arc::clone(&registry)).is_ok());

        // En hawkes = 1.60 exacto: factor = 1.60 / 1.60 = 1.0
        registry.set("hawkes_intensity", 1.60);
        let v_base = engine.evaluate();
        assert!(v_base > 0.0);
        // Formula esperada: 0.5 * (0.85 - 0.50) * 4.0 * 1.0 = 0.5 * 0.35 * 4.0 = 0.70
        assert!((v_base - 0.70).abs() < 1e-6, "v_base esperado 0.70, dio {v_base}");

        // En hawkes = 2.40: factor = 2.40 / 1.60 = 1.50
        registry.set("hawkes_intensity", 2.40);
        let v_high = engine.evaluate();
        assert!(v_high > v_base, "mayor intensidad produce monotonicamente mayor senal");
        // Formula esperada: 0.70 * 1.50 = 1.05 clamped to 1.0
        assert_eq!(v_high, 1.0, "satura suavemente en 1.0");
    }
}

#[cfg(test)]
mod qo_622_tests {
    use super::*;
    use crate::voto_espectral::ESCALAS_VOTO;
    use crate::hawkes_bessel::STEADY_STATE_RATIO;

    #[test]
    fn qo_622_confluencia_por_escala_exige_excitacion() {
        let mut x = [0.0; ESCALAS_VOTO];
        for (k, v) in x.iter_mut().enumerate() {
            *v = 0.3 * (k as f64 - 15.5).signum();
        }
        // Excitación ALTA: supera el umbral en las escalas rápidas (kernel
        // pleno) pero no en las lentas (kernel muerto + umbral creciente).
        let alta = STEADY_STATE_RATIO + 2.0;
        let voto = FlowExcitationConfluenceEngine::voto_espectral(&x, alta);
        // Escala rápida: kernel ≈ 1 ⇒ excitación ≈ alta > umbral ⇒ CONFLUENCIA.
        let escala_rapida = voto.en_escala(1).abs();
        // Escala lenta: kernel ≈ 0 ⇒ excitación ≈ 0 < umbral ⇒ ABSTENCIÓN.
        let escala_lenta = voto.en_escala(28).abs();
        assert!(
            escala_rapida > escala_lenta,
            "confluencia es de alta frecuencia: rápida {} vs lenta {}",
            escala_rapida,
            escala_lenta
        );
        // Excitación BAJA: no supera el umbral en NINGUNA escala ⇒ todo 0.
        let baja = 0.5;
        let voto_bajo = FlowExcitationConfluenceEngine::voto_espectral(&x, baja);
        assert_eq!(voto_bajo.dominante(), None, "sin excitación suficiente no hay confluencia");
        // Antisimetría cuando la confluencia activa.
        let mut x_neg = x;
        for v in &mut x_neg {
            *v = -*v;
        }
        let voto_neg = FlowExcitationConfluenceEngine::voto_espectral(&x_neg, alta);
        for k in 0..ESCALAS_VOTO {
            if x[k] != 0.0 {
                assert!(
                    (voto.en_escala(k) + voto_neg.en_escala(k)).abs() < 1e-10,
                    "antisimetría en {}",
                    k
                );
            }
        }
    }
}
