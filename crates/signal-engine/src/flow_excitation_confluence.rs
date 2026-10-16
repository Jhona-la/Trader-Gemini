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
    /// #649 (Ola 49) — VOTO ESPECTRAL de la confluencia flujo×excitación,
    /// kernel-honesto: `voto_k = tanh(x(τ_k)) · √(|x(τ_k)|·|excit|)` — la
    /// COHERENCIA GEOMÉTRICA (media proporcional) de la magnitud del flujo
    /// a esa escala y del exceso real de excitación λ/μ̂ sobre SS. La
    /// confluencia exige AMBAS: flujo fuerte Y proceso excitado; si either
    /// falta, el voto decae suavemente a 0.
    ///
    /// Lo que ELIMINA (auditor A): el kernel e^{−β·τ_s} con τ en segundos
    /// mataba `excitacion_k` en [30 s, 12 h] y el `continue` de umbral era
    /// un ESCALÓN de magnitud completa (violación C∞ de la doctrina);
    /// además excitacion_base llegaba como literal 2.5 — con umbral ≥1.6
    /// sólo votaban escalas con τ ≲ 1 s. La media geométrica es continua,
    /// antisimétrica y acotada por construcción.
    /// Observacional: el voto vivo queda bit a bit (T-1 cero).
    /// OLA Ω133 — VOTO ESPECTRAL CON MODULACIÓN DE LIQUIDEZ OCULTA (KYLE-HASBROUCK)
    /// OLA Ω137 — VOTO ESPECTRAL CON MODULACIÓN HELMHOLTZ-HODGE Y LIQUIDEZ OCULTA
    /// OLA Ω138 — VOTO ESPECTRAL CON MODULACIÓN GAUGE YANG-MILLS SU(N) Y HELMHOLTZ-HODGE
    /// Incorpora absorción iceberg local, pureza laminar L² y corrientes restauradoras covariantes J_i.
    pub fn voto_espectral_con_gauge_y_helmholtz(
        desplazamientos: &[f64; 32],
        ratio_lambda_mu: f64,
        hidden_liq_pressure: f64,
        laminar_fraction: f64,
        yang_mills_current: f64,
    ) -> crate::voto_espectral::VotoEspectral {
        let excitacion = crate::hawkes_bessel::excitacion_hawkes_norm(
            ratio_lambda_mu,
        )
        .max(0.0);
        let mut por_escala = [0.0f64; 32];
        let p_ice = if hidden_liq_pressure.is_finite() {
            hidden_liq_pressure.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let lam_scale = if laminar_fraction.is_finite() {
            laminar_fraction.clamp(0.10, 1.50)
        } else {
            1.0
        };
        let j_ym = if yang_mills_current.is_finite() {
            yang_mills_current.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        for k in 0..32 {
            let x = desplazamientos[k];
            if !x.is_finite() {
                continue;
            }
            // En micro-escalas (k < 12, τ < 1 s), la presión de iceberg modula el flujo local;
            // en macro-escalas (k >= 12), la corriente covariante Yang-Mills modula el campo continuo multiactivo.
            let micro_factor = (1.0 - (k as f64 / 32.0)).max(0.0);
            let macro_factor = (k as f64 / 32.0).clamp(0.0, 1.0);
            let flujo = (x + p_ice * 0.5 * micro_factor + j_ym * 0.4 * macro_factor).clamp(-10.0, 10.0);
            let direccion = flujo.tanh();
            let coherencia = (flujo.abs().min(1.0) * excitacion * lam_scale).sqrt();
            por_escala[k] = (direccion * coherencia).clamp(-1.0, 1.0);
        }
        crate::voto_espectral::VotoEspectral::desde_arr(&por_escala)
    }

    pub fn voto_espectral_con_helmholtz(
        desplazamientos: &[f64; 32],
        ratio_lambda_mu: f64,
        hidden_liq_pressure: f64,
        laminar_fraction: f64,
    ) -> crate::voto_espectral::VotoEspectral {
        Self::voto_espectral_con_gauge_y_helmholtz(desplazamientos, ratio_lambda_mu, hidden_liq_pressure, laminar_fraction, 0.0)
    }

    pub fn voto_espectral_con_liquidez(
        desplazamientos: &[f64; 32],
        ratio_lambda_mu: f64,
        hidden_liq_pressure: f64,
    ) -> crate::voto_espectral::VotoEspectral {
        Self::voto_espectral_con_helmholtz(desplazamientos, ratio_lambda_mu, hidden_liq_pressure, 1.0)
    }

    pub fn voto_espectral(
        desplazamientos: &[f64; 32],
        ratio_lambda_mu: f64,
    ) -> crate::voto_espectral::VotoEspectral {
        Self::voto_espectral_con_helmholtz(desplazamientos, ratio_lambda_mu, 0.0, 1.0)
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
        // TRIAJE B (GLM 109) — CL-15: "no opina es AUSENCIA, no un 0,5".
        // Los fallbacks 0.5 fabricaban lift: con ml_prob 0.36 real y base
        // inventada 0.5, la pata short votaba −0.33 QUE NO EXISTIRÍA con la
        // base honesta (CL-21: bases reales 0.18-0.23; lib.rs:4328 sólo
        // cae a 0.5 cuando NO hay bosque). Sin prob o sin base medible, el
        // motor ML NO OPINA (voto 0) — misma doctrina que el
        // DarkAlpha-fallback-None de B-H1.
        let ml_prob = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "ml_prob_motor",
                "FlowExcitationConfluenceEngine",
            )
            .map(|p| p.get_value())
            .filter(|v| v.is_finite());

        // FIX #643: Guarda de finitud estricta en indicadores de entrada
        if !hawkes.is_finite() || !obi.is_finite() {
            return 0.0;
        }
        let ml_prob = match ml_prob {
            Some(v) if v.is_finite() => v,
            _ => return 0.0, // CL-15: sin prob del modelo, no opina
        };

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
            .filter(|v| v.is_finite() && *v > 0.0 && *v < 1.0);
        let ml_base = match ml_base {
            Some(v) => v,
            None => return 0.0, // CL-15: sin base medida, el lift no es medible — no opina
        };
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

        // OLA Ω133: ACOPLAMIENTO TENSORIAL DE LIQUIDEZ OCULTA (KYLE-HASBROUCK)
        // Leemos la presión neta de iceberg y el estado de absorción desde el registro.
        let hidden_liq_pressure = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "hidden_liquidity_pressure",
                "FlowExcitationConfluenceEngine",
            )
            .map(|p| p.get_value())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
            .clamp(-1.0, 1.0);

        let is_iceberg_active = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "is_iceberg_active",
                "FlowExcitationConfluenceEngine",
            )
            .map(|p| p.get_value())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0) > 0.5;

        let hidden_liq_weight = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "hidden_liquidity_weight",
                "FlowExcitationConfluenceEngine",
            )
            .map(|p| p.get_value())
            .filter(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0)
            .unwrap_or(0.30);

        // Modulación de OBI efectivo: el libro visible se complementa con la absorción oculta
        let obi_eff = (obi + hidden_liq_weight * hidden_liq_pressure).clamp(-1.0, 1.0);

        // #664 (G2-3): pertenencia CONTINUA por exceso (rampas C¹
        // smoothstep de ancho relativo) — con el gate duro el voto
        // saltaba de 0 a obi·lift·4·hawkes_scale al cruzar cualquiera
        // de los dos umbrales. Dentro de la región (exceso ≥ ancho)
        // el peso es 1: el comportamiento en región se conserva.
        let rampa = |exceso: f64, ancho: f64| {
            let t = (exceso / ancho.max(1e-9)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        let w_hawkes = rampa(
            hawkes - effective_hawkes_thresh,
            0.30 * effective_hawkes_thresh.max(0.01),
        );
        let w_obi_raw = rampa(obi.abs() - obi_floor, 0.50 * obi_floor);
        let w_iceberg = if is_iceberg_active {
            rampa(hidden_liq_pressure.abs() - 0.25, 0.25)
        } else {
            0.0
        };
        let w_obi = (w_obi_raw + hidden_liq_weight * w_iceberg).min(1.0);

        // #666 (H2-4): TERCERA puerta CONTINUA — el gate de ML
        // (ml_prob ≥ base ± lift) quedaba binario: el voto saltaba de 0 a
        // obi·lift·4·scale al cruzar el umbral. Rampa smoothstep de ancho
        // lift: dentro (≥ 2·lift de exceso) peso 1, comportamiento en
        // región profunda conservado.
        let w_ml_long = rampa(ml_prob - (ml_base + ml_lift), ml_lift.max(1e-4));
        let w_ml_short = rampa((ml_base - ml_lift) - ml_prob, ml_lift.max(1e-4));
        let is_long = obi_eff > 0.0;
        let is_short = obi_eff < 0.0;

        // Ola 9 (SPECTRAL CONTINUITY): El factor de escala de excitación
        // se normaliza contra el umbral crítico efectivo del proceso (anclado
        // al estado estacionario + genoma), garantizando continuidad espectral:
        // en el umbral exacto vale 1.0 y crece monótonamente hasta saturar en 2.0,
        // eliminando el divisor literal 2.0 que desalineaba la señal según el gen.
        let hawkes_scale = (hawkes / effective_hawkes_thresh.max(0.01)).min(2.0);

        let base_voto = if is_long {
            (obi_eff * (ml_prob - ml_base) * 4.0 * hawkes_scale).clamp(0.0, 1.0) * w_ml_long
        } else if is_short {
            (obi_eff * (ml_base - ml_prob) * 4.0 * hawkes_scale).clamp(-1.0, 0.0) * w_ml_short
        } else {
            0.0
        };

        // Modulación Direccional Tensorial por Liquidez Oculta:
        let mut mult_liq = 1.0;
        if base_voto.abs() > 1e-9 && hidden_liq_pressure.abs() > 1e-4 {
            let intent_dir = base_voto.signum();
            let alignment = intent_dir * hidden_liq_pressure; // in [-1.0, 1.0]

            if alignment >= 0.0 {
                // Flujo ALINEADO con la absorción institucional:
                // La absorción actúa como soporte/trampolín dinámico (+bonus hasta +30%)
                mult_liq = 1.0 + hidden_liq_weight * rampa(alignment, 0.50);
            } else {
                // Flujo OPUESTO a la absorción institucional (ej. Long atacando muro de venta iceberg)
                // Atenuación suave C¹ (smoothstep):
                // x_opp ∈ (0.0, 1.0]. A partir de x_opp = 0.15 comienza a atenuar,
                // y al alcanzar x_opp >= 0.70 (veto conformal) el voto se extingue a 0.0
                let x_opp = -alignment;
                let t_opp = ((x_opp - 0.15) / 0.55).clamp(0.0, 1.0);
                mult_liq = 1.0 - t_opp * t_opp * (3.0 - 2.0 * t_opp);
            }
        }

        // OLA Ω137: MODULACIÓN POR DESCOMPOSICIÓN DE HELMHOLTZ-HODGE (L² = ∇φ ⊕ ∇×A ⊕ h)
        // La convicción modula la señal: flujos laminares genuinos (Λ > 0.35) reciben bonificación,
        // mientras que alta enstrofía rotacional (spoofing / turbulencia) atenúa suavemente vía Hermite C¹.
        let helmholtz_conviction = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "helmholtz_conviction",
                "FlowExcitationConfluenceEngine",
            )
            .map(|p| p.get_value())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(1.0);

        // OLA Ω138: MODULACIÓN POR CURVATURA GAUGE YANG-MILLS SU(N) Y CORRIENTES RESTAURADORAS J_i
        // Leemos la corriente gauge covariante J_i calculada en tiempo real para este activo en la red multiactivo.
        // Si el flujo local está alineado con la fuerza restauradora global del fibrado principal,
        // recibe bonificación de confluencia (+hasta 20%); si colisiona contra una dislocación severa,
        // se atenúa suavemente vía Hermite C¹ (smoothstep).
        let ym_current_absent = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "yang_mills_current_absent",
                "FlowExcitationConfluenceEngine",
            )
            .map(|p| p.get_value())
            .unwrap_or(0.0) > 0.5;

        let ym_current = if !ym_current_absent {
            r.get_scoped_parameter(
                sym_opt,
                cid_opt,
                "yang_mills_current",
                "FlowExcitationConfluenceEngine",
            )
            .map(|p| p.get_value())
            .filter(|v| v.is_finite())
            .unwrap_or(0.0)
            .clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let mut mult_ym = 1.0;
        if !ym_current_absent && base_voto.abs() > 1e-9 && ym_current.abs() > 1e-4 {
            let intent_dir = base_voto.signum();
            let ym_alignment = intent_dir * ym_current; // in [-1.0, 1.0]

            if ym_alignment >= 0.0 {
                // Flujo ALINEADO con la corriente restauradora gauge SU(N):
                // La paridad global impulsa la reversión a favor de nuestra posición (+bonus hasta +20%)
                mult_ym = 1.0 + 0.20 * rampa(ym_alignment, 0.40);
            } else {
                // Flujo OPUESTO a la corriente restauradora gauge SU(N) (ej. Long cuando el activo está sobrecomprado en la tríada):
                // Atenuación suave C¹ (Hermite smoothstep): a partir de -0.15 atenúa, y si la oposición es fuerte (-0.75) atenúa al 20%
                let x_opp = -ym_alignment;
                let t_opp = ((x_opp - 0.15) / 0.60).clamp(0.0, 1.0);
                mult_ym = 1.0 - 0.80 * t_opp * t_opp * (3.0 - 2.0 * t_opp);
            }
        }

        (base_voto * w_hawkes * w_obi * mult_liq * helmholtz_conviction * mult_ym).clamp(-1.0, 1.0)
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
        // GLM 109: base honesta explícita — el fallback 0.5 fue removido
        // (CL-15: sin base medida el motor NO opina).
        registry.set("ml_model_base", 0.30);

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
        registry.set("ml_model_base", 0.30);
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

        // #664 (G2-3): pertenencia CONTINUA — EN el umbral el voto es 0
        // y crece suave hasta el pleno a umbral+ancho (30% del umbral,
        // 0.48 aquí); el factor hawkes_scale sigue valiendo 1.0 en el
        // umbral y saturando en 2.0.
        registry.set("hawkes_intensity", 1.60);
        let v_base = engine.evaluate();
        assert_eq!(v_base, 0.0, "en el umbral exacto el peso de exceso es 0");
        // Punto medio de la rampa (umbral + ancho/2): smoothstep(0.5)=0.5
        // y el factor de escala crece con la intensidad: 1.84/1.60 = 1.15
        registry.set("hawkes_intensity", 1.60 + 0.24);
        let v_medio = engine.evaluate();
        assert!(
            (v_medio - 0.70 * 1.15 * 0.5).abs() < 1e-6,
            "punto medio de la rampa: {v_medio}"
        );
        // Pleno (umbral + ancho): peso 1; el factor de escala es
        // 2.08/1.60 = 1.30 ⇒ 0.70 · 1.30 = 0.91
        registry.set("hawkes_intensity", 1.60 + 0.48);
        let v_pleno = engine.evaluate();
        assert!(
            (v_pleno - 0.70 * 1.30).abs() < 1e-6,
            "v_pleno esperado 0.91, dio {v_pleno}"
        );
        assert!(v_medio < v_pleno, "monotonia en la rampa");

        // En hawkes = 2.40: pleno y factor = 2.40 / 1.60 = 1.50
        registry.set("hawkes_intensity", 2.40);
        let v_high = engine.evaluate();
        assert!(v_high > v_pleno, "mayor intensidad produce monotonicamente mayor senal");
        // Formula esperada: 0.70 * 1.50 = 1.05 clamped to 1.0
        assert_eq!(v_high, 1.0, "satura suavemente en 1.0");
    }
}

#[cfg(test)]
mod qo_622_tests {
    use super::*;
    use crate::voto_espectral::ESCALAS_VOTO;
    use crate::hawkes_bessel::STEADY_STATE_RATIO;

    /// #649 — confluencia como COHERENCIA GEOMÉTRICA continua: sin
    /// escalón de umbral, exige flujo Y excitación, funciona en TODA la
    /// banda (el kernel viejo mataba [30 s, 12 h] y el continue era un
    /// salto de magnitud completa).
    #[test]
    fn qo_649_confluencia_geometrica_continua() {
        let mut x = [0.0; ESCALAS_VOTO];
        for (k, v) in x.iter_mut().enumerate() {
            *v = 0.3 * (k as f64 - 15.5).signum();
        }
        // Estado estacionario ⇒ |excit|=0 ⇒ coherencia 0 ⇒ abstención.
        let frio = FlowExcitationConfluenceEngine::voto_espectral(&x, STEADY_STATE_RATIO);
        assert_eq!(frio.dominante(), None);
        // Cascada 3x: voto = tanh(0.3)·sqrt(0.3·|excit|) en TODA escala.
        let cascada = FlowExcitationConfluenceEngine::voto_espectral(&x, 3.0);
        let excit = ((3.0 - STEADY_STATE_RATIO) / STEADY_STATE_RATIO).tanh().abs();
        let esperado = (0.3f64).tanh() * (0.3f64 * excit).sqrt();
        for k in 0..ESCALAS_VOTO {
            let dir = if k as f64 - 15.5 > 0.0 { 1.0 } else { -1.0 };
            assert!(
                (cascada.en_escala(k) - esperado * dir).abs() < 1e-12,
                "escala {k}: {}",
                cascada.en_escala(k)
            );
        }
        // CONTINUIDAD: no hay escalón — x pequeno da voto pequeno.
        let mut x_grad = [0.0; ESCALAS_VOTO];
        x_grad[12] = 1e-6;
        let suave = FlowExcitationConfluenceEngine::voto_espectral(&x_grad, 3.0);
        assert!(suave.en_escala(12).abs() < 1e-5);
        // Sin flujo no hay confluencia (requiere AMBAS magnitudes).
        let cero = [0.0; ESCALAS_VOTO];
        let quieto = FlowExcitationConfluenceEngine::voto_espectral(&cero, 3.0);
        assert_eq!(quieto.dominante(), None);
        // Antisimetría.
        let mut x_neg = x;
        for v in &mut x_neg {
            *v = -*v;
        }
        let neg = FlowExcitationConfluenceEngine::voto_espectral(&x_neg, 3.0);
        for k in 0..ESCALAS_VOTO {
            assert!((cascada.en_escala(k) + neg.en_escala(k)).abs() < 1e-12);
        }
    }
}

#[cfg(test)]
mod qo_663_tests {
    use super::*;

    /// #663 (G2-1): la CALMA abstiene en el voto espectral del
    /// confluence — excit = exceso .max(0). Antes el `.abs()` hacía que
    /// λ/μ̂→0.1 (calma extrema) votara 0.734, MÁS que una cascada 3×.
    #[test]
    fn qo_663_calma_no_vota_mas_que_la_cascada() {
        let desplazamientos = [0.5f64; 32];
        let calma = FlowExcitationConfluenceEngine::voto_espectral(
            &desplazamientos,
            0.1,
        );
        let cascada = FlowExcitationConfluenceEngine::voto_espectral(
            &desplazamientos,
            4.8,
        );
        let ss = FlowExcitationConfluenceEngine::voto_espectral(
            &desplazamientos,
            1.6,
        );
        for k in 0..32 {
            assert_eq!(
                calma.en_escala(k), 0.0,
                "calma extrema debe abstenerse en toda escala"
            );
            assert_eq!(
                ss.en_escala(k), 0.0,
                "estado estacionario debe abstenerse"
            );
        }
        let max_calma = (0..32).map(|k| calma.en_escala(k)).fold(0.0, f64::max);
        let max_cascada =
            (0..32).map(|k| cascada.en_escala(k)).fold(0.0, f64::max);
        assert!(
            max_cascada > max_calma,
            "cascada {max_cascada} debe superar calma {max_calma}"
        );
        assert!(max_cascada > 0.0, "cascada debe votar");
    }
}
