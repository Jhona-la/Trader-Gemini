//! # Fase 19 — Causalidad Dirigida Spot $\to$ Futuros (`SpotFuturesLeadLagEngine`), Cazador Bayesiano de Liquidez Oculta L3 (`IcebergHunterL3Engine`) y Formalismo Cuántico de Dos Vectores de Estado de Aharonov-Bergmann-Lebowitz (`AharonovBergmannLebowitzEngine` / Weak Values)
//!
//! ## 👨‍🏫 MODO PROFESOR (QUÉ, POR QUÉ, PARA QUÉ, CÓMO, CUÁNDO, DÓNDE, QUIÉN)
//!
//! - **QUÉ:** Tres motores deterministas en pila (`O(1)` memoria, cero `unwrap`/`assert!`, 100% Rust nativo):
//!   1. `SpotFuturesLeadLagEngine`: Estimador de liderazgo causal dirigido entre el mercado Spot al contado y los Futuros Perpetuos ($\mathcal{I}(\text{Spot} \to \text{Perp}) - \mathcal{I}(\text{Perp} \to \text{Spot})$) combinado con la tensión de base instantánea $F_t - S_t$.
//!   2. `IcebergHunterL3Engine`: Inferencia bayesiana secuencial de órdenes institucionales ocultas (*Iceberg L3*) que detecta la recarga automática de liquidez cuando el volumen ejecutado acumulado en un nivel supera la profundidad inicialmente visible ($Q_{\text{exec}} > V_{\text{vis}}$).
//!   3. `AharonovBergmannLebowitzEngine`: Implementación del **Formalismo de Dos Vectores de Estado (TSVF)** de Aharonov-Bergmann-Lebowitz (1964) y **Valores Débiles (Weak Values)** de Aharonov-Albert-Vaidman (1988) $A_w = \frac{\langle \Phi_{\text{post}} | \hat{A} | \Psi_{\text{pre}} \rangle}{\langle \Phi_{\text{post}} | \Psi_{\text{pre}} \rangle}$, junto con la distribución de probabilidad condicional simétrica en el tiempo $P_{\text{ABL}}(a_k)$ ($\sum_k P_{\text{ABL}}(a_k) \equiv 1.0$).
//! - **POR QUÉ:** Las rupturas impulsadas sólo por derivados apalancados sin respaldo en Spot suelen revertir violentamente; los muros Iceberg absorben el flujo agresivo sin desplazar el precio; y los valores esperados ordinarios $\langle \Psi | \hat{A} | \Psi \rangle$ no condicionan sobre el atractor futuro post-seleccionado $\langle \Phi_{\text{post}}|$.
//! - **PARA QUÉ:** Confirmar que el flujo direccional provenga de acumulación Spot orgánica, detectar absorción institucional oculta L3 antes de chocar contra un muro Iceberg, y amplificar precursores microestructurales sutiles mediante amplificación anómala de valor débil ($|A_w| > \|\hat{A}\|_{\text{op}}$).
//! - **CÓMO:** Correlación cruzada asimétrica a 8 retardos en pila, actualización Bayesiana log-odds con razón de recarga, y álgebra bilineal en espacio de Hilbert de 4 estados con regularización de ortogonalidad de Tikhonov.
//! - **CUÁNDO:** En cada actualización sincronizada Spot/Futuros y evento de ejecución sobre el mejor nivel L2/L3.
//! - **DÓNDE:** `crates/feature-engine/src/spot_futures_abl_iceberg.rs`.
//! - **QUIÉN:** `SpotFuturesLeadLagEngine`, `IcebergHunterL3Engine`, `AharonovBergmannLebowitzEngine`, `SpotFuturesAblIcebergMasterEngine`.

/// Número de muestras en ventana corta en pila para el estimador de liderazgo Spot-Futuros.
pub const LEAD_LAG_WINDOW: usize = 8;

/// Dimensión del espacio de Hilbert para el formalismo de dos vectores de estado ABL/AAV.
pub const ABL_STATE_DIM: usize = 4;

/// Clasificación del régimen conjunto Spot-Futuros, Iceberg L3 y Valor Débil Cuántico ABL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpotFuturesAblRegime {
    /// Impulso orgánico liderado por el mercado Spot con amplificación constructiva ABL.
    SpotDrivenOrganicImpulse,
    /// Absorción institucional oculta detectada por el cazador Iceberg L3 ($P(\text{Iceberg}) \ge 0.75$).
    HiddenIcebergAbsorptionWall,
    /// Divergencia frágil liderada por derivados apalancados sin soporte Spot (riesgo de *Long/Short Squeeze*).
    PerpFrothSqueezeFragility,
}

/// Diagnóstico de causalidad dirigida Spot $\to$ Futuros y tensión de base.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpotFuturesLeadLagReport {
    /// Correlación adelantada $\text{Corr}(\Delta S_{t-1}, \Delta F_t) \in [-1, 1]$ (Spot lidera a Futuros).
    pub spot_leads_futures_corr: f64,
    /// Correlación inversa $\text{Corr}(\Delta F_{t-1}, \Delta S_t) \in [-1, 1]$ (Futuros lidera a Spot).
    pub futures_leads_spot_corr: f64,
    /// Índice neto de liderazgo causal $\Delta \mathcal{L} = \text{Corr}(S \to F) - \text{Corr}(F \to S) \in [-2, 2]$.
    pub net_spot_leadership: f64,
    /// Base relativa instantánea en puntos básicos $\frac{F_t - S_t}{S_t} \times 10^4$.
    pub relative_basis_bps: f64,
}

/// Motor de Causalidad Dirigida Spot $\to$ Futuros.
#[derive(Debug, Clone, Copy, Default)]
pub struct SpotFuturesLeadLagEngine;

impl SpotFuturesLeadLagEngine {
    /// Evalúa el liderazgo causal entre retornos Spot `spot_returns` y retornos de Futuros `futures_returns`.
    #[inline]
    pub fn evaluate_lead_lag(
        &self,
        spot_returns: &[f64; LEAD_LAG_WINDOW],
        futures_returns: &[f64; LEAD_LAG_WINDOW],
        spot_price: f64,
        futures_price: f64,
    ) -> SpotFuturesLeadLagReport {
        let mut s = [0.0; LEAD_LAG_WINDOW];
        let mut f = [0.0; LEAD_LAG_WINDOW];
        let mut i = 0;
        while i < LEAD_LAG_WINDOW {
            s[i] = if spot_returns[i].is_finite() {
                spot_returns[i].clamp(-1.0, 1.0)
            } else {
                0.0
            };
            f[i] = if futures_returns[i].is_finite() {
                futures_returns[i].clamp(-1.0, 1.0)
            } else {
                0.0
            };
            i += 1;
        }

        // Correlación Spot(t-1) -> Futures(t) vs Futures(t-1) -> Spot(t) sobre N-1 pares
        let mut cov_sf = 0.0;
        let mut cov_fs = 0.0;
        let mut var_s_prev = 0.0;
        let mut var_f_curr = 0.0;
        let mut var_f_prev = 0.0;
        let mut var_s_curr = 0.0;

        let mut k = 0;
        while k + 1 < LEAD_LAG_WINDOW {
            let s_prev = s[k];
            let s_curr = s[k + 1];
            let f_prev = f[k];
            let f_curr = f[k + 1];

            cov_sf += s_prev * f_curr;
            cov_fs += f_prev * s_curr;
            var_s_prev += s_prev * s_prev;
            var_f_curr += f_curr * f_curr;
            var_f_prev += f_prev * f_prev;
            var_s_curr += s_curr * s_curr;
            k += 1;
        }

        let denom_sf = (var_s_prev * var_f_curr).sqrt();
        let denom_fs = (var_f_prev * var_s_curr).sqrt();

        let corr_sf = if denom_sf > 1e-12 {
            (cov_sf / denom_sf).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let corr_fs = if denom_fs > 1e-12 {
            (cov_fs / denom_fs).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let basis_bps = if spot_price.is_finite() && futures_price.is_finite() && spot_price > 1e-9
        {
            (((futures_price - spot_price) / spot_price) * 1.0e4).clamp(-1.0e4, 1.0e4)
        } else {
            0.0
        };

        SpotFuturesLeadLagReport {
            spot_leads_futures_corr: corr_sf,
            futures_leads_spot_corr: corr_fs,
            net_spot_leadership: corr_sf - corr_fs,
            relative_basis_bps: basis_bps,
        }
    }
}

/// Diagnóstico del Cazador Bayesiano de Liquidez Oculta L3 (*Iceberg Hunter*).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IcebergL3Report {
    /// Razón de exceso de ejecución sobre liquidez inicialmente visible $R_{\text{iceberg}} = \frac{\max(0, Q_{\text{exec}} - V_{\text{vis}})}{V_{\text{vis}}} \ge 0$.
    pub replenishment_excess_ratio: f64,
    /// Probabilidad posterior Bayesiana de orden Iceberg activa $P(\text{Iceberg} \mid \mathcal{D}) \in [0, 1]$.
    pub iceberg_posterior_prob: f64,
    /// Reserva oculta estimada restante $\hat{V}_{\text{hidden}} \ge 0$.
    pub estimated_hidden_reserve: f64,
    /// Número de ciclos de recarga sintética detectados en el nivel de precio.
    pub detected_refill_count: u32,
}

/// Motor Bayesiano de Detección de Órdenes Iceberg L3.
#[derive(Debug, Clone, Copy)]
pub struct IcebergHunterL3Engine {
    /// Probabilidad a priori de presencia de orden Iceberg en niveles institucionales $P_0 \in (0, 1)$.
    pub prior_probability: f64,
    /// Sensibilidad log-verosimilitud por unidad de exceso de recarga $\beta_{\text{LR}} > 0$.
    pub log_likelihood_sensitivity: f64,
}

impl Default for IcebergHunterL3Engine {
    #[inline]
    fn default() -> Self {
        Self {
            prior_probability: 0.15,
            log_likelihood_sensitivity: 2.5,
        }
    }
}

impl IcebergHunterL3Engine {
    /// Evalúa la probabilidad posterior de una orden Iceberg oculta dado el tamaño inicial visible `initial_visible_qty`,
    /// el volumen agresivo total ejecutado en el mismo precio `cumulative_executed_qty` y la cantidad aún visible `current_visible_qty`.
    #[inline]
    pub fn evaluate_level(
        &self,
        initial_visible_qty: f64,
        cumulative_executed_qty: f64,
        current_visible_qty: f64,
    ) -> IcebergL3Report {
        let v_init = if initial_visible_qty.is_finite() && initial_visible_qty > 1e-9 {
            initial_visible_qty
        } else {
            return IcebergL3Report {
                replenishment_excess_ratio: 0.0,
                iceberg_posterior_prob: 0.0,
                estimated_hidden_reserve: 0.0,
                detected_refill_count: 0,
            };
        };
        let q_exec = if cumulative_executed_qty.is_finite() && cumulative_executed_qty > 0.0 {
            cumulative_executed_qty
        } else {
            0.0
        };
        let v_curr = if current_visible_qty.is_finite() && current_visible_qty > 0.0 {
            current_visible_qty
        } else {
            0.0
        };

        // Cantidad total que ha debido suministrarse en este nivel sin que colapse:
        // Total suministrado = max(0, q_exec + v_curr - v_init)
        let absorbed_hidden = (q_exec + v_curr - v_init).max(0.0);
        let excess_ratio = (absorbed_hidden / v_init).min(1.0e4);
        let refills = (excess_ratio.floor() as u64).min(u32::MAX as u64) as u32;

        let p0 = if self.prior_probability.is_finite() {
            self.prior_probability.clamp(0.01, 0.99)
        } else {
            0.15
        };
        let beta = if self.log_likelihood_sensitivity.is_finite()
            && self.log_likelihood_sensitivity > 0.0
        {
            self.log_likelihood_sensitivity.min(20.0)
        } else {
            2.5
        };

        let prior_log_odds = (p0 / (1.0 - p0)).ln();
        // Si no hay recarga oculta, la evidencia disminuye el log-odds ligeramente; si absorbed_hidden > 0, crece linealmente con excess_ratio
        let log_lr = if excess_ratio > 0.0 {
            beta * excess_ratio
        } else {
            -0.5
        };
        let post_log_odds = (prior_log_odds + log_lr).clamp(-20.0, 20.0);
        let posterior = 1.0 / (1.0 + (-post_log_odds).exp());

        // Estimación conservadora de la reserva oculta restante proporcional a la punta visible y la tasa de recarga
        let est_hidden = if posterior > 0.50 {
            v_curr * (1.0 + excess_ratio) * posterior
        } else {
            0.0
        };

        IcebergL3Report {
            replenishment_excess_ratio: excess_ratio,
            iceberg_posterior_prob: posterior.clamp(0.0, 1.0),
            estimated_hidden_reserve: if est_hidden.is_finite() {
                est_hidden
            } else {
                0.0
            },
            detected_refill_count: refills,
        }
    }
}

/// Diagnóstico del Formalismo Cuántico de Dos Vectores de Estado de Aharonov-Bergmann-Lebowitz (ABL) y Valores Débiles (AAV).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AblWeakValueReport {
    /// Valor esperado estándar pre-seleccionado $\langle \Psi_{\text{pre}} | \hat{A} | \Psi_{\text{pre}} \rangle$.
    pub standard_expectation_value: f64,
    /// Valor débil cuántico de Aharonov-Albert-Vaidman $A_w = \frac{\langle \Phi_{\text{post}} | \hat{A} | \Psi_{\text{pre}} \rangle}{\langle \Phi_{\text{post}} | \Psi_{\text{pre}} \rangle}$.
    pub weak_value_aw: f64,
    /// Solapamiento entre estado pre-seleccionado y post-seleccionado $|\langle \Phi_{\text{post}} | \Psi_{\text{pre}} \rangle| \in [0, 1]$.
    pub pre_post_overlap: f64,
    /// Probabilidades condicionales simétricas en el tiempo de la regla ABL $P_{\text{ABL}}(a_k)$ ($\sum_k P_{\text{ABL}}(a_k) \equiv 1.0$).
    pub abl_probabilities: [f64; ABL_STATE_DIM],
    /// Indica si $A_w$ presenta amplificación cuántica anómala fuera del espectro clásico de autovalores $[\lambda_{\min}, \lambda_{\max}]$.
    pub is_anomalous_amplification: bool,
}

/// Motor del Formalismo de Dos Vectores de Estado (Aharonov-Bergmann-Lebowitz) y Valores Débiles.
#[derive(Debug, Clone, Copy)]
pub struct AharonovBergmannLebowitzEngine {
    /// Regularización de Tikhonov $\delta > 0$ para el denominador $\langle \Phi_{\text{post}} | \Psi_{\text{pre}} \rangle$ cuando los estados son casi ortogonales.
    pub overlap_regularization: f64,
}

impl Default for AharonovBergmannLebowitzEngine {
    #[inline]
    fn default() -> Self {
        Self {
            overlap_regularization: 0.05,
        }
    }
}

impl AharonovBergmannLebowitzEngine {
    #[inline]
    fn normalize_state(raw: &[f64; ABL_STATE_DIM]) -> [f64; ABL_STATE_DIM] {
        let mut clean = [0.0; ABL_STATE_DIM];
        let mut norm_sq = 0.0;
        let mut i = 0;
        while i < ABL_STATE_DIM {
            let v = if raw[i].is_finite() {
                raw[i].clamp(-1e4, 1e4)
            } else {
                0.0
            };
            clean[i] = v;
            norm_sq += v * v;
            i += 1;
        }
        if norm_sq > 1e-18 && norm_sq.is_finite() {
            let inv = 1.0 / norm_sq.sqrt();
            let mut k = 0;
            while k < ABL_STATE_DIM {
                clean[k] *= inv;
                k += 1;
            }
            clean
        } else {
            [0.5; ABL_STATE_DIM]
        }
    }

    /// Evalúa el valor débil $A_w$ y la regla de probabilidad simétrica en el tiempo de Aharonov-Bergmann-Lebowitz
    /// para un observable diagonal en la base propia $\hat{A} = \text{diag}(a_0, a_1, a_2, a_3)$.
    #[inline]
    pub fn evaluate_weak_measurement(
        &self,
        pre_state_psi: &[f64; ABL_STATE_DIM],
        post_state_phi: &[f64; ABL_STATE_DIM],
        observable_eigenvalues: &[f64; ABL_STATE_DIM],
    ) -> AblWeakValueReport {
        let psi = Self::normalize_state(pre_state_psi);
        let phi = Self::normalize_state(post_state_phi);
        let reg = if self.overlap_regularization.is_finite() && self.overlap_regularization > 1e-6 {
            self.overlap_regularization.min(0.5)
        } else {
            0.05
        };

        let mut a_vals = [0.0; ABL_STATE_DIM];
        let mut a_min = f64::INFINITY;
        let mut a_max = f64::NEG_INFINITY;
        let mut i = 0;
        while i < ABL_STATE_DIM {
            let val = if observable_eigenvalues[i].is_finite() {
                observable_eigenvalues[i].clamp(-1e4, 1e4)
            } else {
                0.0
            };
            a_vals[i] = val;
            if val < a_min {
                a_min = val;
            }
            if val > a_max {
                a_max = val;
            }
            i += 1;
        }

        // 1. Producto interno <Phi | Psi>, <Psi | A | Psi> y elemento de matriz de transición <Phi | A | Psi>
        let mut overlap = 0.0;
        let mut std_exp = 0.0;
        let mut matrix_elem = 0.0;
        let mut abl_weights = [0.0; ABL_STATE_DIM];
        let mut abl_sum = 0.0;

        let mut k = 0;
        while k < ABL_STATE_DIM {
            let prod_k = phi[k] * psi[k];
            overlap += prod_k;
            std_exp += psi[k] * psi[k] * a_vals[k];
            matrix_elem += prod_k * a_vals[k];

            // Regla ABL: w_k = |<Phi|a_k>|^2 * |<a_k|Psi>|^2 = (phi_k * psi_k)^2
            let w_k = prod_k * prod_k;
            abl_weights[k] = w_k;
            abl_sum += w_k;
            k += 1;
        }

        // Regularización suave de Tikhonov: overlap / (overlap^2 + reg^2) evita división por cero y preserva el signo
        let denom = overlap * overlap + reg * reg;
        let aw = (matrix_elem * overlap) / denom;

        let mut abl_probs = [0.25; ABL_STATE_DIM];
        if abl_sum > 1e-18 && abl_sum.is_finite() {
            let inv_sum = 1.0 / abl_sum;
            let mut m = 0;
            while m < ABL_STATE_DIM {
                abl_probs[m] = abl_weights[m] * inv_sum;
                m += 1;
            }
        }

        let is_anomalous = aw < a_min - 1e-9 || aw > a_max + 1e-9;

        AblWeakValueReport {
            standard_expectation_value: if std_exp.is_finite() { std_exp } else { 0.0 },
            weak_value_aw: if aw.is_finite() { aw } else { 0.0 },
            pre_post_overlap: overlap.abs().clamp(0.0, 1.0),
            abl_probabilities: abl_probs,
            is_anomalous_amplification: is_anomalous,
        }
    }
}

/// Reporte unificado de la Fase 19 (Spot-Futures Lead-Lag + Iceberg Hunter L3 + ABL Weak Values).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpotFuturesAblIcebergReport {
    pub lead_lag: SpotFuturesLeadLagReport,
    pub iceberg: IcebergL3Report,
    pub abl: AblWeakValueReport,
    pub unified_conviction: f64,
    pub regime: SpotFuturesAblRegime,
}

/// Motor maestro de Fase 19 que culmina la arquitectura de 20 fases (0..=19).
#[derive(Debug, Clone, Copy, Default)]
pub struct SpotFuturesAblIcebergMasterEngine {
    pub lead_lag_engine: SpotFuturesLeadLagEngine,
    pub iceberg_engine: IcebergHunterL3Engine,
    pub abl_engine: AharonovBergmannLebowitzEngine,
}

impl SpotFuturesAblIcebergMasterEngine {
    /// Evalúa un ciclo completo de Fase 19.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn evaluate_cycle(
        &self,
        spot_returns: &[f64; LEAD_LAG_WINDOW],
        futures_returns: &[f64; LEAD_LAG_WINDOW],
        spot_price: f64,
        futures_price: f64,
        initial_visible_qty: f64,
        cumulative_executed_qty: f64,
        current_visible_qty: f64,
        pre_state_psi: &[f64; ABL_STATE_DIM],
        post_state_phi: &[f64; ABL_STATE_DIM],
        observable_eigenvalues: &[f64; ABL_STATE_DIM],
    ) -> SpotFuturesAblIcebergReport {
        let lead_lag = self.lead_lag_engine.evaluate_lead_lag(
            spot_returns,
            futures_returns,
            spot_price,
            futures_price,
        );
        let iceberg = self.iceberg_engine.evaluate_level(
            initial_visible_qty,
            cumulative_executed_qty,
            current_visible_qty,
        );
        let abl = self.abl_engine.evaluate_weak_measurement(
            pre_state_psi,
            post_state_phi,
            observable_eigenvalues,
        );

        let raw_conv = 0.50 * abl.weak_value_aw.tanh() + 0.35 * lead_lag.spot_leads_futures_corr
            - 0.15 * iceberg.iceberg_posterior_prob;
        let unified_conviction = if raw_conv.is_finite() {
            raw_conv.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let regime = if iceberg.iceberg_posterior_prob >= 0.75 {
            SpotFuturesAblRegime::HiddenIcebergAbsorptionWall
        } else if lead_lag.net_spot_leadership < -0.25 && lead_lag.relative_basis_bps.abs() > 25.0 {
            SpotFuturesAblRegime::PerpFrothSqueezeFragility
        } else {
            SpotFuturesAblRegime::SpotDrivenOrganicImpulse
        };

        SpotFuturesAblIcebergReport {
            lead_lag,
            iceberg,
            abl,
            unified_conviction,
            regime,
        }
    }
}
