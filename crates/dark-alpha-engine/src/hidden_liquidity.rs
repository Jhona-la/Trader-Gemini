//! # Hidden Liquidity & Block Absorption Engine (Kyle-Hasbrouck Microstructure & Conformal Gating)
//!
//! ## QUÉ
//! Detector no-paramétrico de absorción institucional y liquidez oculta (órdenes iceberg, bloques
//! institucionales TWAP/VWAP) basado en la teoría de microestructura de Kyle (1985) y Hasbrouck (1991).
//! Evalúa la compuerta conformal de riesgo para impedir tomas de posición en contra de absorciones
//! activas y modular el ancho de banda conformal.
//!
//! ## POR QUÉ
//! En los mercados de criptoactivos, los creadores de mercado institucionales ("ballenas") rara vez
//! colocan órdenes límite visibles masivas en el libro L2 para evitar slippage y front-running.
//! En su lugar, utilizan icebergs nativos o algoritmos de ejecución que absorben volumen continuo
//! sin desplazar el precio proporcionalmente ($|\Delta P| \approx 0$ ante gran volumen negociado).
//! Operar en contra de esta liquidez oculta resulta en liquidaciones y paradas violentas (*squeezes*).
//!
//! ## PARA QUÉ
//! 1. Detectar absorción oculta en Bid o Ask en nanosegundos (cero alocaciones dinámicas).
//! 2. Modular el factor de escala conformal $\hat{C}_{1-\alpha}(X)$ para estrechar o expandir las cotas de incertidumbre.
//! 3. Vetar o desescalar a sonda órdenes que colisionen frontalmente con un muro de absorción institucional.
//!
//! ## CÓMO
//! Se calcula el índice de absorción relativo al libro visible y al desplazamiento normalizado por ATR:
//! $$\text{AbsorptionIndex}_{\text{bid/ask}} = \frac{V_{\text{trade}}}{\text{Depth}_{\text{visible}} + \epsilon} \cdot \exp\left(-\frac{\gamma \cdot |\Delta P|}{\text{ATR} + \epsilon}\right)$$
//! La presión neta de iceberg se define como:
//! $$\Pi_{\text{iceberg}} = \frac{\text{Abs}_{\text{ask}} - \text{Abs}_{\text{bid}}}{\text{Abs}_{\text{ask}} + \text{Abs}_{\text{bid}} + \epsilon} \in [-1.0, 1.0]$$
//! Si la absorción supera el umbral crítico y la orden candidateada ataca en la dirección opuesta,
//! la compuerta conformal emite un veto o desescalada estricta.

use serde::{Deserialize, Serialize};

/// Parámetros configurables desacoplados para el detector de liquidez oculta.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HiddenLiquidityParams {
    /// Escala para el impacto de Kyle (lambda scale).
    pub kyle_lambda_scale: f64,
    /// Umbral mínimo de ratio de absorción para considerar presencia de bloque iceberg (ej. 1.25).
    pub absorption_ratio_threshold: f64,
    /// Coeficiente gamma de decaimiento por desplazamiento de precio (ej. 2.0).
    pub price_displacement_decay: f64,
    /// Volumen mínimo negociado para evaluar absorción (evita divisiones por cero en frío).
    pub min_trade_volume: f64,
    /// Factor de contracción de banda conformal cuando hay absorción alineada (ej. 0.70).
    pub conformal_shrink_factor: f64,
    /// Factor de expansión de incertidumbre conformal ante choque de liquidez (ej. 1.50).
    pub conformal_expansion_on_opposing: f64,
}

impl Default for HiddenLiquidityParams {
    fn default() -> Self {
        Self {
            kyle_lambda_scale: 1.0,
            absorption_ratio_threshold: 1.25,
            price_displacement_decay: 2.0,
            min_trade_volume: 0.01,
            conformal_shrink_factor: 0.70,
            conformal_expansion_on_opposing: 1.50,
        }
    }
}

impl HiddenLiquidityParams {
    /// Valida que los parámetros pertenezcan a dominios físicamente admisibles.
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.kyle_lambda_scale.is_finite() || self.kyle_lambda_scale <= 0.0 {
            return Err("kyle_lambda_scale debe ser finito y positivo");
        }
        if !self.absorption_ratio_threshold.is_finite() || self.absorption_ratio_threshold <= 0.0 {
            return Err("absorption_ratio_threshold debe ser finito y mayor que cero");
        }
        if !self.price_displacement_decay.is_finite() || self.price_displacement_decay < 0.0 {
            return Err("price_displacement_decay debe ser finito y no-negativo");
        }
        if !self.min_trade_volume.is_finite() || self.min_trade_volume < 0.0 {
            return Err("min_trade_volume debe ser finito y no-negativo");
        }
        if !self.conformal_shrink_factor.is_finite()
            || self.conformal_shrink_factor <= 0.0
            || self.conformal_shrink_factor > 1.0
        {
            return Err("conformal_shrink_factor debe pertenecer a (0.0, 1.0]");
        }
        if !self.conformal_expansion_on_opposing.is_finite()
            || self.conformal_expansion_on_opposing < 1.0
        {
            return Err("conformal_expansion_on_opposing debe ser >= 1.0");
        }
        Ok(())
    }
}

/// Estado instantáneo de absorción de bloques y liquidez oculta.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BlockAbsorptionState {
    /// Ratio de absorción de compras en Ask (institucionales absorbiendo oferta).
    pub ask_absorption_ratio: f64,
    /// Ratio de absorción de ventas en Bid (institucionales absorbiendo demanda).
    pub bid_absorption_ratio: f64,
    /// Presión neta de iceberg en [-1.0, 1.0].
    /// +1.0 = acumulación agresiva en ask; -1.0 = distribución masiva en bid.
    pub net_iceberg_pressure: f64,
    /// Indica si se detecta un bloque iceberg activo sobre el umbral.
    pub is_iceberg_active: bool,
    /// Factor multiplicativo recomendado para la banda conformal.
    pub conformal_band_scale: f64,
}

/// Veredicto de la compuerta conformal de riesgo ante liquidez oculta.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConformalGateVerdict {
    /// Orden admitida con modulación de confianza y factor de banda conformal.
    Admitted {
        confidence_multiplier: f64,
        band_scale: f64,
    },
    /// Veto conformal: intento colisiona directamente contra un bloque de absorción masivo.
    OpposingAbsorptionVeto {
        reason: &'static str,
    },
    /// Desescalada a tamaño sonda: riesgo elevado de squeeze, acotar tamaño de lote.
    ProbeOnlyDeescalation {
        max_lot_scale: f64,
        band_scale: f64,
    },
}

/// Motor de detección de liquidez oculta y compuerta conformal.
#[derive(Debug, Clone, PartialEq)]
pub struct HiddenLiquidityEngine {
    params: HiddenLiquidityParams,
}

impl Default for HiddenLiquidityEngine {
    fn default() -> Self {
        Self::new(HiddenLiquidityParams::default())
    }
}

impl HiddenLiquidityEngine {
    /// Crea un nuevo motor con parámetros validados.
    pub fn new(params: HiddenLiquidityParams) -> Self {
        if params.validate().is_err() {
            Self {
                params: HiddenLiquidityParams::default(),
            }
        } else {
            Self { params }
        }
    }

    /// Retorna una referencia a los parámetros activos.
    pub fn params(&self) -> &HiddenLiquidityParams {
        &self.params
    }

    /// Actualiza los parámetros en caliente garantizando invariantes.
    pub fn update_params(&mut self, new_params: HiddenLiquidityParams) -> Result<(), &'static str> {
        new_params.validate()?;
        self.params = new_params;
        Ok(())
    }

    /// Computa la absorción instantánea de bloques a partir de volumen, profundidad y movimiento de precio.
    ///
    /// # Parámetros
    /// - `trade_volume`: Volumen total negociado en el intervalo.
    /// - `bid_depth`: Profundidad acumulada visible en el top de bids.
    /// - `ask_depth`: Profundidad acumulada visible en el top de asks.
    /// - `delta_price`: Desplazamiento del precio en el intervalo ($P_t - P_{t-1}$).
    /// - `safe_atr`: Volatilidad promedio segura (ATR > 0).
    pub fn compute_absorption(
        &self,
        trade_volume: f64,
        bid_depth: f64,
        ask_depth: f64,
        delta_price: f64,
        safe_atr: f64,
    ) -> BlockAbsorptionState {
        const EPS: f64 = 1e-8;

        if !trade_volume.is_finite()
            || !bid_depth.is_finite()
            || !ask_depth.is_finite()
            || !delta_price.is_finite()
            || !safe_atr.is_finite()
            || trade_volume < self.params.min_trade_volume
            || safe_atr <= EPS
        {
            return BlockAbsorptionState {
                ask_absorption_ratio: 0.0,
                bid_absorption_ratio: 0.0,
                net_iceberg_pressure: 0.0,
                is_iceberg_active: false,
                conformal_band_scale: 1.0,
            };
        }

        let abs_dp = delta_price.abs();
        let displacement_ratio = (self.params.price_displacement_decay * abs_dp) / (safe_atr + EPS);
        let dampening = (-displacement_ratio).exp();

        // Absorción en Ask: volumen ejecutado absorbido sin subir el precio correspondientemente
        let ask_depth_safe = ask_depth.max(EPS);
        let ask_absorption = (trade_volume / ask_depth_safe) * dampening;

        // Absorción en Bid: volumen ejecutado absorbido sin bajar el precio correspondientemente
        let bid_depth_safe = bid_depth.max(EPS);
        let bid_absorption = (trade_volume / bid_depth_safe) * dampening;

        // Diferencial direccional según el signo de delta_price
        // Si el precio subió levemente mientras se absorbía en ask: presión compradora neta
        // Si el precio bajó levemente mientras se absorbía en bid: presión vendedora neta
        let (eff_ask_abs, eff_bid_abs) = if delta_price >= 0.0 {
            (ask_absorption, bid_absorption * 0.5)
        } else {
            (ask_absorption * 0.5, bid_absorption)
        };

        let sum_abs = eff_ask_abs + eff_bid_abs;
        let net_pressure = if sum_abs > EPS {
            ((eff_ask_abs - eff_bid_abs) / sum_abs).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let is_active = eff_ask_abs >= self.params.absorption_ratio_threshold
            || eff_bid_abs >= self.params.absorption_ratio_threshold;

        let band_scale = if is_active {
            // Con absorción activa, la volatilidad local está comprimida; la banda conformal se ajusta
            self.params.conformal_shrink_factor
        } else {
            1.0
        };

        BlockAbsorptionState {
            ask_absorption_ratio: eff_ask_abs,
            bid_absorption_ratio: eff_bid_abs,
            net_iceberg_pressure: net_pressure,
            is_iceberg_active: is_active,
            conformal_band_scale: band_scale,
        }
    }

    /// Evalúa la compuerta conformal de riesgo ante la dirección intencionada de la orden.
    ///
    /// # Parámetros
    /// - `intent_direction`: Dirección de la orden (+1.0 = Long, -1.0 = Short).
    /// - `state`: Estado instantáneo de absorción de bloques.
    pub fn evaluate_conformal_gate(
        &self,
        intent_direction: f64,
        state: &BlockAbsorptionState,
    ) -> ConformalGateVerdict {
        if !intent_direction.is_finite() || intent_direction.abs() < 0.1 || !state.is_iceberg_active {
            return ConformalGateVerdict::Admitted {
                confidence_multiplier: 1.0,
                band_scale: state.conformal_band_scale,
            };
        }

        let dir_sign = intent_direction.signum();

        // Si intentamos ir LONG (dir_sign > 0) pero hay fuerte absorción de distribución en BID (net_pressure < -0.30)
        // O si intentamos ir SHORT (dir_sign < 0) pero hay fuerte absorción de acumulación en ASK (net_pressure > 0.30)
        let is_opposing = (dir_sign > 0.0 && state.net_iceberg_pressure < -0.30)
            || (dir_sign < 0.0 && state.net_iceberg_pressure > 0.30);

        if is_opposing {
            let severity = state.net_iceberg_pressure.abs();
            if severity > 0.70 {
                // Veto total ante colisión frontal extrema contra muro institucional
                ConformalGateVerdict::OpposingAbsorptionVeto {
                    reason: if dir_sign > 0.0 {
                        "VETO_CONFORMAL_ICEBERG_DISTRIBUCION_MASIVA"
                    } else {
                        "VETO_CONFORMAL_ICEBERG_ACUMULACION_MASIVA"
                    },
                }
            } else {
                // Desescalada preventiva a modo sonda con ensanchamiento de incertidumbre
                ConformalGateVerdict::ProbeOnlyDeescalation {
                    max_lot_scale: 0.25,
                    band_scale: self.params.conformal_expansion_on_opposing,
                }
            }
        } else {
            // Orden alineada con el flujo de absorción institucional: premiar confianza
            let alignment_bonus = 1.0 + 0.20 * (dir_sign * state.net_iceberg_pressure).max(0.0);
            ConformalGateVerdict::Admitted {
                confidence_multiplier: alignment_bonus.clamp(1.0, 1.25),
                band_scale: state.conformal_band_scale,
            }
        }
    }
}
