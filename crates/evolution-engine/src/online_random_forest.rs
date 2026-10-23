use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::ensemble::random_forest_regressor::{
    RandomForestRegressor, RandomForestRegressorParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::{accuracy, mean_squared_error};
use crate::lock_seguro::{escribir, leer};
use std::sync::{Arc, RwLock};

/// Estructura de observación individual (Data Point) con paridad direccional de lado
#[derive(Clone, Debug)]
pub struct TradeObservation {
    pub obi: f64,
    pub vol_accel: f64,
    pub spread_bps: f64,
    pub atr_pct: f64,
    pub hurst_exp: f64,
    pub macro_momentum: f64,
    pub actual_pnl_pct: f64,
    pub was_profitable: bool,
    /// LC-F1-18 / LC-F1-67: Lado de la operación para proyectar la dirección real del mercado
    pub is_long: bool,
}

pub struct TrueOnlineRandomForest {
    pub observations: RwLock<Vec<TradeObservation>>,
    classifier: RwLock<Option<Arc<RandomForestClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>>>>>,
    regressor: RwLock<Option<Arc<RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>>>>,
    max_memory_size: usize,
    // Compatibilidad
    pub shadow_threshold_long: RwLock<f32>,
    pub shadow_threshold_short: RwLock<f32>,
    /// QO-E2a — exactitud del ÚLTIMO reentrenamiento: el gate de consumo
    /// (modulación de confianza en el core) la exige >0.55 antes de
    /// dejar opinar al forest. Sin esto, un forest degenerado modularía
    /// entradas con ruido.
    pub last_accuracy: RwLock<f64>,
    /// CL-18 — `true` sólo si el último reentrenamiento IDENTIFICÓ umbrales:
    /// la curva de PnL filtrado cambia con el umbral. Con un clasificador
    /// que sólo devuelve la clase (0/1) todos los umbrales de la rejilla
    /// filtran lo mismo y no hay nada que calibrar.
    thresholds_identified: RwLock<bool>,
    /// Ola Ω136: Motor analítico Newton GBDT en espacio de Hilbert L2 con Taylor de 2º orden y cero alocaciones
    pub gbdt: crate::online_newton_gbdt::OnlineNewtonGbdt,
}

impl TrueOnlineRandomForest {
    pub fn new(max_memory_size: usize) -> Self {
        Self {
            observations: RwLock::new(Vec::with_capacity(max_memory_size)),
            classifier: RwLock::new(None),
            regressor: RwLock::new(None),
            max_memory_size,
            shadow_threshold_long: RwLock::new(0.60),
            shadow_threshold_short: RwLock::new(0.60),
            last_accuracy: RwLock::new(0.0),
            thresholds_identified: RwLock::new(false),
            gbdt: crate::online_newton_gbdt::OnlineNewtonGbdt::new(max_memory_size),
        }
    }

    pub fn shadow_evaluate(&self, _ml_prob: f32, net_pnl_pct: f32, _reserved: f32, is_long: bool) {
        let obs = TradeObservation {
            obi: if is_long { 0.5 } else { -0.5 },
            vol_accel: 0.0,
            spread_bps: 1.0,
            atr_pct: 0.002,
            hurst_exp: 0.50,
            macro_momentum: 0.0,
            actual_pnl_pct: net_pnl_pct as f64,
            was_profitable: net_pnl_pct > 0.0,
            is_long,
        };
        self.feed_observation(obs);
        let features = [if is_long { 0.5 } else { -0.5 }, 0.0, 1.0, 0.002, 0.50, 0.0];
        self.gbdt.feed_observation(features, net_pnl_pct as f64, is_long);
    }

    /// Alimenta la observación con las 6 características microestructurales y macro reales observadas.
    /// Por compatibilidad con tests existentes, deduce el lado del signo del OBI (obi >= 0 => long).
    pub fn shadow_evaluate_with_features(&self, features: [f64; 6], net_pnl_pct: f64) {
        let is_long = features[0] >= 0.0;
        self.shadow_evaluate_with_features_dir(features, net_pnl_pct, is_long);
    }

    /// Alimenta la observación con las 6 características microestructurales e incorpora
    /// paridad explícita de lado (is_long), resolviendo de raíz LC-F1-18 y LC-F1-67.
    pub fn shadow_evaluate_with_features_dir(
        &self,
        features: [f64; 6],
        net_pnl_pct: f64,
        is_long: bool,
    ) {
        let obs = TradeObservation {
            obi: features[0],
            vol_accel: features[1],
            spread_bps: features[2],
            atr_pct: features[3],
            hurst_exp: features[4],
            macro_momentum: features[5],
            actual_pnl_pct: net_pnl_pct,
            was_profitable: net_pnl_pct > 0.0,
            is_long,
        };
        self.feed_observation(obs);
        self.gbdt.feed_observation(features, net_pnl_pct, is_long);
    }

    pub fn is_trained(&self) -> bool {
        leer(&self.classifier).is_some() || *leer(&self.gbdt.is_trained)
    }

    /// CL-18 — umbrales calibrados, o `None` si el último reentrenamiento no
    /// pudo identificarlos (ver `thresholds_identified`). Antes devolvía
    /// siempre un par, y con la calibración degenerada era (0,50; 0,50): el
    /// daemon lo escribía cada 500 ms en los genes del gate ML y lo fijaba
    /// en su lift mínimo.
    pub fn get_optimal_thresholds(&self) -> Option<(f32, f32)> {
        if !*leer(&self.thresholds_identified) {
            return None;
        }
        let th_long = *leer(&self.shadow_threshold_long);
        let th_short = *leer(&self.shadow_threshold_short);
        Some((th_long, th_short))
    }

    pub fn feed_observation(&self, obs: TradeObservation) {
        // FIX #673: Descartar observaciones con flotantes no finitos
        if !obs.obi.is_finite()
            || !obs.vol_accel.is_finite()
            || !obs.spread_bps.is_finite()
            || !obs.atr_pct.is_finite()
            || !obs.hurst_exp.is_finite()
            || !obs.macro_momentum.is_finite()
            || !obs.actual_pnl_pct.is_finite()
        {
            return;
        }

        let mut data = escribir(&self.observations);
        if data.len() >= self.max_memory_size {
            let drain_count = self.max_memory_size / 10;
            data.drain(0..drain_count);
        }
        data.push(obs);
    }

    pub fn retrain_models(&self) -> Result<(f64, f64), String> {
        let data = leer(&self.observations).clone();
        if data.len() < 50 {
            return Err("Insuficientes datos para entrenar el modelo (mínimo 50)".to_string());
        }

        let mut x_features = Vec::with_capacity(data.len() * 6);
        let mut y_class = Vec::with_capacity(data.len());
        let mut y_reg = Vec::with_capacity(data.len());

        for obs in &data {
            x_features.push(obs.obi);
            x_features.push(obs.vol_accel);
            x_features.push(obs.spread_bps);
            x_features.push(obs.atr_pct);
            x_features.push(obs.hurst_exp);
            x_features.push(obs.macro_momentum);

            // LC-F1-18 / LC-F1-67: Paridad direccional del mercado
            // Long: ganancia (>0) indica subida de precio; pérdida (<0) bajada.
            // Short: ganancia (>0) indica bajada de precio; pérdida (<0) subida.
            let directional_pnl = if obs.is_long {
                obs.actual_pnl_pct
            } else {
                -obs.actual_pnl_pct
            };
            let price_went_up = directional_pnl > 0.0;
            y_class.push(if price_went_up { 1 } else { 0 });
            y_reg.push(directional_pnl);
        }

        // FIX #720: Verificar variabilidad de clases para evitar particiones degeneradas en árboles
        let class_0 = y_class.iter().filter(|&&c| c == 0).count();
        let class_1 = y_class.iter().filter(|&&c| c == 1).count();
        if class_0 == 0 || class_1 == 0 {
            return Err(
                "Insuficiente variabilidad de clases (se requieren casos positivos y negativos)"
                    .to_string(),
            );
        }

        let x_matrix = DenseMatrix::new(data.len(), 6, x_features, false);

        let clf_params = RandomForestClassifierParameters::default()
            .with_m(3)
            .with_n_trees(20);
        let classifier = RandomForestClassifier::fit(&x_matrix, &y_class, clf_params)
            .map_err(|e| format!("Error entrenando Classifier: {:?}", e))?;

        let y_class_pred = classifier.predict(&x_matrix).unwrap_or_default();
        let acc = accuracy(&y_class, &y_class_pred);

        let reg_params = RandomForestRegressorParameters::default()
            .with_m(3)
            .with_n_trees(20);
        let regressor = RandomForestRegressor::fit(&x_matrix, &y_reg, reg_params)
            .map_err(|e| format!("Error entrenando Regressor: {:?}", e))?;

        let y_reg_pred = regressor.predict(&x_matrix).unwrap_or_default();
        let mse = mean_squared_error(&y_reg, &y_reg_pred);

        // Auto-calibración de umbrales óptimos maximizando PnL ponderado
        let mut best_th_long = 0.52_f32;
        let mut best_th_short = 0.48_f32;
        let mut best_profit_long = -1e9_f64;
        let mut best_profit_short = -1e9_f64;
        // CL-18: recorrido de cada curva de PnL sobre la rejilla de umbrales.
        let (mut min_l, mut max_l) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut min_s, mut max_s) = (f64::INFINITY, f64::NEG_INFINITY);

        // CERT-M8-C02: el loop ANTERIOR no usaba `th` dentro del cálculo
        // de profit — profit_l/profit_s eran idénticos para cada paso, así
        // que best_th_long siempre era 0.50 (primer paso) y el daemon
        // escribía 0.50/0.50 al arena (el gate ML más laxo) en el entorno
        // donde la autoevolución está armada. Ahora el umbral REALMENTE
        // filtra: sólo cuentan las observaciones cuya probabilidad
        // predicha supera `th`.
        for step in 50..=75 {
            let th = (step as f64) / 100.0;
            let mut profit_l = 0.0_f64;
            let mut profit_s = 0.0_f64;
            for (i, obs) in data.iter().enumerate() {
                let pnl = obs.actual_pnl_pct;
                // CL-18: smartcore 0.3.2 sólo expone `predict`, que devuelve
                // la CLASE (0/1), no una probabilidad: el comentario anterior
                // («probabilidad predicha») no era cierto y ningún `th` de la
                // rejilla discrimina. Se detecta abajo y no se publica nada.
                let prob_win = y_class_pred.get(i).copied().unwrap_or(0) as f64;
                if obs.obi > 0.0 && prob_win >= th {
                    profit_l += pnl;
                }
                if obs.obi < 0.0 && prob_win < (1.0 - th) {
                    profit_s += pnl;
                }
            }
            min_l = min_l.min(profit_l);
            max_l = max_l.max(profit_l);
            min_s = min_s.min(profit_s);
            max_s = max_s.max(profit_s);
            if profit_l > best_profit_long {
                best_profit_long = profit_l;
                best_th_long = th as f32;
            }
            if profit_s > best_profit_short {
                best_profit_short = profit_s;
                best_th_short = (1.0 - th) as f32;
            }
        }

        // CL-18: sólo se publica una calibración que el umbral puede mover. Una
        // curva plana (los 26 umbrales filtran lo mismo) no identifica nada:
        // el «óptimo» sería el primer paso de la rejilla, no un dato.
        let identificados = max_l > min_l && max_s > min_s;
        if identificados {
            *escribir(&self.shadow_threshold_long) = best_th_long.clamp(0.50, 0.75);
            *escribir(&self.shadow_threshold_short) = best_th_short.clamp(0.25, 0.50);
        }
        *escribir(&self.thresholds_identified) = identificados;

        // Ola Ω136: Reentrenar el motor analítico Newton GBDT en espacio Hilbert L2
        let _ = self.gbdt.retrain_models();

        *escribir(&self.classifier) = Some(Arc::new(classifier));
        *escribir(&self.regressor) = Some(Arc::new(regressor));

        {
            *escribir(&self.last_accuracy) = acc;
            Ok((acc, mse))
        }
    }

    pub fn predict_6d(&self, features: [f64; 6]) -> Option<(f64, f64)> {
        // FIX #673: Validar finitud de features antes de predecir
        for f in &features {
            if !f.is_finite() {
                return None;
            }
        }

        // Ola Ω136: Si el motor analítico Newton GBDT está entrenado, usar su predicción
        // continua zero-alloc en stack (<300 ns).
        if *leer(&self.gbdt.is_trained) {
            if let Some((prob, pnl)) = self.gbdt.predict_6d(features) {
                return Some((prob, pnl));
            }
        }

        let clf_guard = leer(&self.classifier);
        let reg_guard = leer(&self.regressor);

        if let (Some(clf), Some(reg)) = (clf_guard.as_ref(), reg_guard.as_ref()) {
            let x_eval = DenseMatrix::from_2d_vec(&vec![features.to_vec()]);

            let class_pred = match clf.predict(&x_eval) {
                Ok(preds) => preds.into_iter().next()? as f64,
                Err(_) => return None,
            };
            let expected_pnl = match reg.predict(&x_eval) {
                Ok(preds) => preds.into_iter().next()?,
                Err(_) => return None,
            };

            // F1-091 extensión (Ola Ω217): finitud estricta fail-closed.
            // Si el modelo no puede predecir o emite no-finitos, no se fabrican ceros ni probabilidades 0.5 sintéticas.
            if !class_pred.is_finite() || !expected_pnl.is_finite() {
                return None;
            }

            // FIX #867 & LC-F1-18: Probabilidad direccional continua suave de que el precio SUBA
            let prob_from_pnl = 1.0 / (1.0 + (-expected_pnl * 50.0).exp());
            let safe_prob = (0.6 * prob_from_pnl + 0.4 * class_pred).clamp(0.01, 0.99);

            Some((safe_prob, expected_pnl))
        } else {
            None
        }
    }
}
