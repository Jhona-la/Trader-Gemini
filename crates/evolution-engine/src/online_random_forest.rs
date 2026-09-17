use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::ensemble::random_forest_regressor::{
    RandomForestRegressor, RandomForestRegressorParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::{accuracy, mean_squared_error};
use std::sync::{Arc, RwLock};

/// Estructura de observación individual (Data Point)
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
}

pub struct TrueOnlineRandomForest {
    pub observations: RwLock<Vec<TradeObservation>>,
    classifier: RwLock<Option<Arc<RandomForestClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>>>>>,
    regressor: RwLock<Option<Arc<RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>>>>,
    max_memory_size: usize,
    // Compatibilidad
    pub shadow_threshold_long: RwLock<f32>,
    pub shadow_threshold_short: RwLock<f32>,
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
        };
        self.feed_observation(obs);
    }

    /// Alimenta la observación con las 6 características microestructurales y macro reales observadas
    pub fn shadow_evaluate_with_features(&self, features: [f64; 6], net_pnl_pct: f64) {
        let obs = TradeObservation {
            obi: features[0],
            vol_accel: features[1],
            spread_bps: features[2],
            atr_pct: features[3],
            hurst_exp: features[4],
            macro_momentum: features[5],
            actual_pnl_pct: net_pnl_pct,
            was_profitable: net_pnl_pct > 0.0,
        };
        self.feed_observation(obs);
    }

    pub fn is_trained(&self) -> bool {
        self.classifier.read().unwrap().is_some()
    }

    pub fn get_optimal_thresholds(&self) -> (f32, f32) {
        let th_long = *self.shadow_threshold_long.read().unwrap();
        let th_short = *self.shadow_threshold_short.read().unwrap();
        (th_long, th_short)
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

        let mut data = self.observations.write().unwrap();
        if data.len() >= self.max_memory_size {
            let drain_count = self.max_memory_size / 10;
            data.drain(0..drain_count);
        }
        data.push(obs);
    }

    pub fn retrain_models(&self) -> Result<(f64, f64), String> {
        let data = self.observations.read().unwrap().clone();
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

            y_class.push(if obs.was_profitable { 1 } else { 0 });
            y_reg.push(obs.actual_pnl_pct);
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

        for step in 50..=75 {
            let th = (step as f64) / 100.0;
            let mut profit_l = 0.0_f64;
            let mut profit_s = 0.0_f64;
            for (i, obs) in data.iter().enumerate() {
                let pnl = obs.actual_pnl_pct;
                let is_predicted_win = y_class_pred.get(i).copied().unwrap_or(0) == 1
                    || y_reg_pred.get(i).copied().unwrap_or(0.0) > 0.0;
                if obs.obi > 0.0 && is_predicted_win {
                    profit_l += pnl;
                }
                if obs.obi < 0.0 && is_predicted_win {
                    profit_s += pnl;
                }
            }
            if profit_l > best_profit_long {
                best_profit_long = profit_l;
                best_th_long = th as f32;
            }
            if profit_s > best_profit_short {
                best_profit_short = profit_s;
                best_th_short = (1.0 - th) as f32;
            }
        }

        *self.shadow_threshold_long.write().unwrap() = best_th_long.clamp(0.50, 0.75);
        *self.shadow_threshold_short.write().unwrap() = best_th_short.clamp(0.25, 0.50);

        *self.classifier.write().unwrap() = Some(Arc::new(classifier));
        *self.regressor.write().unwrap() = Some(Arc::new(regressor));

        Ok((acc, mse))
    }

    pub fn predict_6d(&self, features: [f64; 6]) -> Option<(f64, f64)> {
        // FIX #673: Validar finitud de features antes de predecir
        for f in &features {
            if !f.is_finite() {
                return None;
            }
        }

        let clf_guard = self.classifier.read().unwrap();
        let reg_guard = self.regressor.read().unwrap();

        if let (Some(clf), Some(reg)) = (clf_guard.as_ref(), reg_guard.as_ref()) {
            let x_eval = DenseMatrix::from_2d_vec(&vec![features.to_vec()]);

            let class_pred = clf
                .predict(&x_eval)
                .unwrap_or_default()
                .into_iter()
                .next()
                .unwrap_or(0) as f64;
            let expected_pnl = reg
                .predict(&x_eval)
                .unwrap_or_default()
                .into_iter()
                .next()
                .unwrap_or(0.0);

            // FIX #867: Probabilidad continua suave derivada de sigmoid sobre el PnL esperado y clase
            let prob_from_pnl = 1.0 / (1.0 + (-expected_pnl * 50.0).exp());
            let safe_prob = if expected_pnl.is_finite() && class_pred.is_finite() {
                // Combinar probabilidad de regresión continua con clase discreta
                (0.6 * prob_from_pnl + 0.4 * class_pred).clamp(0.01, 0.99)
            } else if class_pred.is_finite() {
                class_pred.clamp(0.0, 1.0)
            } else {
                0.5
            };
            let safe_pnl = if expected_pnl.is_finite() {
                expected_pnl
            } else {
                0.0
            };

            Some((safe_prob, safe_pnl))
        } else {
            None
        }
    }
}
