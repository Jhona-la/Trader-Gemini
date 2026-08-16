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
        // En ML real, podríamos usar esta función para empujar datos dummy, pero lo haremos con feed_observation
        let obs = TradeObservation {
            obi: if is_long { 1.0 } else { -1.0 },
            vol_accel: 0.1,
            spread_bps: 2.0,
            atr_pct: 0.005,
            hurst_exp: 0.55,
            macro_momentum: 0.0,
            actual_pnl_pct: net_pnl_pct as f64,
            was_profitable: net_pnl_pct > 0.0,
        };
        self.feed_observation(obs);
    }

    pub fn get_optimal_thresholds(&self) -> (f32, f32) {
        let clf = self.classifier.read().unwrap();
        if clf.is_some() {
            // Si el modelo está entrenado, confiamos en thresholds fijos altos
            (0.55, 0.55)
        } else {
            (0.60, 0.60)
        }
    }

    pub fn feed_observation(&self, obs: TradeObservation) {
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

        *self.classifier.write().unwrap() = Some(Arc::new(classifier));
        *self.regressor.write().unwrap() = Some(Arc::new(regressor));

        Ok((acc, mse))
    }

    pub fn predict_6d(&self, features: [f64; 6]) -> Option<(f64, f64)> {
        let clf_guard = self.classifier.read().unwrap();
        let reg_guard = self.regressor.read().unwrap();

        if let (Some(clf), Some(reg)) = (clf_guard.as_ref(), reg_guard.as_ref()) {
            let x_eval = DenseMatrix::from_2d_vec(&vec![features.to_vec()]);

            let prob_win = clf
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

            Some((prob_win, expected_pnl))
        } else {
            None
        }
    }
}
