use smartcore::ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::time::Instant;

pub struct QuantumRandomForest {
    model: Option<RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>>,
    pub is_trained: bool,
}

impl Default for QuantumRandomForest {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantumRandomForest {
    pub fn new() -> Self {
        Self {
            model: None,
            is_trained: false,
        }
    }

    /// Entrena el Random Forest Cuántico usando features puros y labels direccionales (1=Long, -1=Short, 0=Flat).
    /// El proceso se ejecuta in-memory sin dependencias Python.
    pub fn fit(&mut self, features: Vec<Vec<f64>>, targets: Vec<f64>) -> Result<(), String> {
        if features.is_empty() || targets.is_empty() || features.len() != targets.len() {
            return Err("Datos de entrenamiento inválidos o vacíos.".to_string());
        }

        let start = Instant::now();
        let matrix = DenseMatrix::from_2d_vec(&features).unwrap();
        
        let params = RandomForestRegressorParameters::default()
            .with_m(3) // Features at each split
            .with_n_trees(50) // Optimize for latency vs accuracy
            .with_max_depth(10); // Prevent overfitting

        let model = RandomForestRegressor::fit(&matrix, &targets, params)
            .map_err(|e| format!("Fallo al entrenar RF: {:?}", e))?;

        self.model = Some(model);
        self.is_trained = true;

        println!("⚡ [Dark Alpha] Random Forest entrenado nativamente en {:?}", start.elapsed());
        Ok(())
    }

    /// Inferencia en el Hot Path. Retorna la dirección predecida y una señal cruda.
    /// Operación en nanosegundos (O(profundidad_arbol * num_arboles)).
    #[inline(always)]
    pub fn predict_nanos(&self, features: &[f64]) -> f64 {
        if let Some(model) = &self.model {
            let matrix = DenseMatrix::from_2d_vec(&vec![features.to_vec()]).unwrap();
            // Predict returns a vector with 1 element for 1 sample
            if let Ok(predictions) = model.predict(&matrix) {
                return predictions[0];
            }
        }
        0.0 // Flat if not trained or error
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_rf_training_and_inference() {
        let mut qrf = QuantumRandomForest::new();
        
        let features = vec![
            vec![1.0, 0.5, 0.1], 
            vec![-1.0, -0.5, 0.1], 
            vec![1.2, 0.6, 0.15], 
            vec![-1.1, -0.6, 0.12], 
        ];
        
        let targets = vec![1.0, -1.0, 1.0, -1.0];
        
        let train_result = qrf.fit(features, targets);
        assert!(train_result.is_ok());
        assert!(qrf.is_trained);

        let pred_long = qrf.predict_nanos(&[1.0, 0.5, 0.1]);
        assert!(pred_long > 0.0);
        
        let pred_short = qrf.predict_nanos(&[-1.0, -0.5, 0.1]);
        assert!(pred_short < 0.0);
    }
}
