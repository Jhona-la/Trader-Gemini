pub struct CorrelationGuardEngine;

impl CorrelationGuardEngine {
    /// Determines whether to veto an order based on directional correlation risk.
    pub fn is_correlation_vetoed(
        _engine_id: usize,
        _coin_id: usize,
        _is_long: bool,
        same_direction_active_count: usize,
    ) -> bool {
        if same_direction_active_count == 0 {
            return false;
        }
        // FIX #628: Acotar exponente para evitar desbordamiento numérico
        let safe_count = (same_direction_active_count as f64).min(20.0);
        let risk_score = safe_count.exp() / std::f64::consts::E.powi(3);
        risk_score > 1.0
    }

    /// Determines whether to veto based on capital-aware correlation cluster limits
    pub fn is_correlation_vetoed_dynamic(
        same_direction_active_count: usize,
        current_capital: f64,
        max_allowed_cluster: usize,
    ) -> bool {
        if same_direction_active_count == 0 {
            return false;
        }
        // FIX #704: Sanitización defensiva de capital a micro-cuenta ($13 USD) ante NaN
        let safe_capital = if current_capital.is_finite() && current_capital > 0.0 {
            current_capital
        } else {
            13.0
        };
        // Micro cuentas (< $30 USD) no deben abrir más de 2 posiciones en la misma dirección
        // para prevenir aniquilación correlacionada ante un Flash Crash de Bitcoin
        let limit = if safe_capital < 30.0 {
            2.min(max_allowed_cluster.max(1))
        } else {
            max_allowed_cluster.max(2)
        };
        same_direction_active_count >= limit
    }

    /// Determina si vetar una orden considerando de forma desacoplada los horizontes Scalp y Swing
    pub fn is_correlation_vetoed_by_horizon(
        is_scalp: bool,
        same_horizon_same_dir_count: usize,
        current_capital: f64,
        max_allowed_cluster: usize,
    ) -> bool {
        if same_horizon_same_dir_count == 0 {
            return false;
        }
        // FIX #704: Sanitización defensiva de capital a micro-cuenta ($13 USD) ante NaN
        let safe_capital = if current_capital.is_finite() && current_capital > 0.0 {
            current_capital
        } else {
            13.0
        };
        // Scalping tiene un turnover ultra-rápido (< 60s) por lo que el cluster permitido es independiente de Swing
        let limit = if is_scalp {
            if safe_capital < 30.0 {
                // Cuenta de $13 USD permite hasta 2 micro-scalps concurrentes
                2.min(max_allowed_cluster.max(2))
            } else {
                max_allowed_cluster.max(2)
            }
        } else {
            // Swing holding de horas: límite más estricto en micro cuentas (1 posición swing activa)
            if safe_capital < 30.0 {
                1.min(max_allowed_cluster.max(1))
            } else {
                (max_allowed_cluster / 2).max(1)
            }
        };
        same_horizon_same_dir_count >= limit
    }

    /// Determina si vetar una orden cuántica continua unificada
    pub fn is_continuous_correlation_vetoed(
        same_dir_count: usize,
        current_capital: f64,
        max_allowed_cluster: usize,
    ) -> bool {
        if same_dir_count == 0 {
            return false;
        }
        let safe_capital = if current_capital.is_finite() && current_capital > 0.0 {
            current_capital
        } else {
            13.0
        };
        let limit = if safe_capital < 30.0 {
            2.min(max_allowed_cluster.max(2))
        } else {
            max_allowed_cluster.max(2)
        };
        same_dir_count >= limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_micro_account_correlation_veto() {
        // Con $13 USD, el límite es 2 posiciones.
        assert!(!CorrelationGuardEngine::is_correlation_vetoed_dynamic(
            0, 13.0, 5
        ));
        assert!(!CorrelationGuardEngine::is_correlation_vetoed_dynamic(
            1, 13.0, 5
        ));
        // Al intentar la 2da/3ra en la misma dirección -> VETO activado
        assert!(CorrelationGuardEngine::is_correlation_vetoed_dynamic(
            2, 13.0, 5
        ));
    }

    #[test]
    fn test_large_account_correlation_veto() {
        // Con $1000 USD, se respeta max_allowed_cluster
        assert!(!CorrelationGuardEngine::is_correlation_vetoed_dynamic(
            2, 1000.0, 4
        ));
        assert!(!CorrelationGuardEngine::is_correlation_vetoed_dynamic(
            3, 1000.0, 4
        ));
        assert!(CorrelationGuardEngine::is_correlation_vetoed_dynamic(
            4, 1000.0, 4
        ));
    }

    #[test]
    fn test_horizon_decoupled_correlation_veto() {
        // Micro cuenta ($13 USD): Scalp permite hasta 2 micro-scalps
        assert!(!CorrelationGuardEngine::is_correlation_vetoed_by_horizon(
            true, 1, 13.0, 5
        ));
        assert!(CorrelationGuardEngine::is_correlation_vetoed_by_horizon(
            true, 2, 13.0, 5
        ));

        // Swing permite 1 en micro-cuenta
        assert!(!CorrelationGuardEngine::is_correlation_vetoed_by_horizon(
            false, 0, 13.0, 5
        ));
        assert!(CorrelationGuardEngine::is_correlation_vetoed_by_horizon(
            false, 1, 13.0, 5
        ));
    }

    #[test]
    fn test_correlation_guard_zero_and_nan_capital_immunity() {
        // Zero o NaN capital debe comportarse defensivamente como micro cuenta (falla seguro)
        assert!(CorrelationGuardEngine::is_correlation_vetoed_dynamic(
            2, 0.0, 10
        ));
        assert!(CorrelationGuardEngine::is_correlation_vetoed_dynamic(
            2,
            f64::NAN,
            10
        ));
    }
}
