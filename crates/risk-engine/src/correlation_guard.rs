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

    /// U-3 (MOTOR UNIVERSAL CONTINUO): `is_correlation_vetoed_by_horizon`
    /// (límites de clúster bifurcados por is_scalp) EXTIRPADA — el camino
    /// vivo es `is_continuous_correlation_vetoed`, continuo en capital y
    /// agnóstico de horizonte.
    /// Determina si vetar una orden cuántica continua unificada
    pub fn is_continuous_correlation_vetoed(
        same_dir_count: usize,
        current_capital: f64,
        min_notional: f64,
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
        // D-641 (completo): el límite de posiciones correlacionadas deja de
        // saltar en $30. Micro pleno ⇒ 2, como se diseñó; estándar ⇒ el cluster
        // genómico; entre ambos, interpolación redondeada al entero.
        let w = crate::capital_regime::micro_weight(safe_capital, min_notional);
        let standard = max_allowed_cluster.max(2) as f64;
        let limit = crate::capital_regime::lerp(standard, 2.0, w).round().max(2.0) as usize;
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
        // U-3: la variante bifurcada por horizonte fue extirpada; su semántica
        // viva (límites de clúster continuos en capital) la cubre
        // is_correlation_vetoed_dynamic — este test verifica que la micro
        // cuenta sigue permitiendo 2 posiciones concurrentes.
        assert!(!CorrelationGuardEngine::is_correlation_vetoed_dynamic(1, 13.0, 5));
        assert!(CorrelationGuardEngine::is_correlation_vetoed_dynamic(2, 13.0, 5));
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
