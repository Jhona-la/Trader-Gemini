mod types;
mod vecm_arbitrage;
mod multivariate_coint;

#[cfg(test)]
mod r4_ou_diagnostic {
    use super::multivariate_coint::MultivariateCointegrationEngine;
    use super::types::SignalType;
    use super::vecm_arbitrage::ContinuousOrnsteinUhlenbeckSde;

    fn demonstrate(second_ts: u64) {
        let prices = [0.1_f64.exp(), 1.0, 1.0, 1.0];
        let spread = prices[0].ln();
        let mut direct_sde = ContinuousOrnsteinUhlenbeckSde::new(0.1, 0.0, 0.01);
        assert_eq!(direct_sde.update(0.0, 1000).to_bits(), 0.0_f64.to_bits());
        let physical_z = direct_sde.update(spread, second_ts);
        assert_eq!(physical_z.to_bits(), 0.0_f64.to_bits());
        assert_eq!(direct_sde.count, 2);
        assert_eq!(direct_sde.last_ts_ms, second_ts);

        let mut engine = MultivariateCointegrationEngine::new([1.0, 0.0, 0.0, 0.0], 2.0)
            .with_continuous_ou();
        assert!(engine.update_and_evaluate(&[1.0; 4], 1000).is_none());
        let order = engine.update_and_evaluate(&prices, second_ts)
            .expect("diagnostic: current SDE mode falls through to legacy and emits");
        let embedded_sde = engine.physical_sde.as_ref().unwrap();
        assert_eq!(embedded_sde.count, 2);
        assert_eq!(embedded_sde.last_ts_ms, second_ts);
        assert_eq!(embedded_sde.last_value.to_bits(), spread.to_bits());
        assert_eq!(engine.count, 2);
        assert_eq!(order.signal, SignalType::Short);
        assert_eq!(order.expected_duration_ms, 0);
        let legacy_z = (spread - engine.mean_spread) / engine.var_spread.sqrt();
        assert!(legacy_z > 3.55 && legacy_z < 3.56);
        println!("DIAGNOSTIC second_ts={second_ts} physical_z={physical_z} sde_count={} sde_last_ts={} legacy_count={} legacy_z={legacy_z:.12} signal={:?} duration_ms={}",
                 embedded_sde.count, embedded_sde.last_ts_ms, engine.count,
                 order.signal, order.expected_duration_ms);
    }

    #[test]
    fn duplicate_timestamp_returns_zero_physical_z_but_legacy_short() { demonstrate(1000); }

    #[test]
    fn backward_timestamp_returns_zero_physical_z_but_legacy_short() { demonstrate(500); }
}
