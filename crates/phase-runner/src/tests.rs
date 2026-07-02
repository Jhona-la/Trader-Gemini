#[cfg(test)]
mod tests {
    use crate::{AdaptiveTimer, Phase};
    use std::time::Duration;

    #[test]
    fn test_adaptive_timer_init() {
        let timer = AdaptiveTimer::new(5000, 8000);
        assert_eq!(timer.base_interval, Duration::from_millis(5000));
        assert_eq!(timer.max_memory_mb, 8000);
    }

    #[test]
    fn test_phase_transitions() {
        assert_eq!(Phase::Alpha.next(), Phase::Beta);
        assert_eq!(Phase::Beta.next(), Phase::Gamma);
        assert_eq!(Phase::Gamma.next(), Phase::Delta);
        assert_eq!(Phase::Delta.next(), Phase::Epsilon);
        assert_eq!(Phase::Epsilon.next(), Phase::Zeta);
        assert_eq!(Phase::Zeta.next(), Phase::Alpha);
    }
}
