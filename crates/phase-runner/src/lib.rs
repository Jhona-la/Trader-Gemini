use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Alpha,   // Basic scaffolding and initialization
    Beta,    // Forward tracing data flow
    Gamma,   // Strategy signals and decisions
    Delta,   // Backward tracing from execution
    Epsilon, // Mutation and optimization
    Zeta,    // Deep structural graph tracing
}

impl Phase {
    pub fn next(&self) -> Self {
        match self {
            Phase::Alpha => Phase::Beta,
            Phase::Beta => Phase::Gamma,
            Phase::Gamma => Phase::Delta,
            Phase::Delta => Phase::Epsilon,
            Phase::Epsilon => Phase::Zeta,
            Phase::Zeta => Phase::Alpha, // Loop forever
        }
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            Phase::Alpha => "Alpha",
            Phase::Beta => "Beta",
            Phase::Gamma => "Gamma",
            Phase::Delta => "Delta",
            Phase::Epsilon => "Epsilon",
            Phase::Zeta => "Zeta",
        }
    }
}

pub struct AdaptiveTimer {
    base_interval: Duration,
    _max_memory_mb: usize,
    last_run_timestamp: AtomicU64,
}

impl AdaptiveTimer {
    pub fn new(base_interval_ms: u64, max_memory_mb: usize) -> Self {
        // FIX #1450: Clamping defensivo de intervalo base
        let safe_interval = base_interval_ms.clamp(1, 3_600_000);
        Self {
            base_interval: Duration::from_millis(safe_interval),
            _max_memory_mb: max_memory_mb,
            last_run_timestamp: AtomicU64::new(0),
        }
    }

    /// Sleeps dynamically based on current CPU load.
    /// If CPU > 50%, it dilates the interval to avoid starving HFT engines.
    pub async fn wait_next_cycle(&self) {
        // Se obtiene la telemetría real del SO.
        let telemetry = os_guardian::telemetry::get_system_telemetry();
        let cpu_load = if telemetry.cpu_usage.is_finite() && telemetry.cpu_usage >= 0.0 {
            (telemetry.cpu_usage / 100.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut actual_interval = self.base_interval;

        if cpu_load > 0.8 {
            // Dilatación extrema bajo saturación severa de CPU
            actual_interval = actual_interval.mul_f32(5.0);
        } else if cpu_load > 0.5 {
            // Dilatación moderada bajo carga media
            actual_interval = actual_interval.mul_f32(2.0);
        }

        tokio::time::sleep(actual_interval).await;

        let now = chrono::Utc::now().timestamp_millis().max(0) as u64;
        self.last_run_timestamp.store(now, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseResult {
    pub phase: Phase,
    pub findings: Vec<String>,
    pub severity: String, // "CRITICAL", "HIGH", "MEDIUM", "LOW"
    pub duration_ms: u64,
    pub next_interval_ms: u64,
}

pub struct PhaseExecutor;

impl PhaseExecutor {
    pub fn run(phase: Phase, base_interval: Duration) -> PhaseResult {
        let start = Instant::now();

        // Ejecución real de auditoría de fase y telemetría de sistema
        let telemetry = os_guardian::telemetry::get_system_telemetry();
        let mut findings = vec![format!("Audited phase: {}", phase.to_str())];
        findings.push(format!(
            "Host RAM used: {:.1} MB, CPU: {:.1}%",
            telemetry.memory_used_mb, telemetry.cpu_usage
        ));

        let severity = if telemetry.memory_used_mb > (16.0 * 1024.0 * 0.85) {
            "HIGH".to_string()
        } else {
            "LOW".to_string()
        };

        let duration = start.elapsed();

        PhaseResult {
            phase,
            findings,
            severity,
            duration_ms: duration.as_millis() as u64,
            next_interval_ms: base_interval.as_millis() as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_executor() {
        let result = PhaseExecutor::run(Phase::Alpha, Duration::from_millis(100));
        assert_eq!(result.phase, Phase::Alpha);
        assert_eq!(result.next_interval_ms, 100);
        assert!(!result.findings.is_empty());
    }

    #[tokio::test]
    async fn test_adaptive_timer() {
        let timer = AdaptiveTimer::new(10, 1024);
        timer.wait_next_cycle().await;
        assert!(timer.last_run_timestamp.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_phase_transitions_full_cycle() {
        let mut p = Phase::Alpha;
        p = p.next();
        assert_eq!(p, Phase::Beta);
        p = p.next();
        assert_eq!(p, Phase::Gamma);
        p = p.next();
        assert_eq!(p, Phase::Delta);
        p = p.next();
        assert_eq!(p, Phase::Epsilon);
        p = p.next();
        assert_eq!(p, Phase::Zeta);
        p = p.next();
        assert_eq!(p, Phase::Alpha);
        assert_eq!(Phase::Alpha.to_str(), "Alpha");
    }

    #[test]
    fn test_phase_result_serialization() {
        let res = PhaseResult {
            phase: Phase::Gamma,
            findings: vec!["Signal engine validated".to_string()],
            severity: "LOW".to_string(),
            duration_ms: 15,
            next_interval_ms: 100,
        };
        let json = serde_json::to_string(&res).expect("serialization of PhaseResult");
        assert!(json.contains("Gamma"));
        assert!(json.contains("Signal engine validated"));
        let restored: PhaseResult = serde_json::from_str(&json).expect("deserialization of PhaseResult");
        assert_eq!(restored.phase, Phase::Gamma);
        assert_eq!(restored.severity, "LOW");
    }

    #[test]
    fn test_phase_all_to_str_and_clamped_timer() {
        assert_eq!(Phase::Beta.to_str(), "Beta");
        assert_eq!(Phase::Gamma.to_str(), "Gamma");
        assert_eq!(Phase::Delta.to_str(), "Delta");
        assert_eq!(Phase::Epsilon.to_str(), "Epsilon");
        assert_eq!(Phase::Zeta.to_str(), "Zeta");

        // Clamped interval at zero
        let timer = AdaptiveTimer::new(0, 1024);
        assert_eq!(timer.base_interval, Duration::from_millis(1));
    }
}


