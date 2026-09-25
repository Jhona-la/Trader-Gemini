//! Narrow source regressions: do not execute the diagnostic or contact an exchange.
//! These guard the demonstrated calls/claims, not arbitrary future side effects.
const HEALTH: &str = include_str!("../../../src/bin/system_health.rs");
const EXECUTOR: &str = include_str!("../../execution-engine/src/executor.rs");

#[test]
fn diagnostic_does_not_request_account_mode_mutation() {
    assert!(!HEALTH.contains("exec.ensure_hedge_mode().await"));
    assert!(HEALTH.contains("exec.fetch_hedge_mode().await"));
}

#[test]
fn diagnostic_does_not_treat_process_local_watchdog_as_engine_health() {
    assert!(!HEALTH.contains("feed_health::is_stalled()"));
    assert!(HEALTH.contains("DESCONOCIDO"));
}

#[test]
fn income_rows_are_not_reported_as_trade_win_rate() {
    assert!(!HEALTH.contains("WR: {:.1}%"));
    assert!(HEALTH.contains("una página"));
    assert!(HEALTH.contains("filas positivas"));
}

#[test]
fn read_only_mode_method_has_no_post_or_mode_store() {
    let method = EXECUTOR
        .split("pub async fn fetch_hedge_mode(")
        .nth(1)
        .expect("read-only mode getter");
    let method = method
        .split("pub async fn ensure_hedge_mode(")
        .next()
        .unwrap();
    assert!(method.contains("get_payload_account"));
    assert!(!method.contains(".post_payload("));
    assert!(!method.contains(".store("));
}
