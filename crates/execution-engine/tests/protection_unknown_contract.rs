fn host() -> &'static str {
    include_str!("../../../src/bin/god_engine.rs")
}

fn compact(text: &str) -> String {
    text.split_whitespace().collect()
}

#[test]
fn open_algo_fetch_failure_is_unknown_and_cannot_create_legs() {
    let source = host();
    let body = source
        .split("async fn ensure_position_protected(")
        .nth(1)
        .expect("ensure_position_protected")
        .split("/// B3.6")
        .next()
        .expect("function body");
    assert!(
        !compact(body).contains("fetch_open_algo_orders(symbol).await.unwrap_or_default()"),
        "a failed fetch must not become an empty order list"
    );
    let compacted = compact(body);
    assert!(
        compacted.contains("evidence_unknown:true"),
        "failed coverage evidence must be explicit"
    );
    assert!(
        compacted.contains("protection_health::mark_dirty()"),
        "unknown coverage must remain scheduled for reconciliation"
    );
}

#[test]
fn watchdog_does_not_escalate_unknown_coverage() {
    let compacted = compact(host());
    assert!(
        compacted.contains("ifprot.evidence_unknown{"),
        "unknown coverage must bypass gap accounting and emergency escalation"
    );
}

#[test]
fn position_risk_failures_never_use_theoretical_quantities() {
    let source = host();
    assert!(
        !source.contains(".unwrap_or(true); // si fetch falla, no bloquear"),
        "failed positionRisk must not become still_open=true"
    );
    assert!(
        !source.contains(".unwrap_or(qty_intent)"),
        "failed positionRisk must not become the intended OCO quantity"
    );
    assert!(
        !source.contains(".unwrap_or(final_qty.abs())"),
        "failed positionRisk must not become the theoretical emergency-close quantity"
    );
    assert!(
        source.matches("POSITION_EVIDENCE_UNKNOWN").count() >= 2,
        "OCO and emergency-close failures must both leave unknown evidence for reconciliation"
    );
}
