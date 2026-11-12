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
    // I-14 (N44): unknown coverage is an explicit error. The function returns
    // `Result` and the leg read leaves with `?` before anything is placed.
    assert!(
        compacted.contains(")->Result<ProtectionAudit,String>{"),
        "failed coverage evidence must be explicit"
    );
    let lectura = compacted
        .find("cobertura_leida(executor.fetch_open_algo_orders(symbol).await,is_long,)?;")
        .expect("failed coverage evidence must be explicit");
    let colocacion = compacted.find(".place_algo_leg(").expect("leg placement");
    assert!(
        lectura < colocacion,
        "no leg can be placed before the coverage is read"
    );
}

#[test]
fn watchdog_does_not_escalate_unknown_coverage() {
    let compacted = compact(host());
    // Watchdog: the error skips the position before gap accounting and leaves
    // the audit incomplete, so the protection flag is not cleared and the
    // reading is retried (`vigilante::ReintentoLectura`). It does not call
    // `mark_dirty`: that counts a terminal event and would advance its own
    // retry every cycle.
    let llamada = compacted
        .find("\"WATCHDOG\",).await{")
        .expect("watchdog call");
    let resto = &compacted[llamada..];
    let error = resto.find("Err(e)=>{").expect("watchdog error branch");
    let incompleta = resto
        .find("lecturas.error(&p.symbol,&e);continue;")
        .expect("unknown coverage must remain scheduled for reconciliation");
    let huecos = resto
        .find("let(g_tp,g_sl)=(prot.tp_gap,prot.sl_gap);")
        .expect("gap accounting");
    assert!(
        error < incompleta && incompleta < huecos,
        "unknown coverage must bypass gap accounting and emergency escalation"
    );
    // Startup adoption (RESTORE) runs outside the watchdog: there the caller
    // marks the flag dirty so the watchdog audits it.
    let restore = compacted
        .find("|Err(e)=ensure_position_protected(")
        .expect("restore call");
    let r = &compacted[restore..];
    let fin = r
        .find("\"RESTORE\",).await{")
        .expect("restore error branch");
    let marca = r
        .find("protection_health::mark_dirty();")
        .expect("unknown coverage must remain scheduled for reconciliation");
    // `}});` closes the error branch and the spawned task; a bare `});` also
    // appears inside the log's format string (`({});`).
    let siguiente = r.find("}});").expect("end of the restore task");
    assert!(
        fin < marca && marca < siguiente,
        "unknown coverage must remain scheduled for reconciliation"
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

#[test]
fn residual_close_paths_share_reconciliation() {
    let source = host();
    assert!(
        source
            .matches("execute_reduce_only_market_with_reconciliation(")
            .count()
            >= 2,
        "X-008 y X-009 deben compartir la reconciliación conductual"
    );
    assert!(
        !source.contains(
            ".execute_reduce_only_market(&parsed_sym_str, is_long_close, qty, sym_filter.step_size)"
        ),
        "X-008 no debe repetir la cantidad anterior"
    );
    assert!(
        !source.contains(
            ".execute_reduce_only_market(&parsed_sym_str, is_long_close, close_qty, dyn_step_size)"
        ),
        "X-009 no debe repetir la cantidad anterior"
    );
}

#[test]
fn emergency_purge_waits_for_successful_close() {
    let body = host()
        .split("'close_with_known_quantity: {")
        .nth(1)
        .expect("emergency close body");
    let flat_guard = body
        .find("if close_res.is_ok()")
        .expect("flat evidence guard");
    let cancel = body
        .find(".cancel_position_oco_orders(")
        .expect("OCO purge");
    let match_result = body.find("match close_res").expect("result match");
    let err_branch = body.find("Err(es) => {").expect("error branch");
    assert!(
        flat_guard < cancel && cancel < match_result && match_result < err_branch,
        "la purga sólo puede ejecutarse tras prueba de posición plana"
    );
}

#[test]
fn ws_residual_bypasses_rest_dispatch_before_resnapshot() {
    let source = include_str!("../src/executor.rs");
    let branch = source
        .find("if let Some(tras) = ws_tras_cierre")
        .expect("WS settlement branch");
    let rest_dispatch = source
        .find("let coid_rest = uuid::Uuid::now_v7()")
        .expect("REST dispatch");
    assert!(
        branch < rest_dispatch,
        "la rama residual WS debe abrir antes de cualquier envío REST con la cantidad anterior"
    );
}
