use execution_engine::{
    client::BinanceClient,
    order_types::{Fill, IncomeEntry, OpenAlgoOrder, OrderAck},
    reconciliation::PositionRiskEntry,
};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[test]
fn error_codes_distinguish_minimum_notional_and_throttling() {
    use execution_engine::order_types::error_codes::*;
    assert_eq!(MIN_NOTIONAL, -4164);
    assert_eq!(PARAM_EMPTY, -1105);
    assert_eq!(TOO_MANY_PARAMETERS, -1101);
    assert_eq!(TOO_MANY_REQUESTS, -1003);
    assert_ne!(MIN_NOTIONAL, TOO_MANY_REQUESTS);
}

#[test]
fn all_shared_numeric_deserializers_reject_nonfinite_strings() {
    for bad in ["NaN", "inf", "-inf", "Infinity", "1e999"] {
        let field = |name| format!(r#"{{"{name}":"{bad}"}}"#);
        assert!(serde_json::from_str::<OrderAck>(&field("executedQty")).is_err());
        assert!(serde_json::from_str::<Fill>(&field("commission")).is_err());
        assert!(serde_json::from_str::<OpenAlgoOrder>(&field("quantity")).is_err());
        assert!(serde_json::from_str::<IncomeEntry>(&field("income")).is_err());
        assert!(serde_json::from_str::<PositionRiskEntry>(&field("positionAmt")).is_err());
    }
}

#[test]
fn negative_income_rebates_and_short_positions_remain_valid() {
    let income: IncomeEntry = serde_json::from_str(r#"{"income":"-1.25"}"#).unwrap();
    let fill: Fill = serde_json::from_str(r#"{"commission":-0.25}"#).unwrap();
    let pos: PositionRiskEntry = serde_json::from_str(r#"{"positionAmt":"-2.5"}"#).unwrap();
    assert_eq!(income.income, -1.25);
    assert_eq!(fill.commission, -0.25);
    assert_eq!(pos.position_amt, -2.5);
}

// Real HTTP client against an ephemeral loopback server. No exchange, credentials,
// trading process or account is used; a synthetic response is returned once.
async fn response(
    status: &str,
    body: &str,
    advertised_length: Option<usize>,
) -> Result<OrderAck, String> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/audit", listener.local_addr().unwrap());
    let wire = format!("HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", advertised_length.unwrap_or(body.len()));
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = [0; 4096];
        stream.read(&mut request).await.unwrap();
        stream.write_all(wire.as_bytes()).await.unwrap();
        stream.shutdown().await.unwrap();
    });
    let client =
        BinanceClient::with_timeout("audit-noncredential".into(), true, Duration::from_secs(2));
    let result = client
        .execute_order_payload_typed(&url)
        .await
        .map(|(_, ack)| ack);
    server.await.unwrap();
    result
}

#[tokio::test]
async fn unreadable_success_ack_is_ambiguous_not_rejected() {
    for body in ["not json", r#"{"executedQty":"NaN"}"#] {
        let error = response("200 OK", body, None).await.unwrap_err();
        assert!(error.starts_with("AMBIGUOUS:"), "{error}");
    }
}

#[tokio::test]
async fn interrupted_success_body_is_ambiguous() {
    for status in ["200 OK", "400 Bad Request", "503 Service Unavailable"] {
        let error = response(status, "{", Some(100)).await.unwrap_err();
        assert!(error.starts_with("AMBIGUOUS:"), "{error}");
    }
}

#[tokio::test]
async fn backend_timeout_cannot_prove_order_rejection() {
    for (status, body) in [
        ("408 Request Timeout", r#"{"code":-1007,"msg":"Timeout"}"#),
        (
            "400 Bad Request",
            r#"{"code":-1006,"msg":"Unexpected response"}"#,
        ),
        ("400 Bad Request", r#"{"code":-1007,"msg":"Timeout"}"#),
    ] {
        let error = response(status, body, None).await.unwrap_err();
        assert!(error.starts_with("AMBIGUOUS:"), "{error}");
    }
}

#[tokio::test]
async fn known_ack_rejection_limits_and_server_error_keep_their_classes() {
    let ack = response("200 OK", r#"{"orderId":42,"symbol":"AUDITUSDT","clientOrderId":"audit","status":"NEW","origQty":"1","executedQty":"0"}"#, None).await.unwrap();
    assert_eq!(ack.order_id, 42);
    assert!(ack.is_active());
    for (status, body, prefix) in [
        (
            "400 Bad Request",
            r#"{"code":-2010,"msg":"rejected"}"#,
            "REJECTED",
        ),
        ("429 Too Many Requests", "{}", "HTTP_429"),
        ("418 I'm a teapot", "{}", "HTTP_418"),
        ("503 Service Unavailable", "{}", "AMBIGUOUS:"),
    ] {
        assert!(response(status, body, None)
            .await
            .unwrap_err()
            .starts_with(prefix));
    }
}
