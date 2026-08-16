//! TESTS DE INTEGRACIÓN TESTNET (F1.11) — ciclo REAL contra Binance Futures Testnet.
//!
//! EJECUCIÓN (requiere llaves de demo en env; NUNCA mainnet):
//!   BINANCE_TESTNET_API_KEY=... BINANCE_TESTNET_SECRET_KEY=... \
//!   cargo test -p execution-engine --test testnet_integration -- --ignored --nocapture
//!
//! SEGURIDAD:
//! - Solo opera contra testnet.binancefuture.com (hardcodeado).
//! - Nunca envía MARKET: usa LIMIT post-only (GTX) a precio absurdo (-50%)
//!   que jamás cruza el libro → orden NEW garantizada sin fill accidental.

use execution_engine::executor::{ExecutionProvider, OrderExecutor};

#[allow(unused_imports)]
use std::sync::atomic::Ordering;

fn testnet_creds() -> Option<(String, String)> {
    let k = std::env::var("BINANCE_TESTNET_API_KEY").unwrap_or_default();
    let s = std::env::var("BINANCE_TESTNET_SECRET_KEY").unwrap_or_default();
    if k.trim().is_empty() || s.trim().is_empty() {
        None
    } else {
        Some((k, s))
    }
}

fn executor() -> OrderExecutor {
    let (k, s) = testnet_creds().expect("creds testnet");
    let mut e = OrderExecutor::new(k, s, true);
    e.set_paper_trading(false); // testnet real, jamás mainnet
    e
}

/// Precio actual de BTCUSDT desde testnet (endpoint público sin firma).
async fn btc_price(exec: &OrderExecutor) -> f64 {
    let url = "https://testnet.binancefuture.com/fapi/v1/ticker/price?symbol=BTCUSDT";
    let (_, body) = exec.client().get_payload(url).await.expect("ticker price");
    let v: serde_json::Value = serde_json::from_str(&body).expect("ticker json");
    v["price"].as_str().unwrap().parse().expect("precio f64")
}

#[tokio::test]
#[ignore = "requiere BINANCE_TESTNET_API_KEY/SECRET"]
async fn t01_conectividad_tiempo_y_lector_precio() {
    let exec = executor();
    let t = exec.fetch_server_time().await.expect("server time");
    assert!(
        t > 1_600_000_000_000,
        "timestamp de servidor plausible: {}",
        t
    );
    let p = btc_price(&exec).await;
    assert!(p > 0.0, "BTCUSDT testnet price: {}", p);
}

#[tokio::test]
#[ignore = "requiere BINANCE_TESTNET_API_KEY/SECRET"]
async fn t02_ciclo_de_vida_orden_new_query_cancel() {
    let exec = executor();
    // Higiene de DEMO (dinero falso): huérfanas de sesiones previas bloquean
    // el cambio de modo (-4068) y ensucian la reconciliación. Aplanar primero.
    let (cancelled_syms, closed_positions) = exec
        .flatten_all_positions()
        .await
        .expect("flatten demo leftovers");
    println!("flatten demo: {cancelled_syms} símbolos con órdenes canceladas, {closed_positions} posiciones cerradas");

    // El motor es hedge (positionSide LONG/SHORT): la cuenta debe estar en
    // dualSidePosition o TODA orden falla -4061 (hallazgo de este ciclo).
    let changed = exec.ensure_hedge_mode().await.expect("ensure_hedge_mode");
    println!(
        "modo hedge: {}",
        if changed {
            "activado por el test"
        } else {
            "ya estaba activo"
        }
    );
    let price = btc_price(&exec).await;
    // GTX post-only al 50% del precio: garantizado NEW, jamás fill.
    // BTCUSDT futuros: tickSize=0.1 — el precio DEBE alinear al tick o
    // Binance rechaza con -4014 (lección validada en vivo, gap #4 del API map).
    let tick = 0.1;
    let far_price = (price * 0.5 / tick).round() * tick;
    // Notional mínimo del testnet: $50 (-4164). Cantidad dinámica para
    // garantizar ~$100 de notional al precio lejano, alineada al step 0.001.
    let step = 0.001;
    let qty = ((100.0 / far_price) / step).ceil() * step;

    let coid = uuid::Uuid::now_v7().simple().to_string();
    exec.registry()
        .register_intent(&coid, "BTCUSDT", "BUY", "LONG", "LIMIT", qty, 0);

    // Colocar LIMIT GTX (via camino interno del executor).
    let limit_res = exec
        .execute_limit_order("BTCUSDT", true, qty, far_price, step, tick, &coid)
        .await;
    assert!(
        limit_res.is_ok(),
        "LIMIT GTX debe entrar NEW: {:?}",
        limit_res
    );

    // Query por clientOrderId: el corazón de la idempotencia.
    let ack = exec
        .query_order("BTCUSDT", &coid)
        .await
        .expect("query order");
    assert_eq!(ack.client_order_id, coid);
    assert_eq!(ack.status, "NEW", "post-only lejano debe quedar NEW");
    assert!((ack.executed_qty).abs() < 1e-12);

    // Cancel y verificar estado final en el registro.
    exec.cancel_order("BTCUSDT", &coid).await.expect("cancel");
    let final_ack = exec
        .query_order("BTCUSDT", &coid)
        .await
        .expect("query post-cancel");
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    exec.registry().apply_ack(&final_ack, now);
    let tracked = exec.registry().get(&coid).expect("orden registrada");
    assert!(
        tracked.status == execution_engine::OrderStatus::Canceled
            || tracked.status == execution_engine::OrderStatus::Expired,
        "tras cancel: {:?}",
        tracked.status
    );
}

#[tokio::test]
#[ignore = "requiere BINANCE_TESTNET_API_KEY/SECRET"]
async fn t03_reconciliacion_positionrisk_estructural() {
    let exec = executor();
    let entries = exec.fetch_position_risk().await.expect("positionRisk");
    // Puede estar vacía (sin posiciones): lo importante es el parseo estructural.
    for p in &entries {
        assert!(!p.symbol.is_empty());
        if p.is_open() {
            assert!(p.entry_price > 0.0, "posición abierta debe tener entrada");
        }
    }
    println!(
        "positionRisk: {} entradas (abiertas: {})",
        entries.len(),
        entries.iter().filter(|p| p.is_open()).count()
    );
}

#[tokio::test]
#[ignore = "requiere BINANCE_TESTNET_API_KEY/SECRET"]
async fn t04_user_data_y_caps_de_cuenta() {
    let exec = executor();
    // Dos requests con peso distinto: si algo rompe el presupuesto, falla aquí.
    let _ = exec.fetch_server_time().await.expect("server time");
    let _ = exec.fetch_position_risk().await.expect("positionRisk");
}

#[tokio::test]
#[ignore = "requiere BINANCE_TESTNET_API_KEY/SECRET"]
async fn t05_user_data_stream_listenkey() {
    let exec = executor();
    let key = exec.client().create_listen_key().await.expect("listenKey");
    assert!(key.len() > 20, "listenKey plausible: len={}", key.len());
    exec.client()
        .keep_alive_listen_key()
        .await
        .expect("keepalive");
}
