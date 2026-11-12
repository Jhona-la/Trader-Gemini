use execution_engine::client::BinanceClient;
use execution_engine::user_data_stream::UserDataStreamer;
use execution_engine::{OrderAck, OrderRegistry, OrderStatus};
use serde_json::{json, Value};
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

const FIXTURE_DIR: &str = "CODEX_M2_FILL_TEST_DIR";

#[test]
fn replayed_partial_fill_does_not_add_commission_after_cancel() -> Result<(), Box<dyn Error>> {
    let temp = std::env::temp_dir().canonicalize()?;
    if let Some(dir) = std::env::var_os(FIXTURE_DIR) {
        let dir = PathBuf::from(dir);
        if dir.parent() != Some(temp.as_path()) || std::env::current_dir()?.canonicalize()? != dir {
            return Err("unexpected fill test working directory".into());
        }
        return exercise_stream();
    }

    // The real asynchronous journal writes a relative path. Isolate it in a
    // child process so neither the repo's journal nor its CWD can be changed.
    let dir = temp.join(format!("tg-m2-fill-{}", uuid::Uuid::now_v7()));
    std::fs::create_dir(&dir)?;
    let output = std::process::Command::new(std::env::current_exe()?)
        .current_dir(&dir)
        .env(FIXTURE_DIR, &dir)
        .env("TG_GENOME_ENV", "backtest")
        .args([
            "--exact",
            "replayed_partial_fill_does_not_add_commission_after_cancel",
            "--nocapture",
        ])
        .output();
    if dir.canonicalize()? != dir || dir.parent() != Some(temp.as_path()) {
        return Err("fill test directory changed".into());
    }
    std::fs::remove_dir_all(&dir)?;
    let output = output?;
    assert!(
        output.status.success(),
        "isolated stream test failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    Ok(())
}

fn exercise_stream() -> Result<(), Box<dyn Error>> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let _runtime = runtime.enter();
    let registry = Arc::new(OrderRegistry::new());
    let stream = UserDataStreamer::new(BinanceClient::new(String::new(), true), registry.clone());
    let ack: OrderAck = serde_json::from_value(json!({
        "orderId": 42, "symbol": "BTCUSDT", "clientOrderId": "cL_m2_partial",
        "type": "LIMIT", "side": "BUY", "status": "PARTIALLY_FILLED",
        "origQty": "1", "executedQty": "0.5", "avgPrice": "100", "cumQuote": "50",
        "fills": [{"tradeId": 7, "qty": "0.5", "price": "100",
                   "commission": "0.25", "commissionAsset": "USDT"}],
    }))?;
    registry.apply_ack(&ack, 10);
    let first = json!({
        "e": "ORDER_TRADE_UPDATE", "E": 11,
        "o": {
            "s": "BTCUSDT", "c": "cL_m2_partial", "S": "BUY", "ps": "LONG",
            "o": "LIMIT", "x": "TRADE", "X": "PARTIALLY_FILLED", "i": 42,
            "q": "1", "ap": "100", "l": "0.5", "z": "0.5", "L": "100",
            "n": "0.25", "N": "USDT", "t": 7, "T": 11, "m": true
        },
    });
    let mut unpriced = first.clone();
    unpriced
        .get_mut("o")
        .and_then(Value::as_object_mut)
        .ok_or("missing event object")?
        .remove("L");
    stream.dispatch(unpriced.to_string().as_bytes());
    let order = registry
        .get(&ack.client_order_id)
        .ok_or("missing partial order")?;
    assert_eq!(
        order.ws_commission, 0.25,
        "REST must not hide a WS fee with missing price"
    );
    stream.dispatch(first.to_string().as_bytes());
    let canceled = json!({
        "e": "ORDER_TRADE_UPDATE", "E": 12,
        "o": {
            "s": "BTCUSDT", "c": "cL_m2_partial", "S": "BUY", "ps": "LONG",
            "o": "LIMIT", "x": "CANCELED", "X": "CANCELED", "i": 42,
            "q": "1", "ap": "100", "l": "0", "z": "0.5", "T": 12
        },
    });
    stream.dispatch(canceled.to_string().as_bytes());
    stream.dispatch(first.to_string().as_bytes());
    // A distinct late fill is the journal fence: it must still be accounted
    // after cancellation, and observing it means the earlier rows were drained.
    let late = json!({
        "e": "ORDER_TRADE_UPDATE", "E": 13,
        "o": {
            "s": "BTCUSDT", "c": "cL_m2_partial", "S": "BUY", "ps": "LONG",
            "o": "LIMIT", "x": "TRADE", "X": "PARTIALLY_FILLED", "i": 42,
            "q": "1", "ap": "100", "l": "0.25", "z": "0.75", "L": "100",
            "n": "-0.125", "N": "USDT", "t": 8, "T": 13, "m": true
        },
    });
    stream.dispatch(late.to_string().as_bytes());

    let rows = read_until_late_fill(Path::new("data/demo/trade_fills.jsonl"))?;
    assert_eq!(
        rows.iter().filter(|r| r["ts"] == 11).count(),
        1,
        "partial fill journaled twice"
    );
    let after = registry
        .get(&ack.client_order_id)
        .ok_or("missing canceled order")?;
    assert_eq!(after.status, OrderStatus::Canceled);
    assert_eq!(after.executed_qty, 0.75);
    assert_eq!(after.total_commission, 0.125);
    assert_eq!(
        after.ws_commission, 0.125,
        "replay must not add another WS commission"
    );
    Ok(())
}

fn read_until_late_fill(journal: &Path) -> Result<Vec<Value>, Box<dyn Error>> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match std::fs::read_to_string(journal) {
            Ok(text) => {
                let rows: Vec<Value> = text
                    .split_inclusive('\n')
                    .filter(|line| line.ends_with('\n'))
                    .map(serde_json::from_str)
                    .collect::<Result<_, _>>()?;
                if rows.iter().any(|row| row["ts"] == 13) {
                    return Ok(rows);
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        if Instant::now() >= deadline {
            return Err("late fill did not reach the isolated journal".into());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}
