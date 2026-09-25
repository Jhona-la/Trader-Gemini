//! Versioned research export: sampled midpoint barriers over caller-supplied
//! clock horizons. Not trading outcomes, not an operational training format.
use backtest_engine::label_evidence::{
    label_surface, validate_horizons, BarrierSpec, Record, Tape, RECORD_BYTES,
};
use god_engine_core::stateful_engine::StatefulEngine;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};

#[derive(Debug)]
struct Options {
    symbol: String,
    input: String,
    output: String,
    horizons_ms: Vec<u64>,
    stride_ms: u64,
    max_records: usize,
    max_work: u64,
    spec: BarrierSpec,
    allow_legacy: bool,
}
impl Options {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut values = BTreeMap::new();
        let mut allow_legacy = false;
        let mut i = 0;
        while i < args.len() {
            let key = args[i].as_str();
            if key == "--allow-legacy" {
                if allow_legacy {
                    return Err("duplicate --allow-legacy".into());
                }
                allow_legacy = true;
                i += 1;
                continue;
            }
            if ![
                "--symbol",
                "--input",
                "--output",
                "--horizons-ms",
                "--stride-ms",
                "--max-records",
                "--max-work",
                "--tp-return",
                "--sl-return",
                "--timestamp-unit",
            ]
            .contains(&key)
            {
                return Err(format!("unknown argument {key}; use --help"));
            }
            let value = args
                .get(i + 1)
                .ok_or_else(|| format!("missing value for {key}"))?;
            if values.insert(key, value.as_str()).is_some() {
                return Err(format!("duplicate {key}"));
            }
            i += 2;
        }
        let get = |key| {
            values
                .get(key)
                .copied()
                .ok_or_else(|| format!("required argument {key}"))
        };
        if get("--timestamp-unit")? != "ms" {
            return Err(
                "this input contract supports explicitly declared milliseconds only".into(),
            );
        }
        let symbol = get("--symbol")?.to_string();
        if symbol.is_empty()
            || !symbol
                .bytes()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
        {
            return Err("symbol must be nonempty uppercase ASCII alphanumeric".into());
        }
        let horizons_ms = get("--horizons-ms")?
            .split(',')
            .map(|s| s.parse::<u64>().map_err(|_| "invalid horizon".to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        validate_horizons(&horizons_ms)?;
        let positive = |key| -> Result<u64, String> {
            let v = get(key)?
                .parse::<u64>()
                .map_err(|_| format!("invalid {key}"))?;
            if v == 0 {
                Err(format!("{key} must be positive"))
            } else {
                Ok(v)
            }
        };
        let spec = BarrierSpec {
            take_profit_return: get("--tp-return")?
                .parse()
                .map_err(|_| "invalid --tp-return")?,
            stop_loss_return: get("--sl-return")?
                .parse()
                .map_err(|_| "invalid --sl-return")?,
        };
        spec.validate()?;
        Ok(Self {
            symbol,
            input: get("--input")?.to_string(),
            output: get("--output")?.to_string(),
            horizons_ms,
            stride_ms: positive("--stride-ms")?,
            max_records: positive("--max-records")?
                .try_into()
                .map_err(|_| "record budget exceeds platform")?,
            max_work: positive("--max-work")?,
            spec,
            allow_legacy,
        })
    }
}

/// Retain legacy channel positions without inventing external macro values.
/// 0..34 are legacy StatefulEngine outputs (may themselves have cold/default
/// values). None distinguishes unavailable/nonfinite values from numeric zero.
fn feature_snapshot(engine: &StatefulEngine, t: Record) -> Vec<Option<f64>> {
    let mut f = vec![None; 54];
    for (i, value) in engine.get_universal_features().iter().enumerate() {
        f[i] = value.is_finite().then_some(*value as f64);
    }
    // BinTick does not retain an observed buyer-maker flag. Book imbalance
    // cannot establish aggressor side; do not export a fabricated flow delta.
    f[6] = None;
    let depth_proxy = t.bid_qty + t.ask_qty;
    let delta_over_mid = engine.v_t / t.mid();
    let atr_ratio = engine.get_atr_pct();
    f[34] = delta_over_mid
        .is_finite()
        .then(|| delta_over_mid.clamp(-0.1, 0.1));
    f[35] = atr_ratio.is_finite().then_some(atr_ratio);
    f[36] = Some((depth_proxy / 1000.0).tanh());
    f[38] = f[34].map(|v| (v * 10.0).tanh());
    let scaled_atr = atr_ratio * 100.0;
    f[39] = scaled_atr.is_finite().then(|| scaled_atr.min(5.0));
    if depth_proxy > 0.0 {
        f[37] = Some((t.bid_qty - t.ask_qty) / depth_proxy);
    }
    f
}

fn line(out: &mut impl Write, value: &serde_json::Value) -> Result<(), String> {
    serde_json::to_writer(&mut *out, value).map_err(|e| e.to_string())?;
    out.write_all(b"\n").map_err(|e| e.to_string())
}

fn export_to(out: &mut impl Write, bytes: &[u8], options: &Options) -> Result<usize, String> {
    let tape = Tape::parse(bytes, options.allow_legacy)?;
    if tape.len() > options.max_records {
        return Err("record budget exceeded; no prefix sampling".into());
    }
    let source_hash = format!("{:x}", Sha256::digest(bytes));
    line(
        out,
        &json!({
            "kind":"manifest", "schema":"tgm.midpoint_barrier_surface.v2", "research_only":true,
            "symbol":options.symbol, "symbol_identity":"caller_declared_not_encoded_in_tape",
            "source_path":options.input, "source_sha256":source_hash, "source_bytes":bytes.len(),
            "records":tape.len(), "timestamp_unit":"ms", "timestamp_unit_basis":"caller_declaration",
            "provenance":tape.provenance, "observed_l2_certified":false,
            "horizons_ms":options.horizons_ms, "barriers":options.spec,
            "stride_ms":options.stride_ms, "max_work":options.max_work,
            "price_basis":"sampled_midpoint_not_executable_quote_or_fill", "costs_included":false,
            "event_order":"file_ordinal_within_equal_timestamp_not_exchange_sequence",
            "features":"legacy_stateful_34_plus_6_derived; positions_40_53_unavailable",
            "feature_limits":"legacy warmup/defaults and depth proxies remain; not host parity",
            "trade_flow_channel_6":"unavailable; buyer_maker not stored, no depth-to-aggressor inference",
            "transform_34_39":"v_t/mid clipped; ATR ratio; tanh(depth/1000); depth imbalance; tanh(10*f34); min(100*f35,5)",
            "missing_values":"null; not imputed", "completion":"requires terminal complete record"
        }),
    )?;
    let mut engine = StatefulEngine::new();
    let mut next_anchor = 0;
    let mut remaining = options.max_work;
    let mut rows = 0;
    for i in 0..tape.len() {
        let t = tape.get(i).unwrap();
        let depth_proxy = t.bid_qty + t.ask_qty;
        engine
            .try_process_tick(t.mid(), depth_proxy, t.timestamp_ms)
            .map_err(|error| format!("feature state rejected record {i}: {error:?}"))?;
        // No observed aggressor flag in this format: do not infer it from depth.
        let _ = engine.update_ofi(t.bid, t.ask, t.bid_qty, t.ask_qty);
        if t.timestamp_ms < next_anchor {
            continue;
        }
        let horizons = label_surface(&tape, i, &options.horizons_ms, options.spec, &mut remaining)?;
        let information_end = horizons.iter().map(|h| h.information_end_ms).max().unwrap();
        line(
            out,
            &json!({
                "kind":"observation", "symbol":options.symbol, "source_record":i,
                "feature_time_ms":t.timestamp_ms, "feature_history_start_ms":tape.get(0).unwrap().timestamp_ms,
                "feature_history_records":i+1, "features":feature_snapshot(&engine,t),
                "label_information_start_ms":t.timestamp_ms, "label_information_end_ms":information_end,
                "horizons":horizons
            }),
        )?;
        rows += 1;
        // If no subsequent representable anchor exists, all later records
        // still have been validated; no wraparound to timestamp zero.
        let Some(next) = t.timestamp_ms.checked_add(options.stride_ms) else {
            break;
        };
        next_anchor = next;
    }
    line(
        out,
        &json!({"kind":"complete","rows":rows,"source_sha256":source_hash,
        "work_used":options.max_work-remaining,"research_only":true}),
    )?;
    out.flush().map_err(|e| e.to_string())?;
    Ok(rows)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() == 1 && args[0] == "--help" {
        println!("Research v2 (legacy CSV is not overwritten). Required: --symbol SYMBOL --input PATH --output NEW_PATH --timestamp-unit ms --horizons-ms H1,H2,... --stride-ms N --tp-return F --sl-return F --max-records N --max-work N. Optional: --allow-legacy. No trading defaults; no promotion. Trainer legacy does not consume v2.");
        return Ok(());
    }
    let options = Options::parse(&args)?;
    let max_bytes = options
        .max_records
        .checked_mul(RECORD_BYTES)
        .and_then(|n| n.checked_add(8))
        .ok_or("byte budget overflow")?;
    let read_limit = u64::try_from(max_bytes)
        .ok()
        .and_then(|n| n.checked_add(1))
        .ok_or("read limit overflow")?;
    let mut bytes = Vec::new();
    File::open(&options.input)
        .map_err(|e| e.to_string())?
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > max_bytes {
        return Err("input exceeds record budget; refusing silent truncation".into());
    }
    // Validate before creating output; never overwrite an existing dataset.
    Tape::parse(&bytes, options.allow_legacy)?;
    let mut output = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&options.output)
            .map_err(|e| e.to_string())?,
    );
    let rows = export_to(&mut output, &bytes, &options)?;
    output.get_ref().sync_all().map_err(|e| e.to_string())?;
    println!("Research v2 complete: {rows} anchors. No model trained or promoted.");
    Ok(())
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    fn options() -> Options {
        Options {
            symbol: "BTCUSDT".into(),
            input: "synthetic_fixture.bin".into(),
            output: "unused.jsonl".into(),
            horizons_ms: vec![5, 10],
            stride_ms: 10,
            max_records: 10,
            max_work: 1000,
            spec: BarrierSpec {
                take_profit_return: 0.0036,
                stop_loss_return: 0.0018,
            },
            allow_legacy: false,
        }
    }
    fn tape(rows: &[(u64, f64)]) -> Vec<u8> {
        let mut b = b"TGMTICK1".to_vec();
        for &(ts, p) in rows {
            b.extend_from_slice(&ts.to_le_bytes());
            for v in [p, p, 1.0, 1.000000001] {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        b
    }
    fn export(rows: &[(u64, f64)]) -> Vec<serde_json::Value> {
        let mut out = Vec::new();
        export_to(&mut out, &tape(rows), &options()).unwrap();
        String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }
    #[test]
    fn explicit_cli_has_no_hidden_trading_defaults() {
        assert!(Options::parse(&[]).is_err());
        let args="--symbol ETHUSDT --input x --output y --timestamp-unit ms --horizons-ms 1,100,100000 --stride-ms 5 --tp-return 0.01 --sl-return 0.02 --max-records 100 --max-work 1000";
        let a: Vec<String> = args.split_whitespace().map(str::to_string).collect();
        let o = Options::parse(&a).unwrap();
        assert_eq!(o.symbol, "ETHUSDT");
        assert_eq!(o.horizons_ms, vec![1, 100, 100000]);
        for extra in [
            "--bad 1",
            "--symbol BTCUSDT",
            "--allow-legacy --allow-legacy",
        ] {
            let mut b = a.clone();
            b.extend(extra.split_whitespace().map(str::to_string));
            assert!(Options::parse(&b).is_err());
        }
        for (from, to) in [
            ("ms --horizons", "ns --horizons"),
            ("1,100,100000", "1,1"),
            ("--max-work 1000", "--max-work 0"),
            ("ETHUSDT", "../ETHUSDT"),
        ] {
            assert!(Options::parse(
                &args
                    .replace(from, to)
                    .split_whitespace()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            )
            .is_err());
        }
    }
    #[test]
    fn v2_keeps_flat_and_censored_rows_and_has_terminal_manifest_hash() {
        let v = export(&[(0, 100.), (10, 100.), (20, 100.)]);
        assert_eq!(v.len(), 5);
        assert_eq!(v[0]["schema"], "tgm.midpoint_barrier_surface.v2");
        assert_eq!(v[0]["observed_l2_certified"], false);
        assert_eq!(v[1]["horizons"][0]["long"]["kind"], "no_observed_hit");
        assert_eq!(v[3]["horizons"][1]["long"]["kind"], "right_censored");
        assert_eq!(v[4]["kind"], "complete");
        assert_eq!(v[4]["rows"], 3);
        assert_eq!(v[0]["source_sha256"], v[4]["source_sha256"]);
    }
    #[test]
    fn unavailable_external_features_are_null_not_macro_constants() {
        let v = export(&[(0, 100.), (10, 100.)]);
        let f = v[1]["features"].as_array().unwrap();
        assert_eq!(f.len(), 54);
        assert!(f[40..].iter().all(|v| v.is_null()));
        let obi = f[37].as_f64().unwrap();
        assert!(obi.abs() > 0. && obi.abs() < 1e-6); // no six-decimal rounding to zero
    }
    #[test]
    fn future_price_does_not_change_earlier_feature_snapshot() {
        let a = export(&[(0, 100.), (10, 100.4), (20, 101.)]);
        let b = export(&[(0, 100.), (10, 99.6), (20, 90.)]);
        assert_eq!(a[1]["features"], b[1]["features"]);
        assert_ne!(a[1]["horizons"], b[1]["horizons"]);
        assert_ne!(a[0]["source_sha256"], b[0]["source_sha256"]);
    }
    #[test]
    fn budgets_reject_instead_of_silent_prefix_or_success_footer() {
        let b = tape(&[(0, 100.), (10, 100.), (20, 100.)]);
        let mut o = options();
        o.max_records = 2;
        let mut out = Vec::new();
        assert!(export_to(&mut out, &b, &o).is_err());
        assert!(out.is_empty());
        o.max_records = 10;
        o.max_work = 3;
        assert!(export_to(&mut out, &b, &o).is_err());
        assert!(!String::from_utf8(out)
            .unwrap()
            .contains("\"kind\":\"complete\""));
    }
    #[test]
    fn invalid_tape_is_rejected_before_manifest() {
        let mut out = Vec::new();
        assert!(export_to(&mut out, b"TGMTICK2", &options()).is_err());
        assert!(out.is_empty());
    }
    #[test]
    fn writer_and_flush_errors_propagate() {
        struct Fails {
            on_write: bool,
        }
        impl Write for Fails {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                if self.on_write {
                    Err(std::io::Error::other("fixture write failure"))
                } else {
                    Ok(b.len())
                }
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Err(std::io::Error::other("fixture flush failure"))
            }
        }
        for on_write in [true, false] {
            assert!(export_to(&mut Fails { on_write }, &tape(&[(0, 100.)]), &options()).is_err());
        }
    }
    #[test]
    fn derived_overflow_is_missing_before_clipping() {
        let mut engine = StatefulEngine::new();
        engine.v_t = f64::MAX;
        let f = feature_snapshot(
            &engine,
            Record {
                timestamp_ms: 0,
                bid: 1e-100,
                ask: 1e-100,
                bid_qty: 0.,
                ask_qty: 0.,
            },
        );
        assert!(f[34].is_none());
        assert!(f[38].is_none());
        assert!(f[37].is_none());
    }

    #[test]
    fn book_imbalance_does_not_fabricate_trade_aggressor_evidence() {
        let v = export(&[(0, 100.), (10, 100.)]);
        assert!(v[1]["features"][6].is_null());
        assert!(!v[1]["features"][37].is_null()); // separately labelled depth proxy
        assert!(v[0]["trade_flow_channel_6"]
            .as_str()
            .unwrap()
            .contains("unavailable"));
    }

    #[test]
    fn feature_rejection_aborts_export_without_success_footer() {
        let mut bytes = b"TGMTICK1".to_vec();
        bytes.extend_from_slice(&0u64.to_le_bytes());
        for value in [1e200_f64, 1e200, 1e200, 1e200] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let mut out = Vec::new();
        let error = export_to(&mut out, &bytes, &options()).unwrap_err();
        assert!(error.contains("record 0: NonFiniteDerivedValue"));
        let text = String::from_utf8(out).unwrap();
        assert!(!text.contains("\"kind\":\"complete\""));
        assert!(!text.contains("\"kind\":\"observation\""));
    }
}
