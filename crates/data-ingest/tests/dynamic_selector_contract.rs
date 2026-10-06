use data_ingest::DynamicSelector;
use serde_json::json;

#[test]
fn checked_selector_distinguishes_invalid_payload_and_empty_eligibility() {
    use data_ingest::dynamic_selector::SelectorError;
    assert_eq!(
        DynamicSelector::try_parse_and_rank_tickers(&json!({}), 1.0),
        Err(SelectorError::ExpectedTickerArray)
    );
    assert_eq!(
        DynamicSelector::try_parse_and_rank_tickers(&json!([]), -1.0),
        Err(SelectorError::InvalidMinimumVolume)
    );
    assert_eq!(
        DynamicSelector::try_parse_and_rank_tickers(&json!([]), 1.0),
        Ok(vec![])
    );
}

#[test]
fn conflicting_duplicates_have_no_implicit_max_score_resolution() {
    use data_ingest::dynamic_selector::SelectorError;
    let a = json!({"symbol":"BTCUSDT","quoteVolume":"10000000","priceChangePercent":"2"});
    let b = json!({"symbol":"BTCUSDT","quoteVolume":"20000000","priceChangePercent":"2"});
    for data in [json!([a.clone(), b.clone()]), json!([b, a])] {
        assert_eq!(
            DynamicSelector::try_parse_and_rank_tickers(&data, 0.0),
            Err(SelectorError::ConflictingDuplicate {
                symbol: "BTCUSDT".into()
            })
        );
    }
}

#[test]
fn open_debt_zero_daily_net_return_receives_no_slot_regardless_of_path_variation() {
    let data = json!([{"symbol":"BTCUSDT","quoteVolume":"1000000000","priceChangePercent":"0"}]);
    assert!(DynamicSelector::try_parse_and_rank_tickers(&data, 0.0)
        .unwrap()
        .is_empty());
    // XCVIII (triaje A): la versión anterior era `assert!(variation > 0.0)`
    // — una tautología matemática que no ejercitaba NINGÚN código. La
    // aserción REAL: el selector con pct=0 no selecciona AUNQUE la
    // variación intra-día del path fuera no-trivial (ese es el open-debt
    // que el nombre del test describe). Ya está cubierto por el assert
    // anterior (pct=0 ⇒ sin slot); este comentario preserva la intención.
    let _variation = (1.5_f64.ln()).powi(2) + (1.0_f64 / 1.5).ln().powi(2);
}

#[test]
fn empty_or_malformed_feed_does_not_invent_ten_instruments() {
    for data in [
        json!([]),
        json!({"code":-1,"msg":"bad response"}),
        json!([{"symbol":"BTCUSDT","quoteVolume":"NaN","priceChangePercent":"2"}]),
    ] {
        assert!(DynamicSelector::parse_and_rank_tickers(&data, 10000.0).is_empty());
    }
}

#[test]
fn duplicate_instruments_do_not_consume_multiple_slots() {
    let item = json!({"symbol":"BTCUSDT","quoteVolume":"10000000","priceChangePercent":"2"});
    let result = DynamicSelector::parse_and_rank_tickers(&json!([item.clone(), item]), 10000.0);
    assert_eq!(result, vec!["BTCUSDT"]);
}

#[test]
fn equal_scores_have_input_order_independent_ties() {
    let btc = json!({"symbol":"BTCUSDT","quoteVolume":"10000000","priceChangePercent":"2"});
    let eth = json!({"symbol":"ETHUSDT","quoteVolume":"10000000","priceChangePercent":"2"});
    let a = DynamicSelector::parse_and_rank_tickers(&json!([btc.clone(), eth.clone()]), 10000.0);
    let b = DynamicSelector::parse_and_rank_tickers(&json!([eth, btc]), 10000.0);
    assert_eq!(a, b);
}

#[test]
fn invalid_threshold_does_not_fall_back_to_default_universe() {
    assert!(DynamicSelector::parse_and_rank_tickers(&json!([]), f64::NAN).is_empty());
}
