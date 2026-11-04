use execution_engine::order_types::IncomeEntry;

#[test]
fn missing_income_is_rejected_and_explicit_zero_survives() {
    let missing = r#"[{"symbol":"BTCUSDT","incomeType":"COMMISSION","asset":"USDT","time":1700000000000,"tranId":1,"tradeId":"1"}]"#;
    assert!(
        serde_json::from_str::<Vec<IncomeEntry>>(missing).is_err(),
        "un registro sin income no puede fabricar una comisión cero"
    );

    for (amount, expected_bits) in [("0.0", 0.0_f64.to_bits()), ("-0.25", (-0.25_f64).to_bits())] {
        let payload = missing.replace("\"asset\"", &format!("\"income\":\"{amount}\",\"asset\""));
        let parsed = serde_json::from_str::<Vec<IncomeEntry>>(&payload);
        assert!(
            parsed.is_ok(),
            "importe explícito rechazado: {:?}",
            parsed.err()
        );
        if let Ok(rows) = parsed {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].income.to_bits(), expected_bits);
        }
    }
}
