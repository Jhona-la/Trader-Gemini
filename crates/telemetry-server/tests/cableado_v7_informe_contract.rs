//! CABLEADOV7 (pedido CABLEADO HOST v7, §5.1): invariantes del informe que el
//! cableado del host no debe cambiar. No fallan antes: son invariantes, no
//! carencias. El informe no inventa importes y la CLI sigue exigiendo el income.

use telemetry_server::report_demo_run::{build_report, parse_income_jsonl};

#[test]
fn informe_no_inventa_importes_y_cli_rechaza_income_ausente() {
    // Un importe no finito deja el neto en NaN (se muestra N/D), no en cero.
    let income = parse_income_jsonl(
        "{\"symbol\":\"VETUSDT\",\"income_type\":\"COMMISSION\",\"income\":\"x\",\"time_ms\":1,\"trade_id\":\"7\",\"tran_id\":1,\"asset\":\"USDT\"}\n",
    );
    assert_eq!(income.len(), 1);
    let ops = build_report(&[], &income);
    assert_eq!(ops.len(), 1);
    assert!(ops[0].net.is_nan(), "neto sin inventar: {:?}", ops[0].net);
    assert_eq!(ops[0].who_ordered, "N/D");

    // La CLI exige --income y falla si el archivo no existe.
    let dir = std::env::temp_dir().join(format!("tg_cablev7_cli_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let log = dir.join("run.log");
    std::fs::write(&log, "").unwrap();
    let exe = env!("CARGO_BIN_EXE_demo_report");
    let sin_income = std::process::Command::new(exe).arg("--log").arg(&log).output().unwrap();
    assert_eq!(sin_income.status.code(), Some(2), "sin --income: {:?}", sin_income);
    let ausente = dir.join("no_existe.jsonl");
    let con_ausente = std::process::Command::new(exe)
        .arg("--log")
        .arg(&log)
        .arg("--income")
        .arg(&ausente)
        .output()
        .unwrap();
    assert_eq!(con_ausente.status.code(), Some(2), "income inexistente: {:?}", con_ausente);
    assert!(String::from_utf8_lossy(&con_ausente.stderr).contains("no lee"));
    let _ = std::fs::remove_dir_all(&dir);
}
