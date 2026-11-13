//! DEMO_TEL T2 (QRA-TELEMETRIA-DEMO, desviacion 2): TRADE_FEE lleva el tradeId NATIVO del
//! mismo fill, sin sustituto. Prueba de EMISION real: el streamer real despacha un
//! ORDER_TRADE_UPDATE de entrada (`cS_`) y se lee su stdout (las dos lineas son `println!`).
//! El harness de pruebas captura stdout dentro del proceso, asi que la prueba exterior relanza
//! este mismo binario con la interior (`#[ignore]`) y `--nocapture`, con `TGM_DATA_DIR` en una
//! carpeta temporal (el diario de entradas no toca `data/`).

use execution_engine::client::BinanceClient;
use execution_engine::order_registry::OrderRegistry;
use execution_engine::user_data_stream::UserDataStreamer;
use std::sync::Arc;

const COID: &str = "cS_tel2fee";

fn evento(t: u64, n: &str) -> String {
    format!(
        r#"{{"e":"ORDER_TRADE_UPDATE","E":1700000000000,"o":{{"s":"VETUSDT","c":"{COID}","S":"SELL","ps":"BOTH","o":"LIMIT","f":"GTC","q":"996","p":"0.00512","ap":"0.00512","sp":"0","X":"PARTIALLY_FILLED","i":20202,"l":"400","z":"400","L":"0.00512","n":"{n}","N":"USDT","T":1700000000123,"t":{t},"m":false}}}}"#
    )
}

#[test]
#[ignore = "lo lanza trade_fee_lleva_el_trade_id_nativo_del_fill como subproceso"]
fn emisor_fill_de_entrada() {
    let registry = Arc::new(OrderRegistry::new());
    registry.register_intent(COID, "VETUSDT", "SELL", "SHORT", "LIMIT", 996.0, 1_700_000_000_000);
    let streamer = UserDataStreamer::new(BinanceClient::new("k".into(), true), registry);
    streamer.dispatch(evento(7_110_001, "0.00081920").as_bytes());
}

fn campo<'a>(linea: &'a str, k: &str) -> &'a str {
    let pref = format!(" {k}=");
    let i = linea.find(&pref).unwrap_or_else(|| panic!("sin {k}: {linea}")) + pref.len();
    linea[i..].split_whitespace().next().unwrap()
}

#[test]
fn trade_fee_lleva_el_trade_id_nativo_del_fill() {
    let datos = std::env::temp_dir().join(format!("tel2_fee_{}", std::process::id()));
    std::fs::create_dir_all(&datos).unwrap();
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "emisor_fill_de_entrada", "--nocapture", "--test-threads=1"])
        .env("TGM_DATA_DIR", &datos)
        .output()
        .expect("relanza el binario de prueba");
    let salida = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(out.status.success(), "{salida}\n{}", String::from_utf8_lossy(&out.stderr));
    let fill: Vec<&str> = salida.lines().filter(|l| l.starts_with("DEMO_TEL event=FILL ")).collect();
    let fee: Vec<&str> = salida.lines().filter(|l| l.starts_with("DEMO_TEL event=TRADE_FEE ")).collect();
    assert_eq!((fill.len(), fee.len()), (1, 1), "una linea de cada una:\n{salida}");
    assert_eq!(campo(fee[0], "trade_id"), "7110001", "tradeId nativo del fill");
    assert_eq!(campo(fee[0], "trade_id"), campo(fill[0], "trade_id"), "TRADE_FEE y FILL unen por tradeId");
    assert_ne!(campo(fee[0], "trade_id"), COID, "nunca el clientOrderId como sustituto");
    assert_eq!(campo(fee[0], "client_order_id"), COID);
    assert_eq!(campo(fee[0], "fee"), "0.00081920");
    assert_eq!(campo(fee[0], "fee_asset"), "USDT");
    assert_eq!(campo(fee[0], "ts"), campo(fill[0], "ts"));
    assert!(fee[0].ends_with(" alcance=ENTRADA"), "{}", fee[0]);
    // La linea humana sigue igual y una sola vez (el harness antepone `test ... ` en su renglon).
    assert_eq!(salida.lines().filter(|l| l.contains("💧 [FILL] VETUSDT SELL LIMIT cS_tel2fee @ 0.00512 (acum 400/996 status ")).count(), 1, "{salida}");
    let _ = std::fs::remove_dir_all(&datos);
}
