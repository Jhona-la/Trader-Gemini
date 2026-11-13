//! DEMO_TEL T2 (QRA-TELEMETRIA-DEMO, fila T5): coste del stream con las lineas de telemetria.
//!
//! Banco, no contrato: `#[ignore]`, se lanza a mano con el MISMO archivo en cada pin y con stdout
//! redirigido a un archivo (misma captura en todos):
//!   `<bin> --ignored --exact bench_dispatch_fills_de_entrada --nocapture --test-threads=1 > out.txt`
//! Cada evento es un ORDER_TRADE_UPDATE de entrada (`cS_`, FILLED, tradeId nuevo): la rama que
//! imprime `💧 [FILL]`, `DEMO_TEL event=FILL` y `DEMO_TEL event=TRADE_FEE` y anota el diario de
//! entradas. Se mide `dispatch` por evento con `Instant` y se escriben los cuantiles por stderr
//! (`BENCH ...`). Usa solo API que existe en 6a466dd1, 515e3bf4 y la rama. `TGM_DATA_DIR` debe
//! apuntar a una carpeta temporal.

use execution_engine::client::BinanceClient;
use execution_engine::order_registry::OrderRegistry;
use execution_engine::user_data_stream::UserDataStreamer;
use std::sync::Arc;
use std::time::Instant;

const N: usize = 20_000;
const CALENTAMIENTO: usize = 2_000;

fn evento(i: usize) -> String {
    format!(
        r#"{{"e":"ORDER_TRADE_UPDATE","E":1700000000000,"o":{{"s":"VETUSDT","c":"cS_bench{i}","S":"SELL","ps":"BOTH","o":"LIMIT","f":"GTC","q":"996","p":"0.00512","ap":"0.00512","sp":"0","X":"FILLED","i":{oid},"l":"996","z":"996","L":"0.00512","n":"0.00203981","N":"USDT","T":{t},"t":{tid},"m":false}}}}"#,
        oid = 900_000 + i,
        t = 1_700_000_000_000u64 + i as u64,
        tid = 5_000_000 + i
    )
}

#[test]
#[ignore = "banco de latencia T5: se lanza a mano"]
fn bench_dispatch_fills_de_entrada() {
    let registry = Arc::new(OrderRegistry::new());
    let total = CALENTAMIENTO + N;
    for i in 0..total {
        registry.register_intent(&format!("cS_bench{i}"), "VETUSDT", "SELL", "SHORT", "LIMIT", 996.0, 1_700_000_000_000);
    }
    let eventos: Vec<String> = (0..total).map(evento).collect();
    let streamer = UserDataStreamer::new(BinanceClient::new("k".into(), true), registry);
    let mut ns: Vec<u128> = Vec::with_capacity(N);
    for (i, e) in eventos.iter().enumerate() {
        let t0 = Instant::now();
        streamer.dispatch(e.as_bytes());
        let dt = t0.elapsed().as_nanos();
        if i >= CALENTAMIENTO {
            ns.push(dt);
        }
    }
    let media = ns.iter().sum::<u128>() as f64 / ns.len() as f64;
    ns.sort_unstable();
    let q = |p: f64| ns[((ns.len() - 1) as f64 * p).round() as usize];
    eprintln!(
        "BENCH n={} media_ns={:.0} p50_ns={} p90_ns={} p99_ns={} max_ns={}",
        ns.len(),
        media,
        q(0.50),
        q(0.90),
        q(0.99),
        ns[ns.len() - 1]
    );
}
