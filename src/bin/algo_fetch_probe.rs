//! Sonda de diagnóstico B1.3: ¿por qué fetch_open_algo_orders devuelve
//! vacío/error cuando curl sí ve las piernas? Imprime el resultado crudo.
//! LECTURA SOLA — no coloca ni cancela nada.

#[tokio::main]
async fn main() {
    let (key, secret) = (
        std::env::var("BINANCE_DEMO_API_KEY")
            .or_else(|_| std::env::var("BINANCE_TESTNET_API_KEY"))
            .unwrap_or_default(),
        std::env::var("BINANCE_DEMO_SECRET_KEY")
            .or_else(|_| std::env::var("BINANCE_TESTNET_SECRET_KEY"))
            .unwrap_or_default(),
    );
    let symbol = std::env::var("ALGO_PROBE_SYMBOL").unwrap_or_else(|_| "BTCUSDT".into());
    let mut exec = execution_engine::executor::OrderExecutor::new(key, secret, true);
    exec.set_paper_trading(false);
    println!("── fetch_open_algo_orders({}):", symbol);
    match exec.fetch_open_algo_orders(&symbol).await {
        Ok(v) => {
            println!("   OK: {} piernas", v.len());
            for a in &v {
                println!(
                    "   · {} type={} status={} ps={} qty={:.6} trig={:.4}",
                    a.client_algo_id, a.order_type, a.algo_status, a.position_side, a.quantity, a.trigger_price
                );
            }
        }
        Err(e) => println!("   ❌ {}", e),
    }
}
