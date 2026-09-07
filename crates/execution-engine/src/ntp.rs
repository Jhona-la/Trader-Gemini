//! NTP Synchronizer for live timestamp drift correction against Binance Futures.

use crate::client::BinanceClient;
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub async fn start_ntp_synchronizer(
    client: Arc<BinanceClient>,
    arena: Arc<GlobalArena>,
) {
    let mut interval = tokio::time::interval(Duration::from_secs(15));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        interval.tick().await;

        let t0 = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(d) => d.as_millis() as u64,
            Err(_) => continue,
        };

        match client.fetch_server_time().await {
            Ok(server_time) => {
                let t1 = match SystemTime::now().duration_since(UNIX_EPOCH) {
                    Ok(d) => d.as_millis() as u64,
                    Err(_) => continue,
                };

                // Network latency estimate: half round-trip time (RTT / 2)
                let rtt_half = ((t1.saturating_sub(t0)) / 2) as i64;
                let local_estimate = (t0 as i64) + rtt_half;
                let offset_ms = (server_time as i64) - local_estimate;

                arena.server_time_offset_ms.store(offset_ms, Ordering::Relaxed);
            }
            Err(e) => {
                eprintln!("⚠️ [NTP-SYNC] Server time fetch failed: {}", e);
            }
        }
    }
}
