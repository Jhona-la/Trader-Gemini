use quantum_arena::TickEvent;
use crate::historical::Kline;

/// Simula 4 ticks por cada Kline (Open, High, Low, Close)
pub fn kline_to_ticks(coin_id: usize, kline: &Kline) -> [TickEvent; 4] {
    let duration = kline.close_time.saturating_sub(kline.open_time);
    let step = duration / 4;
    
    // Asumimos un spread de 0.01% como mock (realista para BTC/ETH en Binance Futures)
    let spread = 0.0001;

    // Generar un sesgo (skew) direccional base
    let is_green = kline.close >= kline.open;
    let (base_buyer, _base_seller) = if is_green {
        (0.7, 0.3)
    } else {
        (0.3, 0.7)
    };

    // Simple pseudo-random using timestamp and coin_id for deterministic noise
    let hash = kline.open_time.wrapping_add(coin_id as u64);
    
    // Generar 4 variaciones de volumen
    let mut bq = [0.0; 4];
    let mut aq = [0.0; 4];
    
    for i in 0..4 {
        let raw_rand = (hash.wrapping_mul((i + 1) as u64).wrapping_mul(1103515245) >> 16) % 1000;
        
        // Fat-tailed noise: 90% of the time very small noise, 10% of the time huge spike
        let noise_skew = if raw_rand > 950 {
            // Extreme buy spike
            0.4
        } else if raw_rand < 50 {
            // Extreme sell spike
            -0.4
        } else {
            // Micro noise
            (raw_rand as f64 / 1000.0 - 0.5) * 0.05
        };
        
        let buyer_pct = (base_buyer + noise_skew).clamp(0.01, 0.99);
        let seller_pct = 1.0 - buyer_pct;
        
        bq[i] = (kline.volume * buyer_pct) / 4.0;
        aq[i] = (kline.volume * seller_pct) / 4.0;
    }

    [
        // Open
        TickEvent {
            coin_id,
            timestamp: kline.open_time,
            bid_price: kline.open * (1.0 - spread/2.0),
            ask_price: kline.open * (1.0 + spread/2.0),
            bid_qty: bq[0],
            ask_qty: aq[0],
        },
        // Path 1 (Low if green, High if red)
        TickEvent {
            coin_id,
            timestamp: kline.open_time + step,
            bid_price: if is_green { kline.low } else { kline.high } * (1.0 - spread/2.0),
            ask_price: if is_green { kline.low } else { kline.high } * (1.0 + spread/2.0),
            bid_qty: bq[1],
            ask_qty: aq[1],
        },
        // Path 2 (High if green, Low if red)
        TickEvent {
            coin_id,
            timestamp: kline.open_time + step * 2,
            bid_price: if is_green { kline.high } else { kline.low } * (1.0 - spread/2.0),
            ask_price: if is_green { kline.high } else { kline.low } * (1.0 + spread/2.0),
            bid_qty: bq[2],
            ask_qty: aq[2],
        },
        // Close
        TickEvent {
            coin_id,
            timestamp: kline.open_time + step * 3,
            bid_price: kline.close * (1.0 - spread/2.0),
            ask_price: kline.close * (1.0 + spread/2.0),
            bid_qty: bq[3],
            ask_qty: aq[3],
        }
    ]
}

/// Merge-sort O(N log N) de múltiples vectores de Ticks
pub fn multiplex_ticks(coin_ticks: Vec<Vec<TickEvent>>) -> Vec<TickEvent> {
    let total_capacity: usize = coin_ticks.iter().map(|v| v.len()).sum();
    let mut all_ticks = Vec::with_capacity(total_capacity);
    
    for ticks in coin_ticks {
        all_ticks.extend(ticks);
    }
    
    // Ordenamos cronológicamente por timestamp absoluto
    all_ticks.sort_unstable_by_key(|t| t.timestamp);
    
    all_ticks
}
