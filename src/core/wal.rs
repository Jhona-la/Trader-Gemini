use rkyv::{Archive, Deserialize, Serialize};
use crate::core::state::GlobalArena;
use std::sync::atomic::Ordering;
use std::fs::{File, OpenOptions};
use memmap2::MmapOptions;
use std::path::Path;

#[derive(Archive, Serialize, Deserialize, Debug, Clone, Default)]
#[archive(check_bytes)]
pub struct PositionSnapshot {
    pub is_open: bool,
    pub is_long: bool,
    pub entry_price: f64,
    pub quantity: f64,
    pub margin_used: f64,
    pub entry_time_ms: u64,
    pub trail_stop: f64,
    pub trailing_phase: u8,
    pub mfe_atr: f64,
    pub max_pnl_pct: f64,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct CoinStateSnapshot {
    pub scalp_pnl_realized: f64,
    pub scalp_win_rate: f64,
    pub scalp_profit_factor: f64,
    pub scalp_trade_count: usize,
    pub swing_pnl_realized: f64,
    pub swing_win_rate: f64,
    pub swing_profit_factor: f64,
    pub swing_trade_count: usize,
    pub scalp_position: PositionSnapshot,
    pub swing_position: PositionSnapshot,
}

#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct ArenaSnapshot {
    pub unified_capital: f64,
    pub used_margin: f64,
    pub available_margin: f64,
    pub tick_counter: u64,
    pub total_fees_paid: f64,
    pub coins: Vec<CoinStateSnapshot>,
}

impl ArenaSnapshot {
    pub fn capture(arena: &GlobalArena) -> Self {
        let mut coins = Vec::with_capacity(30);
        for i in 0..30 {
            let coin = &arena.coins[i];
            
            let scalp_pos = PositionSnapshot {
                is_open: coin.positions.scalp_position.is_open(),
                is_long: coin.positions.scalp_position.is_long.load(Ordering::Relaxed),
                entry_price: coin.positions.scalp_position.entry_price.load(Ordering::Relaxed),
                quantity: coin.positions.scalp_position.quantity.load(Ordering::Relaxed),
                margin_used: coin.positions.scalp_position.margin_used.load(Ordering::Relaxed),
                entry_time_ms: coin.positions.scalp_position.entry_time_ms.load(Ordering::Relaxed),
                trail_stop: coin.positions.scalp_position.trail_stop.load(Ordering::Relaxed),
                trailing_phase: coin.positions.scalp_position.trailing_phase.load(Ordering::Relaxed),
                mfe_atr: coin.positions.scalp_position.mfe_atr.load(Ordering::Relaxed),
                max_pnl_pct: coin.positions.scalp_position.max_pnl_pct.load(Ordering::Relaxed),
            };

            let swing_pos = PositionSnapshot {
                is_open: coin.positions.swing_position.is_open(),
                is_long: coin.positions.swing_position.is_long.load(Ordering::Relaxed),
                entry_price: coin.positions.swing_position.entry_price.load(Ordering::Relaxed),
                quantity: coin.positions.swing_position.quantity.load(Ordering::Relaxed),
                margin_used: coin.positions.swing_position.margin_used.load(Ordering::Relaxed),
                entry_time_ms: coin.positions.swing_position.entry_time_ms.load(Ordering::Relaxed),
                trail_stop: coin.positions.swing_position.trail_stop.load(Ordering::Relaxed),
                trailing_phase: coin.positions.swing_position.trailing_phase.load(Ordering::Relaxed),
                mfe_atr: coin.positions.swing_position.mfe_atr.load(Ordering::Relaxed),
                max_pnl_pct: coin.positions.swing_position.max_pnl_pct.load(Ordering::Relaxed),
            };

            coins.push(CoinStateSnapshot {
                scalp_pnl_realized: coin.scalp.pnl_realized.load(Ordering::Relaxed),
                scalp_win_rate: coin.scalp.win_rate.load(Ordering::Relaxed),
                scalp_profit_factor: coin.scalp.profit_factor.load(Ordering::Relaxed),
                scalp_trade_count: coin.scalp.trade_count.load(Ordering::Relaxed),
                swing_pnl_realized: coin.swing.pnl_realized.load(Ordering::Relaxed),
                swing_win_rate: coin.swing.win_rate.load(Ordering::Relaxed),
                swing_profit_factor: coin.swing.profit_factor.load(Ordering::Relaxed),
                swing_trade_count: coin.swing.trade_count.load(Ordering::Relaxed),
                scalp_position: scalp_pos,
                swing_position: swing_pos,
            });
        }

        Self {
            unified_capital: arena.unified_capital.load(Ordering::Relaxed),
            used_margin: arena.used_margin.load(Ordering::Relaxed),
            available_margin: arena.available_margin.load(Ordering::Relaxed),
            tick_counter: arena.tick_counter.load(Ordering::Relaxed),
            total_fees_paid: arena.total_fees_paid.load(Ordering::Relaxed),
            coins,
        }
    }
    
    pub fn restore(&self, arena: &GlobalArena) {
        arena.unified_capital.store(self.unified_capital, Ordering::Relaxed);
        arena.used_margin.store(self.used_margin, Ordering::Relaxed);
        arena.available_margin.store(self.available_margin, Ordering::Relaxed);
        arena.tick_counter.store(self.tick_counter, Ordering::Relaxed);
        arena.total_fees_paid.store(self.total_fees_paid, Ordering::Relaxed);
        
        for i in 0..30 {
            if i >= self.coins.len() { break; }
            let s_coin = &self.coins[i];
            let a_coin = &arena.coins[i];
            
            a_coin.scalp.pnl_realized.store(s_coin.scalp_pnl_realized, Ordering::Relaxed);
            a_coin.scalp.win_rate.store(s_coin.scalp_win_rate, Ordering::Relaxed);
            a_coin.scalp.profit_factor.store(s_coin.scalp_profit_factor, Ordering::Relaxed);
            a_coin.scalp.trade_count.store(s_coin.scalp_trade_count, Ordering::Relaxed);
            
            a_coin.swing.pnl_realized.store(s_coin.swing_pnl_realized, Ordering::Relaxed);
            a_coin.swing.win_rate.store(s_coin.swing_win_rate, Ordering::Relaxed);
            a_coin.swing.profit_factor.store(s_coin.swing_profit_factor, Ordering::Relaxed);
            a_coin.swing.trade_count.store(s_coin.swing_trade_count, Ordering::Relaxed);
            
            if s_coin.scalp_position.is_open {
                a_coin.positions.scalp_position.is_open.store(true, Ordering::Relaxed);
                a_coin.positions.scalp_position.is_long.store(s_coin.scalp_position.is_long, Ordering::Relaxed);
                a_coin.positions.scalp_position.entry_price.store(s_coin.scalp_position.entry_price, Ordering::Relaxed);
                a_coin.positions.scalp_position.quantity.store(s_coin.scalp_position.quantity, Ordering::Relaxed);
                a_coin.positions.scalp_position.margin_used.store(s_coin.scalp_position.margin_used, Ordering::Relaxed);
                a_coin.positions.scalp_position.entry_time_ms.store(s_coin.scalp_position.entry_time_ms, Ordering::Relaxed);
                a_coin.positions.scalp_position.trail_stop.store(s_coin.scalp_position.trail_stop, Ordering::Relaxed);
                a_coin.positions.scalp_position.trailing_phase.store(s_coin.scalp_position.trailing_phase, Ordering::Relaxed);
                a_coin.positions.scalp_position.mfe_atr.store(s_coin.scalp_position.mfe_atr, Ordering::Relaxed);
                a_coin.positions.scalp_position.max_pnl_pct.store(s_coin.scalp_position.max_pnl_pct, Ordering::Relaxed);
            }
            
            if s_coin.swing_position.is_open {
                a_coin.positions.swing_position.is_open.store(true, Ordering::Relaxed);
                a_coin.positions.swing_position.is_long.store(s_coin.swing_position.is_long, Ordering::Relaxed);
                a_coin.positions.swing_position.entry_price.store(s_coin.swing_position.entry_price, Ordering::Relaxed);
                a_coin.positions.swing_position.quantity.store(s_coin.swing_position.quantity, Ordering::Relaxed);
                a_coin.positions.swing_position.margin_used.store(s_coin.swing_position.margin_used, Ordering::Relaxed);
                a_coin.positions.swing_position.entry_time_ms.store(s_coin.swing_position.entry_time_ms, Ordering::Relaxed);
                a_coin.positions.swing_position.trail_stop.store(s_coin.swing_position.trail_stop, Ordering::Relaxed);
                a_coin.positions.swing_position.trailing_phase.store(s_coin.swing_position.trailing_phase, Ordering::Relaxed);
                a_coin.positions.swing_position.mfe_atr.store(s_coin.swing_position.mfe_atr, Ordering::Relaxed);
                a_coin.positions.swing_position.max_pnl_pct.store(s_coin.swing_position.max_pnl_pct, Ordering::Relaxed);
            }
        }
    }

    pub fn save_to_disk(&self, path: &str) -> Result<(), std::io::Error> {
        use std::io::Write;
        let temp_path = format!("{}.tmp", path);
        let bytes = rkyv::to_bytes::<_, 1024>(self)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Failed to serialize arena snapshot"))?;
            
        {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&temp_path)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        std::fs::rename(temp_path, path)?;
        Ok(())
    }

    pub fn load_from_disk(path: &str) -> Result<Self, std::io::Error> {
        if !Path::new(path).exists() {
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "Snapshot not found"));
        }

        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        match rkyv::check_archived_root::<Self>(&mmap[..]) {
            Ok(archived) => {
                let snapshot: Self = archived.deserialize(&mut rkyv::Infallible).unwrap();
                Ok(snapshot)
            }
            Err(_) => Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Corrupted snapshot data")),
        }
    }
}
