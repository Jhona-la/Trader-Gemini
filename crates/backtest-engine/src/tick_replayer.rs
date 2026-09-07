//! High-performance memory-mapped binary tick replayer.

use quantum_arena::TickEvent;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BinTick {
    pub timestamp: u64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_qty: f64,
    pub ask_qty: f64,
}

pub fn load_binary_ticks(path: &Path, coin_id: usize) -> std::io::Result<Vec<TickEvent>> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let tick_size = std::mem::size_of::<BinTick>();
    let num_ticks = mmap.len() / tick_size;
    let slice = unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const BinTick, num_ticks) };
    
    let mut ticks = Vec::with_capacity(num_ticks);
    for t in slice {
        ticks.push(TickEvent {
            coin_id,
            timestamp: t.timestamp,
            bid_price: t.bid_price,
            ask_price: t.ask_price,
            bid_qty: t.bid_qty,
            ask_qty: t.ask_qty,
        });
    }
    Ok(ticks)
}

pub fn load_multi_coin_binary_ticks(files: &[(&Path, usize)]) -> std::io::Result<Vec<TickEvent>> {
    let mut all_ticks = Vec::new();
    for (path, cid) in files {
        if let Ok(mut ticks) = load_binary_ticks(path, *cid) {
            all_ticks.append(&mut ticks);
        }
    }
    all_ticks.sort_by_key(|t| t.timestamp);
    Ok(all_ticks)
}
