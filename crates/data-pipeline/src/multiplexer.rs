use quantum_arena::TickEvent;


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
