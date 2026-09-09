use god_engine_core::GodEngineCore;
use quantum_arena::{GlobalArena, genome::SuperGenotype};
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// ShadowForest mantiene N motores en la sombra (universos paralelos)
/// procesando el flujo en vivo (NO backtest, NO datos pasados) para encontrar
/// el genoma que mejor resuena con el micro-régimen actual.
pub struct ShadowForest {
    pub initial_capital: f64,
    pub engines: Vec<GodEngineCore>,
    pub genomes: Vec<SuperGenotype>,
}

impl ShadowForest {
    pub fn new(initial_capital: f64, base_genome: SuperGenotype, num_trees: usize) -> Self {
        let mut engines = Vec::with_capacity(num_trees);
        let mut genomes = Vec::with_capacity(num_trees);

        for i in 0..num_trees {
            let arena = Arc::new(GlobalArena::new(initial_capital));

            // FASE 7: Lock parallel universe memory (Zero Swapping)
            unsafe {
                let _ = os_guardian::memory_compaction::lock_critical_memory(&*arena);
                let _ = os_guardian::memory_compaction::lock_critical_memory_slice(&*arena.coins);
            }

            // Tree 0 es el genoma base (control). Los demás son mutaciones puras.
            let mutation = if i == 0 {
                base_genome.clone()
            } else {
                base_genome.mutate_cmaes(0.15)
            };

            mutation.apply_to_arena(&arena);

            let mut engine = GodEngineCore::new(arena);
            // HyperRealistic para que incluya todos los fees y latencias simuladas
            engine.reality.mode = god_engine_core::reality_physics::EngineMode::HyperRealistic;
            // Simulamos el peor caso estadístico de latencia API Binance (AWS AP-Northeast a Tokyo)
            // para que las mutaciones sobrevivan en el mundo real, no en simulaciones ideales.
            engine
                .arena
                .config
                .latency_penalty_ms
                .store(25.0, std::sync::atomic::Ordering::Relaxed);

            engines.push(engine);
            genomes.push(mutation);
        }

        Self {
            initial_capital,
            engines,
            genomes,
        }
    }

    /// Alimenta un evento en vivo a todos los universos paralelos en la sombra
    #[inline(always)]
    pub fn broadcast_tick(
        &mut self,
        coin_id: usize,
        is_trade: bool,
        is_kline_closed: bool,
        is_depth: bool,
        current_price: f64,
        qty: f64,
        dbp: f64,
        dap: f64,
        dbq: f64,
        daq: f64,
        depth_obi: f64,
        depth_micro_div: f64,
        event_time: u64,
        main_arena: &Arc<GlobalArena>,
        omni_features: &[f64; 54],
        is_buyer_maker: bool,
    ) {
        // FASE 14: Suspensión Cuántica por OS Guardian.
        // Si Windows está asfixiado en RAM, no malgastamos ciclos en los clones de sombra.
        if main_arena.panic_memory_dump.load(Ordering::Relaxed) {
            return;
        }

        for engine in self.engines.iter_mut() {
            // Actualizamos la arena interna del engine
            engine
                .arena
                .update_market_data(coin_id, dbp, dap, dbq, daq, event_time);
            if is_trade {
                engine.arena.coins[coin_id]
                    .current_price
                    .store(current_price, Ordering::Relaxed);
            }

            engine.process_event(
                coin_id,
                is_trade,
                is_kline_closed,
                is_depth,
                current_price,
                qty,
                dbp,
                dap,
                dbq,
                daq,
                depth_obi,
                depth_micro_div,
                event_time,
                false, // No panic latency in shadow
                omni_features,
                is_buyer_maker,
            );
        }
    }

    /// Evalúa todos los genomas y devuelve el mejor si superó al de control,
    /// además devuelve el Leaderboard (PnL de todos los universos).
    pub fn harvest_best_genome(&self) -> (Option<(SuperGenotype, f64)>, Vec<f64>) {
        let mut best_pnl = -999999.0;
        let mut best_idx = 0;
        let mut leaderboard = Vec::with_capacity(self.engines.len());

        let control_cap = self.engines[0]
            .arena
            .unified_capital
            .load(Ordering::Relaxed);
        let control_pnl = control_cap - self.initial_capital;

        for (i, engine) in self.engines.iter().enumerate() {
            let cap = engine.arena.unified_capital.load(Ordering::Relaxed);
            let pnl = cap - self.initial_capital;
            let safe_pnl = if pnl.is_finite() { pnl } else { -999999.0 };
            leaderboard.push(safe_pnl);
            if safe_pnl > best_pnl {
                best_pnl = safe_pnl;
                best_idx = i;
            }
        }

        // Axioma de Inercia: Solo proponemos cambio si la mutación venció al control
        // significativamente (> 0.5% del capital base) y tiene PnL positivo para evitar inestabilidad del sistema.
        let winner = if best_idx != 0
            && (best_pnl - control_pnl > self.initial_capital * 0.005)
            && best_pnl > 0.0
        {
            Some((self.genomes[best_idx].clone(), best_pnl))
        } else {
            None
        };

        (winner, leaderboard)
    }

    /// Resetea los capitales y muta todos los árboles basándose en el nuevo Alpha
    pub fn replant(&mut self, new_alpha: SuperGenotype) {
        for (i, engine) in self.engines.iter_mut().enumerate() {
            // Reset capital
            engine
                .arena
                .unified_capital
                .store(self.initial_capital, Ordering::Relaxed);
            engine
                .arena
                .config
                .base_capital
                .store(self.initial_capital, Ordering::Relaxed);
            // Cerramos todas las posiciones virtuales
            for coin in engine.arena.coins.iter() {
                coin.positions.position.close();
            }

            let mutation = if i == 0 {
                new_alpha.clone()
            } else {
                new_alpha.mutate_cmaes(0.15)
            };

            mutation.apply_to_arena(&engine.arena);
            self.genomes[i] = mutation;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadow_forest_instantiation_and_harvest() {
        let base_genome = SuperGenotype::default();
        let forest = ShadowForest::new(13.0, base_genome, 3);
        assert_eq!(forest.engines.len(), 3);
        assert_eq!(forest.genomes.len(), 3);

        let (winner, leaderboard) = forest.harvest_best_genome();
        assert_eq!(leaderboard.len(), 3);
        assert!(winner.is_none());
    }

    #[test]
    fn test_shadow_forest_replant_and_broadcast_tick() {
        let base_genome = SuperGenotype::default();
        let mut forest = ShadowForest::new(13.0, base_genome.clone(), 2);
        let main_arena = Arc::new(GlobalArena::new(13.0));

        forest.broadcast_tick(
            0,
            true,
            false,
            false,
            50000.0,
            1.0,
            49999.0,
            50001.0,
            10.0,
            10.0,
            0.1,
            0.0,
            1600000000,
            &main_arena,
            &[0.0; 54],
            false,
        );

        forest.replant(base_genome);
        assert_eq!(forest.engines.len(), 2);
        assert_eq!(
            forest.engines[0]
                .arena
                .unified_capital
                .load(Ordering::Relaxed),
            13.0
        );
    }
}
