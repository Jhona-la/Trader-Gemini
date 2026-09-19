use god_engine_core::GodEngineCore;
use quantum_arena::{GlobalArena, genome::SuperGenotype};
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// D-689 (DÉCIMA OLA): operaciones cerradas mínimas del universo ganador desde
/// la última replantación antes de comparar su PnL con el control. Es la misma
/// regla que la aptitud unificada (X-014): por debajo de 15 operaciones la
/// evidencia es inviable. Antes bastaba una ventaja de 6,5 céntimos sobre $13,
/// alcanzable con una sola operación afortunada.
pub const MIN_HARVEST_TRADES: usize = 15;

/// ShadowForest mantiene N motores en la sombra (universos paralelos)
/// procesando el flujo en vivo (NO backtest, NO datos pasados) para encontrar
/// el genoma que mejor resuena con el micro-régimen actual.
pub struct ShadowForest {
    pub initial_capital: f64,
    pub engines: Vec<GodEngineCore>,
    pub genomes: Vec<SuperGenotype>,
    /// D-689: operaciones cerradas de cada universo en la última replantación.
    pub trades_at_replant: Vec<usize>,
    /// QO-E2c — pico de capital por universo: alimenta el término de
    /// drawdown del fitness unificado (antes la cosecha comparaba PnL
    /// crudo — el único promotor fuera del objetivo D-652).
    pub peak_capital: Vec<f64>,
}

/// Operaciones cerradas acumuladas por un universo (todas las monedas).
fn closed_trades(engine: &GodEngineCore) -> usize {
    engine
        .arena
        .coins
        .iter()
        .map(|c| c.metrics.trade_count.load(Ordering::Relaxed))
        .sum()
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

        let trades_at_replant = engines.iter().map(closed_trades).collect();
        let peak_capital = vec![initial_capital; engines.len()];

        Self {
            initial_capital,
            engines,
            genomes,
            trades_at_replant,
            peak_capital,
        }
    }

    /// D-689: operaciones cerradas por el universo `i` desde la última replantación.
    pub fn closed_since_replant(&self, i: usize) -> usize {
        closed_trades(&self.engines[i])
            .saturating_sub(self.trades_at_replant.get(i).copied().unwrap_or(0))
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
        latency_panic: bool,
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
                latency_panic, // CERT-M8-H04: forward the REAL flag
                omni_features,
                is_buyer_maker,
            );
        }
    }

    /// Evalúa todos los genomas y devuelve el mejor si superó al de control,
    /// además devuelve el Leaderboard (PnL de todos los universos).
    pub fn harvest_best_genome(&mut self) -> (Option<(SuperGenotype, f64)>, Vec<f64>) {
        // QO-E2c — OBJETIVO UNIFICADO: la cosecha comparaba PnL CRUDO (el
        // único promotor fuera del fitness D-652 — divergencia de objetivos
        // que la auditoría señaló). Ahora cada universo se puntúa con
        /// fitness::compute (crecimiento log penalizado por drawdown²,
        /// inacción INVIABLE) y el ganador debe superar al CONTROL en
        /// fitness, no en dólares: un mutante con $1 más y +40% de
        /// drawdown YA NO gana.
        for (i, engine) in self.engines.iter().enumerate() {
            let cap = engine.arena.unified_capital.load(Ordering::Relaxed);
            if i < self.peak_capital.len() && cap > self.peak_capital[i] {
                self.peak_capital[i] = cap;
            }
        }

        let fitness_of = |i: usize| -> f64 {
            let engine = &self.engines[i];
            let cap = engine.arena.unified_capital.load(Ordering::Relaxed);
            let peak = self.peak_capital.get(i).copied().unwrap_or(cap.max(self.initial_capital));
            let dd = if peak > 0.0 { (peak - cap) / peak } else { 0.0 };
            crate::fitness::compute(&crate::fitness::FitnessInputs {
                initial_capital: self.initial_capital,
                final_capital: cap,
                max_drawdown_pct: dd,
                total_trades: closed_trades(engine) as u32,
                min_trades_required: MIN_HARVEST_TRADES as u32,
                oos_start_capital: self.initial_capital,
                oos_end_capital: cap,
            })
        };

        let mut leaderboard = Vec::with_capacity(self.engines.len());
        let mut best_fit = f64::NEG_INFINITY;
        let mut best_idx = 0usize;
        for i in 0..self.engines.len() {
            let f = fitness_of(i);
            leaderboard.push(f);
            if f > best_fit {
                best_fit = f;
                best_idx = i;
            }
        }
        let control_fit = fitness_of(0);
        let best_trades = self.closed_since_replant(best_idx);

        // Puerta: muestra mínima (D-689) y superioridad en el MISMO
        // objetivo que los otros promotores (fitness, no pnl crudo).
        let winner = if best_idx != 0
            && best_trades >= MIN_HARVEST_TRADES
            && best_fit.is_finite()
            && best_fit > control_fit
        {
            Some((self.genomes[best_idx].clone(), best_fit))
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
            // CERT-F-56: reset peak TAMBIÉN — sin esto, el fitness post-replant
            // computa drawdown contra peaks pre-replant, castigando sistemáticamente
            // a los universos que alguna vez tuvieron un high watermark alto.
            if let Some(p) = self.peak_capital.get_mut(i) {
                *p = self.initial_capital;
            }
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
        // D-689: la muestra de la siguiente cosecha empieza aquí.
        self.trades_at_replant = self.engines.iter().map(closed_trades).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shadow_forest_instantiation_and_harvest() {
        let base_genome = SuperGenotype::default();
        let mut forest = ShadowForest::new(13.0, base_genome, 3);
        assert_eq!(forest.engines.len(), 3);
        assert_eq!(forest.genomes.len(), 3);

        let (winner, leaderboard) = forest.harvest_best_genome();
        assert_eq!(leaderboard.len(), 3);
        assert!(winner.is_none());
    }

    /// D-689: una ventaja de capital sin muestra mínima no se cosecha; con la
    /// muestra, sí. La replantación reinicia la cuenta.
    #[test]
    fn d689_cosecha_exige_muestra_minima_de_operaciones() {
        let base_genome = SuperGenotype::default();
        let mut forest = ShadowForest::new(13.0, base_genome.clone(), 2);
        forest.engines[1]
            .arena
            .unified_capital
            .store(14.0, Ordering::Relaxed);
        forest.engines[1].arena.coins[0]
            .metrics
            .trade_count
            .store(MIN_HARVEST_TRADES - 1, Ordering::Relaxed);
        let (winner, _) = forest.harvest_best_genome();
        assert!(winner.is_none(), "ventaja sin muestra mínima no debe cosecharse");

        forest.engines[1].arena.coins[0]
            .metrics
            .trade_count
            .store(MIN_HARVEST_TRADES, Ordering::Relaxed);
        let (winner, _) = forest.harvest_best_genome();
        assert!(winner.is_some(), "con muestra mínima y ventaja, se cosecha");

        forest.replant(base_genome);
        assert_eq!(forest.closed_since_replant(1), 0);
        forest.engines[1]
            .arena
            .unified_capital
            .store(14.0, Ordering::Relaxed);
        let (winner, _) = forest.harvest_best_genome();
        assert!(winner.is_none(), "tras replantar la muestra vuelve a empezar");
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
            false, // CERT-M8-H04: sin pánico de latencia en el test
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
