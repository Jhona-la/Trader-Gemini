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
    /// Pico del capital realizado (`unified_capital`) observado por universo
    /// desde la replantación, al terminar cada broadcast y en cada cosecha.
    /// No incluye MTM ni extremos entre observaciones.
    pub peak_capital: Vec<f64>,
    /// E04: máximo drawdown de esas observaciones, conservado entre cosechas.
    /// NaN expone una historia inválida (capital no finito/no positivo); una
    /// recuperación no la convierte en riesgo cero. Sólo replant la reinicia.
    pub max_drawdown_pct: Vec<f64>,
    /// CL-40: generación del almacén alrededor de la cual se plantaron los
    /// universos (0 = ninguna sancionada). Sus núcleos ya no adoptan el
    /// genoma activo por su cuenta (`RecargaGenoma::Fija`): cuando el núcleo
    /// vivo aplica una generación más nueva, el host replanta con
    /// `seguir_generacion`.
    pub generacion_base: u64,
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

/// O(1), sin asignaciones: observa el capital del universo, no sus posiciones.
fn observe_capital(capital: f64, peak: &mut f64, max_dd: &mut f64) {
    if !capital.is_finite() || capital <= 0.0 || !peak.is_finite() || *peak <= 0.0 {
        *max_dd = f64::NAN;
        return;
    }
    if capital > *peak {
        *peak = capital;
    }
    let dd = (*peak - capital) / *peak;
    // Comparar explícitamente conserva NaN; f64::max ocultaría la invalidez.
    if dd > *max_dd {
        *max_dd = dd;
    }
}

/// SOL-R5-02-B (Sol, medición 2026-10-15): equity MARCADA de un universo
/// al instante de la cosecha — caja más el valor a mercado de las
/// posiciones abiertas al último precio transmitido, neto de la fricción
/// de salida (taker + piso de deslizamiento). Misma convención que el
/// examen del demonio (`valor_a_mercado_de_lo_abierto`, LG-7) y que el
/// MTM del replay (GLM-123). Sin posiciones abiertas es bit-idéntico a
/// la caja.
fn capital_marcado(engine: &GodEngineCore) -> f64 {
    let cash = engine.arena.unified_capital.load(Ordering::Relaxed);
    let taker = engine.arena.config.live_taker_fee.load(Ordering::Relaxed);
    let deslizamiento = engine.arena.config.base_slippage_floor.load(Ordering::Relaxed);
    let friccion = if taker.is_finite() && taker >= 0.0 { taker } else { 0.0 }
        + if deslizamiento.is_finite() && deslizamiento >= 0.0 { deslizamiento } else { 0.0 };
    let mut mtm = 0.0;
    for coin in engine.arena.coins.iter() {
        let precio = coin.current_price.load(Ordering::Relaxed);
        if !(precio.is_finite() && precio > 0.0) {
            continue;
        }
        for pos in coin.positions.slots() {
            if !pos.is_open() {
                continue;
            }
            if let Some(snap) = pos.snapshot() {
                let dir = if snap.is_long { 1.0 } else { -1.0 };
                let pnl_bruto = (precio - snap.entry_price) * snap.quantity * dir;
                mtm += pnl_bruto - snap.quantity * precio * friccion;
            }
        }
    }
    cash + mtm
}

impl ShadowForest {
    pub fn new(initial_capital: f64, base_genome: SuperGenotype, num_trees: usize) -> Self {
        let mut engines = Vec::with_capacity(num_trees);
        let mut genomes = Vec::with_capacity(num_trees);

        for i in 0..num_trees {
            let arena = GlobalArena::build_in_own_stack(initial_capital);

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

            // CL-142: universo sombra, sin trazas en el log del host: sus
            // OPEN/CLOSE TRACE no son órdenes de la cuenta.
            let mut engine = GodEngineCore::new_auxiliar(arena);
            // HyperRealistic para que incluya todos los fees y latencias simuladas
            engine.reality.mode = god_engine_core::reality_physics::EngineMode::HyperRealistic;
            // E03: la latencia evaluada es la del genoma aplicado y guardado.

            engines.push(engine);
            genomes.push(mutation);
        }

        let trades_at_replant = engines.iter().map(closed_trades).collect();
        let peak_capital = vec![initial_capital; engines.len()];
        let max_drawdown_pct = vec![
            if initial_capital.is_finite() && initial_capital > 0.0 { 0.0 } else { f64::NAN };
            engines.len()
        ];

        Self {
            initial_capital,
            engines,
            genomes,
            trades_at_replant,
            peak_capital,
            max_drawdown_pct,
            generacion_base: 0,
        }
    }

    /// D-689: operaciones cerradas por el universo `i` desde la última replantación.
    pub fn closed_since_replant(&self, i: usize) -> usize {
        closed_trades(&self.engines[i])
            .saturating_sub(self.trades_at_replant.get(i).copied().unwrap_or(0))
    }

    /// XCIV (F7-A-H2, fase medición): PnL REALIZADO acumulado del universo
    /// de CONTROL (engine 0, genoma sancionado) sobre el capital inicial —
    /// la contraparte REAL para el drift-audit (el shadow 0.95·real del
    /// host es sintético y sólo caza contabilidad podrida, no divergencia
    /// bt↔vivo). Doctrina D-751: PUBLICAR primero, observar la distribución
    /// en vivo, y cablear el veto con calibración cuando la señal medida
    /// lo justifique — el veto NO se toca en esta ola.
    pub fn control_realized_pnl_pct(&self) -> Option<f64> {
        let engine = self.engines.first()?;
        let realized: f64 = engine
            .arena
            .coins
            .iter()
            .map(|c| c.metrics.pnl_realized.load(Ordering::Relaxed))
            .sum();
        if !realized.is_finite() || self.initial_capital <= 0.0 {
            return None;
        }
        Some(realized / self.initial_capital)
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
            // La suspensión del motor no borra la observación contable actual.
            for (i, engine) in self.engines.iter().enumerate() {
                observe_capital(
                    capital_marcado(engine),
                    &mut self.peak_capital[i],
                    &mut self.max_drawdown_pct[i],
                );
            }
            return;
        }

        for (i, engine) in self.engines.iter_mut().enumerate() {
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
            observe_capital(
                engine.arena.unified_capital.load(Ordering::Relaxed),
                &mut self.peak_capital[i],
                &mut self.max_drawdown_pct[i],
            );
        }
    }


    /// Evalúa todos los genomas y devuelve el mejor si superó al de control,
    /// además devuelve el Leaderboard (fitness de todos los universos).
    pub fn harvest_best_genome(&mut self) -> (Option<(SuperGenotype, f64)>, Vec<f64>) {
        // QO-E2c — OBJETIVO UNIFICADO: la cosecha comparaba PnL CRUDO (el
        // único promotor fuera del fitness D-652 — divergencia de objetivos
        // que la auditoría señaló). Ahora cada universo se puntúa con
        // fitness::compute (crecimiento log penalizado por drawdown²,
        // inacción INVIABLE) y el ganador debe superar al CONTROL en
        // fitness, no en dólares: un mutante con $1 más y +40% de
        // drawdown YA NO gana.
        let mut leaderboard = Vec::with_capacity(self.engines.len());
        let mut best_fit = f64::NEG_INFINITY;
        let mut best_idx = 0usize;
        for i in 0..self.engines.len() {
            let engine = &self.engines[i];
            // SOL-R5-02-B: equity MARCADA (caja + lo abierto a mercado),
            // no la caja sola — la cosecha PROMUEVE, así que su objetivo
            // debe ser el mismo que el del replay (GLM-123) y el del
            // examen del demonio (LG-7).
            let cap = capital_marcado(engine);
            observe_capital(cap, &mut self.peak_capital[i], &mut self.max_drawdown_pct[i]);
            // La misma lectura de capital actualiza la historia y puntúa.
            // Sin split OOS en la cosecha: convención legacy documentada
            // en fitness.rs (endpoints iguales ⇒ factor 1.0) — el slot
            // OOS ya NO amplifica las pérdidas IS que `ln` ya penaliza.
            let f = crate::fitness::compute(&crate::fitness::FitnessInputs {
                initial_capital: self.initial_capital,
                final_capital: cap,
                max_drawdown_pct: self.max_drawdown_pct[i],
                total_trades: closed_trades(engine) as u32,
                min_trades_required: MIN_HARVEST_TRADES as u32,
                oos_start_capital: cap,
                oos_end_capital: cap,
            });
            leaderboard.push(f);
            if f > best_fit {
                best_fit = f;
                best_idx = i;
            }
        }
        let control_fit = leaderboard[0];
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

    /// CL-40: replanta alrededor de `genoma` si no es ya el control (un
    /// re-registro del mismo genoma con otra generación no borra la muestra
    /// de la cosecha). Devuelve si replantó.
    ///
    /// CL-40c: decide el genoma, no el número de generación. El host sólo la
    /// llama cuando el almacén cambia (fecha de `active.json`) o tras su
    /// propia promoción, y el almacén puede repetir o bajar números (dos
    /// promotores leen el mismo padre; un `active.json` ilegible cuenta como
    /// generación 0), mientras el demonio y la cosecha aplican su genoma al
    /// arena sin mirar el número. `generacion_base` sólo registra la mayor.
    pub fn seguir_generacion(&mut self, generacion: u64, genoma: &SuperGenotype) -> bool {
        self.generacion_base = self.generacion_base.max(generacion);
        if self
            .genomes
            .first()
            .is_some_and(|control| crate::online_daemon::same_genome(control, genoma))
        {
            return false;
        }
        self.replant(genoma.clone());
        true
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
            if let Some(dd) = self.max_drawdown_pct.get_mut(i) {
                *dd = if self.initial_capital.is_finite() && self.initial_capital > 0.0 {
                    0.0
                } else {
                    f64::NAN
                };
            }
            engine
                .arena
                .config
                .base_capital
                .store(self.initial_capital, Ordering::Relaxed);
            // CL-134 (R6 de la revisión del PR #62): el universo replantado
            // es otra hipótesis y su caída empieza aquí. Antes conservaba el
            // pico del arena y del riesgo, lo medido de las órdenes, la caída
            // en stops y el freno de la estrategia: un universo que había
            // subido a 14,1 USD quedaba vetado por drawdown al volver a 13.
            engine.reiniciar_frenos(self.initial_capital);
            // Cerramos todas las posiciones virtuales en todos los slots espectrales
            for coin in engine.arena.coins.iter() {
                for slot in coin.positions.slots() {
                    slot.close();
                }
            }
            // CL-138 (PR70-INT-10 de la Línea G): sin posiciones no hay margen
            // retenido. `close` devuelve el margen de la ranura y aquí se
            // tiraba: el universo replantado nacía con `used_margin` de las
            // posiciones de la hipótesis vieja y menos margen libre.
            engine.arena.used_margin.store(0.0, Ordering::Relaxed);

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

    fn set_capital(forest: &ShadowForest, i: usize, capital: f64) {
        forest.engines[i].arena.unified_capital.store(capital, Ordering::Relaxed);
    }

    fn set_trades(forest: &ShadowForest, i: usize, trades: usize) {
        forest.engines[i].arena.coins[0].metrics.trade_count.store(trades, Ordering::Relaxed);
    }

    // Exercise the public event path; latency_panic prevents new entries in
    // this accounting fixture. The capital observations are injected, not MTM.
    fn broadcast_observation(forest: &mut ShadowForest, event_time: u64) {
        let main_arena = Arc::clone(&forest.engines[0].arena);
        forest.broadcast_tick(
            0, true, false, false, 100.0, 1.0, 99.99, 100.01, 10.0, 10.0,
            0.0, 0.0, event_time, &main_arena, &[0.0; 54], false, true,
        );
    }

    fn expected_fitness(capital: f64, max_dd: f64, trades: usize) -> f64 {
        crate::fitness::compute(&crate::fitness::FitnessInputs {
            initial_capital: 13.0,
            final_capital: capital,
            max_drawdown_pct: max_dd,
            total_trades: trades as u32,
            min_trades_required: MIN_HARVEST_TRADES as u32,
            oos_start_capital: capital, // sin split: convención legacy (SOL-R5-02-B)
            oos_end_capital: capital,
        })
    }

    fn assert_genome_configurations(forest: &ShadowForest) {
        // Compare against the existing apply_to_arena projection, including
        // horizon curves and their scalar views, without redefining that map.
        let expected_arena = GlobalArena::build_in_own_stack(13.0);
        for (i, (engine, genome)) in forest.engines.iter().zip(&forest.genomes).enumerate() {
            genome.apply_to_arena(&expected_arena);
            assert_eq!(
                SuperGenotype::current_from_arena(&engine.arena).to_vector(),
                SuperGenotype::current_from_arena(&expected_arena).to_vector(),
                "configuration differs from stored genome in universe {i}",
            );
        }
    }

    #[test]
    fn e03_constructor_evaluates_the_stored_50ms_latency() {
        let mut base = SuperGenotype::default();
        base.latency_penalty_ms = 50.0;
        let forest = ShadowForest::new(13.0, base, 1);
        assert_eq!(forest.genomes[0].latency_penalty_ms, 50.0);
        assert_eq!(forest.engines[0].arena.config.latency_penalty_ms.load(Ordering::Relaxed), 50.0);
    }

    #[test]
    fn e03_control_and_mutants_evaluate_their_own_genome_configurations() {
        let mut base = SuperGenotype::default();
        base.latency_penalty_ms = 50.0;
        let forest = ShadowForest::new(13.0, base, 4);
        assert_genome_configurations(&forest);
    }

    #[test]
    fn e03_replant_applies_the_new_control_and_mutant_configurations() {
        let mut forest = ShadowForest::new(13.0, SuperGenotype::default(), 4);
        let mut alpha = SuperGenotype::default();
        alpha.latency_penalty_ms = 75.0;
        forest.replant(alpha);
        assert_eq!(forest.genomes[0].latency_penalty_ms, 75.0);
        assert_genome_configurations(&forest);
        broadcast_observation(&mut forest, 1_600_000_000_000);
        assert_genome_configurations(&forest);
    }

    #[test]
    fn e04_harvest_recovery_keeps_max_dd_and_the_stable_control_wins() {
        let mut forest = ShadowForest::new(13.0, SuperGenotype::default(), 2);
        for i in 0..2 { set_trades(&forest, i, 30); }
        forest.harvest_best_genome(); // initial 13
        set_capital(&forest, 1, 6.5);
        forest.harvest_best_genome(); // observe the 50% loss
        set_capital(&forest, 0, 14.0);
        set_capital(&forest, 1, 15.0);
        for _ in 0..3 {
            let (winner, scores) = forest.harvest_best_genome();
            assert_eq!(scores[0], expected_fitness(14.0, 0.0, 30));
            assert_eq!(scores[1], expected_fitness(15.0, 0.5, 30));
            assert!(scores[0] > scores[1], "a recovered loss must still penalize fitness");
            assert!(winner.is_none(), "control 13->14 must beat mutant 13->6.5->15");
        }
    }

    #[test]
    fn e04_broadcast_records_peak_and_loss_between_harvests() {
        let mut forest = ShadowForest::new(13.0, SuperGenotype::default(), 2);
        for i in 0..2 { set_trades(&forest, i, 30); }
        set_capital(&forest, 1, 26.0);
        broadcast_observation(&mut forest, 1_600_000_000_000);
        set_capital(&forest, 1, 13.0);
        broadcast_observation(&mut forest, 1_600_000_000_001);
        set_capital(&forest, 0, 14.0);
        set_capital(&forest, 1, 15.0);
        broadcast_observation(&mut forest, 1_600_000_000_002);
        let (winner, scores) = forest.harvest_best_genome();
        assert_eq!(scores[1], expected_fitness(15.0, 0.5, 30));
        assert_eq!(forest.peak_capital[1], 26.0);
        assert!(scores[0] > scores[1]);
        assert!(winner.is_none());
    }

    #[test]
    fn e04_same_genome_registration_and_generation_keep_history_and_sample() {
        let base = SuperGenotype::default();
        let mut forest = ShadowForest::new(13.0, base.clone(), 2);
        set_trades(&forest, 1, 30);
        set_capital(&forest, 1, 6.5);
        broadcast_observation(&mut forest, 1_600_000_000_000);
        set_capital(&forest, 1, 15.0);
        for generation in [1, 7, 7, 0] {
            assert!(!forest.seguir_generacion(generation, &base));
            assert_eq!(forest.closed_since_replant(1), 30);
            let (_, scores) = forest.harvest_best_genome();
            assert_eq!(scores[1], expected_fitness(15.0, 0.5, 30));
        }
        assert_eq!(forest.generacion_base, 7);
    }

    #[test]
    fn e04_replant_resets_peak_drawdown_and_recent_trade_gate_together() {
        assert_eq!(MIN_HARVEST_TRADES, 15);
        let base = SuperGenotype::default();
        let mut forest = ShadowForest::new(13.0, base.clone(), 2);
        set_trades(&forest, 1, 30);
        set_capital(&forest, 1, 26.0);
        broadcast_observation(&mut forest, 1_600_000_000_000);
        set_capital(&forest, 1, 6.5);
        broadcast_observation(&mut forest, 1_600_000_000_001);
        forest.harvest_best_genome();
        forest.replant(base);
        assert_eq!(forest.peak_capital, vec![13.0; 2]);
        assert_eq!(forest.closed_since_replant(1), 0);
        set_capital(&forest, 1, 14.0);
        for (recent, eligible) in [(0, false), (MIN_HARVEST_TRADES - 1, false), (MIN_HARVEST_TRADES, true)] {
            set_trades(&forest, 1, 30 + recent);
            let (winner, scores) = forest.harvest_best_genome();
            assert_eq!(scores[1], expected_fitness(14.0, 0.0, 30 + recent));
            assert_eq!(winner.is_some(), eligible, "recent trades = {recent}");
        }
    }

    #[test]
    fn e04_invalid_capital_observations_cannot_disappear_after_recovery() {
        let base = SuperGenotype::default();
        let mut forest = ShadowForest::new(13.0, base.clone(), 2);
        for via_broadcast in [false, true] {
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
                forest.replant(base.clone());
                let trades = closed_trades(&forest.engines[1]) + 30;
                set_trades(&forest, 1, trades);
                set_capital(&forest, 1, invalid);
                if via_broadcast {
                    broadcast_observation(&mut forest, 1_600_000_000_000);
                } else {
                    let (winner, scores) = forest.harvest_best_genome();
                    assert!(winner.is_none());
                    assert_eq!(scores[1], crate::fitness::INVIABLE);
                }
                set_capital(&forest, 1, 15.0);
                assert!(!forest.seguir_generacion(7, &base));
                let (winner, scores) = forest.harvest_best_genome();
                assert_eq!(scores[1], crate::fitness::INVIABLE, "invalid={invalid}, broadcast={via_broadcast}");
                assert!(winner.is_none());
                forest.replant(base.clone());
                set_trades(&forest, 1, trades + MIN_HARVEST_TRADES);
                set_capital(&forest, 1, 14.0);
                assert!(forest.harvest_best_genome().0.is_some(), "replant starts valid observations again");
            }
        }
    }

    #[test]
    fn e04_depth_and_kline_broadcasts_also_record_observed_capital() {
        let base = SuperGenotype::default();
        let mut forest = ShadowForest::new(13.0, base.clone(), 2);
        for (is_depth, is_kline_closed) in [(true, false), (false, true)] {
            forest.replant(base.clone());
            let trades = closed_trades(&forest.engines[1]) + 30;
            set_trades(&forest, 1, trades);
            let main_arena = Arc::clone(&forest.engines[0].arena);
            for (offset, capital) in [(0, 26.0), (1, 13.0), (2, 15.0)] {
                set_capital(&forest, 1, capital);
                forest.broadcast_tick(
                    0, false, is_kline_closed, is_depth, 100.0, 1.0,
                    99.99, 100.01, 10.0, 10.0, 0.0, 0.0,
                    1_600_000_000_000 + offset, &main_arena, &[0.0; 54], false, true,
                );
            }
            let (_, scores) = forest.harvest_best_genome();
            assert_eq!(scores[1], expected_fitness(15.0, 0.5, trades));
        }
    }

    #[test]
    fn e04_memory_suspension_preserves_the_observed_capital_history() {
        let mut forest = ShadowForest::new(13.0, SuperGenotype::default(), 2);
        let main_arena = GlobalArena::build_in_own_stack(13.0);
        main_arena.panic_memory_dump.store(true, Ordering::Relaxed);
        set_trades(&forest, 1, 30);
        for (offset, capital) in [(0, 26.0), (1, 13.0), (2, 15.0)] {
            set_capital(&forest, 1, capital);
            forest.broadcast_tick(
                0, true, false, false, 100.0, 1.0, 99.99, 100.01,
                10.0, 10.0, 0.0, 0.0, 1_600_000_000_000 + offset,
                &main_arena, &[0.0; 54], false, true,
            );
        }
        assert_eq!(forest.engines[1].arena.coins[0].current_price.load(Ordering::Relaxed), 0.0);
        let (_, scores) = forest.harvest_best_genome();
        assert_eq!(scores[1], expected_fitness(15.0, 0.5, 30));
    }

    #[test]
    fn e04_positive_finite_capital_extremes_keep_finite_observed_risk() {
        for initial in [f64::from_bits(2), f64::MIN_POSITIVE, f64::MAX] {
            let mut forest = ShadowForest::new(initial, SuperGenotype::default(), 2);
            set_trades(&forest, 1, 30);
            set_capital(&forest, 1, initial / 2.0);
            let (_, scores) = forest.harvest_best_genome();
            assert!(scores[1].is_finite(), "positive finite endpoints: {initial}");
            set_capital(&forest, 1, initial);
            let (_, recovered) = forest.harvest_best_genome();
            assert_eq!(recovered[1], -crate::fitness::DRAWDOWN_LAMBDA * 0.5 * 0.5);
        }
    }

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

    /// CL-40: el bosque replanta cuando el almacén sanciona una generación
    /// más nueva, y sólo entonces (sus núcleos ya no la adoptan solos).
    #[test]
    fn cl40_el_bosque_sigue_la_generacion_sancionada() {
        let base = SuperGenotype::default();
        let mut forest = ShadowForest::new(13.0, base.clone(), 2);
        assert_eq!(forest.generacion_base, 0);

        let mut nuevo = base.clone();
        nuevo.tech_threshold = 0.29;
        forest.engines[1].arena.unified_capital.store(14.0, Ordering::Relaxed);
        assert!(forest.seguir_generacion(3, &nuevo));
        assert_eq!(forest.generacion_base, 3);
        assert_eq!(forest.genomes[0].tech_threshold, 0.29, "el control es el genoma sancionado");
        assert_eq!(
            forest.engines[1].arena.unified_capital.load(Ordering::Relaxed),
            13.0,
            "replantar reinicia el capital de cada universo"
        );

        // Re-registro del mismo genoma con una generación nueva: avanza la
        // base y conserva la muestra (capital y mutantes intactos).
        forest.engines[1].arena.unified_capital.store(14.0, Ordering::Relaxed);
        let mutante = forest.genomes[1].clone();
        assert!(!forest.seguir_generacion(4, &nuevo));
        assert_eq!(forest.generacion_base, 4);
        assert_eq!(forest.engines[1].arena.unified_capital.load(Ordering::Relaxed), 14.0);
        assert!(crate::online_daemon::same_genome(&forest.genomes[1], &mutante));
    }

    /// CL-40c: el almacén puede repetir o bajar el número de generación (dos
    /// promotores que leen el mismo padre, un `active.json` ilegible que
    /// cuenta como 0) mientras el demonio y la cosecha aplican su genoma al
    /// arena. El bosque sigue al genoma, no al número.
    #[test]
    fn cl40c_el_bosque_sigue_al_genoma_aunque_el_numero_no_avance() {
        let base = SuperGenotype::default();
        let mut forest = ShadowForest::new(13.0, base.clone(), 2);
        let mut nuevo = base.clone();
        nuevo.tech_threshold = 0.29;
        assert!(forest.seguir_generacion(3, &nuevo));

        // Otro promotor escribió la MISMA generación con otro genoma.
        let mut otro = base.clone();
        otro.tech_threshold = 0.31;
        assert!(forest.seguir_generacion(3, &otro), "generación repetida con otro genoma");
        assert_eq!(forest.genomes[0].tech_threshold, 0.31);

        // El almacén retrocedió (generación 1) y se aplicó un genoma nuevo.
        assert!(forest.seguir_generacion(1, &nuevo), "generación menor con otro genoma");
        assert_eq!(forest.genomes[0].tech_threshold, 0.29);
        assert_eq!(forest.generacion_base, 3, "la base registra la mayor vista");

        // El mismo genoma, releído del JSON (serde sin float_roundtrip puede
        // moverlo 1 ulp la primera vez), es estable a partir de ahí.
        let releido: SuperGenotype =
            serde_json::from_str(&serde_json::to_string(&forest.genomes[0]).unwrap()).unwrap();
        forest.seguir_generacion(5, &releido);
        let otra_vez: SuperGenotype =
            serde_json::from_str(&serde_json::to_string(&releido).unwrap()).unwrap();
        assert!(!forest.seguir_generacion(6, &otra_vez), "la copia releída es estable");
    }

    #[test]
    fn test_shadow_forest_replant_and_broadcast_tick() {
        let base_genome = SuperGenotype::default();
        let mut forest = ShadowForest::new(13.0, base_genome.clone(), 2);
        let main_arena = GlobalArena::build_in_own_stack(13.0);

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

    // ── SOL-R5-02-B (Sol, medición 2026-10-15): la cosecha puntúa equity
    // MARCADA, no caja ─────────────────────────────────────────────────
    //
    // Hallazgo: `harvest_best_genome` leía `unified_capital` (caja) para
    // puntuar cada universo. Con una posición abierta ganadora al momento
    // de la cosecha, la marca no entraba en el fitness: el bosque
    // aprendía a preferir universos que cierran (o que dejan el riesgo
    // sin valuar), igual que el replay antes del fix de GLM-123. El slot
    // OOS cargaba además la trayectoria IS (`initial → caja`),
    // amplificando las pérdidas IS que `ln(caja/inicial)` ya penaliza —
    // doble conteo sin ser degradación fuera de muestra.
    use quantum_arena::position::PositionHorizon;

    fn abrir_largo(engine: &GodEngineCore, entry: f64, qty: f64) {
        engine.arena.coins[0].positions.slot_2.open_with_fee(
            true,
            entry,
            qty,
            qty * entry / 10.0, // margen 10×
            1_600_000_000_000,
            entry * 1.10,
            entry * 0.90,
            PositionHorizon::Continuous,
            0.6,
            0.5,
            0.0,
        );
    }

    #[test]
    fn sol_r5_02_b_la_cosecha_marca_la_posicion_abierta() {
        let mut forest = ShadowForest::new(13.0, SuperGenotype::default(), 2);
        for i in 0..2 {
            set_trades(&forest, i, 30);
            forest.engines[i].arena.config.live_taker_fee.store(0.0005, Ordering::Relaxed);
            forest.engines[i].arena.config.base_slippage_floor.store(0.0002, Ordering::Relaxed);
        }
        // Universo 1: caja plana 13.0 y un largo ganador abierto
        // (entry 100, qty 0.1, mid 110 ⇒ bruto +1.0, fricción 11×0.0007).
        abrir_largo(&forest.engines[1], 100.0, 0.1);
        forest.engines[1].arena.coins[0].current_price.store(110.0, Ordering::Relaxed);

        let (winner, scores) = forest.harvest_best_genome();

        // Equity marcada esperada: 13.0 + (1.0 − 11.0×0.0007) = 13.9923.
        let marcado = 13.0 + 1.0 - 11.0 * 0.0007;
        let esperado = crate::fitness::compute(&crate::fitness::FitnessInputs {
            initial_capital: 13.0,
            final_capital: marcado,
            max_drawdown_pct: 0.0,
            total_trades: 30,
            min_trades_required: MIN_HARVEST_TRADES as u32,
            oos_start_capital: marcado, // sin split: convención legacy
            oos_end_capital: marcado,
        });
        assert!(
            (scores[1] - esperado).abs() < 1e-9,
            "score del universo con marca {:.6} != esperado {:.6} — la cosecha debe puntuar equity marcada",
            scores[1],
            esperado
        );
        assert!(
            winner.is_some(),
            "el universo con la posición ganadora abierta debe ganar la cosecha (hoy la caja plana empata)"
        );
        // El control (caja plana, sin posiciones) puntúa exactamente 0.
        assert_eq!(scores[0], 0.0);
    }

    #[test]
    fn sol_r5_02_b_sin_split_el_factor_oos_no_amplifica_perdidas_is() {
        let mut forest = ShadowForest::new(13.0, SuperGenotype::default(), 2);
        for i in 0..2 {
            set_trades(&forest, i, 30);
        }
        set_capital(&forest, 1, 6.5); // −50 % IS, sin posiciones
        let (_, scores) = forest.harvest_best_genome();
        // Sin split, el score es el crecimiento log puro MENOS la
        // penalización de drawdown observado (la cosecha vio la caída
        // 13→6.5 ⇒ dd = 0.5), con factor OOS 1.0 — SIN el amplificador
        // 1+|ln| que medí en RED (−2.3472 vs −1.3863 = ×1.6931).
        let puro = (6.5_f64 / 13.0).ln()
            - crate::fitness::DRAWDOWN_LAMBDA * 0.5 * 0.5;
        assert!(
            (scores[1] - puro).abs() < 1e-9,
            "score {:.6} != log puro {:.6} — el slot OOS no puede amplificar pérdidas IS",
            scores[1],
            puro
        );
    }
}
