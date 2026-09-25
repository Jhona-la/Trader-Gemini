use crate::GodEngineCore;
use quantum_arena::GlobalArena;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use quantum_arena::tick_source::TickEvent;
use rand::RngExt;

/// Axioma X: The Darwin Daemon
/// Legacy optional GA on a recent tick window; not a certification of online
/// adaptation or out-of-sample superiority. XXXV shares replay and drawdown
/// policy for candidate/baseline. Synthetic context, incomplete configuration
/// snapshots and promotion-before-persistence remain open.

#[derive(Debug, Clone)]
pub struct Genotype {
    pub global_leverage: f64,
    pub trend_threshold: f64,
    pub maker_spread_pct: f64,
    pub maker_obi_threshold: f64,
    pub tp_curve_a: f64,
    pub tp_curve_b: f64,
    pub sl_curve_a: f64,
    pub sl_curve_b: f64,
    pub scalp_obi_threshold: f64,
    pub capital_split_scalp: f64,
    pub min_confidence: f64,
    pub explosive_leverage_multiplier: f64,
}

impl Genotype {
    #[inline]
    pub fn scalp_tp(&self) -> f64 {
        quantum_arena::temporal_spectrum::HorizonCurve { a: self.tp_curve_a, b: self.tp_curve_b }
            .eval(quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS)
    }

    #[inline]
    pub fn scalp_sl(&self) -> f64 {
        quantum_arena::temporal_spectrum::HorizonCurve { a: self.sl_curve_a, b: self.sl_curve_b }
            .eval(quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS)
    }

    pub fn new_random() -> Self {
        Self {
            global_leverage: rand::rng().random_range(10.0..125.0),
            trend_threshold: rand::rng().random_range(0.3..0.8),
            maker_spread_pct: rand::rng().random_range(0.0001..0.0020),
            maker_obi_threshold: rand::rng().random_range(0.3..0.9),
            // Curvas continuas espectrales: TP(τ) = exp(a + b·ln(τ))
            tp_curve_a: rand::rng().random_range(-7.5..-4.0),
            tp_curve_b: rand::rng().random_range(0.01..0.25),
            sl_curve_a: rand::rng().random_range(-8.5..-5.0),
            sl_curve_b: rand::rng().random_range(0.01..0.25),
            scalp_obi_threshold: rand::rng().random_range(0.20..0.80),
            capital_split_scalp: rand::rng().random_range(0.1..1.0),
            min_confidence: rand::rng().random_range(0.5..0.95),
            explosive_leverage_multiplier: rand::rng().random_range(1.0..5.0),
        }
    }

    pub fn current_from_arena(arena: &GlobalArena) -> Self {
        Self {
            global_leverage: arena.config.global_leverage.load(Ordering::Relaxed),
            trend_threshold: arena.config.trend_threshold.load(Ordering::Relaxed),
            maker_spread_pct: arena.config.maker_spread_pct.load(Ordering::Relaxed),
            maker_obi_threshold: arena.config.maker_obi_threshold.load(Ordering::Relaxed),
            tp_curve_a: arena.config.tp_curve_a.load(Ordering::Relaxed),
            tp_curve_b: arena.config.tp_curve_b.load(Ordering::Relaxed),
            sl_curve_a: arena.config.sl_curve_a.load(Ordering::Relaxed),
            sl_curve_b: arena.config.sl_curve_b.load(Ordering::Relaxed),
            scalp_obi_threshold: arena.config.scalp_obi_threshold.load(Ordering::Relaxed),
            capital_split_scalp: arena.config.capital_split_scalp.load(Ordering::Relaxed),
            min_confidence: arena.config.min_confidence_btc.load(Ordering::Relaxed),
            explosive_leverage_multiplier: arena
                .config
                .explosive_leverage_multiplier
                .load(Ordering::Relaxed),
        }
    }

    pub fn apply_to_arena(&self, arena: &GlobalArena) {
        // Sanitizar parámetros de genoma antes de almacenar en atómicos
        let g_lev = if self.global_leverage.is_finite() && self.global_leverage >= 1.0 {
            self.global_leverage
        } else {
            10.0
        };
        let t_th = if self.trend_threshold.is_finite() {
            self.trend_threshold
        } else {
            0.5
        };
        let m_sp = if self.maker_spread_pct.is_finite() && self.maker_spread_pct > 0.0 {
            self.maker_spread_pct
        } else {
            0.0005
        };
        let m_obi = if self.maker_obi_threshold.is_finite() {
            self.maker_obi_threshold
        } else {
            0.5
        };
        let tp_a = if self.tp_curve_a.is_finite() {
            self.tp_curve_a.clamp(-9.5, -2.0)
        } else {
            -5.8
        };
        let tp_b = if self.tp_curve_b.is_finite() {
            self.tp_curve_b.clamp(-0.2, 0.35)
        } else {
            0.12
        };
        let sl_a = if self.sl_curve_a.is_finite() {
            self.sl_curve_a.clamp(-10.5, -3.0)
        } else {
            -6.5
        };
        let sl_b = if self.sl_curve_b.is_finite() {
            self.sl_curve_b.clamp(-0.2, 0.35)
        } else {
            0.10
        };

        let cap_sp = if self.capital_split_scalp.is_finite() {
            self.capital_split_scalp
        } else {
            0.5
        };
        let min_conf = if self.min_confidence.is_finite() {
            self.min_confidence
        } else {
            0.65
        };
        let exp_lev = if self.explosive_leverage_multiplier.is_finite()
            && self.explosive_leverage_multiplier >= 1.0
        {
            self.explosive_leverage_multiplier
        } else {
            1.5
        };

        arena.config.global_leverage.store(g_lev, Ordering::Relaxed);
        arena.config.trend_threshold.store(t_th, Ordering::Relaxed);
        arena.config.maker_spread_pct.store(m_sp, Ordering::Relaxed);
        arena
            .config
            .maker_obi_threshold
            .store(m_obi, Ordering::Relaxed);
        arena.config.tp_curve_a.store(tp_a, Ordering::Relaxed);
        arena.config.tp_curve_b.store(tp_b, Ordering::Relaxed);
        arena.config.sl_curve_a.store(sl_a, Ordering::Relaxed);
        arena.config.sl_curve_b.store(sl_b, Ordering::Relaxed);

        // Actualizar anclas derivadas para módulos legacy y observabilidad
        let tp_c = quantum_arena::temporal_spectrum::HorizonCurve { a: tp_a, b: tp_b };
        let sl_c = quantum_arena::temporal_spectrum::HorizonCurve { a: sl_a, b: sl_b };
        arena.config.scalp_tp_base.store(tp_c.eval(quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS), Ordering::Relaxed);
        arena.config.swing_tp_base.store(tp_c.eval(quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS), Ordering::Relaxed);
        arena.config.scalp_sl_base.store(sl_c.eval(quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS), Ordering::Relaxed);
        arena.config.swing_sl_base.store(sl_c.eval(quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS), Ordering::Relaxed);
        let s_obi = if self.scalp_obi_threshold.is_finite() {
            self.scalp_obi_threshold.clamp(0.05, 0.95)
        } else {
            0.55
        };
        arena.config.scalp_obi_threshold.store(s_obi, Ordering::Relaxed);
        arena
            .config
            .capital_split_scalp
            .store(cap_sp, Ordering::Relaxed);
        arena
            .config
            .min_confidence_btc
            .store(min_conf, Ordering::Relaxed);
        // D-715 (DÉCIMA OLA · auditoría integral): UN GENOTIPO NO SOBRESCRIBE
        // GENES QUE NO LLEVA.
        //
        // Aquí se derivaban `ml_threshold_long`/`ml_threshold_short` de
        // `min_confidence`, que no tiene nada que ver con ellos: `Genotype` sólo
        // porta doce genes y esos dos no están entre ellos. Como `min_confidence`
        // vive acotada a [0,50; 0,95], el `.abs()` era inerte y el resultado
        // exacto era `ml_long = min_confidence`, `ml_short = 1 − min_confidence`:
        // dos umbrales forzados a ser espejos, sin asimetría posible entre largo
        // y corto. Con el valor por defecto, la puerta que decide TODAS las
        // entradas pasaba del par validado cross-month (0,5698 / 0,4302) a
        // (0,65 / 0,35) cada vez que este daemon aplicaba un genotipo —21 veces
        // por generación sobre arenas nuevas, y también sobre la arena VIVA—,
        // sin que ninguna validación hubiera visto ese par.
        //
        // Los dos umbrales son genes del `SuperGenotype` y se aplican desde él.
        arena
            .config
            .explosive_leverage_multiplier
            .store(exp_lev, Ordering::Relaxed);
    }
}

/// CERT-M2-H05 — Sintetizador OMNI causal para el GA de Darwin.
///
/// El GA copiaba `get_universal_features()[0..34]` en `omni[0..34]`:
/// el slot de funding recibía aceleración (micro[11]) y el bloque macro
/// quedaba en 0 — el GA optimizaba genes contra un tensor que producción
/// JAMÁS ve. Este sintetizador deriva el 54D del PROPIO tick de forma
/// causal (mismo contrato y convenciones que run_backtest_native::omni_sim
/// y que producción con feeds secundarios offline):
///   - spreads: sólo la venue local (binance futures) — el resto 0.0
///     (paridad con "secondary WS feeds offline")
///   - OFI/CVD/taker del imbalance bid/ask REAL del tick
///   - funding/L&S/F&G/VIX/skew/micro_vol derivados del retorno PREVIO
///     (causal: calculado con el mid anterior, nunca el futuro)
///   - anclas macro estáticas neutras (idénticas al nativo)
/// Las swing-features NO se inyectan: el core las computa internamente
/// desde los ticks (feature_engines) — inyectarlas era doble conteo.
pub(crate) struct OmniSynth {
    prev_mid: Vec<f64>,
    ewma_turnover: Vec<f64>,
}

impl OmniSynth {
    pub(crate) fn new(n_coins: usize) -> Self {
        Self {
            prev_mid: vec![0.0; n_coins],
            ewma_turnover: vec![0.0; n_coins],
        }
    }

    pub(crate) fn tick(
        &mut self,
        coin_id: usize,
        bid: f64,
        ask: f64,
        bid_qty: f64,
        ask_qty: f64,
    ) -> [f64; 54] {
        let mut omni = [0.0f64; 54];
        let mid = if bid > 0.0 && ask > 0.0 { (bid + ask) * 0.5 } else { 0.0 };
        let prev_mid = self.prev_mid[coin_id];
        let prev_ret = if mid > 0.0 && prev_mid > 0.0 {
            ((mid - prev_mid) / prev_mid).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let spread_pct = if mid > 0.0 { (ask - bid) / mid } else { 0.0 };
        let ofi = if bid_qty + ask_qty > 0.0 {
            (bid_qty - ask_qty) / (bid_qty + ask_qty)
        } else {
            0.0
        };
        let turnover = (bid_qty + ask_qty) * mid;
        self.ewma_turnover[coin_id] = if self.ewma_turnover[coin_id] <= 0.0 {
            turnover
        } else {
            self.ewma_turnover[coin_id] * 0.98 + turnover * 0.02
        };
        let cvd_unit = if self.ewma_turnover[coin_id] > 1e-12 {
            (bid_qty - ask_qty) * mid / self.ewma_turnover[coin_id]
        } else {
            0.0
        };

        // 0..10 — cotizaciones cross-exchange (sólo venue local viva)
        omni[0] = 0.0;
        omni[1] = (spread_pct * 10.0).clamp(-5.0, 5.0); // binance_futures
        // 2..10 = 0.0 — paridad feeds secundarios offline
        omni[10] = ofi.abs() * 10.0; // liquidaciones proxy

        // 11..30 — sentiment/tasas/macro dinámicos del retorno PREVIO
        omni[11] = (prev_ret * 0.005).clamp(-0.001, 0.001); // funding
        omni[12] = (self.ewma_turnover[coin_id] / 1.0e6).clamp(0.0, 100.0); // OI proxy
        omni[13] = (1.0 + prev_ret * 5.0).clamp(0.5, 2.5); // long/short
        omni[14] = (50.0 + prev_ret * 500.0).clamp(10.0, 90.0); // fear&greed
        omni[18] = cvd_unit.clamp(-5.0, 5.0); // exchange inflows
        omni[19] = (-cvd_unit).clamp(-5.0, 5.0); // exchange outflows
        omni[21] = 104.2; // dxy (ancla estática — paridad nativo)
        omni[22] = 5120.0; // sp500
        omni[23] = 18100.0; // nasdaq
        omni[24] = (15.0 + prev_ret.abs() * 200.0).clamp(10.0, 80.0); // vix
        omni[25] = 4.25; // us10y
        omni[26] = 2320.0; // gold
        omni[27] = 81.0; // oil_wti
        omni[29] = 5.5; // fed funds

        // 30..54 — derivados y flujo
        omni[30] = cvd_unit.clamp(-10.0, 10.0); // spot_cvd (normalizado)
        omni[31] = omni[30] * 1.2; // futures_cvd (convención nativa)
        omni[32] = if ask_qty > 0.0 {
            (bid_qty / ask_qty).clamp(0.1, 10.0)
        } else {
            1.0
        };
        omni[33] = prev_ret * mid * 0.001; // basis premium
        omni[34] = mid * 1.01; // liq cluster shorts
        omni[35] = mid * 0.99; // liq cluster longs
        omni[39] = ofi; // order_flow_imbalance
        omni[40] = (15.0 + prev_ret.abs() * 300.0).clamp(20.0, 150.0); // dvol
        omni[41] = (-prev_ret * 2.0).clamp(-0.25, 0.25); // 25Δ skew
        omni[43] = mid; // max pain
        omni[49] = prev_ret.abs() * 100.0; // micro_volatility

        if mid > 0.0 {
            self.prev_mid[coin_id] = mid;
        }
        omni
    }
}

/// Legacy promotion margin, NOT statistical significance or publication authority.
/// Nonfinite scores (including insufficient evidence) are not measured utilities.
pub fn meets_promotion_margin(candidate: f64, baseline: f64) -> bool {
    if !candidate.is_finite() || !baseline.is_finite() {
        return false;
    }
    if baseline >= 0.0 {
        candidate > (baseline * 1.05).max(baseline + 1e-4)
    } else {
        candidate > baseline && candidate > baseline * 0.95
    }
}

fn replay_arena(genome: &Genotype, initial: f64, max_drawdown: f64) -> Arc<GlobalArena> {
    let arena = GlobalArena::build_in_own_stack(initial);
    genome.apply_to_arena(&arena);
    arena.config.global_max_drawdown.store(max_drawdown, Ordering::Relaxed);
    arena
}

/// Common replay, still with synthetic macros and realized-close drawdown.
/// Defaults outside the copied genes and dynamically loaded models are NOT
/// a frozen live configuration/model snapshot or an out-of-sample experiment.
fn evaluate_genotype(
    genome: &Genotype,
    stream: &[TickEvent],
    initial: f64,
    max_drawdown_policy: f64,
    active_coins: usize,
) -> (f64, f64) {
    if !initial.is_finite() || initial <= 0.0 || !max_drawdown_policy.is_finite() {
        return (initial, f64::NEG_INFINITY);
    }
    let arena = replay_arena(genome, initial, max_drawdown_policy);
    let mut engine = GodEngineCore::new(arena.clone());
    let mut synth = OmniSynth::new(active_coins);
    let mut peak_capital = initial;
    let mut max_drawdown = 0.0_f64;
    let mut trades = 0_u32;
    for tick in stream {
        arena.update_market_data(tick.coin_id, tick.bid_price, tick.ask_price,
            tick.bid_qty, tick.ask_qty, tick.timestamp);
        let omni = synth.tick(tick.coin_id, tick.bid_price, tick.ask_price, tick.bid_qty, tick.ask_qty);
        let (_, closed, _) = engine.process_tick(tick.coin_id, tick.bid_price, tick.ask_price,
            tick.bid_qty, tick.ask_qty, tick.timestamp, &omni);
        if closed.is_some() {
            trades += 1;
            let capital = arena.unified_capital.load(Ordering::Relaxed);
            if !capital.is_finite() || capital <= 0.0 {
                return (capital, f64::NEG_INFINITY);
            }
            peak_capital = peak_capital.max(capital);
            max_drawdown = max_drawdown.max((peak_capital - capital) / peak_capital);
        }
    }
    let final_capital = arena.unified_capital.load(Ordering::Relaxed);
    (final_capital, crate::fitness_compute(initial, final_capital, max_drawdown, trades))
}

pub struct DarwinDaemon {
    pub live_arena: Arc<GlobalArena>,
}

impl DarwinDaemon {
    pub fn new(live_arena: Arc<GlobalArena>) -> Self {
        Self { live_arena }
    }

    /// Extacts the recent ticks from the live arena, sorts them, and runs a fast GA
    pub fn evolve_online(&self) {
        let active_coins = self.live_arena.coins.len().min(30);
        let mut master_stream = Vec::with_capacity(active_coins * 4096);

        // 1. Extract memory snapshot across all active coins (lock-free: snapshot_recent never blocks the writer)
        for coin_id in 0..active_coins {
            let ticks = self.live_arena.coins[coin_id]
                .tick_ring
                .snapshot_recent(4096);

            for tick in ticks {
                master_stream.push(TickEvent {
                    coin_id,
                    timestamp: tick.timestamp,
                    bid_price: tick.bid_price,
                    ask_price: tick.ask_price,
                    bid_qty: tick.bid_qty,
                    ask_qty: tick.ask_qty,
                });
            }
        }

        if master_stream.is_empty() {
            return;
        }

        // FASE FORENSE: Preservar ordenamiento cronológico multiactivo estricto
        master_stream.sort_unstable_by_key(|t| t.timestamp);

        println!(
            "[Darwin] Extracted {} recent ticks. Starting online evolution...",
            master_stream.len()
        );

        let pop_size = 20; // Fast mini-evolution
        let generations = 5;
        let mutation_rate = 0.3;

        let mut population: Vec<Genotype> = (0..pop_size).map(|_| Genotype::new_random()).collect();
        // Ensure current active genotype is in the pool (Elitism baseline)
        let current_active = Genotype::current_from_arena(&self.live_arena);
        population[0] = current_active.clone();

        let initial_capital = self.live_arena.unified_capital.load(Ordering::Relaxed);
        // Capture this risk constraint once for BOTH sides, not a transactional
        // snapshot of the whole live configuration.
        let replay_max_drawdown = self.live_arena.config.global_max_drawdown.load(Ordering::Relaxed);
        if !initial_capital.is_finite() || initial_capital <= 0.0 || !replay_max_drawdown.is_finite() {
            println!("[Darwin] Evaluation skipped: invalid capital or drawdown policy.");
            return;
        }
        // M5-H01: arranca en INVIABLE, no en 0.0 — con 0.0, una corrida donde
        // ningún genoma aclara el gate min_trades (todos -inf) o donde todos
        // pierden deja este valor FANTASMA, y FIX#412 lo leería como "mejor
        // que cualquier baseline negativo/gateado" promoviendo population[0]
        // sin evidencia. Con -inf, sólo un fitness real lo reemplaza.
        let mut best_all_time = (population[0].clone(), f64::NEG_INFINITY);

        for generation in 1..=generations {
            let mut results: Vec<_> = population
                .par_iter()
                .map(|genome| {
                    let (final_cap, fitness) = evaluate_genotype(
                        genome, &master_stream, initial_capital, replay_max_drawdown, active_coins,
                    );
                    (genome.clone(), final_cap, fitness)
                })
                .collect();

            results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            let best_gen = &results[0];

            if best_gen.2 > best_all_time.1 {
                best_all_time = (best_gen.0.clone(), best_gen.2);
            }

            if generation == generations {
                break;
            }

            let mut next_gen = Vec::with_capacity(pop_size);
            for i in 0..(pop_size / 4) {
                next_gen.push(results[i].0.clone());
            } // Top 25% elites

            while next_gen.len() < pop_size {
                let p1 = &results[rand::rng().random_range(0..(pop_size / 2))].0;
                let p2 = &results[rand::rng().random_range(0..(pop_size / 2))].0;

                let mut child = Genotype {
                    global_leverage: if rand::rng().random_bool(0.5) {
                        p1.global_leverage
                    } else {
                        p2.global_leverage
                    },
                    trend_threshold: if rand::rng().random_bool(0.5) {
                        p1.trend_threshold
                    } else {
                        p2.trend_threshold
                    },
                    maker_spread_pct: if rand::rng().random_bool(0.5) {
                        p1.maker_spread_pct
                    } else {
                        p2.maker_spread_pct
                    },
                    maker_obi_threshold: if rand::rng().random_bool(0.5) {
                        p1.maker_obi_threshold
                    } else {
                        p2.maker_obi_threshold
                    },
                    tp_curve_a: if rand::rng().random_bool(0.5) {
                        p1.tp_curve_a
                    } else {
                        p2.tp_curve_a
                    },
                    tp_curve_b: if rand::rng().random_bool(0.5) {
                        p1.tp_curve_b
                    } else {
                        p2.tp_curve_b
                    },
                    sl_curve_a: if rand::rng().random_bool(0.5) {
                        p1.sl_curve_a
                    } else {
                        p2.sl_curve_a
                    },
                    sl_curve_b: if rand::rng().random_bool(0.5) {
                        p1.sl_curve_b
                    } else {
                        p2.sl_curve_b
                    },
                    scalp_obi_threshold: if rand::rng().random_bool(0.5) {
                        p1.scalp_obi_threshold
                    } else {
                        p2.scalp_obi_threshold
                    },
                    capital_split_scalp: if rand::rng().random_bool(0.5) {
                        p1.capital_split_scalp
                    } else {
                        p2.capital_split_scalp
                    },
                    min_confidence: if rand::rng().random_bool(0.5) {
                        p1.min_confidence
                    } else {
                        p2.min_confidence
                    },
                    explosive_leverage_multiplier: if rand::rng().random_bool(0.5) {
                        p1.explosive_leverage_multiplier
                    } else {
                        p2.explosive_leverage_multiplier
                    },
                };

                if rand::rng().random_bool(mutation_rate) {
                    child.global_leverage *= rand::rng().random_range(0.8..1.2);
                }
                if rand::rng().random_bool(mutation_rate) {
                    child.trend_threshold *= rand::rng().random_range(0.9..1.1);
                }
                if rand::rng().random_bool(mutation_rate) {
                    child.maker_spread_pct *= rand::rng().random_range(0.5..2.0);
                }
                if rand::rng().random_bool(mutation_rate) {
                    child.maker_obi_threshold *= rand::rng().random_range(0.8..1.2);
                }
                if rand::rng().random_bool(mutation_rate) {
                    child.tp_curve_a += rand::rng().random_range(-0.25..0.25);
                }
                if rand::rng().random_bool(mutation_rate) {
                    child.tp_curve_b += rand::rng().random_range(-0.02..0.02);
                }
                if rand::rng().random_bool(mutation_rate) {
                    child.sl_curve_a += rand::rng().random_range(-0.25..0.25);
                }
                if rand::rng().random_bool(mutation_rate) {
                    child.sl_curve_b += rand::rng().random_range(-0.02..0.02);
                }
                if rand::rng().random_bool(mutation_rate) {
                    child.scalp_obi_threshold *= rand::rng().random_range(0.9..1.1);
                }
                if rand::rng().random_bool(mutation_rate) {
                    child.capital_split_scalp *= rand::rng().random_range(0.8..1.2);
                }
                if rand::rng().random_bool(mutation_rate) {
                    child.min_confidence *= rand::rng().random_range(0.9..1.1);
                }
                if rand::rng().random_bool(mutation_rate) {
                    child.explosive_leverage_multiplier *= rand::rng().random_range(0.5..2.0);
                }

                child.global_leverage = child.global_leverage.clamp(1.0, 125.0);
                child.trend_threshold = child.trend_threshold.clamp(0.1, 0.9);
                child.maker_spread_pct = child.maker_spread_pct.clamp(0.0001, 0.05);
                child.maker_obi_threshold = child.maker_obi_threshold.clamp(0.1, 0.95);
                child.tp_curve_a = child.tp_curve_a.clamp(-9.5, -2.0);
                child.tp_curve_b = child.tp_curve_b.clamp(-0.2, 0.35);
                child.sl_curve_a = child.sl_curve_a.clamp(-10.5, -3.0);
                child.sl_curve_b = child.sl_curve_b.clamp(-0.2, 0.35);
                child.scalp_obi_threshold = child.scalp_obi_threshold.clamp(0.05, 0.95);
                child.capital_split_scalp = child.capital_split_scalp.clamp(0.1, 1.0);
                child.min_confidence = child.min_confidence.clamp(0.50, 0.95);
                child.explosive_leverage_multiplier =
                    child.explosive_leverage_multiplier.clamp(1.0, 10.0);

                next_gen.push(child);
            }
            population = next_gen;
        }

        let (_, baseline_fitness) = evaluate_genotype(
            &current_active, &master_stream, initial_capital, replay_max_drawdown, active_coins,
        );

        println!("[Darwin] Online Evolution Complete.");
        println!("         Current Active Fitness: {:.4}", baseline_fitness);
        println!("         Evolved Genome Fitness: {:.4}", best_all_time.1);

        // Missing/invalid evidence is not a measured loss. The inherited finite
        // margin is a policy, not significance or a structural-change detector.
        let clears_margin = meets_promotion_margin(best_all_time.1, baseline_fitness);

        let allow_hotswap = std::env::var("ENABLE_ONLINE_DARWIN_MUTATION")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);
        if clears_margin && allow_hotswap {
            println!("[Darwin] 🧬 Candidate cleared the configured in-window margin; authorized legacy promotion.");
            best_all_time.0.apply_to_arena(&self.live_arena);

            let mut full_genotype =
                quantum_arena::genome::SuperGenotype::current_from_arena(&self.live_arena);
            full_genotype.global_leverage = best_all_time.0.global_leverage;
            full_genotype.trend_threshold = best_all_time.0.trend_threshold;
            full_genotype.maker_spread_pct = best_all_time.0.maker_spread_pct;
            full_genotype.maker_obi_threshold = best_all_time.0.maker_obi_threshold;
            full_genotype.scalp_obi_threshold = best_all_time.0.scalp_obi_threshold;
            full_genotype.tp_horizon_curve = quantum_arena::temporal_spectrum::HorizonCurve {
                a: best_all_time.0.tp_curve_a,
                b: best_all_time.0.tp_curve_b,
            };
            full_genotype.sl_horizon_curve = quantum_arena::temporal_spectrum::HorizonCurve {
                a: best_all_time.0.sl_curve_a,
                b: best_all_time.0.sl_curve_b,
            };
            full_genotype.derive_anchors_from_curves();
            full_genotype.capital_split_scalp = best_all_time.0.capital_split_scalp;
            full_genotype.min_confidence_btc = best_all_time.0.min_confidence;
            full_genotype.explosive_leverage_multiplier =
                best_all_time.0.explosive_leverage_multiplier;

            match quantum_arena::genome_store::GenomeEnvelope::promote(
                full_genotype,
                "darwin_daemon",
                &format!(
                    "fitness {:.4} (baseline {:.4})",
                    best_all_time.1, baseline_fitness
                ),
            ) {
                Ok(env) => println!(
                    "🧬 [DARWIN] Genoma generación {} promovido vía almacén (padre {}).",
                    env.generation, env.parent_generation
                ),
                Err(e) => println!("⚠️ [DARWIN] Fallo al promover genoma al almacén: {}", e),
            }
        } else {
            println!("[Darwin] No promotion: missing comparable evidence, margin not met, or mutation disabled. No optimality claim.");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genotype_random_and_apply_to_arena() {
        let arena = GlobalArena::build_in_own_stack(13.0);
        let genome = Genotype::new_random();

        assert!(genome.global_leverage >= 10.0 && genome.global_leverage <= 125.0);
        assert!(genome.scalp_tp() > 0.0);
        assert!(genome.scalp_sl() > 0.0);

        genome.apply_to_arena(&arena);

        let roundtrip = Genotype::current_from_arena(&arena);
        assert_eq!(roundtrip.global_leverage, genome.global_leverage);
        assert!((roundtrip.scalp_tp() - genome.scalp_tp()).abs() < 1e-6);
    }

    #[test]
    fn test_genotype_nan_immunity_when_applying_to_arena() {
        let arena = GlobalArena::build_in_own_stack(13.0);
        let nan_genome = Genotype {
            global_leverage: f64::NAN,
            trend_threshold: f64::NAN,
            maker_spread_pct: f64::NAN,
            maker_obi_threshold: f64::NAN,
            tp_curve_a: f64::NAN,
            tp_curve_b: f64::NAN,
            sl_curve_a: f64::NAN,
            sl_curve_b: f64::NAN,
            scalp_obi_threshold: f64::NAN,
            capital_split_scalp: f64::NAN,
            min_confidence: f64::NAN,
            explosive_leverage_multiplier: f64::NAN,
        };

        nan_genome.apply_to_arena(&arena);

        let safe_genome = Genotype::current_from_arena(&arena);
        assert!(safe_genome.global_leverage.is_finite());
        assert!(safe_genome.trend_threshold.is_finite());
        assert!(safe_genome.scalp_tp().is_finite());
        assert!(safe_genome.scalp_sl().is_finite());
        assert!(safe_genome.min_confidence.is_finite());
    }

    #[test]
    fn test_darwin_daemon_instantiation() {
        let arena = GlobalArena::build_in_own_stack(13.0);
        let daemon = DarwinDaemon::new(arena);
        assert!(
            daemon
                .live_arena
                .unified_capital
                .load(std::sync::atomic::Ordering::Relaxed)
                > 0.0
        );
    }

    #[test]
    fn replay_candidate_and_baseline_use_the_same_captured_drawdown_policy() {
        let source = GlobalArena::build_in_own_stack(100.0);
        let genome = Genotype::current_from_arena(&source);
        let candidate = replay_arena(&genome, 100.0, 0.2);
        let baseline = replay_arena(&genome, 100.0, 0.2);
        assert_eq!(candidate.config.global_max_drawdown.load(Ordering::Relaxed), 0.2);
        assert_eq!(baseline.config.global_max_drawdown.load(Ordering::Relaxed), 0.2);
        assert_eq!(candidate.config.global_leverage.load(Ordering::Relaxed),
            baseline.config.global_leverage.load(Ordering::Relaxed));
    }

    #[test]
    fn empty_replay_preserves_missing_evidence_instead_of_a_finite_loss() {
        let source = GlobalArena::build_in_own_stack(100.0);
        let genome = Genotype::current_from_arena(&source);
        let (capital, score) = evaluate_genotype(&genome, &[], 100.0, 0.2, 1);
        assert_eq!(capital, 100.0);
        assert_eq!(score, f64::NEG_INFINITY);
        assert!(!meets_promotion_margin(0.1, score));
    }

    #[test]
    fn invalid_replay_policy_is_rejected_before_constructing_an_engine() {
        let source = GlobalArena::build_in_own_stack(100.0);
        let genome = Genotype::current_from_arena(&source);
        assert_eq!(evaluate_genotype(&genome, &[], 100.0, f64::NAN, 1).1, f64::NEG_INFINITY);
        assert_eq!(evaluate_genotype(&genome, &[], 0.0, 0.2, 1).1, f64::NEG_INFINITY);
    }

    #[test]
    fn same_genome_and_short_tape_have_equal_replay_results() {
        let source = GlobalArena::build_in_own_stack(100.0);
        let genome = Genotype::current_from_arena(&source);
        let stream = [1_000, 2_000, 3_000].map(|timestamp| TickEvent {
            coin_id: 0, timestamp, bid_price: 100.0, ask_price: 100.02,
            bid_qty: 1.0, ask_qty: 1.0,
        });
        let candidate = evaluate_genotype(&genome, &stream, 100.0, 0.2, 1);
        let baseline = evaluate_genotype(&genome, &stream, 100.0, 0.2, 1);
        assert_eq!(candidate, baseline);
        assert!(candidate.0.is_finite());
        assert_eq!(candidate.1, f64::NEG_INFINITY); // three ticks are not thirty closes
    }
}
