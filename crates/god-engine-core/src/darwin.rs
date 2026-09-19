use crate::GodEngineCore;
use quantum_arena::GlobalArena;
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use quantum_arena::tick_source::TickEvent;
use rand::RngExt;

/// Axioma X: The Darwin Daemon
/// Continuous Online Evolution. Evaluates the recent market microstructure
/// and dynamically hot-swaps parameters without stopping the live engine.

#[derive(Debug, Clone)]
pub struct Genotype {
    pub global_leverage: f64,
    pub trend_threshold: f64,
    pub maker_spread_pct: f64,
    pub maker_obi_threshold: f64,
    pub scalp_tp: f64,
    pub scalp_sl: f64,
    pub swing_tp: f64,
    pub swing_sl: f64,
    pub scalp_z_target: f64,
    pub capital_split_scalp: f64,
    pub min_confidence: f64,
    pub explosive_leverage_multiplier: f64,
}

impl Genotype {
    pub fn new_random() -> Self {
        Self {
            global_leverage: rand::rng().random_range(10.0..125.0),
            trend_threshold: rand::rng().random_range(0.3..0.8),
            maker_spread_pct: rand::rng().random_range(0.0001..0.0020),
            maker_obi_threshold: rand::rng().random_range(0.3..0.9),
            // FIX #1505: Rango asimétrico favorable (R:R >= 2:1 a 5:1) en genomas iniciales
            scalp_tp: rand::rng().random_range(0.003..0.012),
            scalp_sl: rand::rng().random_range(0.0005..0.0025),
            swing_tp: rand::rng().random_range(0.006..0.025),
            swing_sl: rand::rng().random_range(0.0015..0.0060),
            scalp_z_target: rand::rng().random_range(1.0..4.0),
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
            scalp_tp: arena.config.scalp_tp_base.load(Ordering::Relaxed),
            scalp_sl: arena.config.scalp_sl_base.load(Ordering::Relaxed),
            swing_tp: arena.config.swing_tp_base.load(Ordering::Relaxed),
            swing_sl: arena.config.swing_sl_base.load(Ordering::Relaxed),
            scalp_z_target: arena.config.scalp_obi_threshold.load(Ordering::Relaxed),
            capital_split_scalp: arena.config.capital_split_scalp.load(Ordering::Relaxed),
            min_confidence: arena.config.min_confidence_btc.load(Ordering::Relaxed),
            explosive_leverage_multiplier: arena
                .config
                .explosive_leverage_multiplier
                .load(Ordering::Relaxed),
        }
    }

    pub fn apply_to_arena(&self, arena: &GlobalArena) {
        // FIX #685: Sanitizar parámetros de genoma antes de almacenar en atómicos
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
        let sc_tp = if self.scalp_tp.is_finite() && self.scalp_tp > 0.0 {
            self.scalp_tp
        } else {
            0.003
        };
        let sc_sl = if self.scalp_sl.is_finite() && self.scalp_sl > 0.0 {
            self.scalp_sl
        } else {
            0.002
        };
        let sw_tp = if self.swing_tp.is_finite() && self.swing_tp > 0.0 {
            self.swing_tp
        } else {
            0.010
        };
        let sw_sl = if self.swing_sl.is_finite() && self.swing_sl > 0.0 {
            self.swing_sl
        } else {
            0.005
        };
        // D-728: `scalp_z_target` ya no se escribe en ningún gen (ver abajo);
        // se conserva en el genotipo porque la evolución lo muta, pero no tiene
        // destino en la configuración hasta que tenga su propio atómico.
        let _sc_z = if self.scalp_z_target.is_finite() {
            self.scalp_z_target
        } else {
            2.0
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
        arena.config.scalp_tp_base.store(sc_tp, Ordering::Relaxed);
        arena.config.scalp_sl_base.store(sc_sl, Ordering::Relaxed);
        arena.config.swing_tp_base.store(sw_tp, Ordering::Relaxed);
        arena.config.swing_sl_base.store(sw_sl, Ordering::Relaxed);
        arena.config.update_tp_curve(sc_tp, sw_tp);
        arena.config.update_sl_curve(sc_sl, sw_sl);
        // D-728 (DÉCIMA OLA · auditoría integral): UN GENOTIPO NO SOBRESCRIBE
        // GENES QUE NO LLEVA (la misma regla de D-715).
        //
        // Aquí se escribía `scalp_z_target` —una desviación típica, inicializada
        // en [1; 4] y con respaldo 2,0— dentro de `scalp_obi_threshold`, que es
        // el gen de desequilibrio del libro y vive en [0,05; 1,0]. Como este
        // daemon aplica sobre la arena VIVA, la puerta de OBI quedaba clavada en
        // su cota máxima y `current_from_arena` releía ese 2,0 como si fuera el
        // gen, corroyendo el genoma en cada lectura. `scalp_z_target` necesita su
        // propio atómico si ha de evolucionar; no el de otro gen.
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

        let initial_capital = self.live_arena.unified_capital.load(Ordering::Relaxed); // Dynamic fitness baseline
        let mut best_all_time = (population[0].clone(), 0.0_f64);

        for generation in 1..=generations {
            let mut results: Vec<_> = population
                .par_iter()
                .map(|genome| {
                    // D-714: construcción con pila suficiente (este sitio corre en un worker de rayon).
                    let arena = GlobalArena::build_in_own_stack(initial_capital);
                    genome.apply_to_arena(&arena);
                    arena
                        .config
                        .global_max_drawdown
                        .store(0.95, Ordering::Relaxed);

                    let mut engine = GodEngineCore::new(arena.clone());
                    let mut max_drawdown = 0.0;
                    let mut peak_capital = initial_capital;
                    // CERT-M2-H05: tensor 54D CAUSAL del propio tick — mismo
                    // contrato que producción/nativo (ver OmniSynth). Antes:
                    // copia de swing-features en omni[0..34] con funding=
                    // aceleración y macro=0.
                    let mut synth = OmniSynth::new(active_coins);

                    for tick in &master_stream {
                        arena.update_market_data(
                            tick.coin_id,
                            tick.bid_price,
                            tick.ask_price,
                            tick.bid_qty,
                            tick.ask_qty,
                            tick.timestamp,
                        );
                        let dynamic_omni = synth.tick(
                            tick.coin_id,
                            tick.bid_price,
                            tick.ask_price,
                            tick.bid_qty,
                            tick.ask_qty,
                        );

                        let (_new_pos, closed_pos, _) = engine.process_tick(
                            tick.coin_id,
                            tick.bid_price,
                            tick.ask_price,
                            tick.bid_qty,
                            tick.ask_qty,
                            tick.timestamp,
                            &dynamic_omni,
                        );

                        if closed_pos.is_some() {
                            let current_cap = arena.unified_capital.load(Ordering::Relaxed);
                            if current_cap > peak_capital {
                                peak_capital = current_cap;
                            }
                            let dd = (peak_capital - current_cap) / peak_capital;
                            if dd > max_drawdown {
                                max_drawdown = dd;
                            }
                        }
                    }

                    let final_cap = arena.unified_capital.load(Ordering::Relaxed);
                    // CERT-M5-H01: fitness UNIFICADO — antes (final−initial)×(1−dd):
                    // PnL crudo sin log-utility, sin INVIABLE para inacción, escalado
                    // por dólares (no por crecimiento relativo). Era la 5ª fn compitiendo.
                    let fitness = crate::fitness_compute(
                        initial_capital,
                        final_cap,
                        max_drawdown,
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
                    scalp_tp: if rand::rng().random_bool(0.5) {
                        p1.scalp_tp
                    } else {
                        p2.scalp_tp
                    },
                    scalp_sl: if rand::rng().random_bool(0.5) {
                        p1.scalp_sl
                    } else {
                        p2.scalp_sl
                    },
                    swing_tp: if rand::rng().random_bool(0.5) {
                        p1.swing_tp
                    } else {
                        p2.swing_tp
                    },
                    swing_sl: if rand::rng().random_bool(0.5) {
                        p1.swing_sl
                    } else {
                        p2.swing_sl
                    },
                    scalp_z_target: if rand::rng().random_bool(0.5) {
                        p1.scalp_z_target
                    } else {
                        p2.scalp_z_target
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
                    child.scalp_tp *= rand::rng().random_range(0.7..1.5);
                }
                if rand::rng().random_bool(mutation_rate) {
                    child.scalp_sl *= rand::rng().random_range(0.7..1.5);
                }
                if rand::rng().random_bool(mutation_rate) {
                    child.swing_tp *= rand::rng().random_range(0.7..1.5);
                }
                if rand::rng().random_bool(mutation_rate) {
                    child.swing_sl *= rand::rng().random_range(0.7..1.5);
                }
                if rand::rng().random_bool(mutation_rate) {
                    child.scalp_z_target *= rand::rng().random_range(0.8..1.2);
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
                child.scalp_tp = child.scalp_tp.clamp(0.0005, 0.02);
                child.scalp_sl = child.scalp_sl.clamp(0.0005, 0.01);
                child.swing_tp = child.swing_tp.clamp(0.001, 0.03);
                child.swing_sl = child.swing_sl.clamp(0.0005, 0.01);
                child.scalp_z_target = child.scalp_z_target.clamp(0.5, 5.0);
                child.capital_split_scalp = child.capital_split_scalp.clamp(0.1, 1.0);
                child.min_confidence = child.min_confidence.clamp(0.50, 0.95);
                child.explosive_leverage_multiplier =
                    child.explosive_leverage_multiplier.clamp(1.0, 10.0);

                next_gen.push(child);
            }
            population = next_gen;
        }

        let baseline_results = [current_active];
        let baseline_fitness = {
            // D-714: construcción con pila suficiente (este sitio corre en un worker de rayon).
                    let arena = GlobalArena::build_in_own_stack(initial_capital);
            baseline_results[0].apply_to_arena(&arena);
            let mut engine = GodEngineCore::new(arena.clone());
            let mut max_drawdown = 0.0;
            let mut peak_capital = initial_capital;
            for tick in &master_stream {
                arena.update_market_data(
                    tick.coin_id,
                    tick.bid_price,
                    tick.ask_price,
                    tick.bid_qty,
                    tick.ask_qty,
                    tick.timestamp,
                );
                let mut dynamic_omni = [0.0f64; 54];
                let swing_feats = engine.feature_engines[tick.coin_id].get_universal_features();
                for (i, &f) in swing_feats.iter().enumerate() {
                    dynamic_omni[i] = f as f64;
                }

                let (_new_pos, closed_pos, _) = engine.process_tick(
                    tick.coin_id,
                    tick.bid_price,
                    tick.ask_price,
                    tick.bid_qty,
                    tick.ask_qty,
                    tick.timestamp,
                    &dynamic_omni,
                );
                if closed_pos.is_some() {
                    let cap = arena.unified_capital.load(Ordering::Relaxed);
                    if cap > peak_capital {
                        peak_capital = cap;
                    }
                    let dd = if peak_capital > 0.0 {
                        (peak_capital - cap) / peak_capital
                    } else {
                        0.0
                    };
                    if dd > max_drawdown && dd.is_finite() {
                        max_drawdown = dd;
                    }
                }
            }
            let final_cap = arena.unified_capital.load(Ordering::Relaxed);
            // CERT-M5-H01: fitness UNIFICADO (baseline también)
            let raw_fitness = crate::fitness_compute(
                initial_capital,
                final_cap,
                max_drawdown,
            );
            if raw_fitness.is_finite() {
                raw_fitness
            } else {
                -999999.0
            }
        };

        println!("[Darwin] Online Evolution Complete.");
        println!("         Current Active Fitness: {:.4}", baseline_fitness);
        println!("         Evolved Genome Fitness: {:.4}", best_all_time.1);

        // FIX #412: El nuevo genoma debe ser estrictamente mejor y superar un margen del 5% sin inversión de signo
        let is_significantly_better = if baseline_fitness >= 0.0 {
            best_all_time.1 > (baseline_fitness * 1.05).max(baseline_fitness + 1e-4)
        } else {
            // Para fitness negativo (ej. -100.0), mejorar un 5% significa acercarse a cero (ej. > -95.0)
            best_all_time.1 > baseline_fitness && best_all_time.1 > (baseline_fitness * 0.95)
        };

        let allow_hotswap = std::env::var("ENABLE_ONLINE_DARWIN_MUTATION")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);
        if is_significantly_better && allow_hotswap {
            println!("[Darwin] 🧬 HOT-SWAPPING ACTIVE GENOME! Market regime shift detected.");
            best_all_time.0.apply_to_arena(&self.live_arena);

            let mut full_genotype =
                quantum_arena::genome::SuperGenotype::current_from_arena(&self.live_arena);
            full_genotype.global_leverage = best_all_time.0.global_leverage;
            full_genotype.trend_threshold = best_all_time.0.trend_threshold;
            full_genotype.maker_spread_pct = best_all_time.0.maker_spread_pct;
            full_genotype.maker_obi_threshold = best_all_time.0.maker_obi_threshold;
            full_genotype.scalp_tp_base = best_all_time.0.scalp_tp;
            full_genotype.scalp_sl_base = best_all_time.0.scalp_sl;
            full_genotype.swing_tp_base = best_all_time.0.swing_tp;
            full_genotype.swing_sl_base = best_all_time.0.swing_sl;
            full_genotype.scalp_obi_threshold = best_all_time.0.scalp_z_target;
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
            println!("[Darwin] 🛡️ Current genome is still optimal for this regime.");
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
        assert!(genome.scalp_tp > 0.0);
        assert!(genome.scalp_sl > 0.0);

        genome.apply_to_arena(&arena);

        let roundtrip = Genotype::current_from_arena(&arena);
        assert_eq!(roundtrip.global_leverage, genome.global_leverage);
        assert_eq!(roundtrip.scalp_tp, genome.scalp_tp);
    }

    #[test]
    fn test_genotype_nan_immunity_when_applying_to_arena() {
        let arena = GlobalArena::build_in_own_stack(13.0);
        let nan_genome = Genotype {
            global_leverage: f64::NAN,
            trend_threshold: f64::NAN,
            maker_spread_pct: f64::NAN,
            maker_obi_threshold: f64::NAN,
            scalp_tp: f64::NAN,
            scalp_sl: f64::NAN,
            swing_tp: f64::NAN,
            swing_sl: f64::NAN,
            scalp_z_target: f64::NAN,
            capital_split_scalp: f64::NAN,
            min_confidence: f64::NAN,
            explosive_leverage_multiplier: f64::NAN,
        };

        nan_genome.apply_to_arena(&arena);

        let safe_genome = Genotype::current_from_arena(&arena);
        assert!(safe_genome.global_leverage.is_finite());
        assert!(safe_genome.trend_threshold.is_finite());
        assert!(safe_genome.scalp_tp.is_finite());
        assert!(safe_genome.scalp_sl.is_finite());
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
}
