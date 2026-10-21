use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;

/// FASE 8: Portfolio Orchestrator (Capa 3)
/// Responsable de analizar la exposición cruzada (correlación direccional) en todo el portafolio
/// y bloquear posiciones (Flash Crash Protection) si el riesgo se vuelve asimétrico o sistémico.
pub struct PortfolioOrchestrator<'a> {
    arena: &'a GlobalArena,
}

impl<'a> PortfolioOrchestrator<'a> {
    pub fn new(arena: &'a GlobalArena) -> Self {
        Self { arena }
    }

    /// Calcula la asignación dinámica de capital (Fase 8: Redistribución basada en rendimiento)
    #[inline(always)]
    pub fn calculate_dynamic_allocation(&self, coin_id: usize, base_leverage: f64) -> f64 {
        if coin_id >= self.arena.coins.len() || !base_leverage.is_finite() || base_leverage <= 0.0 {
            return 1.0;
        }
        let coin = &self.arena.coins[coin_id];

        let win_rate = coin.metrics.win_rate.load(Ordering::Relaxed);
        let profit_factor = coin.metrics.profit_factor.load(Ordering::Relaxed);

        // CONTINUOUS Performance Multiplier (sigmoid-based, no step functions)
        // Maps WR×PF product into a smooth [0.3, 2.0] range via generalized logistic
        // Center at WR=0.50, PF=1.0 (breakeven point)
        let safe_wr = if win_rate.is_finite() && win_rate >= 0.0 {
            win_rate.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let safe_pf = if profit_factor.is_finite() && profit_factor > 0.0 {
            profit_factor.max(0.1)
        } else {
            1.0
        };
        let performance_score = safe_wr * safe_pf;
        let portfolio_perf_mult_steepness = self
            .arena
            .config
            .portfolio_perf_mult_steepness
            .load(Ordering::Relaxed);
        let portfolio_perf_mult_min = self
            .arena
            .config
            .portfolio_perf_mult_min
            .load(Ordering::Relaxed);
        let portfolio_perf_mult_max = self
            .arena
            .config
            .portfolio_perf_mult_max
            .load(Ordering::Relaxed);
        let portfolio_perf_mult_center = self
            .arena
            .config
            .portfolio_perf_mult_center
            .load(Ordering::Relaxed);

        let range = portfolio_perf_mult_max - portfolio_perf_mult_min;
        let performance_multiplier = portfolio_perf_mult_min
            + range
                / (1.0
                    + (-portfolio_perf_mult_steepness
                        * (performance_score - portfolio_perf_mult_center))
                        .exp());

        // CONTINUOUS Drawdown Penalty (exponential decay, no step functions)
        let mut global_unrealized: f64 = 0.0;
        for c in self.arena.coins.iter() {
            global_unrealized += c.metrics.pnl_unrealized.load(Ordering::Relaxed);
        }

        let capital = self.arena.unified_capital.load(Ordering::Relaxed);
        let portfolio_dd_penalty_decay = self
            .arena
            .config
            .portfolio_dd_penalty_decay
            .load(Ordering::Relaxed);
        let drawdown_penalty = if capital > 0.0 && global_unrealized < 0.0 {
            let dd_pct = (global_unrealized.abs() / capital).clamp(0.0, 1.0);
            // Smooth exponential decay: at 0% DD = 1.0, at 5% DD ≈ 0.47, at 10% DD ≈ 0.22
            (-dd_pct * portfolio_dd_penalty_decay).exp()
        } else {
            1.0
        };

        let raw_alloc = base_leverage * performance_multiplier * drawdown_penalty;
        if raw_alloc.is_finite() && raw_alloc > 0.0 {
            raw_alloc
        } else {
            1.0
        }
    }

    /// F1-083: Límite de exposición continuo C1 sobre el régimen adverso (D-641 / P2).
    ///
    /// Para p_sys in [0.0, 0.80]:
    ///   E(P) = cushion - 0.25 * max(p_coin, p_sys) (contracción lineal canónica,
    ///   preserva exactamente las cotas contractuales auditadas: coin_flux=1.0 -> C-0.25,
    ///   p_sys=0.40 -> C-0.10, p_sys=0.60 -> C-0.15, p_sys=0.80 -> C-0.20).
    ///
    /// Para p_sys in (0.80, 0.90):
    ///   Hermite cúbico C1 suave en p_sys: conecta el valor y_0 = cushion - 0.25 * max(p_coin, 0.80)
    ///   con derivada dE/d(p_sys) = -0.25 hasta aterrizar suavemente en (0.90, 0.0) con derivada 0.0.
    ///   Erradica por completo el acantilado discontinuo de ~76% de capital (F1-083) y
    ///   previene órdenes suicidas de ~$9.85 USD sobre cuentas micro de $13 USD a p=0.899.
    ///
    /// Para p_sys >= 0.90:
    ///   E(P) = 0.0 (extinción natural y suave del margen direccional ante riesgo macro extremo).
    #[inline(always)]
    pub fn continuous_exposure_limit(cushion: f64, p_coin: f64, p_sys: f64) -> f64 {
        if !cushion.is_finite() || cushion <= 0.0 {
            return 0.0;
        }
        let coin = if p_coin.is_finite() {
            p_coin.clamp(0.0, 1.0)
        } else {
            return 0.0;
        };
        let sys = if p_sys.is_finite() {
            p_sys.max(0.0)
        } else {
            return 0.0;
        };

        if sys <= 0.80 {
            let adverse = coin.max(sys);
            (cushion - 0.25 * adverse).max(0.0)
        } else if sys < 0.90 {
            let u = (sys - 0.80) / 0.10;
            let y0 = (cushion - 0.25 * coin.max(0.80)).max(0.0);
            let s0 = -0.025; // dE/du = (dE/dp_sys) * 0.10 = -0.25 * 0.10
            let u2 = u * u;
            let u3 = u2 * u;
            let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
            let h10 = u3 - 2.0 * u2 + u;
            (h00 * y0 + h10 * s0).max(0.0)
        } else {
            0.0
        }
    }

    /// Calcula el margen máximo admisible que la cartera puede alojar para una nueva posición,
    /// respetando tanto el límite de exposición bruta como el direccional y las presiones sistémicas.
    #[inline]
    pub fn available_margin_headroom(
        &self,
        intent_is_long: bool,
        min_notional_simbolo: f64,
    ) -> f64 {
        let capital = self.arena.unified_capital.load(Ordering::Relaxed);
        if !capital.is_finite() || capital <= 0.0 {
            return 0.0;
        }
        let p_crash = self.arena.regime_p_crash.load(Ordering::Relaxed);
        let p_bull = self.arena.regime_p_bull.load(Ordering::Relaxed);
        let p_chaos = self.arena.regime_p_chaos.load(Ordering::Relaxed);
        if !p_crash.is_finite() || !p_bull.is_finite() || !p_chaos.is_finite() {
            return 0.0;
        }
        let mut directional_max = 0.0f64;
        for coin in self.arena.coins.iter() {
            let tide = coin.spectral_coherence.load(Ordering::Relaxed);
            let flux = coin.spectral_crash_flux.load(Ordering::Relaxed);
            if !tide.is_finite() || !flux.is_finite() {
                return 0.0;
            }
            if (intent_is_long && tide < 0.0) || (!intent_is_long && tide > 0.0) {
                directional_max = directional_max.max(flux);
            }
        }
        let directional_adverse = if intent_is_long { p_crash } else { p_bull };
        let systemic_pressure = directional_adverse.max(0.50 * p_chaos.clamp(0.0, 1.0));

        let escasez = crate::capital_regime::micro_weight(
            capital,
            crate::capital_regime::effective_min_notional(min_notional_simbolo),
        );
        let cushion = crate::capital_regime::margin_cushion(
            self.arena.config.margin_cushion_pct.load(Ordering::Relaxed),
            escasez,
        );
        // F1-083: manifold continuo C1 sin acantilado ni piso muerto .max(0.05)
        let exposure_limit = Self::continuous_exposure_limit(cushion, directional_max, systemic_pressure);

        let max_capital = capital * exposure_limit;

        let mut total_long_margin = 0.0;
        let mut total_short_margin = 0.0;
        for coin in self.arena.coins.iter() {
            for pos in coin.positions.slots() {
                if pos.is_open() {
                    let margin = pos.margin_used.load(Ordering::Relaxed);
                    if !margin.is_finite() || margin < 0.0 {
                        return 0.0;
                    }
                    if pos.is_long.load(Ordering::Relaxed) {
                        total_long_margin += margin;
                    } else {
                        total_short_margin += margin;
                    }
                }
            }
        }

        let gross_headroom = (max_capital - (total_long_margin + total_short_margin)).max(0.0);
        let dir_headroom = if intent_is_long {
            (max_capital - total_long_margin).max(0.0)
        } else {
            (max_capital - total_short_margin).max(0.0)
        };
        gross_headroom.min(dir_headroom)
    }

    /// Evalúa si el portafolio permite la apertura de una nueva posición direccional
    #[inline(always)]
    pub fn allow_trade(
        &self,
        intent_is_long: bool,
        required_margin: f64,
        regime: crate::regime::MarketRegime,
        // D-750 — NOCIONAL MÍNIMO DEL SÍMBOLO QUE SE PRETENDE ABRIR.
        //
        // La escasez de capital que estrecha el colchón de margen se medía
        // contra `arena.config.min_notional`, literal 5,0 que nadie escribe
        // nunca. El mínimo real lo publica el exchange por símbolo y el
        // evaluador ya lo tiene resuelto cuando llega aquí: se lo pasa, en vez
        // de que esta capa vuelva a leer el número congelado.
        min_notional_simbolo: f64,
    ) -> bool {
        if !required_margin.is_finite() || required_margin <= 0.0 {
            return false;
        }
        // Unknown financial state is not zero exposure. Preserve the existing
        // collateral policy only inside its declared finite domain.
        let capital = self.arena.unified_capital.load(Ordering::Relaxed);
        let drawdown_budget = self
            .arena
            .config
            .global_max_drawdown
            .load(Ordering::Relaxed);
        if !capital.is_finite()
            || capital <= 0.0
            || !drawdown_budget.is_finite()
            || !(0.0..=1.0).contains(&drawdown_budget)
        {
            return false;
        }
        // Regime Orchestration (Fase 13: Kill-Switch macro)
        let p_crash = self.arena.regime_p_crash.load(Ordering::Relaxed);
        let p_bull = self.arena.regime_p_bull.load(Ordering::Relaxed);
        let p_chaos = self.arena.regime_p_chaos.load(Ordering::Relaxed);
        if !p_crash.is_finite() || !p_bull.is_finite() || !p_chaos.is_finite() {
            return false;
        }

        let systemic_crash_veto = p_crash >= 0.90;
        // F1-119: En producción viva, el símplex continuo (p_crash >= 0.90) gobierna el veto duro.
        // fallback_discrete_veto actúa como red de seguridad para fixtures de prueba y arranque en frío
        // donde p_crash aún no ha sido inicializado por el pipeline continuo (p_crash <= 0.0).
        let fallback_discrete_veto = p_crash <= 0.0
            && regime == crate::regime::MarketRegime::Crash;
        if (systemic_crash_veto || fallback_discrete_veto) && intent_is_long {
            return false; // Bloqueo absoluto de compras en caída libre sistémica.
        }

        let systemic_squeeze_veto = p_bull >= 0.90;
        if systemic_squeeze_veto && !intent_is_long {
            return false; // Bloqueo absoluto de ventas en euforia sistémica.
        }

        let headroom = self.available_margin_headroom(intent_is_long, min_notional_simbolo);
        if !headroom.is_finite() || headroom <= 0.0 {
            return false;
        }
        required_margin <= headroom + 1e-9
    }
}
