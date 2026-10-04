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
        // (Ola XLII·B1-wire) CRASH POLICY CONTINUA: la crash-ness MEDIDA del
        // campo espectral (publicada por símbolo) contrae el margen admisible
        // continuamente — el largo pierde hasta 25 PUNTOS PORCENTUALES de
        // colchón (0,25 de la fracción de capital) a crash-ness plena. El veto
        // binario del enum queda como el extremo declarado (caída libre
        // sistémica); el rango medio es gradiente, no caja.
        //
        // Ola XLIV: `crash_flux` usa |marea| («describe el evento, no el
        // lado»), así que una tendencia ALCISTA intensa también lo eleva.
        // Tomar el máximo de todas las monedas sin mirar el signo recortaba
        // el margen de TODOS los largos cuando cualquier moneda subía con
        // fuerza. Sólo cuenta la crash-ness de las monedas cuya marea
        // portadora (`spectral_coherence`) es BAJISTA — la misma regla que
        // Ola XLIV & AGY-AUD-P07: SIMETRÍA DIRECCIONAL EN PRESIÓN ESPECTRAL DE COLCHÓN.
        // Para largos: contrae por crash_pressure (marea portadora bajista + crash flux).
        // Para cortos: contrae simétricamente por squeeze_pressure (marea portadora alcista + crash flux),
        // protegiendo posiciones cortas contra short squeezes violentos y blow-off tops.
        // AGY-AUD-P31: Conexión del símplex continuo de régimen de mercado al colchón direccional de riesgo.
        // A la presión de crash_max por moneda se añade la probabilidad sistémica p_crash del mercado,
        // contrayendo el margen admisible continuamente sin escalones discretos.
        // Para cortos: contrae simétricamente por squeeze_pressure y p_bull, protegiendo
        // posiciones cortas contra short squeezes violentos y blow-off tops.
        // P2: unknown pressure is not calm. Validate both systemic inputs and
        // every spectral pair before filtering by direction; finite cold zeros
        // remain admissible. Reuse the values checked here for this decision.
        let p_crash = self.arena.regime_p_crash.load(Ordering::Relaxed);
        let p_bull = self.arena.regime_p_bull.load(Ordering::Relaxed);
        if !p_crash.is_finite() || !p_bull.is_finite() {
            return false;
        }
        let mut directional_max = 0.0f64;
        for coin in self.arena.coins.iter() {
            let tide = coin.spectral_coherence.load(Ordering::Relaxed);
            let flux = coin.spectral_crash_flux.load(Ordering::Relaxed);
            if !tide.is_finite() || !flux.is_finite() {
                return false;
            }
            if (intent_is_long && tide < 0.0) || (!intent_is_long && tide > 0.0) {
                directional_max = directional_max.max(flux);
            }
        }
        let systemic_pressure = if intent_is_long { p_crash } else { p_bull };
        let directional_pressure = 0.25
            * directional_max
                .clamp(0.0, 1.0)
                .max(systemic_pressure.clamp(0.0, 1.0));
        let systemic_crash_veto = p_crash >= 0.90;
        if (regime == crate::regime::MarketRegime::Crash || systemic_crash_veto) && intent_is_long {
            return false; // Bloqueo absoluto de compras en caída libre sistémica.
        }
        // D-403: Permitir operaciones Short durante BullRun (scalping contratendencia con stops ceñidos)
        // en cumplimiento del mandato supremo: operar Long y Short simétricamente.

        let mut total_long_margin = 0.0;
        let mut total_short_margin = 0.0;

        // Sum collateral across spectral slots; this is NOT market delta.
        // Individual atomic loads do not provide a coherent portfolio snapshot.
        for coin in self.arena.coins.iter() {
            for pos in coin.positions.slots() {
                if pos.is_open() {
                    let margin = pos.margin_used.load(Ordering::Relaxed);
                    if !margin.is_finite() || margin < 0.0 {
                        return false;
                    }
                    if pos.is_long.load(Ordering::Relaxed) {
                        total_long_margin += margin;
                    } else {
                        total_short_margin += margin;
                    }
                }
            }
        }

        // (Ola XLI·D6) Una sola definición: la segunda era una sombra literal
        // de la primera, y el límite TOTAL contra el mismo exposure_limit era
        // redundante con el direccional (siempre dispara antes o igual).
        let total_exposure = total_long_margin + total_short_margin + required_margin;
        if !total_exposure.is_finite() {
            return false;
        }

        // D-744: el tope de margen ya NO se deriva del gen de drawdown. Eran
        // dos conceptos distintos leyendo el mismo número: con el gen base
        // (0,95) el `min(0,20)` lo aplastaba a 0,20 y el colchón quedaba
        // clavado en 0,80 pasara lo que pasara — evolucionar el drawdown
        // movía, de paso y sin decirlo, cuánto capital podía comprometerse.
        // El colchón tiene su propio gen (`margin_cushion_pct`) y su propia
        // fuente única, la misma que usa el núcleo al comprobar margen libre.
        // D-750: escasez medida contra el mínimo DEL SÍMBOLO.
        let escasez = crate::capital_regime::micro_weight(
            capital,
            crate::capital_regime::effective_min_notional(min_notional_simbolo),
        );
        let exposure_limit = (crate::capital_regime::margin_cushion(
            self.arena.config.margin_cushion_pct.load(Ordering::Relaxed),
            escasez,
        ) - directional_pressure)
            .max(0.05);

        // GROSS exposure cap: margen comprometido en AMBAS direcciones a la
        // vez — el límite direccional no lo captura (40 long + 50 short caben
        // por lado y suman 90 sobre un techo de 90). (Ola XLI·D6 corrección:
        // el total NO era redundante; sólo lo era la doble computación.)
        if total_exposure > capital * exposure_limit {
            return false;
        }

        // Directional collateral limit; leverage/notional are not modeled here.
        if intent_is_long {
            if (total_long_margin + required_margin) > capital * exposure_limit {
                return false;
            }
        } else {
            if (total_short_margin + required_margin) > capital * exposure_limit {
                return false;
            }
        }

        true
    }
}
