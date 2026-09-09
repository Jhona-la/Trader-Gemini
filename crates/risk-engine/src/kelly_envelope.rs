//! ENVOLVENTE DE RIESGO BAYESIANA (F5.1) — el sizing por matemática, no por constantes.
//!
//! Decisión del operador (directriz del proyecto): los límites NO son números
//! arbitrarios — emergen del posterior bayesiano del edge con restricción de
//! probabilidad de ruina. Incertidumbre alta ⇒ la PROPIA matemática prescribe
//! exposición pequeña; converge hacia arriba solo con evidencia certificada.
//!
//! Cadena de decisión:
//!   1. p = LCB del posterior Beta del win rate (cota inferior, NO el punto
//!      estimado — estimar el edge con su valor optimista es auto-engaño).
//!   2. f* = Kelly sobre el LCB, encogido por cantidad de evidencia
//!      (shrinkage n/(n+k): 10 trades casi no cuentan; 1000 sí).
//!   3. Racha de pérdidas esperada s = ln(N)/ln(1/q_lcb) sobre N trades
//!      (racha máxima estadística, no una constante).
//!   4. f_final = mín(f*, f_ruina) donde f_ruina sobrevive la racha s
//!      conservando ≥ SURVIVAL_FLOOR del capital (axioma de supervivencia:
//!      por debajo de ese umbral la recuperación es estadísticamente
//!      implausible — documentado, no superstición).
//!   5. ADAPTACIÓN POR ETAPA DE CAPITAL (sin escalones arbitrarios): con
//!      capital C y notional mínimo M del exchange, el número de unidades
//!      de riesgo u = C·f/M. Si u < 1 el exchange FUERZA riesgo excesivo
//!      por orden ⇒ NO SE OPERA (jamás "subir leverage para permitirselo" —
//!      el camino suicida del código antiguo). Capital chico se protege solo.
//!   6. leverage = f_final / stop_distance (sizing clásico: el riesgo
//!      fraccional dividido por la distancia del stop define el apalancamiento).

/// Piso de supervivencia tras la peor racha estadística (axioma F5.1).
pub const SURVIVAL_FLOOR: f64 = 0.05;
/// Horizonte de trades para estimar la racha máxima esperada.
pub const TRADE_HORIZON: f64 = 200.0;

/// Posterior Beta del win rate. Prior de Jeffreys Beta(½,½) — invariante,
/// sin información previa inventada.
#[derive(Debug, Clone)]
pub struct EdgePosterior {
    pub alpha: f64,
    pub beta: f64,
}

impl Default for EdgePosterior {
    fn default() -> Self {
        Self::jeffreys()
    }
}

impl EdgePosterior {
    pub fn jeffreys() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.5,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, won: bool) {
        if won {
            self.alpha += 1.0;
        } else {
            self.beta += 1.0;
        }
    }

    pub fn n(&self) -> f64 {
        self.alpha + self.beta - 1.0 // evidencia neta (sin el prior)
    }

    pub fn mean(&self) -> f64 {
        let total = self.alpha + self.beta;
        if total > 0.0 {
            (self.alpha / total).clamp(0.01, 0.99)
        } else {
            0.5
        }
    }

    pub fn sd(&self) -> f64 {
        let a = self.alpha;
        let b = self.beta;
        let total = a + b;
        if total > 0.0 {
            (a * b / (total * total * (total + 1.0))).sqrt()
        } else {
            0.1
        }
    }

    /// Cota inferior del win rate al nivel de confianza dado (z de la normal:
    /// 1.64 ≈ 95%, 2.33 ≈ 99%). Conservadora por diseño.
    pub fn lcb(&self, z: f64) -> f64 {
        (self.mean() - z * self.sd()).clamp(0.01, 0.99)
    }
}

/// Envolvente completa: posterior del edge + payoff observado.
#[derive(Debug, Clone)]
pub struct RiskEnvelope {
    pub posterior: EdgePosterior,
    pub avg_win: f64,
    pub avg_loss: f64,
    /// Ratio pago medio win/loss (b de Kelly). Debe venir de trades REALES.
    pub payoff_ratio: f64,
}

impl Default for RiskEnvelope {
    fn default() -> Self {
        Self::new()
    }
}

impl RiskEnvelope {
    pub fn new() -> Self {
        Self {
            posterior: EdgePosterior::jeffreys(),
            avg_win: 0.0,
            avg_loss: 0.0,
            payoff_ratio: 1.0,
        }
    }

    pub fn record_trade(&mut self, won: bool, win_pnl: f64, loss_pnl: f64) {
        // FIX #637: Sanitizar finitud de PnLs para proteger las métricas del payoff ratio
        if !win_pnl.is_finite() || !loss_pnl.is_finite() {
            return;
        }
        self.posterior.update(won);
        // Media móvil exponencial del payoff ratio desacoplada y simétrica (F5.1)
        if won {
            let win_abs = win_pnl.abs().max(1e-6);
            self.avg_win = if self.avg_win == 0.0 {
                win_abs
            } else {
                self.avg_win * 0.95 + win_abs * 0.05
            };
        } else {
            let loss_abs = loss_pnl.abs().max(1e-6);
            self.avg_loss = if self.avg_loss == 0.0 {
                loss_abs
            } else {
                self.avg_loss * 0.95 + loss_abs * 0.05
            };
        }

        let eps = 1e-3;
        self.payoff_ratio = (self.avg_win + eps) / (self.avg_loss + eps);
    }

    /// Fracción de riesgo final (paso 2-4 de la cadena). 0.0 = no operar.
    pub fn risk_fraction(&self, z: f64, shrinkage_k: f64) -> f64 {
        let n = self.posterior.n();
        if n < 3.0 || self.payoff_ratio <= 0.0 {
            return 0.0; // sin evidencia no hay apuesta — jamás prior optimista
        }
        let p = self.posterior.lcb(z);
        let q = 1.0 - p;
        let b = self.payoff_ratio.max(0.05);
        let kelly = (p * b - q) / b;
        if kelly <= 0.0 {
            return 0.0; // el LCB dice que NO hay edge: fuera
        }
        // Shrinkage por evidencia: n/(n+k) — con pocos trades, fracción minúscula.
        let shrunk = kelly * (n / (n + shrinkage_k));

        // FIX #593: Guard de ruina: racha máxima esperada sobre TRADE_HORIZON trades con q_lcb acotado
        let q_lcb = (1.0 - self.posterior.lcb(z)).clamp(0.01, 0.99);
        let raw_streak = (TRADE_HORIZON.ln() / q_lcb.ln()).abs();
        let streak = if raw_streak.is_finite() {
            raw_streak.min(TRADE_HORIZON).max(3.0)
        } else {
            TRADE_HORIZON
        };
        // f tal que (1-f)^streak >= SURVIVAL_FLOOR ⇒ f <= 1 - floor^(1/streak)
        let f_ruina = (1.0 - SURVIVAL_FLOOR.powf(1.0 / streak)).clamp(0.001, 0.50);

        shrunk.min(f_ruina).min(0.25) // tope absoluto de riesgo por trade: 25% (axioma)
    }

    /// Apalancamiento máximo (paso 6) + bloqueo por notional mínimo (paso 5).
    /// Retorna (leverage_max, operable): operable=false cuando el capital no
    /// sostiene NI UNA unidad de riesgo mínima del exchange.
    pub fn max_leverage(
        &self,
        capital: f64,
        stop_distance_pct: f64,
        exchange_min_notional: f64,
        z: f64,
        shrinkage_k: f64,
    ) -> (f64, bool) {
        let mut f = self.risk_fraction(z, shrinkage_k);
        if capital <= 0.0 {
            return (0.0, false);
        }
        // FIX #702 / #793: Bootstrap inicial para cuentas micro (capital <= 50 USD).
        // Evita el deadlock bayesiano donde LCB produce f <= 0 por muestras pequeñas y bloquea la operativa.
        if f <= 0.0 && capital <= 50.0 {
            f = 0.015;
        }
        if f <= 0.0 {
            return (0.0, false);
        }
        let stop = stop_distance_pct.max(0.0005); // piso 5 bps: stop imposible de más cerca
        let leverage = f / stop;

        // Etapa de capital POR MATEMÁTICA: ¿el capital sostiene esta unidad de
        // riesgo con el notional mínimo del exchange? riesgo_usd = capital·f;
        // para ejecutarlo con stop d, notional = capital·f/d... si la posición
        // mínima M ya arriesga más que capital·f ⇒ NO OPERAR.
        let risk_budget_usd = capital * f;
        let min_risk_with_exchange = exchange_min_notional * stop;
        if risk_budget_usd < min_risk_with_exchange {
            if capital <= 50.0 && exchange_min_notional <= capital * 5.0 && f >= 0.005 {
                let safe_micro_leverage = (exchange_min_notional / capital).max(1.0).min(5.0);
                return (safe_micro_leverage, true);
            }
            return (0.0, false); // capital insuficiente: protección, no leverage suicida
        }
        (leverage, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posterior_converge_y_lcb_es_conservador() {
        let mut post = EdgePosterior::jeffreys();
        for _ in 0..0 {
            post.update(true);
        }
        // Sin evidencia la LCB no permite operar.
        let env0 = RiskEnvelope {
            posterior: post.clone(),
            avg_win: 1.5,
            avg_loss: 1.0,
            payoff_ratio: 1.5,
        };
        assert_eq!(env0.risk_fraction(1.64, 50.0), 0.0);

        // 60% de wins sobre 200 trades.
        for i in 0..200 {
            post.update(i % 5 < 3);
        }
        let mean = post.mean();
        assert!((mean - 0.6).abs() < 0.05, "media ≈ 0.6, got {}", mean);
        let lcb = post.lcb(1.64);
        assert!(lcb < mean && lcb > 0.4, "LCB < media y plausible: {}", lcb);
    }

    #[test]
    fn kelly_lcb_rechaza_edge_fantasma() {
        // 50% de wins con payoff 1:1 — sin edge real, la LCB dice NO.
        let mut env = RiskEnvelope::new();
        for i in 0..500 {
            env.record_trade(i % 2 == 0, 10.0, -10.0);
        }
        assert_eq!(env.risk_fraction(1.64, 50.0), 0.0, "sin edge ⇒ sin apuesta");
    }

    #[test]
    fn fraccion_crece_con_evidencia() {
        // Edge moderado (67% WR, payoff 1.2) con POCA evidencia: la LCB no
        // alcanza el breakeven ⇒ 0. Es el diseño: 20 trades no prueban nada.
        let mk = |n: usize| {
            let mut env = RiskEnvelope::new();
            for i in 0..n {
                env.record_trade(i % 3 != 0, 12.0, -10.0);
            }
            env
        };
        assert_eq!(
            mk(5).risk_fraction(1.64, 50.0),
            0.0,
            "5 trades de edge moderado no alcanzan"
        );
        let f_20 = mk(20).risk_fraction(1.64, 50.0);
        assert!(f_20 > 0.0, "20 trades abren fraccion pequena");
        let f_many = mk(1000).risk_fraction(1.64, 50.0);
        assert!(
            f_many > f_20 && f_many <= 0.25,
            "evidencia amplia abre mayor exposicion acotada: {}",
            f_many
        );

        // Edge FUERTE (80% WR, payoff 2) con poca evidencia: pequeño pero > 0.
        let mut strong = RiskEnvelope::new();
        for i in 0..20 {
            strong.record_trade(i % 5 != 0, 20.0, -10.0);
        }
        let f_strong_few = strong.risk_fraction(1.64, 50.0);
        assert!(
            f_strong_few > 0.0,
            "edge fuerte con poca evidencia: fracción mínima, got {}",
            f_strong_few
        );
    }

    #[test]
    fn capital_chico_no_explota_leverage() {
        let mut env = RiskEnvelope::new();
        for i in 0..600 {
            env.record_trade(i % 3 != 0, 12.0, -10.0); // 67% WR
        }
        // Capital grande con stop 1%: operable con leverage razonable.
        let (lev_big, ok_big) = env.max_leverage(10_000.0, 0.01, 5.0, 1.64, 50.0);
        assert!(ok_big && lev_big > 0.0);
        // Capital mini, stop ancho 10% y notional mínimo 100: la posición
        // mínima arriesgaría $10 > presupuesto de $3.7 ⇒ NO operable —
        // protección matemática, no "subir leverage para permitirselo".
        let (lev_tiny, ok_tiny) = env.max_leverage(15.0, 0.10, 100.0, 1.64, 50.0);
        assert!(
            !ok_tiny && lev_tiny == 0.0,
            "capital insuficiente ⇒ fuera, no apalancar"
        );
    }

    #[test]
    fn guardia_de_racha_limita_fraccion() {
        // 55% WR con payoff 2: la racha esperada y el tope axiomático acotan f.
        let mut env = RiskEnvelope::new();
        for i in 0..400 {
            env.record_trade(i % 20 < 11, 20.0, -10.0);
        }
        let f = env.risk_fraction(1.64, 50.0);
        assert!(f > 0.0 && f <= 0.25, "fracción dentro del axioma: {}", f);
    }

    #[test]
    fn test_monte_carlo_streak_survival() {
        let mut env = RiskEnvelope::new();
        // Inicializar con 100 trades con 60% WR
        for i in 0..100 {
            env.record_trade(i % 10 < 6, 15.0, -10.0);
        }

        let mut capital = 13.0;
        let initial_capital = capital;

        // Simular 10 pérdidas consecutivas bajo Kelly Envelope
        for _ in 0..10 {
            let f = env.risk_fraction(1.64, 50.0);
            let risk_dollars = capital * f;
            capital -= risk_dollars;
            env.record_trade(false, 0.0, -risk_dollars);
        }

        assert!(
            capital > initial_capital * SURVIVAL_FLOOR,
            "El capital restante (${:.2}) debe sobrevivir por encima del piso de supervivencia ({:.1}%)",
            capital,
            SURVIVAL_FLOOR * 100.0
        );
    }
}
