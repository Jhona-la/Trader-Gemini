use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// ♟️ ALGORITMO #74: MOTOR DE TEORÍA DE JUEGOS Y EQUILIBRIO DE NASH (GAME THEORETIC NASH ENGINE)
/// Modela la interacción competitiva entre participantes agresivos y Market Makers mediante juegos de Stackelberg,
/// calculando el precio óptimo del equilibrio de Nash para evitar trampas de liquidez y cazas de Stop Loss.
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct GameTheoreticNashEngine {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for GameTheoreticNashEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GameTheoreticNashEngine").finish()
    }
}

impl GameTheoreticNashEngine {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// Calcula la posición óptima del equilibrio de Nash para colocar órdenes pasivas
    #[inline(always)]
    pub fn compute_nash_equilibrium_price(
        best_bid: f64,
        best_ask: f64,
        liquidity_imbalance: f64,
    ) -> f64 {
        if best_bid <= 0.0 || best_ask <= best_bid || !best_bid.is_finite() || !best_ask.is_finite()
        {
            return if best_bid.is_finite() && best_bid > 0.0 {
                best_bid
            } else {
                0.0
            };
        }
        let mid = (best_bid + best_ask) * 0.5;
        let spread = (best_ask - best_bid).max(0.0);
        let safe_imb = if liquidity_imbalance.is_finite() {
            liquidity_imbalance.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let nash_offset = spread * 0.25 * safe_imb;
        (mid + nash_offset).clamp(best_bid, best_ask)
    }

    /// Calcula la estrategia óptima de Minimax bajo presencia de creadores de mercado adversarios
    #[inline(always)]
    pub fn compute_minimax_strategy(
        long_payoff: f64,
        short_payoff: f64,
        adversarial_pressure: f64,
    ) -> f64 {
        // FIX #647: Sanitizar parámetros de teoría de juegos
        let safe_long = if long_payoff.is_finite() {
            long_payoff
        } else {
            0.0
        };
        let safe_short = if short_payoff.is_finite() {
            short_payoff
        } else {
            0.0
        };
        let safe_adv = if adversarial_pressure.is_finite() {
            adversarial_pressure.clamp(0.0, 0.9)
        } else {
            0.1
        };

        let net_payoff = safe_long - safe_short;
        let defense_factor = 1.0 - safe_adv;
        let res = (net_payoff * defense_factor).tanh();
        if res.is_finite() {
            res.clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// #618 (Ola 40) — VOTO ESPECTRAL del equilibrio de Nash: el juego se
    /// juega a CADA escala. El desplazamiento x(τ_k) ES el payoff direccional
    /// a esa escala (long_payoff = x si x > 0, short_payoff = |x| si x < 0).
    /// La PRESIÓN ADVERSARIAL es espectral: las escalas LENTAS (k alto) son
    /// el adversario que castiga (la inercia macro controla más del juego
    /// cuanto más se extiende en τ) — presión = (k/31)·máx. El minimax
    /// defiende: a mayor escala (adversario más fuerte), el voto se contrae.
    /// Las escalas rápidas juegan el juego casi puro (defensa ≈ 1); las
    /// lentas juegan contra la inercia estructural (defensa → 1−máx).
    /// Observacional: el voto vivo queda bit a bit (T-1 cero).
    pub fn voto_espectral(
        desplazamientos: &[f64; 32],
        presion_adversarial_max: f64,
    ) -> crate::voto_espectral::VotoEspectral {
        let max_adv = if presion_adversarial_max.is_finite() {
            presion_adversarial_max.clamp(0.0, 0.9)
        } else {
            0.1
        };
        let mut por_escala = [0.0f64; 32];
        for k in 0..32 {
            let x = desplazamientos[k];
            if !x.is_finite() || x == 0.0 {
                continue;
            }
            // Payoff direccional: x > 0 ⇒ long domina; x < 0 ⇒ short domina.
            let (long_payoff, short_payoff) = if x > 0.0 {
                (x, 0.0)
            } else {
                (0.0, x.abs())
            };
            // Presión adversarial espectral: crece linealmente con k/31.
            // La escala 0 juega casi sin adversario; la 31 juega contra
            // la inercia estructural completa.
            let adversarial = max_adv * (k as f64 / 31.0);
            por_escala[k] = Self::compute_minimax_strategy(long_payoff, short_payoff, adversarial);
        }
        crate::voto_espectral::VotoEspectral::desde_arr(&por_escala)
    }
}

impl QuantumStrategy for GameTheoreticNashEngine {
    fn name(&self) -> &str {
        "GameTheoreticNashEngine"
    }

    fn init(&mut self, registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        self.registry = Some(registry);
        Ok(())
    }

    fn evaluate(&self) -> f64 {
        self.evaluate_for_coin(0, "")
    }

    fn evaluate_for_coin(&self, coin_id: usize, symbol: &str) -> f64 {
        let sym_opt = if symbol.is_empty() {
            None
        } else {
            Some(symbol)
        };
        let cid_opt = if symbol.is_empty() {
            None
        } else {
            Some(coin_id)
        };
        let r = match self.registry.as_ref() {
            Some(reg) => reg,
            None => return 0.0,
        };

        let ofi = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "order_book_imbalance",
                "GameTheoreticNashEngine",
            )
            .or_else(|| {
                r.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "order_flow_imbalance",
                    "GameTheoreticNashEngine",
                )
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0);

        let long_payoff = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "game_theory_long_payoff",
                "GameTheoreticNashEngine",
            )
            .map(|p| p.get_value())
            .unwrap_or_else(|| ofi.max(0.0));
        let short_payoff = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "game_theory_short_payoff",
                "GameTheoreticNashEngine",
            )
            .map(|p| p.get_value())
            .unwrap_or_else(|| (-ofi).max(0.0));
        let adversarial = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "game_theory_adversarial_pressure",
                "GameTheoreticNashEngine",
            )
            .or_else(|| {
                r.get_scoped_parameter(sym_opt, cid_opt, "cvpin", "GameTheoreticNashEngine")
            })
            .map(|p| p.get_value())
            .unwrap_or(0.1);

        if !long_payoff.is_finite() || !short_payoff.is_finite() || !adversarial.is_finite() {
            return 0.0;
        }

        Self::compute_minimax_strategy(long_payoff, short_payoff, adversarial)
    }

    fn horizon(&self) -> strategy_core::TradeHorizon {
        strategy_core::TradeHorizon::Continuous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nash_equilibrium_and_minimax() {
        let price = GameTheoreticNashEngine::compute_nash_equilibrium_price(60000.0, 60010.0, 0.5);
        assert!(price >= 60000.0 && price <= 60010.0);

        let signal = GameTheoreticNashEngine::compute_minimax_strategy(1.0, -0.2, 0.1);
        assert!(signal > 0.0);
    }

    #[test]
    fn test_game_theoretic_nash_nan_and_inverted_spread_immunity() {
        // Inverted or invalid spread
        let price_inv =
            GameTheoreticNashEngine::compute_nash_equilibrium_price(60010.0, 60000.0, 0.5);
        assert_eq!(price_inv, 60010.0);

        // NaN inputs
        let price_nan =
            GameTheoreticNashEngine::compute_nash_equilibrium_price(f64::NAN, 60010.0, 0.5);
        assert!(price_nan.is_nan() || price_nan == 60010.0 || price_nan <= 0.0);

        let signal_nan =
            GameTheoreticNashEngine::compute_minimax_strategy(f64::NAN, f64::INFINITY, -100.0);
        assert!(signal_nan.is_finite());
        assert!((-1.0..=1.0).contains(&signal_nan));
    }

    #[test]
    fn test_game_theoretic_nash_evaluate_with_registry() {
        let mut engine = GameTheoreticNashEngine::new();
        assert_eq!(engine.evaluate(), 0.0); // uninitialized

        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("order_book_imbalance", 0.75);
        registry.set("cvpin", 0.05);

        engine.init(registry).expect("init should succeed");
        let signal = engine.evaluate();
        assert!(signal > 0.0);
        assert!((-1.0..=1.0).contains(&signal));
    }
}

#[cfg(test)]
mod qo_618_tests {
    use super::*;
    use crate::voto_espectral::ESCALAS_VOTO;

    #[test]
    fn qo_618_nash_espectral_adversario_macro_contrae() {
        let mut x = [0.0; ESCALAS_VOTO];
        for (k, v) in x.iter_mut().enumerate() {
            *v = 0.3 * (k as f64 - 15.5).signum();
        }
        let voto = GameTheoreticNashEngine::voto_espectral(&x, 0.5);
        // Antisimetría: el juego es simétrico en payoff.
        let mut x_neg = x;
        for v in &mut x_neg {
            *v = -*v;
        }
        let voto_neg = GameTheoreticNashEngine::voto_espectral(&x_neg, 0.5);
        for k in 0..ESCALAS_VOTO {
            if x[k] != 0.0 {
                assert!(
                    (voto.en_escala(k) + voto_neg.en_escala(k)).abs() < 1e-10,
                    "antisimetría en {}: {} vs {}",
                    k,
                    voto.en_escala(k),
                    voto_neg.en_escala(k)
                );
            }
        }
        // El ADVERSARIO espectral (k alto ⇒ más presión) CONTRAE el voto:
        // a igual payoff, la escala lenta vota MENOS que la rápida.
        let idx_rapido = 2;
        let idx_lento = 28;
        assert!(
            voto.en_escala(idx_rapido).abs() > voto.en_escala(idx_lento).abs(),
            "adversario macro contrae: rápido {} vs lento {}",
            voto.en_escala(idx_rapido).abs(),
            voto.en_escala(idx_lento).abs()
        );
        // Sin adversario: ambas escalas votan igual (juego puro).
        let voto_libre = GameTheoreticNashEngine::voto_espectral(&x, 0.0);
        let dif = (voto_libre.en_escala(idx_rapido).abs() - voto_libre.en_escala(idx_lento).abs()).abs();
        assert!(dif < 1e-10, "sin adversario, juego puro en todas las escalas");
        // Payoff nulo ⇒ voto 0 (sin juego no hay equilibrio).
        let cero = GameTheoreticNashEngine::voto_espectral(&[0.0; ESCALAS_VOTO], 0.5);
        assert_eq!(cero.dominante(), None);
    }
}
