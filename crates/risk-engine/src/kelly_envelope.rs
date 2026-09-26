//! ENVOLVENTE DE RIESGO (F5.1 / FMT-098/099/113).
//!
//! La beta modela resultados Bernoulli bajo supuestos de la población elegida.
//! El LCB normal aproximado, el shrinkage n/(n+k), el guard de racha y el
//! máximo del 25% son componentes diferentes: no certifican conjuntamente
//! una probabilidad de ruina ni rentabilidad. z y k son políticas del caller.
//!
//! Cadena dimensional, independiente de etiquetas de horizonte:
//! 1. p = media beta - z·desviación (aproximación, no cuantil beta exacto).
//! 2. f = Kelly conservador, shrinkage y topes de política; cero significa
//!    que esta envolvente NO autoriza exposición, también en cuentas micro.
//! 3. Presupuesto B=C·f, en la moneda de C. No se financia exploración implícita.
//! 4. Pérdida nominal N·(d+c), con N=|Q|·P, distancia relativa d y coste c.
//! 5. N_max=B/(d+c). Si el mínimo realizable excede N_max, no hay entrada.
//!
//! `max_leverage` conserva la API antigua, pero expresa N_max/C, NO el
//! multiplicador de margen del exchange. Cambiar margen N/L sin cambiar Q
//! no cambia la pérdida al stop. `exposure_budget` expone el presupuesto y
//! su proyección a lotes; el caller debe usar y revalidar la Q resultante.
//! No hay aquí modelo de gaps/liquidación ni garantía sobre fills futuros.
//!
//! El host/replay aún poseen un bootstrap explícito n<30 y un adaptador de
//! margen independientes. Corregir esta API no certifica esos consumidores
//! (FMT-113); no desplegar atribuyéndole un cierre del riesgo extremo a extremo.

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
            (self.alpha / total).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }

    pub fn sd(&self) -> f64 {
        let a = self.alpha;
        let b = self.beta;
        let total = a + b;
        if total > 0.0 {
            // Algebraically a*b/(total^2*(total+1)), without overflowing
            // intermediate products for finite, large posterior counts.
            ((a / total) * (b / total) / (total + 1.0)).sqrt()
        } else {
            0.1
        }
    }

    /// Aproximación normal inferior. Los niveles nominales asociados a z
    /// no garantizan cobertura beta exacta con muestras pequeñas/dependientes.
    pub fn lcb(&self, z: f64) -> f64 {
        // A numerical bound may reach zero. A positive floor is NOT a
        // confidence bound: it can invent an edge for rare-win payoffs.
        (self.mean() - z * self.sd()).clamp(0.0, 1.0)
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

/// Presupuesto nominal en moneda de cuenta, sin garantía frente a gaps.
/// Sólo se construye desde una envolvente válida; sus campos son inmutables
/// para que la proyección no pueda recibir un presupuesto corrupto.
#[derive(Debug, Clone, Copy)]
pub struct ExposureBudget {
    risk_budget_usd: f64,
    loss_per_notional: f64,
    max_notional: f64,
}

impl ExposureBudget {
    pub fn risk_budget_usd(&self) -> f64 {
        self.risk_budget_usd
    }

    /// Distancia al stop + reserva de fees/slippage, ambas fracciones.
    pub fn loss_per_notional(&self) -> f64 {
        self.loss_per_notional
    }

    pub fn max_notional(&self) -> f64 {
        self.max_notional
    }

    /// Proyecta cantidad (unidades del activo) hacia abajo a lotes factibles.
    /// `price` es moneda/unidad, `step_size` y `min_qty` son unidades;
    /// `min_notional` está en la moneda del presupuesto. No aumenta la propuesta.
    /// None significa dato inválido, precisión insuficiente o conjunto vacío.
    /// Los mínimos deben ser los del símbolo vigente; esta API no consulta red.
    /// Margen, inventario agregado y precio efectivo de fill se validan aparte.
    pub fn project_quantity(
        &self,
        proposed_qty: f64,
        price: f64,
        step_size: f64,
        min_qty: f64,
        min_notional: f64,
    ) -> Option<f64> {
        if [proposed_qty, price, step_size, min_notional]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
            || !min_qty.is_finite()
            || min_qty < 0.0
        {
            return None;
        }
        let limit_qty = proposed_qty.min(self.max_notional / price);
        let lots = (limit_qty / step_size).floor();
        // Beyond 2^53-1, decrementing a binary64 lot count cannot reliably
        // move exactly one integer lot. Reject rather than claim precision.
        if !lots.is_finite() || !(1.0..=9_007_199_254_740_991.0).contains(&lots) {
            return None;
        }
        let within_budget = |qty: f64| {
            let notional = qty * price;
            let loss = notional * self.loss_per_notional;
            qty.is_finite()
                && qty > 0.0
                && qty <= limit_qty
                && notional.is_finite()
                && notional > 0.0
                && notional <= self.max_notional
                && loss.is_finite()
                && loss > 0.0
                && loss <= self.risk_budget_usd
        };
        let mut qty = lots * step_size;
        if !within_budget(qty) {
            // A division/multiplication round trip can overshoot by an ULP.
            // Drop a lot and recheck; never relax a risk inequality by epsilon.
            qty = (lots - 1.0) * step_size;
        }
        if !within_budget(qty) || qty < min_qty || qty * price < min_notional {
            return None;
        }
        Some(qty)
    }
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
        if !self.posterior.alpha.is_finite()
            || self.posterior.alpha <= 0.0
            || !self.posterior.beta.is_finite()
            || self.posterior.beta <= 0.0
            || !n.is_finite()
            || n < 3.0
            || !z.is_finite()
            || z < 0.0
            || !shrinkage_k.is_finite()
            || shrinkage_k < 0.0
            || !(n + shrinkage_k).is_finite()
            || !self.payoff_ratio.is_finite()
            || self.payoff_ratio <= 0.0
        {
            return 0.0; // sin evidencia no hay apuesta — jamás prior optimista
        }
        let p = self.posterior.lcb(z);
        if !p.is_finite() {
            return 0.0;
        }
        let q = 1.0 - p;
        let b = self.payoff_ratio;
        // b was validated positive above. Flooring it would invent a more
        // favorable payoff; p-q/b also avoids the unnecessary product p*b.
        let kelly = p - q / b;
        if !kelly.is_finite() || kelly <= 0.0 {
            return 0.0; // el LCB dice que NO hay edge: fuera
        }
        // Shrinkage por evidencia: n/(n+k) — con pocos trades, fracción minúscula.
        let shrunk = kelly * (n / (n + shrinkage_k));

        // FIX #593 / CERT-M5-H03: streak-bound extraído a `ruin` — la MISMA
        // función que ahora acota bootstrap/micro/leverage_matrix (antes este
        // era el único path con protección de ruina).
        let q_lcb = (1.0 - self.posterior.lcb(z)).clamp(0.01, 0.99);
        shrunk.min(crate::ruin::streak_ruin_cap(q_lcb)).min(0.25) // axioma 25%
    }

    /// B=C*f y N_max=B/(distancia+coste). No usa pisos de posición ni bootstrap.
    /// El coste debe incluir la reserva de ida/vuelta elegida por el caller.
    /// Valida representabilidad; no reemplaza inputs corruptos por un trade.
    pub fn exposure_budget(
        &self,
        capital: f64,
        stop_distance_pct: f64,
        roundtrip_cost_pct: f64,
        z: f64,
        shrinkage_k: f64,
    ) -> Option<ExposureBudget> {
        if !capital.is_finite()
            || capital <= 0.0
            || !stop_distance_pct.is_finite()
            || stop_distance_pct <= 0.0
            || !roundtrip_cost_pct.is_finite()
            || roundtrip_cost_pct < 0.0
        {
            return None;
        }
        let f = self.risk_fraction(z, shrinkage_k);
        let risk_budget_usd = capital * f;
        let loss_per_notional = stop_distance_pct + roundtrip_cost_pct;
        let max_notional = risk_budget_usd / loss_per_notional;
        if [risk_budget_usd, loss_per_notional, max_notional]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return None;
        }
        Some(ExposureBudget {
            risk_budget_usd,
            loss_per_notional,
            max_notional,
        })
    }

    /// Compatibilidad: techo de exposición nominal N/C, NO leverage de margen.
    /// Conserva el piso prudencial histórico de 5 bps, sin inventar edge.
    /// Excluye costes: para sizing final usar exposure_budget y project_quantity.
    /// Si el mínimo excede B, devuelve (0,false); no concede excepciones micro.
    pub fn max_leverage(
        &self,
        capital: f64,
        stop_distance_pct: f64,
        exchange_min_notional: f64,
        z: f64,
        shrinkage_k: f64,
    ) -> (f64, bool) {
        // (fusión PR #5, fail-closed FMT): NaN no es capital positivo ni stop
        // ni mínimo ejecutables; se rechaza antes de que propague. Un
        // posterior INVÁLIDO (alpha/beta NaN) tampoco admite candidatos:
        // f64::min ignora NaN y la sonda mínima colaría sin este guard.
        if !capital.is_finite() || capital <= 0.0 {
            return (0.0, false);
        }
        if !(self.posterior.alpha.is_finite() && self.posterior.beta.is_finite()) {
            return (0.0, false);
        }
        if !stop_distance_pct.is_finite()
            || stop_distance_pct <= 0.0
            || !exchange_min_notional.is_finite()
            || exchange_min_notional <= 0.0
        {
            return (0.0, false);
        }
        let stop = stop_distance_pct.max(0.0005); // piso 5 bps: stop imposible de más cerca

        // D-750 — LA FRACCIÓN MÍNIMA EJECUTABLE ES UNA MEDIDA, NO UN NÚMERO.
        //
        // La orden más pequeña que el símbolo acepta tiene nocional
        // `exchange_min_notional`; con este stop pierde
        // `exchange_min_notional · stop` dólares si se toca, es decir esta
        // fracción del capital. Es la unidad de riesgo indivisible de este
        // símbolo: por debajo de ella no existe orden alguna. Nótese que el
        // apalancamiento NO aparece — reparte el mismo nocional entre margen y
        // préstamo, pero no cambia lo que se pierde.
        let f_min_ejecutable = exchange_min_notional * stop / capital;

        let mut f = self.risk_fraction(z, shrinkage_k);
        // FIX #702 / #793: arranque para cuentas en régimen micro — evita el
        // bloqueo bayesiano en que la cota inferior da `f ≤ 0` con muestras
        // pequeñas y el motor nunca genera la evidencia que necesita.
        //
        // D-750 — LA APUESTA DE EXPLORACIÓN ERA `0,015 · micro_w`: un 1,5 % del
        // capital arriesgado SIN EDGE, inventado. La única apuesta defendible
        // cuando no hay edge probado es la MÍNIMA EJECUTABLE: exactamente lo que
        // el exchange obliga a arriesgar para que exista una orden, ni un
        // céntimo más. Y se somete al mismo control de ruina que cualquier otra
        // fracción del sistema: si ni el mínimo cabe bajo el tope de ruina, la
        // exploración no es viable y se rechaza (abajo).
        let micro_w = crate::capital_regime::micro_weight(capital, exchange_min_notional);
        if f <= 0.0 && micro_w > 0.0 {
            let q_lcb = (1.0 - self.posterior.lcb(z)).clamp(0.01, 0.99);
            f = crate::ruin::clamp_ruin(f_min_ejecutable, q_lcb);
        }
        if f <= 0.0 || !f.is_finite() {
            return (0.0, false);
        }

        // D-750 — SI NO CABE, SE RECHAZA; JAMÁS SE INFLA.
        //
        // Antes, cuando el presupuesto de riesgo no alcanzaba para la orden
        // mínima, una puerta de escape devolvía `operable = true` con un
        // apalancamiento escrito a mano —`(min_notional/capital).max(1).min(5)`,
        // tras exigir `min_notional ≤ capital·5` y `f ≥ 0,005`: cuatro literales
        // sin derivación— e inflaba así la apuesta por encima de lo que el
        // propio control de ruina acababa de autorizar. Es el modo de fallo que
        // CERT-M5-C02 declaró cerrado, sobreviviendo aquí.
        //
        // La respuesta correcta es la de la aritmética: si la unidad de riesgo
        // indivisible del símbolo excede la fracción que el riesgo permite, NO
        // HAY ORDEN POSIBLE en este símbolo con este capital y este stop.
        if f < f_min_ejecutable {
            return (0.0, false); // capital insuficiente: protección, no leverage suicida
        }
        (f / stop, true)
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

    /// D-750 — LA PUERTA DE ESCAPE QUE INFLABA LA APUESTA.
    ///
    /// EL DEFECTO: cuando el presupuesto de riesgo no alcanzaba para la orden
    /// mínima del símbolo, en lugar de rechazar, el código devolvía
    /// `operable = true` con un apalancamiento escrito a mano
    /// —`(min_notional/capital).max(1).min(5)`— siempre que
    /// `min_notional ≤ capital · 5` y `f ≥ 0,005`. Con 13 $ de capital, un
    /// símbolo de 50 $ de mínimo y un stop del 10 %, la unidad de riesgo
    /// indivisible es el 38 % del capital mientras el riesgo autorizado ronda
    /// el 18 %: la puerta se abría y operaba igual, DUPLICANDO la fracción que
    /// el control de ruina acababa de permitir.
    ///
    /// Con el código viejo este test falla: devolvía `(3,85, true)`.
    #[test]
    fn si_la_orden_minima_no_cabe_se_rechaza_en_vez_de_inflar() {
        let mut env = RiskEnvelope::new();
        for i in 0..400 {
            env.record_trade(i % 20 < 11, 15.0, -10.0); // 55 % WR, payoff 1,5
        }
        let f = env.risk_fraction(1.64, 50.0);
        let capital = 13.0;
        let min_notional = 50.0;
        let stop = 0.10;
        let f_min_ejecutable = min_notional * stop / capital;
        // La premisa del test: hay edge medido, pero NO alcanza para la unidad
        // de riesgo indivisible de este símbolo.
        assert!(
            f > 0.005 && f < f_min_ejecutable,
            "premisa del test: {f} debe estar entre 0,005 y {f_min_ejecutable}"
        );
        // Y la puerta de escape vieja habría vinculado (min_notional ≤ 5·cap).
        assert!(min_notional <= capital * 5.0, "premisa del test");

        let (lev, operable) = env.max_leverage(capital, stop, min_notional, 1.64, 50.0);
        assert!(
            !operable && lev == 0.0,
            "la orden mínima no cabe en el riesgo permitido ⇒ se rechaza, \
             no se infla la apuesta; got ({lev}, {operable})"
        );
    }

    /// D-750 — LA APUESTA DE EXPLORACIÓN ES LA MÍNIMA EJECUTABLE, NO EL 1,5 %.
    ///
    /// EL DEFECTO: sin edge probado, el arranque micro apostaba
    /// `0,015 · micro_w` del capital — un número inventado. Ahora arriesga
    /// exactamente lo que el exchange obliga a arriesgar para que exista una
    /// orden, y ni un céntimo más: el apalancamiento resultante es justo el que
    /// realiza el nocional mínimo del símbolo.
    #[test]
    fn la_exploracion_arriesga_lo_minimo_ejecutable() {
        let env = RiskEnvelope::new(); // sin evidencia: risk_fraction = 0
        assert_eq!(env.risk_fraction(1.64, 50.0), 0.0, "premisa: sin edge");

        let capital = 13.0;
        let min_notional = 5.0;
        let stop = 0.02;
        let (lev, operable) = env.max_leverage(capital, stop, min_notional, 1.64, 50.0);
        assert!(operable, "la orden mínima cabe: debe poder explorarse");
        // f = min_notional·stop/capital  ⇒  L = f/stop = min_notional/capital.
        let esperado = min_notional / capital;
        assert!(
            (lev - esperado).abs() < 1e-9,
            "la exploración realiza EXACTAMENTE el nocional mínimo: {lev} vs {esperado}"
        );
        // El viejo `0,015 · micro_w` daba, con esta cuenta en micro pleno,
        // L = 0,015/0,02 = 0,75: arriesgaba 0,195 $ cuando la orden mínima sólo
        // obliga a arriesgar 0,10 $. Es decir, apostaba casi el DOBLE de lo
        // necesario sin ningún edge que lo respaldase.
        assert!(
            lev < 0.75,
            "explorar debe costar lo mínimo ejecutable, no el 1,5 % inventado: {lev}"
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
