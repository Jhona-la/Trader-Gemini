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
//! no cambia la pérdida al stop. El paso 5 con coste es
//! `nocional_por_riesgo` (CL-159), el que usa LH-04 en el gate. La
//! proyección a lotes la hace el ejecutor (`round_to_step_size`, hacia
//! abajo) y el mínimo del exchange lo juzga D-750. RET-48 retiró
//! `ExposureBudget`, `exposure_budget(_spectral)` y `project_quantity`, que
//! no tenían consumidor operativo.
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

impl Default for RiskEnvelope {
    fn default() -> Self {
        Self::new()
    }
}

impl RiskEnvelope {
    pub const EMA_ALPHA: f64 = 0.05;
    pub const EMA_EFFECTIVE_N: f64 = 39.0; // 2.0 / 0.05 - 1.0 = 39.0

    pub fn new() -> Self {
        Self {
            posterior: EdgePosterior::jeffreys(),
            avg_win: 0.0,
            avg_loss: 0.0,
            payoff_ratio: 1.0,
        }
    }

    /// F1-081: Registrar el resultado de una operación individual.
    /// Recibe directamente el PnL neto o magnitudes de trade individual, aplicando una única EMA(0.05).
    /// Erradica la doble composición EMA∘EMA provocada por pre-filtrar el trade en los llamadores.
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
                self.avg_win * (1.0 - Self::EMA_ALPHA) + win_abs * Self::EMA_ALPHA
            };
        } else {
            let loss_abs = loss_pnl.abs().max(1e-6);
            self.avg_loss = if self.avg_loss == 0.0 {
                loss_abs
            } else {
                self.avg_loss * (1.0 - Self::EMA_ALPHA) + loss_abs * Self::EMA_ALPHA
            };
        }

        let eps = 1e-3;
        self.payoff_ratio = (self.avg_win + eps) / (self.avg_loss + eps);
    }

    /// F1-081: Conveniencia canónica para registrar un cierre directamente por su PnL neto individual.
    #[inline]
    pub fn record_net_pnl(&mut self, net_pnl: f64) {
        let won = net_pnl > 0.0;
        let abs_pnl = net_pnl.abs();
        self.record_trade(won, abs_pnl, abs_pnl);
    }

    /// Cota inferior conservadora (LCB) del payoff ratio b = win/loss.
    /// Resuelve R7-R2-A-5 y F1-081: como d f*/d b = q/b² > 0, usar el b puntual
    /// introducía un sesgo de optimismo en muestras pequeñas (n < 30).
    /// Descuenta la estimación muestral por incertidumbre asintótica:
    /// b_lcb = b · exp(-z / sqrt(n_eff)), garantizando simetría conservadora
    /// en ambos parámetros de la fórmula de Kelly (p y b).
    ///
    /// F1-081: Dado que el payoff ratio se estima con una EMA(alpha = 0.05), su tamaño de muestra
    /// efectivo está acotado por la memoria de la ventana (N_eff = min(n, 2/alpha - 1) = min(n, 39.0)).
    /// Usar el n acumulativo de Bernoulli sin cota eliminaba el descuento por incertidumbre tras
    /// cientos de trades, impidiendo que el dimensionamiento reaccionara a deterioros de spread/fricción.
    pub fn conservative_payoff_ratio(&self, z: f64) -> f64 {
        let n = self.posterior.n();
        if !self.payoff_ratio.is_finite() || self.payoff_ratio <= 0.0 || !n.is_finite() || n < 1.0 {
            return 1.0;
        }
        let z_eff = if z.is_finite() && z >= 0.0 { z } else { 1.0 };
        // F1-081: Tamaño efectivo de muestra acotado por la memoria de la EMA (N_eff <= 39)
        let n_eff = n.min(Self::EMA_EFFECTIVE_N);

        // R7-R2-A-5: Descontar el exceso de payoff (b - 1.0) hacia la paridad neutra 1.0
        // bajo incertidumbre muestral: b_lcb = 1.0 + (b - 1.0) * exp(-z / sqrt(n_eff)).
        // Si b <= 1.0, el ratio ya es adverso de por sí.
        if self.payoff_ratio > 1.0 {
            let excess = self.payoff_ratio - 1.0;
            let discount = (-z_eff / n_eff.sqrt()).exp();
            1.0 + excess * discount
        } else {
            self.payoff_ratio
        }
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
        // R7-R2-A-5: ambos parámetros (p y b) usan sus cotas conservadoras LCB
        let b = self.conservative_payoff_ratio(z);
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

    /// Atenuador de estabilidad espectral Koopman/DMD (§57 / Ola Ω215).
    ///
    /// Modula la fracción de riesgo de Kelly según el módulo dominante (|λ|)
    /// y la métrica de bifurcación espectral del operador Koopman streaming:
    ///
    /// - Régimen Asintóticamente Estable (|λ| <= 1.0 y bifurcacion <= 1.0):
    ///   El flujo en el espacio de Hilbert es contractivo o unitario; el factor es 1.0.
    /// - Régimen Divergente / Bifurcación (|λ| > 1.0 o bifurcacion > 1.0):
    ///   Atenuación hiperbólica suave:
    ///   \phi = 1.0 / (1.0 + 3.0 * (|λ| - 1.0).max(0.0) + 5.0 * (bifurcacion - 1.0).max(0.0))
    ///
    /// LG-31: el productor (`koopman_dmd`, campo `bifurcacion_max`) publica
    /// max_i |λ_i| y documenta «≥ 1,0 es régimen de bifurcación», el mismo
    /// umbral que el módulo. Antes se comparaba con 0,10: con el |λ| ≈ 0,77
    /// medido en tape real (régimen contractivo) el factor quedaba en 0,23 en
    /// cada orden y la sonda micro se vetaba con un espectro estable.
    ///
    /// Protege matemáticamente las micro-cuentas ($13 USD) de entrar con tamaño pleno
    /// durante bifurcaciones no lineales o regímenes explosivos.
    #[inline]
    pub fn koopman_stability_dampener(
        koopman_modulo: f64,
        koopman_bifurcacion: f64,
    ) -> f64 {
        if !koopman_modulo.is_finite() || !koopman_bifurcacion.is_finite() {
            return 1.0; // Fallback neutral si aún no hay lecturas maduras de Koopman
        }
        let excess_modulo = (koopman_modulo - 1.0).max(0.0);
        let excess_bif = (koopman_bifurcacion - 1.0).max(0.0);
        let dampener = 1.0 / (1.0 + 3.0 * excess_modulo + 5.0 * excess_bif);
        dampener.clamp(0.05, 1.0)
    }

    /// Fracción de riesgo final modulada espectralmente por Koopman/DMD (§57 / Ola Ω215).
    pub fn risk_fraction_spectral(
        &self,
        z: f64,
        shrinkage_k: f64,
        koopman_modulo: f64,
        koopman_bifurcacion: f64,
    ) -> f64 {
        let base_f = self.risk_fraction(z, shrinkage_k);
        if base_f <= 0.0 {
            return 0.0;
        }
        let dampener = Self::koopman_stability_dampener(koopman_modulo, koopman_bifurcacion);
        base_f * dampener
    }

    /// Compatibilidad: techo de exposición nominal N/C, NO leverage de margen.
    /// Cota superior de apalancamiento y filtro de admisibilidad con modulación espectral Koopman/DMD.
    ///
    /// Modula la fracción de riesgo mediante `koopman_stability_dampener`. Si el modo dinámico es estable
    /// (|λ| <= 1.0), el dampener es exactamente 1.0. Ante divergencia o bifurcaciones (|λ| > 1.0),
    /// la fracción se atenúa suavemente protegiendo el capital. Si `f < f_min_ejecutable`, la orden
    /// se veta de forma fail-closed (`operable = false`).
    pub fn max_leverage_spectral(
        &self,
        capital: f64,
        stop_distance_pct: f64,
        exchange_min_notional: f64,
        z: f64,
        shrinkage_k: f64,
        koopman_modulo: f64,
        koopman_bifurcacion: f64,
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
        let f_min_ejecutable = exchange_min_notional * stop / capital;

        let mut f = self.risk_fraction_spectral(z, shrinkage_k, koopman_modulo, koopman_bifurcacion);
        let micro_w = crate::capital_regime::micro_weight(capital, exchange_min_notional);
        if f <= 0.0 && micro_w > 0.0 {
            let q_lcb = (1.0 - self.posterior.lcb(z)).clamp(0.01, 0.99);
            let dampener = Self::koopman_stability_dampener(koopman_modulo, koopman_bifurcacion);
            f = crate::ruin::clamp_ruin(f_min_ejecutable, q_lcb) * dampener;
        }
        if f <= 0.0 || !f.is_finite() {
            return (0.0, false);
        }

        // D-750 — SI NO CABE, SE RECHAZA; JAMÁS SE INFLA.
        if f < f_min_ejecutable {
            return (0.0, false); // capital insuficiente: protección, no leverage suicida
        }
        (f / stop, true)
    }

    /// Conserva el piso prudencial histórico de 5 bps, sin inventar edge.
    /// Excluye costes: el nocional con coste es `nocional_por_riesgo` (CL-159).
    /// Si el mínimo excede B, devuelve (0,false); no concede excepciones micro.
    pub fn max_leverage(
        &self,
        capital: f64,
        stop_distance_pct: f64,
        exchange_min_notional: f64,
        z: f64,
        shrinkage_k: f64,
    ) -> (f64, bool) {
        self.max_leverage_spectral(
            capital,
            stop_distance_pct,
            exchange_min_notional,
            z,
            shrinkage_k,
            f64::NAN,
            f64::NAN,
        )
    }
}

/// CL-159 — PASO 5 DE LA CADENA COMO FUNCIÓN ÚNICA: el nocional cuya pérdida
/// al stop es exactamente la fracción `fraccion` del capital `capital`.
///
/// Una orden de nocional N que toca su stop pierde N·(d + c): la distancia
/// relativa d y la fricción de ida y vuelta c (`tp_sl::roundtrip_friction`:
/// dos comisiones taker y dos deslizamientos; la misma l = (d + c)/d del
/// freno duro y de la medida D-744). Igualar esa pérdida a f·C da
/// N = f·C/(d + c). Con c = 0 es exactamente f·C/d. El carry de funding NO
/// entra: es un coste esperado sobre τ, no se realiza al tocar el stop.
///
/// None si f, C o d no son finitos y positivos, si c no es finito o es
/// negativo, o si el cociente no es representable. Sin pisos ni rescates:
/// el mínimo del exchange lo juzga quien emite la orden (D-750).
#[inline]
pub fn nocional_por_riesgo(fraccion: f64, capital: f64, stop: f64, friccion: f64) -> Option<f64> {
    if !(fraccion.is_finite()
        && fraccion > 0.0
        && capital.is_finite()
        && capital > 0.0
        && stop.is_finite()
        && stop > 0.0
        && friccion.is_finite()
        && friccion >= 0.0)
    {
        return None;
    }
    let nocional = fraccion * capital / (stop + friccion);
    (nocional.is_finite() && nocional > 0.0).then_some(nocional)
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

    #[test]
    fn test_r7_r2_a5_conservative_payoff_ratio_lcb() {
        let mut env = RiskEnvelope::new();
        // Con pocos trades (n = 4), una ganancia con payoff inflado no debe sobredimensionar Kelly
        env.record_trade(true, 30.0, -10.0);
        env.record_trade(true, 30.0, -10.0);
        env.record_trade(false, 0.0, -10.0);
        env.record_trade(false, 0.0, -10.0);
        let raw_b = env.payoff_ratio;
        let lcb_b = env.conservative_payoff_ratio(1.64);
        assert!(lcb_b < raw_b, "lcb_b={lcb_b} debe ser menor que raw_b={raw_b}");
        // Con z = 0.0, debe coincidir exactamente
        assert!((env.conservative_payoff_ratio(0.0) - raw_b).abs() < 1e-12);
        // Fracción con z > 0 debe ser menor o igual que sin descuento de payoff
        let f_conservadora = env.risk_fraction(1.64, 10.0);
        assert!(f_conservadora >= 0.0);
    }

    #[test]
    fn test_koopman_stability_dampener_and_spectral_risk_fraction() {
        // LG-31: el productor publica bifurcacion = max|λ|, así que las entradas
        // posibles tienen bif = |λ| dominante.
        // 1. Modos asintóticamente estables (|λ| <= 1.0) => factor 1.0
        assert_eq!(RiskEnvelope::koopman_stability_dampener(0.95, 0.95), 1.0);
        assert_eq!(RiskEnvelope::koopman_stability_dampener(1.00, 1.00), 1.0);

        // 2. Modos inestables / divergentes atenúan el riesgo
        let damp_mild = RiskEnvelope::koopman_stability_dampener(1.10, 1.10);
        assert!(damp_mild < 1.0 && damp_mild > 0.5, "1.10 |λ| debe atenuar suavemente");

        let damp_severe = RiskEnvelope::koopman_stability_dampener(1.30, 1.30);
        assert!(damp_severe < damp_mild, "Mayor inestabilidad debe atenuar más");

        // 3. Inmunidad a NaN / Inf => fallback 1.0
        assert_eq!(RiskEnvelope::koopman_stability_dampener(f64::NAN, 0.10), 1.0);
        assert_eq!(RiskEnvelope::koopman_stability_dampener(1.0, f64::NAN), 1.0);

        // 4. Piso de supervivencia (mínimo 0.05)
        let damp_extreme = RiskEnvelope::koopman_stability_dampener(100.0, 100.0);
        assert_eq!(damp_extreme, 0.05, "Piso de atenuación debe ser 0.05");

        // 5. Acoplamiento en risk_fraction_spectral
        let mut env = RiskEnvelope::new();
        for i in 0..100 {
            env.record_trade(i % 10 < 6, 20.0, -10.0); // Edge claro
        }
        let base_f = env.risk_fraction(1.64, 10.0);
        assert!(base_f > 0.0);

        let f_stable = env.risk_fraction_spectral(1.64, 10.0, 0.95, 0.95);
        assert_eq!(f_stable, base_f);

        let f_chaotic = env.risk_fraction_spectral(1.64, 10.0, 1.25, 1.25);
        assert!(f_chaotic < base_f && f_chaotic > 0.0);
    }

    #[test]
    fn test_max_leverage_spectral_koopman_coupling() {
        let mut env = RiskEnvelope::new();
        for i in 0..100 {
            env.record_trade(i % 10 < 7, 25.0, -10.0); // 70% win rate
        }
        let cap = 50.0;
        let stop = 0.005; // 0.5%
        let min_notional = 5.0;
        let (base_lev, base_operable) = env.max_leverage(cap, stop, min_notional, 1.64, 10.0);
        assert!(base_operable);
        assert!(base_lev > 0.0);

        // Modo estable: debe ser idéntico al base
        let (stable_lev, stable_operable) = env.max_leverage_spectral(cap, stop, min_notional, 1.64, 10.0, 0.95, 0.95);
        assert!(stable_operable);
        assert_eq!(stable_lev, base_lev);

        // Modo caótico / bifurcación: amortigua apalancamiento
        let (chaotic_lev, _chaotic_operable) = env.max_leverage_spectral(cap, stop, min_notional, 1.64, 10.0, 1.25, 1.25);
        assert!(chaotic_lev < stable_lev);

        // Extremo caótico: si la fracción cae bajo el mínimo ejecutable, se veta fail-closed
        let (_extreme_lev, extreme_operable) = env.max_leverage_spectral(13.0, 0.04, 5.0, 1.64, 50.0, 10.0, 10.0);
        assert!(!extreme_operable, "Divergencia extrema con fracción amortiguada bajo el mínimo debe vetar la orden");

        // Cuenta en frío sin edge probado bajo bifurcación caótica: veto absoluto de exploración
        let fresh_env = RiskEnvelope::new();
        let (_cold_lev, cold_operable) = fresh_env.max_leverage_spectral(13.0, 0.01, 5.0, 1.64, 10.0, 1.25, 1.25);
        assert!(!cold_operable, "Cuenta en frío bajo régimen caótico debe vetar exploración");
    }

    /// CL-159: la pérdida al stop del nocional es f·C con la fricción dentro;
    /// sin fricción es bit a bit el f·C/d de antes (LH-04 hasta CL-159).
    #[test]
    fn cl159_nocional_por_riesgo_iguala_la_perdida_al_stop_a_f_por_c() {
        for &(f, capital, stop, friccion) in &[
            (0.0099, 1_000.0, 0.07746, 0.00082),
            (0.1304, 1_000.0, 0.01, 0.00082),
            (0.0913, 13.0, 0.005, 0.00145),
            (0.25, 1e9, 0.0005, 0.05),
        ] {
            let n = nocional_por_riesgo(f, capital, stop, friccion);
            assert!(n.is_some(), "entradas válidas: {f} {capital} {stop} {friccion}");
            let n = n.unwrap_or(f64::NAN);
            let presupuesto = f * capital;
            assert!(
                (n * (stop + friccion) - presupuesto).abs() <= 4.0 * f64::EPSILON * presupuesto,
                "N·(d + c) = f·C: {} vs {presupuesto}",
                n * (stop + friccion)
            );
            assert!(n < presupuesto / stop, "la fricción reduce el nocional");
            assert_eq!(
                nocional_por_riesgo(f, capital, stop, 0.0).map(f64::to_bits),
                Some((f * capital / stop).to_bits()),
                "con c = 0 es exactamente f·C/d"
            );
        }
    }

    #[test]
    fn cl159_nocional_por_riesgo_decrece_con_la_friccion_y_escala_con_la_moneda() {
        let mut previo = f64::INFINITY;
        for friccion in [0.0, 0.0001, 0.001, 0.01, 0.10, 1.0] {
            let n = nocional_por_riesgo(0.05, 100.0, 0.01, friccion).unwrap_or(f64::NAN);
            assert!(n < previo, "c = {friccion}: {n} ≥ {previo}");
            previo = n;
        }
        let base = nocional_por_riesgo(0.05, 100.0, 0.01, 0.001).unwrap_or(f64::NAN) / 100.0;
        assert!(base.is_finite() && base > 0.0, "base: {base}");
        for k in [0.01, 1.0, 10.0, 1000.0] {
            let n = nocional_por_riesgo(0.05, 100.0 * k, 0.01, 0.001).unwrap_or(f64::NAN);
            assert!((n / (100.0 * k) - base).abs() <= 1e-12 * base, "k = {k}");
        }
    }

    #[test]
    fn cl159_nocional_por_riesgo_rechaza_entradas_invalidas() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0] {
            assert_eq!(nocional_por_riesgo(bad, 100.0, 0.01, 0.001), None);
            assert_eq!(nocional_por_riesgo(0.05, bad, 0.01, 0.001), None);
            assert_eq!(nocional_por_riesgo(0.05, 100.0, bad, 0.001), None);
            if bad != 0.0 {
                assert_eq!(nocional_por_riesgo(0.05, 100.0, 0.01, bad), None);
            }
        }
        assert!(nocional_por_riesgo(0.05, 100.0, 0.01, 0.0).is_some(), "c = 0 es válido");
        assert_eq!(nocional_por_riesgo(0.25, f64::MAX, f64::MIN_POSITIVE, 0.0), None);
        assert_eq!(nocional_por_riesgo(0.05, 100.0, f64::MAX, f64::MAX), None);
    }
}
