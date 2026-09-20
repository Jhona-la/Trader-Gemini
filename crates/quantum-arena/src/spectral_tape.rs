//! ESPECTRO DEL TAPE — las tasas del mercado a todas las escalas y su
//! pronóstico honesto: volatilidad, volumen, flujo de dinero, intensidad y
//! concentración de entes.
//!
//! QUÉ ES. Cada trade real del exchange (tiempo, precio, cantidad, lado del
//! agresor) alimenta, a cada escala τ del espectro temporal, un estimador de
//! TASA con núcleo exponencial en tiempo continuo:
//!
//! ```text
//!   A_τ(t) = Σ_i v_i · e^{−(t − t_i)/τ}        tasa_τ(t) = A_τ(t) / (τ · (1 − e^{−T/τ}))
//! ```
//!
//! donde T es el tiempo observado. El denominador es la masa del núcleo que
//! los datos han llenado de verdad: con T ≪ τ la escala no ha visto aún un
//! τ entero y su tasa sale de lo observado, sin semillas inventadas. Los
//! cinco observables son la varianza realizada (Σ r²), el nocional
//! negociado, el nocional con signo del agresor (dinero que entra menos
//! dinero que sale), el número de trades y el nocional al cuadrado.
//!
//! LOS ENTES SIN UMBRALES. De esos observables sale, a cada escala, la
//! concentración `R = N·Σv² / (Σv)²` —el tamaño del trade en el que se
//! sienta el dólar medio dividido por el tamaño del trade medio—: vale 1 si
//! todos los trades son iguales y crece cuando pocas impresiones enormes
//! cargan el volumen (ballenas, bloques institucionales, liquidaciones en
//! cascada). No hay «tamaño de ballena» fijado a mano: la escala τ se anula
//! en el cociente y el propio tape define qué es grande.
//!
//! EL PRONÓSTICO. Para un horizonte h cualquiera, el valor que el mercado
//! acumulará en (t, t+h] se predice como la PERSISTENCIA —lo que la escala
//! más cercana a h dice hoy— más una corrección lineal en la FORMA del
//! espectro (log-cocientes entre escalas), aprendida en línea por mínimos
//! cuadrados recursivos. Es la generalización espectral del HAR de Corsi
//! (2009): en lugar de tres ventanas fijas (día, semana, mes), la ventana
//! del espectro centrada en el horizonte. El modelo sólo se actualiza cuando
//! el objetivo ya ocurrió, así que cada predicción puntuada es fuera de
//! muestra por construcción, y se compara contra la persistencia —el
//! pronóstico ingenuo que en volatilidad y volumen es difícil de batir— y
//! contra la climatología (media móvil del objetivo).

use crate::temporal_spectrum::SPECTRUM_SCALES_MS;
use std::collections::VecDeque;

/// Índices de los observables acumulados.
pub const OBS_VAR: usize = 0;
pub const OBS_NOTIONAL: usize = 1;
pub const OBS_SIGNED: usize = 2;
pub const OBS_COUNT: usize = 3;
pub const OBS_NOTIONAL_SQ: usize = 4;
pub const N_OBS: usize = 5;

/// Tasas del mercado a varias escalas del espectro temporal.
#[derive(Clone, Debug)]
pub struct SpectralTape {
    taus: Vec<f64>,
    acc: Vec<[f64; N_OBS]>,
    totals: [f64; N_OBS],
    first_ts: u64,
    last_ts: u64,
    last_price: f64,
    /// |r| no nulo más pequeño visto: el tick efectivo del instrumento,
    /// medido en el tape (suelo de la varianza de un horizonte sin cambios).
    min_abs_ret: f64,
    /// Nocional no nulo más pequeño visto (suelo del volumen de un horizonte
    /// sin trades).
    min_notional: f64,
    started: bool,
}

impl SpectralTape {
    /// Espectro con las escalas dadas (ms).
    pub fn new(taus_ms: &[f64]) -> Self {
        let taus: Vec<f64> = taus_ms
            .iter()
            .copied()
            .filter(|t| t.is_finite() && *t > 0.0)
            .collect();
        let n = taus.len();
        Self {
            taus,
            acc: vec![[0.0; N_OBS]; n],
            totals: [0.0; N_OBS],
            first_ts: 0,
            last_ts: 0,
            last_price: 0.0,
            min_abs_ret: f64::INFINITY,
            min_notional: f64::INFINITY,
            started: false,
        }
    }

    /// Las escalas del espectro global comprendidas en [min, max] ms. La
    /// cota inferior útil es la resolución del reloj del tape (1 ms en
    /// Binance): por debajo, dos escalas no se distinguen. La superior es lo
    /// que el operador quiera observar; una escala más larga que los datos no
    /// estorba —su tasa sale de lo observado— pero tampoco informa.
    pub fn with_band(min_tau_ms: f64, max_tau_ms: f64) -> Self {
        let taus: Vec<f64> = SPECTRUM_SCALES_MS
            .iter()
            .copied()
            .filter(|t| *t >= min_tau_ms && *t <= max_tau_ms)
            .collect();
        Self::new(&taus)
    }

    pub fn taus(&self) -> &[f64] {
        &self.taus
    }

    /// Sumas acumuladas desde el arranque (para medir lo que ocurre en una
    /// ventana: diferencia de dos instantáneas).
    pub fn totals(&self) -> [f64; N_OBS] {
        self.totals
    }

    pub fn last_ts(&self) -> u64 {
        self.last_ts
    }

    pub fn last_price(&self) -> f64 {
        self.last_price
    }

    pub fn is_started(&self) -> bool {
        self.started
    }

    /// Suelo de la varianza acumulada en una ventana sin cambios de precio.
    pub fn variance_floor(&self) -> f64 {
        if self.min_abs_ret.is_finite() {
            self.min_abs_ret * self.min_abs_ret
        } else {
            0.0
        }
    }

    /// Suelo del nocional acumulado en una ventana sin trades.
    pub fn notional_floor(&self) -> f64 {
        if self.min_notional.is_finite() {
            self.min_notional
        } else {
            0.0
        }
    }

    /// Un trade real. `buyer_aggressor` = el comprador cruzó el spread.
    pub fn on_trade(&mut self, ts_ms: u64, price: f64, qty: f64, buyer_aggressor: bool) {
        if !price.is_finite() || price <= 0.0 || !qty.is_finite() || qty <= 0.0 {
            return;
        }
        if !self.started {
            self.started = true;
            self.first_ts = ts_ms;
            self.last_ts = ts_ms;
            self.last_price = price;
        }
        if ts_ms < self.last_ts {
            // Reloj hacia atrás: el evento no puede reordenarse sin romper la
            // causalidad de los núcleos; se descarta.
            return;
        }
        let dt = (ts_ms - self.last_ts) as f64;
        if dt > 0.0 {
            for (k, tau) in self.taus.iter().enumerate() {
                let d = (-dt / tau).exp();
                for v in self.acc[k].iter_mut() {
                    *v *= d;
                }
            }
        }
        let r = (price / self.last_price).ln();
        let r2 = r * r;
        if r != 0.0 && r.abs() < self.min_abs_ret {
            self.min_abs_ret = r.abs();
        }
        let notional = price * qty;
        if notional < self.min_notional {
            self.min_notional = notional;
        }
        let signed = if buyer_aggressor { notional } else { -notional };
        let obs = [r2, notional, signed, 1.0, notional * notional];
        for a in self.acc.iter_mut() {
            for (v, o) in a.iter_mut().zip(obs.iter()) {
                *v += o;
            }
        }
        for (t, o) in self.totals.iter_mut().zip(obs.iter()) {
            *t += o;
        }
        self.last_ts = ts_ms;
        self.last_price = price;
    }

    /// Tiempo observado hasta `now_ms`.
    pub fn elapsed_ms(&self, now_ms: u64) -> f64 {
        if !self.started {
            return 0.0;
        }
        now_ms.saturating_sub(self.first_ts) as f64
    }

    /// Fracción del núcleo de la escala k que los datos ya llenaron.
    pub fn warm(&self, k: usize, now_ms: u64) -> f64 {
        let t = self.elapsed_ms(now_ms);
        1.0 - (-t / self.taus[k]).exp()
    }

    /// Suma del núcleo de la escala k llevada a `now_ms` (sin tocar estado).
    fn kernel(&self, k: usize, obs: usize, now_ms: u64) -> f64 {
        let dt = now_ms.saturating_sub(self.last_ts) as f64;
        self.acc[k][obs] * (-dt / self.taus[k]).exp()
    }

    /// Tasa (por ms) del observable a la escala k, corregida por la masa del
    /// núcleo observada. `None` antes de observar nada.
    pub fn rate(&self, k: usize, obs: usize, now_ms: u64) -> Option<f64> {
        let w = self.warm(k, now_ms);
        if !self.started || w <= 0.0 {
            return None;
        }
        Some(self.kernel(k, obs, now_ms) / (self.taus[k] * w))
    }

    /// Desequilibrio del flujo a la escala k: (compras − ventas)/total ∈ [−1, 1].
    pub fn flow_imbalance(&self, k: usize, now_ms: u64) -> f64 {
        let n = self.kernel(k, OBS_NOTIONAL, now_ms);
        if n > 0.0 {
            (self.kernel(k, OBS_SIGNED, now_ms) / n).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// Concentración de entes a la escala k: N·Σv²/(Σv)² ≥ 1.
    pub fn concentration(&self, k: usize, now_ms: u64) -> f64 {
        let n = self.kernel(k, OBS_NOTIONAL, now_ms);
        if n > 0.0 {
            (self.kernel(k, OBS_COUNT, now_ms) * self.kernel(k, OBS_NOTIONAL_SQ, now_ms) / (n * n))
                .max(1.0)
        } else {
            1.0
        }
    }

    /// Escala del espectro más cercana (en log τ) a `tau_ms`.
    pub fn nearest_scale(&self, tau_ms: f64) -> usize {
        let lt = tau_ms.max(1e-9).ln();
        let mut best = 0;
        let mut best_d = f64::INFINITY;
        for (k, t) in self.taus.iter().enumerate() {
            let d = (t.ln() - lt).abs();
            if d < best_d {
                best_d = d;
                best = k;
            }
        }
        best
    }
}

/// Lo que se pronostica sobre la ventana (t, t+h].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForecastTarget {
    /// ln de la varianza realizada Σ r².
    Variance,
    /// ln del nocional negociado.
    Volume,
    /// ln del número de trades.
    Intensity,
    /// Desequilibrio del flujo de dinero ∈ [−1, 1].
    Flow,
    /// ln de la concentración de entes.
    Concentration,
}

impl ForecastTarget {
    pub const ALL: [ForecastTarget; 5] = [
        ForecastTarget::Variance,
        ForecastTarget::Volume,
        ForecastTarget::Intensity,
        ForecastTarget::Flow,
        ForecastTarget::Concentration,
    ];

    pub fn nombre(self) -> &'static str {
        match self {
            ForecastTarget::Variance => "volatilidad (ln Σr²)",
            ForecastTarget::Volume => "volumen (ln nocional)",
            ForecastTarget::Intensity => "intensidad (ln trades)",
            ForecastTarget::Flow => "flujo de dinero (desequilibrio)",
            ForecastTarget::Concentration => "concentración de entes (ln R)",
        }
    }
}

/// Mínimos cuadrados recursivos con olvido exponencial.
#[derive(Clone, Debug)]
pub struct Rls {
    p: usize,
    theta: Vec<f64>,
    pm: Vec<f64>,
    lambda: f64,
}

impl Rls {
    /// `lambda` = 1 − 1/memoria (en muestras). `prior_var` = varianza a
    /// priori de cada coeficiente alrededor de cero: con ella el primer
    /// ajuste es una cresta y no una inversión mal condicionada.
    pub fn new(p: usize, lambda: f64, prior_var: f64) -> Self {
        let mut pm = vec![0.0; p * p];
        for i in 0..p {
            pm[i * p + i] = prior_var;
        }
        Self {
            p,
            theta: vec![0.0; p],
            pm,
            lambda: lambda.clamp(0.5, 1.0),
        }
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        self.theta.iter().zip(x.iter()).map(|(a, b)| a * b).sum()
    }

    pub fn update(&mut self, x: &[f64], y: f64) {
        if !y.is_finite() || x.iter().any(|v| !v.is_finite()) {
            return;
        }
        let p = self.p;
        // Px
        let mut px = vec![0.0; p];
        for i in 0..p {
            let mut s = 0.0;
            for j in 0..p {
                s += self.pm[i * p + j] * x[j];
            }
            px[i] = s;
        }
        let denom = self.lambda + x.iter().zip(px.iter()).map(|(a, b)| a * b).sum::<f64>();
        if !denom.is_finite() || denom <= 1e-300 {
            return;
        }
        let err = y - self.predict(x);
        for i in 0..p {
            self.theta[i] += px[i] * err / denom;
        }
        // P ← (P − Px·xᵀP / denom) / λ, simétrica.
        for i in 0..p {
            for j in 0..p {
                let v = (self.pm[i * p + j] - px[i] * px[j] / denom) / self.lambda;
                self.pm[i * p + j] = v;
            }
        }
    }

    pub fn theta(&self) -> &[f64] {
        &self.theta
    }
}

/// Puntuación prequential: cada predicción se puntúa antes de que el modelo
/// vea su objetivo.
#[derive(Clone, Debug, Default)]
pub struct ForecastScore {
    pub n: u64,
    pub sse_model: f64,
    pub sse_persist: f64,
    pub sse_clim: f64,
}

impl ForecastScore {
    /// Habilidad frente a la persistencia: 1 − SSE_modelo/SSE_persistencia.
    /// Positiva = el modelo bate al pronóstico ingenuo.
    pub fn skill_vs_persistence(&self) -> f64 {
        if self.sse_persist > 0.0 {
            1.0 - self.sse_model / self.sse_persist
        } else {
            0.0
        }
    }

    /// R² fuera de muestra frente a la climatología (media móvil).
    pub fn skill_vs_climatology(&self) -> f64 {
        if self.sse_clim > 0.0 {
            1.0 - self.sse_model / self.sse_clim
        } else {
            0.0
        }
    }

    /// Habilidad de la propia persistencia frente a la climatología: cuánto
    /// del objetivo ya explica el pronóstico ingenuo.
    pub fn persistence_vs_climatology(&self) -> f64 {
        if self.sse_clim > 0.0 {
            1.0 - self.sse_persist / self.sse_clim
        } else {
            0.0
        }
    }
}

#[derive(Clone, Debug)]
struct Pending {
    t0: u64,
    x: Vec<f64>,
    persist: f64,
    totals0: [f64; N_OBS],
}

/// Pronosticador de un objetivo a un horizonte.
#[derive(Clone, Debug)]
pub struct HorizonForecaster {
    pub target: ForecastTarget,
    pub horizon_ms: f64,
    feat: Vec<usize>,
    base: usize,
    rls: Rls,
    pending: VecDeque<Pending>,
    stride_ms: u64,
    next_sample_ms: u64,
    matured: u64,
    warmup: u64,
    clim_sum: f64,
    clim_n: f64,
    clim_lambda: f64,
    pub score: ForecastScore,
    last_prediction: Option<f64>,
}

impl HorizonForecaster {
    /// Ventana del espectro que informa el horizonte h: dos escalas de base 4
    /// por debajo y tres por encima (h/16 … 64·h), la extensión espectral de
    /// las ventanas día/semana/mes del HAR. La memoria del ajuste es la de la
    /// escala más larga que usa: el modelo no recuerda más lejos de lo que
    /// sus propias variables ven. Se muestrea cada h/4 (objetivos solapados).
    pub fn new(tape: &SpectralTape, target: ForecastTarget, horizon_ms: f64) -> Self {
        let base = tape.nearest_scale(horizon_ms);
        let lo = base.saturating_sub(2);
        let hi = (base + 3).min(tape.taus().len() - 1);
        let feat: Vec<usize> = (lo..=hi).filter(|k| *k != base).collect();
        let p = feat.len() + 1;
        let stride_ms = ((horizon_ms / 4.0).round() as u64).max(1);
        let memory_ms = tape.taus()[hi].max(horizon_ms);
        let memory_samples = (memory_ms / stride_ms as f64).max(4.0 * p as f64);
        let lambda = 1.0 - 1.0 / memory_samples;
        Self {
            target,
            horizon_ms,
            feat,
            base,
            rls: Rls::new(p, lambda, 1.0),
            pending: VecDeque::new(),
            stride_ms,
            next_sample_ms: 0,
            matured: 0,
            warmup: memory_samples.ceil() as u64,
            clim_sum: 0.0,
            clim_n: 0.0,
            clim_lambda: lambda,
            score: ForecastScore::default(),
            last_prediction: None,
        }
    }

    /// Valor del observable acumulado en una ventana, en las unidades del
    /// objetivo (log para los positivos).
    fn realized(&self, tape: &SpectralTape, d: &[f64; N_OBS]) -> Option<f64> {
        match self.target {
            ForecastTarget::Variance => Some((d[OBS_VAR] + tape.variance_floor()).ln()),
            ForecastTarget::Volume => Some((d[OBS_NOTIONAL] + tape.notional_floor()).ln()),
            ForecastTarget::Intensity => Some((d[OBS_COUNT] + 1.0).ln()),
            ForecastTarget::Flow => {
                if d[OBS_NOTIONAL] > 0.0 {
                    Some((d[OBS_SIGNED] / d[OBS_NOTIONAL]).clamp(-1.0, 1.0))
                } else {
                    None
                }
            }
            ForecastTarget::Concentration => {
                if d[OBS_NOTIONAL] > 0.0 && d[OBS_COUNT] >= 2.0 {
                    Some(
                        (d[OBS_COUNT] * d[OBS_NOTIONAL_SQ] / (d[OBS_NOTIONAL] * d[OBS_NOTIONAL]))
                            .max(1.0)
                            .ln(),
                    )
                } else {
                    None
                }
            }
        }
    }

    /// Lo que la escala k dice que se acumulará en h, en unidades del objetivo.
    fn implied(&self, tape: &SpectralTape, k: usize, now: u64) -> Option<f64> {
        let h = self.horizon_ms;
        match self.target {
            ForecastTarget::Variance => {
                Some((tape.rate(k, OBS_VAR, now)? * h + tape.variance_floor()).ln())
            }
            ForecastTarget::Volume => {
                Some((tape.rate(k, OBS_NOTIONAL, now)? * h + tape.notional_floor()).ln())
            }
            ForecastTarget::Intensity => Some((tape.rate(k, OBS_COUNT, now)? * h + 1.0).ln()),
            ForecastTarget::Flow => Some(tape.flow_imbalance(k, now)),
            ForecastTarget::Concentration => Some(tape.concentration(k, now).ln()),
        }
    }

    /// Vector de rasgos: sesgo + forma del espectro respecto de la escala
    /// base, y el pronóstico de persistencia.
    fn features(&self, tape: &SpectralTape, now: u64) -> Option<(Vec<f64>, f64)> {
        let persist = self.implied(tape, self.base, now)?;
        let mut x = Vec::with_capacity(self.feat.len() + 1);
        x.push(1.0);
        for &k in &self.feat {
            x.push(self.implied(tape, k, now)? - persist);
        }
        Some((x, persist))
    }

    /// Pronóstico actual para (now, now+h], en unidades del objetivo.
    pub fn predict(&self, tape: &SpectralTape, now: u64) -> Option<f64> {
        let (x, persist) = self.features(tape, now)?;
        Some(persist + self.rls.predict(&x))
    }

    pub fn last_prediction(&self) -> Option<f64> {
        self.last_prediction
    }

    pub fn warmed_up(&self) -> bool {
        self.matured >= self.warmup
    }

    /// Llamar ANTES de `tape.on_trade(ts, …)`: madura las muestras cuya
    /// ventana (t0, t0+h] terminó antes de este evento.
    pub fn before_trade(&mut self, tape: &SpectralTape, ts_ms: u64) {
        let h = self.horizon_ms as u64;
        let totals = tape.totals();
        while let Some(front) = self.pending.front() {
            if front.t0 + h >= ts_ms {
                break;
            }
            let s = self.pending.pop_front().expect("frente comprobado");
            let mut d = [0.0; N_OBS];
            for i in 0..N_OBS {
                d[i] = totals[i] - s.totals0[i];
            }
            if let Some(y) = self.realized(tape, &d) {
                if self.matured >= self.warmup {
                    let pred = s.persist + self.rls.predict(&s.x);
                    let clim = if self.clim_n > 0.0 {
                        self.clim_sum / self.clim_n
                    } else {
                        s.persist
                    };
                    self.score.n += 1;
                    self.score.sse_model += (y - pred).powi(2);
                    self.score.sse_persist += (y - s.persist).powi(2);
                    self.score.sse_clim += (y - clim).powi(2);
                }
                self.rls.update(&s.x, y - s.persist);
                self.clim_sum = self.clim_sum * self.clim_lambda + y;
                self.clim_n = self.clim_n * self.clim_lambda + 1.0;
                self.matured += 1;
            }
        }
    }

    /// Llamar DESPUÉS de `tape.on_trade(ts, …)`: toma una muestra si toca.
    pub fn after_trade(&mut self, tape: &SpectralTape, ts_ms: u64) {
        if ts_ms < self.next_sample_ms {
            return;
        }
        self.next_sample_ms = ts_ms + self.stride_ms;
        if let Some((x, persist)) = self.features(tape, ts_ms) {
            self.last_prediction = Some(persist + self.rls.predict(&x));
            self.pending.push_back(Pending {
                t0: ts_ms,
                x,
                persist,
                totals0: tape.totals(),
            });
        }
    }

    pub fn coefficients(&self) -> &[f64] {
        self.rls.theta()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tasa_de_un_proceso_constante_converge_a_su_valor() {
        // Un trade de 1 unidad a precio 100 cada 10 ms: nocional 100/10 ms
        // = 10 por ms; la tasa debe verlo en todas las escalas calentadas,
        // también en las más largas que el tiempo observado.
        let mut tape = SpectralTape::new(&[100.0, 1_000.0, 100_000.0]);
        for i in 0..20_000u64 {
            tape.on_trade(i * 10, 100.0, 1.0, i % 2 == 0);
        }
        // Se consulta a mitad del intervalo entre trades: justo en el instante
        // de un salto el núcleo exponencial sobrestima la tasa en ≈ Δ/(2τ)
        // (la suma discreta de e^{−jΔ/τ} frente a su integral).
        let now = tape.last_ts() + 5;
        for k in 0..3 {
            let r = tape.rate(k, OBS_NOTIONAL, now).unwrap();
            assert!((r - 10.0).abs() / 10.0 < 0.05, "escala {} tasa {}", k, r);
        }
        // Alternancia perfecta de agresores ⇒ desequilibrio ≈ 0.
        assert!(tape.flow_imbalance(2, now).abs() < 0.01);
        // Trades iguales ⇒ concentración = 1.
        assert!((tape.concentration(2, now) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn concentracion_detecta_impresiones_gigantes_sin_umbral() {
        let mut tape = SpectralTape::new(&[10_000.0]);
        for i in 0..1_000u64 {
            let qty = if i % 100 == 0 { 500.0 } else { 1.0 };
            tape.on_trade(i * 10, 100.0, qty, true);
        }
        let c = tape.concentration(0, tape.last_ts());
        assert!(c > 10.0, "pocos bloques enormes deben elevar R, got {}", c);
    }

    #[test]
    fn rls_recupera_una_relacion_lineal() {
        let mut rls = Rls::new(2, 1.0, 100.0);
        for i in 0..500 {
            let x = (i as f64 * 0.37).sin();
            rls.update(&[1.0, x], 0.5 + 2.0 * x);
        }
        assert!((rls.theta()[0] - 0.5).abs() < 1e-3);
        assert!((rls.theta()[1] - 2.0).abs() < 1e-3);
    }

    #[test]
    fn el_pronostico_nunca_ve_su_objetivo() {
        // Volumen que cambia de régimen: el objetivo de cada muestra sólo se
        // conoce al vencer su ventana; antes de eso no hay puntuación.
        let mut tape = SpectralTape::with_band(1_000.0, 300_000.0);
        let mut f = HorizonForecaster::new(&tape, ForecastTarget::Volume, 4_294.967296);
        let mut ts = 0u64;
        for i in 0..200_000u64 {
            ts += if (i / 20_000) % 2 == 0 { 5 } else { 50 };
            f.before_trade(&tape, ts);
            tape.on_trade(ts, 100.0 + (i % 7) as f64 * 0.01, 1.0, i % 3 == 0);
            f.after_trade(&tape, ts);
        }
        assert!(f.score.n > 0);
        // La persistencia ya explica los regímenes largos; el modelo no debe
        // hacerlo peor que ella en un proceso tan regular.
        assert!(f.score.skill_vs_persistence() > -0.05, "{:?}", f.score);
    }
}
