use omniscient_registry::OmniscientRegistry;
use std::collections::VecDeque;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// QO-M2.2 — PROCESO DE HAWKES REAL con kernel exponencial.
///
/// HALLAZGO (auditoría matemática): el "kernel de Bessel" anterior era
/// `sqrt(1+dt²) − dt` — NO es una función de Bessel en ningún sentido, y
/// no había estructura de Hawkes: sin historia de eventos, sin suma de
/// excitación Σα·e^(−β(t−tᵢ)), sin condición de estacionariedad.
///
/// MATEMÁTICA REAL (Hawkes 1971, proceso auto-excitado):
///   λ(t) = μ + Σᵢ α·e^(−β·(t − tᵢ))   para tᵢ < t
/// donde:
///   - μ = intensidad base (llegada exógena de eventos)
///   - α = magnitud de auto-excitación (cada evento eleva λ en α)
///   - β = tasa de decaimiento del kernel exponencial
///   - tᵢ = timestamps de eventos anteriores
///
/// RATIO DE RAMIFICACIÓN: n = α/β
///   - n < 1: estacionario (la cascada se agota — régimen operativo)
///   - n ≥ 1: explosivo (cascada infinita — clúster de pánico)
///
/// La intensidad λ(t) mide el RITMO esperido de próximos eventos dado el
/// pasado: alta λ = cascada en curso (liquidaciones auto-excitándose);
/// el decay hacia μ mide cuánto queda de la cascada.
pub struct HawkesBesselEngine {
    registry: Option<Arc<OmniscientRegistry>>,
    /// CERT-M2-C02: historia con INTERIOR MUTABILITY — el trait da &self,
    /// así que record_event necesita Mutex para funcionar a través del
    /// orchestrator. Antes sin Mutex: record_event era INCALLABLE desde
    /// el trait, y el QO-M2.2 entero era código muerto.
    events: std::sync::Mutex<VecDeque<f64>>,
    /// Intensidad base μ.
    mu: f64,
    /// Auto-excitación α.
    alpha: f64,
    /// Decaimiento β (1/segundos).
    beta: f64,
    /// Último timestamp (para compute_dt entre eventos).
    last_ts: f64,
    /// #535 — eventos observados (para el seeding de μ̂).
    n_seen: u64,
}

impl Clone for HawkesBesselEngine {
    fn clone(&self) -> Self {
        Self {
            registry: self.registry.clone(),
            events: std::sync::Mutex::new(
                self.events.lock().map(|e| e.clone()).unwrap_or_default()
            ),
            mu: self.mu,
            alpha: self.alpha,
            beta: self.beta,
            last_ts: self.last_ts,
            n_seen: self.n_seen,
        }
    }
}

impl Default for HawkesBesselEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for HawkesBesselEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HawkesBesselEngine")
            .field("mu", &self.mu)
            .field("alpha", &self.alpha)
            .field("beta", &self.beta)
            .field("n_events", &self.events.lock().map(|e| e.len()).unwrap_or(0))
            .finish()
    }
}

/// Parámetros por defecto calibrados a liquidaciones de crypto:
/// cada liquidación excita la siguiente dentro de ~2 segundos.
pub const DEFAULT_MU: f64 = 0.5; // evento exógeno cada ~2s (prior inicial de μ̂)
pub const DEFAULT_ALPHA: f64 = 0.3; // cada evento suma 0.3 a λ
pub const DEFAULT_BETA: f64 = 0.5; // decae con τ = 2s
pub const MAX_EVENTS: usize = 128;

/// #535 — Ancla del estado estacionario del proceso: con ritmo constante r,
/// λ_ss = μ + α·r/β y μ̂ → r, así que λ/μ → 1 + α/β. Éste es el "ritmo
/// normal" del símbolo en unidades del propio proceso; las ráfagas lo
/// superan, las pausas quedan por debajo (piso 1.0 con ventana vacía).
pub const STEADY_STATE_RATIO: f64 = 1.0 + DEFAULT_ALPHA / DEFAULT_BETA;

/// #649 (Ola 49) — EXCESO DE EXCITACIÓN HAWKES λ/μ̂ sobre el estado
/// estacionario, saturado a (−1, 1): 0 = régimen normal (abstención),
/// →+1 cascada, →−1 calma extrema. La MONEDA de la casa (#582, #617,
/// #625, PPO): todo consumidor de λ/μ̂ expresa su convicción en esta
/// escala-libre. Vive AQUÍ (junto a STEADY_STATE_RATIO, su única
/// constante) y el core la re-exporta — fuente única.
#[inline]
pub fn excitacion_hawkes_norm(ratio: f64) -> f64 {
    const SS: f64 = STEADY_STATE_RATIO;
    if ratio.is_finite() && ratio > 0.0 {
        ((ratio - SS) / SS).tanh()
    } else {
        0.0
    }
}

/// #535 — Memoria del estimador de μ (EWMA de la tasa de llegada). τ = 60 s
/// separa la banda de adaptación (minutos) de la banda del kernel (τ = 2 s):
/// una ráfaga de ≤10 s mueve μ̂ a lo sumo ~15 %, preservando la señal de
/// excitación en lugar de absorberla en la línea base.
const MU_TAU_S: f64 = 60.0;
const MU_FLOOR: f64 = 0.05;
const MU_INST_MIN: f64 = 0.01;
const MU_INST_MAX: f64 = 200.0;
const MU_SEED_MAX: f64 = 50.0;

impl HawkesBesselEngine {
    /// #649 (Ola 49) — VOTO ESPECTRAL de la excitación Hawkes, versión
    /// kernel-honesta. El voto de la escala k es `tanh(x(τ_k)) ·
    /// excitacion_hawkes_norm(ratio)`: la DIRECCIÓN es el momentum de esa
    /// escala (suave, sin escalón de signo) y la CONVICCIÓN es el exceso
    /// REAL λ/μ̂ sobre el estado estacionario — una propiedad GLOBAL del
    /// proceso, no una función de la escala.
    ///
    /// Lo que ELIMINA esta versión (auditor A, transversal): el kernel
    /// `e^{−β·τ_s}` con τ en SEGUNDOS daba e^{−15}…e^{−21600} en la banda
    /// operable [30 s, 12 h] — una "intensidad a escala 12 h" evaluada con
    /// un decaimiento de 2 s no existe físicamente. Con el literal
    /// excitacion_base=2.5 de la composición, el voto degeneraba a ±0.9
    /// plano (un sign(x) disfrazado); la mezcla dimensional sumaba la
    /// tasa β=0.5 al ratio λ/μ̂. La per-escala vive en x(τ_k) — que ya ES
    /// la resolución espectral del flujo — y la excitación modula global.
    /// Observacional: el voto vivo queda bit a bit (T-1 cero).
    pub fn voto_espectral(
        desplazamientos: &[f64; 32],
        ratio_lambda_mu: f64,
    ) -> crate::voto_espectral::VotoEspectral {
        let excitacion = excitacion_hawkes_norm(ratio_lambda_mu);
        let mut por_escala = [0.0f64; 32];
        for k in 0..32 {
            let x = desplazamientos[k];
            if !x.is_finite() {
                continue;
            }
            por_escala[k] = (x.clamp(-10.0, 10.0).tanh() * excitacion).clamp(-1.0, 1.0);
        }
        crate::voto_espectral::VotoEspectral::desde_arr(&por_escala)
    }

    pub fn new() -> Self {
        Self {
            registry: None,
            events: std::sync::Mutex::new(VecDeque::with_capacity(MAX_EVENTS)),
            mu: DEFAULT_MU,
            alpha: DEFAULT_ALPHA,
            beta: DEFAULT_BETA,
            last_ts: 0.0,
            n_seen: 0,
        }
    }

    /// #535 — μ̂ EMPÍRICO: la base de normalización λ/μ debe ser el ritmo
    /// exógeno OBSERVADO del símbolo, no una constante. Con μ = 0.5 fijo y
    /// símbolos líquidos (10-50 trades/s excitando el proceso por cada
    /// trade), el estado estacionario de λ/μ era 1 + α·r/(β·μ) ≈ 7-25:
    /// ningún umbral genético en [1.0, 1.9] podía discriminar — el gate era
    /// tautológico precisamente en los pares que el sistema opera. Con μ̂
    /// convergido al ritmo real, λ/μ → 1 + α/β (= STEADY_STATE_RATIO) en
    /// régimen normal para TODOS los símbolos: escala-libre.
    ///
    /// Estimación: siembra con la primera tasa observada (2º evento) para
    /// converger instantáneamente en streams regulares, luego EWMA con
    /// τ = 60 s (ver MU_TAU_S).
    #[inline]
    pub fn base_rate(&self) -> f64 {
        self.mu
    }

    /// Registra un evento (timestamp en segundos desde epoch o relativo).
    /// Purga eventos más viejos que 5/β (la contribución es < e^-5 ≈ 0.7%).
    pub fn record_event(&mut self, ts: f64) {
        if !ts.is_finite() || ts < self.last_ts {
            return; // monotonicidad estricta
        }
        if self.n_seen >= 1 && ts > self.last_ts {
            let dt = ts - self.last_ts;
            if dt.is_finite() && dt > 0.0 {
                let inst = (1.0 / dt).clamp(MU_INST_MIN, MU_INST_MAX);
                if self.n_seen == 1 {
                    // Siembra: un solo salto al primer estimador — evita ~τ
                    // de warmup en el que el ratio seguiría inflado.
                    self.mu = inst.min(MU_SEED_MAX).max(MU_FLOOR);
                } else {
                    let w = (-dt / MU_TAU_S).exp();
                    self.mu = (self.mu * w + inst * (1.0 - w)).max(MU_FLOOR);
                }
            }
        }
        self.n_seen = self.n_seen.saturating_add(1);
        if let Ok(mut e) = self.events.lock() { e.push_back(ts); }
        self.last_ts = self.last_ts.max(ts);
        // Purga: eventos con contribución < e^-5 son ruido computacional
        let cutoff = ts - 5.0 / self.beta.max(0.01);
        if let Ok(mut e) = self.events.lock() {
            while let Some(&oldest) = e.front() {
                if oldest < cutoff {
                    e.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    /// λ(t) = μ + Σ α·e^(−β·(t − tᵢ)) — la intensidad REAL de Hawkes.
    /// Un evento EN t contribuye con α (dt=0 ⇒ e^0 = 1).
    #[inline]
    pub fn intensity(&self, t: f64) -> f64 {
        let mut lambda = self.mu;
        // #535 (perf): iterar BAJO el lock — el clone del deque era una
        // asignación por llamada en el hot path (una por trade).
        if let Ok(events) = self.events.lock() {
            for &ti in events.iter() {
                let dt = t - ti;
                if dt >= 0.0 {
                    lambda += self.alpha * (-self.beta * dt).exp();
                }
            }
        }
        if lambda.is_finite() {
            lambda
        } else {
            self.mu
        }
    }

    /// Ratio de ramificación n = α/β. n < 1 = estacionario.
    #[inline]
    pub fn branching_ratio(&self) -> f64 {
        if self.beta > 0.0 {
            self.alpha / self.beta
        } else {
            f64::INFINITY
        }
    }

    /// ¿Está el proceso en régimen estacionario? (n < 1)
    #[inline]
    pub fn is_stationary(&self) -> bool {
        self.branching_ratio() < 1.0
    }

    /// Intensidad NORMALIZADA contra μ̂: λ/μ > STEADY_STATE_RATIO = cascada
    /// activa sobre el ritmo normal DEL SÍMBOLO. Un valor de 3.0 significa
    /// "3× el ritmo base observado". Con μ̂ empírico (#535) la cantidad es
    /// escala-libre: 1.6 ≈ régimen normal tanto a 30 tps como a 0.3 tps.
    #[inline]
    pub fn intensity_ratio(&self, t: f64) -> f64 {
        let lambda = self.intensity(t);
        if self.mu > 1e-9 {
            (lambda / self.mu).max(0.0)
        } else {
            1.0
        }
    }

    /// API legacy: computar intensidad para un dt dado (sin historia).
    /// Mantiene compatibilidad con el código que la llamaba — pero ahora
    /// la fórmula es el KERNEL EXPONENCIAL de Hawkes, no sqrt(1+dt²)−dt.
    #[inline]
    pub fn compute_bessel_hawkes_intensity(base_lambda: f64, alpha: f64, dt: f64) -> f64 {
        let safe_dt = if dt.is_finite() && dt >= 0.0 { dt } else { 0.1 };
        let safe_lambda = if base_lambda.is_finite() { base_lambda } else { 1.0 };
        let safe_alpha = if alpha.is_finite() { alpha } else { 0.5 };
        // QO-M2.2: kernel EXPONENCIAL e^(−β·dt) con β=0.5 (τ=2s) — la
        // forma funcional del proceso de Hawkes. Antes: sqrt(1+dt²)−dt
        // (no era Bessel ni Hawkes, sólo una sigmoide arbitraria).
        let res = safe_lambda + safe_alpha * (-0.5 * safe_dt).exp();
        if res.is_finite() {
            res
        } else {
            safe_lambda
        }
    }
}

impl QuantumStrategy for HawkesBesselEngine {
    fn name(&self) -> &str {
        "HawkesBesselEngine"
    }

    fn init(&mut self, registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        self.registry = Some(registry);
        Ok(())
    }

    fn evaluate(&self) -> f64 {
        self.evaluate_for_coin(0, "")
    }

    fn evaluate_for_coin(&self, _coin_id: usize, symbol: &str) -> f64 {
        let sym_opt = if symbol.is_empty() {
            None
        } else {
            Some(symbol)
        };
        let r = match self.registry.as_ref() {
            Some(reg) => reg,
            None => return 0.0,
        };

        let direction = r
            .get_scoped_parameter(
                sym_opt,
                if symbol.is_empty() { None } else { Some(_coin_id) },
                "order_flow_direction",
                "HawkesBesselEngine",
            )
            .map(|p| p.get_value())
            .unwrap_or(0.0);

        if direction.abs() <= 1e-6 || !direction.is_finite() {
            return 0.0;
        }

        // M2-C02 — CERRADO (R9, 2026-09-19; verificado H2-8 RONDA 3,
        // 2026-10-07): el core excita el proceso real por SÍMBOLO
        // (`hawkes_by_coin`, record_event por trade) y publica λ/μ
        // VERDADERO al registry como 'hawkes_intensity'
        // (god-engine-core/src/lib.rs:~4180,
        // `hawkes_ratio_real.clamp(0.1, 10)`).
        //
        // HISTORIA CERRADA (no re-parar — era el riesgo de este bloque
        // viejo, hallazgo H2-8): antes de M2-C02 el slot 'hawkes_intensity'
        // llevaba un PROXY de aceleración 1+|a_t|/ATR (mislabel de la
        // auditoría décima). Aquel texto advertía "FIX REAL pendiente, NO
        // hecho" — PENDIENTE YA CUMPLIDO: la matemática Σα·e^(−βΔt) de
        // este engine está VIVA en producción vía el proceso por símbolo
        // del core. Los consumidores del slot
        // (flow_excitation_confluence, flow_impulse) leen intensidad real
        // desde entonces.
        let core_intensity = r
            .get_scoped_parameter(
                sym_opt,
                if symbol.is_empty() { None } else { Some(_coin_id) },
                "hawkes_intensity",
                "HawkesBesselEngine",
            )
            .map(|p| p.get_value())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(1.0);

        // #657 (F2-A1) — ERRADICACIÓN sombra/vivo: el evaluate VIVO usa la
        // MISMA moneda de la casa que el voto espectral #649 — exceso
        // λ/μ̂ sobre STEADY_STATE_RATIO (0 en régimen normal = ABSTENCIÓN),
        // firmado por el momentum CONTINUO (no signum: salto en dir=0).
        // Antes: signum·tanh(λ/μ̂) votaba ±0.92 constante en régimen
        // normal — el fallback escalar D-754 heredaba la física rota.
        // #659 (F2-A4): la CALMA se abstiene (excit ≥ 0, paridad con
        // flow_impulse #657) — antes la excitación negativa INVERTÍA el
        // sentido del momentum: calma + flujo alcista votaba bajista.
        // #666 (H2-2): divisor O(1) — direction es un flujo normalizado
        // O(1); /1e-3 saturaba a signum encubierto.
        direction.tanh() * excitacion_hawkes_norm(core_intensity).max(0.0)
    }

    fn horizon(&self) -> strategy_core::TradeHorizon {
        strategy_core::TradeHorizon::Continuous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qo_m22_kernel_exponencial_decae_monotonamente() {
        // e^(−0.5·dt) es monótono decreciente: dt=0 ⇒ α+μ; dt→∞ ⇒ μ
        let i0 = HawkesBesselEngine::compute_bessel_hawkes_intensity(1.0, 2.0, 0.0);
        assert!((i0 - 3.0).abs() < 1e-6, "dt=0: μ + α = 3.0");
        let i10 = HawkesBesselEngine::compute_bessel_hawkes_intensity(1.0, 2.0, 10.0);
        assert!(i10 > 1.0 && i10 < 3.0, "decay monótono: {i10}");
        // Con dt grande converge a μ
        let i100 = HawkesBesselEngine::compute_bessel_hawkes_intensity(1.0, 2.0, 100.0);
        assert!((i100 - 1.0).abs() < 0.01, "dt→∞ converge a μ: {i100}");
    }

    #[test]
    fn qo_m22_intensidad_con_historia_de_eventos() {
        let mut h = HawkesBesselEngine::new();
        // Sin eventos: λ = μ (prior inicial)
        assert!((h.intensity(0.0) - DEFAULT_MU).abs() < 1e-6);

        // Un evento en t=0: λ(0) = μ + α (contribución plena). El 1er evento
        // no actualiza μ̂ (necesita un dt previo).
        h.record_event(0.0);
        assert!((h.intensity(0.0) - (DEFAULT_MU + DEFAULT_ALPHA)).abs() < 1e-6);

        // 3 eventos: λ(0) = μ̂ + Σα — μ̂ ya aprendió la tasa (dt=0.1s ⇒ 10 tps)
        h.record_event(0.1);
        h.record_event(0.2);
        let i = h.intensity(0.2);
        assert!(i > DEFAULT_MU + 2.0 * DEFAULT_ALPHA, "3 eventos suman: {i}");
        // La contribución decae: en t=10 (5/β), ≈ μ̂ APRENDIDO (no DEFAULT_MU)
        let i_far = h.intensity(10.0);
        assert!(
            i_far < h.base_rate() + 0.1,
            "decay a μ̂ en 5/β: {i_far} vs μ̂={}",
            h.base_rate()
        );
    }

    #[test]
    fn qo_535_mu_empirico_es_escala_libre() {
        // La invariante teórica es ERGÓDICA: E[λ/μ̂] = 1 + α/β sobre la fase
        // uniforme de evaluación. El motor responde "λ AHORA" (el purge sigue
        // al último evento), así que el muestreo debe ser FORWARD-ONLY —
        // exactamente el patrón vivo: registrar evento y evaluar entre
        // llegadas, nunca retroactivamente.
        fn avg_ratio_forward(
            h: &mut HawkesBesselEngine,
            dt_event: f64,
            cycles: usize,
            n_sub: usize,
        ) -> f64 {
            let mut t = 0.0_f64;
            let mut sum = 0.0_f64;
            let mut n = 0.0_f64;
            for _ in 0..cycles {
                h.record_event(t);
                let sub = dt_event / (n_sub as f64);
                for k in 0..=n_sub {
                    sum += h.intensity_ratio(t + sub * (k as f64));
                    n += 1.0;
                }
                t += dt_event;
            }
            sum / n
        }

        // Régimen normal a 10 tps: E[λ/μ̂] → 1 + α/β = STEADY_STATE_RATIO
        let mut h = HawkesBesselEngine::new();
        let r_liquido = avg_ratio_forward(&mut h, 0.1, 600, 4);
        assert!(
            (r_liquido - STEADY_STATE_RATIO).abs() < 0.1,
            "10 tps estacionario ⇒ ~1.6, got {r_liquido}"
        );

        // Régimen normal a 0.5 tps (alt tranquilo): MISMO ratio estacionario
        let mut h2 = HawkesBesselEngine::new();
        let r_quieto = avg_ratio_forward(&mut h2, 2.0, 60, 4);
        assert!(
            (r_quieto - STEADY_STATE_RATIO).abs() < 0.15,
            "0.5 tps estacionario ⇒ ~1.6, got {r_quieto}"
        );
        // La diferencia absoluta de escala NO debe trasladarse al ratio:
        // sin μ̂ empírico, el motor a 10 tps habría dado λ/μ ≈ 13.
        assert!((r_liquido - r_quieto).abs() < 0.2);
    }

    #[test]
    fn qo_535_tautologia_muerta_en_simbolo_liquido() {
        // 30 tps CONSTANTES (BTCUSDT en horas pico): antes λ/μ ≈ 37 con
        // μ=0.5 fijo — el gate pasaba SIEMPRE (tautología #535). Ahora el
        // stream constante es "ritmo normal del símbolo": ratio ≈ 1.6.
        let mut h = HawkesBesselEngine::new();
        let mut t = 0.0;
        while t < 90.0 {
            h.record_event(t);
            t += 1.0 / 30.0;
        }
        let r = h.intensity_ratio(90.0);
        assert!(
            r < STEADY_STATE_RATIO + 0.15,
            "30 tps constantes NO es cascada: {r}"
        );
        // Y una ráfaga de 4× sobre ese ritmo SÍ debe superar el ancla
        let mut tb = t;
        while tb < t + 3.0 {
            h.record_event(tb);
            tb += 1.0 / 120.0;
        }
        let r_burst = h.intensity_ratio(tb);
        assert!(
            r_burst > STEADY_STATE_RATIO + 0.3,
            "ráfaga 4× ⇒ excitación genuina: {r_burst}"
        );
    }

    #[test]
    fn qo_535_siembra_de_mu_con_segundo_evento() {
        let mut h = HawkesBesselEngine::new();
        h.record_event(0.0);
        assert!((h.base_rate() - DEFAULT_MU).abs() < 1e-9, "1er evento: prior");
        h.record_event(0.1); // dt = 0.1s ⇒ 10 tps
        assert!(
            (h.base_rate() - 10.0).abs() < 1e-9,
            "siembra inmediata: μ̂=10, got {}",
            h.base_rate()
        );
        // EWMA posterior: sin deriva si la tasa se mantiene
        h.record_event(0.2);
        assert!((h.base_rate() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn qo_m22_ratio_de_ramicacion_estacionario() {
        let h = HawkesBesselEngine::new();
        // Defaults: α=0.3, β=0.5 ⇒ n = 0.6 < 1 (estacionario)
        assert!(h.is_stationary(), "defaults deben ser estacionarios");
        assert!((h.branching_ratio() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn qo_m22_purga_de_eventos_viejos() {
        let mut h = HawkesBesselEngine::new();
        h.record_event(0.0);
        h.record_event(100.0); // t=100 con β=0.5 ⇒ cutoff = 100-10 = 90
        assert!(h.events.lock().unwrap().len() <= 2, "purga mantiene ≤ cutoff");
        // El evento de t=0 fue purgado (contribución < e^-50)
        assert!(h.intensity(100.0) < DEFAULT_MU + DEFAULT_ALPHA * 1.01);
    }

    #[test]
    fn test_hawkes_nan_immunity() {
        let res = HawkesBesselEngine::compute_bessel_hawkes_intensity(f64::NAN, f64::NAN, f64::NAN);
        assert!(res.is_finite());
    }

    #[test]
    fn test_hawkes_engine_evaluate_with_registry() {
        // #657 (F2-A1): la física viva = #649 — ABSTENCIÓN en régimen
        // normal (ratio SS ⇒ voto 0), cascada firma con el momentum.
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("order_flow_direction", 1.0);
        registry.set("hawkes_intensity", STEADY_STATE_RATIO);
        let mut engine = HawkesBesselEngine::new();
        assert!(engine.init(Arc::clone(&registry)).is_ok());
        let en_ss = engine.evaluate();
        assert!(
            en_ss.abs() < 1e-9,
            "régimen normal (λ/μ̂=SS) se abstiene, votó {en_ss}"
        );

        registry.set("hawkes_intensity", 4.0);
        let mut engine2 = HawkesBesselEngine::new();
        assert!(engine2.init(registry).is_ok());
        let cascada = engine2.evaluate();
        assert!(cascada > 0.3, "cascada con momentum + vota alto, votó {cascada}");
        assert!(cascada <= 1.0);
    }
}

#[cfg(test)]
mod qo_617_tests {
    use super::*;
    use crate::voto_espectral::ESCALAS_VOTO;

    /// #649 — el voto es la dirección suave de la escala por el exceso
    /// REAL de excitación: régimen normal ⇒ abstención TOTAL (el defecto
    /// viejo: con base literal 2.5 votaba ±0.9 plano en TODA escala).
    #[test]
    fn qo_649_régimen_normal_abstiene_y_cascada_convence() {
        let mut x = [0.0; ESCALAS_VOTO];
        for (k, v) in x.iter_mut().enumerate() {
            *v = if k % 2 == 0 { 0.5 } else { -0.5 };
        }
        // Estado estacionario: exceso 0 ⇒ voto 0 en TODAS las escalas.
        let frio = HawkesBesselEngine::voto_espectral(&x, STEADY_STATE_RATIO);
        for k in 0..ESCALAS_VOTO {
            assert!(
                frio.en_escala(k).abs() < 1e-12,
                "SS ⇒ abstención en {k}: {}",
                frio.en_escala(k)
            );
        }
        // Cascada 3×: convicción = tanh((3−1.6)/1.6) ≈ 0.70 en TODAS las
        // escalas — la excitación es GLOBAL, no función de la escala (el
        // kernel viejo la mataba en [30 s, 12 h]).
        let cascada = HawkesBesselEngine::voto_espectral(&x, 3.0);
        let esperado = ((3.0 - STEADY_STATE_RATIO) / STEADY_STATE_RATIO).tanh() * 0.5f64.tanh();
        for k in 0..ESCALAS_VOTO {
            assert!(
                (cascada.en_escala(k) - esperado * if k % 2 == 0 { 1.0 } else { -1.0 }).abs()
                    < 1e-12,
                "escala {k}: {}",
                cascada.en_escala(k)
            );
        }
        // Antisimetría estricta.
        let mut x_neg = x;
        for v in &mut x_neg {
            *v = -*v;
        }
        let neg = HawkesBesselEngine::voto_espectral(&x_neg, 3.0);
        for k in 0..ESCALAS_VOTO {
            assert!((cascada.en_escala(k) + neg.en_escala(k)).abs() < 1e-12);
        }
        // CONTINUIDAD en x=0 (el viejo saltaba ±convicción por sign()).
        let mut x_grad = [0.0; ESCALAS_VOTO];
        x_grad[7] = 1e-9;
        let suave = HawkesBesselEngine::voto_espectral(&x_grad, 3.0);
        assert!(suave.en_escala(7).abs() < 1e-9);
        // Ratio no finito o ≤0 ⇒ abstención (sin inventar).
        for malo in [f64::NAN, 0.0, -3.0] {
            let v = HawkesBesselEngine::voto_espectral(&x, malo);
            assert_eq!(v.en_escala(4), 0.0);
        }
    }

    /// #649 — la moneda de la casa vive junto a su constante: exceso
    /// saturado, SS⇒0, calma negativa, guardias.
    #[test]
    fn qo_649_moneda_excitacion_hawkes_norm() {
        assert_eq!(excitacion_hawkes_norm(STEADY_STATE_RATIO), 0.0);
        assert!(excitacion_hawkes_norm(3.0) > 0.5);
        assert!(excitacion_hawkes_norm(0.1) < -0.5);
        assert_eq!(excitacion_hawkes_norm(f64::NAN), 0.0);
        assert_eq!(excitacion_hawkes_norm(0.0), 0.0);
    }
}
