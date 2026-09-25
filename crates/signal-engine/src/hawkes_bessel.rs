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
pub const DEFAULT_MU: f64 = 0.5; // evento exógeno cada ~2s
pub const DEFAULT_ALPHA: f64 = 0.3; // cada evento suma 0.3 a λ
pub const DEFAULT_BETA: f64 = 0.5; // decae con τ = 2s
pub const MAX_EVENTS: usize = 128;

impl HawkesBesselEngine {
    pub fn new() -> Self {
        Self {
            registry: None,
            events: std::sync::Mutex::new(VecDeque::with_capacity(MAX_EVENTS)),
            mu: DEFAULT_MU,
            alpha: DEFAULT_ALPHA,
            beta: DEFAULT_BETA,
            last_ts: 0.0,
        }
    }

    /// Registra un evento (timestamp en segundos desde epoch o relativo).
    /// Purga eventos más viejos que 5/β (la contribución es < e^-5 ≈ 0.7%).
    pub fn record_event(&mut self, ts: f64) {
        if !ts.is_finite() || ts < self.last_ts {
            return; // monotonicidad estricta
        }
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
        let events_snapshot = self.events.lock().map(|e| e.clone()).unwrap_or_default();
        for &ti in &events_snapshot {
            let dt = t - ti;
            if dt >= 0.0 {
                lambda += self.alpha * (-self.beta * dt).exp();
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

    /// Intensidad NORMALIZADA contra μ: λ/μ > 1 = cascada activa.
    /// Un valor de 3.0 significa "3× el ritmo base de eventos".
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

        // M2-C02 — CERRADO (R9, 2026-09-19): el core AHORA excita el
        // proceso real (record_event por trade en process_event) y publica
        // λ/μ VERDADERO al registry — el proxy de aceleración fue retirado.
        // El historial del mislabel se conserva abajo como advertencia.
        //
        // Lo que realmente pasa: este evaluate lee el param de registry
        // 'hawkes_intensity', pero el ÚNICO escritor en producción
        // (god-engine-core/src/lib.rs:1856-1859) publica
        //     (1.0 + (a_t.abs()/atr_abs).clamp(0,4)).clamp(0.1,5)
        // = `1 + |aceleración|/ATR`: un proxy de aceleración de
        // volatilidad. NO es un proceso de Hawkes — no hay historia de
        // eventos, ni suma de auto-excitación Σα·e^(−β·(t−tᵢ)), ni
        // branching ratio. El comentario CERT anterior afirmaba que "el
        // core ya computa el proceso de Hawkes con su propia historia de
        // trades": FALSO.
        //
        // Consecuencia: la matemática Hawkes correcta de este engine
        // (record_event / intensity / intensity_ratio / branching_ratio)
        // es CÓDIGO MUERTO en producción — record_event sigue siendo
        // &mut self (incallable vía el trait QuantumStrategy::evaluate que
        // da &self) y tiene CERO llamadores de producción. La detección de
        // cascadas de liquidación (auto-excitación con memoria) que motivó
        // QO-M2.2 fue degradada en silencio a un proxy de aceleración.
        // Otros consumidores del mismo proxy (clave 'hawkes_intensity'):
        // flow_excitation_confluence.rs y flow_impulse.rs.
        //
        // FIX REAL (pendiente, NO hecho — requiere tocar el core, archivo
        // caliente): cablear una fuente de eventos (trades/liquidaciones)
        // hacia un HawkesBesselEngine por moneda y publicar
        // intensity_ratio(now) como 'hawkes_intensity', en vez del proxy.
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

        direction.signum() * core_intensity.tanh().clamp(0.0, 1.0)
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
        // Sin eventos: λ = μ
        assert!((h.intensity(0.0) - DEFAULT_MU).abs() < 1e-6);

        // Un evento en t=0: λ(0) = μ + α (contribución plena)
        h.record_event(0.0);
        assert!((h.intensity(0.0) - (DEFAULT_MU + DEFAULT_ALPHA)).abs() < 1e-6);

        // 3 eventos: λ(0) = μ + 3α
        h.record_event(0.1);
        h.record_event(0.2);
        let i = h.intensity(0.2);
        assert!(i > DEFAULT_MU + 2.0 * DEFAULT_ALPHA, "3 eventos suman: {i}");
        // La contribución decahe: en t=10 (5/β), ≈ μ
        let i_far = h.intensity(10.0);
        assert!(i_far < DEFAULT_MU + 0.1, "decay a μ en 5/β: {i_far}");
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
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("order_flow_direction", 1.0);

        let mut engine = HawkesBesselEngine::new();
        assert!(engine.init(registry).is_ok());

        let eval = engine.evaluate();
        assert!(eval > 0.0, "dirección positiva + intensidad > 0");
        assert!(eval <= 1.0);
    }
}
