//! Fixtures compartidos de las pruebas de la base futuro-spot (CL-97):
//! núcleo con BTCUSDT, eventos del futuro con el spot llegando entre ellos,
//! y un mercado con la base en OU o en caminata.
#![allow(dead_code)]

use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

static TEST_LOCK: Mutex<()> = Mutex::new(());

/// Serializa las pruebas que tocan el universo y el registro de símbolos
/// (globales). Tolera el envenenamiento: una prueba que falla no arrastra a
/// las demás.
pub fn bloqueo() -> MutexGuard<'static, ()> {
    TEST_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

pub struct Lcg(u64);
impl Lcg {
    pub fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    pub fn n(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

pub fn nucleo() -> (Arc<GlobalArena>, GodEngineCore) {
    symbols::update_dynamic_universe(vec!["BTCUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec("BTCUSDT")]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
    (arena, core)
}

/// Un evento del futuro en `ts`; si `spot`, el spot llega justo antes.
pub fn evento(arena: &GlobalArena, core: &mut GodEngineCore, ts: u64, futuro: f64, spot: Option<f64>) {
    if let Some(s) = spot {
        arena.update_spot_data(0, s - 0.1, s + 0.1, 1.0, 1.0);
    }
    let _ = core.process_tick_dual(0, futuro - 0.5, futuro + 0.5, 1.0, 1.0, ts, &[0.0; 54], true, true);
}

pub fn leer(arena: &GlobalArena, k: &str) -> Option<f64> {
    arena.registry.get_scoped_strict("BTCUSDT", k)
}

/// La guarda del lector (`StatArbEngine::evaluate_for_coin`): z publicada
/// finita y no nula, y t½ publicada ≤ 2τ*.
pub fn guarda_pasa(arena: &GlobalArena) -> bool {
    match (
        leer(arena, "statarb_ou_zscore"),
        leer(arena, "statarb_half_life_ms"),
        leer(arena, "dominant_tau_ms"),
    ) {
        (Some(z), Some(th), Some(tau)) => z.is_finite() && z != 0.0 && th <= 2.0 * tau,
        _ => false,
    }
}

/// Generador de la base y el futuro: futuro en caminata log-normal y base
/// OU con vida media `t_media_s` (∞: caminata independiente del spot).
pub struct Mercado {
    g: Lcg,
    lf: f64,
    base: f64,
    rho: f64,
    innov: f64,
}

impl Mercado {
    pub fn new(semilla: u64, t_media_s: f64) -> Self {
        let (rho, innov) = if t_media_s.is_finite() {
            let rho = (-(std::f64::consts::LN_2) / t_media_s).exp();
            // Desviación estacionaria de la base: 1 pb.
            (rho, 1e-4 * (1.0 - rho * rho).sqrt())
        } else {
            // Caminata: la base suma la diferencia de dos caminatas de 2 pb/s.
            (1.0, 2e-4 * std::f64::consts::SQRT_2)
        };
        Self {
            g: Lcg(0x9E37_79B9_7F4A_7C15 ^ semilla),
            lf: 50_000f64.ln(),
            base: 0.0,
            rho,
            innov,
        }
    }
    /// Avanza un segundo y devuelve (futuro, spot). El spot llega ENTRE dos
    /// eventos del futuro: refleja el futuro vigente (el anterior) más la
    /// base nueva. El futuro salta en su evento, después.
    pub fn paso(&mut self) -> (f64, f64) {
        self.base = self.rho * self.base + self.innov * self.g.n();
        let spot = (self.lf - self.base).exp();
        self.lf += 2e-4 * self.g.n();
        (self.lf.exp(), spot)
    }
}

/// Núcleo con una base OU rápida (t½ = 14 s ≈ τ_18/5) hasta que la guarda
/// pasa, con al menos 30 min de pares (con menos, el estadístico apenas
/// cruza el crítico). Devuelve el mercado y la marca siguiente. Falla si en
/// 60 min no pasa.
pub fn nucleo_demostrado(semilla: u64) -> (Arc<GlobalArena>, GodEngineCore, Mercado, u64) {
    let (arena, mut core) = nucleo();
    let mut m = Mercado::new(semilla, 14.0);
    let mut ts = 1_000u64;
    for i in 0..3_600u64 {
        let (f, sp) = m.paso();
        evento(&arena, &mut core, ts, f, Some(sp));
        ts += 1_000;
        if i >= 1_800 && guarda_pasa(&arena) {
            return (arena, core, m, ts);
        }
    }
    panic!("precondición: la base OU rápida demuestra su reversión en 60 min");
}

