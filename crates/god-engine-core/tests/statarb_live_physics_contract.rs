//! CONTRATO FORMAL — OLA 73 (R6-A3/B1/B2/A5): StatArb con física VIVA y
//! PARIDAD lector/escritor en el core.
//!
//! Erradica R6-A3/B1 (la SDE del orquestador jamás corría y el voto leía
//! `vecm_zscore` = basis/ATR sin cointegración) y R6-B2 (θ congelada en el
//! default ⇒ guarda t½ ≤ 2τ* decorativa), SIN romper la señal viva:
//!
//! 1. PARIDAD LECTOR/ESCRITOR (R6-A3): el core SÓLO publica
//!    `statarb_ou_zscore` cuando la física OU EXISTE para esa moneda (SDE
//!    madura: ≥ 10 pares spot-futuro causalmente crecientes). Con la clave
//!    AUSENTE, el lector conserva su fallback documentado (`vecm_zscore`)
//!    bit a bit — publicar 0.0 incondicional desde el arranque lo sombrearía
//!    para siempre y silenciaría una señal viva sin evidencia.
//! 2. Con spot y reloj físico, `statarb_ou_engines` calibra θ EN PRODUCCIÓN:
//!    `statarb_half_life_ms` abandona el valor congelado del default
//!    (ln 2 / 0.1 = 6 931,47 ms) — la guarda t½ ≤ 2τ* del lector deja de
//!    ser decorativa (R6-B2).
//! 3. Al madurar la SDE se publica z finito y la clave queda PRESENTE.
//! 4. TTL: si el feed de spot desaparece > 30 s, el core re-publica 0.0
//!    aunque la SDE esté madura (abstención honesta anti-staleness, R6-C4).

use god_engine_core::GodEngineCore;
use quantum_arena::{GlobalArena, symbol_registry, symbols};
use std::sync::Mutex;
use strategy_core::stat_arb::StatArbEngine;
use strategy_core::QuantumStrategy;

static TEST_LOCK: Mutex<()> = Mutex::new(());

/// t½ del default θ = 0.1 s⁻¹ (ln 2 / 0.1 · 1000 ms) — el valor CONGELADO
/// que R6-B2 denuncia: la SDE jamás calibró θ en producción.
const THETA_CONGELADA_HALF_LIFE_MS: f64 = 6_931.471_805_599_453;

fn setup_btcusdt() -> (std::sync::Arc<GlobalArena>, GodEngineCore) {
    symbols::update_dynamic_universe(vec!["BTCUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(
        "BTCUSDT",
    )]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
    (arena, core)
}

fn tick(core: &mut GodEngineCore, bid: f64, ask: f64, ts_ms: u64) {
    core.process_tick_dual(0, bid, ask, 1.0, 1.0, ts_ms, &[0.0; 54], true, true);
}

/// El mismo lector que el orquestador registra (instancia pura, sin SDE).
fn lector(arena: &GlobalArena) -> StatArbEngine {
    let mut engine = StatArbEngine::new(30, 1.5);
    assert!(engine.init(arena.registry.clone()).is_ok());
    engine
}

#[test]
fn ola73_sin_spot_no_publica_fisica_propia_y_el_lector_conserva_el_fallback() {
    let _guard = TEST_LOCK.lock().unwrap();
    let (arena, mut core) = setup_btcusdt();

    // Sin spot (arena recién construido): la SDE no observa ningún par.
    tick(&mut core, 50_000.0, 50_001.0, 1_000);

    // R6-A3: la clave está AUSENTE, no publicada en 0.0. 0.0 en el arranque
    // sería una abstención SIN evidencia que sombrea el fallback para siempre.
    assert!(
        arena.registry.get_value_fast("statarb_ou_zscore").is_none(),
        "sin física OU madura el core no debe publicar statarb_ou_zscore"
    );
    assert!(arena.registry.get_value_fast("statarb_half_life_ms").is_none());
    assert!(arena.registry.get_value_fast("statarb_beta").is_none());

    // PARIDAD: con la clave ausente el lector usa el fallback legacy
    // (`vecm_zscore` = basis/ATR o desviación al EMA lento) bit a bit como el
    // árbol previo — la señal VIVA no se silencia.
    arena.registry.set("vecm_zscore", 2.5);
    let esperado = -((2.5_f64 - 1.5) / 1.5).tanh();
    let voto = lector(&arena).evaluate();
    assert!(
        (voto - esperado).abs() < 1e-12,
        "fallback legacy vivo con vecm_zscore=2.5: esperado {esperado}, obtenido {voto}"
    );

    // Y la razón exacta de la publicación CONDICIONAL: si el core publicara
    // 0.0 incondicionalmente, ese mismo voto legítimo desaparecería.
    arena.registry.set("statarb_ou_zscore", 0.0);
    assert_eq!(
        lector(&arena).evaluate(),
        0.0,
        "0.0 publicado (abstención) debe tener autoridad sobre el fallback"
    );
}

#[test]
fn ola73_con_spot_la_theta_se_calibra_viva_y_publica_z_finita() {
    let _guard = TEST_LOCK.lock().unwrap();
    let (arena, mut core) = setup_btcusdt();

    // Basis futuro-spot oscilante con reloj físico de 1 s entre ticks:
    // θ del default (congelada) debe moverse — la SDE observa ≥ 12 pares.
    let spot_bid = 50_000.0;
    let spot_ask = 50_000.2;
    arena.update_spot_data(0, spot_bid, spot_ask, 1.0, 1.0);

    let mut ts = 1_000_u64;
    for i in 0..12 {
        // futuros oscilan ±25 pb alrededor del centro de la basis.
        let off = if i % 2 == 0 { 1.0025 } else { 0.9975 };
        let fut = 50_000.0 * off;
        tick(&mut core, fut - 0.5, fut + 0.5, ts);
        ts += 1_000;
    }

    // SDE madura ⇒ la física propia SÍ se publica (clave presente).
    assert!(arena.registry.get_value_fast("statarb_ou_zscore").is_some());

    let half_ms = arena.registry.get_value_or("statarb_half_life_ms", -1.0);
    assert!(half_ms.is_finite() && half_ms > 0.0, "t½ publicada finita y positiva: {half_ms}");
    assert!(
        (half_ms - THETA_CONGELADA_HALF_LIFE_MS).abs() > 1.0,
        "R6-B2: θ sigue congelada en el default ({half_ms} ms ≈ {:.3} ms) — \
         el reloj físico no está calibrando la SDE",
        THETA_CONGELADA_HALF_LIFE_MS
    );

    // β adaptativa on en el escritor (R6-A4): se publica el estado real de la
    // RLS, no un literal decorativo.
    let beta = arena.registry.get_value_or("statarb_beta", -1.0);
    assert!(beta.is_finite() && beta > 0.0, "statarb_beta finita y positiva: {beta}");

    let z = arena.registry.get_value_or("statarb_ou_zscore", -1.0);
    assert!(z.is_finite(), "z OU madura debe ser finita, obtenido: {z}");
    // SDE madura con spot fresco: no es la abstención 0.0 del camino stale.
    // (Un z exactamente 0.0 sólo ocurriría si last_value = μ: la oscilación
    // alternativa con θ calibrada asimétrica el estado final.)
    assert_ne!(z, 0.0, "con SDE madura y feed fresco la física propia debe votar");
}

#[test]
fn ola73_spot_stale_mas_alla_del_ttl_re_publica_cero() {
    let _guard = TEST_LOCK.lock().unwrap();
    let (arena, mut core) = setup_btcusdt();

    let spot_bid = 50_000.0;
    let spot_ask = 50_000.2;
    arena.update_spot_data(0, spot_bid, spot_ask, 1.0, 1.0);

    let mut ts = 1_000_u64;
    for i in 0..12 {
        let off = if i % 2 == 0 { 1.0025 } else { 0.9975 };
        let fut = 50_000.0 * off;
        tick(&mut core, fut - 0.5, fut + 0.5, ts);
        ts += 1_000;
    }
    let z_madura = arena.registry.get_value_or("statarb_ou_zscore", -1.0);
    assert!(z_madura.is_finite() && z_madura != 0.0, "precondición SDE madura: {z_madura}");

    // El feed de spot DESAPARECE (spot = 0): el core ya no actualiza la SDE,
    // pero los ticks de futuros siguen. Tras el TTL (30 s) la z publicada
    // debe volver a la abstención honesta 0.0 aunque la SDE esté madura. Con
    // la física ya existente, ese 0.0 SÍ es honesto y el lector debe callar
    // (no reabrir el fallback: la autoridad de la clave publicada es el core).
    arena.update_spot_data(0, 0.0, 0.0, 0.0, 0.0);
    ts += 31_000;
    tick(&mut core, 50_100.0, 50_101.0, ts);

    let z_stale = arena.registry.get_value_or("statarb_ou_zscore", -1.0);
    assert_eq!(
        z_stale, 0.0,
        "spot stale > TTL (30 s): el core debe re-publicar abstención 0.0"
    );
    assert_eq!(lector(&arena).evaluate(), 0.0, "abstención honesta leída por el orquestador");
}
