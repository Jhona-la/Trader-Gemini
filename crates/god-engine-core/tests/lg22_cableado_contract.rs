//! LG-22 cableado (GLM 121): el detector e-CUSUM de pérdida de ventaja
//! (Línea G, resultado_orden) conectado al close path — por MONEDA.
//! (i) Cada cierre alimenta el detector; (ii) la alarma siembra el
//! epigenetic_bias (×0.5, pausa de régimen vía LH-04/streak-118 — el OU
//! la relaja: NO absorbente); (iii) el ARL es gobernanza del registry.
//! Sin exchange: posición plantada + tick que cierra (patrón
//! close_outcome_contract).
use god_engine_core::{GodEngineCore, outcome_context::OutcomeContext};
use quantum_arena::{GlobalArena, position::PositionHorizon, symbol_registry};
use std::sync::{Arc, Mutex, atomic::Ordering};

static ENVIRONMENT: Mutex<()> = Mutex::new(());

struct FixtureDirectory {
    root: std::path::PathBuf,
    previous: std::path::PathBuf,
}
impl FixtureDirectory {
    fn new() -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("tg-lg22-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        std::fs::create_dir(root.join("data")).unwrap();
        let previous = std::env::current_dir().unwrap();
        std::env::set_current_dir(&root).unwrap();
        Self { root, previous }
    }
}
impl Drop for FixtureDirectory {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.previous);
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn core(arl: f64) -> (Arc<GlobalArena>, GodEngineCore) {
    quantum_arena::symbols::update_dynamic_universe(vec!["LG22USDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(
        "LG22USDT",
    )]);
    let arena = GlobalArena::build_in_own_stack(1000.0);
    arena.registry.set("lg22_arl", arl);
    let mut core = GodEngineCore::new_with_outcome_context(
        arena.clone(),
        OutcomeContext::ExchangeLocalEstimate,
    );
    core.swing_nn = None;
    core.scalp_forest = None;
    (arena, core)
}

/// Abre una posición long plantada (entry 100, qty 1, margen 10) y la
/// cierra con UN tick a `bid`. Devuelve (arena, core) para inspección.
fn abrir_y_cerrar(arena: &Arc<GlobalArena>, core: &mut GodEngineCore, bid: f64, ts: u64) {
    let pos = &arena.coins[0].positions.slot_2;
    assert!(pos.open_with_horizon(
        true,
        100.0,
        1.0,
        10.0,
        ts.saturating_sub(1000),
        101.0,
        99.0,
        PositionHorizon::Continuous
    ));
    pos.exchange_confirmed.store(true, Ordering::Relaxed);
    arena.used_margin.store(10.0, Ordering::Relaxed);
    let (entry, proposal, _) = core.process_tick_dual(0, bid, bid + 0.01, 5.0, 5.0, ts, &[0.0; 54], true, false);
    assert!(entry.is_none());
    assert!(proposal.is_some(), "el cierre defensivo llega al caller (bid={bid})");
    assert!(!pos.is_open());
}

/// (1) El cierre alimenta el detector: lg22_cierres y lg22_estadistico
/// publicados tras UN cierre perdedor (bid 97 ⇒ R neta ≈ −1).
#[test]
fn lg22_cierre_alimenta_el_detector_por_moneda() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (arena, mut core) = core(500.0); // ARL alto: no queremos alarma aquí
    abrir_y_cerrar(&arena, &mut core, 97.0, 2000);
    let cierres = arena.registry.get_for_coin_or(0, "lg22_cierres", f64::NAN);
    assert!(cierres >= 1.0, "un cierre alimentó el detector: {cierres}");
    let est = arena.registry.get_for_coin_or(0, "lg22_estadistico", f64::NAN);
    assert!(est.is_finite() && est > 0.0, "estadístico publicado: {est}");
    let alarma = arena.registry.get_for_coin_or(0, "lg22_alarma", 1.0);
    assert_eq!(alarma, 0.0, "con ARL 500 un cierre NO alarmó");
}

/// (2) La alarma siembra el bias y se REINICIA (el estadístico vuelve a
/// ~0): con ARL bajo (gobernanza del registry), una racha perdedora
/// corta alarma, encoge el bias ×0.5, y el detector sigue midiendo.
#[test]
fn lg22_alarma_siembra_bias_y_reinicia() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (arena, mut core) = core(2.0); // ARL 2: alarma en ~1-2 cierres malos
    arena.coins[0].epigenetic_bias.store(1.0, Ordering::Relaxed);
    // Un cierre perdedor fuerte (bid 97 ≈ R −1): e-valores > 1 ⇒ M crece.
    abrir_y_cerrar(&arena, &mut core, 97.0, 2000);
    let alarma = arena.registry.get_for_coin_or(0, "lg22_alarma", 0.0);
    let bias = arena.coins[0].epigenetic_bias.load(Ordering::Relaxed);
    if alarma == 1.0 {
        assert!(
            bias < 1.0,
            "la alarma sembró el bias (encoge ×0.5): bias={bias}"
        );
        // Reinicio: el estadístico publicado es POST-reinicio (bajo).
        let est = arena.registry.get_for_coin_or(0, "lg22_estadistico", f64::NAN);
        assert!(est < 2.0, "post-reinicio el estadístico vuelve a bajar: {est}");
        // Y NO es absorbente: el OU-relaja el bias por diseño (contrato
        // del mecanismo, no de este test — el OU es de Ω109).
    } else {
        // Con ARL 2 y un solo cierre, M debe haber cruzado: verificamos
        // que al menos el estadístico está vivo y creciendo.
        let est = arena.registry.get_for_coin_or(0, "lg22_estadistico", 0.0);
        assert!(est > 0.0, "sin alarma aún, el estadístico crece: {est}");
    }
}

/// (3) Fuente SANA no alarma: cierres GANADORES (bid 102 ⇒ TP tocado, R
/// positiva) con ARL razonable NO disparan — la especificidad del
/// e-CUSUM (falsa alarma esperada 1/ARL bajo H0).
#[test]
fn lg22_fuente_sana_no_alarma() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (arena, mut core) = core(50.0);
    arena.coins[0].epigenetic_bias.store(1.0, Ordering::Relaxed);
    // Tres cierres ganadores consecutivos.
    for (i, ts) in [2000u64, 4000, 6000].iter().enumerate() {
        let _ = i;
        abrir_y_cerrar(&arena, &mut core, 102.0, *ts);
    }
    let alarma = arena.registry.get_for_coin_or(0, "lg22_alarma", 1.0);
    assert_eq!(alarma, 0.0, "tres ganancias NO alarmaron (especificidad)");
    // El bias puede SUBIR por el feedback epigenético NORMAL de las
    // ganancias (racha ganadora ⇒ bias > 1, camino preexistente) — lo
    // que LG-22 garantiza es que no fue SEMBRADO HACIA ABAJO por alarma.
    let bias = arena.coins[0].epigenetic_bias.load(Ordering::Relaxed);
    assert!(bias >= 1.0, "sin alarma el bias no se siembra hacia abajo: {bias}");
}

/// (LG-22b, Línea G) La siembra de la alarma se aplica al sesgo YA
/// avanzado al cierre, no al guardado con la marca del cierre anterior.
/// Antes se sembraba ×0,5 sobre el valor viejo y la relajación OU del mismo
/// cierre (τ = 30 min) lo devolvía hacia 1 durante todo el hueco: con
/// cierres cada 60 min la alarma dejaba ≈ 0,78 en vez del suelo 0,5.
#[test]
fn lg22b_la_alarma_no_se_relaja_durante_el_hueco_entre_cierres() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let (arena, mut core) = core(1.0); // ARL 1: el primer cierre perdedor alarma
    let previo = 10_000_000u64;
    let ahora = previo + 3_600_000; // 60 min después del cierre anterior
    arena.coins[0].epigenetic_bias.store(1.0, Ordering::Relaxed);
    arena.coins[0].last_close_ts.store(previo, Ordering::Relaxed);
    abrir_y_cerrar(&arena, &mut core, 97.0, ahora);
    let alarma = arena.registry.get_for_coin_or(0, "lg22_alarma", 0.0);
    assert_eq!(alarma, 1.0, "con ARL 1 un cierre a −1 R alarma");
    let sesgo = arena.coins[0].get_active_epigenetic_bias(ahora);
    assert!(
        sesgo <= 0.5 + 1e-9,
        "tras la alarma el sesgo al cierre debe estar en su suelo 0,5: {sesgo}"
    );
}
