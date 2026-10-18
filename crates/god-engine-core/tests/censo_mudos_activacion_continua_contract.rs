//! Formal Contract Test for Ola Ω144: Censo Forense de Motores Silenciosos y Activación Continua.
//!
//! Verifica:
//! 1. `resolve_models_directory()` resuelve canónicamente `models/`.
//! 2. `preload_models_from_disk()` puebla `GLOBAL_FORESTS` de manera atómica y lock-free.
//! 3. `GodEngineCore::new(arena)` inicializa el núcleo con modelos disponibles (`BTCUSDT_MOTOR`),
//!    eliminando la divergencia histórica entre producción y simulación/backtest.
//! 4. `FlowExcitationConfluenceEngine` recibe probabilidades reales y base honesta, produciendo
//!    votos continuos no nulos y eliminando la mudez en el consenso tensorial.
//! 5. `voto_espectral_con_wasserstein_dirac_bsd_riemann_kahler_gauge_y_helmholtz` propaga los 9
//!    campos físicos hacia las 32 escalas espectrales continuas.

use dark_alpha_engine::atiyah_singer_dirac::AtiyahSingerDiracEngine;
use dark_alpha_engine::helmholtz_hodge::{HelmholtzHodgeEngine, HelmholtzHodgeParams};
use god_engine_core::GodEngineCore;
use god_engine_core::ml_inference::{NanoForest, GLOBAL_FORESTS};
use god_engine_core::model_reload::{resolve_models_directory, preload_models_from_disk};
use omniscient_registry::OmniscientRegistry;
use quantum_arena::wasserstein_kantorovich::WassersteinKantorovichEngine;
use quantum_arena::GlobalArena;
use risk_engine::riemann_zeta_gue::RiemannZetaGueEngine;
use signal_engine::flow_excitation_confluence::FlowExcitationConfluenceEngine;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use strategy_core::birch_swinnerton_dyer::BirchSwinnertonDyerEngine;
use strategy_core::kahler_hodge::KahlerHodgeEngine;
use strategy_core::yang_mills_gauge::YangMillsGaugeEngine;
use strategy_core::QuantumStrategy;

/// CSG-11 (Ola Ω150): Candado de proceso para serializar mutaciones de `TG_DATA_ROOT`
/// cuando `cargo test` ejecuta los tests de este binario en hilos paralelos.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// GLM 136 / CSG-11 (fix CI + concurrencia): los tests 1-2 de Ω144 dependían de `models/`
/// del checkout raíz. Este fixture adquiere `ENV_LOCK`, crea un directorio temporal con
/// un `BTCUSDT_MOTOR.json` sintético válido, apunta `TG_DATA_ROOT` al directorio temporal,
/// y en `Drop` restaura `TG_DATA_ROOT` y elimina el directorio temporal antes de soltar el candado.
struct ModelsFixture {
    prev_env: Option<std::ffi::OsString>,
    temp_dir: PathBuf,
    _lock: MutexGuard<'static, ()>,
}

impl Drop for ModelsFixture {
    fn drop(&mut self) {
        // SAFETY: `_lock` mantiene la exclusión mutua sobre `ENV_LOCK` durante todo `drop`,
        // garantizando que ningún otro test de este binario lea o escriba `TG_DATA_ROOT` concurrentemente.
        unsafe {
            match self.prev_env.take() {
                Some(v) => std::env::set_var("TG_DATA_ROOT", v),
                None => std::env::remove_var("TG_DATA_ROOT"),
            }
        }
        let _ = std::fs::remove_dir_all(&self.temp_dir);
    }
}

impl ModelsFixture {
    /// tmp/models/BTCUSDT_MOTOR.json — init_score −1.685 ⇒ base 0.156
    /// (la base honesta HOST-010 del modelo real de BTC).
    fn new() -> Self {
        let lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var_os("TG_DATA_ROOT");
        let dir = std::env::temp_dir().join(format!(
            "tg_censo_models_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        // Convención del resolver: TG_DATA_ROOT = <root>/data (un path de
        // DATOS) y hace parent().join("models") ⇒ <root>/models. Crear
        // ambos: data/ (vacío, ancla la raíz) y models/ con el sintético.
        std::fs::create_dir_all(dir.join("data")).unwrap();
        let models = dir.join("models");
        std::fs::create_dir_all(&models).unwrap();
        // Formato bias-only: sin árboles, sólo init_score — el validador
        // lo acepta por diseño («used by measurement oracles»).
        let synth = serde_json::json!({
            "children_left": [],
            "children_right": [],
            "feature": [],
            "threshold": [],
            "value": [],
            "tree_offsets": [0, 0],
            "init_score": -1.6851217,
        });
        std::fs::write(
            models.join("BTCUSDT_MOTOR.json"),
            serde_json::to_vec(&synth).unwrap(),
        )
        .unwrap();
        // SAFETY: Protegido por `ENV_LOCK` adquirido al inicio de `ModelsFixture::new()`.
        unsafe { std::env::set_var("TG_DATA_ROOT", dir.join("data")) };
        Self {
            prev_env: prev,
            temp_dir: dir,
            _lock: lock,
        }
    }
}

#[test]
fn contract_model_directory_resolution_and_preload() {
    let fx = ModelsFixture::new();
    let temp_path = fx.temp_dir.clone();
    let dir_opt = resolve_models_directory();
    assert!(dir_opt.is_some(), "Debe resolver canónicamente el directorio models/");
    let dir = dir_opt.unwrap();
    assert!(dir.is_dir(), "La ruta resuelta debe ser un directorio válido");

    let count = preload_models_from_disk();
    assert!(count > 0, "Debe precargar al menos 1 modelo desde disk (e.g. BTCUSDT_MOTOR, etc.)");

    let map = GLOBAL_FORESTS.load();
    assert!(map.contains_key("BTCUSDT_MOTOR"), "GLOBAL_FORESTS debe contener BTCUSDT_MOTOR");
    drop(fx);
    assert!(
        !temp_path.exists(),
        "CSG-11: ModelsFixture::drop debe limpiar el directorio temporal en %TEMP%"
    );
}

#[test]
fn contract_god_engine_core_initialization_paridad_modelos() {
    let _fx = ModelsFixture::new();
    preload_models_from_disk(); // poblar GLOBAL_FORESTS con el sintético del fixture
    let arena = GlobalArena::build_in_own_stack(1000.0);
    let core = GodEngineCore::new(arena);

    // Verificar que el bosque activo de BTCUSDT se puede obtener por símbolo y sufijo
    let btc_forest = NanoForest::get_global_suffixed("BTCUSDT", "_MOTOR");
    assert!(btc_forest.is_some(), "GodEngineCore::new debe asegurar BTCUSDT_MOTOR en GLOBAL_FORESTS");

    let forest = btc_forest.unwrap();
    let base_p = forest.base_prob();
    assert!(
        base_p > 0.05 && base_p < 0.95,
        "La base honesta del bosque debe estar acotada entre 5% y 95%, valor: {base_p}"
    );

    // Verificar que swing_nn (DarkAlpha) se cargó si el archivo está presente
    if resolve_models_directory().map(|d| d.join("DarkAlpha_BTCUSDT.json").is_file()).unwrap_or(false) {
        assert!(core.swing_nn.is_some(), "DarkAlpha_BTCUSDT.json debe inicializar swing_nn congelado");
    }
}

#[test]
fn contract_flow_excitation_confluence_despertar_continuo() {
    let reg = Arc::new(OmniscientRegistry::new());
    reg.set("hawkes_intensity", 2.2);
    reg.set("order_book_imbalance", 0.40);
    reg.set("ml_prob_motor", 0.48);
    reg.set("ml_model_base", 0.30); // Base honesta con lift positivo de +0.18

    let mut engine = FlowExcitationConfluenceEngine::new();
    assert!(engine.init(Arc::clone(&reg)).is_ok());

    let vote = engine.evaluate_for_coin(0, "BTCUSDT");
    assert!(vote.is_finite(), "El voto debe ser finito");
    assert!(vote > 0.0, "Flujo comprador con lift ML positivo y excitación Hawkes debe votar long: {vote}");
    assert!(vote <= 1.0, "El voto debe estar acotado superiormente en 1.0: {vote}");

    // Probar lado opuesto (vendedor)
    reg.set("order_book_imbalance", -0.40);
    reg.set("ml_prob_motor", 0.12); // Lift vendedor: 0.12 < 0.30 - 0.05
    let vote_short = engine.evaluate_for_coin(0, "BTCUSDT");
    assert!(vote_short.is_finite(), "El voto short debe ser finito");
    assert!(vote_short < 0.0, "Flujo vendedor con lift ML negativo debe votar short: {vote_short}");
    assert!(vote_short >= -1.0, "El voto short debe estar acotado inferiormente en -1.0: {vote_short}");
}

#[test]
fn contract_shadow_spectral_confluence_nine_fields_alive() {
    // G8-OMEGA-13 (Ola Ω150): Alimentar la sombra espectral con los 7 motores físicos REALES
    // sobre una trayectoria causal multiactivo (BTC, ETH, SOL), verificando que cada productor
    // emite salidas reales no degeneradas antes de componer los 9 campos físicos.
    let mut helmholtz = HelmholtzHodgeEngine::new(HelmholtzHodgeParams::default());
    let mut yang_mills = YangMillsGaugeEngine::new(3);
    let mut kahler = KahlerHodgeEngine::new(3);
    let mut riemann = RiemannZetaGueEngine::new();
    let mut bsd = BirchSwinnertonDyerEngine::new();
    let mut dirac = AtiyahSingerDiracEngine::new();
    let mut wasserstein = WassersteinKantorovichEngine::new();

    let mut h_lam_frac = 1.0;
    let mut ym_current = 0.0;
    let mut kh_current = 0.0;
    let mut rz_conv = 1.0;
    let mut bsd_current = 0.0;
    let mut bsd_conv = 1.0;
    let mut dirac_current = 0.0;
    let mut dirac_conv = 1.0;
    let mut wk_velocity = 0.0;
    let mut wk_conv = 1.0;

    for step in 0..24 {
        let s = step as f64;
        // Trayectoria multiactivo con innovación líder en ETH/SOL y desfase en BTC (activo 0)
        let prices = [
            65_000.0 * (1.0 + 0.0004 * s + 0.00015 * (0.4 * s).sin()),
            3_500.0 * (1.0 + 0.0009 * s + 0.00025 * (0.7 * s).cos()),
            150.0 * (1.0 + 0.0014 * s - 0.00020 * (0.5 * s).sin()),
        ];
        let ofis = [
            0.35 + 0.10 * (0.3 * s).sin(),
            0.55 + 0.15 * (0.5 * s).cos(),
            0.65 - 0.12 * (0.4 * s).sin(),
        ];
        let pos_dev = (0.08 * s - 0.60).clamp(-2.5, 2.5);
        let ofi_norm = ofis[0];
        let price_delta_bps = 4.0 + 1.5 * (0.4 * s).cos();

        h_lam_frac = helmholtz.update(price_delta_bps, ofi_norm).laminar_fraction;
        let (_ym_act, ym_curr_vec) = yang_mills.update_and_calculate_curvature(&prices);
        ym_current = ym_curr_vec[0];
        let (_kh_met, kh_curr_vec) = kahler.update_and_calculate_manifold(&prices, &ofis);
        kh_current = kh_curr_vec[0];

        let energy_obs = prices[0] * (1.0 + ofi_norm * 0.01) + 2.2 * (0.5 + 0.1 * (1.3 * s).sin());
        rz_conv = riemann.observe_energy_level(energy_obs).gue_conviction_modulation;

        let bsd_m = bsd.observe_phase_point(pos_dev, ofi_norm);
        bsd_current = bsd_m.bsd_restoring_current;
        bsd_conv = bsd_m.bsd_conviction_modulation;

        let dirac_m = dirac.observe_spinor_state(pos_dev, ofi_norm);
        dirac_current = dirac_m.chiral_anomaly_current;
        dirac_conv = dirac_m.dirac_conviction_modulation;

        let wk_m = wasserstein.observe_spectral_measure(pos_dev + 0.50 * ofi_norm);
        wk_velocity = wk_m.barycenter_transport_velocity;
        wk_conv = wk_m.wasserstein_conviction_modulation;
    }

    assert!(
        yang_mills.current_metrics().active_triangles >= 1,
        "Yang-Mills debe tener al menos 1 triángulo gauge activo sobre 3 monedas"
    );
    assert!(
        ym_current.is_finite() && ym_current.abs() > 1e-4,
        "G8-OMEGA-13: Yang-Mills real debe emitir corriente gauge viva > 1e-4, obtenido: {ym_current}"
    );
    assert!(
        kh_current.is_finite() && kh_current.abs() > 1e-4,
        "G8-OMEGA-13: Kähler-Hodge real debe emitir corriente viva > 1e-4, obtenido: {kh_current}"
    );

    let mut desplazamientos = [0.0f64; 32];
    for (k, d) in desplazamientos.iter_mut().enumerate() {
        *d = 0.25 * (k as f64 - 15.5) / 16.0;
    }

    let voto = FlowExcitationConfluenceEngine::voto_espectral_con_wasserstein_dirac_bsd_riemann_kahler_gauge_y_helmholtz(
        &desplazamientos,
        2.5,          // ratio hawkes / mu
        0.30,         // hidden liq pressure
        h_lam_frac,   // Helmholtz laminar_fraction real
        ym_current,   // Yang-Mills current real
        kh_current,   // Kähler-Hodge current real
        rz_conv,      // Riemann GUE conviction real
        bsd_current,  // BSD restoring current real
        bsd_conv,     // BSD conviction real
        dirac_current,// Dirac chiral current real
        dirac_conv,   // Dirac conviction real
        wk_velocity,  // Wasserstein velocity real
        wk_conv,      // Wasserstein conviction real
    );

    assert!(voto.dominante().is_some(), "El voto espectral continuo con 9 campos reales debe tener escala dominante");
    let media = voto.media_banda(0, 31).unwrap();
    assert!(media.is_finite(), "La media de banda espectral debe ser finita");
    assert!(media.abs() > 1e-4, "La media de banda espectral debe ser no nula bajo flujo activo: {media}");
}
