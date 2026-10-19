//! Contrato formal para Ola Ω169:
//! - F1-056: Erradicación de `nn_penalty` en `ModelEnsemble::combine()`, unificando el `SkillTracker`
//!   como telemetría pura (lib.rs:8915) y permitiendo que el algoritmo Hedge por Brier pondere los modelos.
//! - F1-057: Persistencia del calibrador conformal en `MonedaGuardada` / `EstadoAprendido`,
//!   erradicando la vulnerabilidad de fail-open tras reinicio del host y optimizando `eval_prediction`.
//! - F1-058: Carga directa desde la fuente JSON (`load_model_from_source` / `load_global_from_source`)
//!   y escritura atómica de la caché `.bin` (tmp + rename), evitando el servicio de modelos obsoletos.

use god_engine_core::conformal::ConformalCalibrator;
use god_engine_core::ensemble::{ModelEnsemble, ModelId};
use god_engine_core::estado_aprendido::{self, EpigeneticaGuardada, EstadoAprendido, MetricasGuardadas, MonedaGuardada};
use god_engine_core::ml_inference::NanoForest;
use std::path::PathBuf;

struct TempFixture {
    path: PathBuf,
}

impl TempFixture {
    fn new(name: &str) -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("tg-{name}-{}-{stamp}", std::process::id()));
        std::fs::create_dir_all(&path).expect("crea dir temporal");
        Self { path }
    }
}

impl Drop for TempFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[test]
fn f1_056_ensemble_combines_without_nn_penalty_and_preserves_brier_hedge_weights() {
    let mut ensemble = ModelEnsemble::new();

    // En el estado inicial, ambos modelos tienen log_weights iguales (0.0).
    ensemble.submit(ModelId::MotorForest, 0.70);
    ensemble.submit(ModelId::DarkAlphaNN, 0.30);
    let combined_init = ensemble.combined().expect("debe combinar");
    assert!((combined_init - 0.50).abs() < 1e-12, "pesos iguales dan media exacta 0.50");

    // Simulamos 800 resultados (> SKILL_SPAN_BARS = 720) donde el bosque acierta y la red falla.
    // El SkillTracker aprende de la predicción combinada.
    for i in 0..800 {
        ensemble.submit(ModelId::MotorForest, 0.90);
        ensemble.submit(ModelId::DarkAlphaNN, 0.10);
        let y = if i % 10 != 0 { 1.0 } else { 0.0 };
        ensemble.update_with_outcome(y);
    }

    let weights = ensemble.weights();
    assert!(
        weights[ModelId::MotorForest as usize] > weights[ModelId::DarkAlphaNN as usize],
        "Hedge debe premiar al bosque y penalizar a la red: {:?}",
        weights
    );

    // Ahora evaluamos si `combine()` usa directamente los log_weights sin nn_penalty.
    // Sometemos predicciones idénticas a ambos modelos.
    ensemble.submit(ModelId::MotorForest, 0.60);
    ensemble.submit(ModelId::DarkAlphaNN, 0.60);
    let combined_equal = ensemble.combined().expect("debe combinar");
    assert!(
        (combined_equal - 0.60).abs() < 1e-12,
        "con opiniones idénticas la salida debe ser exactamente 0.60 sin distorsión de nn_penalty"
    );

    // Verificamos que el skill_z existe como telemetría diagnóstica (> 720 barras observadas)
    assert!(ensemble.skill_z().is_some(), "con 800 barras skill_z debe ser Some");
}

#[test]
fn f1_057_conformal_persistence_in_moneda_guardada_prevents_fail_open_on_restart() {
    // 1. Calibrador con 40 observaciones (is_warm == true)
    let mut calib = ConformalCalibrator::new();
    calib.set_target_alpha(0.08);
    for i in 0..40 {
        calib.update(0.85, i % 2 == 0);
    }
    assert!(calib.is_warm(), "calibrador original debe estar caliente");
    let (p_orig, acc_orig) = calib.eval_prediction(0.85);

    // 2. Exportamos snapshot
    let snapshot = calib.snapshot();
    assert_eq!(snapshot.scores.len(), 40);

    // 3. Empaquetamos en EstadoAprendido
    let moneda = MonedaGuardada {
        simbolo: "BTCUSDT".to_string(),
        metricas: MetricasGuardadas {
            trade_count: 40,
            win_rate: 0.50,
            profit_factor: 1.2,
            kelly_fraction: 0.05,
            gross_wins: 10.0,
            gross_losses: 8.0,
            pnl_realized: 2.0,
            pnl_gross: 2.5,
            roi_pre_fee: 0.15,
            roi_post_fee: 0.12,
            zombie_promotions: 0,
        },
        calibrador: vec![],
        calibrador_reloj_ms: None,
        epigenetica: EpigeneticaGuardada {
            sesgo: 1.0,
            umbral: 1.0,
            ultimo_cierre_ms: 100_000,
            ultimo_cierre_largo: true,
            ultimo_cierre_gano: true,
        },
        siniestros: vec![],
        detector_lg22: None,
        conformal: Some(snapshot.clone()),
    };

    let estado = EstadoAprendido {
        version: estado_aprendido::VERSION_ESTADO,
        monedas: vec![moneda],
        calibrador_global: vec![],
        calibrador_global_reloj_ms: None,
        conformal_global: Some(snapshot),
        freno_estrategia: estado_aprendido::FrenoGuardado {
            e_value: 1.0,
            log_wealth: 0.0,
            peak_e_value: 1.0,
            count: 10,
            running_mean: 0.0,
            running_sq_mean: 1.0,
            last_lambda: 0.5,
            rechazada: false,
        },
        riesgo_por_operacion: 0.01,
        riesgo_por_operacion_usd: 0.13,
        ganancia_en_stops: 0.0,
        perdida_en_stops: 0.0,
        resultado_en_stops: 0.0,
        pico_en_stops: 0.0,
    };

    // 4. Validar el estado
    assert!(estado_aprendido::validar(&estado).is_ok());

    // 5. Simular restauración en calibrador frío
    let mut calib_restaurado = ConformalCalibrator::new();
    assert!(!calib_restaurado.is_warm(), "calibrador nuevo arranca frío (fail-open)");
    calib_restaurado.restaurar(estado.monedas[0].conformal.as_ref().unwrap());

    // 6. Certificar que tras restaurar está caliente e inmunizado contra fail-open
    assert!(calib_restaurado.is_warm(), "tras restaurar debe estar caliente");
    let (p_rest, acc_rest) = calib_restaurado.eval_prediction(0.85);
    assert!((p_orig - p_rest).abs() < 1e-12);
    assert_eq!(acc_orig, acc_rest);
    assert_eq!(calib.accepts(0.85), calib_restaurado.accepts(0.85));
}

#[test]
fn f1_058_model_reload_from_source_bypasses_stale_cache_and_writes_atomically() {
    let fixture = TempFixture::new("f1_058_reload");
    let json_path = fixture.path.join("TEST_MOTOR.json");
    let bin_path = fixture.path.join("TEST_MOTOR.bin");

    // Modelo inicial con init_score = 0.42
    let model_json = r#"{
        "tree_offsets": [0, 1],
        "feature": [0],
        "threshold": [0.5],
        "children_left": [-1],
        "children_right": [-1],
        "value": [0.1],
        "init_score": 0.42
    }"#;
    std::fs::write(&json_path, model_json).expect("escribe json");

    // 1. Cargar desde la fuente
    let forest = NanoForest::load_model_from_source(json_path.to_str().unwrap())
        .expect("carga desde fuente");
    assert!((forest.init_value() - 0.42).abs() < 1e-5);

    // 2. Comprobar que se generó el bin atómicamente
    assert!(bin_path.exists(), "debe haberse escrito la caché binaria");

    // 3. Simular que un proceso externo escribe un .bin obsoleto con init_score = -0.99
    // y fija su timestamp más nuevo que el JSON
    let model_obsoleto = r#"{
        "tree_offsets": [0, 1],
        "feature": [0],
        "threshold": [0.5],
        "children_left": [-1],
        "children_right": [-1],
        "value": [0.1],
        "init_score": -0.99
    }"#;
    let stale_data: god_engine_core::ml_inference::NanoForestData =
        serde_json::from_str(model_obsoleto).unwrap();
    let stale_encoded = bincode::serialize(&stale_data).unwrap();
    std::fs::write(&bin_path, stale_encoded).unwrap();

    // 4. Con `load_model_from_source` se EXIGE leer el JSON fuente ignorando el .bin obsoleto
    let reloaded = NanoForest::load_model_from_source(json_path.to_str().unwrap())
        .expect("recarga desde fuente");
    assert!(
        (reloaded.init_value() - 0.42).abs() < 1e-5,
        "load_model_from_source debe leer el JSON original (0.42) e ignorar el bin obsoleto (-0.99)"
    );

    // 5. Carga global desde la fuente
    NanoForest::load_global_from_source("TEST_KEY_F1_058", json_path.to_str().unwrap())
        .expect("load_global_from_source exitoso");
    let pred = NanoForest::predict_global("TEST_KEY_F1_058", &[0.0]);
    assert!(pred.is_some());
}
