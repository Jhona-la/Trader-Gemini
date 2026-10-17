//! LG-23 (Línea G) — construir un núcleo no lee disco ni pisa el estado global.
//!
//! Ω144 metió `preload_models_from_disk()` dentro de
//! `GodEngineCore::new_with_outcome_context`. La clave de cada modelo es el
//! nombre del archivo y `store_global` sustituye la entrada, así que un
//! predictor ya registrado en memoria con la misma clave queda pisado por el
//! del disco. Los oráculos T-1, T-2 y T-3 registran un `BTCUSDT_MOTOR`
//! sintético y LUEGO construyen el núcleo: desde el checkout principal
//! (donde `../../models` existe) medían con el modelo real de BTC; desde un
//! worktree o en CI, con el sintético. El mismo oráculo daba números
//! distintos según la carpeta.
//!
//! El host carga sus modelos por su cuenta (`refresh_ml_models`) antes de
//! construir el núcleo, así que en vivo nada depende de esa precarga. Quien
//! quiera los modelos del disco en una herramienta los carga explícitamente.
//!
//! Un único test en su propio binario: muta `TG_DATA_ROOT` del proceso.
use god_engine_core::ml_inference::{NanoForest, NanoForestData};
use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;

fn bosque_constante(init_score: f32) -> NanoForestData {
    // Formato «sólo sesgo» que el validador acepta para oráculos.
    NanoForestData {
        children_left: vec![],
        children_right: vec![],
        feature: vec![],
        threshold: vec![],
        value: vec![],
        tree_offsets: vec![0, 0],
        init_score,
    }
}

#[test]
fn lg23_construir_el_nucleo_no_pisa_un_predictor_registrado() {
    let raiz = std::env::temp_dir().join(format!(
        "tg_lg23_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    let modelos = raiz.join("models");
    std::fs::create_dir_all(&modelos).unwrap();
    std::fs::create_dir_all(raiz.join("data")).unwrap();
    // En disco: base ≈ 0,12.
    std::fs::write(
        modelos.join("BTCUSDT_MOTOR.json"),
        serde_json::to_vec(&bosque_constante(-2.0)).unwrap(),
    )
    .unwrap();
    // SAFETY: único test de este binario; nadie más lee el entorno a la vez.
    unsafe { std::env::set_var("TG_DATA_ROOT", raiz.join("data")) };

    // Control positivo: la carga EXPLÍCITA sí ve ese directorio.
    assert!(god_engine_core::model_reload::preload_models_from_disk() >= 1);
    let del_disco = NanoForest::get_global("BTCUSDT_MOTOR").unwrap().base_prob();
    assert!((del_disco - 0.1192).abs() < 1e-3, "control: {del_disco}");

    // Lo que hace el oráculo: registra su predictor (base ≈ 0,88) y construye.
    NanoForest::store_global(
        "BTCUSDT_MOTOR",
        NanoForest::from_data(bosque_constante(2.0)).unwrap(),
    );
    let _core = GodEngineCore::new(GlobalArena::build_in_own_stack(1000.0));
    let tras_construir = NanoForest::get_global("BTCUSDT_MOTOR").unwrap().base_prob();

    let _ = std::fs::remove_dir_all(&raiz);
    assert!(
        (tras_construir - 0.8808).abs() < 1e-3,
        "construir el núcleo cambió el predictor registrado: base {tras_construir} (esperada 0,881)"
    );
}
