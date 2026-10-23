//! Contrato F1-093 (Sol) — UN LOCK ENVENENADO YA NO ABORTA EL EVOLVER ONLINE.
//!
//! # QUÉ
//! `std::sync::RwLock` se envenena cuando un hilo panica con el guard tomado, y
//! desde entonces cada `read()`/`write()` devuelve `Err`. El evolver online
//! tenía 43 `read().unwrap()` / `write().unwrap()` en producción (16 en
//! `online_random_forest.rs`, 27 en `online_newton_gbdt.rs`): con
//! `panic = "abort"` (LG-28), un pánico en cualquier parte del proceso
//! envenenaba el lock y el siguiente acceso del bosque abortaba el motor con
//! posiciones abiertas.
//!
//! # PRUEBA
//! Este contrato envenena un lock REAL de `TrueOnlineRandomForest` —con un
//! pánico de verdad dentro de un hilo con scope, no con un mock— y verifica que
//! los accesores públicos siguen devolviendo el estado en vez de abortar.
//! `lock_seguro::leer`/`escribir` recuperan el guard con `into_inner()`.
//!
//! # DÓNDE
//! `crates/evolution-engine/src/lock_seguro.rs` (helper único, usado por los
//! dos motores online) y sus 43 sitios de consumo.

use evolution_engine::lock_seguro::{escribir, leer};
use evolution_engine::online_random_forest::TrueOnlineRandomForest;
use std::sync::RwLock;

/// Envenena un lock REAL del bosque: un hilo panica mientras mantiene el guard
/// de escritura de `shadow_threshold_long`. Es el mecanismo auténtico de
/// `PoisonError`, no una simulación.
///
/// Usa `thread::scope`, así que el hilo puede pedir prestado el lock del
/// `forest` sin `unsafe` ni `transmute`: el scope garantiza que el hilo termina
/// antes de que la referencia deje de vivir.
fn envenenar_umbral_del_bosque(forest: &TrueOnlineRandomForest) {
    let lock: &RwLock<f32> = &forest.shadow_threshold_long;
    std::thread::scope(|s| {
        s.spawn(move || {
            let mut g = lock.write().expect("guard adquirido antes del pánico");
            *g = 0.77;
            panic!("pánico ajeno que envenena el lock");
        });
    });
}

#[test]
fn f1_093_lock_envenenado_no_aborta_el_accesor_del_bosque() {
    let forest = TrueOnlineRandomForest::new(512);

    // Estado sano: el umbral se lee normal.
    let umbral_inicial = *leer(&forest.shadow_threshold_long);
    assert!(
        umbral_inicial.is_finite(),
        "el umbral inicial debe ser finito, leído {umbral_inicial}"
    );

    envenenar_umbral_del_bosque(&forest);

    // El lock está realmente envenenado: la lectura estándar devuelve Err.
    assert!(
        forest.shadow_threshold_long.read().is_err(),
        "precondición: el lock debe estar envenenado"
    );

    // Y aun así el acceso seguro devuelve el valor escrito antes del pánico,
    // en vez de abortar el proceso como hacía `.read().unwrap()`.
    assert_eq!(
        *leer(&forest.shadow_threshold_long),
        0.77,
        "la lectura recuperada debe ver el último valor escrito"
    );
}

#[test]
fn f1_093_escritura_recuperada_persiste_sobre_lock_envenenado() {
    let forest = TrueOnlineRandomForest::new(512);
    envenenar_umbral_del_bosque(&forest);

    *escribir(&forest.shadow_threshold_long) = 0.61;
    assert_eq!(*leer(&forest.shadow_threshold_long), 0.61);
}

#[test]
fn f1_093_estado_del_bosque_sigue_siendo_legible_con_un_lock_envenenado() {
    // El bosque consume varios locks en su camino de decisión
    // (`is_trained`, `get_optimal_thresholds`). Con un lock envenenado ninguno
    // debe abortar.
    let forest = TrueOnlineRandomForest::new(512);
    envenenar_umbral_del_bosque(&forest);

    assert!(leer(&forest.last_accuracy).is_finite());
    let _entrenado = forest.is_trained();
    let _umbrales = forest.get_optimal_thresholds();
    let _obs = leer(&forest.observations).len();
}

#[test]
fn f1_093_helper_es_generico_sobre_arc() {
    // Los locks del bosque guardan `Option<Arc<...>>`; el helper debe servir
    // igual para ese tipo (no sólo para `Copy`).
    static LOCK: RwLock<Option<std::sync::Arc<u32>>> = RwLock::new(None);
    *escribir(&LOCK) = Some(std::sync::Arc::new(7));
    assert_eq!(*leer(&LOCK).as_ref().expect("valor sembrado"), 7);
}
