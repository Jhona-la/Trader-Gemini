//! Contrato F1-093 (Sol) — UN LOCK ENVENENADO YA NO ABORTA EL EVOLVER ONLINE.
//!
//! # QUÉ
//! `std::sync::RwLock` se envenena cuando un hilo panica con el guard tomado, y
//! desde entonces cada `read()`/`write()` devuelve `Err`. El evolver online
//! tenía 43 `read().unwrap()` / `write().unwrap()` en producción (16 en
//! `online_random_forest.rs`, 27 en `online_newton_gbdt.rs`). Con unwinding,
//! un pánico mientras se mantiene el guard de escritura envenena el lock y
//! el siguiente `unwrap()` vuelve a panicar. Con `panic = "abort"`, el primer
//! pánico termina el proceso: este contrato no certifica su supervivencia.
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
fn envenenar_lock<T: Send + Sync>(lock: &RwLock<T>, valor: T) {
    std::thread::scope(|s| {
        let hilo = s.spawn(move || {
            let mut g = lock.write().expect("guard adquirido antes del pánico");
            *g = valor;
            panic!("pánico ajeno que envenena el lock");
        });
        assert!(hilo.join().is_err(), "precondición: pánico intencional");
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

    envenenar_lock(&forest.shadow_threshold_long, 0.77);

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
    envenenar_lock(&forest.shadow_threshold_long, 0.77);

    *escribir(&forest.shadow_threshold_long) = 0.61;
    assert_eq!(*leer(&forest.shadow_threshold_long), 0.61);
}

#[test]
fn f1_093_estado_del_bosque_sigue_siendo_legible_con_un_lock_envenenado() {
    // is_trained consulta el flag GBDT cuando el clasificador está ausente.
    // Envenenar ese mismo lock ejercita el consumidor público real; no prueba
    // entrenamiento ni consistencia de todo el estado del modelo.
    let forest = TrueOnlineRandomForest::new(512);
    envenenar_lock(&forest.gbdt.is_trained, true);

    assert!(forest.gbdt.is_trained.read().is_err());
    assert!(forest.is_trained());
}

#[test]
fn f1_093_helper_es_generico_sobre_arc() {
    // Los locks del bosque guardan `Option<Arc<...>>`; el helper debe servir
    // igual para ese tipo (no sólo para `Copy`).
    static LOCK: RwLock<Option<std::sync::Arc<u32>>> = RwLock::new(None);
    *escribir(&LOCK) = Some(std::sync::Arc::new(7));
    let sembrado = leer(&LOCK).clone().expect("valor sembrado");
    assert_eq!(*sembrado, 7);
}
