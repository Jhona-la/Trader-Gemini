//! I-13 (Línea I, fila N26 del censo C-F7b): el ejecutor que el host publica al terminar el
//! calentamiento hereda del vigente lo que mide el exchange y no el ejecutor.
//!
//! El freno 429/418 y la cuota (`X-MBX-USED-WEIGHT-1M`, `X-MBX-ORDER-COUNT-*`) son de la IP y de
//! la cuenta. El host de main construía el sucesor con `new_with_shared`, que estrena un freno y
//! un uso a cero: durante el Retry-After de un 429 o el ban de un 418 el sucesor volvía a
//! enviar, y por encima del 80 % de la cuota seguía admitiendo lecturas. Las pruebas de conducta
//! del freno, del uso y de las plazas en vuelo viven en `executor.rs` (`tests_n26`), donde se
//! puede armar el freno sin red. Aquí, con la sesión OMS y la arena reales, sin red ni claves:
//! la API pública de `OrderExecutor::sucesor_de`, la publicación por `paso_transicion` y la
//! guarda de fuente del host.
use arc_swap::ArcSwap;
use execution_engine::executor::{ExecutionProvider, HerenciaDeCuota, OrderExecutor};
use quantum_arena::GlobalArena;
use quantum_engine::host_oms::{OmsSession, ReintentoTransicion, ResultadoTransicion};
use std::sync::{atomic::AtomicBool, Arc};

/// El ejecutor del arranque con la arena que le inyecta el host (god_engine.rs:
/// `exec.load().arena.store(Some(Arc::clone(&arena_real)))`).
fn previo(clave: &str, testnet: bool, arena: Option<Arc<GlobalArena>>) -> OrderExecutor {
    let ejecutor = OrderExecutor::new(clave.into(), "offline-contract".into(), testnet);
    ejecutor.arena.store(arena);
    ejecutor
}

/// Con la misma URL y la misma arena comparte el reloj, también con otra clave: el peso y el 418
/// son de la IP. El kill-switch del vigente cruza el cambio (N58).
#[test]
fn i13_1_con_la_misma_url_y_arena_el_sucesor_comparte_el_reloj() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let anterior = previo("k", true, Some(Arc::clone(&arena)));
    anterior.trigger_kill_switch();
    let (sucesor, herencia) = OrderExecutor::sucesor_de(
        &anterior,
        "k".into(),
        "otro-secreto".into(),
        true,
        Arc::clone(&arena),
    );
    assert_eq!(herencia, HerenciaDeCuota::Compartida);
    assert_eq!(herencia.motivo_sin_herencia(), None);
    assert!(
        Arc::ptr_eq(&sucesor.arena, &anterior.arena),
        "un solo reloj: el del freno compartido"
    );
    assert!(Arc::ptr_eq(&sucesor.registry(), &anterior.registry()));
    assert!(
        sucesor.is_kill_switch_active(),
        "el kill-switch es un latch hasta reiniciar: cruza el cambio"
    );
    assert_eq!(sucesor.api_secret(), "otro-secreto");

    let (otra_clave, herencia) = OrderExecutor::sucesor_de(
        &anterior,
        "otra-clave".into(),
        "s".into(),
        true,
        Arc::clone(&arena),
    );
    assert_eq!(
        herencia,
        HerenciaDeCuota::Compartida,
        "la misma IP: el mismo freno"
    );
    assert!(Arc::ptr_eq(&otra_clave.arena, &anterior.arena));
}

#[test]
fn i13_2_con_otra_url_o_arena_estrena_freno_y_uso_y_lo_dice() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let otra = GlobalArena::build_in_own_stack(100.0);
    let casos = [
        (
            previo("k", true, Some(Arc::clone(&arena))),
            "k",
            false,
            HerenciaDeCuota::OtraUrl,
        ),
        (
            previo("k", true, Some(otra)),
            "k",
            true,
            HerenciaDeCuota::OtraArena,
        ),
        (
            previo("k", true, None),
            "k",
            true,
            HerenciaDeCuota::OtraArena,
        ),
    ];
    for (anterior, clave, testnet, esperado) in casos {
        anterior.trigger_kill_switch();
        let (sucesor, herencia) = OrderExecutor::sucesor_de(
            &anterior,
            clave.into(),
            "s".into(),
            testnet,
            Arc::clone(&arena),
        );
        assert_eq!(herencia, esperado);
        assert!(herencia.motivo_sin_herencia().is_some(), "{esperado:?}");
        assert!(
            sucesor.is_kill_switch_active(),
            "{esperado:?}: el kill-switch cruza aunque no comparta el freno"
        );
        assert!(
            !Arc::ptr_eq(&sucesor.arena, &anterior.arena),
            "{esperado:?}"
        );
        assert!(
            sucesor
                .arena
                .load_full()
                .is_some_and(|propia| Arc::ptr_eq(&propia, &arena)),
            "{esperado:?}: el sucesor lleva la arena pedida"
        );
        assert!(
            Arc::ptr_eq(&sucesor.registry(), &anterior.registry()),
            "{esperado:?}"
        );
    }
}

/// La transición real: la sesión publica el sucesor construido desde su ejecutor vigente.
#[test]
fn i13_3_la_transicion_publica_un_sucesor_que_comparte_el_reloj_del_vigente() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let ready = Arc::new(AtomicBool::new(false));
    let executors = Arc::new(ArcSwap::from(Arc::new(previo(
        "k",
        true,
        Some(Arc::clone(&arena)),
    ))));
    let session = OmsSession::new(Arc::clone(&executors), Arc::clone(&ready));
    let vigente = executors.load_full();
    let reintento = ReintentoTransicion::new();
    let paso = session.paso_transicion(&arena, None, &reintento, || {
        let (sucesor, herencia) =
            OrderExecutor::sucesor_de(&vigente, "k".into(), "s".into(), true, Arc::clone(&arena));
        assert_eq!(herencia, HerenciaDeCuota::Compartida);
        Arc::new(sucesor)
    });
    let ResultadoTransicion::Publicada {
        ejecutor: publicado,
        ..
    } = paso.resultado
    else {
        panic!("la sesión libre publica el sucesor");
    };
    assert!(Arc::ptr_eq(&publicado, &executors.load_full()));
    assert!(!Arc::ptr_eq(&publicado, &vigente));
    assert!(Arc::ptr_eq(&publicado.arena, &vigente.arena));
}

/// Guarda de fuente: el host construye el sucesor desde el ejecutor vigente, dentro del paso de
/// la transición, y no queda ningún `new_with_shared` en el host.
#[test]
fn i13_4_el_host_construye_el_sucesor_desde_el_ejecutor_vigente() {
    let host = include_str!("../src/bin/god_engine.rs");
    let produccion = host.split("#[cfg(test)]").next().unwrap_or("");
    assert!(
        produccion.contains("oms_session.paso_transicion("),
        "corte de producción"
    );
    assert_eq!(
        produccion.matches("new_with_shared(").count(),
        0,
        "el host no estrena freno ni uso en la transición"
    );
    assert_eq!(produccion.matches("OrderExecutor::sucesor_de(").count(), 1);
    let bloque = host
        .split("if newly_transitioned {")
        .nth(2)
        .expect("bloque de la transición del ejecutor");
    let paso = bloque
        .find("oms_session.paso_transicion(")
        .expect("la transición pasa por OmsSession::paso_transicion");
    let sucesor = bloque
        .find("OrderExecutor::sucesor_de(")
        .expect("el sucesor se construye en el bloque");
    let publicada = bloque
        .find("ResultadoTransicion::Publicada")
        .expect("publicación del sucesor");
    assert!(
        paso < sucesor && sucesor < publicada,
        "dentro del paso de la transición"
    );
    let cierre = &bloque[paso..publicada];
    assert!(cierre.contains("let previo = exec.load_full();"));
    assert!(cierre.contains("&previo,"));
    assert!(cierre.contains("Arc::clone(&arena_real),"));
    assert!(
        cierre.contains(
            "telemetry_engine::telemetry_err!(\"[OMS TRANSITION] el ejecutor nuevo estrena"
        ),
        "no compartir es una anomalía: va al canal de errores"
    );
    assert!(
        cierre.contains("herencia.motivo_sin_herencia()"),
        "el host dice cuándo el sucesor estrena freno y uso"
    );
}
