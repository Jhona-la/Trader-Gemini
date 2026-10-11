//! C-F3.5 (Línea C), hallazgo F3.5-03: DarkAlpha se entrena con la entrada
//! cruda y se sirve en z-scores; el núcleo instala por la recarga en caliente
//! una red sin sistema de coordenadas declarado.
//!
//! Teoría que se prueba:
//! - `auto_trainer_daemon` parte de `DarkAlphaEngine::new(54, 64, 32)`
//!   (`scaler = None`, `freeze_normalizers = false`), entrena con x CRUDAS y
//!   valida con `predict`, que sin scaler y sin congelar actualiza los
//!   Welford y evalúa en z. Guarda la red sin scaler y sin congelar.
//! - `predict_in_context` de una red así, una vez congelada, normaliza con
//!   esos Welford (z), no con las coordenadas crudas del ajuste: los pesos se
//!   aplican a entradas en una escala distinta de la del entrenamiento.
//! - La recarga en caliente del núcleo (`process_event`, bloque `model_rx`)
//!   sólo hace `init_buffers`, `sanitize_denormals` y `freeze`, y acepta la
//!   red.
//!
//! Generador: una red como la que guarda el demonio, con Welford poblados
//! por 200 `predict` sin congelar sobre entradas en escala cruda (0..540).
//!
//! Propiedad: el núcleo no sirve una red sin sistema de coordenadas
//! declarado. Tras la recarga, `swing_nn` es None o lleva un Scaler.
//! Booleano exacto. Escéptico (refutar): exigir un Scaler es UNA salida; otra
//! válida es entrenar en las z de normalizadores ya ajustados y congelados.
//! El assert la admite: una red que llega SIN scaler y SIN congelar (el caso
//! del generador) debe rechazarse con cualquiera de las dos políticas.
//! La consecuencia viva es condicional: el repo no tiene
//! `DarkAlpha_BTCUSDT.json`; depende del artefacto del PC del operador.
//!
//! Convención de la fase C-F3 (docs/audit/HALLAZGOS_C-F3_2026-10-11.md): un
//! test que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto;
//! documenta el defecto y se activa con el arreglo. Para correrlo:
//! `cargo test ... -- --include-ignored`.

use dark_alpha_engine::DarkAlphaEngine;
use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};

#[test]
#[ignore = "OPEN F3.5-03 (Línea C, núcleo: rechazar en la recarga y en la carga inicial una red sin sistema de coordenadas; pedido a Línea E para el demonio): la recarga en caliente sólo hace init_buffers, sanitize y freeze, y acepta una red entrenada en crudo que se sirve en z de Welford"]
fn cf3_nucleo_no_sirve_red_sin_sistema_de_coordenadas() {
    symbols::update_dynamic_universe(vec!["BTCUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec("BTCUSDT")]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;

    // Como la guarda el demonio: sin scaler, sin congelar, Welford poblados por la validación.
    let mut red = DarkAlphaEngine::new(54, 64, 32);
    for k in 0..200 {
        let x: Vec<f64> = (0..54).map(|i| i as f64 * 10.0 + (k % 7) as f64).collect();
        let _ = red.predict(&x);
    }
    assert!(
        red.scaler.is_none() && !red.freeze_normalizers,
        "precondición del generador"
    );

    let (tx, rx) = std::sync::mpsc::channel();
    core.set_model_rx(rx);
    tx.send(red).unwrap();
    let _ = core.process_event(
        0, false, false, true, 100.0, 0.0, 100.0, 100.01, 5.0, 5.0, 0.0, 0.0, 10_000, false,
        &[0.0; 54], false,
    );
    println!(
        "observado: swing_nn = {}, scaler = {}, congelada = {}",
        if core.swing_nn.is_some() {
            "Some"
        } else {
            "None"
        },
        core.swing_nn.as_ref().is_some_and(|nn| nn.scaler.is_some()),
        core.swing_nn
            .as_ref()
            .is_some_and(|nn| nn.freeze_normalizers),
    );
    assert!(
        core.swing_nn
            .as_ref()
            .map_or(true, |nn| nn.scaler.is_some()),
        "el núcleo instala una red sin scaler: entrenada en crudo, servida en z de Welford"
    );
}
