//! C-F5c · R6 de la revisión posterior del PR #62 (CL-134): replantar el
//! bosque sombra reinicia los frenos de cada universo.
//!
//! Teoría. `ShadowForest::replant` devuelve a cada universo al capital
//! inicial con un genoma nuevo: es otra hipótesis (ADR-0017 §3), y su caída
//! empieza en ese capital. Hasta ahora reponía el capital y el pico propios
//! del bosque, pero no el estado de riesgo del núcleo del universo: el pico
//! compartido del arena (CL-99), el pico del motor de riesgo, lo medido de
//! las órdenes (CL-106), la caída en stops (CL-130) ni el freno de la
//! estrategia (CL-107). Un universo que había subido a 14,1 USD quedaba, al
//! volver a 13, con una caída del 7,8 % desde un pico que ya no es suyo: el
//! freno duro le vetaba las entradas y nunca llegaba a los 15 cierres que
//! pide la cosecha. Si era el control, su aptitud quedaba muda.
//!
//! Fixture: bosque de 3 universos con 13 USD. Antes de replantar, cada uno
//! tiene un pico de 14,1 USD, lo medido de unas órdenes con stop del 0,5 %,
//! y el freno de la estrategia echado.
//!
//! Esperado: tras replantar, cada universo empieza como uno nuevo.

use evolution_engine::random_forest::ShadowForest;
use quantum_arena::genome::SuperGenotype;
use risk_engine::freno_estrategia::FrenoEstrategia;
use std::sync::atomic::Ordering::Relaxed;

fn bosque_con_historia() -> ShadowForest {
    let forest = ShadowForest::new(13.0, SuperGenotype::default(), 3);
    for engine in &forest.engines {
        risk_engine::drawdown::registrar_pico(&engine.arena.pico_capital, 14.1);
        engine.arena.riesgo_por_operacion.store(0.002, Relaxed);
        engine.arena.riesgo_por_operacion_usd.store(0.0255, Relaxed);
        engine.arena.ganancia_en_stops.store(1.96, Relaxed);
        engine.arena.perdida_en_stops.store(1.29, Relaxed);
    }
    forest
}

#[test]
fn cl134_replantar_reinicia_el_estado_de_riesgo_de_cada_universo() {
    let mut forest = bosque_con_historia();
    for engine in forest.engines.iter_mut() {
        engine.risk_engine.peak_capital = 14.1;
        engine.risk_engine.freno_estrategia.rechazada = true;
    }
    forest.replant(SuperGenotype::default());
    for (i, engine) in forest.engines.iter().enumerate() {
        let a = &engine.arena;
        println!(
            "observado: universo {i}: pico {} / {}, freno {}, riesgo {}",
            a.pico_capital.load(Relaxed),
            engine.risk_engine.peak_capital,
            engine.risk_engine.freno_estrategia.rechazada,
            a.riesgo_por_operacion_usd.load(Relaxed)
        );
        assert_eq!(a.pico_capital.load(Relaxed), 13.0, "universo {i}: pico compartido");
        assert_eq!(engine.risk_engine.peak_capital, 13.0, "universo {i}: pico del riesgo");
        assert_eq!(
            engine.risk_engine.freno_estrategia,
            FrenoEstrategia::new(),
            "universo {i}: freno de la estrategia"
        );
        for (nombre, v) in [
            ("riesgo_por_operacion", a.riesgo_por_operacion.load(Relaxed)),
            ("riesgo_por_operacion_usd", a.riesgo_por_operacion_usd.load(Relaxed)),
            ("ganancia_en_stops", a.ganancia_en_stops.load(Relaxed)),
            ("perdida_en_stops", a.perdida_en_stops.load(Relaxed)),
        ] {
            assert_eq!(v, 0.0, "universo {i}: {nombre}");
        }
    }
}

/// Esperado: la caída en stops de las posiciones (CL-130) también vuelve a
/// cero; el universo replantado no hereda un freno duro echado.
#[test]
fn cl134_replantar_reinicia_la_caida_en_stops() {
    let mut forest = bosque_con_historia();
    for engine in &forest.engines {
        for _ in 0..50 {
            risk_engine::drawdown::registrar_resultado_en_stops(&engine.arena, -1.2);
        }
        assert!(risk_engine::drawdown::caida_en_stops(&engine.arena) > 59.0);
    }
    forest.replant(SuperGenotype::default());
    for (i, engine) in forest.engines.iter().enumerate() {
        assert_eq!(risk_engine::drawdown::caida_en_stops(&engine.arena), 0.0, "universo {i}");
        assert_eq!(engine.arena.resultado_en_stops.load(Relaxed), 0.0, "universo {i}");
        assert_eq!(engine.arena.pico_en_stops.load(Relaxed), 0.0, "universo {i}");
    }
}
