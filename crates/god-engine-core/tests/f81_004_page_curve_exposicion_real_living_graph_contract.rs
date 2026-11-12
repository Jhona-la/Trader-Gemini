//! Contrato del cableado de la Curva de Page (Fase 81) en el grafo vivo.
//!
//! El grafo pasa al motor la exposición de la moneda en el libro: Λ = slots abiertos × la cota
//! por slot del dimensionado (riesgo × apalancamiento efectivos), o la exposición real
//! Σ|cantidad·mid| / capital si el llamador la tiene y es mayor; ρ = 1 con el propio
//! instrumento, y la volatilidad de la CWT sólo con su ventana llena. Antes pasaba los pesos
//! SU(3) de la tríada como exposición y su producto como correlación: con pesos ≈ 1 no había
//! isla y la cuenta de 13 USD con el libro plano se vetaba en cada tick.

use god_engine_core::GodEngineCore;
use god_engine_core::omniscient_living_graph::{
    OmniscientLivingGraphOrchestrator, OmniscientLivingGraphReport,
};
use quantum_arena::GlobalArena;
use signal_engine::continuous_volatility_operator::RETURN_WINDOW_SIZE;
use signal_engine::quantum_slot_orchestrator::{
    QuantumSlotOrchestrator, QuantumSlotOrchestratorParams,
};
use strategy_core::AssetColorState;

const CAPITAL_USD: f64 = 13.0;
const APALANCAMIENTO: f64 = 20.0;
const RIESGO: f64 = 0.02;
const TOL: f64 = 1e-9;
const TICK_INICIAL_MS: u64 = 1_700_000_000_000;

/// La tríada de `f78_002`: pesos ≈ 1, así que su producto (la «correlación» que se pasaba)
/// valía 1 y la isla no se formaba.
fn triada() -> [AssetColorState; 3] {
    [
        AssetColorState {
            normalized_return: 0.0005,
            order_flow_imbalance: 0.10,
            relative_weight: 1.05,
        },
        AssetColorState {
            normalized_return: -0.0002,
            order_flow_imbalance: -0.05,
            relative_weight: 0.98,
        },
        AssetColorState {
            normalized_return: 0.0003,
            order_flow_imbalance: 0.02,
            relative_weight: 1.0,
        },
    ]
}

/// Libro de un tick: mid que oscila unos pocos pb para que la CWT vea retornos no nulos.
fn libro(i: usize) -> (f64, f64) {
    let desplazamiento = if i.is_multiple_of(2) { 0.0 } else { 0.02 };
    (99.99 + desplazamiento, 100.01 + desplazamiento)
}

/// Recorre `ticks` pulsos tranquilos con los slots dados y devuelve el informe de cada uno.
fn recorrer(slots_abiertos: [bool; 3], ticks: usize) -> Vec<OmniscientLivingGraphReport> {
    let mut grafo = OmniscientLivingGraphOrchestrator::new(3);
    let mut slots = QuantumSlotOrchestrator::new(QuantumSlotOrchestratorParams::default());
    let triad = triada();
    let taus = [60_000.0; 3];
    let largos = [true; 3];
    let mut informes = Vec::with_capacity(ticks);
    for i in 0..ticks {
        let (bid, ask) = libro(i);
        informes.push(grafo.evaluate_and_synthesize(
            0,
            bid,
            ask,
            10.0,
            10.0,
            TICK_INICIAL_MS + 50 * i as u64,
            &triad,
            CAPITAL_USD,
            APALANCAMIENTO,
            RIESGO,
            &mut slots,
            &slots_abiertos,
            &taus,
            &largos,
        ));
    }
    informes
}

/// Radiación esperada del motor para una exposición Λ en un solo instrumento con la
/// volatilidad que el grafo midió en ese tick (configuración por defecto, σ_ref = 0,02).
fn radiacion_esperada(exposicion: f64, informe: &OmniscientLivingGraphReport) -> f64 {
    let vol_norm = (informe.volatility_state.effective_volatility * 100.0).max(0.01);
    (1.0 + exposicion * vol_norm.clamp(1e-4, 10.0) / 0.02).ln()
}

#[test]
fn c1_libro_plano_de_13_usd_no_se_veta_en_ningun_tick() {
    // Con el cableado anterior esta tríada vetaba todos los ticks: «correlación» 1, sin isla,
    // y S_Rad ≈ 1,6–1,8 (Shannon de los pesos más el término de volatilidad) > S_BH = ln3,6.
    for (i, informe) in recorrer([false; 3], RETURN_WINDOW_SIZE + 8)
        .iter()
        .enumerate()
    {
        let page = &informe.page_curve_island_report;
        assert_eq!(
            page.radiation_entropy, 0.0,
            "tick {i}: un libro plano no radia"
        );
        assert!(page.quantum_island_present, "tick {i}");
        assert!(
            !page.island_veto_induced,
            "tick {i}: el libro plano no se veta"
        );
    }
}

#[test]
fn c2_slot_abierto_con_la_ventana_de_la_cwt_incompleta_veta() {
    // Hasta llenar la ventana la CWT no estima (0.0) o subestima la volatilidad.
    let informes = recorrer([true, false, false], RETURN_WINDOW_SIZE - 1);
    for (i, informe) in informes.iter().enumerate() {
        assert!(
            informe.page_curve_island_report.island_veto_induced,
            "tick {i}: volatilidad sin estimación estacionaria con exposición abierta debe vetar"
        );
    }
}

#[test]
fn c3_con_la_ventana_llena_la_radiacion_es_la_de_la_cota_del_dimensionado() {
    let por_slot =
        QuantumSlotOrchestrator::exposicion_bruta_maxima_por_slot(APALANCAMIENTO, RIESGO);
    assert!(
        (por_slot - 0.4).abs() <= TOL,
        "0,02 × 20 = 0,4× equity por slot"
    );

    for (slots_abiertos, n) in [([true, false, false], 1.0), ([true, true, false], 2.0)] {
        let informes = recorrer(slots_abiertos, RETURN_WINDOW_SIZE + 8);
        // La ventana se llena en la actualización número RETURN_WINDOW_SIZE.
        for (i, informe) in informes.iter().enumerate().skip(RETURN_WINDOW_SIZE - 1) {
            let esperada = radiacion_esperada(n * por_slot, informe);
            let page = &informe.page_curve_island_report;
            assert!(
                (page.radiation_entropy - esperada).abs() <= TOL,
                "{n} slots, tick {i}: radiación {} frente a {esperada}",
                page.radiation_entropy
            );
            assert!(
                !page.quantum_island_present,
                "ρ = 1 con el propio instrumento"
            );
        }
    }
}

#[test]
fn c4_god_engine_core_pasa_la_exposicion_real_del_slot() {
    // Un slot de 0,26 unidades a 100 USD: 26 USD de nocional con 13 USD de capital, Λ = 2,0;
    // la cota del dimensionado (0,02 × 20) sólo daría 0,4.
    let mut core = GodEngineCore::new(GlobalArena::build_in_own_stack(CAPITAL_USD));
    core.arena.coins[0].positions.slots()[0].open(
        true,
        100.0,
        0.26,
        1.3,
        TICK_INICIAL_MS,
        102.0,
        99.0,
    );
    let triad = triada();
    for i in 0..RETURN_WINDOW_SIZE + 8 {
        let (bid, ask) = libro(i);
        let mid = (bid + ask) * 0.5;
        let informe = core.evaluate_omniscient_living_graph(
            0,
            bid,
            ask,
            10.0,
            10.0,
            TICK_INICIAL_MS + 50 * i as u64,
            &triad,
            CAPITAL_USD,
            APALANCAMIENTO,
            RIESGO,
        );
        let page = &informe.page_curve_island_report;
        if i + 1 < RETURN_WINDOW_SIZE {
            assert!(
                page.island_veto_induced,
                "tick {i}: ventana de la CWT incompleta"
            );
        } else {
            let real = 0.26 * mid / CAPITAL_USD;
            let esperada = radiacion_esperada(real, &informe);
            assert!(
                (page.radiation_entropy - esperada).abs() <= TOL,
                "tick {i}: radiación {} frente a {esperada} (Λ real {real})",
                page.radiation_entropy
            );
        }
    }
}
