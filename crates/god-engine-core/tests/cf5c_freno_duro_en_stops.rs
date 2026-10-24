//! C-F5c · R1 de la revisión posterior del PR #62 (CL-130): el freno duro de
//! la cuenta mide la caída en stops DE CADA POSICIÓN.
//!
//! Teoría. `K(w, l)` es el cuantil 95 % de la caída máxima en stops de 200
//! operaciones bajo H0 = +0,05 R (ADR-0017 §2), con operaciones que ganan `w`
//! o pierden `l` stops de su propia posición. Hasta ahora la caída se medía en
//! USD y se dividía por `s̄`, la EWMA del riesgo al stop de las últimas
//! órdenes. Con el dimensionado en espacio de riesgo (LH-04) y el factor de
//! racha (GLM 118), el tamaño baja durante una racha perdedora: `s̄` encoge y
//! la misma caída en USD cuenta como muchos más stops. En el Monte Carlo de la
//! revisión, con L = 10 disparaba tras 13 SL en vez de 31, y la falsa alarma
//! con +0,30 R de ventaja subía del 0 % al 4,5-23 %.
//!
//! Fixture. Cuenta de 100 USD, largos a 100 con stop en 99 (1 %) y TP en
//! 102,25 (RR 2,25), todos cerrados en 98,9 por el camino de cierre del
//! núcleo. En la primera fase la cantidad baja un 15 % tras cada pérdida, como
//! al encoger la fracción de riesgo; en la segunda queda fija. El arena tiene
//! la geometría y el riesgo medidos que deja la última orden, como en
//! producción.
//!
//! Esperado: el riesgo veta por drawdown si y sólo si la caída acumulada en
//! stops de las posiciones cerradas llega a K. Antes vetaba tras 12 pérdidas,
//! con unos 14 stops.

use god_engine_core::{outcome_context::OutcomeContext, GodEngineCore};
use quantum_arena::{position::PositionHorizon, symbol_registry, GlobalArena};
use risk_engine::{RiskEngine, REJECT_NAMES};
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc};

const CAPITAL: f64 = 100.0;
const ENTRADA: f64 = 100.0;
const SL: f64 = 99.0;
const TP: f64 = 102.25;
const SALIDA: f64 = 98.9;
/// Geometría en stops de una orden con stop del 1 %, RR 2,25 y 14,5 pb de
/// fricción: `w = (TP − f)/SL`, `l = (SL + f)/SL`.
const W: f64 = (0.0225 - 0.00145) / 0.01;
const L: f64 = (0.01 + 0.00145) / 0.01;

fn nucleo() -> (Arc<GlobalArena>, GodEngineCore) {
    quantum_arena::symbols::update_dynamic_universe(vec!["CFQAUSDT".into(), "CFQBUSDT".into()]);
    symbol_registry::update_registry(vec![
        symbol_registry::get_official_binance_spec("CFQAUSDT"),
        symbol_registry::get_official_binance_spec("CFQBUSDT"),
    ]);
    let arena = GlobalArena::build_in_own_stack(CAPITAL);
    let mut core = GodEngineCore::new_with_outcome_context(
        arena.clone(),
        OutcomeContext::IsolatedSimulation,
    );
    core.swing_nn = None;
    arena.ganancia_en_stops.store(W, Relaxed);
    arena.perdida_en_stops.store(L, Relaxed);
    (arena, core)
}

/// Abre un largo de `qty` y lo cierra en su stop por el camino de cierre del
/// núcleo. Devuelve la pérdida en USD.
fn perder(arena: &GlobalArena, core: &mut GodEngineCore, qty: f64, t: u64) -> f64 {
    let antes = arena.unified_capital.load(Relaxed);
    let pos = &arena.coins[0].positions.slot_2;
    assert!(pos.open_with_horizon(true, ENTRADA, qty, qty * ENTRADA / 10.0, t, TP, SL, PositionHorizon::Continuous));
    arena.used_margin.store(qty * ENTRADA / 10.0, Relaxed);
    // Riesgo al stop en USD de la última orden, como lo deja el riesgo.
    arena.riesgo_por_operacion_usd.store(qty * (ENTRADA - SL), Relaxed);
    let (_, cierre, _) =
        core.process_tick_dual(0, SALIDA, SALIDA + 0.01, 5.0, 5.0, t + 1_000, &[0.0; 54], true, false);
    assert!(cierre.is_some(), "el stop se toca en {SALIDA}");
    assert!(!pos.is_open());
    antes - arena.unified_capital.load(Relaxed)
}

/// ¿Veta por drawdown un motor de riesgo nuevo sobre este arena?
fn veta(arena: &GlobalArena) -> bool {
    let larga = SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        win_probability: 0.9,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    };
    let orden = RiskEngine::new(CAPITAL).evaluate_quantum_order(1, &larga, arena);
    orden.signal == SignalType::Flat
        && risk_engine::ultimo_rechazo().map(|i| REJECT_NAMES[i]) == Some("drawdown")
}

#[test]
fn cl130_el_freno_duro_cuenta_stops_de_cada_posicion_aunque_el_tamano_baje() {
    let (arena, mut core) = nucleo();
    let k = risk_engine::drawdown::caida_alfa_freno_duro(W, L).expect("K de la geometría");
    let (mut stops, mut qty, mut t) = (0.0_f64, 1.0_f64, 1_000_u64);
    let mut disparo = None;
    for n in 1..=60 {
        if n <= 12 {
            qty = 0.85_f64.powi(n - 1);
        }
        let perdida = perder(&arena, &mut core, qty, t);
        t += 10_000;
        stops += perdida / (qty * (ENTRADA - SL));
        let v = veta(&arena);
        let caida_usd = (CAPITAL - arena.unified_capital.load(Relaxed)) / CAPITAL;
        if n == 12 {
            println!("observado: tras 12 pérdidas, {stops:.2} stops ({:.2} %), veta {v}", caida_usd * 100.0);
        }
        if (stops - k).abs() > 1e-6 {
            assert_eq!(
                v,
                stops >= k,
                "pérdida {n}: {stops:.2} stops de sus posiciones frente a K = {k:.2} ({:.2} % en USD)",
                caida_usd * 100.0
            );
        }
        if v {
            disparo = Some((n, stops));
            break;
        }
    }
    let (n, s) = disparo.expect("con la caída en stops ≥ K el freno tiene que disparar");
    println!("observado: K = {k:.2} stops; dispara en la pérdida {n}, con {s:.2} stops");
}
