//! C-F3 · F3.7-02, v2 (Línea C, CL-97): StatArb sólo vota con reversión de
//! la base demostrada a la escala de decisión.
//!
//! Teoría que se prueba. La base b = ln F − ln S entre el futuro y el spot de
//! una moneda. StatArb vota su reversión, y el lector la deja votar si la
//! vida media publicada cumple t½ ≤ 2τ* (τ* = `dominant_tau_ms`). Esa guarda
//! sólo vale si la t½ publicada es una cota demostrada: con dos caminatas
//! independientes (b sin reversión) debe pasar en ≤ 5 % de las decisiones, y
//! con una reversión de t½ ≥ 2τ* tampoco.
//!
//! Defectos de `main` que estas pruebas fijan (revisión adversarial de CL-84):
//! - la SDE OU por evento confunde el ruido de tick y la asincronía con
//!   reversión: la guarda pasa con caminatas independientes;
//! - el futuro se compara en cada evento con un spot que no se ha refrescado;
//! - la β RLS deja un residuo paso-alto que revierte por construcción;
//! - tras un hueco largo del spot la estimación vieja sigue valiendo.
//! La reconexión (`reset_engines`) reiniciaba la prueba; desde CL-109 (W-1
//! de la revisión del PR #60) sólo corta el emparejamiento, porque los nodos
//! lentos no maduraban nunca con el corte diario del WS.
//!
//! Configuración: núcleo vivo con BTCUSDT, un evento del futuro por segundo
//! y el spot escrito justo antes de cada evento: llega entre dos eventos,
//! como el feed, y refleja el futuro vigente (el del evento anterior) más la
//! base. Caminatas log-normales con σ = 2e-4 por segundo. La τ*
//! la elige el espectro del núcleo (en estos fixtures, casi siempre τ_18 =
//! 68,7 s; a veces τ_19 o τ_20). LCG de 64 bits con Box-Muller.

mod comun_base_spot;

use comun_base_spot::*;
use strategy_core::{stat_arb::StatArbEngine, QuantumStrategy};

/// Esperado: con dos caminatas independientes (base sin reversión), la
/// guarda pasa en ≤ 4 de 40 semillas al final de 30 min. Con un contraste al
/// 5 % por decisión, P(Binomial(40; 0,05) > 4) ≈ 0,05. Medido en `main`:
/// pasa en 40/40 (la SDE estima t½ de pocos segundos).
#[test]
fn cl97_caminatas_independientes_no_pasan_la_guarda() {
    let _g = bloqueo();
    let semillas = 40u64;
    let mut pasan = 0u64;
    for s in 0..semillas {
        let (arena, mut core) = nucleo();
        let mut m = Mercado::new(s + 1, f64::INFINITY);
        for i in 0..1_800u64 {
            let (f, sp) = m.paso();
            evento(&arena, &mut core, 1_000 + i * 1_000, f, Some(sp));
        }
        if guarda_pasa(&arena) {
            pasan += 1;
        }
    }
    println!("observado: caminatas independientes que pasan la guarda: {pasan}/{semillas}");
    assert!(pasan * 10 <= semillas, "pasan {pasan}/{semillas}");
}

/// Esperado: una base OU con t½ = 1 100 s (= τ_20) no demuestra t½ < 2τ_k en
/// los nodos 17 a 19 (H0 cierta en ellos), y en 40 min el nodo 20 no madura:
/// la guarda pasa en ≤ 2 de 20 semillas. Medido en `main`: pasa casi
/// siempre.
#[test]
fn cl97_una_reversion_lenta_no_pasa_la_guarda() {
    let _g = bloqueo();
    let semillas = 20u64;
    let mut pasan = 0u64;
    for s in 0..semillas {
        let (arena, mut core) = nucleo();
        let mut m = Mercado::new(100 + s, 1_100.0);
        for i in 0..2_400u64 {
            let (f, sp) = m.paso();
            evento(&arena, &mut core, 1_000 + i * 1_000, f, Some(sp));
        }
        if guarda_pasa(&arena) {
            pasan += 1;
        }
    }
    println!("observado: reversiones lentas (t½ = 1 100 s) que pasan la guarda: {pasan}/{semillas}");
    assert!(pasan * 10 <= semillas, "pasan {pasan}/{semillas}");
}

/// Verdad: la base es ln F − ln S con β = 1 fija. El núcleo no estima β
/// (con β estimada sobre los mismos datos el residuo revierte por
/// construcción) y publica `statarb_beta` = 1 exacto.
#[test]
fn cl97_la_beta_de_la_base_es_uno() {
    let _g = bloqueo();
    let (arena, mut core) = nucleo();
    let mut m = Mercado::new(7, f64::INFINITY);
    for i in 0..600u64 {
        let (f, sp) = m.paso();
        evento(&arena, &mut core, 1_000 + i * 1_000, f, Some(sp));
    }
    let e = &core.statarb_ou_engines[0];
    println!(
        "observado: adaptive_beta = {}, β = {}, β publicada = {:?}",
        e.adaptive_beta,
        e.beta_hedge_ratio,
        leer(&arena, "statarb_beta")
    );
    assert!(!e.adaptive_beta, "el núcleo no debe estimar β");
    assert_eq!(leer(&arena, "statarb_beta").map(f64::to_bits), Some(1f64.to_bits()));
}

/// Verdad: un spot que no se refresca no forma pares. Con el spot escrito
/// una vez (fresco: < 30 s) y 25 eventos del futuro alternando ±25 pb, no
/// hay serie de la base y la guarda no pasa en ningún evento. En `main` la
/// SDE madura a los 10 eventos contra el spot fijo y la guarda pasa.
#[test]
fn cl97_un_spot_sin_refrescar_no_forma_pares() {
    let _g = bloqueo();
    let (arena, mut core) = nucleo();
    arena.update_spot_data(0, 49_999.9, 50_000.1, 1.0, 1.0);
    let mut alguna = 0u32;
    for i in 0..25u64 {
        let f = 50_000.0 * if i % 2 == 0 { 1.0025 } else { 0.9975 };
        evento(&arena, &mut core, 1_000 + i * 1_000, f, None);
        if guarda_pasa(&arena) {
            alguna += 1;
        }
    }
    println!("observado: eventos en que pasa la guarda con el spot sin refrescar: {alguna}/25");
    assert_eq!(alguna, 0);
}

/// Verdad (CL-109, W-1 de la revisión del PR #60): una reconexión sin hueco
/// no rehace la prueba. En el minuto siguiente la guarda pasa en los mismos
/// eventos que en un control sin reconexión, salvo el par que se pierde al
/// cortar el emparejamiento. Antes (CL-97) la reconexión la reiniciaba y la
/// guarda no pasaba nunca en ese minuto; el hueco largo (siguiente prueba)
/// es lo que debe hacerla madurar de nuevo.
#[test]
fn cl109_la_reconexion_sin_hueco_conserva_la_prueba() {
    let _g = bloqueo();
    let minuto = |reconectar: bool| {
        let (arena, mut core, mut m, mut ts) = nucleo_demostrado(3);
        if reconectar {
            core.reset_engines();
        }
        let mut pasa = 0u32;
        for _ in 0..60 {
            let (f, sp) = m.paso();
            evento(&arena, &mut core, ts, f, Some(sp));
            ts += 1_000;
            if guarda_pasa(&arena) {
                pasa += 1;
            }
        }
        pasa
    };
    let (control, tras) = (minuto(false), minuto(true));
    println!("observado: eventos con la guarda abierta en el minuto siguiente: control {control}/60, tras la reconexión {tras}/60");
    assert!(control >= 30, "precondición: la reversión está demostrada ({control}/60)");
    assert!(tras + 2 >= control, "la reconexión no debe cerrar la guarda: {tras} frente a {control}");
}

/// Verdad: tras 40 min sin spot (el futuro sigue), la evidencia vieja decae
/// y no hay pares que la renueven: al volver el spot, la guarda no pasa en
/// el primer minuto (el nodo vuelve a madurar). Coherente con CL-87.
#[test]
fn cl97_tras_un_hueco_largo_del_spot_vuelve_a_madurar() {
    let _g = bloqueo();
    let (arena, mut core, mut m, mut ts) = nucleo_demostrado(5);
    for _ in 0..2_400 {
        let (f, _) = m.paso();
        evento(&arena, &mut core, ts, f, None);
        ts += 1_000;
    }
    let mut pasa = 0u32;
    for _ in 0..60 {
        let (f, sp) = m.paso();
        evento(&arena, &mut core, ts, f, Some(sp));
        ts += 1_000;
        if guarda_pasa(&arena) {
            pasa += 1;
        }
    }
    println!("observado: eventos con la guarda abierta en el minuto tras el hueco: {pasa}/60");
    assert_eq!(pasa, 0);
}

/// Verdad (potencia y cableado): con la reversión demostrada, una
/// dislocación de la base de +5 pb (el spot cae; σ de la base = 1 pb) da
/// z > 1,5 y el lector vota corto en el futuro. Se mira el evento de la
/// dislocación sólo si en él la guarda sigue abierta (τ* puede saltar a un
/// nodo sin madurar); debe ocurrir en alguno de 20 intentos.
#[test]
fn cl97_con_la_reversion_demostrada_el_lector_vota() {
    let _g = bloqueo();
    let (arena, mut core, mut m, mut ts) = nucleo_demostrado(9);
    let mut lector = StatArbEngine::new(30, 1.5);
    assert!(lector.init(arena.registry.clone()).is_ok());
    for _ in 0..20 {
        let (f, sp) = m.paso();
        // El spot llega dislocado; el par se forma en este evento.
        evento(&arena, &mut core, ts, f, Some(sp * (1.0 - 5e-4)));
        ts += 1_000;
        if guarda_pasa(&arena) {
            let z = leer(&arena, "statarb_ou_zscore");
            let voto = lector.evaluate_for_coin(0, "BTCUSDT");
            println!("observado: z = {z:?}, voto = {voto}");
            assert!(matches!(z, Some(v) if v > 1.5), "z = {z:?}");
            assert!(voto < 0.0, "voto = {voto}");
            return;
        }
        for _ in 0..600 {
            let (f, sp) = m.paso();
            evento(&arena, &mut core, ts, f, Some(sp));
            ts += 1_000;
            if guarda_pasa(&arena) {
                break;
            }
        }
    }
    panic!("en 20 intentos la guarda no siguió abierta en el evento de la dislocación");
}
