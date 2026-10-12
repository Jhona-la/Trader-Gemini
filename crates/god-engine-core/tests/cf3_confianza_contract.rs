//! C-F3.3 (Línea C): la convicción de una rama (D-752) contra soluciones
//! conocidas.
//!
//! Teoría que se prueba:
//! - `conviccion_de_rama` toma el intervalo de Wilson (z = Z95) del historial
//!   de la rama. Hasta CL-75 devolvía la cota inferior si lo > ½, la
//!   superior si hi < ½ y el piso por magnitud si el intervalo contenía ½.
//!   Desde CL-75 (`conviccion_de_rama_en`, con el equilibrio p* de la orden)
//!   devuelve 0 si hi < p*, máx(lo, piso) si lo > p* y el piso en otro caso.
//! - Una convicción construida sobre evidencia debe ser MONÓTONA en la
//!   evidencia: con la misma n y el mismo piso, más aciertos no pueden bajar
//!   la convicción.
//! - El umbral que separa ventaja de desventaja no es ½ sino el equilibrio de
//!   la geometría. Con TP = RR·SL y una fricción de ida y vuelta f = φ·SL,
//!   la esperanza por operación en unidades de SL es
//!   `E = p·(RR − φ) − (1 − p)·(1 + φ)`, que se anula en
//!   `p* = (1 + φ)/(RR + 1)`. Con el genoma activo (tp_rr_ratio_btc = 2,25 en
//!   config_dir/genomes/{prod,demo,backtest}/active.json) y φ = 0,2,
//!   p* = 1,2/3,25 ≈ 0,369. Una rama cuya cota inferior supera p* tiene
//!   EV > 0 demostrado y no debe quedar «condenada» bajo ½.
//! - El doc de D-752 dice que una desventaja demostrada (hi < ½) «queda bajo
//!   ½ y el gate la rechaza». Los multiplicadores sin estado que se aplican
//!   después (modulación espectral hasta ×1,35, `confianza_modulada`) no
//!   pueden devolverla por encima de ½.
//!
//! Son pruebas de solución conocida de la fase C-F3
//! (docs/audit/HALLAZGOS_C-F3_2026-10-11.md, F3.3-01 y F3.3-02): los valores
//! de Wilson son aritmética cerrada. Convención: un test que hoy falla por la
//! causa de un hallazgo lleva `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con
//! el assert intacto; documenta el defecto y se activa con el arreglo. Para
//! correrlos: `cargo test ... -- --include-ignored`.

use god_engine_core::diffusion::Z95;
use god_engine_core::{
    confianza_modulada, conviccion_de_rama, conviccion_de_rama_en,
    p_equilibrio as p_equilibrio_orden, GodEngineCore, TasaAcierto,
};
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;

/// Equilibrio de la geometría: `p* = (1 + φ)/(RR + 1)` (ver doc del módulo).
fn p_equilibrio(rr: f64, phi: f64) -> f64 {
    (1.0 + phi) / (rr + 1.0)
}

/// RR del genoma activo (tp_rr_ratio_btc).
const RR_GENOMA: f64 = 2.25;

/// T1 (F3.3-01). Solución conocida (Wilson, z = 1,96):
/// - 21/30 ⇒ [0,5212; 0,8334]: lo > ½, devuelve 0,5212.
/// - 16/30 ⇒ [0,3614; 0,6977]: contiene ½, devuelve el piso 0,75.
///
/// Sin tolerancia: es una desigualdad entre dos salidas de la misma función.
#[test]
fn cf3_conviccion_monotona_en_la_evidencia() {
    // Misma n y mismo piso por magnitud; sólo cambia el número de aciertos.
    let piso = 0.75;
    let r_mas = TasaAcierto {
        n: 30,
        aciertos: 21,
    };
    let r_menos = TasaAcierto {
        n: 30,
        aciertos: 16,
    };
    let (lo_mas, hi_mas) = r_mas.intervalo(Z95);
    let (lo_menos, hi_menos) = r_menos.intervalo(Z95);
    let mas = conviccion_de_rama(&r_mas, piso);
    let menos = conviccion_de_rama(&r_menos, piso);
    println!(
        "observado: 21/30 ⇒ Wilson [{lo_mas:.4}; {hi_mas:.4}] ⇒ {mas:.4}; \
         16/30 ⇒ [{lo_menos:.4}; {hi_menos:.4}] ⇒ {menos:.4}"
    );
    // Premisa: los valores de Wilson son los de la solución conocida.
    assert!(
        (lo_mas - 0.5212).abs() < 5e-4,
        "premisa Wilson 21/30: lo = {lo_mas}"
    );
    assert!(lo_menos < 0.5 && hi_menos > 0.5, "premisa Wilson 16/30");
    assert!(
        mas >= menos,
        "21/30 da {mas} y 16/30 da {menos}: más aciertos no puede bajar la convicción"
    );
}

/// T2 (F3.3-01). Una rama con 180/400 cierres netos ganadores (45 %):
/// Wilson [0,4020; 0,4990]. Su cota inferior supera p* para toda fricción
/// φ < lo·(RR + 1) − 1 ≈ 0,306 (el test imprime ese límite); con φ = 0,2,
/// p* ≈ 0,369. Tiene EV > 0 demostrado y la función, que compara con ½, la
/// condena con hi = 0,499.
///
/// Nota del escéptico de refutar: p* depende de la fricción de cada orden;
/// `suelo_tp_sl` admite φ hasta 0,65, y ahí p* = 0,508 > ½. Por eso la
/// premisa se verifica explícitamente con φ = 0,2. El arreglo probablemente
/// añadirá p* como parámetro de `conviccion_de_rama`: entonces se actualiza
/// la llamada, no la aserción.
#[test]
fn cf3_conviccion_no_condena_una_rama_sobre_su_equilibrio() {
    // Genoma activo: TP = 2,25·SL; fricción ≈ 0,2·SL.
    let phi = 0.2;
    let p_star = p_equilibrio(RR_GENOMA, phi); // ≈ 0,369
    let rama = TasaAcierto {
        n: 400,
        aciertos: 180,
    }; // 45 % de cierres netos ganadores
    let (lo, hi) = rama.intervalo(Z95);
    let phi_max = lo * (RR_GENOMA + 1.0) - 1.0;
    // CL-75: la función recibe p* (`conviccion_de_rama_en`); la aserción no cambia.
    let c = conviccion_de_rama_en(&rama, 0.75, p_star);
    println!(
        "observado: 180/400 ⇒ Wilson [{lo:.4}; {hi:.4}], p*(φ = {phi}) = {p_star:.4}, \
         la premisa vale para φ < {phi_max:.3}; convicción = {c:.4}"
    );
    assert!(lo > p_star, "premisa: cota inferior {lo} > p* {p_star}");
    assert!(c >= 0.5, "EV > 0 demostrado y la rama recibe {c} < ½");
}

/// T3 (F3.3-02). Una rama con 80/200 aciertos: Wilson [0,335; 0,4692], así
/// que D-752 emite hi = 0,4692 < ½. Los multiplicadores de la modulación
/// espectral (hasta ×1,35, techo 0,98) se aplican sin estado: con ×1,10 ya
/// da 0,516 ≥ ½ y con ×1,35, 0,633.
///
/// La parte del impulso ML de las puertas (hasta ×1,5) vive en
/// `puertas_del_continuo`, que es privada; queda fuera de esta prueba.
#[test]
fn cf3_desventaja_demostrada_no_se_reinfla() {
    let perdedora = TasaAcierto {
        n: 200,
        aciertos: 80,
    };
    let c = conviccion_de_rama(&perdedora, 0.90);
    let modulados: Vec<(f64, f64)> = [1.0_f64, 1.10, 1.25, 1.35]
        .iter()
        .map(|&m| (m, confianza_modulada(c, m)))
        .collect();
    println!("observado: 80/200 ⇒ convicción {c:.4}; modulada: {modulados:?}");
    assert!(c < 0.5, "premisa D-752: cota superior bajo ½, obtenida {c}");
    for (m, x) in modulados {
        assert!(x < 0.5, "la modulación espectral ×{m} reinfla {c} a {x}");
    }
}

/// CL-75. Verdad: con n, piso y p* fijos, la convicción es no decreciente en
/// los aciertos, para toda n ≤ 200, piso ∈ {0,55; 0,75; 0,95} y p* ∈ {0,30;
/// 0,369; 0,5; 0,6}. Es la propiedad que T1 comprueba en un solo punto.
#[test]
fn cf3_conviccion_monotona_en_toda_la_rejilla() {
    for &p_star in &[0.30, 0.369, 0.5, 0.6] {
        for &piso in &[0.55, 0.75, 0.95] {
            for n in 1..=200u32 {
                let mut previa = f64::NEG_INFINITY;
                for aciertos in 0..=n {
                    let c = conviccion_de_rama_en(&TasaAcierto { n, aciertos }, piso, p_star);
                    assert!(
                        c >= previa,
                        "no monótona: n = {n}, {aciertos} aciertos, piso {piso}, p* {p_star}: {c} < {previa}"
                    );
                    previa = c;
                }
            }
        }
    }
}

/// CL-75. Verdad: `p_equilibrio(tp, sl, f)` anula la esperanza neta del veto
/// EV del risk-engine, `p·tp − (1 − p)·sl − f = 0`, y con RR = 2,25 y
/// f = 0,2·SL da (1 + 0,2)/(2,25 + 1) = 0,3692.
#[test]
fn cf3_p_equilibrio_anula_la_esperanza() {
    for &(tp, sl, f) in &[
        (0.0225, 0.01, 0.002),
        (0.004, 0.002, 0.0014),
        (0.05, 0.01, 0.0),
    ] {
        let p = p_equilibrio_orden(tp, sl, f).expect("geometría válida");
        let e = p * tp - (1.0 - p) * sl - f;
        assert!(e.abs() < 1e-15, "E(p*) = {e} con tp {tp}, sl {sl}, f {f}");
    }
    let p = p_equilibrio_orden(0.0225, 0.01, 0.002).unwrap();
    assert!((p - 1.2 / 3.25).abs() < 1e-12, "p* = {p}");
    assert!(p_equilibrio_orden(0.0, 0.01, 0.001).is_none());
    assert!(p_equilibrio_orden(0.01, 0.01, f64::NAN).is_none());
}

/// CL-75. Verdad: sin precio ni ATR no hay sonda y el núcleo devuelve ½ (la
/// orden la vetaría QO-586 de todos modos); con precio y ATR, p* está en
/// (0; ½), porque el RR del gen (≥ 2,25) supera de sobra la fricción.
#[test]
fn cf3_p_equilibrio_de_banda_del_nucleo() {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let core = GodEngineCore::new(arena.clone());
    assert_eq!(core.p_equilibrio_de_banda(0, 600_000.0), 0.5);
    arena.coins[0].current_price.store(100.0, Ordering::Relaxed);
    arena.coins[0].current_atr.store(0.5, Ordering::Relaxed);
    let p = core.p_equilibrio_de_banda(0, 600_000.0);
    println!("observado: p* a 10 min con ATR 0,5 % = {p:.4}");
    assert!(p > 0.0 && p < 0.5, "p* = {p}");
}
