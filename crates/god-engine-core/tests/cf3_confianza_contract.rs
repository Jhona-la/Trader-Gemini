//! C-F3.3 (Línea C): la convicción de una rama (D-752) contra soluciones
//! conocidas.
//!
//! Teoría que se prueba:
//! - `conviccion_de_rama` toma el intervalo de Wilson (z = Z95) del historial
//!   de la rama y devuelve: la cota inferior si lo > ½, la superior si
//!   hi < ½, y el piso por magnitud si el intervalo contiene ½.
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
use god_engine_core::{confianza_modulada, conviccion_de_rama, TasaAcierto};

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
#[ignore = "OPEN F3.3-01 (Línea C): conviccion_de_rama no es monótona: con lo > ½ devuelve lo (0,521) y con el intervalo sobre ½ devuelve el piso (0,75), así que más aciertos bajan la convicción"]
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
#[ignore = "OPEN F3.3-01 (Línea C): conviccion_de_rama compara con ½ y no con el equilibrio p* de la geometría: una rama con lo = 0,402 > p* = 0,369 recibe hi = 0,499 < ½"]
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
    let c = conviccion_de_rama(&rama, 0.75);
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
#[ignore = "OPEN F3.3-02 (Línea C): los multiplicadores sin estado posteriores a D-752 (confianza_modulada ×1,10…×1,35) reinflan una desventaja demostrada (hi = 0,469) por encima de ½"]
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
