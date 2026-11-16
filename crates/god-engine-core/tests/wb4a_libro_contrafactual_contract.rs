//! WB-4a (especificación GLM-APP v3, §4.4.1 y §4.4.4) — libro de decisiones:
//! contrafactuales de lo vetado por triple barrera y calibración EN SOMBRA.
//! Que el vivo no cambie lo fijan `wb45_sombra_cable_contract.rs` y la traza de
//! identidad del replay (`backtest-engine/tests/wb45_sombra_identidad_ordenes.rs`).

use god_engine_core::calibration::PlattCalibrator;
use god_engine_core::libro_decisiones::{Decision, LibroDecisiones};
use risk_engine::medida_sombra::GeometriaSombra;

const G: GeometriaSombra = GeometriaSombra {
    tp_pct: 0.010,
    sl_pct: 0.005,
    friccion_pct: 0.001,
    tau_ms: 60_000.0,
};
const T0: u64 = 1_800_000_000_000;

fn vetada(score: f64, ts: u64) -> Option<Decision> {
    Decision::desde_geometria(0, true, 100.0, Some(G), score, score, 0, ts)
}

/// Selección sesgada: el vivo ejecuta lo que pierde (30 % de acierto) y el
/// consejo veta lo que la cinta paga (90 %), con los mismos scores. La
/// frecuencia real de la población es 0,66.
fn poblacion(libro: &mut LibroDecisiones, vivo: &mut PlattCalibrator, etiquetar: bool) -> (u64, f64) {
    let scores = [0.6, 0.7, 0.8, 0.9];
    let mut t = T0;
    let (mut ganadas, mut total) = (0u32, 0u32);
    for k in 0..40u32 {
        let s = scores[(k % 4) as usize];
        let won = k % 10 < 3;
        t += 1_000;
        vivo.update_at(s, won, t);
        libro.alimentar_ejecutada(0, s, won, t);
        ganadas += u32::from(won);
        total += 1;
    }
    for k in 0..60u32 {
        let s = scores[(k % 4) as usize];
        let won = k % 10 < 9;
        t += 1_000;
        assert!(libro.registrar_vetada(vetada(s, t)).is_some());
        // La cinta: toca el TP (gana) o el SL (pierde); sin etiquetar, se
        // queda dentro y vence el plazo.
        let mid = match (etiquetar, won) {
            (true, true) => 101.0,
            (true, false) => 99.5,
            (false, _) => 100.2,
        };
        assert_eq!(libro.observar(0, mid, t + 1), usize::from(etiquetar));
        if !etiquetar {
            assert_eq!(libro.observar(0, 100.2, t + 60_001), 1, "vence τ");
        }
        ganadas += u32::from(won);
        total += 1;
    }
    (t + 2, f64::from(ganadas) / f64::from(total))
}

/// 4.4.1 — con el libro, la p calibrada queda MÁS CERCA de la frecuencia real
/// de la población que la del calibrador de solo ejecutadas (brecha
/// direccional; el MAP de Platt con prior no converge exacto). Control: sin
/// etiquetar, la sombra es bit a bit la del vivo.
#[test]
fn calibracion_contrafactual_cura_el_sesgo_de_seleccion() {
    let mut libro = LibroDecisiones::new(1);
    let mut vivo = PlattCalibrator::new();
    let (ahora, real) = poblacion(&mut libro, &mut vivo, true);
    assert!((real - 0.66).abs() < 1e-12);
    assert_eq!((libro.ejecutadas, libro.contrafactuales, libro.no_resueltas), (40, 60, 0));
    for s in [0.6, 0.7, 0.8, 0.9] {
        let p_vivo = vivo.calibrate_at(s, ahora);
        let p_libro = libro.p_sombra(0, s, ahora);
        assert!(
            (p_libro - real).abs() < (p_vivo - real).abs(),
            "score {s}: con libro {p_libro}, solo ejecutadas {p_vivo}, real {real}"
        );
    }
    // Control: si el libro no etiqueta, no corrige nada.
    let mut libro = LibroDecisiones::new(1);
    let mut vivo = PlattCalibrator::new();
    let (ahora, _) = poblacion(&mut libro, &mut vivo, false);
    assert_eq!((libro.contrafactuales, libro.no_resueltas), (0, 60));
    for s in [0.6, 0.7, 0.8, 0.9] {
        assert_eq!(libro.p_sombra(0, s, ahora).to_bits(), vivo.calibrate_at(s, ahora).to_bits());
    }
}

/// 4.4.4 — una fila que vence τ sin tocar barrera se cuenta y se excluye;
/// nunca entra como pérdida, aunque el mid toque el SL después del plazo.
#[test]
fn contrafactual_no_resuelta_no_imputa() {
    let mut libro = LibroDecisiones::new(1);
    assert!(libro.registrar_vetada(vetada(0.8, T0)).is_some());
    assert_eq!(libro.observar(0, 100.2, T0 + 30_000), 0, "dentro, sin tocar");
    assert_eq!(libro.observar(0, 99.0, T0 + 60_001), 1, "vencida: se cierra");
    assert_eq!((libro.no_resueltas, libro.contrafactuales), (1, 0));
    assert_eq!(libro.observaciones().count(), 0, "no entra ninguna observación");
    assert_eq!(libro.p_sombra(0, 0.8, T0 + 60_002), PlattCalibrator::new().calibrate(0.8));
    // Control: dentro del plazo el mismo SL sí etiqueta pérdida.
    assert!(libro.registrar_vetada(vetada(0.8, T0 + 70_000)).is_some());
    assert_eq!(libro.observar(0, 99.0, T0 + 70_001), 1);
    assert_eq!(libro.contrafactuales, 1);
    assert!(libro.observaciones().all(|o| !o.won));
}

