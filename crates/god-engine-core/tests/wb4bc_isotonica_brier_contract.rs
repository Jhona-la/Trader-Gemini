//! WB-4b/c (especificación GLM-APP v3, §4.4.2, §4.4.3 y §4.4.5) — isotónica
//! (PAVA) como comparador, Brier fuera de muestra con corte temporal y Brier por
//! grupo con el mínimo de 30, más el diagnóstico del libro. Diagnóstico puro:
//! ninguna decisión lo lee.

use god_engine_core::calibration::{
    brier_platt_fuera_de_muestra, brier_por_grupo, brier_score, descomposicion_brier, isotonic_pava,
    isotonic_pava_ponderada, PlattCalibrator, MIN_OBS_BRIER,
};
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

/// 4.4.2 — PAVA: entrada monótona ⇒ salida idéntica; violación ⇒ pooling
/// exacto con conservación de masa ponderada; empates de score en un bloque.
#[test]
fn isotonica_pava_monotona_y_exacta_en_escalera() {
    let monotona = [(0.1, 0.0), (0.2, 0.0), (0.3, 1.0), (0.4, 1.0)];
    assert_eq!(isotonic_pava(&monotona), monotona.to_vec());
    // Violación local: [1, 0] se funde en 0,5.
    assert_eq!(
        isotonic_pava(&[(0.3, 1.0), (0.1, 1.0), (0.2, 0.0)]),
        vec![(0.1, 0.5), (0.2, 0.5), (0.3, 1.0)],
        "ordena por score y funde exacto"
    );
    // Pesos: (1·3 + 0·1)/(3 + 1) = 0,75, y la masa Σw·y se conserva.
    let filas = [(0.1, 1.0, 3.0), (0.2, 0.0, 1.0), (0.3, 1.0, 2.0)];
    let ajuste = isotonic_pava_ponderada(&filas);
    assert_eq!(ajuste, vec![(0.1, 0.75), (0.2, 0.75), (0.3, 1.0)]);
    let masa = |v: &[(f64, f64, f64)]| v.iter().map(|&(_, y, w)| w * y).sum::<f64>();
    let masa_ajuste: f64 = ajuste.iter().zip(filas.iter()).map(|(&(_, a), &(_, _, w))| w * a).sum();
    assert_eq!(masa(&filas), masa_ajuste);
    // Empates: los dos 0,5 van al mismo bloque (media 0,5) antes de comparar;
    // luego violan con el 1 anterior y se funden los tres: 2/3.
    let empate = isotonic_pava(&[(0.4, 1.0), (0.5, 1.0), (0.5, 0.0)]);
    assert_eq!(empate.len(), 3);
    assert!(empate.iter().all(|&(_, a)| (a - 2.0 / 3.0).abs() < 1e-15), "{empate:?}");
    // Nunca decrece.
    let ruido: Vec<(f64, f64)> = (0..50).map(|k| (f64::from(k) / 50.0, f64::from((k * 7) % 3 == 0))).collect();
    let curva = isotonic_pava(&ruido);
    assert!(curva.windows(2).all(|w| w[0].1 <= w[1].1));
}

/// Pares (score, won) con, por cada score, `por_score` observaciones de las
/// que ganan `round(por_score · f(score))`.
fn sintetico(f: impl Fn(f64) -> f64, por_score: u32, t_inicio: u64) -> Vec<(f64, bool, u64)> {
    let mut v = Vec::new();
    let mut t = t_inicio;
    for k in 0..7u32 {
        let s = 0.3 + 0.1 * f64::from(k);
        let ganan = (f64::from(por_score) * f(s).clamp(0.0, 1.0)).round() as u32;
        for j in 0..por_score {
            t += 1_000;
            v.push((s, j < ganan, t));
        }
    }
    v
}

/// 4.4.3 — desvío: el Platt calibra bien la mitad que lo ajustó y la tardía
/// está sobreconfiada ⇒ Brier fuera > Brier en muestra por margen. Control: si
/// se mide en muestra (ajustar con la tardía), la brecha desaparece. Segunda
/// señal, aparte: la brecha Platt-vs-isotónica EN MUESTRA detecta la familia
/// mal especificada (verdad en escalón) y no la bien especificada.
#[test]
fn brier_oos_detecta_sobreconfianza() {
    let temprana = sintetico(|s| s, 40, T0);
    let tardia: Vec<(f64, bool)> = sintetico(|s| s - 0.35, 40, T0 + 10_000_000)
        .iter()
        .map(|&(s, w, _)| (s, w))
        .collect();
    let Some(r) = brier_platt_fuera_de_muestra(&temprana, &tardia) else {
        panic!("hay evidencia en las dos mitades");
    };
    assert!(r.brier_fuera > r.brier_en_muestra + 0.03, "{r:?}");
    let tardia_fechada = sintetico(|s| s - 0.35, 40, T0 + 10_000_000);
    let Some(en_muestra) = brier_platt_fuera_de_muestra(&tardia_fechada, &tardia) else {
        panic!("hay evidencia");
    };
    assert!(
        (en_muestra.brier_fuera - en_muestra.brier_en_muestra).abs() < 1e-12,
        "medido en muestra no hay brecha: {en_muestra:?}"
    );
    assert!(en_muestra.brier_fuera < r.brier_fuera - 0.03);

    let bien = sintetico(|s| s, 40, T0);
    let escalon = sintetico(|s| if s > 0.55 { 0.8 } else { 0.2 }, 40, T0);
    let b = brier_platt_fuera_de_muestra(&bien, &tardia).map(|r| r.brecha_familia_en_muestra);
    let e = brier_platt_fuera_de_muestra(&escalon, &tardia).map(|r| r.brecha_familia_en_muestra);
    let (Some(b), Some(e)) = (b, e) else {
        panic!("hay evidencia");
    };
    assert!(b >= 0.0 && e >= 0.0, "la isotónica nunca pierde en muestra: {b} {e}");
    assert!(e > 3.0 * b && e > 1e-3, "escalón {e} frente a logística {b}");
    // La descomposición CORP cierra exacta.
    let preds: Vec<(f64, f64)> = bien.iter().map(|&(s, w, _)| (s, f64::from(w))).collect();
    let Some(d) = descomposicion_brier(&preds) else {
        panic!("hay pares");
    };
    assert!((d.brier - (d.fiabilidad - d.resolucion + d.incertidumbre)).abs() < 1e-12);
}

/// 4.4.5 — grupo con menos de 30 ⇒ None (SIN EVIDENCIA), no un Brier
/// fabricado; con 30 sí.
#[test]
fn brier_condicional_por_grupo_sin_evidencia() {
    assert_eq!(MIN_OBS_BRIER, 30);
    let mut preds = Vec::new();
    for k in 0..29 {
        preds.push(("corto", 0.6, f64::from(k % 2 == 0)));
    }
    for k in 0..30 {
        preds.push(("largo", 0.6, f64::from(k % 2 == 0)));
    }
    let g = brier_por_grupo(&preds);
    assert_eq!(g[0], ("corto", None, 29));
    assert_eq!(g[1].0, "largo");
    assert_eq!(g[1].2, 30);
    let Some(bs) = g[1].1 else {
        panic!("30 observaciones dan Brier");
    };
    assert_eq!(bs, brier_score(&preds[29..].iter().map(|&(_, p, y)| (p, y)).collect::<Vec<_>>()).0);
    // Y el Brier fuera de muestra exige el mínimo en las dos partes.
    let pocas = sintetico(|s| s, 4, T0);
    assert!(pocas.len() < MIN_OBS_BRIER);
    assert!(brier_platt_fuera_de_muestra(&pocas, &[(0.5, true); 40]).is_none());
}

/// El diagnóstico del libro publica las dos calibraciones fuera de muestra
/// cuando hay evidencia.
#[test]
fn diagnostico_del_libro_compara_las_dos_calibraciones() {
    let mut libro = LibroDecisiones::new(1);
    let mut vivo = PlattCalibrator::new();
    let _ = poblacion(&mut libro, &mut vivo, true);
    let d = libro.diagnostico();
    assert_eq!((d.observaciones, d.ejecutadas, d.contrafactuales), (100, 40, 60));
    assert!(d.con_contrafactuales.is_some(), "{d:?}");
    // La mitad temprana tiene 40 ejecutadas y 10 contrafactuales.
    assert!(d.solo_ejecutadas.is_some(), "{d:?}");
}
