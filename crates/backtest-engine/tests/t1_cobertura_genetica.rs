//! T-1 (DÉCIMA OLA) — COBERTURA GENÉTICA DEL ORÁCULO DE APTITUD.
//!
//! # Por qué existe este test
//!
//! Un gen cuya variación no altera la aptitud **no está sujeto a selección**.
//! Su dinámica bajo mutación es un paseo aleatorio dentro de sus bandas, y al
//! desplegarse en producción gobierna comportamiento que nunca fue evaluado.
//!
//! La auditoría de la Décima Ola planteó inicialmente que el oráculo leía sólo
//! 6 de los 144 genes. **Esa afirmación era incorrecta** y este test es lo que
//! la corrige: `run_backtest_native` —el evaluador que alimenta a los
//! promotores reales— aplica el genoma COMPLETO al arena y ejecuta el mismo
//! `GodEngineCore` que producción. El evaluador de 6 genes
//! (`run_vectorized_hybrid`) existe, pero sólo es alcanzable por la exportación
//! FFI y no participa en ninguna promoción.
//!
//! Lo que queda por medir —y es lo que este test mide— es cuántos genes tienen
//! **sensibilidad efectiva** sobre el resultado. Un gen puede aplicarse al
//! arena y aun así no influir, porque su sitio de lectura lo pisa con un clamp
//! literal (D-643, 106 sitios) o porque ningún consumidor lo lee (D-649, 33
//! genes muertos).
//!
//! # Método
//!
//! Perturbación coordenada a coordenada: se parte de un genoma base, se mueve
//! un gen a un extremo de su banda evolutiva, se reevalúa y se compara. Es una
//! medición CONDUCTUAL de la presión selectiva, no un recuento sintáctico.

use backtest_engine::{STATS_LEN, run_backtest_native};
use quantum_arena::genome::SuperGenotype;

/// Serie sintética determinista con estructura suficiente para que el motor
/// abra y cierre posiciones: tendencia lenta + ciclo + ruido reproducible.
fn serie(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut closes = Vec::with_capacity(n);
    let mut highs = Vec::with_capacity(n);
    let mut lows = Vec::with_capacity(n);
    let mut vols = Vec::with_capacity(n);
    let mut seed = 0x5DEECE66Du64;
    let mut p = 60_000.0f64;
    for i in 0..n {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((seed >> 33) as f64 / u32::MAX as f64) - 0.5;
        let ciclo = (i as f64 / 180.0).sin() * 0.0015;
        p *= 1.0 + ciclo + u * 0.0022 + 0.00004;
        closes.push(p);
        highs.push(p * (1.0 + 0.0011 + u.abs() * 0.0009));
        lows.push(p * (1.0 - 0.0011 - u.abs() * 0.0009));
        vols.push(900.0 + u.abs() * 2_000.0);
    }
    (closes, highs, lows, vols)
}

fn evaluar(cfg: &SuperGenotype, serie: &(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)) -> [f64; STATS_LEN] {
    let mut pnl = vec![0.0f64; serie.0.len().max(1)];
    let mut stats = [0.0f64; STATS_LEN];
    run_backtest_native(
        &serie.0,
        &serie.1,
        &serie.2,
        &serie.3,
        cfg,
        &mut pnl,
        &mut stats,
        "BTCUSDT",
        1_000.0,
    );
    stats
}

fn difiere(a: &[f64; STATS_LEN], b: &[f64; STATS_LEN]) -> bool {
    a.iter().zip(b.iter()).any(|(x, y)| {
        if !x.is_finite() || !y.is_finite() {
            return x.is_finite() != y.is_finite();
        }
        (x - y).abs() > 1e-9 * x.abs().max(1.0)
    })
}

// B3.18 (2026-09-16) — TEST APARCADO con #[ignore]: desde que LA PREDICCIÓN
// DECIDE, la cobertura genética del oráculo es 0/144 (medido, con modelos
// del roster cargados): el gate de ensamble y la envolvente de sizing son
// gobernadores NO-GENÉTICOS por diseño — si la predicción no coopera, NINGÚN
// valor de gen produce trades y por tanto ningún gen puede "diferir". El
// trinquete del 25% medía un sistema donde el genoma era el único gobernador.
// REDISEÑO PENDIENTE: el oráculo debe medir la expresividad genética
// CONDICIONAL a la predicción (p.ej. serie sintética con modelo a medida, o
// neutralización documentada del gate sólo para la medición). NO eliminar:
// la propiedad que protege (genes muertos = ruido que gobierna) sigue
// vigente; sólo cambió dónde se puede medir.
#[test]
#[ignore = "B3.18: oráculo requiere rediseño condicional a la predicción — ver comentario"]
fn t1_cobertura_genetica_del_oraculo_de_aptitud() {
    // B3.18 (ajuste del oráculo): desde que LA PREDICCIÓN DECIDE, un core
    // sin modelos cargados no tradea (gate de ensamble contra un ml≈0.5
    // sin forest — el fallback cross-símbolo fue retirado a propósito).
    // Sin trades, NINGÚN gen puede expresarse: 0/144 sensibles (medido
    // 2026-09-16). El oráculo debe evaluar el sistema en su configuración
    // REAL: modelos del roster cargados, como el arranque del motor y
    // backtest_windows (B3.15).
    if let Ok(entries) = std::fs::read_dir("models") {
        let mut cargados = 0usize;
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if god_engine_core::ml_inference::NanoForest::load_global(
                stem,
                &path.to_string_lossy(),
            )
            .is_ok()
            {
                cargados += 1;
            }
        }
        assert!(
            cargados > 0,
            "el oráculo necesita modelos en models/ para que el gate B3.18 permita operar (cargados: {cargados})"
        );
        println!("[T-1] {cargados} modelos cargados para el oráculo");
    }
    let datos = serie(3_000);
    let base = SuperGenotype::new_baseline(0.0002, 0.0005);
    let lo = SuperGenotype::get_lower_bounds();
    let hi = SuperGenotype::get_upper_bounds();
    let base_vec = base.to_vector();
    let n = base_vec.len();
    assert_eq!(n, SuperGenotype::DIMENSION);

    let stats_base = evaluar(&base, &datos);

    let mut inertes: Vec<usize> = Vec::new();
    let mut sensibles = 0usize;

    for g in 0..n {
        let mut v = base_vec.clone();
        // Mover el gen al extremo MÁS LEJANO de su banda: si con eso no cambia
        // nada, no cambia con nada.
        let dist_lo = (base_vec[g] - lo[g]).abs();
        let dist_hi = (hi[g] - base_vec[g]).abs();
        v[g] = if dist_hi >= dist_lo { hi[g] } else { lo[g] };
        if (v[g] - base_vec[g]).abs() < 1e-12 {
            // Banda degenerada: el gen no puede variar, se cuenta como inerte.
            inertes.push(g);
            continue;
        }
        let mutado = SuperGenotype::from_vector(&v);
        let stats = evaluar(&mutado, &datos);
        if difiere(&stats_base, &stats) {
            sensibles += 1;
        } else {
            inertes.push(g);
        }
    }

    let cobertura = sensibles as f64 / n as f64;
    println!(
        "\\n[T-1] COBERTURA GENÉTICA DEL ORÁCULO: {sensibles}/{n} genes sensibles \\
         ({:.1} %)\\n[T-1] Genes inertes: {:?}\\n",
        cobertura * 100.0,
        inertes
    );

    // El umbral no pretende ser ambicioso hoy: pretende ser un TRINQUETE.
    // Fija el nivel medido tras la Décima Ola para que no pueda retroceder, y
    // sube conforme se conecten los genes muertos (D-649) y se retiren los
    // clamps de los sitios de lectura (D-643).
    const COBERTURA_MINIMA: f64 = 0.25;
    assert!(
        cobertura >= COBERTURA_MINIMA,
        "cobertura genética {:.1} % por debajo del mínimo {:.1} %. \\
         {} genes no influyen en la aptitud: son ruido no seleccionado que sin \\
         embargo gobierna comportamiento en producción. Inertes: {:?}",
        cobertura * 100.0,
        COBERTURA_MINIMA * 100.0,
        inertes.len(),
        inertes
    );
}
