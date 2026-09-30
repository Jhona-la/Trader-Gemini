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
//!
//! GO (2026-09-30): precisión del alcance de las afirmaciones históricas:
//! se detecta cambio de OCHO ESTADÍSTICAS bajo UN extremo/fixture/predictor,
//! no aptitud canónica ni inercia global. Interiores, interacciones y otros
//! contextos pueden revelar efectos ausentes aquí. from_vector puede borrar
//! o acoplar coordenadas; las trazas siguientes exponen el cambio realizado.
//! Fixture, comparador histórico, predictor y trinquete permanecen intactos.

use backtest_engine::{STATS_LEN, run_backtest_native};
use quantum_arena::genome::SuperGenotype;

#[path = "support/t1_measurement.rs"]
mod measurement;
use measurement::{changed_slots, diagnostic_stats_line, difiere, furthest_endpoint, serie};

/// Evalúa un candidato sobre la serie histórica, sin cambiar sus entradas.
fn evaluar(
    cfg: &SuperGenotype,
    serie: &(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>),
) -> [f64; STATS_LEN] {
    let mut pnl = vec![0.0f64; serie.0.len().max(1)];
    let mut stats = [0.0f64; STATS_LEN];
    run_backtest_native(
        &serie.0, &serie.1, &serie.2, &serie.3, cfg, &mut pnl, &mut stats, "BTCUSDT", 1_000.0,
    );
    stats
}

/// DIAGNÓSTICO (B3.18, 2026-09-16): una SOLA evaluación con el predictor
/// confiado y trazas de los contadores de veto del core — localiza EN QUÉ
/// ETAPA mueren los trades del camino nativo (señales jamás disparadas /
/// consejo / gate ml / risk-engine) sin pagar los 22 min del oráculo.
#[test]
fn t1_diag_camino_nativo_una_evaluacion() {
    // (Ola XLI) El predictor CONSTANTE no puede cooperar con un gate por lift
    // (base 0.953 ⇒ umbrales inalcanzables): es artefacto, no diagnóstico. El
    // direccional (base 0.5, ±3 según signo) es el neutralizador correcto.
    let confident = god_engine_core::ml_inference::NanoForestData {
        children_left: vec![1, -1, -1],
        children_right: vec![2, -1, -1],
        feature: vec![0, -1, -1],
        threshold: vec![0.0, 0.0, 0.0],
        value: vec![0.0, -3.0, 3.0],
        tree_offsets: vec![0, 3],
        init_score: 0.0,
    };
    let forest = god_engine_core::ml_inference::NanoForest::from_data(confident)
        .expect("forest sintético fuera de contrato");
    god_engine_core::ml_inference::NanoForest::store_global("BTCUSDT_MOTOR", forest);
    // Safety: test single-threaded antes de spawn de hilos del runner.
    unsafe { std::env::set_var("TG_TRACE_NATIVO", "1") };

    let datos = serie(3_000);
    let base = SuperGenotype::new_baseline(0.0002, 0.0005);
    let stats = evaluar(&base, &datos);
    println!("{}", diagnostic_stats_line(&stats));
    // Rechazos del risk-engine (contadores globales del proceso): "sin
    // rechazos" ⇒ las SEÑALES jamás produjeron intención; un motivo
    // dominante ⇒ el risk-engine es el estrangulamiento.
    println!("[T1-DIAG] risk-rejects: {}", risk_engine::reject_report());
}

// HISTORIA DEL ORÁCULO (2026-09-16, cuarta medición): tres corridas dieron
// 0/144 (con modelos reales, sin modelos, y con predictor confiado). El
// diagnóstico por etapas (t1_diag + contadores del core + reject_report)
// encontró la CAUSA RAÍZ: el registro dinámico de símbolos nace VACÍO por
// diseño (lo puebla el symbol manager del motor en vivo) y el runner nativo
// jamás registró specs ⇒ evaluate_quantum_order rechazaba TODO con "spec"
// (432/432) ⇒ cero trades ⇒ ningún gen podía diferir. El oráculo llevaba
// muerto desde que el registro se hizo dinámico — NO desde B3.18. Fix B3.20
// en run_backtest_native (registra spec estándar idempotente).
// El PREDICTOR SINTÉTICO SIEMPRE-CONFIADO se mantiene: mide la expresividad
// genética CONDICIONAL a la cooperación de la predicción (el gate B3.18 es
// un gobernador no-genético por diseño; con el predictor cooperando, los
// genes —incluidos los de umbral ml— vuelven a poder expresarse).
/// POST-FUSIÓN PR #5 (2026-09-25, medido): 0/144 — el oráculo vuelve a
/// estar muerto sobre el FIXTURE SINTÉTICO, esta vez por la física de
/// viabilidad del PR: `dynamic_max_spread = (σ(τ)·escala − fricción_ida_y_
/// vuelta).max(tick_pct)` es la compuerta honesta calibrada sobre TAPE REAL
/// (BTC 34M trades, D-751/D-756), pero el fixture sintético simula spread
/// 2×maker_spread_pct = 4 pb con fricción 7 pb y τ dominante corta en frío:
/// σ(τ)−7pb < 0 ⇒ el piso colapsa a tick_pct (0,1/60000 = 0,17 pb) ⇒
/// INVIABLE perpetuo (diag t1_diag: «spread 0.000400 ≤ max 0.000002»,
/// 0 intents al risk-engine). La re-expresión genética exige re-calibrar el
/// fixture o el neutralizador sobre tape REAL — trabajo de la Ola XL, no
/// un revert de la auditoría D-75x. Se ignora con diagnóstico; sólo puede
/// re-activarse subiendo desde una re-medición documentada.
#[test]
fn t1_cobertura_genetica_del_oraculo_de_aptitud() {
    // Neutralización documentada del gate para la MEDICIÓN (ver comentario
    // del test). B3.36 (gate por LIFT sobre base_prob del modelo) invalidó el
    // viejo predictor constante: con init_score=3 su base era 0.953 y el
    // gate largo 0.953+lift > 1.0 — INALCANZABLE por construcción; la
    // medición quedó short-only y 3 genes perdieron expresividad aparente
    // (medido 11.8% vs ratchet 13.5%). Un predictor constante JAMÁS puede
    // cooperar con un gate por lift — esa es exactamente la tesis del lift
    // (un modelo sin información no cruza). El neutralizador correcto bajo
    // lift: un forest DIRECCIONAL con base 0.5 — un split en dim 0
    // (price_change) que predice ±3 según el signo. p varía 0.047/0.953
    // alrededor de base 0.5 y cruza los gates de AMBAS direcciones cuando
    // la señal del genoma apunta: el gate coopera y la medición conserva su
    // espíritu original (expresividad genética condicional a la predicción).
    let confident = god_engine_core::ml_inference::NanoForestData {
        children_left: vec![1, -1, -1],
        children_right: vec![2, -1, -1],
        feature: vec![0, -1, -1],
        threshold: vec![0.0, 0.0, 0.0],
        value: vec![0.0, -3.0, 3.0],
        tree_offsets: vec![0, 3],
        init_score: 0.0, // base 0.5 — lift gate alcanzable en ambas direcciones
    };
    let forest = god_engine_core::ml_inference::NanoForest::from_data(confident)
        .expect("forest sintético del oráculo fuera de contrato");
    // El runner nativo mapea su serie a coin 0; B3.18b resuelve la clave del
    // símbolo con default BTCUSDT cuando el registry no lo registra.
    god_engine_core::ml_inference::NanoForest::store_global("BTCUSDT_MOTOR", forest);
    println!(
        "[T-1] predictor sintético direccional cargado (gate por lift neutralizado para medir)"
    );

    let datos = serie(3_000);
    let base = SuperGenotype::new_baseline(0.0002, 0.0005);
    let lo = SuperGenotype::get_lower_bounds();
    let hi = SuperGenotype::get_upper_bounds();
    let base_vec = base.to_vector();
    let n = base_vec.len();
    assert_eq!(n, SuperGenotype::DIMENSION);

    let roundtrip = SuperGenotype::from_vector(&base_vec).to_vector();
    println!(
        "[T1-PROBE] baseline roundtrip changed_slots={:?}",
        changed_slots(&base_vec, &roundtrip)
    );

    let stats_base = evaluar(&base, &datos);

    let mut inertes: Vec<usize> = Vec::new();
    let mut sensibles = 0usize;

    for g in 0..n {
        let mut v = base_vec.clone();
        // Mismo extremo MÁS LEJANO histórico. No observar diferencia ahí
        // NO prueba inercia para valores interiores o perturbaciones conjuntas.
        v[g] = furthest_endpoint(base_vec[g], lo[g], hi[g]);
        if (v[g] - base_vec[g]).abs() < 1e-12 {
            // Banda degenerada: el gen no puede variar, se cuenta como inerte.
            inertes.push(g);
            continue;
        }
        let mutado = SuperGenotype::from_vector(&v);
        let realized = mutado.to_vector();
        println!(
            "[T1-PROBE] gene={g} requested={} realized={} changed_slots={:?}",
            v[g],
            realized[g],
            changed_slots(&base_vec, &realized)
        );
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
         ({:.1} %)\\n[T-1] Sin cambio observado bajo esta perturbación/fixture: {:?}\\n",
        cobertura * 100.0,
        inertes
    );

    // El umbral no pretende ser ambicioso hoy: pretende ser un TRINQUETE.
    // Historia de la base: 25% (Décima Ola, genoma gobernador único) → 0%
    // (bug B3.20: registro de specs vacío en el runner nativo — el oráculo
    // llevaba muerto desde que el registro se hizo dinámico) → 13.9%
    // (20/144, medido 2026-09-16 con spec registrado + predictor sintético
    // confiado: expresividad CONDICIONAL a la cooperación de la predicción
    // — el gate B3.18 es un gobernador no-genético por diseño) → 11.8%
    // (17/144, medido 2026-09-18 tras B3.36: los genes ml_threshold se
    // reinterpretaron como LIFT sobre base_prob del modelo — la banda de
    // expresividad efectiva se estrechó de [0.51,0.95] (ancho 0.44) a
    // [0.52,0.75] (ancho 0.23) por DISEÑO: es el precio documentado de la
    // invariancia de escala (un cambio de geometría de labels ya no rompe
    // la selectividad). Ningún cableado de genes se retiró: el neutralizador
    // direccional (ver arriba) verifica que no es artefacto del predictor
    // constante — con ambos predictores la medición da el MISMO 11.8%).
    // Dirección de recuperación: conectar genes muertos (D-649), retirar
    // clamps (D-643), o ensanchar la banda de lift con calibración Brier
    // real del evolver. Sólo puede SUBIR desde aquí.
    //
    // RE-CERTIFICACIÓN CONSCIENTE (CL-2, 2026-09-28, autorizada por el
    // operador): 19/144 → 16/144 (11,1 %). Medido gen a gen, sólo cambian
    // tres genes y los tres por el arreglo del freno de apalancamiento:
    //   · 142/143 (`sl_horizon_curve`): su ÚNICA lectura efectiva en el
    //     fixture era el stop imaginario del freno (τ del gen + curva
    //     genómica). La orden usa el stop de `compute_tp_sl`; esa
    //     sensibilidad era falsa y desaparece con el defecto.
    //   · 20 (`veto_threshold_btc`): sigue cableado a la fracción Kelly de la
    //     matriz, pero en el fixture su efecto queda por debajo del paso de
    //     la cuantización entera del apalancamiento (D-730).
    // Ningún cableado se retiró. El trinquete vuelve a ser el valor medido.
    //
    // RE-CERTIFICACIÓN 2 (GLM/LI, 2026-09-30, bajo mandato permanente del
    // operador): 16/144 (11,1 %) → 12/144 (8,3 %). Aislamiento medido:
    // CX-solo = 11,1 % VERDE (XLIX·G, 45,7 min); CX+CL-30..32 = 8,3 %
    // (XLIX·G bis, 46,5 min con lista completa de inertes capturada) ⇒
    // exactamente 4 genes pierden sensibilidad con la persistencia
    // corregida. Mecanismo (mismo patrón que CL-2): la persistencia
    // legada comparaba desviaciones consecutivas de la MISMA EWMA — su
    // acuerdo en caminata aleatoria era (2/π)·asin(e^(−Δt/τ)) ≈ +0,94,
    // tendencia alucinada en todas las escalas; los genes cuya lectura
    // efectiva en el fixture pasaba por esa tendencia fantasma pierden
    // una sensibilidad que era FALSA. Con la persistencia centrada en 0
    // (bloques no solapados ≥ τ, teórico iid −⅓), el motor ve el
    // fixture como lo que es. Activos tras la corrección: [1, 17, 18,
    // 24, 27, 32, 68, 69, 129, 130, 131, 141]. Ningún cableado se
    // retiró. El trinquete vuelve a ser el valor medido.
    // Dirección de recuperación (heredada): conectar genes muertos
    // (D-649), retirar clamps (D-643), calibración Brier del evolver —
    // con las trazas solicitada-vs-realizada del diagnóstico GO, la
    // próxima re-certificación es verificable gen a gen.
    const COBERTURA_MINIMA: f64 = 0.083;
    assert!(
        cobertura >= COBERTURA_MINIMA,
        "cobertura genética {:.1} % por debajo del mínimo {:.1} %. \
         {} coordenadas no cambiaron las estadísticas bajo ESTE extremo y fixture; \
         no es prueba de inercia global ni autorización para bajar el umbral. Sin cambio: {:?}",
        cobertura * 100.0,
        COBERTURA_MINIMA * 100.0,
        inertes.len(),
        inertes
    );
}
