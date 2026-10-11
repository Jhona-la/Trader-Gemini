//! C-F3 / F3.4 (Línea C, consejo): pesos de los asientos y vetos duplicados.
//!
//! Teoría que se prueba:
//! - F3.4-01. El peso de un asiento debe premiar la ESPERANZA de lo que
//!   respalda, no la tasa de acierto. Con geometría asimétrica (RR > 1) una
//!   estrategia con acierto < ½ tiene EV > 0: el asiento que la respalda no
//!   debe pesar menos que el que se opone. Hoy `record_outcome` puntúa el
//!   signo del retorno neto (consejo_seniors.rs, `was_correct`) y
//!   `compute_weights` da acierto/acierto medio recortado a [0,5; 2]: con
//!   acierto 0,35 frente a 0,65 (media 0,5) salen w0 = 0,70 y w1 = 1,30.
//!   Mismo defecto que F2.2-02 (Ville contra ½ en vez del EV en R), en otro
//!   consumidor.
//! - F3.4-02. Un dato, un veto. `do_calculus_risk` (el VPIN vivo) alimenta
//!   dos asientos: Causal veta por encima de 0,75 (veto único: admite override
//!   por supermayoría) y el Auditor por encima de 0,92. Con VPIN > 0,92 hay
//!   dos vetos y el quórum cae en la rama irrevocable: el override desaparece
//!   por duplicación del mismo dato, no por una decisión explícita. Es el
//!   acoplamiento que XLI·D7 retiró para el drawdown. El valor esperado
//!   (approved(0,93) == approved(0,90)) es una POLÍTICA («un dato, un
//!   veto»); si el dueño quiere un techo absoluto de VPIN debe ser una rama
//!   explícita, como Riesgo y Ente.
//!
//! Regla de C-F2 para las rejillas del consejo (FICHAS_C-F2:872): incluir
//! tracker con n ∈ {0, 500} y multiplicadores de Meta ∈ {1, 2}. Las dos
//! pruebas la cumplen.
//!
//! Convención: una prueba que hoy falla por la causa de un hallazgo lleva
//! `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert intacto. Para
//! correrlas: `cargo test -p metacortex-engine --test cf3_consejo_vetos --
//! --include-ignored`.

use metacortex_engine::consejo_seniors::{
    ConsejoDeliberacion, MarketSnapshotPayload, SeniorPerformanceTracker, SeniorRole,
};

// ───────────────────────────── F3.4-01 ─────────────────────────────

/// Pesos del tracker en bruto tras 100 resultados con el asiento 0 a favor
/// de la operación ejecutada y el asiento 1 en contra; patrón determinista de
/// `ganadoras_de_20` ganadoras por cada 20.
fn pesos_tracker(ganancia: f64, perdida: f64, ganadoras_de_20: usize) -> [f64; 11] {
    let mut t = SeniorPerformanceTracker::new(500);
    let mut s = [0.0_f64; 11];
    s[0] = 1.0; // respalda la operación ejecutada
    s[1] = -1.0; // se opone
    for i in 0..100 {
        let r = if i % 20 < ganadoras_de_20 {
            ganancia
        } else {
            -perdida
        };
        t.record_outcome(&s, r);
    }
    t.compute_weights()
}

/// Mismo experimento por la API viva `record_trade_outcome` con un CORTO
/// (pedido del escéptico de completitud): cubre la inversión
/// agreement = señal·lado y la máscara de moduladores. Asientos 0
/// (Microestructura) y 1 (SeriesTemporales), direccionales: el corto lo
/// respalda quien vota −1.
fn pesos_corto_api_viva(ganancia: f64, perdida: f64, ganadoras_de_20: usize) -> [f64; 11] {
    let consejo = ConsejoDeliberacion::new();
    let mut s = [0.0_f64; 11];
    s[SeniorRole::Microestructura as usize] = -1.0; // respalda el corto
    s[SeniorRole::SeriesTemporales as usize] = 1.0; // se opone
    for i in 0..100 {
        let r = if i % 20 < ganadoras_de_20 {
            ganancia
        } else {
            -perdida
        };
        consejo.record_trade_outcome(&s, r, false);
    }
    let pesos = consejo.tracker.read().expect("tracker").compute_weights();
    pesos
}

/// Esperado: con EV > 0, w(respalda) ≥ w(opone); con EV < 0, al revés.
/// Comparación exacta (los pesos salen de cocientes de conteos enteros).
/// Hoy: EV > 0 (RR 2:1, acierto 35 %) da w0 = 0,70 y w1 = 1,30.
#[test]
#[ignore = "OPEN F3.4-01 (Línea C, consejo C-F3): el tracker puntúa el signo del retorno neto (acierto), no su magnitud; con RR > 1 y acierto < ½ (EV > 0) el asiento que respalda pierde peso frente al que se opone (w0 = 0,70, w1 = 1,30)"]
fn cf3_tracker_premia_ev_no_acierto() {
    // RR 2:1 y acierto 35 %: EV = 0,35·0,02 − 0,65·0,01 = +0,0005 por operación.
    let w_pos = pesos_tracker(0.02, 0.01, 7);
    // Espejo: RR 1:3 y acierto 65 %: EV = 0,65·0,005 − 0,35·0,015 = −0,002.
    let w_neg = pesos_tracker(0.005, 0.015, 13);
    let c_pos = pesos_corto_api_viva(0.02, 0.01, 7);
    let c_neg = pesos_corto_api_viva(0.005, 0.015, 13);
    println!(
        "observado (tracker): EV > 0: w0 = {:.3}, w1 = {:.3}; EV < 0: w0 = {:.3}, w1 = {:.3}",
        w_pos[0], w_pos[1], w_neg[0], w_neg[1]
    );
    println!(
        "observado (API viva, corto): EV > 0: w0 = {:.3}, w1 = {:.3}; EV < 0: w0 = {:.3}, w1 = {:.3}",
        c_pos[0], c_pos[1], c_neg[0], c_neg[1]
    );
    assert!(
        w_pos[0] >= w_pos[1],
        "EV > 0: el asiento que respalda pierde peso (w0 = {:.3}, w1 = {:.3})",
        w_pos[0],
        w_pos[1]
    );
    assert!(
        w_neg[0] <= w_neg[1],
        "EV < 0: el asiento que respalda gana peso (w0 = {:.3}, w1 = {:.3})",
        w_neg[0],
        w_neg[1]
    );
    assert!(
        c_pos[0] >= c_pos[1],
        "corto, EV > 0: el asiento que respalda pierde peso (w0 = {:.3}, w1 = {:.3})",
        c_pos[0],
        c_pos[1]
    );
    assert!(
        c_neg[0] <= c_neg[1],
        "corto, EV < 0: el asiento que respalda gana peso (w0 = {:.3}, w1 = {:.3})",
        c_neg[0],
        c_neg[1]
    );
}

// ───────────────────────────── F3.4-02 ─────────────────────────────

/// Payload fijo; sólo cambia `do_calculus_risk`. Largo, OBI 0,2 (< 0,25: sin
/// bypass de ruptura alineada de Causal), fused 0,8, persistencia 0,8, ML 0,6
/// sobre base 0,3, Hurst 0,7, τ = 1 h (s ≈ 0,658), sin deslizamiento.
fn payload(vpin: f64) -> MarketSnapshotPayload {
    MarketSnapshotPayload {
        intended_direction: 1.0,
        book_imbalance: 0.2,
        fused_score: 0.8,
        persistence: 0.8,
        ml_prob: 0.6,
        ml_model_base: 0.3,
        hurst_exponent: 0.7,
        atr_pct: 0.001,
        dominant_tau_ms: 3_600_000.0,
        estimated_slippage_bps: 0.0,
        do_calculus_risk: vpin,
        ..Default::default()
    }
}

/// Consejo con tracker de n resultados neutros (señales nulas: no mueven los
/// pesos, sólo el n del encogimiento de wr; con wr = 0,5 safe_wr = 0,5).
fn consejo_con_n(n: usize) -> ConsejoDeliberacion {
    let c = ConsejoDeliberacion::new();
    for _ in 0..n {
        c.record_outcome(&[0.0; 11], 0.001);
    }
    c
}

/// Esperado: approved(0,93) == approved(0,90) en toda la rejilla
/// n ∈ {0, 500} × Meta ∈ {1, 2}; precondición approved(0,90) = true (un
/// veto único de Causal con supermayoría se sobreescribe). Hoy, con 0,93
/// veta también el Auditor: dos vetos ⇒ rechazo irrevocable.
#[test]
#[ignore = "OPEN F3.4-02 (Línea C, consejo): do_calculus_risk (VPIN) alimenta dos vetos (Causal > 0,75 y Auditor > 0,92); con VPIN > 0,92 hay dos vetos y el override por supermayoría desaparece por duplicación del mismo dato"]
fn cf3_vpin_un_dato_un_veto() {
    let mut rejilla = Vec::new();
    for n in [0usize, 500] {
        let consejo = consejo_con_n(n);
        for meta in [1.0_f64, 2.0] {
            let mut mult = [1.0_f64; 11];
            mult[SeniorRole::Metacognitivo as usize] = meta;
            let r090 = consejo.deliberar_with_weights(&payload(0.90), 0.5, Some(&mult));
            let r093 = consejo.deliberar_with_weights(&payload(0.93), 0.5, Some(&mult));
            let vetos_093: Vec<SeniorRole> = r093
                .dissenting_log
                .iter()
                .filter(|o| o.is_veto)
                .map(|o| o.role)
                .collect();
            println!(
                "observado: n = {n}, Meta ×{meta}: VPIN 0,90 → approved {} (vetado por {:?}, final {:.3}); \
                 VPIN 0,93 → approved {} (vetado por {:?}, vetos {:?})",
                r090.approved, r090.vetoed_by, r090.final_signal, r093.approved, r093.vetoed_by, vetos_093
            );
            rejilla.push((n, meta, r090, r093, vetos_093));
        }
    }
    for (n, meta, r090, r093, vetos_093) in &rejilla {
        assert!(
            r090.approved,
            "precondición (n = {n}, Meta ×{meta}): un veto único (Causal) con supermayoría se sobreescribe; vetado por {:?}",
            r090.vetoed_by
        );
        assert_eq!(
            r093.approved, r090.approved,
            "n = {n}, Meta ×{meta}: VPIN 0,93 veta dos veces ({vetos_093:?}) y el override deja de existir"
        );
    }
}
