//! C-F2.5 (Línea C): conducta de masas contra soluciones conocidas.
//!
//! Teoría que se prueba:
//! - Kahneman-Tversky (1992): la función de valor V(x) = x^α (x ≥ 0) y
//!   V(x) = −λ(−x)^β (x < 0) con α = β = 0,88 y λ = 2,25, y la ponderación
//!   w(p) = p^γ/(p^γ + (1 − p)^γ)^(1/γ), con su S invertida (un solo cruce
//!   con la diagonal en (0,01; 0,99)).
//! - El asiento EnteMercado del consejo: su convicción es el producto de
//!   factores de masa (ballena, cascada, OI, spoofing, LS, taker, staleness y
//!   Prospect) recortado a [0,20; 1]. Se prueba que el impulso contrarian de
//!   Prospect llega a la convicción (F2.5-03), que un mismo dato de masa
//!   entra una sola vez (F2.5-02) y por qué canal decide el Ente (F2.5-01):
//!   sin vetos es inerte por construcción (identidad exacta de la
//!   agregación), y con un veto único decide el suelo del override
//!   (|final_signal| ≥ 0,28), porque final_signal incluye a los moduladores.
//! - La presión de masa de Prospect no tiene punto de referencia (F2.5-06):
//!   P(LS, 0, δ)/δ^0,88 no depende de δ.
//!
//! Son pruebas de solución conocida de la fase C-F2
//! (docs/audit/HALLAZGOS_C-F2_2026-10-11.md). Convención: un test que hoy
//! falla por la causa de un hallazgo lleva `#[ignore = "OPEN F2.x-nn (dueño):
//! causa"]` con el assert intacto; documenta el defecto y se activa con el
//! arreglo. Para correrlos: `cargo test ... -- --include-ignored`.

use metacortex_engine::consejo_seniors::{
    ConsejoDeliberacion, MarketSnapshotPayload, SeniorAgent, SeniorEnteMercado, SeniorRole,
};
use metacortex_engine::prospect_theory::ProspectTheoryEngine;

/// LCG de 64 bits (constantes de Knuth/MMIX) con salida de 53 bits en (0, 1).
struct Lcg(u64);
impl Lcg {
    fn unif(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

/// Carga útil neutra para el Ente: todos sus factores valen exactamente 1.
fn payload_ente_neutro(dir: f64) -> MarketSnapshotPayload {
    MarketSnapshotPayload {
        intended_direction: dir,
        ..Default::default()
    }
}

/// Ente extremo adverso a `dir`: cada factor cae a su suelo y el producto
/// (0,6·0,8·0,8·0,7·0,75·0,4·0,5 ≈ 0,020) se recorta a 0,20.
fn ente_extremo(base: &MarketSnapshotPayload) -> MarketSnapshotPayload {
    let dir = base.intended_direction;
    let adverso = if dir > 0.0 { 20.0 } else { 0.05 };
    MarketSnapshotPayload {
        whale_burst_z: 10.0,
        open_interest_norm: 1.0,
        spoof_score: 1.0,
        crowd_ls_ratio: adverso,
        crowd_taker_ratio: adverso,
        macro_staleness_ms: 3_600_000,
        prospect_pressure: 50.0 * dir,
        ..base.clone()
    }
}

/// Rejilla de cargas útiles del auditor (sin campos del Ente): OBI, ML,
/// espectro, Hurst, do-calculus, DD ≤ 0,3, slippage ≤ 100 pb, τ
/// log-uniforme en [30 s; 12 h], dir ±1 y liq ≤ 0,85 (sin cascada).
fn payload_aleatoria(rng: &mut Lcg) -> MarketSnapshotPayload {
    let dir = if rng.unif() < 0.5 { 1.0 } else { -1.0 };
    MarketSnapshotPayload {
        book_imbalance: 2.0 * rng.unif() - 1.0,
        hurst_exponent: rng.unif(),
        ml_prob: rng.unif(),
        fused_score: 2.0 * rng.unif() - 1.0,
        persistence: 2.0 * rng.unif() - 1.0,
        atr_pct: 0.001 + 0.01 * rng.unif(),
        loss_streak: (4.0 * rng.unif()) as u32,
        intended_direction: dir,
        do_calculus_risk: rng.unif(),
        current_drawdown_pct: 0.3 * rng.unif(),
        estimated_slippage_bps: 100.0 * rng.unif(),
        dominant_tau_ms: 30_000.0 * 1_440f64.powf(rng.unif()),
        liquidation_severity: 0.85 * rng.unif(),
        ml_model_base: 0.2 + 0.3 * rng.unif(),
        ..Default::default()
    }
}

/// Consejo con 500 resultados en el tracker (el tope de la ventana), como en
/// vivo tras unas horas (god-engine-core lib.rs:4244 alimenta record_trade_outcome).
/// Con n = 500, safe_wr = (wr·500 + 4)/508 ya no es 0,5 fijo
/// (consejo_seniors.rs:1119). Las señales sembradas dan exactitudes distintas
/// por asiento, así que compute_weights devuelve pesos ≠ 1 (en [0,5; 2,0]).
fn consejo_con_historial(rng: &mut Lcg) -> ConsejoDeliberacion {
    let consejo = ConsejoDeliberacion::new();
    // Exactitud sembrada por asiento direccional: Meta acierta casi siempre,
    // el resto poco. Causal, Ejecución y Auditor votan 0 en vivo.
    let exactitud = [
        (SeniorRole::Microestructura, 0.35),
        (SeniorRole::SeriesTemporales, 0.40),
        (SeniorRole::Ml, 0.30),
        (SeniorRole::Metacognitivo, 0.95),
        (SeniorRole::Teleonomia, 0.45),
    ];
    for _ in 0..500 {
        let ret: f64 = if rng.unif() < 0.5 { 0.01 } else { -0.01 };
        let mut s = [0.0f64; 11];
        for &(rol, p) in &exactitud {
            let acierta = rng.unif() < p;
            s[rol as usize] = if acierta { ret.signum() } else { -ret.signum() };
        }
        consejo.record_outcome(&s, ret);
    }
    consejo
}

/// TK92: V(x) = x^0,88 si x ≥ 0 y −2,25·(−x)^0,88 si x < 0.
/// Tolerancia relativa 1e-12: son las mismas operaciones IEEE (powf y un
/// producto). V(4) = 3,387 ≠ 4 comprueba que α se ejerce (el test existente
/// sólo mira x = ±1, donde x^α = 1 para cualquier α). V(150) = 100^0,88 por el
/// recorte de dominio documentado (prospect_theory.rs:91).
#[test]
fn cf2_kt_funcion_de_valor_forma_cerrada() {
    let kt = ProspectTheoryEngine::new();
    let v_cerrada = |x: f64| {
        if x >= 0.0 {
            x.powf(0.88)
        } else {
            -2.25 * (-x).powf(0.88)
        }
    };
    for i in 0..=2000 {
        let x = -100.0 + 0.1 * i as f64;
        let (v, e) = (kt.value(x), v_cerrada(x));
        assert!(
            (v - e).abs() <= 1e-12 * e.abs().max(1.0),
            "x={x}: {v} frente a {e}"
        );
    }
    assert!((kt.value(4.0) - 4f64.powf(0.88)).abs() < 1e-12);
    assert!(
        (kt.value(4.0) - 4.0).abs() > 0.5,
        "alfa no se ejerce: V(4) = {}",
        kt.value(4.0)
    );
    assert_eq!(kt.value(150.0), 100f64.powf(0.88));
}

/// TK92: w(p) = p^γ/(p^γ + (1 − p)^γ)^(1/γ) con γ ∈ {0,61; 0,65; 0,69}
/// (0,61 ganancias y 0,69 pérdidas en TK92). Error absoluto ≤ 1e-12 porque
/// la implementación hace las mismas operaciones en el mismo orden
/// (prospect_theory.rs:114-120); la rejilla k/1000 queda fuera de los
/// recortes en 1e-6. S invertida: w(p) − p cambia de signo una sola vez en
/// (0,01; 0,99).
#[test]
fn cf2_tk92_ponderacion_forma_cerrada() {
    let w_cerrada = |p: f64, g: f64| {
        let (a, b) = (p.powf(g), (1.0 - p).powf(g));
        a / (a + b).powf(1.0 / g)
    };
    for &g in &[0.65, 0.61, 0.69] {
        let kt = ProspectTheoryEngine::with_params(2.25, 0.88, 0.88, g);
        for k in 1..1000 {
            let p = k as f64 / 1000.0;
            let (w, e) = (kt.probability_weight(p), w_cerrada(p, g));
            assert!((w - e).abs() <= 1e-12, "g={g} p={p}: {w} frente a {e}");
        }
        let mut cruces = 0;
        let mut prev = kt.probability_weight(0.01) - 0.01;
        for k in 2..100 {
            let p = k as f64 / 100.0;
            let d = kt.probability_weight(p) - p;
            if d.signum() != prev.signum() {
                cruces += 1;
            }
            prev = d;
        }
        assert_eq!(cruces, 1, "S invertida, g={g}");
    }
}

/// Con la carga neutra todos los factores del Ente valen 1, así que su
/// convicción es exactamente el factor de Prospect. Con dir = +1 y P = −4
/// (pánico de la masa, contrarian a favor), modulation_factor = 1 + 0,3·tanh(1)
/// = 1,22848 (tolerancia 1e-12, misma fórmula). La teoría del asiento
/// promete una amplificación en [1,0; 1,30]: la confianza con P = −4 debe
/// superar a la neutra. Hoy el producto se recorta a 1,0
/// (consejo_seniors.rs:615) y las dos valen 1,0.
#[test]
fn cf2_ente_el_impulso_contrarian_de_prospect_llega_a_la_conviccion() {
    let kt = ProspectTheoryEngine::new();
    let factor = kt.modulation_factor(1.0, -4.0);
    assert!(
        (factor - (1.0 + 0.30 * 1.0f64.tanh())).abs() < 1e-12,
        "factor {factor}"
    );
    let ente = SeniorEnteMercado;
    let neutro = ente.evaluate(&payload_ente_neutro(1.0), 0.5).confidence;
    let mut panico = payload_ente_neutro(1.0);
    panico.prospect_pressure = -4.0;
    let c = ente.evaluate(&panico, 0.5).confidence;
    assert!(
        c > neutro,
        "factor {factor} recortado en consejo_seniors.rs:615: confianza con P=-4 {c} = neutra {neutro}"
    );
}

/// Un mismo dato de masa, una sola entrada. En vivo, crowd_ls_ratio y la
/// presión de respaldo salen del MISMO ls_account_ratio (god-engine-core
/// lib.rs:7692-7695 y :7744-7749, con δ = (atr_pct·100).clamp(0,1; 10)).
/// Sólo con LS = 2 la confianza es p_crowd = 1/(1 + 0,25·ln²2) = 0,892767
/// (tolerancia 1e-12, misma fórmula de consejo_seniors.rs:592-593). Con la
/// presión de respaldo P(2, 0, 0,2) = +0,1066 debe ser idéntica (1e-12),
/// porque no hay información nueva. Hoy el factor de Prospect
/// 1 + 0,5·tanh(−0,0267) = 0,98667 la baja a 0,88087.
///
/// Revisión cruzada (CL-65): es un CONTRATO DEL ASIENTO. Sólo se pone verde si
/// el arreglo va al asiento (ignorar P derivada del mismo LS). Si va al núcleo
/// (no derivar P del mismo ls_account_ratio en lib.rs:7727-7753), esta prueba
/// se sustituye por otra sobre el MarketSnapshotPayload que arma
/// GodEngineCore, en el mismo commit.
#[test]
fn cf2_ente_un_dato_de_masa_una_entrada() {
    let kt = ProspectTheoryEngine::new();
    let ente = SeniorEnteMercado;
    let (ls, atr_pct) = (2.0f64, 0.002f64);
    let delta = (atr_pct * 100.0).clamp(0.1, 10.0);
    let p = kt.compute_crowd_net_prospect_pressure(ls, 0.0, delta);
    assert!(p > 0.0, "presión de respaldo {p}");
    let mut solo_ls = payload_ente_neutro(1.0);
    solo_ls.crowd_ls_ratio = ls;
    let mut como_en_vivo = solo_ls.clone();
    como_en_vivo.prospect_pressure = p;
    let p_crowd = 1.0 / (1.0 + 0.25 * ls.ln().powi(2));
    let a = ente.evaluate(&solo_ls, 0.5).confidence;
    assert!((a - p_crowd).abs() < 1e-12, "sólo LS: {a} frente a {p_crowd}");
    let b = ente.evaluate(&como_en_vivo, 0.5).confidence;
    assert!(
        (a - b).abs() < 1e-12,
        "LS contado dos veces: sólo LS {a} frente a LS + respaldo P={p} {b}"
    );
}

/// Identidad exacta de la agregación (consejo_seniors.rs:1166-1331): sin
/// ningún veto, la aprobación exige consenso direccional, directional_signal
/// (que excluye al Ente) con el signo de intended_direction, y final_signal
/// con ese mismo signo. Los tres moduladores (Volatilidad, Riesgo, Ente)
/// votan intended_direction con confianza > 0, así que si D tiene el signo de
/// dir, final_signal también, sea cual sea la convicción del Ente, en las
/// cargas APROBADAS. Por tanto approved coincide EXACTAMENTE entre una carga
/// y su gemela con el Ente extremo adverso, para cualquier tracker y pesos, y
/// el signo de final_signal coincide en las aprobadas. Revisión cruzada
/// (CL-65): en una carga NO aprobada con D·dir ∈ (−(M_otros + c_base·w),
/// −(M_otros + 0,2·w)) el Ente puede invertir el signo de final_signal; no
/// es identidad y ya no se exige.
///
/// Corrección de los escépticos: la versión del auditor usaba sólo un consejo
/// nuevo (n = 0 ⇒ safe_wr = 0,5 y pesos 1 sea cual sea el wr), y su
/// «inercia salvo cascada» es falsa con un veto único (ver
/// cf2_ente_decide_por_el_suelo_del_override). Aquí la rejilla se limita a
/// cargas SIN vetos y recorre los regímenes n ∈ {0; 500}, pesos del tracker y
/// multiplicadores explícitos en [0,5; 2,0] (Meta en {1; 2}). Se exige un
/// mínimo de casos sin veto y de aprobaciones para que la rejilla ejerza el
/// camino (no es tolerancia estadística: la igualdad es exacta).
#[test]
fn cf2_ente_inerte_sin_vetos() {
    let mut rng = Lcg(0xC0F2_0025);
    let fria = ConsejoDeliberacion::new();
    let viva = consejo_con_historial(&mut rng);
    let pesos_vivos = viva.tracker.read().expect("tracker").compute_weights();
    assert!(
        pesos_vivos[SeniorRole::Metacognitivo as usize] > 1.5,
        "el tracker sembrado no da el régimen vivo: {pesos_vivos:?}"
    );
    let (mut sin_veto, mut aprobadas) = (0usize, 0usize);
    for caso in 0..8_000usize {
        let base = payload_aleatoria(&mut rng);
        let extremo = ente_extremo(&base);
        let wr = rng.unif();
        let mut w = [1.0f64; 11];
        for x in w.iter_mut() {
            *x = 0.5 + 1.5 * rng.unif();
        }
        w[SeniorRole::Metacognitivo as usize] = if rng.unif() < 0.5 { 1.0 } else { 2.0 };
        let (consejo, pesos) = match caso % 4 {
            0 => (&fria, None),
            1 => (&fria, Some(&w)),
            2 => (&viva, None),
            _ => (&viva, Some(&w)),
        };
        let traza = consejo.deliberar_traced(&base, wr, pesos);
        assert!(!traza.opinions.is_empty(), "fallo de integridad: {:?}", traza.consensus.vetoed_by);
        if traza.opinions.iter().any(|o| o.is_veto) {
            continue;
        }
        sin_veto += 1;
        let a = traza.consensus;
        let b = consejo.deliberar_with_weights(&extremo, wr, pesos);
        assert_eq!(
            a.approved, b.approved,
            "caso {caso}: el Ente cambia la aprobación sin veto ({} frente a {})",
            a.final_signal, b.final_signal
        );
        if a.approved {
            assert_eq!(
                (a.final_signal > 0.0, a.final_signal < 0.0),
                (b.final_signal > 0.0, b.final_signal < 0.0),
                "caso {caso}: signo de final_signal {} frente a {}",
                a.final_signal,
                b.final_signal
            );
        }
        aprobadas += a.approved as usize;
    }
    println!("cargas sin veto {sin_veto}, aprobadas {aprobadas}, pesos del tracker vivo {pesos_vivos:?}");
    assert!(
        sin_veto >= 2_000 && aprobadas >= 100,
        "rejilla sin ejercer: {sin_veto} cargas sin veto, {aprobadas} aprobadas"
    );
}

/// Contraejemplo exacto del escéptico de F2.5 (lente refutar), en el régimen
/// vivo: 500 resultados en el tracker, wr = 0,8 ⇒ safe_wr = 404/508, y
/// Metacognitivo ×2 (tope de compute_weights, consejo_seniors.rs:1507).
/// Carga: un único veto, el Causal (do-calculus 0,80 > 0,75), con evidencia
/// direccional fina (sólo el espectro, 0,05·tanh(1)). Forma cerrada, con e la
/// convicción del Ente:
///   s = 0,05·tanh(1); s_meta = 0,4·s/3; w_meta = 2·(1 + (wr − 0,2)/0,8);
///   f(e) = (s² + 0,4·0,9 + 0,4 + s_meta·wr·w_meta + e)
///          / (s + 0,4·0,9 + 0,4 + wr·w_meta + e).
/// f(1) = 0,38834 ≥ 0,28: override, se aprueba con final 0,75·f(1).
/// f(0,2) = 0,25861 < 0,28: se mantiene el veto Causal.
/// Tolerancia 1e-12: mismas operaciones salvo el orden de la suma.
///
/// El canal es estrecho: exige un veto único, consenso ≥ 0,8 y una
/// final_signal cerca del suelo, lo que sólo ocurre con evidencia direccional
/// fina y mucha capacidad del Metacognitivo. En la rejilla aleatoria del
/// auditor (20 000 cargas de payload_aleatoria con liq ≤ 0,85) el Ente
/// extremo no cambia ninguna aprobación en ninguno de cinco regímenes (n = 0
/// con pesos 1 o Meta ×3; n = 500 con pesos del tracker, Meta ×2 o Meta ×3),
/// medido en 1d1f8bdd.
#[test]
fn cf2_ente_decide_por_el_suelo_del_override() {
    let consejo = ConsejoDeliberacion::new();
    for _ in 0..500 {
        consejo.record_outcome(&[0.0; 11], 0.0);
    }
    let mut w = [1.0f64; 11];
    w[SeniorRole::Metacognitivo as usize] = 2.0;
    let base = MarketSnapshotPayload {
        book_imbalance: 0.0,
        hurst_exponent: 0.5,
        ml_prob: 0.30,
        ml_model_base: 0.30,
        fused_score: 0.05,
        persistence: 0.5,
        // F1-008 (Ω151): a la τ por defecto (≈ 19 min, s ≈ 0,5) las bandas de
        // la volatilidad se ensanchan ×2,25 (calma 22,5 pb, peligro 90 pb).
        // Con 100 pb su convicción sigue saturada en 0,4, como la de 50 pb
        // con las bandas base: es el 0,4·0,9 de la forma cerrada.
        atr_pct: 0.010,
        loss_streak: 3,
        intended_direction: 1.0,
        do_calculus_risk: 0.80,
        ..Default::default()
    };
    let extremo = MarketSnapshotPayload {
        whale_burst_z: 10.0,
        open_interest_norm: 1.0,
        spoof_score: 1.0,
        crowd_ls_ratio: 20.0,
        crowd_taker_ratio: 20.0,
        macro_staleness_ms: 3_600_000,
        prospect_pressure: 50.0,
        ..base.clone()
    };
    let wr = (0.8 * 500.0 + 0.5 * 8.0) / (500.0 + 8.0);
    let s = 0.05 * 1.0f64.tanh();
    let s_meta = s / 3.0 * 0.4;
    let w_meta = 2.0 * (1.0 + (wr - 0.20) / 0.80);
    let f = |e: f64| {
        (s * s + 0.4 * 0.9 + 0.4 + s_meta * wr * w_meta + e) / (s + 0.4 * 0.9 + 0.4 + wr * w_meta + e)
    };
    let (f1, f02) = (f(1.0), f(0.2));
    assert!(f1 >= 0.28 && f02 < 0.28, "forma cerrada: f(1) {f1}, f(0,2) {f02}");
    let a = consejo.deliberar_with_weights(&base, 0.8, Some(&w));
    let b = consejo.deliberar_with_weights(&extremo, 0.8, Some(&w));
    assert!(
        a.approved && a.vetoed_by.is_none() && (a.final_signal - 0.75 * f1).abs() < 1e-12,
        "override esperado con el Ente neutro: aprobada {}, veto {:?}, final {} frente a {}",
        a.approved,
        a.vetoed_by,
        a.final_signal,
        0.75 * f1
    );
    assert!(
        !b.approved && b.vetoed_by == Some(SeniorRole::Causal),
        "el Ente extremo debía sostener el veto Causal (f = {f02}): aprobada {}, veto {:?}",
        b.approved,
        b.vetoed_by
    );
}

/// Contraejemplo del escéptico de completitud de F2.5: consejo nuevo
/// (safe_wr = 0,5), Metacognitivo ×3 (el clamp del consejo admite 0,1..5),
/// veto Causal único y evidencia direccional fina (OBI 0,05; ML en zona
/// muerta; Teleonomía con utilidad 0,08·0,4·0,76). Forma cerrada:
///   s_meta = 0,4·0,05/3; w_meta = 3·1,375; u = 0,08·0,4·0,76;
///   f(e) = (0,05·1,2 + 0,36 + 0,1 + s_meta·0,5·w_meta + u² + e)
///          / (0,05·1,2 + 0,36 + 0,1 + 0,5·w_meta + u + e).
/// f(1) = 0,42540 (override, aprueba) y f(0,2) = 0,26163 (veto Causal).
/// Tolerancia 1e-12 frente a la forma cerrada.
#[test]
fn cf2_ente_decide_el_override_de_un_veto_unico() {
    let consejo = ConsejoDeliberacion::new();
    let mut w = [1.0f64; 11];
    w[SeniorRole::Metacognitivo as usize] = 3.0;
    let base = MarketSnapshotPayload {
        book_imbalance: 0.05,
        ml_prob: 0.54,
        ml_model_base: 0.50,
        // F1-008 (Ω151): a la τ por defecto (≈ 19 min, s ≈ 0,5) las bandas de
        // la volatilidad se ensanchan ×2,25 (calma 22,5 pb, peligro 90 pb).
        // Con 100 pb su convicción sigue saturada en 0,4, como la de 50 pb
        // con las bandas base: es el 0,4·0,9 de la forma cerrada.
        atr_pct: 0.010,
        loss_streak: 18,
        intended_direction: 1.0,
        do_calculus_risk: 0.80,
        ..Default::default()
    };
    let adverso = MarketSnapshotPayload {
        whale_burst_z: 10.0,
        open_interest_norm: 1.0,
        spoof_score: 1.0,
        crowd_ls_ratio: 20.0,
        crowd_taker_ratio: 20.0,
        macro_staleness_ms: 3_600_000,
        ..base.clone()
    };
    let s_meta = 0.05 / 3.0 * 0.4;
    let w_meta = 3.0 * (1.0 + (0.5 - 0.20) / 0.80);
    let u = ((0.54 - 0.50) * 2.0) * 0.4 * (1.0 - 0.30 * 0.80);
    let f = |e: f64| {
        (0.05 * 1.2 + 0.36 + 0.1 + s_meta * 0.5 * w_meta + u * u + e)
            / (0.05 * 1.2 + 0.36 + 0.1 + 0.5 * w_meta + u + e)
    };
    let (f1, f02) = (f(1.0), f(0.2));
    assert!(f1 >= 0.28 && f02 < 0.28, "forma cerrada: f(1) {f1}, f(0,2) {f02}");
    let a = consejo.deliberar_with_weights(&base, 0.5, Some(&w));
    let b = consejo.deliberar_with_weights(&adverso, 0.5, Some(&w));
    assert!(
        a.approved && (a.final_signal - 0.75 * f1).abs() < 1e-12,
        "override esperado: aprobada {}, veto {:?}, final {} frente a {}",
        a.approved,
        a.vetoed_by,
        a.final_signal,
        0.75 * f1
    );
    assert!(
        !b.approved && b.vetoed_by == Some(SeniorRole::Causal),
        "el Ente adverso debía sostener el veto Causal (f = {f02}): aprobada {}, veto {:?}",
        b.approved,
        b.vetoed_by
    );
}

/// Homogeneidad de la presión de masa (F2.5-06). Con α = β = 0,88, V(±δ) =
/// ±{1; 2,25}·δ^0,88, así que con liq = 0:
///   P(LS, 0, δ) = δ^0,88·{[w(x) − 2,25·w((1 − x)/2)] − [w(1 − x) − 2,25·w(x/2)]},
/// con x = LS/(1 + LS) y w la ponderación TK92 (γ = 0,65). P/δ^0,88 no depende
/// de δ: la presión es una transformación fija de ln LS en unidades
/// arbitrarias, sin precio de referencia (dos masas con el mismo LS y distinto
/// coste medio no se pueden distinguir). Esta prueba de forma cerrada pasa hoy
/// y documenta la degeneración; el sucesor con punto de referencia (rotación
/// de Grinblatt-Han, según el escéptico de completitud) tendrá que
/// sustituirla. Tolerancia absoluta 1e-12: mismas potencias, sumas en otro
/// orden (|P| ≤ 3,25·δ^0,88 < 10, error de redondeo ~1e-15).
#[test]
fn cf2_kt_presion_de_masa_sin_punto_de_referencia() {
    let kt = ProspectTheoryEngine::new();
    let w = |p: f64| {
        let (a, b) = (p.powf(0.65), (1.0 - p).powf(0.65));
        a / (a + b).powf(1.0 / 0.65)
    };
    for &ls in &[0.5f64, 2.0, 5.0] {
        let x = ls / (1.0 + ls);
        let g = (w(x) - 2.25 * w((1.0 - x) * 0.5)) - (w(1.0 - x) - 2.25 * w(x * 0.5));
        for &delta in &[0.1f64, 0.3, 1.0, 3.0] {
            let p = kt.compute_crowd_net_prospect_pressure(ls, 0.0, delta);
            let escala = delta.powf(0.88);
            assert!(
                (p - escala * g).abs() < 1e-12,
                "LS={ls} delta={delta}: P {p} frente a la forma cerrada {}",
                escala * g
            );
            assert!(
                (p / escala - g).abs() < 1e-12,
                "LS={ls}: P/delta^0,88 = {} depende de delta={delta} (g = {g})",
                p / escala
            );
        }
    }
}
