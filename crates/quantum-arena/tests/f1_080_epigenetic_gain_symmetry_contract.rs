//! Contrato Formal de Verificación F1-080:
//! Ganancia Epigenética Espectral: Gradiente Exponenciado Simétrico, Paridad de EV y Homeostasis por Escala.
//!
//! # QUÉ:
//! Erradicación de la asimetría arbitraria de literales (0.06 boost vs 0.10 penalty, ratio 1.67x,
//! equilibrio espurio que exigía WR >= 62.5% para no colapsar al suelo 0.20) en `apply_epigenetic_outcome`,
//! sustitución por un gradiente multiplicativo/exponenciado simétrico en el espacio de Hilbert,
//! homeostásis dependiente de la escala tau (tau_homeo = max(16 * tau, 30 min)) y protección de
//! horizonte de apertura en el camino de cierre del núcleo (`tau_de_la_posicion > 0`).
//!
//! # POR QUÉ:
//! En estrategias asimétricas de trading cuantitativo de alto beneficio/riesgo (RR >= 2.25) con
//! win rate realista (35% - 50%) y esperanza matemática positiva (EV > 0), la fórmula anterior
//! castigaba sistemáticamente a las escalas más rentables, reduciendo su ganancia al suelo 0.20.
//! Además, la homeostasis fija de 30 minutos borraba la memoria de escalas lentas (4 horas) antes
//! de que un solo trade concluyera, mientras que posiciones adoptadas sin horizonte decisionable
//! contaminaban las ganancias espectrales (F3.2-06).
//!
//! # PARA QUÉ:
//! Garantizar convergencia matemática exacta, preservación de martingala bajo ruido simétrico,
//! amplificación de escalas con ventaja demostrada (EV > 0) independientemente del win rate,
//! y preservación de memoria espectral coherente con cada longitud de onda.
//!
//! # CÓMO:
//! 1. `delta_ln_gain = eta * kernel * r_norm`, con r_norm = clamp(pnl_pct / 0.01, -3.0, 3.0).
//! 2. `g_new = clamp(g_old * exp(delta_ln_gain), 0.20, 3.00)`.
//! 3. `kernel = exp(-0.5 * (delta_ln / ln(4))^2)`.
//! 4. `tau_homeo = max(16.0 * tau, 1_800_000.0)`.
//! 5. `god-engine-core` ejecuta `apply_epigenetic_outcome` solo si `tau_de_la_posicion > 0`.
//!
//! # QUIÉN:
//! Consejo de 10 Roles Senior / Quant Sr. — Ola Ω184 (Cierre F1-080 / F3.2-06).

use quantum_arena::temporal_spectrum::{TemporalSpectrum, SPECTRUM_SCALES_MS};

/// Contrato 1: Martingala Exacta y Simetría ante Retornos Iguales (+x, -x).
/// Una secuencia de retornos opuestos preserva el valor de la ganancia sin deriva descendente espuria.
#[test]
fn f1_080_contrato_1_martingala_exacta_ante_retornos_simetricos() {
    let mut spec = TemporalSpectrum::new();
    let tau_target = SPECTRUM_SCALES_MS[18]; // ~60 s
    let g_inicial = spec.scales[18].epigenetic_gain;
    assert_eq!(g_inicial, 1.0);

    // 50 pares de trades simétricos: +1.5% y -1.5%
    for _ in 0..50 {
        spec.apply_epigenetic_outcome(tau_target, true, 0.015);
        spec.apply_epigenetic_outcome(tau_target, false, -0.015);
    }

    let g_final = spec.scales[18].epigenetic_gain;
    assert!(
        (g_final - g_inicial).abs() < 1e-12,
        "La ganancia final ({g_final}) debe coincidir analíticamente con la inicial ({g_inicial}) tras retornos simétricos"
    );
}

/// Contrato 2: Estrategia de Esperanza Positiva (EV > 0) con Win Rate Bajo (40%) Amplifica Ganancia.
/// Con WR = 40%, ganando +2.5% y perdiendo -1.0% (EV = 0.40 * 2.5 - 0.60 * 1.0 = +0.40 > 0),
/// la ganancia DEBE aumentar por encima de 1.0 (en el código viejo caía al suelo 0.20).
#[test]
fn f1_080_contrato_2_estrategia_ev_positivo_wr_bajo_amplifica_ganancia() {
    let mut spec = TemporalSpectrum::new();
    let tau_target = SPECTRUM_SCALES_MS[18];

    // Simular 20 ciclos de 4 wins (+2.5%) y 6 losses (-1.0%) = 40% WR, EV > 0
    for _ in 0..20 {
        for _ in 0..4 {
            spec.apply_epigenetic_outcome(tau_target, true, 0.025);
        }
        for _ in 0..6 {
            spec.apply_epigenetic_outcome(tau_target, false, -0.010);
        }
    }

    let g_final = spec.scales[18].epigenetic_gain;
    assert!(
        g_final > 1.0,
        "Una estrategia con EV positivo debe incrementar su ganancia espectral (obtenido: {g_final} <= 1.0)"
    );
    assert!(g_final <= 3.00, "Debe respetar el techo de seguridad de 3.00");
}

/// Contrato 3: Localización Gaussiana Espectral en Malla Diádica con sigma = ln(4).
#[test]
fn f1_080_contrato_3_localizacion_gaussiana_en_malla_diadica() {
    let mut spec = TemporalSpectrum::new();
    let idx = 18;
    let tau_target = SPECTRUM_SCALES_MS[idx];

    // Aplicar un retorno ganador significativo
    spec.apply_epigenetic_outcome(tau_target, true, 0.020);

    let g_target = spec.scales[idx].epigenetic_gain;
    let g_adj_1 = spec.scales[idx + 1].epigenetic_gain;
    let g_adj_2 = spec.scales[idx + 2].epigenetic_gain;
    let g_far = spec.scales[idx + 5].epigenetic_gain;

    // La escala resonante exacta debe experimentar la máxima amplificación
    assert!(g_target > g_adj_1);
    assert!(g_adj_1 > g_adj_2);
    assert!(g_adj_2 > g_far);

    // Las escalas alejadas (5 saltos diádicos) deben permanecer prácticamente inalteradas
    assert!(
        (g_far - 1.0).abs() < 1e-4,
        "Escala lejana no debe contaminarse (g_far = {g_far})"
    );
}

/// Contrato 4: Homeostasis Proporcional a la Escala Temporal (Memoria Relativa al Horizonte).
#[test]
fn f1_080_contrato_4_homeostasis_proporcional_al_horizonte_temporal() {
    let mut spec = TemporalSpectrum::new();
    let idx_fast = 18; // ~60 s
    let idx_slow = 24; // ~4 h (~14,400,000 ms)

    // Amplificar ambas escalas artificialmente a 2.0
    spec.scales[idx_fast].epigenetic_gain = 2.0;
    spec.scales[idx_slow].epigenetic_gain = 2.0;

    let t0 = 1_000_000u64;
    spec.update(100.0, t0);

    // Avanzar 30 minutos (1,800,000 ms)
    let t_30m = t0 + 1_800_000;
    spec.update(100.0, t_30m);

    let g_fast_30m = spec.scales[idx_fast].epigenetic_gain;
    let g_slow_30m = spec.scales[idx_slow].epigenetic_gain;

    // La escala rápida (tau = 60s) tiene tau_homeo = 30 min, tras 30 min decae a ~ 1.0 + 1.0 * e^(-1) = ~1.367
    assert!(
        (g_fast_30m - (1.0 + (-1.0_f64).exp())).abs() < 0.05,
        "Escala rápida debe decaer con tau_homeo de 30 min (g = {g_fast_30m})"
    );

    // La escala lenta (tau = 4h) tiene tau_homeo = 16 * 4h = 64h. Tras 30 min, apenas ha disipado memoria (~0.7%)
    assert!(
        g_slow_30m > 1.90,
        "Escala lenta no debe perder memoria en 30 minutos (g_slow_30m = {g_slow_30m})"
    );
}

/// Contrato 5: Inmunidad IEEE-754 y Clamping Riguroso ante Valores Extremos.
#[test]
fn f1_080_contrato_5_inmunidad_ieee754_y_cotas_extremas() {
    let mut spec = TemporalSpectrum::new();
    let tau_target = SPECTRUM_SCALES_MS[18];

    // 1. Valores no finitos son ignorados sin panic ni corrupción
    for bad_pnl in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        spec.apply_epigenetic_outcome(tau_target, true, bad_pnl);
        assert_eq!(spec.scales[18].epigenetic_gain, 1.0);
    }
    for bad_tau in [0.0, -100.0, f64::NAN, f64::INFINITY] {
        spec.apply_epigenetic_outcome(bad_tau, true, 0.01);
        assert_eq!(spec.scales[18].epigenetic_gain, 1.0);
    }

    // 2. Ganancias masivas saturan limpiamente en 3.00
    for _ in 0..100 {
        spec.apply_epigenetic_outcome(tau_target, true, 0.50); // +50% por trade
    }
    assert_eq!(spec.scales[18].epigenetic_gain, 3.00);

    // 3. Pérdidas masivas saturan limpiamente en 0.20
    for _ in 0..100 {
        spec.apply_epigenetic_outcome(tau_target, false, -0.50); // -50% por trade
    }
    assert_eq!(spec.scales[18].epigenetic_gain, 0.20);
}
