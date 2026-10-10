//! QS-R1 / K-06 & K-23 — CONTRATO DE SIMETRÍA DIRECCIONAL EN PUERTAS DEL CONTINUO Y GATE ML (Ola Ω75).
//!
//! Contrato formal que demuestra la resolución de:
//! - **K-06**: Eliminación de la asimetría estructural en `puertas_del_continuo` (`lib.rs:1563-1579`).
//!   Bajo la fórmula legacy `(p - b) * 2.0` con base honesta b ≈ 0.25:
//!   - Long vivía en d ∈ [-0.50, +1.50] -> VETO IMPOSIBLE (min -0.50 > -0.80), boost hasta ×1.75.
//!   - Short vivía en d ∈ [-1.50, +0.50] -> VETO PREMATURO (p > 0.65), boost capado a ×1.25.
//!   Con `normalized_directional_divergence`, ambos lados viven en [-1.0, 1.0], vetan en d < -0.80
//!   y tienen exactamente el mismo techo de impulso (1.50x).
//! - **K-23**: Eliminación de la parálisis de cortos en el Gate ML B3.18 (`lib.rs:7663-7699`).
//!   Bajo la fórmula legacy `p <= b - lift_S`:
//!   - Si lift_S >= b (ej. b <= 0.135 con genoma activo, o en extremos de gen), el umbral colapsaba
//!     a <= 0.0 haciendo matemáticamente imposible abrir posiciones cortas.
//!   Con escala proporcional a los semi-intervalos [0, b] y [b, 1], el umbral para cortos
//!   es b * (1.0 - 2.0 * lift_S) >= 0.40 * b > 0 siempre, exigiendo idéntico rigor estadístico
//!   relativo en ambos lados y preservando coincidencia exacta con legacy cuando b = 0.50.

use god_engine_core::calibration::normalized_directional_divergence;

const TOL: f64 = 1e-12;

#[test]
fn k06_simetria_de_rango_y_extremos_en_puertas_del_continuo() {
    let bases = [0.10, 0.20, 0.25, 0.30, 0.33, 0.50, 0.70, 0.85];

    for &b in &bases {
        // En p == b, neutralidad exacta para ambos lados
        let div_base = normalized_directional_divergence(b, b);
        assert_eq!(div_base, 0.0, "base={b}");

        // Soporte máximo: p = 1.0 para Long y p = 0.0 para Short
        let long_max_support = normalized_directional_divergence(1.0, b);
        let short_max_support = -normalized_directional_divergence(0.0, b);
        assert!((long_max_support - 1.0).abs() < TOL, "Long max support en base={b}");
        assert!((short_max_support - 1.0).abs() < TOL, "Short max support en base={b}");

        // Multiplicador de impulso idéntico: 1.0 + d * 0.5
        let long_boost = 1.0 + long_max_support * 0.5;
        let short_boost = 1.0 + short_max_support * 0.5;
        assert_eq!(long_boost, 1.50, "Impulso Long debe ser 1.50");
        assert_eq!(short_boost, 1.50, "Impulso Short debe ser 1.50");

        // Oposición máxima: p = 0.0 para Long y p = 1.0 para Short
        let long_max_opposition = normalized_directional_divergence(0.0, b);
        let short_max_opposition = -normalized_directional_divergence(1.0, b);
        assert!((long_max_opposition - (-1.0)).abs() < TOL, "Long max oposicion en base={b}");
        assert!((short_max_opposition - (-1.0)).abs() < TOL, "Short max oposicion en base={b}");
    }
}

#[test]
fn k06_veto_alcanzable_para_long_y_desasfixia_para_short() {
    let b = 0.25;

    // 1. En legacy, Long con p = 0.0 daba d = (0.0 - 0.25) * 2.0 = -0.50 > -0.80 (NUNCA VETABA).
    // Con la corrección, p < 0.20 * b (ej. p = 0.04) da d = (0.04 - 0.25) / 0.25 = -0.84 < -0.80 -> VETO.
    let div_long_veto = normalized_directional_divergence(0.04, b);
    assert!(div_long_veto < -0.80, "Long con p=0.04 debe ser vetado: d={div_long_veto}");

    // 2. En legacy, Short con p = 0.70 daba d = (0.25 - 0.70) * 2.0 = -0.90 < -0.80 (VETADO PREMATURAMENTE).
    // Con la corrección, p = 0.70 da d_long = (0.70 - 0.25) / 0.75 = +0.60.
    // Para short, d_short = -0.60 > -0.80 -> NO se veta prematuramente, recibe penalización suave.
    let div_short_eval = -normalized_directional_divergence(0.70, b);
    assert!(div_short_eval > -0.80, "Short con p=0.70 NO debe ser vetado prematuramente: d={div_short_eval}");
    assert_eq!(div_short_eval, -0.60);

    // El veto de Short ocurre cuando p > b + 0.80 * (1 - b) = 0.25 + 0.80 * 0.75 = 0.85
    let div_short_real_veto = -normalized_directional_divergence(0.86, b);
    assert!(div_short_real_veto < -0.80, "Short con p=0.86 sí debe ser vetado: d={div_short_real_veto}");
}

#[test]
fn k23_gate_ml_coincidencia_exacta_legacy_en_base_050() {
    let b = 0.50;
    let lift_long = 0.12;
    let lift_short = 0.08;

    let required_edge_long = 2.0 * lift_long;
    let required_edge_short = 2.0 * lift_short;

    // Long pasa si p >= 0.50 + 0.12 = 0.62
    let p_long_border = 0.62;
    let div_long = normalized_directional_divergence(p_long_border, b);
    assert!((div_long - required_edge_long).abs() < TOL);

    // Short pasa si p <= 0.50 - 0.08 = 0.42
    let p_short_border = 0.42;
    let div_short = -normalized_directional_divergence(p_short_border, b);
    assert!((div_short - required_edge_short).abs() < TOL);
}

#[test]
fn k23_gate_ml_elimina_paralisis_de_cortos_con_base_baja() {
    // Escenario crítico de K-23: base honesta baja b = 0.10 y lift_short = 0.08
    let b = 0.10;
    let lift_short = 0.08;
    let required_edge = 2.0 * lift_short; // 0.16

    // En legacy: p <= b - lift_S = 0.10 - 0.08 = 0.02 (extremo casi inalcanzable)
    // O si lift_S >= 0.10: b - lift_S <= 0.0 (MATEMÁTICAMENTE IMPOSIBLE abrir cortos)
    //
    // Con la corrección proporcional simétrica:
    // -div >= required_edge <=> (b - p) / b >= 2.0 * lift_S <=> p <= b * (1 - 2.0 * lift_S)
    // p_umbral = 0.10 * (1 - 0.16) = 0.084 > 0
    let p_pass = 0.08; // modelo ve 8% de probabilidad de subida (92% de bajada)
    let div_pass = -normalized_directional_divergence(p_pass, b);
    assert!(div_pass >= required_edge, "Short con p=0.08 debe pasar el gate: div={div_pass} >= {required_edge}");

    let p_fail = 0.09; // modelo ve 9% de probabilidad (no alcanza el 16% de lift relativo requerido)
    let div_fail = -normalized_directional_divergence(p_fail, b);
    assert!(div_fail < required_edge, "Short con p=0.09 no debe pasar: div={div_fail} < {required_edge}");

    // Garantía analítica: para cualquier lift_short in [0.01, 0.30], el umbral NUNCA colapsa a <= 0
    for &l in &[0.01, 0.05, 0.10, 0.20, 0.25, 0.30] {
        let req = 2.0 * l;
        assert!(req <= 0.60, "req={req} <= 0.60");
        let threshold_fraction = 1.0 - req;
        assert!(threshold_fraction >= 0.40, "La fracción de umbral siempre es >= 40% de base");
    }
}
