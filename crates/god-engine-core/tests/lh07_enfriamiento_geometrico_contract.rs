//! LH-07 (Línea H — GLM 119): el enfriamiento post-racha documentado como
//! lo que ES. La condición `t ≥ base·2^(k−⌊log2(1+t/base)⌋)` resuelta da
//! espera efectiva t ≈ base·(√2)^k — retroceso geométrico auto-relajante
//! (forma corregida H3V-07 de Línea G, aceptada por Claude), NO el
//! base·2^k que el doc viejo prometía. Este contrato PINNEA el diseño:
//! para cada k, la espera efectiva (buscada empíricamente) debe caer en
//! [base·2^((k−1)/2), base·2^((k+1)/2)] — la banda de (√2)^k con un
//! escalón de tolerancia por el floor de la amortización.
use god_engine_core::stateful_engine::StatefulEngine;

fn motor_con_racha(racha: u32, t_salida: u64, ahora: u64) -> StatefulEngine {
    let mut e = StatefulEngine::new();
    // Sembrar la racha de pérdidas de la banda rápida y el cierre.
    e.fastband_loss_streak = racha;
    e.last_exit_fastband_ms = t_salida;
    e.last_event_ms = ahora;
    e
}

/// Espera efectiva: el mínimo t (desde el cierre) con el que el motor
/// abre, buscado por bisección sobre `can_open_position_ms`.
fn espera_efectiva_ms(racha: u32, base_ms: f64) -> f64 {
    let t_salida: u64 = 1_000_000;
    let mut lo = 0.0f64;
    let mut hi = base_ms * 2f64.powi(racha as i32 + 1) + base_ms * 2.0;
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        let e = motor_con_racha(racha, t_salida, t_salida + mid as u64);
        if e.can_open_position_ms(base_ms) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// (1) La curva PINNEADA: espera(k) ∈ [base·2^((k−1)/2), base·2^((k+1)/2)]
/// para k=1..8 — geométrica de razón √2, NO 2^k (que sería base·2^(k−1)
/// a base·2^k, un factor √2 más arriba por cada nivel).
#[test]
fn lh07_espera_efectiva_es_geometrica_raiz2_no_binaria() {
    let base = 60_000.0f64; // 60 s
    for k in 1u32..=8 {
        let t = espera_efectiva_ms(k, base);
        let piso = base * 2f64.powf((k as f64 - 1.0) / 2.0);
        let techo = base * 2f64.powf((k as f64 + 1.0) / 2.0);
        assert!(
            t >= piso && t <= techo,
            "k={k}: espera {t} fuera de banda (√2)^k [{piso}, {techo}]"
        );
        // Y NO es binaria: la espera 2^k estaría ~√2^k por ENCIMA del
        // techo de la banda (para k≥3 inconfundible).
        if k >= 3 {
            let binaria = base * 2f64.powi(k as i32 - 1);
            assert!(
                t < binaria,
                "k={k}: espera {t} NO debe alcanzar la binaria {binaria} — el freno es (√2)^k"
            );
        }
    }
}

/// (2) Monotonía estricta en k: cada pérdida adicional espera más.
#[test]
fn lh07_espera_monotona_estricta_en_racha() {
    let base = 60_000.0f64;
    let mut prev = 0.0f64;
    for k in 1u32..=8 {
        let t = espera_efectiva_ms(k, base);
        assert!(t > prev, "k={k}: espera {t} debe superar la anterior {prev}");
        prev = t;
    }
}

/// (3) El techo de 12 h acota el régimen asintótico: con racha extrema,
/// la espera no supera TAU_ANCHOR_SLOW_MS (el requisito se clampa).
#[test]
fn lh07_techo_12h_respeta_racha_extrema() {
    let base = 3_600_000.0f64; // 1 h de base: racha 20 ⇒ 2^20 h sin techo
    let t = espera_efectiva_ms(20, base);
    let techo = quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS;
    assert!(
        t <= techo * 1.5,
        "racha 20 con base 1h: espera {t} acotada por el techo 12h ({techo})"
    );
}

/// (4) Sin cierre previo no hay enfriamiento (contrato existente,
/// re-pinneado junto a la familia).
#[test]
fn lh07_sin_cierre_no_enfria() {
    let e = StatefulEngine::new();
    assert!(e.can_open_position_ms(60_000.0));
}
