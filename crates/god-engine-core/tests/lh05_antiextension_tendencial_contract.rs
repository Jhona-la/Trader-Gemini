//! CONTRATO FORMAL LH-05: RESOLUCIÓN DE LA CONTRADICCIÓN ALGEBRAICA EN RAMAS TENDENCIALES (Ola Ω95).
//!
//! # Origen del Hallazgo (LH-05)
//! En `crates/god-engine-core/src/lib.rs:6107-6134`, la guarda de anti-extensión (No Chasing Law)
//! imponía:
//! ```text
//! let (min_stretch_short, max_stretch_long) = if is_mean_reverting {
//!     (0.05, -0.05)
//! } else {
//!     (0.00, 0.00)
//! };
//! let not_overextended_long = price_stretch <= max_stretch_long && stretch_z >= -Z95;
//! ```
//! En régimen tendencial (`!is_mean_reverting`), la rama 1 (Trend Long, líneas 6559-6573) exigía:
//! `macro_trend >= 0.0 && micro_trend >= 0.0 && composite_score > u_tech_thr`.
//! En un avance tendencial alcista, el precio cotiza naturalmente por encima de la media lenta
//! (`price_stretch > 0.0`). Al imponer `price_stretch <= 0.00`, la conjunción exigía:
//! `price >= EMA_slow && price <= EMA_slow => price == EMA_slow` exacto.
//! Esto redujo el espacio de estados operables a un conjunto de medida nula, provocando que
//! 0 de 604 operaciones tendenciales pudieran abrirse en pruebas de ventaja conocida.
//!
//! # Resolución Canónica
//! - En reversión a la media: preserva la exigencia de descuento/prima (`price_stretch <= -0.05` / `>= 0.05`).
//! - En tendencia: la disciplina anti-extensión es gobernada por el envolvente difusivo canónico D-621:
//!   `|stretch_z| <= Z95`. Esto permite capturar tendencias alcistas y bajistas activas sin perseguir
//!   fuera del intervalo de confianza del 95%.

use god_engine_core::diffusion::{atr_stretch_z, EMA_SLOW_BARS, Z95};

fn evaluar_antiextension(
    is_mean_reverting: bool,
    price_stretch: f64,
) -> (bool, bool) {
    let stretch_z = atr_stretch_z(price_stretch, EMA_SLOW_BARS);
    if is_mean_reverting {
        (
            price_stretch <= -0.05 && stretch_z >= -Z95,
            price_stretch >= 0.05 && stretch_z <= Z95,
        )
    } else {
        (
            stretch_z >= -Z95 && stretch_z <= Z95,
            stretch_z >= -Z95 && stretch_z <= Z95,
        )
    }
}

fn evaluar_legacy(
    is_mean_reverting: bool,
    price_stretch: f64,
) -> (bool, bool) {
    let (min_stretch_short, max_stretch_long) = if is_mean_reverting {
        (0.05, -0.05)
    } else {
        (0.00, 0.00)
    };
    let stretch_z = atr_stretch_z(price_stretch, EMA_SLOW_BARS);
    (
        price_stretch <= max_stretch_long && stretch_z >= -Z95,
        price_stretch >= min_stretch_short && stretch_z <= Z95,
    )
}

#[test]
fn lh05_tendencial_permite_avances_dentro_del_envolvente_difusivo() {
    // Escenario: Un avance tendencial alcista moderado (precio 0.50 ATR sobre la EMA slow).
    let stretch_avance_long = 0.50;
    let (legacy_long, _) = evaluar_legacy(false, stretch_avance_long);
    let (resuelto_long, _) = evaluar_antiextension(false, stretch_avance_long);

    // En legacy: bloqueado por la contradicción price_stretch <= 0.00
    assert!(
        !legacy_long,
        "Legacy debió haber bloqueado el avance tendencial por el bug LH-05"
    );

    // En resolución canónica: permitido por estar dentro de |z| <= 1.96
    let z = atr_stretch_z(stretch_avance_long, EMA_SLOW_BARS);
    assert!(
        z.abs() < Z95,
        "El z del avance ({z}) debe estar dentro de la banda Z95 ({Z95})"
    );
    assert!(
        resuelto_long,
        "La resolución LH-05 debe admitir avances tendenciales dentro de la banda difusiva"
    );

    // Escenario espejo: Un avance tendencial bajista moderado (precio -0.50 ATR bajo la EMA slow).
    let stretch_avance_short = -0.50;
    let (_, legacy_short) = evaluar_legacy(false, stretch_avance_short);
    let (_, resuelto_short) = evaluar_antiextension(false, stretch_avance_short);

    assert!(
        !legacy_short,
        "Legacy debió haber bloqueado el corto tendencial por el bug LH-05"
    );
    assert!(
        resuelto_short,
        "La resolución LH-05 debe admitir cortos tendenciales dentro de la banda difusiva"
    );
}

#[test]
fn lh05_tendencial_veta_persecucion_extrema_en_ambos_lados() {
    // Escenario: Estiramiento extremo de 3.0 ATR (burbuja/euforia o pánico).
    let stretch_extremo_alza = 3.0;
    let z_alza = atr_stretch_z(stretch_extremo_alza, EMA_SLOW_BARS);
    assert!(
        z_alza > Z95,
        "Un estiramiento de 3 ATR debe superar Z95 (obtenido z={z_alza})"
    );

    let (permite_long_extremo, _) = evaluar_antiextension(false, stretch_extremo_alza);
    assert!(
        !permite_long_extremo,
        "La disciplina anti-extensión debe vetar persecución alcista cuando z > Z95"
    );

    let stretch_extremo_baja = -3.0;
    let z_baja = atr_stretch_z(stretch_extremo_baja, EMA_SLOW_BARS);
    assert!(
        z_baja < -Z95,
        "Un estiramiento de -3 ATR debe caer bajo -Z95 (obtenido z={z_baja})"
    );

    let (_, permite_short_extremo) = evaluar_antiextension(false, stretch_extremo_baja);
    assert!(
        !permite_short_extremo,
        "La disciplina anti-extensión debe vetar persecución bajista cuando z < -Z95"
    );
}

#[test]
fn lh05_reversion_media_preserva_rigor_de_descuento_y_sobreprecio() {
    // En reversión a la media:
    // Long requiere comprar con descuento (price_stretch <= -0.05)
    let (long_en_descuento, _) = evaluar_antiextension(true, -0.20);
    let (long_sin_descuento, _) = evaluar_antiextension(true, 0.01);
    assert!(
        long_en_descuento,
        "Reversión a la media debe permitir comprar con descuento (-0.20 ATR)"
    );
    assert!(
        !long_sin_descuento,
        "Reversión a la media debe prohibir comprar sin descuento (+0.01 ATR)"
    );

    // Short requiere vender con sobreprecio (price_stretch >= 0.05)
    let (_, short_en_sobreprecio) = evaluar_antiextension(true, 0.20);
    let (_, short_sin_sobreprecio) = evaluar_antiextension(true, -0.01);
    assert!(
        short_en_sobreprecio,
        "Reversión a la media debe permitir vender con sobreprecio (+0.20 ATR)"
    );
    assert!(
        !short_sin_sobreprecio,
        "Reversión a la media debe prohibir vender sin sobreprecio (-0.01 ATR)"
    );
}

#[test]
fn lh05_simetria_especular_exacta_long_short() {
    let valores_stretch = [-2.5, -1.8, -0.8, -0.1, -0.05, 0.0, 0.05, 0.1, 0.8, 1.8, 2.5];

    for &s in &valores_stretch {
        // En tendencia: la condición Long para +s debe ser exactamente idéntica a la condición Short para -s.
        let (long_pos, _) = evaluar_antiextension(false, s);
        let (_, short_neg) = evaluar_antiextension(false, -s);
        assert_eq!(
            long_pos, short_neg,
            "Simetría tendencial rota para s={s}: long({s}) != short({neg})",
            neg = -s
        );

        // En reversión: la condición Long para -s (descuento) debe coincidir con Short para +s (sobreprecio).
        let (rev_long, _) = evaluar_antiextension(true, -s.abs());
        let (_, rev_short) = evaluar_antiextension(true, s.abs());
        assert_eq!(
            rev_long, rev_short,
            "Simetría de reversión rota para |s|={abs}: rev_long != rev_short",
            abs = s.abs()
        );
    }
}
