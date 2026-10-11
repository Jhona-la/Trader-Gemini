//! CL-49: el host publica el valor ACTUAL de los entes del consejo (ballena y
//! spoofing) en cada medición, también el nulo. Antes sólo escribía en evento
//! (burst, score > 0,05) y el último valor alto quedaba pegado horas.
//!
//! El binario `god_engine` no es testeable por unidad: el contrato se fija
//! sobre su fuente, como `kill_switch_host_contract.rs` (sin comentarios de
//! línea completa y sin espacios). La regla de valores y la lectura del núcleo
//! se prueban en `god_engine_core::entes_consejo`.

fn host() -> String {
    include_str!("../../../src/bin/god_engine.rs")
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<String>()
        .split_whitespace()
        .collect()
}

fn tramo<'a>(h: &'a str, desde: &str, hasta: &str) -> &'a str {
    let i = h.find(desde).unwrap_or_else(|| panic!("ancla «{desde}» ausente"));
    let f = i + h[i..]
        .find(hasta)
        .unwrap_or_else(|| panic!("ancla «{hasta}» ausente tras «{desde}»"));
    &h[i..f]
}

#[test]
fn cl49_cada_trade_publica_su_ente_ballena() {
    let h = host();
    let t = tramo(&h, "whale_trackers[coin_id].update(notional)", "}elseifis_depth{");
    assert!(
        t.contains("entes_consejo::publicar(") && t.contains("valor_ballena(z,is_burst)"),
        "el trade publica su valor, burst o no"
    );
    assert!(!t.contains("ifis_burst{"), "la publicación no depende de que haya burst");
}

#[test]
fn cl49_cada_evaluacion_de_profundidad_publica_su_ente_spoofing() {
    let h = host();
    let t = tramo(&h, "spoof_detectors[sym_id].evaluate_wall_decay(", "letntp_offset_actual");
    assert!(
        t.contains("entes_consejo::publicar(") && t.contains("valor_spoof(score)"),
        "la evaluación publica el score decaído"
    );
    assert!(!t.contains("score>0.05{"), "la publicación no depende del suelo de ruido");
}

/// El núcleo lee los dos entes con la misma regla con la que se escriben.
#[test]
fn cl49_el_nucleo_lee_los_entes_por_el_mismo_modulo() {
    let nucleo: String = include_str!("../src/lib.rs").split_whitespace().collect();
    assert!(nucleo.contains("whale_burst_z:entes_consejo::leer("));
    assert!(nucleo.contains("spoof_score:entes_consejo::leer("));
}
