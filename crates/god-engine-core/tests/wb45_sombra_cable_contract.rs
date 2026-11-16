//! WB-4/WB-5 EN SOMBRA — contrato de fuente del cable en el núcleo: el libro de
//! decisiones y el f encogido se CALCULAN y se REGISTRAN, pero ninguna
//! decisión los lee. La orden sigue usando la p calibrada de hoy
//! (`calibrator_by_coin` / `confidence_calibrator`) y el `kelly_fraction` del
//! legado. La conducta (órdenes idénticas byte a byte a C3) la fija la traza
//! del replay de referencia (`backtest-engine/tests/wb45_sombra_identidad_ordenes.rs`).

const NUCLEO: &str = include_str!("../src/lib.rs");
const RIESGO: &str = include_str!("../../risk-engine/src/lib.rs");
const LIBRO: &str = include_str!("../src/libro_decisiones.rs");

/// Líneas de código (sin comentarios de línea), recortadas.
fn lineas(fuente: &str) -> Vec<String> {
    fuente
        .replace("\r\n", "\n")
        .lines()
        .map(|l| l.split("//").next().unwrap_or("").trim().to_string())
        .filter(|l| !l.is_empty())
        .collect()
}

fn sin_espacios(fuente: &str) -> String {
    lineas(fuente).concat().split_whitespace().collect()
}

/// El núcleo sólo ESCRIBE en el libro de decisiones (y lee su diagnóstico
/// para la telemetría dentro del propio módulo).
#[test]
fn wb4_el_nucleo_solo_escribe_en_el_libro() {
    let permitidos = [
        ("pub mod libro_decisiones;", 1),
        ("pub libro_decisiones: libro_decisiones::LibroDecisiones,", 1),
        ("libro_decisiones: libro_decisiones::LibroDecisiones::new(n_coins),", 1),
        ("if self.libro_decisiones.observar(", 1),
        ("self.libro_decisiones.avisar(self.trazas_diagnosticas);", 2),
        ("let _ = self.libro_decisiones.registrar_vetada(libro_decisiones::Decision::desde_geometria(", 2),
        ("self.libro_decisiones.alimentar_ejecutada(", 1),
        ("libro_decisiones::fuente_vetada(", 1),
    ];
    let usos: Vec<String> = lineas(NUCLEO)
        .into_iter()
        .filter(|l| l.contains("libro_decisiones"))
        .collect();
    for uso in &usos {
        assert!(
            permitidos.iter().any(|(p, _)| uso.starts_with(p)),
            "uso del libro no permitido en el núcleo: `{uso}`"
        );
    }
    for (p, veces) in permitidos {
        assert_eq!(usos.iter().filter(|u| u.starts_with(p)).count(), veces, "`{p}`");
    }
    // Nada del libro llega a la p de la orden: la p calibrada sale de los
    // calibradores vivos, como en C3.
    let nucleo = sin_espacios(NUCLEO);
    assert!(nucleo.contains(
        "self.calibrator_by_coin[coin_id].calibrate_at(raw_confidence_score,event_time_ms)"
    ));
    assert!(nucleo.contains(".calibrate_at(raw_confidence_score,event_time_ms)};calibrated_intent.win_probability=cal_prob;"));
    assert!(!nucleo.contains("p_sombra("), "el núcleo no consulta la p en sombra");
    assert!(!nucleo.contains(".diagnostico()"), "ni su diagnóstico");
    // El libro no toca los calibradores vivos ni el registro.
    let libro = sin_espacios(LIBRO);
    for prohibido in ["calibrator_by_coin", "confidence_calibrator", "registry", "arena"] {
        assert!(!libro.contains(prohibido), "libro_decisiones.rs usa `{prohibido}`");
    }
}

/// El f encogido se calcula junto al legado y SÓLO se registra: el
/// `kelly_fraction` que se guarda sigue siendo el de `calculate_kelly_fraction`
/// y nadie lee las claves `sombra_wb5_*`.
#[test]
fn wb5_el_f_encogido_solo_se_registra() {
    let nucleo = sin_espacios(NUCLEO);
    let legado = nucleo
        .find("letkelly_f=risk_engine::kelly::calculate_kelly_fraction(new_wr,pf_para_kelly,")
        .expect("el legado sigue calculando kelly_f");
    let guarda = nucleo
        .find("coin.metrics.kelly_fraction.store(kelly_f,Ordering::Relaxed);")
        .expect("y se guarda tal cual");
    let sombra = nucleo
        .find("letsombra_wb5=risk_engine::kelly::kelly_con_encogimiento(")
        .expect("la sombra se calcula en el cierre");
    assert!(legado < guarda && guarda < sombra, "la sombra va después de guardar el legado");
    assert_eq!(nucleo.matches("kelly_con_encogimiento(").count(), 1);
    assert_eq!(nucleo.matches(".kelly_fraction.store(").count(), 1, "un solo escritor de kelly_fraction");
    // `sombra_wb5` y `f_sombra_wb5` sólo van al registro y al log.
    for uso in lineas(NUCLEO).iter().filter(|l| l.contains("sombra_wb5")) {
        let permitido = uso.starts_with("let sombra_wb5 = risk_engine::kelly::kelly_con_encogimiento(")
            || uso.starts_with("let f_sombra_wb5 = sombra_wb5.f_ajustada * koopman_dampener;")
            || uso.starts_with("(\"sombra_wb5_")
            || uso.starts_with("sombra_wb5.")
            || uso.starts_with("f_sombra_wb5,");
        assert!(permitido, "uso de la sombra WB-5 no permitido: `{uso}`");
    }
    for fuente in [NUCLEO, RIESGO] {
        let f = sin_espacios(fuente);
        assert!(!f.contains("get_for_coin_strict(coin_id,\"sombra_wb5"), "nadie lee sombra_wb5_*");
        assert!(!f.contains("get_for_coin_or(coin_id,\"sombra_wb5"), "nadie lee sombra_wb5_*");
    }
}

/// LH-04 (la ruta viva del tamaño) y la ruta nueva comparten la cota de
/// Wilson: una sola implementación (`evidence::wilson_lcb`).
#[test]
fn wb5_lh04_usa_la_cota_de_wilson_unica() {
    let riesgo = sin_espacios(RIESGO);
    assert!(riesgo.contains("letp_lcb=crate::evidence::wilson_lcb(lh04_wr,n,z_lcb_lh04);"));
    assert!(!riesgo.contains("letcentro="), "sin una segunda copia de Wilson en LH-04");
}
