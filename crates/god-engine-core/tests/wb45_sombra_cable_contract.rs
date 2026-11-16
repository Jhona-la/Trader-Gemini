//! WB-4/WB-5 EN SOMBRA — contrato de fuente del cable en el núcleo: el libro de
//! decisiones y el f encogido se CALCULAN y se REGISTRAN, pero ninguna
//! decisión los lee. La orden sigue usando la p calibrada de hoy
//! (`calibrator_by_coin` / `confidence_calibrator`) y el `kelly_fraction` del
//! legado. La conducta (órdenes idénticas byte a byte a C3) la fija la traza
//! del replay de referencia (`backtest-engine/tests/wb45_sombra_identidad_ordenes.rs`).

const NUCLEO: &str = include_str!("../src/lib.rs");
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
