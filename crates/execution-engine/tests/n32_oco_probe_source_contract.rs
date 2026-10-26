//! N32 (CL-205): contratos de fuente de la sonda OCO (src/bin/oco_probe.rs).
//! No la ejecutan ni tocan el exchange: la sonda abre una posición real en
//! testnet y sólo se puede probar leyendo su código.
const SONDA: &str = include_str!("../../../src/bin/oco_probe.rs");
const CLIENTE: &str = include_str!("../src/client.rs");

/// La sonda sin su módulo de pruebas.
fn produccion() -> &'static str {
    SONDA.split("#[cfg(test)]").next().unwrap_or("")
}

/// Posición de la `}` que cierra la llave abierta en `abre`, contando llaves
/// fuera de cadenas y de comentarios de línea (las de los formatos de
/// println! van en cadenas).
fn cierre(fuente: &str, abre: usize) -> Option<usize> {
    let b = fuente.as_bytes();
    let (mut nivel, mut i, mut en_cadena) = (0usize, abre, false);
    while let Some(&c) = b.get(i) {
        if en_cadena {
            match c {
                b'\\' => i += 1,
                b'"' => en_cadena = false,
                _ => {}
            }
        } else if c == b'/' && b.get(i + 1) == Some(&b'/') {
            while b.get(i).is_some_and(|&c| c != b'\n') {
                i += 1;
            }
        } else if c == b'"' {
            en_cadena = true;
        } else if c == b'{' {
            nivel += 1;
        } else if c == b'}' {
            nivel = nivel.checked_sub(1)?;
            if nivel == 0 {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

/// RED antes de CL-205: dos `.unwrap()` de la hora en código que no es de
/// prueba, y un recorte de texto por bytes que entraba en pánico con un error
/// acentuado («precisión», «—») en el punto de corte. La guarda es literal:
/// prohíbe esas formas y las macros que entran en pánico, no cualquier
/// pánico (println! lo hace con stdout cerrado; la cabecera lo advierte).
#[test]
fn n32_la_sonda_no_entra_en_panico() {
    let codigo = produccion();
    assert!(codigo.contains("// 7. Limpieza total"), "sin su módulo de pruebas");
    for prohibido in [".unwrap()", ".expect(", "panic!", "&s[..max]", "unreachable!", "todo!", "unimplemented!", "assert!"] {
        assert!(!codigo.contains(prohibido), "{prohibido}");
    }
}

/// RED antes de CL-205: tras enviar la entrada MARKET, un fallo de
/// positionRisk o una posición sin magnitud volvía de main sin limpiar, y la
/// posición quedaba abierta y sin TP/SL. Desde la entrada hasta la limpieza
/// no hay ningún `return`: los fallos salen del bloque etiquetado.
#[test]
fn n32_tras_la_entrada_todo_pasa_por_la_limpieza() {
    let bloque = SONDA.find("'sonda: {").unwrap_or(usize::MAX);
    let entrada = SONDA.find(".execute_raw_qty_with_client_id(").unwrap_or(usize::MAX);
    let limpieza = SONDA.find("// 7. Limpieza total").unwrap_or(0);
    assert!(bloque < entrada && entrada < limpieza, "bloque, entrada y limpieza en ese orden");
    let tramo = SONDA.get(bloque..limpieza).unwrap_or("return");
    assert!(!tramo.contains("return"), "{tramo}");
    assert_eq!(SONDA.matches(".flatten_all_positions()").count(), 2, "estado previo y limpieza");
    // RV24-T: el bloque cierra antes de la limpieza y todas sus salidas están
    // dentro; si no, una limpieza movida dentro del bloque pasaba la guarda.
    let fin = cierre(SONDA, bloque + "'sonda: ".len()).unwrap_or(usize::MAX);
    assert!(fin < limpieza, "el bloque cierra antes de la limpieza");
    let salidas: Vec<usize> = SONDA.match_indices("break 'sonda").map(|(i, _)| i).collect();
    assert!(!salidas.is_empty() && salidas.iter().all(|&i| bloque < i && i < fin), "{salidas:?}");
}

/// RED antes de CL-205: `unwrap_or((0, 0))` daba un aplanado fallido por
/// «0 posiciones cerradas, órdenes purgadas», y el primer campo de la tupla
/// (las cancelaciones aceptadas, hasta dos por símbolo) se imprimía como
/// posiciones cerradas.
#[test]
fn n32_la_limpieza_dice_su_error() {
    let limpieza = SONDA.split("// 7. Limpieza total").nth(1).unwrap_or("");
    assert!(!limpieza.contains("unwrap_or((0, 0))"), "{limpieza}");
    assert!(limpieza.contains("Ok((_, cerradas)) =>"), "{limpieza}");
    assert!(limpieza.contains("limpieza incompleta"), "{limpieza}");
}

/// RED antes de CL-207: un 429 en el paso 4 arma el freno, y el aplanado de
/// la limpieza, rechazado en local sin enviar nada, dejaba la posición
/// abierta aunque el freno venciera a los dos segundos; y el aplanado sólo
/// avisa de una purga fallida, que se leía como «órdenes purgadas». La
/// limpieza repite el aplanado mientras el freno lo rechace (con los textos
/// de `FrenoExchange::comprobar`) y lee las órdenes que queden del símbolo.
#[test]
fn cl207_la_limpieza_espera_al_freno_y_lee_lo_que_queda() {
    let limpieza = produccion().split("// 7. Limpieza total").nth(1).unwrap_or("");
    let repite = limpieza.find("loop {").unwrap_or(usize::MAX);
    let aplana = limpieza.find(".flatten_all_positions()").unwrap_or(0);
    assert!(repite < aplana, "el aplanado va dentro del bucle");
    assert!(limpieza.contains("espera_del_freno_ms(e)"), "{limpieza}");
    assert!(!limpieza.contains("órdenes purgadas\""), "{limpieza}");
    for lectura in [".fetch_open_orders(&symbol)", ".fetch_open_algo_orders(&symbol)"] {
        assert!(limpieza.contains(lectura), "{lectura}");
    }
    for texto in [
        "\"RATE_BRAKE: freno temporal activo hasta {} (ahora {})\"",
        "\"RATE_LIMIT_COOLDOWN: activo hasta {} (ahora {})\"",
    ] {
        assert!(CLIENTE.contains(texto), "{texto}");
    }
}
