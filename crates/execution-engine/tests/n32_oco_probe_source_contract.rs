//! N32 (CL-205): contratos de fuente de la sonda OCO (src/bin/oco_probe.rs).
//! No la ejecutan ni tocan el exchange: la sonda abre una posición real en
//! testnet y sólo se puede probar leyendo su código.
const SONDA: &str = include_str!("../../../src/bin/oco_probe.rs");

/// RED antes de CL-205: dos `.unwrap()` de la hora en código que no es de
/// prueba, y un recorte de texto por bytes que entraba en pánico con un error
/// acentuado («precisión», «—») en el punto de corte.
#[test]
fn n32_la_sonda_no_entra_en_panico() {
    for prohibido in [".unwrap()", ".expect(", "panic!", "&s[..max]"] {
        assert!(!SONDA.contains(prohibido), "{prohibido}");
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
}

/// RED antes de CL-205: `unwrap_or((0, 0))` daba un aplanado fallido por
/// «0 posiciones cerradas, órdenes purgadas», y el primer campo de la tupla
/// (los símbolos cancelados) se imprimía como posiciones cerradas.
#[test]
fn n32_la_limpieza_dice_su_error() {
    let limpieza = SONDA.split("// 7. Limpieza total").nth(1).unwrap_or("");
    assert!(!limpieza.contains("unwrap_or((0, 0))"), "{limpieza}");
    assert!(limpieza.contains("Ok((_, cerradas)) =>"), "{limpieza}");
    assert!(limpieza.contains("limpieza incompleta"), "{limpieza}");
}
