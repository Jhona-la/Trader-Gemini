//! RET-51b (CL-160, censo C-F7b, F155): `telemetry!` y `telemetry_err!` de
//! telemetry-engine escriben en la misma cola que `telemetry_log!`
//! (LOG_QUEUE) cuando el proceso conecta `linea_de_telemetria`, como hace el
//! host. Antes iban a una segunda cola que sólo un hilo del host pasaba a la
//! del log, sondeándola cada ≈1 ms cuando estaba vacía: el apagado
//! (`vaciar_telemetria`) no la vaciaba y sus líneas podían salir después de
//! otras posteriores del mismo hilo.
//!
//! Archivo propio porque la prueba inicializa el log global del proceso.

use telemetry_server::macros::{
    Rotacion, init_telemetry_logger_rotativo, linea_de_telemetria, vaciar_telemetria,
};

/// Esperado: tres líneas emitidas en orden por las dos familias de macros
/// llegan al log rotativo en ese orden y el vaciado del apagado las deja en
/// disco. Con la cola binaria de antes (sin el hilo del host) el archivo
/// sólo tenía «DOS 2».
#[test]
fn ret51b_las_dos_familias_de_macros_comparten_cola_orden_y_apagado() {
    let id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("ret51b_{}_{id}", std::process::id()));
    assert!(std::fs::create_dir_all(&dir).is_ok(), "directorio temporal {dir:?}");
    let rot = Rotacion { ruta: dir.join("motor.log"), tope_bytes: 1 << 20, conservar: 2 };
    assert!(init_telemetry_logger_rotativo(64, rot.clone()));
    assert!(telemetry_engine::conectar(linea_de_telemetria));
    assert!(!telemetry_engine::conectar(linea_de_telemetria), "un solo destino");

    telemetry_engine::telemetry!("UNO {}", 1);
    telemetry_server::telemetry_log!("DOS {}", 2);
    telemetry_engine::telemetry_err!("TRES {}", 3);
    let _ = vaciar_telemetria();

    let log = std::fs::read_to_string(&rot.ruta).unwrap_or_default();
    assert_eq!(log.lines().collect::<Vec<_>>(), ["UNO 1", "DOS 2", "⚠️ ERROR: TRES 3"]);
    let _ = std::fs::remove_dir_all(&dir);
}
