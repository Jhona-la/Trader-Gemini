//! Macros de telemetría `telemetry!` y `telemetry_err!`. Las usan el host
//! (god_engine.rs) y los crates que están por debajo de telemetry-server:
//! quantum-arena y os-guardian (el servidor depende de ellos, así que usar
//! sus macros haría un ciclo). storage-engine también las usa.
//!
//! RET-51b (CL-160, censo C-F7b, F155): las macros entregan la línea, en el
//! hilo que la emite, al destino que el proceso conecta con `conectar`. El
//! host conecta `telemetry_server::macros::linea_de_telemetria`, que la
//! escribe en la cola del log (LOG_QUEUE) igual que `telemetry_log!`. Antes
//! iban a una segunda cola (BINARY_QUEUE, 33 MiB en el host) que un hilo del
//! host sondeaba cada ≈1 ms cuando estaba vacía y pasaba a la del log: el
//! apagado (`vaciar_telemetria`) no la vaciaba y sus líneas podían salir
//! después de otras posteriores del mismo hilo. Sin destino (herramientas,
//! backtests y evolver no conectan ninguno) la línea se descarta, como
//! antes, pero sin reservar una cola que nadie lee.

use std::fmt;
use std::sync::OnceLock;

/// Destino de las líneas de `telemetry!` y `telemetry_err!`. Recibe el
/// mensaje sin formatear y corre en el hilo que emite: no debe bloquear.
pub type Destino = fn(fmt::Arguments<'_>);

static DESTINO: OnceLock<Destino> = OnceLock::new();

/// Conecta el destino de todo el proceso. Devuelve false si ya había uno,
/// que no se cambia.
#[must_use = "false: ya había un destino y no se cambió"]
pub fn conectar(destino: Destino) -> bool {
    conectar_en(&DESTINO, destino)
}

/// Lo llaman las macros.
#[doc(hidden)]
pub fn emitir(args: fmt::Arguments<'_>) {
    emitir_en(&DESTINO, args);
}

fn conectar_en(celda: &OnceLock<Destino>, destino: Destino) -> bool {
    celda.set(destino).is_ok()
}

fn emitir_en(celda: &OnceLock<Destino>, args: fmt::Arguments<'_>) {
    if let Some(destino) = celda.get() {
        destino(args);
    }
}

/// Como `println!`, pero hacia el destino conectado. No reserva memoria.
#[macro_export]
macro_rules! telemetry {
    ($($arg:tt)*) => {
        $crate::emitir(::core::format_args!($($arg)*))
    };
}

/// Como `telemetry!`, con el prefijo «⚠️ ERROR: ».
#[macro_export]
macro_rules! telemetry_err {
    ($($arg:tt)*) => {
        $crate::emitir(::core::format_args!("⚠️ ERROR: {}", ::core::format_args!($($arg)*)))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static LOCALES: Mutex<Vec<String>> = Mutex::new(Vec::new());
    static GLOBALES: Mutex<Vec<String>> = Mutex::new(Vec::new());

    fn lineas(celda: &Mutex<Vec<String>>) -> Vec<String> {
        celda.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    fn a_locales(args: fmt::Arguments<'_>) {
        LOCALES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(args.to_string());
    }

    fn a_globales(args: fmt::Arguments<'_>) {
        GLOBALES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(args.to_string());
    }

    fn a_ninguna_parte(_: fmt::Arguments<'_>) {}

    /// RET-51b: sin destino la línea se descarta sin efecto; con él llega
    /// entera y en el orden en que se emitió, y un segundo `conectar` no lo
    /// cambia. Sobre una celda local, no la global.
    #[test]
    fn ret51b_la_linea_va_al_destino_conectado_o_a_ninguna_parte() {
        let celda = OnceLock::new();
        emitir_en(&celda, format_args!("antes {}", 0));
        assert!(lineas(&LOCALES).is_empty());
        assert!(conectar_en(&celda, a_locales));
        assert!(!conectar_en(&celda, a_ninguna_parte), "no se cambia");
        emitir_en(&celda, format_args!("uno {}", 1));
        emitir_en(&celda, format_args!("dos {}", 2));
        assert_eq!(lineas(&LOCALES), ["uno 1", "dos 2"]);
    }

    /// RET-51b: las macros dan el texto que daba la cola binaria, con el
    /// prefijo de error, y siguen valiendo como expresión de un brazo de
    /// `match` (así las usa el host en purge_orphan_legs). Es la única
    /// prueba que conecta el destino global.
    #[test]
    fn ret51b_las_macros_conservan_texto_prefijo_y_forma() {
        assert!(conectar(a_globales));
        crate::telemetry!("TEST_SYSTEM_NOMINAL_TICK: {}", 42);
        crate::telemetry_err!("TEST_SIMULATED_REJECTION: code={}", 500);
        let r: Result<(), &str> = Err("x");
        match r {
            Ok(()) => {}
            Err(e) => crate::telemetry_err!("brazo {}", e),
        }
        assert_eq!(
            lineas(&GLOBALES),
            [
                "TEST_SYSTEM_NOMINAL_TICK: 42",
                "⚠️ ERROR: TEST_SIMULATED_REJECTION: code=500",
                "⚠️ ERROR: brazo x",
            ]
        );
    }
}
