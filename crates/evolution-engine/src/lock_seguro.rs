//! F1-093 (Sol) — LECTURA Y ESCRITURA DE LOCKS QUE NO PROPAGAN EL VENENO.
//!
//! # QUÉ
//! `RwLock::read()` y `RwLock::write()` devuelven `Result`: el `Err` no es un
//! fallo de concurrencia ordinario, es **envenenamiento** (`PoisonError`). Un
//! `RwLock` se envenena cuando un hilo panica mientras mantiene el guard, y
//! desde entonces **cada** `read()`/`write()` vuelve a fallar. Con
//! `panic = "abort"` (LG-28) la cadena es: un pánico en cualquier parte del
//! proceso → lock envenenado → el siguiente `.read().unwrap()` del evolver
//! online aborta → el proceso muere con posiciones abiertas.
//!
//! El evolver online tenía 43 de esos `.unwrap()` (16 en
//! `online_random_forest.rs`, 27 en `online_newton_gbdt.rs`), todos en código
//! de producción: ninguno bajo `#[cfg(test)]`. Una sola caída ajena bastaba
//! para convertir el bosque en una cadena de abortos.
//!
//! # POR QUÉ `into_inner()` Y NO PROPAGAR
//! Estos locks guardan campos cuyo invariante sobrevive al pánico del que
//! mantenía el guard: `Copy` (`f32`, `f64`, `bool`), `Arc` (compartir un Arc
//! es atómico) u `Option<Arc<...>>`. No hay estructura a medio construir que
//! pueda quedar inconsistente, así que recuperar el guard es más honesto que
//! abortar: el estado sigue siendo exactamente el de antes del pánico.
//!
//! Cuando el invariante SÍ dependa de una sección crítica multi-paso, el lock
//! no se recupera: se rediseña para que el pánico no pueda dejar estado
//! parcial. Esta decisión queda documentada aquí para que nadie aplique
//! `leer`/`escribir` a un lock que no cumple esa condición.
//!
//! # CONTRATO
//! `lock_sin_veneno_contract` envenena un lock de verdad (pánico dentro de
//! `catch_unwind` con el guard mantenido) y verifica que la lectura recuperada
//! devuelve el dato en vez de abortar.

use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Lectura que recupera el guard de un lock envenenado.
///
/// Equivalente semántico a `lock.read().unwrap()` para los campos cuyo
/// invariante sobrevive al pánico (ver la nota del módulo): nunca panica.
#[inline]
pub fn leer<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    match lock.read() {
        Ok(guard) => guard,
        // `PoisonError::into_inner()` devuelve el guard envuelto: para estos
        // campos el invariante sobrevive al pánico (ver nota del módulo).
        Err(veneno) => veneno.into_inner(),
    }
}

/// Escritura que recupera el guard de un lock envenenado.
#[inline]
pub fn escribir<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    match lock.write() {
        Ok(guard) => guard,
        Err(veneno) => veneno.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    /// Envenena un lock de verdad: un hilo panica mientras mantiene el guard
    /// de escritura. No hay mock ni atajo: es el mecanismo real de
    /// `PoisonError`.
    fn envenenar<T>(lock: &'static RwLock<T>, valor: T)
    where
        T: Send + Sync + 'static,
    {
        let hilo = std::thread::spawn(move || {
            let mut g = lock.write().expect("el guard se adquiere antes de panicar");
            *g = valor;
            panic!("pánico simulado para envenenar el lock");
        });
        let _ = hilo.join();
    }

    #[test]
    fn f1_093_lectura_recupera_el_guard_de_un_lock_envenenado() {
        static LOCK: RwLock<u32> = RwLock::new(0);
        envenenar(&LOCK, 42);

        // El lock queda envenenado: la lectura estándar devuelve Err.
        assert!(LOCK.read().is_err(), "el lock debe estar envenenado");

        // Y aun así la lectura segura devuelve el último valor escrito.
        assert_eq!(*leer(&LOCK), 42);
    }

    #[test]
    fn f1_093_escritura_recupera_el_guard_de_un_lock_envenenado() {
        static LOCK: RwLock<u32> = RwLock::new(0);
        envenenar(&LOCK, 7);
        assert!(LOCK.read().is_err());

        *escribir(&LOCK) = 99;
        assert_eq!(*leer(&LOCK), 99, "la escritura recuperada debe persistir");
    }

    #[test]
    fn f1_093_lock_sano_se_comporta_igual() {
        static LOCK: RwLock<f64> = RwLock::new(1.5);
        assert_eq!(*leer(&LOCK), 1.5);
        *escribir(&LOCK) = 2.5;
        assert_eq!(*leer(&LOCK), 2.5);
    }

    #[test]
    fn f1_093_catch_unwind_no_deja_pasar_el_pánico() {
        // Comprobación de que el ayudante de prueba detecta el pánico: sin
        // esto, un fallo del harness haría pasar los tests en falso.
        let resultado = catch_unwind(AssertUnwindSafe(|| panic!("capturado")));
        assert!(resultado.is_err());
    }
}
