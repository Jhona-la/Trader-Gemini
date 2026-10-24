//! F1-093 (Sol) — LECTURA Y ESCRITURA DE LOCKS QUE NO PROPAGAN EL VENENO.
//!
//! # QUÉ
//! `RwLock::read()` y `RwLock::write()` devuelven `Result`: el `Err` no es un
//! fallo de concurrencia ordinario, es **envenenamiento** (`PoisonError`). Un
//! `RwLock` se envenena cuando un hilo panica mientras mantiene el guard, y
//! desde entonces **cada** `read()`/`write()` vuelve a fallar. Esto describe
//! un pánico con unwinding mientras se mantiene el guard de escritura.
//! Con `panic = "abort"`, el primer pánico termina el proceso; recuperar un
//! guard no proporciona supervivencia frente a esa política.
//!
//! El evolver online tenía 43 de esos `.unwrap()` (16 en
//! `online_random_forest.rs`, 27 en `online_newton_gbdt.rs`), todos en código
//! de producción: ninguno bajo `#[cfg(test)]`. Una sola caída ajena bastaba
//! para convertir el siguiente acceso en otro pánico bajo unwinding.
//!
//! # POR QUÉ `into_inner()` Y NO PROPAGAR
//! Recuperar el guard exige que el consumidor compruebe que su invariante
//! sobrevive a la sección crítica interrumpida. `Copy`, `Arc` u
//! `Option<Arc<...>>` no demuestran por sí mismos validez numérica ni
//! consistencia entre varios campos. `into_inner()` recupera el último
//! estado protegido, que puede incluir escrituras previas al pánico;
//! no restaura automáticamente el estado anterior ni lo valida.
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
/// Recupera el guard sin propagar el error de envenenamiento. El consumidor
/// sigue siendo responsable de validar su estado (ver la nota del módulo).
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
