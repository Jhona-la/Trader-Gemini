//! I-2b (Línea I) — EDAD DEL DATO DESCOMPUESTA.
//!
//! El host medía `latency_ms = ahora − E` al procesar cada evento, con `E` el
//! sello de emisión de Binance. Esa cifra suma tres cosas que el código no
//! separaba: el trayecto exchange → socket (red), el error del reloj local y
//! la espera del mensaje en la cola entre el lector WS y el bucle consumidor.
//! Medido el 2026-10-24 (`docs/audit/linea_i/SONDAS_WS_2026-10-24.md`), la red
//! sola da p50 ≈ 150 ms y p99 ≤ 420 ms, frente a un umbral de pánico de unos
//! 3141 ms: los más de 200 avisos de «latencia de red» no los explica la red.
//!
//! Con el sello de recepción del lector (`cola_ws::MensajeWs::recibido`) la
//! espera local se mide con el reloj monótono y el resto es red más reloj:
//!
//! ```text
//! total        = ahora_srv − E
//! espera_local = ahora_mono − recibido_mono         (≥ 0)
//! red_reloj    = total − espera_local               (con signo)
//! ```
//!
//! `red_reloj` incluye todo lo que no es la cola local: el retardo del lote
//! del exchange, la red, lo que tarda el lector en sellar y el error de la
//! proyección del reloj del servidor. Mientras I-5 no unifique el reloj, el
//! host proyecta `ahora_srv` con el reloj de pared del arranque del bucle más
//! el tiempo monótono transcurrido y el offset NTP (medido contra el reloj de
//! pared): cualquier corrección del reloj de pared durante la sesión (`D`,
//! ver `deriva_pared_frente_monotono`) entra en `red_reloj` con el signo
//! contrario. Un `red_reloj` negativo no se recorta: la red no puede ser
//! negativa, así que delata que la proyección quedó por detrás de `E`
//! (error de reloj), y es información, no ruido.

use std::time::Duration;

/// Edad de un evento al procesarse, en milisegundos.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdadEvento {
    /// `ahora − E`: lo que el host medía como «latencia».
    pub total_ms: i64,
    /// Trayecto exchange → socket más el error del reloj local.
    pub red_reloj_ms: i64,
    /// Espera en la cola local, medida con el reloj monótono.
    pub espera_local_ms: i64,
}

impl EdadEvento {
    /// Descompone la edad de un evento emitido en `e_srv_ms` (sello de
    /// Binance) y procesado en `ahora_srv_ms` (reloj del host proyectado al
    /// del servidor), que esperó `espera_local` en la cola.
    ///
    /// Aritmética saturada: un sello o un reloj absurdos no desbordan.
    #[inline]
    pub fn medir(e_srv_ms: i64, ahora_srv_ms: i64, espera_local: Duration) -> Self {
        let total_ms = ahora_srv_ms.saturating_sub(e_srv_ms);
        let espera_local_ms = i64::try_from(espera_local.as_millis()).unwrap_or(i64::MAX);
        let red_reloj_ms = total_ms.saturating_sub(espera_local_ms);
        Self {
            total_ms,
            red_reloj_ms,
            espera_local_ms,
        }
    }
}

/// Deriva `D` del reloj de pared frente a la proyección monótona que fija el
/// host al arrancar su bucle: `D = pared_ahora − (pared_base + transcurrido)`.
///
/// `D > 0`: el reloj de pared avanzó más que el monótono (una corrección
/// hacia delante); la proyección queda por detrás y `red_reloj` baja en `D`.
/// `D < 0`: al revés. Es la parte del error de reloj que crece con la sesión
/// y la única que puede separarse de la red antes de I-5.
#[inline]
pub fn deriva_pared_frente_monotono(pared_ahora_ms: i64, pared_base_ms: i64, transcurrido: Duration) -> i64 {
    let transcurrido_ms = i64::try_from(transcurrido.as_millis()).unwrap_or(i64::MAX);
    pared_ahora_ms.saturating_sub(pared_base_ms).saturating_sub(transcurrido_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i2_un_evento_que_espero_en_cola_no_es_red_lenta() {
        // Emitido en t=1000; llegó al socket en t≈1150 y esperó 5 s en cola.
        let e = EdadEvento::medir(1_000, 6_150, Duration::from_millis(5_000));
        assert_eq!(e.total_ms, 5_150, "la cifra que el host tomaba por red");
        assert_eq!(e.espera_local_ms, 5_000);
        assert_eq!(e.red_reloj_ms, 150);
    }

    #[test]
    fn i2_sin_espera_toda_la_edad_es_red_y_reloj() {
        let e = EdadEvento::medir(1_000, 1_180, Duration::ZERO);
        assert_eq!(e.total_ms, 180);
        assert_eq!(e.red_reloj_ms, 180);
        assert_eq!(e.espera_local_ms, 0);
    }

    #[test]
    fn i2_un_error_de_reloj_da_red_negativa_visible() {
        // La proyección del reloj del servidor quedó 100 ms por detrás de E.
        let e = EdadEvento::medir(1_000, 900, Duration::ZERO);
        assert_eq!(e.total_ms, -100);
        assert_eq!(e.red_reloj_ms, -100, "no se recorta: delata el reloj");
    }

    #[test]
    fn i2_la_deriva_del_reloj_de_pared_tiene_signo() {
        // Sin corrección: la pared avanzó lo mismo que el monótono.
        assert_eq!(deriva_pared_frente_monotono(61_000, 1_000, Duration::from_secs(60)), 0);
        // w32time adelantó la pared 2 s durante la sesión.
        assert_eq!(deriva_pared_frente_monotono(63_000, 1_000, Duration::from_secs(60)), 2_000);
        // Y la retrasó 500 ms.
        assert_eq!(deriva_pared_frente_monotono(60_500, 1_000, Duration::from_secs(60)), -500);
        // Valores absurdos saturan.
        assert_eq!(deriva_pared_frente_monotono(i64::MIN, i64::MAX, Duration::MAX), i64::MIN);
    }

    #[test]
    fn i2_la_deriva_entra_en_red_reloj_con_signo_contrario() {
        // Red real 150 ms. La pared se adelantó D = 400 ms tras fijar la
        // proyección, y el offset NTP la corrige: la proyección queda 400 ms
        // por detrás del servidor y el host ve red+reloj = 150 − 400.
        let d = deriva_pared_frente_monotono(61_400, 1_000, Duration::from_secs(60));
        let e_srv = 100_000;
        let ahora_proyectado = e_srv + 150 - d;
        let edad = EdadEvento::medir(e_srv, ahora_proyectado, Duration::ZERO);
        assert_eq!(edad.red_reloj_ms + d, 150, "sumar D recupera la red real");
    }

    #[test]
    fn i2_la_descomposicion_suma_exacta() {
        for (e_srv, ahora, ms) in [(0, 0, 0), (5, 7_000, 6_000), (1_000, 2_000, 3_000), (-50, 10, 1)] {
            let d = EdadEvento::medir(e_srv, ahora, Duration::from_millis(ms));
            assert_eq!(d.red_reloj_ms + d.espera_local_ms, d.total_ms);
        }
    }

    #[test]
    fn i2_valores_extremos_saturan_sin_desbordar() {
        let d = EdadEvento::medir(i64::MIN, i64::MAX, Duration::MAX);
        assert_eq!(d.total_ms, i64::MAX);
        assert_eq!(d.espera_local_ms, i64::MAX);
        assert_eq!(d.red_reloj_ms, 0);
    }
}
