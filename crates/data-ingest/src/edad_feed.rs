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

/// I-3a (Línea I): lo que un ciclo del sistema inmune hace con los strikes
/// de latencia.
///
/// Reproduce las ramas de CERT-M4-H04 con la componente red+reloj (I-2b) en
/// vez de la edad total. Sólo diverge del inmune anterior en dos casos, y en
/// ninguno suma un strike que el anterior no sumara:
///
/// - el exceso de edad viene de la cola local (red+reloj ≤ umbral/2): conserva
///   en vez de sumar. Un atraso del consumidor no es una red degradada y
///   aplanar a mercado no lo arregla; el núcleo sigue bloqueando ENTRADAS con
///   la edad total;
/// - sin eventos públicos nuevos desde el ciclo anterior y sin stall del
///   watchdog: conserva en vez de reiniciar. La muestra es la del último
///   evento, congelada: no prueba que el feed esté sano.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CicloStrikes {
    /// red+reloj por encima del umbral.
    SumaRedLenta,
    /// red+reloj por encima de la mitad del umbral con el watchdog en stall
    /// (rama de stall de CERT-M4-H04).
    SumaRedMediaConStall,
    /// red+reloj por encima de la mitad del umbral con la edad por encima del
    /// umbral (el núcleo está en LATENCY PANIC).
    SumaRedMediaConDatosViejos,
    /// Sin prueba de que el feed esté sano: stall, sin eventos públicos nuevos
    /// o edad por encima del umbral por la cola local.
    Conserva,
    /// Feed avanzando y edad dentro del umbral.
    Reinicia,
}

impl CicloStrikes {
    /// Aplica el ciclo a los strikes previos.
    #[inline]
    pub fn aplicar(self, previos: u32) -> u32 {
        match self {
            Self::SumaRedLenta | Self::SumaRedMediaConStall | Self::SumaRedMediaConDatosViejos => {
                previos.saturating_add(1)
            }
            Self::Conserva => previos,
            Self::Reinicia => 0,
        }
    }

    /// Texto de la rama para los avisos y el motivo del aplanado.
    pub fn motivo(self) -> &'static str {
        match self {
            Self::SumaRedLenta => "red+reloj por encima del umbral",
            Self::SumaRedMediaConStall => "red+reloj por encima de la mitad del umbral con el feed en stall",
            Self::SumaRedMediaConDatosViejos => {
                "red+reloj por encima de la mitad del umbral con la edad del dato por encima del umbral"
            }
            Self::Conserva => "feed sin muestra nueva o cola local atrasada",
            Self::Reinicia => "feed sano",
        }
    }
}

/// Clasifica un ciclo del inmune (I-3a). Las comparaciones son estrictas,
/// como las de CERT-M4-H04.
///
/// La edad es `red_reloj + espera_local` (la edad total que el núcleo compara
/// con el mismo umbral para bloquear entradas). Un red+reloj negativo (error
/// de reloj, ver `EdadEvento`) cuenta como 0 ms de red y nunca suma un strike.
#[inline]
pub fn ciclo_strikes(
    red_reloj_ms: i64,
    espera_local_ms: u64,
    umbral_ms: u64,
    stalled: bool,
    sin_eventos_nuevos: bool,
) -> CicloStrikes {
    let red = u64::try_from(red_reloj_ms).unwrap_or(0);
    let edad = i128::from(red_reloj_ms) + i128::from(espera_local_ms);
    let datos_viejos = edad > i128::from(umbral_ms);
    let red_media = red > umbral_ms / 2;
    if red > umbral_ms {
        CicloStrikes::SumaRedLenta
    } else if stalled && red_media {
        CicloStrikes::SumaRedMediaConStall
    } else if datos_viejos && red_media {
        CicloStrikes::SumaRedMediaConDatosViejos
    } else if stalled || sin_eventos_nuevos || datos_viejos {
        CicloStrikes::Conserva
    } else {
        CicloStrikes::Reinicia
    }
}

/// I-3a (Línea I): strikes de latencia del sistema inmune tras un ciclo.
/// Ver [`CicloStrikes`] y [`ciclo_strikes`].
#[inline]
pub fn strikes_latencia(
    previos: u32,
    red_reloj_ms: i64,
    espera_local_ms: u64,
    umbral_ms: u64,
    stalled: bool,
    sin_eventos_nuevos: bool,
) -> u32 {
    ciclo_strikes(red_reloj_ms, espera_local_ms, umbral_ms, stalled, sin_eventos_nuevos).aplicar(previos)
}

/// I-3c (Línea I, GLM-174): ¿le toca evaluar al bosque sombra en ESTE
/// mensaje? El bosque corre en el mismo hilo que el núcleo: si la cola
/// local está atrasada (este evento esperó más que el umbral), evaluar
/// sus 10 universos suma retraso al retraso. Predicado puro sobre la
/// espera local ya medida por mensaje (I-2b) — el bosque es aprendizaje
/// estadístico: con cola sana toma todas sus muestras; con cola atrasada
/// cede el hilo para que el núcleo alcance al feed. El umbral lo fija el
/// llamador (en el host, la mitad del umbral de pánico del genoma —
/// adaptativo, no un número mágico).
#[inline]
pub fn toca_sombra(espera_local_ms: u64, umbral_atraso_ms: u64) -> bool {
    espera_local_ms <= umbral_atraso_ms
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

    /// El inmune anterior (CERT-M4-H04) sobre la edad total del dato.
    fn strikes_anteriores(previos: u32, edad_total_ms: u64, umbral_ms: u64, stalled: bool) -> u32 {
        if edad_total_ms > umbral_ms || (stalled && edad_total_ms > umbral_ms / 2) {
            previos.saturating_add(1)
        } else if stalled {
            previos
        } else {
            0
        }
    }

    #[test]
    fn i3_atraso_local_no_aplana() {
        // La cifra mezclada de hoy (5 150 ms) sumaría 3 strikes y aplanaría;
        // con la cola atrasada los strikes se conservan (en 0) y no crecen.
        let edad = EdadEvento::medir(1_000, 6_150, Duration::from_millis(5_000));
        let espera = u64::try_from(edad.espera_local_ms).unwrap_or(0);
        let mut s = 0;
        for _ in 0..3 {
            s = strikes_latencia(s, edad.red_reloj_ms, espera, 3_141, false, false);
        }
        assert_eq!(s, 0);
        assert_eq!(ciclo_strikes(edad.red_reloj_ms, espera, 3_141, false, false), CicloStrikes::Conserva);
    }

    #[test]
    fn i3_datos_viejos_no_borran_strikes_previos() {
        // Con la cola 5 s atrasada, una red sana medida en eventos viejos no
        // reinicia los strikes que ya había: se conservan.
        assert_eq!(strikes_latencia(2, 150, 5_000, 3_141, false, false), 2);
        // Y una red media con la edad por encima del umbral suma, como antes.
        assert_eq!(strikes_latencia(2, 1_600, 5_000, 3_141, false, false), 3);
    }

    #[test]
    fn i3_banda_mixta_red_degradada_con_cola_moderada_suma() {
        // Red 2,5 s (6 veces el p99 medido) y 700 ms de cola: edad 3,2 s por
        // encima del umbral y el núcleo en LATENCY PANIC. El inmune anterior
        // sumaba; este también, aunque la cola sola no pase el umbral.
        assert_eq!(strikes_latencia(0, 2_500, 700, 3_141, false, false), 1);
        assert_eq!(
            ciclo_strikes(2_500, 700, 3_141, false, false),
            CicloStrikes::SumaRedMediaConDatosViejos
        );
        // Con la red bajo la mitad del umbral el exceso es de la cola: conserva.
        assert_eq!(strikes_latencia(2, 1_000, 2_500, 3_141, false, false), 2);
    }

    #[test]
    fn i3_feed_quieto_sin_stall_conserva_y_no_suma_por_la_banda_media() {
        // Sin eventos públicos nuevos la muestra es la del último evento: no
        // reinicia, pero tampoco suma por la banda media antes de que el
        // watchdog marque stall (paridad con CERT-M4-H04).
        let mut s = 2;
        for _ in 0..3 {
            s = strikes_latencia(s, 1_600, 0, 3_141, false, true);
        }
        assert_eq!(s, 2);
        // Una red congelada por encima del umbral sí suma, como antes.
        assert_eq!(strikes_latencia(2, 3_500, 0, 3_141, false, true), 3);
    }

    #[test]
    fn i3_red_lenta_sostenida_si_dispara() {
        let mut s = 0;
        for _ in 0..3 {
            s = strikes_latencia(s, 4_000, 0, 3_141, false, false);
        }
        assert_eq!(s, 3);
    }

    #[test]
    fn i3_paridad_con_cert_m4_h04() {
        // stall puro: conserva; stall con red media: suma; sano: reinicia.
        assert_eq!(strikes_latencia(2, 100, 0, 3_141, true, false), 2);
        assert_eq!(strikes_latencia(2, 1_600, 0, 3_141, true, false), 3);
        assert_eq!(strikes_latencia(2, 1_600, 0, 3_141, false, false), 0);
        assert_eq!(strikes_latencia(2, 3_142, 0, 3_141, false, false), 3);
        assert_eq!(strikes_latencia(2, 3_141, 0, 3_141, false, false), 0, "el umbral es estricto");
    }

    #[test]
    fn i3_limites_estrictos() {
        // La edad igual al umbral no son datos viejos.
        assert_eq!(strikes_latencia(2, 150, 2_991, 3_141, false, false), 0);
        assert_eq!(strikes_latencia(2, 150, 2_992, 3_141, false, false), 2);
        // La mitad del umbral es estricta (3_141 / 2 = 1_570).
        assert_eq!(strikes_latencia(2, 1_570, 0, 3_141, true, false), 2);
        assert_eq!(strikes_latencia(2, 1_571, 0, 3_141, true, false), 3);
        assert_eq!(strikes_latencia(2, 1_570, 1_572, 3_141, false, false), 2);
        assert_eq!(strikes_latencia(2, 1_571, 1_571, 3_141, false, false), 3);
    }

    #[test]
    fn i3_nunca_suma_ni_reinicia_donde_el_inmune_anterior_no_lo_hacia() {
        // Barrido: donde el nuevo suma, el anterior (sobre la edad total)
        // también sumaba; donde el nuevo reinicia, el anterior también. Las
        // únicas diferencias son conservar.
        let umbrales = [500_u64, 3_141, 10_000];
        let redes = [
            -10_000_i64, -500, 0, 100, 249, 250, 251, 499, 500, 501, 1_570, 1_571, 2_500, 3_141, 3_142, 5_000, 10_001,
        ];
        let esperas = [0_u64, 1, 250, 700, 1_572, 2_991, 2_992, 5_000, 12_000];
        for &u in &umbrales {
            for &r in &redes {
                for &e in &esperas {
                    for &stalled in &[false, true] {
                        for &quieto in &[false, true] {
                            let total = u64::try_from(r.saturating_add(e as i64)).unwrap_or(0);
                            let nuevo = strikes_latencia(1, r, e, u, stalled, quieto);
                            let anterior = strikes_anteriores(1, total, u, stalled);
                            let caso = format!("red {r}, espera {e}, umbral {u}, stall {stalled}, quieto {quieto}");
                            if nuevo == 2 {
                                assert_eq!(anterior, 2, "suma de más: {caso}");
                            }
                            if nuevo == 0 {
                                assert_eq!(anterior, 0, "reinicia de más: {caso}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn i3b_sin_reinicio_un_stall_puro_tras_el_latch_volveria_a_aplanar() {
        // Tras un latch por latencia los strikes quedan en 3; un stall puro
        // los conserva, así que sin el reinicio del rearme el primer ciclo
        // volvía a cumplir `>= 3` sin ninguna muestra lenta nueva.
        assert_eq!(strikes_latencia(3, 100, 0, 3_141, true, false), 3);
        assert_eq!(strikes_latencia(0, 100, 0, 3_141, true, false), 0);
    }

    #[test]
    fn i3_reloj_retrasado_cuenta_como_cero_y_satura() {
        assert_eq!(strikes_latencia(5, -10_000, 0, 3_141, false, false), 0);
        assert_eq!(strikes_latencia(u32::MAX, i64::MAX, 0, 3_141, false, false), u32::MAX);
        assert_eq!(strikes_latencia(1, i64::MIN, u64::MAX, 3_141, true, true), 1);
    }
}
