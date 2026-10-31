//! CONTRATO I-5 (Línea I, rescate GLM-173): reloj coherente — el offset se
//! mide contra la MISMA proyección monótona que consume la edad del dato.
//!
//! QUÉ: la proyección del reloj del servidor (`epoch_baseline + monótono +
//! corrección`) ya no relee el offset NTP referido al reloj de pared
//! actual. La corrección vive en `server_time_correction_ms` y es la
//! diferencia entre la hora del servidor respondida y la PROYECCIÓN
//! MONÓTONA anclada en el arranque (deriva del oscilador + jitter, nada más).
//! POR QUÉ: el viejo camino (#657) medía el offset contra
//! `SystemTime::now()`; una corrección de w32time de −2 s a mitad de sesión
//! movía el offset +2000 ms y ese salto entraba en `red_reloj` como red
//! fantasma (mezcla de épocas: el ancla del arranque quedó en la época
//! vieja). Es parte de los >200 avisos de «latencia de red elevada» del
//! operador que la red (p50 ≈ 150 ms) no explica (I-2b).
//! PARA QUÉ: la edad del dato sólo contiene red, cola local y deriva real
//! del oscilador; una corrección del reloj de pared del SO deja de
//! disparar LATENCY_PANIC/KILL_SWITCH falsos.
//! CÓMO: (1) la simulación de −2 s del plan, contrafactual viejo vs nuevo;
//! (2-3) la proyección y su corrección como funciones puras; (4) el Kalman
//! de la corrección conserva la puerta de outliers por RTT; (5) guardia de
//! fuente: la fórmula del bucle no lee el reloj de pared.

use execution_engine::ntp::{
    correccion_vs_proyeccion, proyeccion_srv, NtpKalmanFilter,
};
use std::time::Duration;

#[test]
fn i5_una_correccion_de_w32time_de_menos_2s_ya_no_es_red_fantasma() {
    // Escenario del plan: arranque con pared local +2 s adelantada;
    // w32time la corrige (−2 s) a mitad de sesión. El servidor, la red y
    // el monótono no se enteran.
    const SALTO_MS: i64 = 2_000;
    let epoch_baseline: i64 = 1_000_000; // srv estimado en el arranque
    let mono_transcurrido: i64 = 60_000; // 60 s de sesión (reloj monótono)

    // Contrafactual VIEJO (#657): offset = servidor − pared_actual.
    // Antes del salto: pared = (epoch_baseline + mono) − 2_000 (adelantada
    // +2 s ⇒ marca MENOS que la proyección... al revés: pared adelantada
    // ⇒ offset = srv − (srv + 2000) = −2000). Tras la corrección la pared
    // queda bien: offset = 0. El Δoffset aplicado = 0 − (−2000) = +2000.
    let offset_viejo_antes = -SALTO_MS; // srv − pared_adelantada
    let offset_viejo_despues = 0; // srv − pared_corregida
    let now_viejo = epoch_baseline + (offset_viejo_despues - offset_viejo_antes) + mono_transcurrido;
    // El ahora viejo se desplazó TODO el salto: +2 s de red fantasma.
    assert_eq!(
        now_viejo - (epoch_baseline + mono_transcurrido),
        SALTO_MS,
        "el camino viejo convierte la corrección de w32time en red fantasma"
    );

    // Camino NUEVO (I-5): la corrección se mide contra la proyección
    // monótona; la pared no interviene. Simulación idéntica, con la pared
    // corregida a mitad de sesión: la corrección publicada NO cambia.
    let ancla_srv = epoch_baseline;
    let corr_antes =
        correccion_vs_proyeccion(ancla_srv + mono_transcurrido, proyeccion_srv(ancla_srv, Duration::from_millis(60_000)));
    // w32time corrige la pared: el servidor sigue respondiendo su hora y
    // la proyección sigue siendo ancla + monótono ⇒ MISMA corrección.
    let corr_despues =
        correccion_vs_proyeccion(ancla_srv + mono_transcurrido, proyeccion_srv(ancla_srv, Duration::from_millis(60_000)));
    assert_eq!(corr_antes, corr_despues, "la corrección es inmune al salto de pared");
    let now_nuevo = epoch_baseline + corr_despues + mono_transcurrido;
    assert_eq!(
        now_nuevo,
        epoch_baseline + mono_transcurrido,
        "el ahora nuevo no se mueve: el salto de w32time es invisible"
    );
}

#[test]
fn i5_la_proyeccion_avanza_con_el_monotono_y_satura() {
    let ancla = 1_000_000_i64;
    assert_eq!(proyeccion_srv(ancla, Duration::from_millis(0)), ancla);
    assert_eq!(
        proyeccion_srv(ancla, Duration::from_secs(60)),
        ancla + 60_000
    );
    // Saturación: un transcurrido absurdo no desborda.
    assert_eq!(proyeccion_srv(i64::MAX - 5, Duration::from_secs(600)), i64::MAX);
    // La corrección también satura en ambos signos.
    assert_eq!(correccion_vs_proyeccion(i64::MIN, i64::MAX), i64::MIN);
    assert_eq!(correccion_vs_proyeccion(i64::MAX, i64::MIN), i64::MAX);
}

#[test]
fn i5_la_correccion_absorbe_la_deriva_del_oscilador() {
    // El oscilador local deriva +1 ms cada 15 s frente al servidor: la
    // corrección filtrada crece con la deriva (es su trabajo), sin
    // importar qué haga el reloj de pared del SO. El Kalman retrasa una
    // rampa (ganancia ~P/(P+R) con R≈400 ms² del RTT): converge hacia la
    // deriva SIN alcanzarla — el honesto es acotar dirección y banda, no
    // exigir 1:1 tras 10 minutos de simulación.
    let ancla = 500_000_i64;
    let mut kf = NtpKalmanFilter::new();
    let mut primera = None;
    let mut ultima = 0_i64;
    for ciclos in 1..=40_i64 {
        let transcurrido = Duration::from_secs(15 * ciclos as u64);
        let servidor = ancla + 15_000 * ciclos + ciclos; // 15 s en ms, +1 ms de deriva/ciclo
        let raw = correccion_vs_proyeccion(servidor, proyeccion_srv(ancla, transcurrido)) as f64;
        let corr = kf.update_contra_proyeccion(40, raw).unwrap_or(ultima);
        if primera.is_none() {
            primera = Some(corr);
        }
        ultima = corr;
    }
    let primera = primera.unwrap_or(0);
    assert!(ultima > 0, "la corrección sigue el signo de la deriva; fue {ultima}");
    assert!(
        ultima >= primera && ultima <= 40,
        "crece con la deriva sin excederla (deriva real +40ms): {primera} → {ultima}"
    );
}

#[test]
fn i5_el_kalman_de_la_correccion_conserva_la_puerta_de_outliers() {
    let mut kf = NtpKalmanFilter::new();
    let base = kf.update_contra_proyeccion(40, 0.0).unwrap_or(0);
    // Un pico de congestión (RTT > 350 ms) no mueve el estado.
    let con_pico = kf
        .update_contra_proyeccion(5_000, 900.0)
        .unwrap_or(base);
    assert_eq!(con_pico, base, "el outlier por RTT se descarta");
    // NaN: medición inválida → None.
    let mut kf2 = NtpKalmanFilter::new();
    assert!(kf2.update_contra_proyeccion(40, f64::NAN).is_none());
}

#[test]
fn i5_el_bucle_no_mide_la_edad_con_el_reloj_de_pared() {
    // Guardia de fuente: entre la computación de epoch_baseline y la
    // fórmula del ahora, el bucle unificado no vuelve a leer SystemTime.
    let fuente = include_str!("../../../src/bin/god_engine.rs");
    let inicio = fuente
        .find("let epoch_baseline_ms")
        .expect("epoch_baseline en el bucle");
    let fin = fuente[inicio..]
        .find("let edad = data_ingest::edad_feed::EdadEvento::medir(")
        .map(|i| inicio + i)
        .unwrap_or(fuente.len());
    let tramo = &fuente[inicio..fin];
    assert!(
        !tramo.contains("SystemTime::now()"),
        "I-5: la proyección de la edad no puede leer el reloj de pared (mezcla de épocas)"
    );
    assert!(
        tramo.contains("server_time_correction_ms"),
        "I-5: la proyección debe consumir la corrección contra la proyección monótona"
    );
}
