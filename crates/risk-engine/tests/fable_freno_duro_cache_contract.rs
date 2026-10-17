//! F-A4 (Fable, 2026-10-17; hallazgo F1-012) — la caché del umbral del freno
//! duro no bloquea el hilo del evento.
//!
//! Teoría que se prueba. `K(w̄, l̄)` es el cuantil 95 % de la caída máxima en
//! 200 operaciones bajo H0 (+0,05 R), estimado por Monte Carlo determinista
//! (20 000 caminos). Hasta F-A4 ese Monte Carlo corría en el hilo del evento
//! cada vez que la geometría redondeada (0,02 stops) cambiaba. El contrato:
//! (a) el umbral exacto no cambia: la caché devuelve exactamente la función
//! pura; (b) con un vecino ya calculado a ≤ 0,10 stops la respuesta es
//! inmediata y CONSERVADORA (el mínimo K de los vecinos: el freno dispara
//! antes, nunca después) mientras un hilo de fondo calcula el exacto; (c) en
//! frío sin vecinos se calcula una vez, de forma síncrona; (d) geometrías no
//! finitas no tienen umbral.
//!
//! Las pruebas comparten la caché estática del proceso: cada una usa una
//! región de geometría separada ≥ 0,30 stops de las demás (radio 0,10).
use risk_engine::drawdown::{
    caida_alfa_en_stops, caida_alfa_freno_duro, freno_duro_calculos_pendientes,
    freno_duro_esperar_pendientes, ALFA_FALSACION, PASO_CACHE_FRENO_DURO,
    RADIO_VECINO_FRENO_DURO, VENTAJA_H0_FRENO_DURO_R,
};
use std::sync::Mutex;
use std::time::{Duration, Instant};

static SERIAL_TEST: Mutex<()> = Mutex::new(());

fn exacto(w: f64, l: f64) -> Option<f64> {
    caida_alfa_en_stops(w, l, VENTAJA_H0_FRENO_DURO_R, ALFA_FALSACION)
}

fn espera() -> Duration {
    // El Monte Carlo en depuración tarda segundos; en release, milisegundos.
    Duration::from_secs(120)
}

#[test]
fn en_frio_sin_vecinos_el_umbral_es_el_exacto_y_no_queda_pendiente() {
    let _g = SERIAL_TEST.lock().unwrap_or_else(|e| e.into_inner());
    // Región w ≈ 1,50 (sin vecinos previos en esta región).
    let k = caida_alfa_freno_duro(1.50, 1.10);
    assert_eq!(k, exacto(1.50, 1.10), "en frío la caché devuelve la función pura");
    assert!(k.is_some_and(|k| k.is_finite() && k > 0.0));
    // Un segundo acceso es un acierto: mismo valor.
    assert_eq!(caida_alfa_freno_duro(1.50, 1.10), k);
    assert!(freno_duro_esperar_pendientes(espera()));
}

#[test]
fn con_un_vecino_la_respuesta_es_inmediata_conservadora_y_luego_exacta() {
    let _g = SERIAL_TEST.lock().unwrap_or_else(|e| e.into_inner());
    // Región w ≈ 3,00.
    let k_vecino = caida_alfa_freno_duro(3.00, 1.00).expect("umbral del vecino");
    assert!(freno_duro_esperar_pendientes(espera()));
    // Geometría a un paso (0,02) del vecino: todavía no calculada.
    let t0 = Instant::now();
    let k_rapido = caida_alfa_freno_duro(3.00 + PASO_CACHE_FRENO_DURO, 1.00);
    let tardo = t0.elapsed();
    assert_eq!(
        k_rapido,
        Some(k_vecino),
        "con un único vecino el respaldo es su K (mínimo de los vecinos)"
    );
    assert!(
        tardo < Duration::from_millis(250),
        "la ruta con vecino no corre el Monte Carlo en el hilo del evento: tardó {tardo:?}"
    );
    // El exacto llega por detrás y sustituye al respaldo.
    assert!(
        freno_duro_esperar_pendientes(espera()),
        "el cálculo de fondo debe terminar"
    );
    let k_exacto = caida_alfa_freno_duro(3.00 + PASO_CACHE_FRENO_DURO, 1.00);
    assert_eq!(k_exacto, exacto(3.00 + PASO_CACHE_FRENO_DURO, 1.00));
    assert_eq!(freno_duro_calculos_pendientes(), 0);
}

#[test]
fn con_varios_vecinos_el_respaldo_es_el_minimo() {
    let _g = SERIAL_TEST.lock().unwrap_or_else(|e| e.into_inner());
    // Región w ≈ 0,80: dos vecinos exactos a ambos lados de la consulta.
    let k_a = caida_alfa_freno_duro(0.80, 1.00).expect("K(0,80)");
    assert!(freno_duro_esperar_pendientes(espera()));
    // 0,86 está a 3 pasos de 0,80: entra con respaldo y se calcula de fondo.
    let _ = caida_alfa_freno_duro(0.86, 1.00);
    assert!(freno_duro_esperar_pendientes(espera()));
    let k_b = caida_alfa_freno_duro(0.86, 1.00).expect("K(0,86) exacto");
    assert_eq!(Some(k_b), exacto(0.86, 1.00));
    // La consulta en medio ve a los dos como vecinos.
    let k_medio = caida_alfa_freno_duro(0.82, 1.00).expect("respaldo");
    assert_eq!(k_medio, k_a.min(k_b), "el respaldo es el mínimo de los vecinos");
    assert!(k_medio <= k_a && k_medio <= k_b, "conservador: nunca mayor que un vecino");
    assert!(freno_duro_esperar_pendientes(espera()));
    assert_eq!(caida_alfa_freno_duro(0.82, 1.00), exacto(0.82, 1.00));
}

#[test]
fn fuera_del_radio_no_hay_respaldo_y_se_calcula_sincrono() {
    let _g = SERIAL_TEST.lock().unwrap_or_else(|e| e.into_inner());
    assert!(freno_duro_esperar_pendientes(espera()));
    // Región w ≈ 5,00; la consulta siguiente queda a RADIO + 1 pasos.
    let _ = caida_alfa_freno_duro(5.00, 1.00);
    assert!(freno_duro_esperar_pendientes(espera()));
    let lejos = 5.00 + PASO_CACHE_FRENO_DURO * (RADIO_VECINO_FRENO_DURO + 1) as f64;
    let k = caida_alfa_freno_duro(lejos, 1.00);
    assert_eq!(k, exacto(lejos, 1.00), "sin vecino en el radio se calcula el exacto al momento");
    assert_eq!(freno_duro_calculos_pendientes(), 0);
}

#[test]
fn geometrias_no_finitas_no_tienen_umbral() {
    assert_eq!(caida_alfa_freno_duro(f64::NAN, 1.0), None);
    assert_eq!(caida_alfa_freno_duro(1.0, f64::INFINITY), None);
    assert_eq!(caida_alfa_freno_duro(f64::NEG_INFINITY, 1.0), None);
}
