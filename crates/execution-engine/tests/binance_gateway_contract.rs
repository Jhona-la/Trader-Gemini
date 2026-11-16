//! PUERTA-REST: contrato de la puerta REST única (`binance_gateway`).
//!
//! Sin red a Binance: las pruebas de contabilidad usan un reloj inyectado y
//! las de transporte (P4) un servidor local en 127.0.0.1.

use execution_engine::binance_api::{
    COSTE_CONSULTA_ORDEN, COSTE_ORDEN, COSTE_POSITION_RISK, LIMITE_ORDENES_10S, LIMITE_PESO_1M,
    VENTANA_ORDENES_10S_MS, VENTANA_ORDENES_1M_MS, VENTANA_PESO_MS,
};
use execution_engine::binance_gateway::{
    CallPriority, Desenlace, LimitesExchange, PresupuestoPuerta, RechazoPuerta, CONTADOR_ORDENES_10S,
    CONTADOR_ORDENES_1M, CONTADOR_PESO_1M, FRACCION_FRENO_LECTURA,
};
use execution_engine::client::BinanceRateLimits;

/// Un instante cualquiera del reloj de prueba, lejos del origen.
const T0: u64 = 1_800_000_000_000;

fn cabecera_peso(uso: usize) -> BinanceRateLimits {
    BinanceRateLimits {
        weight_1m: Some(uso),
        ..BinanceRateLimits::default()
    }
}

// ---------------------------------------------------------------- P1 ----

/// P1: las tres ventanas, con `max(local, cabecera)` como
/// `OrderExecutor::observar_limites`.
#[test]
fn gateway_presupuesto_tres_ventanas() {
    let mut p = PresupuestoPuerta::nuevo(LimitesExchange::respaldo_lg12());
    assert_eq!(p.uso_efectivo(T0), [0, 0, 0]);

    // Una crítica de peso (positionRisk) reserva su coste antes de salir:
    // estimación local y libro en vuelo.
    let c = COSTE_POSITION_RISK.peso;
    let reserva = p.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0);
    assert!(reserva.is_ok());
    assert_eq!(p.uso_local(T0)[CONTADOR_PESO_1M], c);
    assert_eq!(p.en_vuelo()[CONTADOR_PESO_1M], c);
    assert_eq!(p.uso_efectivo(T0)[CONTADOR_PESO_1M], c);

    // Su respuesta trae una cabecera mayor que lo local (consumo de otros
    // procesos con la misma IP): la cabecera manda y el coste sale del libro
    // en vuelo (la cabecera ya lo cuenta).
    let cab = c * 20;
    if let Ok(r) = reserva {
        p.cerrar(r, Desenlace::Respuesta(cabecera_peso(cab)), T0 + 1);
    }
    assert_eq!(p.en_vuelo(), [0, 0, 0]);
    assert_eq!(p.uso_efectivo(T0 + 1)[CONTADOR_PESO_1M], cab);
    // Una cabecera menor que llega después no la rebaja dentro de la
    // ventana (máximo de las vigentes, CL-180).
    p.observar(&cabecera_peso(cab / 2), T0 + 2);
    assert_eq!(p.uso_efectivo(T0 + 2)[CONTADOR_PESO_1M], cab);
    // Con una petición en vuelo, su coste se suma a la cabecera.
    let en_vuelo = p.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0 + 2);
    assert_eq!(p.uso_efectivo(T0 + 2)[CONTADOR_PESO_1M], cab + c);

    // Lo local manda cuando las respuestas no traen cabecera.
    let mut q = PresupuestoPuerta::nuevo(LimitesExchange::respaldo_lg12());
    q.observar(&cabecera_peso(1), T0);
    for _ in 0..3 {
        if let Ok(r) = q.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0) {
            q.cerrar(r, Desenlace::Respuesta(BinanceRateLimits::default()), T0);
        }
    }
    assert_eq!(q.en_vuelo(), [0, 0, 0]);
    assert_eq!(q.uso_efectivo(T0)[CONTADOR_PESO_1M], c * 3);

    // Sin respuesta: el coste queda perdido sobre la cabecera hasta que
    // caduca la ventana (SinRespuesta del ejecutor, CL-201).
    let mut s = PresupuestoPuerta::nuevo(LimitesExchange::respaldo_lg12());
    if let Ok(r) = s.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0) {
        s.cerrar(r, Desenlace::SinRespuesta, T0);
    }
    s.observar(&cabecera_peso(cab), T0 + 1);
    assert_eq!(s.uso_efectivo(T0 + 1)[CONTADOR_PESO_1M], cab + c);
    assert_eq!(s.uso_efectivo(T0 + VENTANA_PESO_MS)[CONTADOR_PESO_1M], cab);
    // Un 429/418 no registra su cabecera y sale del libro en vuelo.
    let mut z = PresupuestoPuerta::nuevo(LimitesExchange::respaldo_lg12());
    if let Ok(r) = z.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0) {
        z.cerrar(r, Desenlace::LimiteExchange, T0);
    }
    assert_eq!(z.en_vuelo(), [0, 0, 0]);
    assert_eq!(z.uso_efectivo(T0)[CONTADOR_PESO_1M], c, "lo local sigue contando");

    // Las órdenes suman a los dos contadores ORDERS; cada uno caduca con su
    // ventana.
    let mut r = PresupuestoPuerta::nuevo(LimitesExchange::respaldo_lg12());
    if let Ok(res) = r.admitir(CallPriority::Critica, COSTE_ORDEN, T0) {
        assert_eq!(r.uso_efectivo(T0), [0, COSTE_ORDEN.ordenes, COSTE_ORDEN.ordenes]);
        r.cerrar(res, Desenlace::Respuesta(BinanceRateLimits::default()), T0);
    }
    r.observar(
        &BinanceRateLimits {
            weight_1m: None,
            orders_10s: Some(7),
            orders_1m: Some(9),
        },
        T0,
    );
    assert_eq!(r.uso_efectivo(T0), [0, 7, 9]);
    let tras_10s = T0 + VENTANA_ORDENES_10S_MS;
    let uso = r.uso_efectivo(tras_10s);
    assert_eq!(uso[CONTADOR_ORDENES_10S], 0, "la ventana de 10 s caducó");
    assert_eq!(uso[CONTADOR_ORDENES_1M], 9, "la de 1 min sigue");
    assert_eq!(r.uso_efectivo(T0 + VENTANA_ORDENES_1M_MS), [0, 0, 0]);

    // El peso caduca con su ventana, lo local y la cabecera.
    if let Ok(res) = en_vuelo {
        p.cerrar(res, Desenlace::Respuesta(BinanceRateLimits::default()), T0 + 2);
    }
    assert_eq!(p.uso_efectivo(T0 + 2 + VENTANA_PESO_MS)[CONTADOR_PESO_1M], 0);
}

/// P1: Crítica reserva su coste y pasa en la franja (0,8·L, L]; Alta,
/// Lectura y Batch frenan por encima de 0,8·L (CL-187). Regla de ruina: lo
/// que admite una prioridad menor lo admite cualquier mayor, en todo el
/// rango del uso.
#[test]
fn gateway_prioridades_admiten_en_orden() {
    let limites = LimitesExchange::respaldo_lg12();
    let l = limites.peso_1m.limite;
    assert_eq!(l, LIMITE_PESO_1M);
    let franja = (l as f64 * FRACCION_FRENO_LECTURA) as usize;
    let lectura = COSTE_CONSULTA_ORDEN;

    let con_uso = |uso: usize| {
        let mut p = PresupuestoPuerta::nuevo(limites);
        p.observar(&cabecera_peso(uso), T0);
        p
    };

    // En la franja del 80 %: todas pasan.
    for pr in CallPriority::TODAS {
        assert!(con_uso(franja).cabe(pr, lectura, T0).is_ok(), "{pr:?} en 0,8·L");
    }
    // Por encima: sólo Crítica.
    for pr in [CallPriority::Alta, CallPriority::Lectura, CallPriority::Batch] {
        assert!(
            matches!(con_uso(franja + 1).cabe(pr, lectura, T0), Err(RechazoPuerta::Presupuesto(_))),
            "{pr:?} por encima de 0,8·L"
        );
    }
    assert!(con_uso(franja + 1)
        .cabe(CallPriority::Critica, COSTE_POSITION_RISK, T0)
        .is_ok());

    // Crítica: hasta uso + coste = L.
    let c = COSTE_POSITION_RISK.peso;
    assert!(con_uso(l - c).cabe(CallPriority::Critica, COSTE_POSITION_RISK, T0).is_ok());
    assert!(con_uso(l - c + 1)
        .cabe(CallPriority::Critica, COSTE_POSITION_RISK, T0)
        .is_err());
    // Una orden (peso 0) sale con el peso justo en L: no gasta peso.
    assert!(con_uso(l).cabe(CallPriority::Critica, COSTE_ORDEN, T0).is_ok());
    assert!(con_uso(l + 1).cabe(CallPriority::Critica, COSTE_ORDEN, T0).is_err());

    // Crítica RESERVA: dos que juntas pasarían del límite no caben las dos.
    let mut p = con_uso(l - c - 1);
    assert!(p.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0).is_ok());
    assert!(p.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0).is_err());

    // Orden de admisión en todo el rango del peso y del contador de 10 s.
    let orden = CallPriority::TODAS;
    for coste in [lectura, COSTE_POSITION_RISK, COSTE_ORDEN] {
        for uso in 0..=l + 1 {
            let admite: Vec<bool> = orden
                .iter()
                .map(|pr| con_uso(uso).cabe(*pr, coste, T0).is_ok())
                .collect();
            for i in 1..orden.len() {
                assert!(
                    !admite[i] || admite[i - 1],
                    "uso {uso}, coste {coste:?}: {:?} admite y {:?} no",
                    orden[i],
                    orden[i - 1]
                );
            }
        }
        for uso in 0..=LIMITE_ORDENES_10S + 1 {
            let admite: Vec<bool> = orden
                .iter()
                .map(|pr| {
                    let mut p = PresupuestoPuerta::nuevo(limites);
                    p.observar(
                        &BinanceRateLimits {
                            weight_1m: None,
                            orders_10s: Some(uso),
                            orders_1m: None,
                        },
                        T0,
                    );
                    p.cabe(*pr, coste, T0).is_ok()
                })
                .collect();
            for i in 1..orden.len() {
                assert!(!admite[i] || admite[i - 1], "ordenes_10s {uso}, coste {coste:?}");
            }
        }
    }
}

/// P1: el 0,8 de la puerta es el literal del tramo `Lectura` del ejecutor
/// (executor.rs, `const HEADROOM`), no un número nuevo.
#[test]
fn gateway_fraccion_lectura_es_la_del_ejecutor() {
    let ejecutor = include_str!("../src/executor.rs").replace("\r\n", "\n");
    let ancla = "const HEADROOM: f64 = ";
    assert_eq!(ejecutor.matches(ancla).count(), 1, "un solo HEADROOM en el ejecutor");
    let resto = ejecutor.split(ancla).nth(1).unwrap_or("");
    let literal = resto.split(';').next().unwrap_or("").trim();
    let valor: f64 = literal.parse().unwrap_or(f64::NAN);
    assert_eq!(valor, FRACCION_FRENO_LECTURA, "literal del ejecutor: {literal}");
}
