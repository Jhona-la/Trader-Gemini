//! PUERTA-REST: contrato de la puerta REST única (`binance_gateway`).
//!
//! Sin red a Binance: las pruebas de contabilidad usan un reloj inyectado y
//! las de transporte (P4) un servidor local en 127.0.0.1.

use execution_engine::binance_api::{
    CosteBinance, COSTE_ALGO_ABIERTAS_SIMBOLO, COSTE_CANCELACION, COSTE_CONSULTA_ORDEN, COSTE_ORDEN,
    COSTE_PIERNA_ALGO, COSTE_POSITION_RISK, LIMITE_ORDENES_10S, LIMITE_ORDENES_1M, LIMITE_PESO_1M,
    VENTANA_ORDENES_10S_MS, VENTANA_ORDENES_1M_MS, VENTANA_PESO_MS,
};
use execution_engine::binance_gateway::{
    coste_de_firma, Alcance, BinanceGateway, CallPriority, CintaJsonl, CintaMemoria, Desenlace, FilaPeso, LimitesExchange, Metodo, OrigenLimite, PesoFirma,
    Pedido, PresupuestoPuerta, RechazoPuerta, ResultadoLlamada, CONTADOR_ORDENES_10S, TABLA_PESOS,
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

    // Respuestas sin cabecera: su coste queda sin reconciliar SOBRE la última
    // cabecera conocida (C1, dictamen R1). Antes de C1 este caso esperaba
    // 3·c: la cabecera previa de 1 se perdía, que es el hueco de R1.
    let mut q = PresupuestoPuerta::nuevo(LimitesExchange::respaldo_lg12());
    q.observar(&cabecera_peso(1), T0);
    for _ in 0..3 {
        if let Ok(r) = q.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0) {
            q.cerrar(r, Desenlace::Respuesta(BinanceRateLimits::default()), T0);
        }
    }
    assert_eq!(q.en_vuelo(), [0, 0, 0]);
    assert_eq!(q.uso_efectivo(T0)[CONTADOR_PESO_1M], 1 + c * 3);

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
    // ventana. (C1: las cabeceras vienen en la respuesta de la orden, que así
    // queda reconciliada; una cabecera suelta de `observar` no cubriría su
    // coste.)
    let mut r = PresupuestoPuerta::nuevo(LimitesExchange::respaldo_lg12());
    if let Ok(res) = r.admitir(CallPriority::Critica, COSTE_ORDEN, T0) {
        assert_eq!(r.uso_efectivo(T0), [0, COSTE_ORDEN.ordenes, COSTE_ORDEN.ordenes]);
        r.cerrar(
            res,
            Desenlace::Respuesta(BinanceRateLimits {
                weight_1m: None,
                orders_10s: Some(7),
                orders_1m: Some(9),
            }),
            T0,
        );
    }
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

/// C1 (dictamen R1): con una cabecera previa vigente a `c` del límite, una
/// respuesta sin cabeceras no devuelve el uso a la cabecera: la segunda
/// admisión se rechaza. Valores del contraejemplo del dictamen: L = 2400,
/// c = 5 (positionRisk), H = L - c.
#[test]
fn c1_sin_cabecera_tras_cabecera_cerca_del_limite() {
    let c = COSTE_POSITION_RISK.peso;
    let l = LIMITE_PESO_1M;
    let mut p = PresupuestoPuerta::nuevo(LimitesExchange::respaldo_lg12());
    // Una primera petición trae la cabecera H = L - c (cuenta la suya).
    let r0 = p.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0);
    assert!(r0.is_ok());
    if let Ok(r0) = r0 {
        p.cerrar(r0, Desenlace::Respuesta(cabecera_peso(l - c)), T0);
    }
    assert_eq!(p.uso_efectivo(T0)[CONTADOR_PESO_1M], l - c);
    // Entra una crítica: H + c = L.
    let r1 = p.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0 + 1);
    assert!(r1.is_ok(), "H + c = L cabe");
    // Responde SIN cabeceras: el exchange la contó, nadie lo dice.
    if let Ok(r1) = r1 {
        p.cerrar(r1, Desenlace::Respuesta(BinanceRateLimits::default()), T0 + 2);
    }
    assert_eq!(p.en_vuelo(), [0, 0, 0]);
    assert_eq!(p.uso_efectivo(T0 + 2)[CONTADOR_PESO_1M], l, "el coste no desaparece");
    assert!(
        matches!(
            p.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0 + 3),
            Err(RechazoPuerta::Presupuesto(_))
        ),
        "la segunda admisión pasaría a H + 2c > L"
    );
    // Vence su ventana: sale.
    assert_eq!(p.uso_efectivo(T0 + 2 + VENTANA_PESO_MS)[CONTADOR_PESO_1M], 0);

    // Sin contarlo dos veces: una cabecera de una petición admitida DESPUÉS
    // de esa respuesta lo cubre.
    let mut q = PresupuestoPuerta::nuevo(LimitesExchange::respaldo_lg12());
    if let Ok(a) = q.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0) {
        q.cerrar(a, Desenlace::Respuesta(cabecera_peso(100)), T0);
    }
    if let Ok(b) = q.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0) {
        q.cerrar(b, Desenlace::Respuesta(BinanceRateLimits::default()), T0);
    }
    assert_eq!(q.uso_efectivo(T0)[CONTADOR_PESO_1M], 100 + c);
    if let Ok(d) = q.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0) {
        // El servidor contó b y d: 100 + 2c.
        q.cerrar(d, Desenlace::Respuesta(cabecera_peso(100 + 2 * c)), T0);
    }
    assert_eq!(
        q.uso_efectivo(T0)[CONTADOR_PESO_1M],
        100 + 2 * c,
        "la cabecera posterior lo cubre: sin contarlo dos veces"
    );

    // Una cabecera de una petición admitida ANTES de esa respuesta no lo
    // cubre (pudo no contarla): se queda, lo conservador.
    let mut s = PresupuestoPuerta::nuevo(LimitesExchange::respaldo_lg12());
    let antes = s.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0);
    if let Ok(b) = s.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0) {
        s.cerrar(b, Desenlace::Respuesta(BinanceRateLimits::default()), T0);
    }
    if let Ok(a) = antes {
        s.cerrar(a, Desenlace::Respuesta(cabecera_peso(50)), T0);
    }
    assert_eq!(s.uso_efectivo(T0)[CONTADOR_PESO_1M], 50 + c);

    // Una cabecera suelta (`observar`) no cubre nada.
    let mut z = PresupuestoPuerta::nuevo(LimitesExchange::respaldo_lg12());
    if let Ok(b) = z.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0) {
        z.cerrar(b, Desenlace::Respuesta(BinanceRateLimits::default()), T0);
    }
    z.observar(&cabecera_peso(50), T0 + 1);
    assert_eq!(z.uso_efectivo(T0 + 1)[CONTADOR_PESO_1M], 50 + c);

    // Un 429/418 (su cabecera no entra) tampoco hace desaparecer el coste.
    let mut x = PresupuestoPuerta::nuevo(LimitesExchange::respaldo_lg12());
    if let Ok(a) = x.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0) {
        x.cerrar(a, Desenlace::Respuesta(cabecera_peso(l - c)), T0);
    }
    if let Ok(b) = x.admitir(CallPriority::Critica, COSTE_POSITION_RISK, T0) {
        x.cerrar(b, Desenlace::LimiteExchange, T0);
    }
    assert_eq!(x.uso_efectivo(T0)[CONTADOR_PESO_1M], l);
}

/// C1: cabeceras parciales. Cada contador se reconcilia sólo con SU
/// cabecera: si la respuesta trae el peso pero no los contadores de
/// órdenes, el coste de órdenes queda sin reconciliar sobre su cabecera
/// previa y la siguiente orden que pasaría del límite se rechaza.
#[test]
fn c1_cabeceras_parciales() {
    let l10 = LIMITE_ORDENES_10S;
    // Una petición que gasta en los tres contadores.
    let mixto = CosteBinance { peso: 1, ordenes: 1 };
    let mut p = PresupuestoPuerta::nuevo(LimitesExchange::respaldo_lg12());
    if let Ok(a) = p.admitir(CallPriority::Critica, COSTE_ORDEN, T0) {
        p.cerrar(
            a,
            Desenlace::Respuesta(BinanceRateLimits {
                weight_1m: Some(10),
                orders_10s: Some(l10 - 1),
                orders_1m: Some(l10 - 1),
            }),
            T0,
        );
    }
    assert_eq!(p.uso_efectivo(T0), [10, l10 - 1, l10 - 1]);
    let b = p.admitir(CallPriority::Critica, mixto, T0 + 1);
    assert!(b.is_ok(), "cabe justo en L del contador de 10 s");
    if let Ok(b) = b {
        // Respuesta con el peso, sin los contadores de órdenes.
        p.cerrar(b, Desenlace::Respuesta(cabecera_peso(11)), T0 + 2);
    }
    let uso = p.uso_efectivo(T0 + 2);
    assert_eq!(uso[CONTADOR_PESO_1M], 11, "el peso se reconcilia con su cabecera");
    assert_eq!(uso[CONTADOR_ORDENES_10S], l10, "las órdenes no desaparecen");
    assert_eq!(uso[CONTADOR_ORDENES_1M], l10);
    assert!(
        matches!(
            p.admitir(CallPriority::Critica, COSTE_ORDEN, T0 + 3),
            Err(RechazoPuerta::Presupuesto(_))
        ),
        "la siguiente orden pasaría de L en el contador de 10 s"
    );
    // Vence la ventana de 10 s: el contador de 10 s se libera, el de 1 min no.
    let tras = p.uso_efectivo(T0 + 2 + VENTANA_ORDENES_10S_MS);
    assert_eq!(tras[CONTADOR_ORDENES_10S], 0);
    assert_eq!(tras[CONTADOR_ORDENES_1M], l10);
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

// ---------------------------------------------------------------- P2 ----

/// El mapa, tal como está en el repo.
const MAPA: &str = include_str!("../../../docs/BINANCE_API_MAP.md");

/// Una fila esperada, leída del mapa: (método, ruta, alcance, peso).
type Esperada = (Metodo, String, Alcance, PesoFirma);

/// El primer entero de `s`.
fn primer_entero(s: &str) -> Option<usize> {
    let ini = s.find(|c: char| c.is_ascii_digit())?;
    let resto = &s[ini..];
    let fin = resto.find(|c: char| !c.is_ascii_digit()).unwrap_or(resto.len());
    resto[..fin].parse().ok()
}

/// El peso de una celda: «0 ✔ (y 1 en cada contador de órdenes…)»,
/// «20 ✔ (antes figuraba 1)», «1-10», «5».
fn peso_de_celda(celda: &str) -> Option<PesoFirma> {
    if !celda.contains('✔') {
        return Some(PesoFirma::NoVerificado);
    }
    let peso = primer_entero(celda)?;
    let ordenes = if celda.contains("contador de órdenes") {
        let tras = celda.split("(y ").nth(1)?;
        primer_entero(tras)?
    } else {
        0
    };
    Some(PesoFirma::Verificado(CosteBinance { peso, ordenes }))
}

/// Lee las filas de §1 «Endpoints EN USO» del mapa.
fn filas_del_mapa() -> Vec<Esperada> {
    let mapa = MAPA.replace("\r\n", "\n");
    let ini = mapa.find("## 1. Endpoints EN USO").unwrap_or(mapa.len());
    let seccion = &mapa[ini..];
    let fin = seccion[3..].find("\n#").map(|k| k + 3).unwrap_or(seccion.len());
    let mut filas = Vec::new();
    for linea in seccion[..fin].lines() {
        if !linea.starts_with("| `/") {
            continue;
        }
        let celdas: Vec<&str> = linea.split('|').map(str::trim).collect();
        // celdas[0] es vacía (la línea empieza por '|').
        let ruta = celdas[1].trim_matches('`').to_string();
        let metodos = celdas[2];
        // Forma «GET 30 ✔ / POST 1 ✔ (...)»: método y peso en la misma celda.
        let pares: Vec<&str> = metodos.split(" / ").collect();
        let en_pares = pares.iter().all(|p| {
            let mut t = p.split_whitespace();
            t.next().and_then(Metodo::de_str).is_some()
                && t.next().is_some_and(|n| n.chars().all(|c| c.is_ascii_digit()))
        });
        if en_pares {
            for p in pares {
                let metodo = p.split_whitespace().next().and_then(Metodo::de_str);
                let peso = peso_de_celda(p);
                assert!(metodo.is_some() && peso.is_some(), "fila del mapa ilegible: {linea}");
                if let (Some(m), Some(w)) = (metodo, peso) {
                    filas.push((m, ruta.clone(), Alcance::Unico, w));
                }
            }
            continue;
        }
        let celda = celdas[3];
        for m in metodos.split('/') {
            let metodo = Metodo::de_str(m);
            assert!(metodo.is_some(), "método ilegible en el mapa: {linea}");
            let Some(metodo) = metodo else { continue };
            if celda.contains("con símbolo") && celda.contains("sin él") {
                let mut partes = celda.split('/');
                let con = partes.next().and_then(peso_de_celda_con_marca(celda));
                let sin = partes.next().and_then(peso_de_celda_con_marca(celda));
                assert!(con.is_some() && sin.is_some(), "fila del mapa ilegible: {linea}");
                if let (Some(c), Some(s)) = (con, sin) {
                    filas.push((metodo, ruta.clone(), Alcance::ConSimbolo, c));
                    filas.push((metodo, ruta.clone(), Alcance::SinSimbolo, s));
                }
            } else {
                let peso = peso_de_celda(celda);
                assert!(peso.is_some(), "peso ilegible en el mapa: {linea}");
                if let Some(w) = peso {
                    filas.push((metodo, ruta.clone(), Alcance::Unico, w));
                }
            }
        }
    }
    filas
}

/// En «1 con símbolo / 40 sin él ✔» la marca ✔ va al final y vale para las
/// dos partes.
fn peso_de_celda_con_marca(celda: &str) -> impl Fn(&str) -> Option<PesoFirma> + '_ {
    move |parte: &str| {
        if celda.contains('✔') {
            primer_entero(parte).map(|peso| PesoFirma::Verificado(CosteBinance { peso, ordenes: 0 }))
        } else {
            Some(PesoFirma::NoVerificado)
        }
    }
}

/// P2: la tabla de la puerta es el mapa §1, fila a fila, ni una más ni una
/// menos; lo no verificado queda N/D.
#[test]
fn gateway_tabla_pesos_paridad_con_mapa() {
    let mapa = filas_del_mapa();
    // Control del parser: el mapa tiene filas, verificadas y N/D.
    assert!(!mapa.is_empty(), "el parser no leyó filas del mapa");
    assert!(mapa.iter().any(|f| f.3 == PesoFirma::NoVerificado));
    assert!(mapa.iter().any(|f| matches!(f.3, PesoFirma::Verificado(_))));

    for (metodo, ruta, alcance, peso) in &mapa {
        let en_tabla: Vec<&FilaPeso> = TABLA_PESOS
            .iter()
            .filter(|f| f.metodo == *metodo && f.ruta == ruta && f.alcance == *alcance)
            .collect();
        assert_eq!(en_tabla.len(), 1, "{metodo:?} {ruta} {alcance:?}: una fila en la tabla");
        assert_eq!(en_tabla[0].peso, *peso, "{metodo:?} {ruta} {alcance:?}: peso del mapa");
    }
    for f in TABLA_PESOS {
        assert!(
            mapa.iter()
                .any(|(m, r, a, _)| *m == f.metodo && r == f.ruta && *a == f.alcance),
            "{:?} {} {:?}: está en la tabla y no en el mapa §1",
            f.metodo,
            f.ruta,
            f.alcance
        );
    }
    assert_eq!(TABLA_PESOS.len(), mapa.len());
}

/// P2: lo que no está en el mapa, o está sin ✔, se rechaza; nunca se supone
/// peso 1.
#[test]
fn gateway_firma_desconocida_falla_cerrado() {
    for (metodo, ruta) in [
        (Metodo::Get, "/fapi/v1/income"),
        (Metodo::Get, "/fapi/v1/openInterest"),
        (Metodo::Get, "/fapi/v1/premiumIndex"),
        (Metodo::Get, "/fapi/v1/fundingRate"),
        (Metodo::Get, "/futures/data/openInterestHist"),
        (Metodo::Get, "/futures/data/topLongShortAccountRatio"),
        (Metodo::Put, "/fapi/v1/order"),
        (Metodo::Get, "/fapi/v1/order/"),
        (Metodo::Get, "/fapi/v9/positionRisk"),
    ] {
        for con_simbolo in [false, true] {
            assert!(
                matches!(
                    coste_de_firma(metodo, ruta, con_simbolo),
                    Err(RechazoPuerta::FirmaDesconocida(_))
                ),
                "{metodo:?} {ruta}"
            );
        }
    }
    let mut nd = 0;
    for f in TABLA_PESOS {
        let con_simbolo = f.alcance == Alcance::ConSimbolo;
        match f.peso {
            PesoFirma::NoVerificado => {
                nd += 1;
                assert!(
                    matches!(
                        coste_de_firma(f.metodo, f.ruta, con_simbolo),
                        Err(RechazoPuerta::PesoNoVerificado(_))
                    ),
                    "{:?} {}: N/D falla cerrado",
                    f.metodo,
                    f.ruta
                );
            }
            PesoFirma::Verificado(c) => {
                assert_eq!(coste_de_firma(f.metodo, f.ruta, con_simbolo), Ok(c));
            }
        }
    }
    assert!(nd > 0, "control: hay filas N/D");
    // Con y sin símbolo: el peso cambia con el alcance.
    assert_ne!(
        coste_de_firma(Metodo::Get, "/fapi/v1/openOrders", true),
        coste_de_firma(Metodo::Get, "/fapi/v1/openOrders", false)
    );
}

/// P2 y regla de ruina: las firmas de cierre y de protección tienen peso
/// verificado (nunca fallan cerrado) y es el coste del ejecutor.
#[test]
fn gateway_firmas_criticas_verificadas() {
    for (metodo, ruta, con_simbolo, coste) in [
        (Metodo::Post, "/fapi/v1/order", true, COSTE_ORDEN),
        (Metodo::Get, "/fapi/v1/order", true, COSTE_CONSULTA_ORDEN),
        (Metodo::Delete, "/fapi/v1/order", true, COSTE_CANCELACION),
        (Metodo::Post, "/fapi/v1/algoOrder", true, COSTE_PIERNA_ALGO),
        (Metodo::Delete, "/fapi/v1/algoOrder", true, COSTE_CANCELACION),
        (Metodo::Delete, "/fapi/v1/algoOpenOrders", true, COSTE_CANCELACION),
        (Metodo::Delete, "/fapi/v1/allOpenOrders", true, COSTE_CANCELACION),
        (Metodo::Get, "/fapi/v2/positionRisk", false, COSTE_POSITION_RISK),
        (Metodo::Get, "/fapi/v1/openAlgoOrders", true, COSTE_ALGO_ABIERTAS_SIMBOLO),
    ] {
        assert_eq!(coste_de_firma(metodo, ruta, con_simbolo), Ok(coste), "{metodo:?} {ruta}");
    }
}

// ---------------------------------------------------------------- P3 ----

/// Cuerpo de exchangeInfo de prueba con límites distintos de LG-12 (para que
/// se vea de dónde sale cada uno), en otro orden, con un RAW_REQUESTS que no
/// es de ningún contador y campos de más.
fn exchange_info_de_prueba(peso: usize, o10: usize, o1m: usize) -> String {
    format!(
        concat!(
            "{{\"timezone\":\"UTC\",\"serverTime\":1,\"rateLimits\":[",
            "{{\"rateLimitType\":\"ORDERS\",\"interval\":\"MINUTE\",\"intervalNum\":1,\"limit\":{o1m}}},",
            "{{\"rateLimitType\":\"RAW_REQUESTS\",\"interval\":\"MINUTE\",\"intervalNum\":5,\"limit\":61000}},",
            "{{\"rateLimitType\":\"REQUEST_WEIGHT\",\"interval\":\"MINUTE\",\"intervalNum\":1,\"limit\":{peso}}},",
            "{{\"rateLimitType\":\"ORDERS\",\"interval\":\"SECOND\",\"intervalNum\":10,\"limit\":{o10}}}",
            "],\"symbols\":[]}}"
        ),
        peso = peso,
        o10 = o10,
        o1m = o1m
    )
}

/// P3: los tres contadores salen de `exchangeInfo.rateLimits`, cada uno con
/// su ventana, y el presupuesto los usa.
#[test]
fn gateway_limites_de_exchangeinfo() {
    let (peso, o10, o1m) = (LIMITE_PESO_1M / 2, LIMITE_ORDENES_10S / 2, LIMITE_ORDENES_1M / 2);
    let lectura = LimitesExchange::desde_exchange_info(&exchange_info_de_prueba(peso, o10, o1m));
    assert!(lectura.is_ok(), "{lectura:?}");
    let Ok(lectura) = lectura else { return };
    let l = lectura.limites;
    assert_eq!(
        (l.peso_1m.limite, l.ordenes_10s.limite, l.ordenes_1m.limite),
        (peso, o10, o1m)
    );
    assert_eq!(
        (l.peso_1m.ventana_ms, l.ordenes_10s.ventana_ms, l.ordenes_1m.ventana_ms),
        (VENTANA_PESO_MS, VENTANA_ORDENES_10S_MS, VENTANA_ORDENES_1M_MS)
    );
    for v in l.como_arreglo() {
        assert_eq!(v.origen, OrigenLimite::ExchangeInfo);
    }
    assert_eq!(lectura.ignorados.len(), 1, "{:?}", lectura.ignorados);
    assert!(lectura.ignorados[0].contains("RAW_REQUESTS"));

    // El límite del exchange también puede ser MAYOR que LG-12 (VIP).
    let mayor = LimitesExchange::desde_exchange_info(&exchange_info_de_prueba(
        LIMITE_PESO_1M * 2,
        LIMITE_ORDENES_10S,
        LIMITE_ORDENES_1M,
    ));
    assert!(matches!(mayor, Ok(ref m) if m.limites.peso_1m.limite == LIMITE_PESO_1M * 2));

    // El presupuesto frena con el límite leído, no con LG-12.
    let mut p = PresupuestoPuerta::nuevo(l);
    p.observar(&cabecera_peso(peso), T0);
    assert!(p.cabe(CallPriority::Critica, COSTE_POSITION_RISK, T0).is_err());
    let mut q = PresupuestoPuerta::nuevo(LimitesExchange::respaldo_lg12());
    q.observar(&cabecera_peso(peso), T0);
    assert!(q.cabe(CallPriority::Critica, COSTE_POSITION_RISK, T0).is_ok());

    // Dos entradas para el mismo contador: queda la menor.
    let doble = exchange_info_de_prueba(peso, o10, o1m).replace(
        "\"rateLimits\":[",
        &format!(
            "\"rateLimits\":[{{\"rateLimitType\":\"REQUEST_WEIGHT\",\"interval\":\"MINUTE\",\"intervalNum\":1,\"limit\":{}}},",
            peso - 1
        ),
    );
    assert!(matches!(
        LimitesExchange::desde_exchange_info(&doble),
        Ok(ref d) if d.limites.peso_1m.limite == peso - 1
    ));
}

/// P3: sin exchangeInfo, ilegible o con un contador ausente o inválido, ese
/// contador cae a su constante LG-12 de binance_api.rs.
#[test]
fn gateway_respaldo_lg12_sin_exchangeinfo() {
    let lg12 = LimitesExchange::respaldo_lg12();
    assert_eq!(
        (lg12.peso_1m.limite, lg12.ordenes_10s.limite, lg12.ordenes_1m.limite),
        (LIMITE_PESO_1M, LIMITE_ORDENES_10S, LIMITE_ORDENES_1M)
    );
    assert_eq!(
        (lg12.peso_1m.ventana_ms, lg12.ordenes_10s.ventana_ms, lg12.ordenes_1m.ventana_ms),
        (VENTANA_PESO_MS, VENTANA_ORDENES_10S_MS, VENTANA_ORDENES_1M_MS)
    );
    for v in lg12.como_arreglo() {
        assert_eq!(v.origen, OrigenLimite::RespaldoLg12);
    }

    for cuerpo in [None, Some("no es json"), Some("{\"symbols\":[]}"), Some("{\"rateLimits\":7}")] {
        let r = LimitesExchange::desde_exchange_info_o_respaldo(cuerpo);
        assert_eq!(r.limites, lg12, "{cuerpo:?}");
        assert_eq!(r.ignorados.len(), 1, "{cuerpo:?}: con su motivo");
    }

    // Sin la entrada de 10 s, sólo ese contador cae a LG-12.
    let (peso, o10, o1m) = (LIMITE_PESO_1M / 2, LIMITE_ORDENES_10S / 2, LIMITE_ORDENES_1M / 2);
    let sin_10s = exchange_info_de_prueba(peso, o10, o1m).replace(
        &format!(",{{\"rateLimitType\":\"ORDERS\",\"interval\":\"SECOND\",\"intervalNum\":10,\"limit\":{o10}}}"),
        "",
    );
    let r = LimitesExchange::desde_exchange_info_o_respaldo(Some(&sin_10s));
    assert_eq!(r.limites.ordenes_10s, lg12.ordenes_10s);
    assert_eq!(r.limites.peso_1m.limite, peso);
    assert_eq!(r.limites.ordenes_1m.limite, o1m);

    // Límite 0, ventana que no es de ningún contador o entrada mal formada:
    // no se usan.
    let raro = exchange_info_de_prueba(0, o10, o1m)
        .replace("\"intervalNum\":10,", "\"intervalNum\":7,")
        .replace("\"limit\":61000", "\"limit\":\"mucho\"");
    let r = LimitesExchange::desde_exchange_info_o_respaldo(Some(&raro));
    assert_eq!(r.limites.peso_1m, lg12.peso_1m);
    assert_eq!(r.limites.ordenes_10s, lg12.ordenes_10s);
    assert_eq!(r.limites.ordenes_1m.limite, o1m);
    assert_eq!(r.ignorados.len(), 3, "{:?}", r.ignorados);
}

// ---------------------------------------------------------------- P4 ----

/// Una respuesta del servidor local: estado y cabeceras extra.
#[derive(Clone)]
struct Guion {
    estado: u16,
    cabeceras: Vec<(&'static str, String)>,
    cuerpo: &'static str,
}

fn guion(estado: u16, cabeceras: &[(&'static str, &str)]) -> Guion {
    Guion {
        estado,
        cabeceras: cabeceras.iter().map(|(n, v)| (*n, v.to_string())).collect(),
        cuerpo: "{}",
    }
}

/// Servidor HTTP/1.1 mínimo en 127.0.0.1 (nunca Binance): responde, en
/// orden, cada `Guion` y apunta la línea de petición que recibió. Pasado el
/// guion responde 500.
struct Servidor {
    base: String,
    recibidas: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl Servidor {
    async fn levantar(guiones: Vec<Guion>) -> Servidor {
        let oyente = tokio::net::TcpListener::bind("127.0.0.1:0").await;
        assert!(oyente.is_ok(), "no se pudo abrir 127.0.0.1");
        let Ok(oyente) = oyente else { unreachable!() };
        let direccion = oyente.local_addr().map(|a| a.to_string()).unwrap_or_default();
        let recibidas = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let apuntes = recibidas.clone();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut cola = std::collections::VecDeque::from(guiones);
            loop {
                let Ok((mut socket, _)) = oyente.accept().await else { return };
                let mut leido = Vec::new();
                let mut buf = [0u8; 4096];
                while !leido.windows(4).any(|w| w == b"\r\n\r\n") {
                    match socket.read(&mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => leido.extend_from_slice(&buf[..n]),
                    }
                }
                let cabeza = String::from_utf8_lossy(&leido).to_string();
                let linea = cabeza.lines().next().unwrap_or("").to_string();
                apuntes.lock().unwrap_or_else(|e| e.into_inner()).push(linea);
                let g = cola.pop_front().unwrap_or_else(|| guion(500, &[]));
                let mut r = format!("HTTP/1.1 {} X\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n", g.estado, g.cuerpo.len());
                for (n, v) in &g.cabeceras {
                    r.push_str(&format!("{n}: {v}\r\n"));
                }
                r.push_str("\r\n");
                r.push_str(g.cuerpo);
                let _ = socket.write_all(r.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });
        Servidor {
            base: format!("http://{direccion}"),
            recibidas,
        }
    }

    fn recibidas(&self) -> Vec<String> {
        self.recibidas.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// Cliente HTTP de prueba: sin proxy del sistema, para que nada salga de 127.0.0.1.
fn cliente() -> reqwest::Client {
    reqwest::Client::builder().no_proxy().build().unwrap_or_default()
}

fn puerta(base: &str, cinta: std::sync::Arc<CintaMemoria>) -> BinanceGateway {
    BinanceGateway::nueva(base, LimitesExchange::respaldo_lg12(), cliente(), cinta)
}

fn position_risk(prioridad: CallPriority) -> Pedido {
    Pedido::nuevo(Metodo::Get, "/fapi/v2/positionRisk", prioridad)
        .consulta("timestamp=1&signature=SECRETO")
        .cabecera("X-MBX-APIKEY", "CLAVE")
}

/// Un pedido de cada prioridad con firma verificada.
fn uno_de_cada() -> Vec<Pedido> {
    CallPriority::TODAS.iter().map(|p| position_risk(*p)).collect()
}

/// P4: un 200 reconcilia la cabecera: `max(local, cabecera)` y la reserva
/// sale del libro en vuelo.
#[tokio::test]
async fn gateway_200_reconcilia_cabeceras() {
    let cab = (COSTE_POSITION_RISK.peso * 100).to_string();
    let menor = (COSTE_POSITION_RISK.peso * 50).to_string();
    let srv = Servidor::levantar(vec![
        guion(200, &[("X-MBX-USED-WEIGHT-1M", &cab), ("X-MBX-ORDER-COUNT-10S", "3"), ("X-MBX-ORDER-COUNT-1M", "4")]),
        guion(200, &[("X-MBX-USED-WEIGHT-1M", &menor)]),
        guion(200, &[]),
    ])
    .await;
    let cinta = std::sync::Arc::new(CintaMemoria::nueva());
    let g = puerta(&srv.base, cinta.clone());

    let r = g.enviar(position_risk(CallPriority::Lectura)).await;
    assert!(matches!(r, Ok(ref r) if r.estado == 200), "{r:?}");
    assert_eq!(g.en_vuelo(), [0, 0, 0]);
    let uso = g.uso_efectivo();
    assert_eq!(uso[CONTADOR_PESO_1M], COSTE_POSITION_RISK.peso * 100, "la cabecera manda");
    assert_eq!((uso[CONTADOR_ORDENES_10S], uso[CONTADOR_ORDENES_1M]), (3, 4));

    // Una cabecera menor que llega después no rebaja el uso de la ventana.
    assert!(g.enviar(position_risk(CallPriority::Lectura)).await.is_ok());
    assert_eq!(g.uso_efectivo()[CONTADOR_PESO_1M], COSTE_POSITION_RISK.peso * 100);

    // Lo que salió: método, ruta y consulta, una vez cada pedido.
    let rec = srv.recibidas();
    assert_eq!(rec.len(), 2);
    assert!(rec[0].starts_with("GET /fapi/v2/positionRisk?timestamp=1&signature=SECRETO HTTP/1.1"), "{rec:?}");

    // Sin cabecera, lo local manda: tres positionRisk en la ventana.
    let srv2 = Servidor::levantar(vec![guion(200, &[]), guion(200, &[]), guion(200, &[])]).await;
    let g2 = puerta(&srv2.base, std::sync::Arc::new(CintaMemoria::nueva()));
    for _ in 0..3 {
        assert!(g2.enviar(position_risk(CallPriority::Critica)).await.is_ok());
    }
    assert_eq!(g2.uso_efectivo()[CONTADOR_PESO_1M], COSTE_POSITION_RISK.peso * 3);

    // El presupuesto frena antes de enviar: un límite de peso del tamaño de
    // un positionRisk deja salir uno y frena el segundo sin tocar la red.
    let srv3 = Servidor::levantar(vec![guion(200, &[])]).await;
    let mut chico = LimitesExchange::respaldo_lg12();
    chico.peso_1m.limite = COSTE_POSITION_RISK.peso;
    let g3 = BinanceGateway::nueva(&srv3.base, chico, cliente(), std::sync::Arc::new(CintaMemoria::nueva()));
    assert!(g3.enviar(position_risk(CallPriority::Critica)).await.is_ok());
    assert!(matches!(
        g3.enviar(position_risk(CallPriority::Critica)).await,
        Err(RechazoPuerta::Presupuesto(_))
    ));
    assert_eq!(srv3.recibidas().len(), 1, "el rechazado no salió");

    // Firma fuera del mapa o N/D: falla cerrado sin tocar la red.
    let srv4 = Servidor::levantar(vec![]).await;
    let g4 = puerta(&srv4.base, std::sync::Arc::new(CintaMemoria::nueva()));
    assert!(matches!(
        g4.enviar(Pedido::nuevo(Metodo::Get, "/fapi/v1/income", CallPriority::Lectura)).await,
        Err(RechazoPuerta::FirmaDesconocida(_))
    ));
    assert!(matches!(
        g4.enviar(Pedido::nuevo(Metodo::Get, "/fapi/v1/klines", CallPriority::Batch)).await,
        Err(RechazoPuerta::PesoNoVerificado(_))
    ));
    assert!(srv4.recibidas().is_empty());
}

/// C1 por el transporte: una respuesta con cabecera H = L - c, otra admitida
/// que responde sin cabeceras, y la tercera ya no sale (antes de C1 salía y
/// el exchange habría contado H + 2c > L).
#[tokio::test]
async fn c1_puerta_sin_cabecera_no_reabre_el_hueco() {
    let c = COSTE_POSITION_RISK.peso;
    let h = (LIMITE_PESO_1M - c).to_string();
    let srv = Servidor::levantar(vec![
        guion(200, &[("X-MBX-USED-WEIGHT-1M", &h)]),
        guion(200, &[]),
        guion(200, &[]),
    ])
    .await;
    let cinta = std::sync::Arc::new(CintaMemoria::nueva());
    let g = puerta(&srv.base, cinta.clone());
    assert!(g.enviar(position_risk(CallPriority::Critica)).await.is_ok());
    assert!(g.enviar(position_risk(CallPriority::Critica)).await.is_ok(), "H + c = L cabe");
    assert_eq!(g.uso_efectivo()[CONTADOR_PESO_1M], LIMITE_PESO_1M);
    let r = g.enviar(position_risk(CallPriority::Critica)).await;
    assert!(matches!(r, Err(RechazoPuerta::Presupuesto(_))), "{r:?}");
    assert_eq!(srv.recibidas().len(), 2, "la tercera no salió");
    assert_eq!(cinta.filas().len(), 3);
}

/// C2 (dictamen R2): el coste sale de los parámetros que salen. openOrders y
/// openAlgoOrders sin `symbol` cuestan 40 (mapa :32 y :34); declararlos «con
/// símbolo» sin llevarlo no puede reservar 1: se rechaza antes de enviar.
/// Los bytes de la consulta firmada salen sin cambios.
#[tokio::test]
async fn c2_sin_symbol_no_reserva_el_coste_con_symbol() {
    let es_alcance = |r: &Result<_, RechazoPuerta>| matches!(r, Err(e) if e.to_string().starts_with("PUERTA_ALCANCE"));
    for ruta in ["/fapi/v1/openOrders", "/fapi/v1/openAlgoOrders"] {
        let sin = coste_de_firma(Metodo::Get, ruta, false);
        let con = coste_de_firma(Metodo::Get, ruta, true);
        assert_eq!(sin.as_ref().map(|c| c.peso), Ok(40), "{ruta}");
        assert_eq!(con.as_ref().map(|c| c.peso), Ok(1), "{ruta}");

        let srv = Servidor::levantar(vec![guion(200, &[]), guion(200, &[])]).await;
        let cinta = std::sync::Arc::new(CintaMemoria::nueva());
        let g = puerta(&srv.base, cinta.clone());

        // Declarado con símbolo, consulta sin él: rechazo antes de enviar.
        let r = g
            .enviar(
                Pedido::nuevo(Metodo::Get, ruta, CallPriority::Lectura)
                    .consulta("timestamp=1&signature=SECRETO")
                    .con_simbolo(true),
            )
            .await;
        assert!(es_alcance(&r), "{ruta}: {r:?}");
        assert!(srv.recibidas().is_empty(), "{ruta}: no salió");
        assert_eq!(g.en_vuelo(), [0, 0, 0]);
        assert_eq!(g.uso_efectivo(), [0, 0, 0], "{ruta}: nada reservado");

        // Ambiguos: fallan cerrados sin salir.
        for consulta in [
            "symbol=BTCUSDT&symbol=ETHUSDT&timestamp=1",
            "symbol=&timestamp=1",
            "SYMBOL=BTCUSDT&timestamp=1",
            "%73ymbol=BTCUSDT&timestamp=1",
        ] {
            let r = g.enviar(Pedido::nuevo(Metodo::Get, ruta, CallPriority::Lectura).consulta(consulta)).await;
            assert!(es_alcance(&r), "{ruta} {consulta}: {r:?}");
        }
        assert!(srv.recibidas().is_empty());

        // Sin declarar y sin symbol: paga 40.
        let r = g
            .enviar(Pedido::nuevo(Metodo::Get, ruta, CallPriority::Lectura).consulta("timestamp=1&signature=SECRETO"))
            .await;
        assert!(r.is_ok(), "{ruta}: {r:?}");
        // Con symbol: paga 1, y la consulta sale con sus mismos bytes.
        let consulta = "symbol=BTCUSDT&timestamp=1&signature=SECRETO";
        let r = g
            .enviar(Pedido::nuevo(Metodo::Get, ruta, CallPriority::Lectura).consulta(consulta).con_simbolo(true))
            .await;
        assert!(r.is_ok(), "{ruta}: {r:?}");
        let rec = srv.recibidas();
        assert_eq!(rec.len(), 2);
        assert_eq!(rec[1], format!("GET {ruta}?{consulta} HTTP/1.1"));
        let filas = cinta.filas();
        let salidas: Vec<(bool, Option<usize>)> = filas
            .iter()
            .filter(|f| matches!(f.resultado, ResultadoLlamada::Respondida { .. }))
            .map(|f| (f.con_simbolo, f.coste_peso))
            .collect();
        assert_eq!(salidas, vec![(false, Some(40)), (true, Some(1))], "{ruta}");
        assert_eq!(filas.len(), 7, "{ruta}: una fila por pedido");

        // El coste correcto se reserva antes de enviar: con un límite de peso
        // de 39, sin symbol no sale; con symbol sí.
        let srv2 = Servidor::levantar(vec![guion(200, &[])]).await;
        let mut chico = LimitesExchange::respaldo_lg12();
        chico.peso_1m.limite = 39;
        let g2 = BinanceGateway::nueva(&srv2.base, chico, cliente(), std::sync::Arc::new(CintaMemoria::nueva()));
        let r = g2
            .enviar(Pedido::nuevo(Metodo::Get, ruta, CallPriority::Critica).consulta("timestamp=1").con_simbolo(true))
            .await;
        assert!(r.is_err(), "{ruta}: {r:?}");
        let r = g2.enviar(Pedido::nuevo(Metodo::Get, ruta, CallPriority::Critica).consulta("timestamp=1")).await;
        assert!(matches!(r, Err(RechazoPuerta::Presupuesto(_))), "{ruta}: {r:?}");
        assert!(srv2.recibidas().is_empty(), "{ruta}: 40 > 39 no sale");
        let r = g2.enviar(Pedido::nuevo(Metodo::Get, ruta, CallPriority::Critica).consulta("symbol=BTCUSDT&timestamp=1")).await;
        assert!(r.is_ok(), "{ruta}: {r:?}");
        assert_eq!(srv2.recibidas().len(), 1);
    }
}

/// Cabecera de peso de la primera respuesta del servidor demorado.
const H_PREVIA: usize = 100;
/// Cabecera de peso de la respuesta que se cancela durante el cuerpo.
const H_CUERPO: usize = 107;

/// C3: servidor local de DOS conexiones. La primera responde 200 con la
/// cabecera de peso `H_PREVIA`. La segunda apunta su línea de petición y se
/// queda colgada: sin responder (`durante_cuerpo = false`) o tras mandar las
/// cabeceras (peso `H_CUERPO`) y un byte de un cuerpo de 100 (`true`).
async fn levantar_demorado(durante_cuerpo: bool) -> Servidor {
    let oyente = tokio::net::TcpListener::bind("127.0.0.1:0").await;
    assert!(oyente.is_ok(), "no se pudo abrir 127.0.0.1");
    let Ok(oyente) = oyente else { unreachable!() };
    let direccion = oyente.local_addr().map(|a| a.to_string()).unwrap_or_default();
    let recibidas = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let apuntes = recibidas.clone();
    tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        for n in 0..2 {
            let Ok((mut socket, _)) = oyente.accept().await else { return };
            let mut leido = Vec::new();
            let mut buf = [0u8; 4096];
            while !leido.windows(4).any(|w| w == b"\r\n\r\n") {
                match socket.read(&mut buf).await {
                    Ok(0) | Err(_) => break,
                    Ok(k) => leido.extend_from_slice(&buf[..k]),
                }
            }
            let linea = String::from_utf8_lossy(&leido).lines().next().unwrap_or("").to_string();
            apuntes.lock().unwrap_or_else(|e| e.into_inner()).push(linea);
            if n == 0 {
                let r = format!("HTTP/1.1 200 X\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: 2\r\nX-MBX-USED-WEIGHT-1M: {H_PREVIA}\r\n\r\n{{}}");
                let _ = socket.write_all(r.as_bytes()).await;
                let _ = socket.shutdown().await;
                continue;
            }
            if durante_cuerpo {
                let cabeza = format!("HTTP/1.1 200 X\r\nContent-Type: application/json\r\nContent-Length: 100\r\nX-MBX-USED-WEIGHT-1M: {H_CUERPO}\r\n\r\n{{");
                let _ = socket.write_all(cabeza.as_bytes()).await;
                let _ = socket.flush().await;
            }
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            drop(socket);
        }
    });
    Servidor {
        base: format!("http://{direccion}"),
        recibidas,
    }
}

/// C3 (dictamen R3): cancelar el futuro de `enviar` (destruirlo) antes de la
/// respuesta o mientras se lee el cuerpo no deja reservas en vuelo para
/// siempre ni pierde la fila de la cinta; el coste que pudo salir se
/// conserva de forma prudente: antes de la respuesta queda SOBRE la cabecera
/// previa (perdido hasta que vence su ventana); durante el cuerpo lo cuenta
/// la cabecera que llegó.
#[tokio::test]
async fn c3_cancelar_no_deja_reservas_ni_pierde_la_fila() {
    let c = COSTE_POSITION_RISK.peso;
    for durante_cuerpo in [false, true] {
        let srv = levantar_demorado(durante_cuerpo).await;
        let cinta = std::sync::Arc::new(CintaMemoria::nueva());
        let g = puerta(&srv.base, cinta.clone());
        // Una primera respuesta deja la cabecera previa H_PREVIA.
        let r = g.enviar(position_risk(CallPriority::Critica)).await;
        assert!(r.is_ok(), "{r:?}");
        assert_eq!(g.uso_efectivo()[CONTADOR_PESO_1M], H_PREVIA);
        {
            let fut = g.enviar(position_risk(CallPriority::Critica));
            tokio::pin!(fut);
            let mut terminado: Option<String> = None;
            let mut listo = false;
            for _ in 0..500 {
                tokio::select! {
                    r = &mut fut => {
                        terminado = Some(format!("{r:?}"));
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_millis(10)) => {}
                }
                if terminado.is_some() {
                    break;
                }
                // Antes de la respuesta: el servidor la recibió y sigue en
                // vuelo. Durante el cuerpo: además, las cabeceras ya
                // cerraron la reserva.
                let en_vuelo = g.en_vuelo()[CONTADOR_PESO_1M];
                if srv.recibidas().len() == 2 && (en_vuelo == 0) == durante_cuerpo {
                    listo = true;
                    break;
                }
            }
            assert!(terminado.is_none(), "durante_cuerpo={durante_cuerpo}: no debía terminar: {terminado:?}");
            assert!(listo, "durante_cuerpo={durante_cuerpo}: no llegó al punto de cancelar");
            assert_eq!(cinta.filas().len(), 1, "todavía sin la fila del cancelado");
        } // Aquí se destruye el futuro: la cancelación.
        assert_eq!(g.en_vuelo(), [0, 0, 0], "durante_cuerpo={durante_cuerpo}: sin reservas permanentes");
        let filas = cinta.filas();
        assert_eq!(filas.len(), 2, "durante_cuerpo={durante_cuerpo}: una fila por pedido admitido");
        let json = serde_json::to_string(&filas[1]).unwrap_or_default();
        assert!(json.contains("Cancelada"), "{json}");
        assert!(!json.contains("SECRETO") && !json.contains("CLAVE"), "{json}");
        assert_eq!(filas[1].coste_peso, Some(c));
        let uso = g.uso_efectivo()[CONTADOR_PESO_1M];
        if durante_cuerpo {
            assert_eq!(filas[1].cabecera_peso_1m, Some(H_CUERPO), "la cabecera que llegó queda en la fila");
            assert_eq!(uso, H_CUERPO, "la cabecera que llegó lo cuenta");
        } else {
            assert_eq!(uso, H_PREVIA + c, "pudo salir: su coste se conserva sobre la cabecera previa");
        }
        assert_eq!(srv.recibidas().len(), 2);
    }
}
/// Espera hasta que el freno de la puerta se levante (Retry-After de 1 s en
/// el guion), con un tope para no colgar la prueba.
async fn esperar_rearme(g: &BinanceGateway) -> bool {
    for _ in 0..40 {
        if g.freno_activo().is_ok() {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    false
}

/// P4: un 429 arma el cooldown con su Retry-After; mientras dura no sale
/// nada (ninguna prioridad, F1.8) y su cabecera no se registra (CL-196);
/// al vencer se sale otra vez.
#[tokio::test]
async fn gateway_429_respeta_retry_after() {
    let alto = (LIMITE_PESO_1M - 1).to_string();
    let srv = Servidor::levantar(vec![
        guion(429, &[("Retry-After", "1"), ("X-MBX-USED-WEIGHT-1M", &alto)]),
        guion(200, &[]),
    ])
    .await;
    let cinta = std::sync::Arc::new(CintaMemoria::nueva());
    let g = puerta(&srv.base, cinta.clone());

    let r = g.enviar(position_risk(CallPriority::Lectura)).await;
    assert!(matches!(r, Err(RechazoPuerta::LimiteExchange(ref t)) if t.contains("HTTP_429") && t.contains("retry_after=1")), "{r:?}");
    assert!(g.uso_efectivo()[CONTADOR_PESO_1M] < LIMITE_PESO_1M - 1, "la cabecera del 429 no entra");
    assert_eq!(g.en_vuelo(), [0, 0, 0]);
    for p in uno_de_cada() {
        let pr = p.prioridad;
        assert!(matches!(g.enviar(p).await, Err(RechazoPuerta::Frenado(ref t)) if t.contains("RATE_LIMIT_COOLDOWN")), "{pr:?}");
    }
    assert_eq!(srv.recibidas().len(), 1, "durante el cooldown no sale nada");

    assert!(esperar_rearme(&g).await, "el cooldown vence");
    assert!(g.enviar(position_risk(CallPriority::Critica)).await.is_ok());
    assert_eq!(srv.recibidas().len(), 2);
}

/// P4: un 418 es parada TOTAL de la puerta (también Crítica: el exchange
/// banea la IP) hasta su Retry-After, y se rearma sola: nunca permanente.
#[tokio::test]
async fn gateway_418_parada_total_hasta_retry_after() {
    let srv = Servidor::levantar(vec![guion(418, &[("Retry-After", "1")]), guion(200, &[])]).await;
    let g = puerta(&srv.base, std::sync::Arc::new(CintaMemoria::nueva()));

    let r = g.enviar(position_risk(CallPriority::Critica)).await;
    assert!(matches!(r, Err(RechazoPuerta::LimiteExchange(ref t)) if t.contains("HTTP_418")), "{r:?}");
    assert!(matches!(g.freno_activo(), Err(ref t) if t.contains("RATE_BRAKE")));
    // C2: el alcance sale de la consulta; la declaración con symbol exige
    // que la consulta lo lleve (antes de C2 el pedido no traía consulta).
    let orden = Pedido::nuevo(Metodo::Post, "/fapi/v1/order", CallPriority::Critica)
        .consulta("symbol=BTCUSDT&side=BUY&timestamp=1&signature=SECRETO")
        .con_simbolo(true);
    assert!(matches!(g.enviar(orden.clone()).await, Err(RechazoPuerta::Frenado(ref t)) if t.contains("RATE_BRAKE")));
    for p in uno_de_cada() {
        assert!(matches!(g.enviar(p).await, Err(RechazoPuerta::Frenado(_))));
    }
    assert_eq!(srv.recibidas().len(), 1, "durante el baneo no sale nada");

    assert!(esperar_rearme(&g).await, "el freno del 418 se rearma solo");
    let r = g.enviar(orden).await;
    assert!(matches!(r, Ok(ref r) if r.estado == 200), "tras el baneo la orden sale: {r:?}");
    assert_eq!(srv.recibidas().len(), 2);
}

/// P4: una fila de cinta por pedido, haya salido o no, en orden, sin la
/// consulta firmada ni la clave; y la cinta JSONL escribe las mismas filas.
#[tokio::test]
async fn gateway_cinta_registra_cada_pedido() {
    let srv = Servidor::levantar(vec![
        guion(200, &[("X-MBX-USED-WEIGHT-1M", "9")]),
        guion(400, &[("X-MBX-USED-WEIGHT-1M", "10")]),
        guion(429, &[("Retry-After", "1")]),
    ])
    .await;
    let cinta = std::sync::Arc::new(CintaMemoria::nueva());
    let g = puerta(&srv.base, cinta.clone());

    let _ = g.enviar(position_risk(CallPriority::Lectura)).await; // 200
    let _ = g.enviar(position_risk(CallPriority::Alta)).await; // 400
    let _ = g.enviar(Pedido::nuevo(Metodo::Get, "/fapi/v1/income", CallPriority::Lectura).consulta("signature=SECRETO")).await; // firma
    let _ = g.enviar(position_risk(CallPriority::Batch)).await; // 429
    let _ = g.enviar(position_risk(CallPriority::Critica)).await; // freno

    let filas = cinta.filas();
    let resultados: Vec<ResultadoLlamada> = filas.iter().map(|f| f.resultado.clone()).collect();
    assert_eq!(
        resultados,
        vec![
            ResultadoLlamada::Respondida { estado: 200 },
            ResultadoLlamada::Respondida { estado: 400 },
            ResultadoLlamada::RechazadaFirma,
            ResultadoLlamada::LimiteExchange { estado: 429 },
            ResultadoLlamada::RechazadaFreno,
        ]
    );
    assert_eq!(srv.recibidas().len(), 3);
    assert_eq!(filas[0].cabecera_peso_1m, Some(9));
    assert_eq!(filas[1].cabecera_peso_1m, Some(10), "la cabecera de un 400 también cuenta (CL-196)");
    assert_eq!(filas[0].coste_peso, Some(COSTE_POSITION_RISK.peso));
    assert_eq!(filas[2].coste_peso, None);
    assert_eq!(filas[3].prioridad, CallPriority::Batch);
    for f in &filas {
        let json = serde_json::to_string(f).unwrap_or_default();
        assert!(!json.contains("SECRETO") && !json.contains("CLAVE"), "la cinta no lleva la firma: {json}");
    }

    // La cinta JSONL: una línea por pedido.
    let ruta = std::env::temp_dir().join(format!("tg_puerta_cinta_{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&ruta);
    let jsonl = CintaJsonl::abrir(&ruta);
    assert!(jsonl.is_ok());
    let Ok(jsonl) = jsonl else { return };
    let jsonl = std::sync::Arc::new(jsonl);
    let srv2 = Servidor::levantar(vec![guion(200, &[])]).await;
    let g2 = BinanceGateway::nueva(&srv2.base, LimitesExchange::respaldo_lg12(), cliente(), jsonl.clone());
    let _ = g2.enviar(position_risk(CallPriority::Lectura)).await;
    let _ = g2.enviar(Pedido::nuevo(Metodo::Get, "/fapi/v1/time", CallPriority::Lectura)).await;
    let texto = std::fs::read_to_string(&ruta).unwrap_or_default();
    let _ = std::fs::remove_file(&ruta);
    assert_eq!(jsonl.fallos(), 0);
    let lineas: Vec<&str> = texto.lines().collect();
    assert_eq!(lineas.len(), 2, "{texto}");
    assert!(lineas[0].contains("\"Respondida\"") && lineas[1].contains("RechazadaFirma"), "{texto}");
}
