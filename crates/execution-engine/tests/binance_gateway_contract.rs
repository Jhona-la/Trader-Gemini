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
    coste_de_firma, Alcance, CallPriority, Desenlace, FilaPeso, LimitesExchange, Metodo, OrigenLimite, PesoFirma,
    PresupuestoPuerta, RechazoPuerta, CONTADOR_ORDENES_10S, TABLA_PESOS,
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
