//! C-F4 · D4 (ADR-0016; Línea C, CL-98, revisión del PR #61, F1): la orden
//! mínima es la del LOTE del símbolo.
//!
//! Teoría. El exchange acepta cantidades múltiplo del paso (`stepSize`), con
//! `qty ≥ minQty` y `qty·precio ≥ min_notional`, y el ejecutor redondea la
//! cantidad hacia abajo al paso (`executor::round_to_step_size`). Antes el
//! riesgo fijaba el nocional de la orden mínima en
//! `(min_notional + 0,1)·1,0005 ≈ 5,10 USD`, sin mirar el paso:
//! - BNB a 600 USD con paso 0,01: 0,0085 BNB, que el ejecutor deja en 0
//!   («Volumen 0»);
//! - paso 1 a 3 USD: 1,7 unidades, que quedan en 1 (3 USD, bajo el mínimo).
//! El IOC no salía, el rollback no sube `trade_count` y la moneda no salía
//! nunca de la fase de sonda. Ahora la orden mínima es el menor múltiplo del
//! paso que cumple los dos mínimos (redondeo hacia arriba), y la viabilidad
//! D-750 juzga su riesgo al stop: si supera el tope de ruina, se rechaza.
//!
//! Fixture. Tres símbolos con nocional mínimo de 5 USD:
//! - 0: BNB, paso 0,01 y cantidad mínima 0,01, a 600 USD;
//! - 1: paso 1 y cantidad mínima 1, a 3 USD;
//! - 2: paso 1 y cantidad mínima 1, a 600 USD (orden mínima de 600 USD).
//! ATR del 1 % del precio. «Ejecutable» se juzga con el redondeo del
//! ejecutor replicado (hacia abajo, con su tolerancia de múltiplo exacto).

#[path = "support/net_payoff_fixture.rs"]
mod net_payoff_fixture;

use quantum_arena::GlobalArena;
use risk_engine::capital_regime::{
    cantidad_de_la_orden, cantidad_minima_ejecutable, nocional_minimo_ejecutable,
};
use risk_engine::{RiskEngine, REJECT_NAMES};
use signal_engine::{SignalIntent, SignalType};
use std::sync::{atomic::Ordering::Relaxed, Arc, Once};

const NOCIONAL_MINIMO: f64 = 5.0;
/// (símbolo, paso = cantidad mínima, precio)
const SIMBOLOS: [(&str, f64, f64); 3] =
    [("BNBUSDT", 0.01, 600.0), ("LOTE1USDT", 1.0, 3.0), ("LOTE1CARO", 1.0, 600.0)];

fn arena(capital: f64) -> Arc<GlobalArena> {
    static REGISTRO: Once = Once::new();
    REGISTRO.call_once(|| {
        quantum_arena::symbol_registry::update_registry(
            SIMBOLOS
                .iter()
                .map(|(s, paso, _)| quantum_arena::symbol_registry::SymbolSpec {
                    symbol: (*s).into(),
                    step_size: *paso,
                    tick_size: 0.01,
                    min_qty: *paso,
                    min_notional: NOCIONAL_MINIMO,
                    max_leverage: 20,
                    maker_fee: 0.0002,
                    taker_fee: 0.0004,
                    is_shadow: false,
                })
                .collect(),
        )
    });
    let arena = GlobalArena::build_in_own_stack(capital);
    for (i, (_, _, precio)) in SIMBOLOS.iter().enumerate() {
        poner_precio(&arena, i, *precio);
        arena.coins[i].hurst_exponent.store(0.5, Relaxed);
    }
    arena.config.latency_penalty_ms.store(0.0, Relaxed);
    arena.config.live_taker_fee.store(0.0004, Relaxed);
    arena.config.base_slippage_floor.store(0.00001, Relaxed);
    arena.config.global_max_drawdown.store(0.5, Relaxed);
    arena.config.kelly_clamp_min.store(0.01, Relaxed);
    arena.config.kelly_clamp_max.store(0.25, Relaxed);
    arena.config.global_correlation_threshold.store(0.5, Relaxed);
    arena.riesgo_por_operacion.store(1e-4, Relaxed);
    arena
}

fn poner_precio(arena: &GlobalArena, coin: usize, precio: f64) {
    arena.coins[coin].current_price.store(precio, Relaxed);
    arena.coins[coin].current_atr.store(precio * 0.01, Relaxed);
}

/// Moneda con historial (fuera de la fase de sonda): 100 cierres, acierto
/// 0,75 y PF 2.
fn con_historial(arena: &GlobalArena, coin: usize) {
    let m = &arena.coins[coin].metrics;
    m.trade_count.store(100, Relaxed);
    m.win_rate.store(0.75, Relaxed);
    m.gross_wins.store(200.0, Relaxed);
    m.gross_losses.store(100.0, Relaxed);
    m.profit_factor.store(2.0, Relaxed);
    m.kelly_fraction.store(0.15, Relaxed);
}

fn larga() -> SignalIntent {
    SignalIntent {
        signal: SignalType::Long,
        confidence: 0.9,
        expected_duration_ms: 60_000,
        ..SignalIntent::default()
    }
}

/// Nocional de la orden admitida, o el nombre del motivo del rechazo.
fn evaluar(arena: &GlobalArena, capital: f64, coin: usize) -> Result<f64, &'static str> {
    let mut intent = larga();
    if arena.coins[coin].metrics.trade_count.load(Relaxed) >= risk_engine::sonda::CIERRES_DE_SONDA {
        intent.net_payoff = Some(net_payoff_fixture::positive(coin, intent.signal));
    }
    let orden = RiskEngine::new(capital).evaluate_quantum_order(coin, &intent, arena);
    if orden.signal == SignalType::Flat {
        Err(risk_engine::ultimo_rechazo().map_or("sin motivo", |i| REJECT_NAMES[i]))
    } else {
        Ok(orden.volume_usd * orden.leverage)
    }
}

/// Réplica del redondeo del ejecutor (`round_to_step_size`): hacia abajo al
/// paso, salvo que el valor ya sea múltiplo exacto con tolerancia relativa.
fn cantidad_enviada(qty: f64, paso: f64) -> f64 {
    let escalado = qty / paso;
    let r = escalado.round();
    let pasos = if (escalado - r).abs() <= 1e-9 * r.abs().max(1.0) { r } else { escalado.floor() };
    pasos * paso
}

/// ¿El exchange aceptaría la orden de ese nocional al precio dado?
fn ejecutable(nocional: f64, coin: usize, precio: f64) -> bool {
    let paso = SIMBOLOS[coin].1;
    let q = cantidad_enviada(nocional / precio, paso);
    q >= paso && q * precio >= NOCIONAL_MINIMO
}

/// Esperado: la sonda de BNB a 600 USD (paso 0,01) sale con una cantidad que
/// el exchange acepta. Antes: 5,10 USD son 0,0085 BNB, que el ejecutor deja
/// en 0 («Volumen 0»).
#[test]
fn cl98b_sonda_de_bnb_a_600_con_paso_001_es_ejecutable() {
    let a = arena(1_000.0);
    let nocional = evaluar(&a, 1_000.0, 0).expect("precondición: la sonda pasa");
    let q = cantidad_enviada(nocional / 600.0, 0.01);
    println!("observado: nocional {nocional:.4} USD, cantidad enviada {q} BNB ({:.4} USD)", q * 600.0);
    assert!(ejecutable(nocional, 0, 600.0), "la sonda no se puede ejecutar: {nocional} USD ⇒ {q} BNB");
}

/// Esperado: con paso 1 a 3 USD, la sonda son 2 unidades (6 USD). Antes:
/// 1,7 unidades, que el ejecutor deja en 1 (3 USD, bajo el mínimo).
#[test]
fn cl98b_sonda_con_paso_1_a_3_es_ejecutable() {
    let a = arena(1_000.0);
    let nocional = evaluar(&a, 1_000.0, 1).expect("precondición: la sonda pasa");
    let q = cantidad_enviada(nocional / 3.0, 1.0);
    println!("observado: nocional {nocional:.4} USD, cantidad enviada {q} ({:.4} USD)", q * 3.0);
    assert!(ejecutable(nocional, 1, 3.0), "la sonda no se puede ejecutar: {nocional} USD ⇒ {q}");
}

/// Esperado: con paso 1, la sonda es ejecutable a cualquier precio de
/// [1,5; 8] USD. Antes se rechazaba en el 97 % de los precios.
#[test]
fn cl98b_con_paso_1_la_sonda_es_ejecutable_en_todo_el_rango() {
    let a = arena(1_000.0);
    let mut malos = Vec::new();
    let mut n = 0;
    let mut precio = 1.5_f64;
    while precio <= 8.0 + 1e-9 {
        poner_precio(&a, 1, precio);
        n += 1;
        match evaluar(&a, 1_000.0, 1) {
            Ok(nocional) if ejecutable(nocional, 1, precio) => {}
            otro => malos.push((precio, otro)),
        }
        precio += 0.05;
    }
    println!("observado: {} de {n} precios sin orden ejecutable", malos.len());
    assert!(malos.is_empty(), "sin orden ejecutable en {} de {n} precios: {:?}", malos.len(), &malos[..malos.len().min(5)]);
}

/// Falso positivo: la sonda sigue siendo la orden mínima, no una mayor.
#[test]
fn cl98b_la_sonda_sigue_siendo_la_orden_minima_del_lote() {
    let a = arena(1_000.0);
    for (coin, (_, paso, precio)) in SIMBOLOS.iter().enumerate().take(2) {
        let nocional = evaluar(&a, 1_000.0, coin).expect("precondición: la sonda pasa");
        let minimo = nocional_minimo_ejecutable(NOCIONAL_MINIMO + 0.1, *paso, *paso, *precio).unwrap();
        println!("observado: moneda {coin}: nocional {nocional:.4} USD, orden mínima del lote {minimo:.4} USD");
        assert!(nocional >= minimo, "por debajo de la orden mínima: {nocional} < {minimo}");
        assert!(nocional <= minimo * 1.0005 + 1e-9, "la sonda no es la orden mínima: {nocional} > {minimo}");
    }
}

/// Esperado: con 13 USD, una orden mínima de 600 USD (paso 1 a 600) arriesga
/// al stop más de lo que el tope de ruina permite: se rechaza por
/// viabilidad antes de abrir la ranura. Antes pasaba con 5,10 USD, una orden
/// que el exchange no acepta.
#[test]
fn cl98b_la_orden_minima_del_lote_que_no_cabe_se_rechaza_por_viabilidad() {
    let a = arena(13.0);
    let r = evaluar(&a, 13.0, 2);
    println!("observado: orden mínima de 600 USD con 13 USD de capital ⇒ {r:?}");
    assert_eq!(r, Err("viabilidad"));
}

/// Esperado: fuera de la fase de sonda, el rescate al nocional mínimo (D-750)
/// también lleva a una orden ejecutable. Con 13 USD y una ventaja pequeña
/// (acierto 0,55, PF 1,1) el Kelly queda bajo el mínimo y el rescate decide
/// el tamaño. Antes rescataba a 5,10 USD: 0,0085 BNB, que salen como 0.
#[test]
fn cl98b_el_rescate_al_minimo_tambien_es_del_lote() {
    let a = arena(13.0);
    con_historial(&a, 0);
    let m = &a.coins[0].metrics;
    m.win_rate.store(0.55, Relaxed);
    m.gross_wins.store(110.0, Relaxed);
    m.gross_losses.store(100.0, Relaxed);
    m.profit_factor.store(1.1, Relaxed);
    m.kelly_fraction.store(0.01, Relaxed);
    let nocional = evaluar(&a, 13.0, 0).expect("precondición: la orden pasa");
    let q = cantidad_enviada(nocional / 600.0, 0.01);
    println!("observado: con historial y 13 USD, nocional {nocional:.4} USD, cantidad enviada {q} BNB");
    assert!(ejecutable(nocional, 0, 600.0), "el rescate no es ejecutable: {nocional} USD ⇒ {q} BNB");
}

/// La función pura: un múltiplo exacto no sube un paso, manda el mayor de
/// los dos mínimos y las entradas inválidas no dan orden.
#[test]
fn cl98b_cantidad_minima_ejecutable_es_el_menor_multiplo_del_paso() {
    // 5,1/100 = 0,051, múltiplo exacto de 0,001.
    assert_eq!(cantidad_minima_ejecutable(5.1, 0.001, 0.001, 100.0), Some(51.0 * 0.001));
    assert_eq!(cantidad_minima_ejecutable(5.1, 0.01, 0.01, 600.0), Some(0.01));
    assert_eq!(cantidad_minima_ejecutable(5.1, 1.0, 1.0, 3.0), Some(2.0));
    // La cantidad mínima manda cuando es mayor que la del nocional.
    assert_eq!(cantidad_minima_ejecutable(5.0, 1.0, 10.0, 3.0), Some(10.0));
    for (mn, paso, qmin, precio) in [
        (0.0, 0.01, 0.01, 600.0),
        (5.0, 0.0, 0.01, 600.0),
        (5.0, f64::NAN, 0.01, 600.0),
        (5.0, 0.01, 0.01, 0.0),
        (5.0, 0.01, 0.01, f64::INFINITY),
    ] {
        assert_eq!(cantidad_minima_ejecutable(mn, paso, qmin, precio), None, "{mn} {paso} {qmin} {precio}");
    }
    // En toda una rejilla de precios cumple los dos mínimos y es la menor.
    for i in 0..2_000 {
        let precio = 0.5 + i as f64 * 0.37;
        for paso in [0.001, 0.01, 0.1, 1.0] {
            let q = cantidad_minima_ejecutable(5.1, paso, paso, precio).unwrap();
            assert!(q * precio >= 5.1 && q >= paso, "precio {precio} paso {paso}: q {q}");
            let menor = q - paso;
            assert!(menor < paso * (1.0 - 1e-9) || menor * precio < 5.1, "precio {precio} paso {paso}: {q} no es el menor");
        }
    }
}

/// Esperado: el núcleo convierte el nocional en cantidad con el precio de
/// entrada con deslizamiento. La orden mínima del lote (6,003 USD de BNB a
/// 600) a un ask 10 pb más caro serían 0,009996 BNB, que el ejecutor deja en
/// 0; la cantidad no baja de 0,01. Por encima del mínimo no cambia.
#[test]
fn cl98b_el_nucleo_no_baja_de_la_orden_minima_con_deslizamiento() {
    let entrada = 600.0 * 1.001;
    let q = cantidad_de_la_orden(6.003, entrada, NOCIONAL_MINIMO, 0.01, 0.01);
    println!("observado: 6,003 USD a {entrada} ⇒ {q} BNB (sin la guarda {})", 6.003 / entrada);
    assert_eq!(cantidad_enviada(6.003 / entrada, 0.01), 0.0, "precondición: sin la guarda sale 0");
    assert!(cantidad_enviada(q, 0.01) >= 0.01 && cantidad_enviada(q, 0.01) * entrada >= NOCIONAL_MINIMO);
    let grande = cantidad_de_la_orden(60.0, entrada, NOCIONAL_MINIMO, 0.01, 0.01);
    assert_eq!(grande, 60.0 / entrada, "por encima del mínimo la cantidad no cambia");
}
