//! LC-F0-20: el panel del replay de libro (comisiones, PnL bruto, WR antes de
//! comisiones y turnover) debe informar lo que el núcleo COBRÓ en cada
//! cierre, no una tarifa supuesta.
//!
//! El núcleo cobra la entrada como taker y la salida como maker sólo si el
//! cierre es un TP con trade-through; el resto de salidas (SL, trailing,
//! zombie, tiempo) son taker. Antes el replay estimaba
//! `qty·mid_salida·(max(maker, 2 pb) + max(taker, 4 pb))` = 7 pb del nocional
//! de salida, y su turnover como `2·qty·mid_salida`: ni el precio de entrada
//! ni el papel de la salida entraban en el panel.
//!
//! Fixture: posiciones inyectadas desde el observer (misma técnica que
//! `booktick_terminal_mark_contract.rs`) en filas cuyo precio ya cruza su TP o
//! su SL, de modo que es el núcleo real quien las cierra en ese evento. Sin
//! spec registrado no hay entradas propias del núcleo (las tarifas no son
//! observadas), sólo cierres defensivos de lo inyectado; el observer lo
//! comprueba en cada fila. La comisión de entrada la fija la inyección (la
//! misma convención del núcleo: nocional de entrada × taker) y se descuenta
//! de la caja como haría la apertura real.
//!
//! El valor esperado de cada cierre se calcula aquí, sin leer el recibo:
//! precio de salida TP = `entry·(1 ± tp)` (fill maker al límite exacto),
//! precio de salida SL = `last_close_exit_price` del arena (único cierre SL
//! del intervalo observado), comisión de salida = nocional de salida × tarifa
//! del papel. La variación de caja de cada intervalo entre observaciones debe
//! cuadrar con Σ(bruto − comisión de salida): eso valida, contra el arena, que
//! esas son las tarifas que aplicó el núcleo.

use backtest_engine::booktick_replay::{
    run_booktick_replay_with_observer, ReplayConfig, ReplayStats, ReplayTick,
};
use quantum_arena::genome::SuperGenotype;
use quantum_arena::position::PositionHorizon;
use std::sync::atomic::Ordering;

const ROWS: usize = 26;
const WARMUP: usize = 2;
const BASE_TS: u64 = 1_700_000_000_000;
const STEP_MS: u64 = 1_000;
const CAPITAL: f64 = 1_000.0;
const ENTRY: f64 = 100.0;
const QTY: f64 = 1.0;
const MARGIN: f64 = 10.0;
const FLAT: f64 = 100.0;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Salida {
    /// TP con trade-through: salida maker al precio límite.
    Tp,
    /// Stop: salida taker con deslizamiento físico.
    Sl,
}

#[derive(Clone, Copy, Debug)]
struct Inyeccion {
    fila: usize,
    slot: usize,
    is_long: bool,
    tp_price: f64,
    sl_price: f64,
    salida: Salida,
}

/// Escenarios en orden de fila. En la fila 18 hay dos posiciones: el slot 0
/// (TP) cierra en el primer evento de la fila y el slot 1 (SL) en el
/// siguiente (evento trade en modo libro, fila 19 en modo trade-only): el
/// panel debe contar los DOS cierres, cada uno con su propio recibo.
const INYECCIONES: [Inyeccion; 5] = [
    // Largo que cae al SL (taker): bruto < 0.
    Inyeccion {
        fila: 6,
        slot: 0,
        is_long: true,
        tp_price: 110.0,
        sl_price: 99.0,
        salida: Salida::Sl,
    },
    // Corto que llega al TP (maker): bruto > 0.
    Inyeccion {
        fila: 10,
        slot: 0,
        is_long: false,
        tp_price: 99.0,
        sl_price: 105.0,
        salida: Salida::Tp,
    },
    // Largo con TP de 3 pb (maker): bruto > 0 pero neto < 0 tras comisiones,
    // el caso que distingue el WR antes de comisiones del WR neto.
    Inyeccion {
        fila: 14,
        slot: 0,
        is_long: true,
        tp_price: 100.03,
        sl_price: 95.0,
        salida: Salida::Tp,
    },
    Inyeccion {
        fila: 18,
        slot: 0,
        is_long: true,
        tp_price: 102.0,
        sl_price: 90.0,
        salida: Salida::Tp,
    },
    Inyeccion {
        fila: 18,
        slot: 1,
        is_long: false,
        tp_price: 90.0,
        sl_price: 101.0,
        salida: Salida::Sl,
    },
];

fn mid_de_fila(i: usize) -> f64 {
    match i {
        6 | 10 => 97.0,
        14 => 100.10,
        18 | 19 => 103.0,
        _ => FLAT,
    }
}

fn cinta() -> Vec<ReplayTick> {
    (0..ROWS)
        .map(|i| {
            let mid = mid_de_fila(i);
            ReplayTick {
                ts_ms: BASE_TS + i as u64 * STEP_MS,
                bid: mid - 0.01,
                ask: mid + 0.01,
                bid_qty: 2.0,
                ask_qty: 2.0,
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct Abierta {
    inyeccion: Inyeccion,
    generation: u64,
    entry_fee: f64,
}

#[derive(Clone, Copy, Debug)]
struct CierreEsperado {
    salida: Salida,
    entry_fee: f64,
    exit_fee: f64,
    gross: f64,
    net: f64,
    turnover: f64,
}

/// Precio de salida TP tal como lo reconstruye la gestión del núcleo desde el
/// nivel guardado en la posición (fracción y vuelta), no el literal inyectado.
fn precio_tp(iny: &Inyeccion) -> f64 {
    let tp = ((iny.tp_price - ENTRY) / ENTRY).abs();
    if iny.is_long {
        ENTRY * (1.0 + tp)
    } else {
        ENTRY * (1.0 - tp)
    }
}

fn bruto(iny: &Inyeccion, exit_price: f64) -> f64 {
    if iny.is_long {
        (exit_price - ENTRY) * QTY
    } else {
        (ENTRY - exit_price) * QTY
    }
}

fn cerca(actual: f64, esperado: f64, escala: f64, que: &str) {
    let tol = 1e-12 * escala.abs().max(esperado.abs()).max(1.0);
    assert!(
        actual.is_finite() && (actual - esperado).abs() <= tol,
        "{que}: actual={actual:.17e} esperado={esperado:.17e} diferencia={:.3e}",
        actual - esperado
    );
}

fn replay(trade_only: bool) -> (ReplayStats, Vec<CierreEsperado>) {
    let cfg = ReplayConfig {
        initial_capital: CAPITAL,
        warmup_ticks: WARMUP,
        trade_only,
        shift_atr_frac: 0.0,
    };
    let filas = cinta();
    assert!(ROWS - WARMUP > 10, "el replay exige más de 10 filas tras W");
    let genome = SuperGenotype::new_baseline(0.0002, 0.0005);
    let mut abiertas: Vec<Abierta> = Vec::new();
    let mut cierres: Vec<CierreEsperado> = Vec::new();
    let mut caja_previa = CAPITAL;
    let mut visitadas = 0usize;

    let stats = run_booktick_replay_with_observer(&filas, &genome, None, &cfg, |i, core| {
        visitadas += 1;
        let arena = &core.arena;
        let coin = &arena.coins[0];
        let Some((maker, taker)) = arena.config.fee_rates_for(0) else {
            panic!("el escenario explícito exige tarifas del arena");
        };

        // 1. Cierres del intervalo [observación anterior, ahora).
        let caja = arena.unified_capital.load(Ordering::Relaxed);
        let (cerradas, siguen): (Vec<Abierta>, Vec<Abierta>) = abiertas.drain(..).partition(|a| {
            let p = coin.positions.get_slot(a.inyeccion.slot);
            !p.is_open() || p.generation.load(Ordering::Acquire) != a.generation
        });
        abiertas = siguen;
        let sl_cerrados = cerradas
            .iter()
            .filter(|a| a.inyeccion.salida == Salida::Sl)
            .count();
        assert!(
            sl_cerrados <= 1,
            "fila {i}: el fixture sólo admite un SL por intervalo"
        );
        if sl_cerrados == 1 {
            // La gestión recorre las ranuras en orden y cierra una por evento:
            // el SL (ranura mayor) es el último cierre del intervalo.
            let ultima = cerradas.iter().map(|a| a.inyeccion.slot).max();
            let sl = cerradas.iter().find(|a| a.inyeccion.salida == Salida::Sl);
            assert_eq!(
                sl.map(|a| a.inyeccion.slot),
                ultima,
                "fila {i}: el SL debe cerrar el último"
            );
            assert_eq!(
                coin.last_close_reason.load(Ordering::Relaxed),
                2,
                "fila {i}: motivo SL"
            );
        } else if cerradas.len() == 1 {
            assert_eq!(
                coin.last_close_reason.load(Ordering::Relaxed),
                1,
                "fila {i}: motivo TP"
            );
        }
        let mut variacion_esperada = 0.0;
        for a in &cerradas {
            let iny = a.inyeccion;
            let exit_price = match iny.salida {
                Salida::Tp => precio_tp(&iny),
                Salida::Sl => coin.last_close_exit_price.load(Ordering::Relaxed),
            };
            if cerradas.len() == 1 {
                assert_eq!(
                    coin.last_close_exit_price.load(Ordering::Relaxed).to_bits(),
                    exit_price.to_bits(),
                    "fila {i}: precio de salida reconstruido"
                );
            }
            match iny.salida {
                Salida::Tp => assert!(
                    if iny.is_long {
                        exit_price > ENTRY
                    } else {
                        exit_price < ENTRY
                    },
                    "fila {i}: un TP sale con beneficio bruto"
                ),
                Salida::Sl => assert!(
                    if iny.is_long {
                        exit_price < iny.sl_price
                    } else {
                        exit_price > iny.sl_price
                    },
                    "fila {i}: el SL del fixture cruza su nivel con deslizamiento taker"
                ),
            }
            let tarifa = match iny.salida {
                Salida::Tp => maker,
                Salida::Sl => taker,
            };
            let gross = bruto(&iny, exit_price);
            let exit_fee = (QTY * exit_price) * tarifa;
            let net = (gross - exit_fee) - a.entry_fee;
            variacion_esperada += gross - exit_fee;
            cierres.push(CierreEsperado {
                salida: iny.salida,
                entry_fee: a.entry_fee,
                exit_fee,
                gross,
                net,
                turnover: QTY * ENTRY + QTY * exit_price,
            });
        }
        let variacion = caja - caja_previa;
        assert!(
            (variacion - variacion_esperada).abs() <= 1e-9,
            "fila {i}: la caja del núcleo ({variacion:.17e}) no cuadra con Σ(bruto − comisión \
             de salida) = {variacion_esperada:.17e}: las tarifas del fixture no son las del núcleo"
        );

        // 2. Ninguna posición que el fixture no haya inyectado.
        for (slot, p) in coin.positions.slots().into_iter().enumerate() {
            if p.is_open() {
                assert!(
                    abiertas.iter().any(|a| a.inyeccion.slot == slot
                        && p.generation.load(Ordering::Acquire) == a.generation),
                    "fila {i}: el núcleo abrió una posición propia en la ranura {slot}"
                );
            }
        }
        assert!(
            arena
                .coins
                .iter()
                .skip(1)
                .all(|c| !c.positions.is_any_open()),
            "fila {i}: posición en otra moneda"
        );

        // 3. Inyecciones de esta fila.
        let mut caja_tras = caja;
        for iny in INYECCIONES.iter().filter(|iny| iny.fila == i) {
            let slot = coin.positions.get_slot(iny.slot);
            assert!(!slot.is_open(), "fila {i}: ranura {} ocupada", iny.slot);
            let entry_fee = (QTY * ENTRY) * taker;
            let abierta = slot.open_with_fee(
                iny.is_long,
                ENTRY,
                QTY,
                MARGIN,
                BASE_TS + (i as u64 - 1) * STEP_MS,
                iny.tp_price,
                iny.sl_price,
                PositionHorizon::Continuous,
                0.5,
                0.5,
                entry_fee,
            );
            assert!(
                abierta,
                "fila {i}: la inyección debe abrir la ranura {}",
                iny.slot
            );
            let generation = slot.generation.load(Ordering::Acquire);
            let snap = slot.snapshot().expect("ranura recién abierta");
            assert_eq!(snap.entry_fee.to_bits(), entry_fee.to_bits());
            assert_eq!(snap.quantity.to_bits(), QTY.to_bits());
            // La apertura real descuenta la comisión de entrada de la caja y
            // reserva el margen; el cierre libera el margen y suma bruto − salida.
            caja_tras -= entry_fee;
            arena.unified_capital.store(caja_tras, Ordering::Relaxed);
            arena.used_margin.fetch_add(MARGIN, Ordering::Relaxed);
            abiertas.push(Abierta {
                inyeccion: *iny,
                generation,
                entry_fee,
            });
        }
        caja_previa = caja_tras;
    });

    assert_eq!(visitadas, ROWS, "el fixture debe recorrer toda la cinta");
    assert!(
        abiertas.is_empty(),
        "todas las inyecciones deben cerrar: {abiertas:?}"
    );
    assert_eq!(
        cierres.len(),
        INYECCIONES.len(),
        "cada inyección produce un cierre"
    );
    assert!(
        cierres.iter().any(|c| c.salida == Salida::Sl),
        "el fixture cubre una salida taker"
    );
    assert!(
        cierres.iter().any(|c| c.salida == Salida::Tp),
        "el fixture cubre una salida maker"
    );
    assert!(
        cierres.iter().any(|c| c.gross > 0.0 && c.net < 0.0),
        "el fixture cubre un cierre con bruto > 0 y neto < 0"
    );
    (stats, cierres)
}

fn comprobar_panel(trade_only: bool) {
    let (stats, cierres) = replay(trade_only);
    let modo = if trade_only { "trade-only" } else { "libro" };
    let fees: f64 = cierres.iter().map(|c| c.entry_fee + c.exit_fee).sum();
    let gross: f64 = cierres.iter().map(|c| c.gross).sum();
    let net: f64 = cierres.iter().map(|c| c.net).sum();
    let turnover: f64 = cierres.iter().map(|c| c.turnover).sum();
    let wins_gross = cierres.iter().filter(|c| c.gross > 0.0).count() as u64;
    let wins_net = cierres.iter().filter(|c| c.net > 0.0).count() as u64;
    for (k, c) in cierres.iter().enumerate() {
        println!(
            "[LC-F0-20 {modo}] cierre {k}: {:?} entrada={:.6e} salida={:.6e} bruto={:.6e} neto={:.6e}",
            c.salida, c.entry_fee, c.exit_fee, c.gross, c.net
        );
    }
    println!(
        "[LC-F0-20 {modo}] panel: trades={} fees={:.17e} (esperado {fees:.17e}) bruto={:.17e} \
         (esperado {gross:.17e}) neto={:.17e} (esperado {net:.17e}) wins_gross={} wins_net={} \
         turnover/día={:.17e}",
        stats.trades,
        stats.fees_est,
        stats.gross_pnl,
        stats.net_pnl,
        stats.wins_gross,
        stats.wins_net,
        stats.metrics.turnover_per_day
    );

    // El neto y la caja son del núcleo: este átomo no los toca.
    assert_eq!(
        stats.trades,
        cierres.len() as u64,
        "{modo}: cierres contados"
    );
    cerca(
        stats.net_pnl,
        net,
        net,
        &format!("{modo}: PnL neto del núcleo"),
    );
    cerca(
        stats.final_capital,
        CAPITAL + net,
        CAPITAL,
        &format!("{modo}: caja final"),
    );
    assert_eq!(stats.open_positions_end, 0, "{modo}: sin MTM terminal");
    assert_eq!(stats.wins_net, wins_net, "{modo}: WR neto");

    // Lo que corrige LC-F0-20: comisiones, bruto, WR antes de comisiones y
    // turnover salen de lo que cobró el núcleo.
    cerca(
        stats.fees_est,
        fees,
        fees,
        &format!("{modo}: comisiones = entrada + salida cobradas"),
    );
    cerca(
        stats.gross_pnl,
        gross,
        gross.abs() + fees,
        &format!("{modo}: PnL bruto"),
    );
    cerca(
        stats.gross_pnl - stats.fees_est,
        stats.net_pnl,
        stats.gross_pnl.abs() + stats.fees_est.abs(),
        &format!("{modo}: bruto − comisiones = neto"),
    );
    assert_eq!(
        stats.wins_gross, wins_gross,
        "{modo}: WR antes de comisiones usa el bruto"
    );
    let dias = stats.metrics.span_days;
    assert!(dias > 0.0, "{modo}: span ejecutado");
    cerca(
        stats.metrics.turnover_per_day,
        turnover / CAPITAL / dias,
        turnover / CAPITAL / dias,
        &format!("{modo}: turnover = Σ qty·(entrada + salida)"),
    );
}

fn en_hilo_grande(nombre: &str, f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(nombre.into())
        .stack_size(64 << 20)
        .spawn(f)
        .expect("lanzar el hilo del contrato")
        .join()
        .expect("contrato LC-F0-20 fallido");
}

#[test]
fn panel_del_replay_trade_only_usa_lo_que_cobro_el_nucleo() {
    en_hilo_grande("lc-f0-20-trade-only", || comprobar_panel(true));
}

#[test]
fn panel_del_replay_de_libro_usa_lo_que_cobro_el_nucleo() {
    en_hilo_grande("lc-f0-20-libro", || comprobar_panel(false));
}
