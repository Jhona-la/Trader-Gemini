//! SIGNO-COMISION (pedido 20261117-2355-signocomision; hallazgo de CODEX-APP en
//! `.agent_tx/diagnostics/auditoria_perdidas/codex_app_demo_testnet.md` §4).
//!
//! El income de Binance es flujo de caja: una COMISSION negativa es un costo pagado.
//! `signed_commission_cost` la convierte en costo positivo; el agregado de CUENTA debe restarlo
//! (`flujo = funding - costo`), porque el profiler, el breaker de Produccion y el certificado demo
//! leen `bruto + flujo` como neto. Antes el host sumaba `costo + funding` y una comision pagada
//! SUBIA el neto. Prueba pura: sin red ni disco. Los valores esperados salen del caso
//! (30 filas, bruto y funding cero, comision total -0,02 USDT), no de una cifra a mano.

use god_engine_core::orchestrator::{EvidenciaDemo, PhaseOrchestrator, SystemPhase};
use quantum_engine::host_oms::{
    agregado_income_de_cuenta, breaker_de_ingreso, ev_neto_por_cierre, signed_commission_cost,
};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

const TOL: f64 = 1e-12;

/// Un simbolo de la pasada, acumulado como lo hace el lazo del host (`god_engine.rs`, match del
/// income): REALIZED_PNL a `registros`, FUNDING_FEE a `funding` (flujo), COMMISSION a `fees` por
/// `signed_commission_cost` (costo).
struct Simbolo {
    registros: Vec<(u64, f64)>,
    funding: f64,
    fees: f64,
}

fn simbolo(filas: &[(&str, f64)]) -> Simbolo {
    let mut s = Simbolo { registros: Vec::new(), funding: 0.0, fees: 0.0 };
    for (i, (tipo, income)) in filas.iter().enumerate() {
        match *tipo {
            "REALIZED_PNL" => s.registros.push((1_000 * i as u64, *income)),
            "FUNDING_FEE" => s.funding += income,
            "COMMISSION" => {
                if let Some(costo) = signed_commission_cost(*income) {
                    s.fees += costo;
                }
            }
            otro => panic!("tipo de prueba desconocido {otro}"),
        }
    }
    s
}

fn agregar(simbolos: &[Simbolo]) -> (f64, f64, u64, u64) {
    agregado_income_de_cuenta(simbolos.iter().map(|s| (s.registros.as_slice(), s.funding, s.fees)))
}

/// 30 filas REALIZED_PNL de bruto 0 y las filas de caja dadas (comision, funding).
fn treinta_filas_con(caja: &[(&str, f64)]) -> Simbolo {
    let mut filas: Vec<(&str, f64)> = vec![("REALIZED_PNL", 0.0); 30];
    filas.extend_from_slice(caja);
    simbolo(&filas)
}

/// Neto por simbolo del FEE-BREAKER (`bruto + funding - costo`, god_engine.rs): la referencia.
fn neto_por_simbolo(s: &Simbolo) -> f64 {
    s.registros.iter().map(|(_, p)| p).sum::<f64>() + s.funding - s.fees
}

fn orquestador(fase: SystemPhase) -> PhaseOrchestrator {
    let mut o = PhaseOrchestrator::new(0, false, Arc::new(AtomicBool::new(true)));
    o.current_phase = fase;
    o
}

#[test]
fn signo_comision_neto_ev_certificado_y_breaker() {
    // --- Caso del pedido: comision de caja -0,02 USDT en total (dos filas), bruto y funding 0.
    let comision_total = -0.015 + -0.005;
    let s = treinta_filas_con(&[("COMMISSION", -0.015), ("COMMISSION", -0.005)]);
    let (bruto, flujo, filas, ganadoras) = agregar(std::slice::from_ref(&s));
    assert_eq!((filas, ganadoras), (30, 0));
    assert_eq!(bruto, 0.0);
    let neto = bruto + flujo;
    assert!((neto - comision_total).abs() < TOL, "neto de cuenta {neto:+.8}, esperado {comision_total:+.8}");
    assert!((neto - neto_por_simbolo(&s)).abs() < TOL, "cuenta y simbolo discrepan");
    let ev = ev_neto_por_cierre(filas, bruto, flujo).expect("EV finito");
    let ev_esperado = comision_total / 30.0;
    assert!(ev < 0.0, "una comision pagada dio EV {ev:+.7} (signo invertido)");
    assert!((ev - ev_esperado).abs() < TOL, "EV {ev:+.10}, esperado {ev_esperado:+.10}");
    assert_eq!(format!("{ev:+.7}"), "-0.0006667");

    // Certificado: con ese EV la puerta de 30 cierres a Produccion queda cerrada.
    let mut o = orquestador(SystemPhase::DemoVerify);
    o.set_evidencia_demo(Some(EvidenciaDemo { cierres: filas, ev_neto_por_trade: ev, wins: ganadoras }));
    assert_eq!(o.on_tick(), SystemPhase::DemoVerify, "un certificado con EV negativo promovio a Produccion");
    // Breaker de Produccion con el mismo agregado: dispara.
    let mut p = orquestador(SystemPhase::ProductionMainnet);
    assert!(breaker_de_ingreso(&mut p, filas, bruto, flujo), "el breaker no disparo con EV negativo");
    assert_eq!(p.current_phase, SystemPhase::DemoVerify);

    // --- Rebate (comision positiva): el neto sube y la puerta abre (control: el gate muerde).
    let s = treinta_filas_con(&[("COMMISSION", 0.02)]);
    let (bruto, flujo, filas, _) = agregar(std::slice::from_ref(&s));
    let ev = ev_neto_por_cierre(filas, bruto, flujo).expect("EV finito");
    assert!((bruto + flujo - 0.02).abs() < TOL);
    assert!((ev - 0.02 / 30.0).abs() < TOL, "rebate: EV {ev:+.10}");
    let mut o = orquestador(SystemPhase::DemoVerify);
    o.set_evidencia_demo(Some(EvidenciaDemo { cierres: filas, ev_neto_por_trade: ev, wins: 0 }));
    assert_eq!(o.on_tick(), SystemPhase::ProductionMainnet, "control: EV positivo con 30 cierres debe promover");
    let mut p = orquestador(SystemPhase::ProductionMainnet);
    assert!(!breaker_de_ingreso(&mut p, filas, bruto, flujo), "rebate: el breaker no debe disparar");

    // --- Funding de ambos signos con la misma comision pagada.
    for (funding, neto_esperado) in [(0.01, 0.01 - 0.02), (-0.01, -0.01 - 0.02)] {
        let s = treinta_filas_con(&[("COMMISSION", -0.02), ("FUNDING_FEE", funding)]);
        let (bruto, flujo, filas, _) = agregar(std::slice::from_ref(&s));
        assert!((bruto + flujo - neto_esperado).abs() < TOL, "funding {funding:+}: neto {:+.8}", bruto + flujo);
        assert!((bruto + flujo - neto_por_simbolo(&s)).abs() < TOL);
        let ev = ev_neto_por_cierre(filas, bruto, flujo).expect("EV finito");
        assert!((ev - neto_esperado / 30.0).abs() < TOL);
    }

    // --- Varios simbolos: el agregado de cuenta es la suma de los netos por simbolo.
    let a = treinta_filas_con(&[("COMMISSION", -0.02), ("FUNDING_FEE", 0.004)]);
    let b = simbolo(&[("REALIZED_PNL", 0.05), ("REALIZED_PNL", -0.01), ("COMMISSION", 0.003), ("FUNDING_FEE", -0.002)]);
    let (bruto, flujo, filas, ganadoras) = agregar(&[a, b]);
    let esperado = (0.0 + 0.004 - 0.02) + (0.05 - 0.01 - 0.002 + 0.003);
    assert_eq!((filas, ganadoras), (32, 1));
    assert!((bruto + flujo - esperado).abs() < TOL, "cuenta {:+.8} != suma por simbolo {esperado:+.8}", bruto + flujo);

    // --- Bruto ausente o no finito: N/D, nunca 0.
    let s = simbolo(&[("COMMISSION", -0.02)]);
    let (bruto, flujo, filas, _) = agregar(std::slice::from_ref(&s));
    assert_eq!(filas, 0);
    assert_eq!(ev_neto_por_cierre(filas, bruto, flujo), None, "sin filas REALIZED_PNL el EV es N/D");
    let mut p = orquestador(SystemPhase::ProductionMainnet);
    assert!(!breaker_de_ingreso(&mut p, filas, bruto, flujo), "sin filas el breaker no decide");
    for malo in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let s = treinta_filas_con(&[("COMMISSION", -0.02), ("REALIZED_PNL", malo)]);
        let (bruto, flujo, filas, _) = agregar(std::slice::from_ref(&s));
        assert_eq!(ev_neto_por_cierre(filas, bruto, flujo), None, "bruto {malo}: EV debe ser N/D");
        // El breaker trata el N/D como no finito y dispara (fail-closed, como antes).
        let mut p = orquestador(SystemPhase::ProductionMainnet);
        assert!(breaker_de_ingreso(&mut p, filas, bruto, flujo), "bruto {malo}: el breaker no disparo");
    }
    let s = treinta_filas_con(&[("FUNDING_FEE", f64::NAN)]);
    let (bruto, flujo, filas, _) = agregar(std::slice::from_ref(&s));
    assert_eq!(ev_neto_por_cierre(filas, bruto, flujo), None, "funding no finito: EV debe ser N/D");
}

/// El lazo del host usa el agregado de cuenta (no una copia propia) y el breaker del bin delega.
/// Ancla de texto: la logica vive en `god_engine.rs`, sin punto de entrada propio.
#[test]
fn signo_comision_el_host_usa_el_agregado() {
    let host = include_str!("../src/bin/god_engine.rs").replace("\r\n", "\n");
    assert!(
        host.contains("let (x_gross, x_fees, x_closes, x_wins) = quantum_engine::host_oms::agregado_income_de_cuenta("),
        "el host no toma el agregado de cuenta de host_oms"
    );
    assert!(!host.contains("x_fees += a.fees + a.funding"), "volvio la suma con el signo del costo");
    assert!(host.contains("quantum_engine::host_oms::breaker_de_ingreso(orch, cierres, bruto, fees)"));
    // El neto por simbolo del FEE-BREAKER es la referencia del signo.
    assert!(host.contains("let net = bruto + a.funding - a.fees;"));
}

/// PENDIENTE separado (no se arregla en este pedido): el contador del certificado y del umbral
/// de 30 cierres (`orchestrator.rs` MIN_CIERRES_PARA_PROD) cuenta FILAS REALIZED_PNL. Una orden
/// cerrada en 3 llenados parciales cuenta 3 "cierres". Si se corrige, esta caracterizacion cambia.
#[test]
fn signo_comision_pendiente_el_contador_cuenta_filas() {
    let s = simbolo(&[("REALIZED_PNL", 0.01), ("REALIZED_PNL", 0.01), ("REALIZED_PNL", 0.01)]);
    let (_, _, filas, ganadoras) = agregar(std::slice::from_ref(&s));
    assert_eq!((filas, ganadoras), (3, 3), "caracterizacion: 3 llenados de una orden = 3 filas");
}
