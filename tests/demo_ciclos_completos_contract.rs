//! CICLOS-2 (pedido 20261118-0535-cicloscompletos; espec. `ciclos_completos_v2.md` §3, matriz 01-10 y
//! regresiones). El productor real (`ciclos_completos::certificar_ciclos`) recibe fills, cashflows,
//! fotos de borde y cobertura inyectados en memoria; su recibo pasa por el escritor y el lector
//! reales (`escribir_certificado_v2` / `cargar_evidencia_ciclos_v2`, en un directorio temporal) y de
//! ahi a la puerta real (`PhaseOrchestrator`). Ningun contador se inyecta ya calculado.
//! Importes del venue en punto fijo (1e-8): la cuadratura es por igualdad exacta, sin tolerancia.
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use execution_engine::importe_venue::{importe_de_texto, CausaImporteNd, ImporteVenue};
use god_engine_core::orchestrator::{EvidenciaDemo, PhaseOrchestrator, SystemPhase, MIN_CIERRES_PARA_PROD};
use quantum_engine::ciclos_completos::*;

const T0: u64 = 1_000_000;
const T1: u64 = 900_000_000;
const CUENTA: &str = "cuenta-testnet-1";

fn libro_en(entorno: &str, cuenta: &str, simbolo: &str) -> Libro {
    Libro { entorno: entorno.into(), venue: "binance-usdm".into(), cuenta: cuenta.into(), simbolo: simbolo.into() }
}
#[allow(dead_code)]
fn libro(simbolo: &str) -> Libro {
    libro_en(ENTORNO_TESTNET, CUENTA, simbolo)
}
fn ambito_en(entorno: &str) -> AmbitoCertificado {
    AmbitoCertificado {
        entorno: entorno.into(),
        venue: "binance-usdm".into(),
        cuenta: CUENTA.into(),
        sesion: "sesion-1".into(),
        estrategia: "estrategia-firmada".into(),
        sha_binario: "b".repeat(64),
        sha_config: "c".repeat(64),
        roster: vec!["BTCUSDT".into(), "ETHUSDT".into()],
    }
}
fn esperado() -> AmbitoEsperado {
    AmbitoEsperado { ambito: ambito_en(ENTORNO_TESTNET), ventana_desde_ms: T0, ventana_hasta_ms: T1 }
}
fn imp(t: &str) -> Result<ImporteVenue, CausaImporteNd> {
    importe_de_texto(t)
}
fn e8(t: &str) -> i64 {
    match importe_de_texto(t) {
        Ok(i) => i.e8,
        Err(c) => panic!("fixture {t}: {c:?}"),
    }
}

/// Generador de evidencia: cada fill avanza el reloj 1 s y lleva su fila COMMISSION explicita
/// (salvo que se pida lo contrario) y, si se indica, su fila REALIZED_PNL.
struct Gen {
    entorno: String,
    cuenta: String,
    fills: Vec<FillVenue>,
    caja: Vec<CajaVenue>,
    t: u64,
    tid: u64,
    tran: u64,
}

impl Gen {
    fn nuevo() -> Self {
        Self { entorno: ENTORNO_TESTNET.into(), cuenta: CUENTA.into(), fills: vec![], caja: vec![], t: T0 + 1_000, tid: 1, tran: 1 }
    }
    fn libro(&self, s: &str) -> Libro {
        libro_en(&self.entorno, &self.cuenta, s)
    }
    #[allow(clippy::too_many_arguments)]
    fn fill_con(&mut self, s: &str, lado: Lado, pierna: Pierna, qty: &str, rp: Result<ImporteVenue, CausaImporteNd>, realized: Option<&str>, comision: Option<&str>, causa: CausaSalida) -> u64 {
        let id = self.tid;
        self.tid += 1;
        self.t += 1_000;
        self.fills.push(FillVenue {
            libro: self.libro(s),
            trade_id: id,
            order_id: id,
            lado: Some(lado),
            pierna,
            qty_e8: Some(e8(qty)),
            precio_e8: Some(e8("100")),
            tiempo_ms: self.t,
            rp,
            canal: Canal::Ws,
            causa,
        });
        if let Some(r) = realized {
            self.caja_con(s, TipoCaja::RealizedPnl, Some(id), self.t, "USDT", imp(r), None);
        }
        if let Some(c) = comision {
            self.caja_con(s, TipoCaja::Comision, Some(id), self.t, "USDT", imp(c), None);
        }
        id
    }
    /// One-way: reduccion con su fila REALIZED_PNL igual al rp; apertura con rp "0" y sin fila.
    fn ow(&mut self, s: &str, lado: Lado, qty: &str, rp: &str) -> u64 {
        let realized = if rp == "0" { None } else { Some(rp) };
        self.fill_con(s, lado, Pierna::Both, qty, imp(rp), realized, Some("0"), CausaSalida::Normal)
    }
    #[allow(clippy::too_many_arguments)]
    fn caja_con(&mut self, s: &str, tipo: TipoCaja, trade_id: Option<u64>, t: u64, asset: &str, importe: Result<ImporteVenue, CausaImporteNd>, tramo: Option<Tramo>) {
        let tran = self.tran;
        self.tran += 1;
        self.caja.push(CajaVenue { libro: self.libro(s), tipo, tran_id: tran, trade_id, tiempo_ms: t, asset: asset.into(), importe, tramo });
    }
    fn entrada(&self, modo: ModoPosicion) -> EntradaCiclos {
        EntradaCiclos {
            ambito: ambito_en(&self.entorno),
            modo,
            ventana_desde_ms: T0,
            ventana_hasta_ms: T1,
            fills: self.fills.clone(),
            caja: self.caja.clone(),
            foto_t0: Some(BTreeMap::new()),
            foto_t1: Some(BTreeMap::new()),
            cobertura_fills_completa: true,
            cobertura_funding_completa: true,
        }
    }
}

/// Diez episodios: compra de 3 a 100 y tres ventas de 1 a 101, cada venta con rp y REALIZED_PNL +1.
fn diez_episodios() -> Gen {
    let mut g = Gen::nuevo();
    for _ in 0..10 {
        g.ow("BTCUSDT", Lado::Compra, "3", "0");
        for _ in 0..3 {
            g.ow("BTCUSDT", Lado::Venta, "1", "1");
        }
    }
    g
}

fn episodios(n: usize, rp: &str) -> Gen {
    let mut g = Gen::nuevo();
    for _ in 0..n {
        g.ow("BTCUSDT", Lado::Compra, "1", "0");
        g.ow("BTCUSDT", Lado::Venta, "1", rp);
    }
    g
}

struct Dir(PathBuf);
impl Dir {
    fn nuevo(etiqueta: &str) -> Self {
        let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let d = std::env::temp_dir().join(format!("tg-ciclos-{etiqueta}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap_or_else(|e| panic!("dir temporal: {e}"));
        Dir(d)
    }
    fn cert(&self) -> PathBuf {
        self.0.join(ARCHIVO_CERTIFICADO_V2)
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fase_con(ev: Option<EvidenciaDemo>, oms: bool) -> SystemPhase {
    let mut p = PhaseOrchestrator::new_with_oms(1, false, Arc::new(AtomicBool::new(true)), Arc::new(AtomicBool::new(oms)));
    p.set_evidencia_demo(ev);
    for _ in 0..5 {
        p.on_tick();
    }
    p.current_phase
}

/// Productor -> escritor -> lector (como tras un reinicio) -> puerta.
fn por_la_puerta(etiqueta: &str, cert: &CertificadoCiclosV2) -> (Result<EvidenciaDemo, MotivoSinEvidencia>, SystemPhase) {
    let d = Dir::nuevo(etiqueta);
    let _ = escribir_certificado_v2(&d.cert(), cert);
    let ev = cargar_evidencia_ciclos_v2(&d.cert(), Some(&esperado()));
    let fase = fase_con(ev.ok(), true);
    (ev, fase)
}

fn tiene(cert: &CertificadoCiclosV2, causa: CausaCiclosNd) -> bool {
    let c = format!("{causa:?}");
    cert.causas_nd.contains(&c) || cert.ciclos.iter().any(|x| x.causas_nd.contains(&c))
}

#[test]
fn demo_ciclos_01_treinta_filas_diez_ciclos_puerta_cerrada() {
    let cert = certificar_ciclos(&diez_episodios().entrada(ModoPosicion::OneWay));
    assert_eq!(cert.filas_realized_pnl, 30, "las 30 filas se conservan como diagnostico");
    assert_eq!(cert.n_ciclos_validos, 10, "10 ciclos completos, no 30 cierres: {cert:?}");
    assert_eq!(cert.ciclos_estructurales, 10);
    assert_eq!(cert.estado, ESTADO_VALIDO, "comision y funding cero acreditados: {:?}", cert.causas_nd);
    assert_eq!(cert.neto_total_e8, e8("30"));
    assert!(cert.ciclos.iter().all(|c| c.comision_e8 == 0 && c.funding_e8 == 0 && c.neto_e8 == Some(e8("3"))));
    assert_eq!(cert.ev_neto_por_ciclo, Some(3.0));
    let (ev, fase) = por_la_puerta("c01", &cert);
    assert_eq!(ev, Ok(EvidenciaDemo { cierres: 10, ev_neto_por_trade: 3.0, wins: 10 }));
    assert_eq!(fase, SystemPhase::DemoVerify, "10 ciclos < {MIN_CIERRES_PARA_PROD}: la puerta sigue cerrada");
}

/// Hallazgo: la ruta por filas da 30 "cierres" con EV +1 por fila y esa evidencia abre la puerta.
/// Desde CICLOS-2 esa ruta queda solo para diagnostico, el freno, el profiler y el panel.
#[test]
fn demo_ciclos_01b_hallazgo_la_ruta_por_filas_cuenta_30() {
    let g = diez_episodios();
    let registros: Vec<(u64, f64)> = g.caja.iter().filter(|c| c.tipo == TipoCaja::RealizedPnl).map(|c| (c.tiempo_ms, 1.0)).collect();
    let (bruto, flujo, filas, ganadoras) = quantum_engine::host_oms::agregado_income_de_cuenta([(registros.as_slice(), 0.0, 0.0)]);
    assert_eq!((filas, ganadoras), (30, 30));
    let ev = quantum_engine::host_oms::ev_neto_por_cierre(filas, bruto, flujo);
    assert_eq!(ev, Some(1.0));
    let legado = EvidenciaDemo { cierres: filas, ev_neto_por_trade: 1.0, wins: ganadoras };
    assert_eq!(fase_con(Some(legado), true), SystemPhase::ProductionMainnet, "control: con 30 la puerta muerde");
    let d = Dir::nuevo("c01b");
    std::fs::write(d.cert(), serde_json::to_string(&legado).unwrap_or_default()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(cargar_evidencia_ciclos_v2(&d.cert(), Some(&esperado())), Err(MotivoSinEvidencia::Legado));
}

#[test]
fn demo_ciclos_02_umbral_y_signo_del_ev() {
    let fase = |g: Gen, et: &str| por_la_puerta(et, &certificar_ciclos(&g.entrada(ModoPosicion::OneWay)));
    let (ev29, f29) = fase(episodios(29, "1"), "c02a");
    assert_eq!(ev29.map(|e| e.cierres), Ok(29));
    assert_eq!(f29, SystemPhase::DemoVerify, "29 ciclos deniegan");
    let (ev30, f30) = fase(episodios(30, "1"), "c02b");
    assert_eq!(ev30.map(|e| (e.cierres, e.ev_neto_por_trade)), Ok((30, 1.0)));
    assert_eq!(f30, SystemPhase::ProductionMainnet, "control: 30 con EV > 0 admiten");
    for (rp, et) in [("0", "c02c"), ("-1", "c02d")] {
        let (ev, f) = fase(episodios(30, rp), et);
        assert_eq!(ev.map(|e| e.cierres), Ok(30), "rp {rp}");
        assert_eq!(f, SystemPhase::DemoVerify, "30 con EV {rp} deniegan");
    }
    let mut g = episodios(30, "1");
    let ultimo = g.fills.len() - 1;
    g.fills[ultimo].rp = Err(CausaImporteNd::Ausente);
    let (ev, f) = fase(g, "c02e");
    assert_eq!(ev, Err(MotivoSinEvidencia::Incompleto), "30 con EV N/D: sin evidencia");
    assert_eq!(f, SystemPhase::DemoVerify);
    // Los vetos vigentes tienen precedencia: sin OMS reconciliado, 30 validos no abren.
    let cert = certificar_ciclos(&episodios(30, "1").entrada(ModoPosicion::OneWay));
    let d = Dir::nuevo("c02f");
    let _ = escribir_certificado_v2(&d.cert(), &cert);
    let ev = cargar_evidencia_ciclos_v2(&d.cert(), Some(&esperado()));
    assert!(ev.is_ok());
    assert_ne!(fase_con(ev.ok(), false), SystemPhase::ProductionMainnet);
}

#[test]
fn demo_ciclos_03_parciales_refuerzos_y_remanente() {
    let mut g = Gen::nuevo();
    g.ow("BTCUSDT", Lado::Compra, "3", "0");
    g.ow("BTCUSDT", Lado::Venta, "1", "1");
    g.ow("BTCUSDT", Lado::Compra, "1", "0");
    g.ow("BTCUSDT", Lado::Venta, "2", "2");
    let mut e = g.entrada(ModoPosicion::OneWay);
    e.foto_t1 = Some(BTreeMap::from([("BTCUSDT".to_string(), Foto { both_e8: e8("1"), ..Foto::default() })]));
    let abierto = certificar_ciclos(&e);
    assert_eq!((abierto.ciclos_estructurales, abierto.n_ciclos_validos, abierto.abiertos), (0, 0, 1), "queda 1: 0 ciclos");
    assert_eq!(abierto.filas_realized_pnl, 2);
    assert_eq!(abierto.estado, ESTADO_ND);
    g.ow("BTCUSDT", Lado::Venta, "1", "1");
    let cerrado = certificar_ciclos(&g.entrada(ModoPosicion::OneWay));
    assert_eq!((cerrado.n_ciclos_validos, cerrado.filas_realized_pnl), (1, 3));
    assert_eq!(cerrado.ciclos[0].fills, 5);
    assert_eq!(cerrado.neto_total_e8, e8("4"));
    // Remanente exacto, por pequeno que sea (1e-8), mantiene el ciclo abierto.
    let mut p = Gen::nuevo();
    p.ow("BTCUSDT", Lado::Compra, "1", "0");
    p.ow("BTCUSDT", Lado::Venta, "0.99999999", "1");
    let mut e = p.entrada(ModoPosicion::OneWay);
    e.foto_t1 = Some(BTreeMap::from([("BTCUSDT".to_string(), Foto { both_e8: 1, ..Foto::default() })]));
    let polvo = certificar_ciclos(&e);
    assert_eq!((polvo.n_ciclos_validos, polvo.abiertos), (0, 1));
}

#[test]
fn demo_ciclos_04_hedge_y_separacion_de_identidades() {
    let mut g = Gen::nuevo();
    g.fill_con("BTCUSDT", Lado::Compra, Pierna::Long, "2", imp("0"), None, Some("0"), CausaSalida::Normal);
    g.fill_con("BTCUSDT", Lado::Venta, Pierna::Short, "2", imp("0"), None, Some("0"), CausaSalida::Normal);
    g.fill_con("BTCUSDT", Lado::Venta, Pierna::Long, "2", imp("1"), Some("1"), Some("0"), CausaSalida::Normal);
    let mut e = g.entrada(ModoPosicion::Hedge);
    e.foto_t1 = Some(BTreeMap::from([("BTCUSDT".to_string(), Foto { short_e8: e8("2"), ..Foto::default() })]));
    let medio = certificar_ciclos(&e);
    assert_eq!((medio.ciclos_estructurales, medio.abiertos), (0, 1), "(0,2): la pierna SHORT sigue abierta");
    g.fill_con("BTCUSDT", Lado::Compra, Pierna::Short, "2", imp("-1"), Some("-1"), Some("0"), CausaSalida::Normal);
    let fin = certificar_ciclos(&g.entrada(ModoPosicion::Hedge));
    assert_eq!((fin.n_ciclos_validos, fin.estado.as_str()), (1, ESTADO_VALIDO), "{:?}", fin.causas_nd);
    assert_eq!(fin.ciclos[0].fills, 4);
    // Dos simbolos intercalados: dos libros, dos ciclos.
    let mut d = Gen::nuevo();
    d.ow("BTCUSDT", Lado::Compra, "1", "0");
    d.ow("ETHUSDT", Lado::Compra, "1", "0");
    d.ow("BTCUSDT", Lado::Venta, "1", "1");
    d.ow("ETHUSDT", Lado::Venta, "1", "2");
    let dos = certificar_ciclos(&d.entrada(ModoPosicion::OneWay));
    assert_eq!((dos.n_ciclos_validos, dos.neto_total_e8), (2, e8("3")));
    // El mismo tradeId en otra cuenta y en otro entorno es otra historia: no contradice ni se suma.
    let mut otra = d.fills.clone();
    for (cuenta, entorno) in [("otra-cuenta", ENTORNO_TESTNET), (CUENTA, "mainnet")] {
        for f in &d.fills {
            let mut x = f.clone();
            x.libro = libro_en(entorno, cuenta, &f.libro.simbolo);
            x.qty_e8 = Some(e8("5"));
            otra.push(x);
        }
    }
    let mut e = d.entrada(ModoPosicion::OneWay);
    e.fills = otra;
    let separado = certificar_ciclos(&e);
    assert_eq!((separado.n_ciclos_validos, separado.estado.as_str()), (2, ESTADO_VALIDO), "{:?}", separado.causas_nd);
    assert_eq!(separado.hash_ciclos, dos.hash_ciclos);
}

#[test]
fn demo_ciclos_05_kill_switch_liquidacion_y_adl() {
    for causa in [CausaSalida::KillSwitch, CausaSalida::Liquidacion, CausaSalida::Adl] {
        let mut g = Gen::nuevo();
        g.ow("BTCUSDT", Lado::Compra, "3", "0");
        g.ow("BTCUSDT", Lado::Venta, "1", "-1");
        g.ow("BTCUSDT", Lado::Venta, "1", "-1");
        let id = g.fill_con("BTCUSDT", Lado::Venta, Pierna::Both, "1", imp("-1"), Some("-1"), Some("0"), causa);
        let t = g.t;
        g.caja_con("BTCUSDT", TipoCaja::CargoForzoso, Some(id), t, "USDT", imp("-0.5"), None);
        let cert = certificar_ciclos(&g.entrada(ModoPosicion::OneWay));
        assert_eq!((cert.n_ciclos_validos, cert.ganadores), (1, 0), "{causa:?}: un ciclo perdedor incluido");
        assert_eq!(cert.ciclos[0].cargos_e8, e8("-0.5"));
        assert_eq!(cert.neto_total_e8, e8("-3.5"), "{causa:?}");
        // Cargo forzoso de importe desconocido: EV N/D.
        if let Some(c) = g.caja.last_mut() {
            c.importe = Err(CausaImporteNd::Formato);
        }
        let nd = certificar_ciclos(&g.entrada(ModoPosicion::OneWay));
        assert!(tiene(&nd, CausaCiclosNd::CargoForzosoIlegible), "{causa:?}");
        assert_eq!((nd.n_ciclos_validos, nd.ev_neto_por_ciclo), (0, None));
    }
    // Aviso o ACK aislado con posicion pendiente: 0 ciclos y N/D.
    let mut g = Gen::nuevo();
    g.ow("BTCUSDT", Lado::Compra, "3", "0");
    g.fill_con("BTCUSDT", Lado::Venta, Pierna::Both, "1", imp("-1"), Some("-1"), Some("0"), CausaSalida::KillSwitch);
    let mut e = g.entrada(ModoPosicion::OneWay);
    e.foto_t1 = Some(BTreeMap::from([("BTCUSDT".to_string(), Foto { both_e8: e8("2"), ..Foto::default() })]));
    let ack = certificar_ciclos(&e);
    assert_eq!((ack.ciclos_estructurales, ack.estado.as_str(), ack.ev_neto_por_ciclo), (0, ESTADO_ND, None));
}

#[test]
fn demo_ciclos_06_bordes_de_ventana() {
    // Arrastrado al inicio (cerrado dentro de la ventana): fuera del conteo, comprobante N/D.
    let mut g = Gen::nuevo();
    g.ow("BTCUSDT", Lado::Venta, "2", "1");
    let mut e = g.entrada(ModoPosicion::OneWay);
    e.foto_t0 = Some(BTreeMap::from([("BTCUSDT".to_string(), Foto { both_e8: e8("2"), ..Foto::default() })]));
    let arrastre = certificar_ciclos(&e);
    assert_eq!((arrastre.ciclos_estructurales, arrastre.arrastres, arrastre.estado.as_str()), (0, 1, ESTADO_ND));
    // Ultimo fill exactamente en T1: fuera de la ventana; el ciclo queda abierto.
    let mut g = episodios(1, "1");
    g.fills[1].tiempo_ms = T1;
    let mut e = g.entrada(ModoPosicion::OneWay);
    e.foto_t1 = Some(BTreeMap::from([("BTCUSDT".to_string(), Foto { both_e8: e8("1"), ..Foto::default() })]));
    let en_t1 = certificar_ciclos(&e);
    assert_eq!((en_t1.n_ciclos_validos, en_t1.abiertos), (0, 1));
    // Sin prueba de posicion de borde: N/D.
    let mut e = episodios(1, "1").entrada(ModoPosicion::OneWay);
    e.foto_t0 = None;
    let sin_foto = certificar_ciclos(&e);
    assert!(tiene(&sin_foto, CausaCiclosNd::FotoBordeAusente));
    assert_eq!(sin_foto.ev_neto_por_ciclo, None);
    // Foto de cierre que no cuadra con la reconstruccion: N/D.
    let mut e = episodios(1, "1").entrada(ModoPosicion::OneWay);
    e.foto_t1 = Some(BTreeMap::from([("BTCUSDT".to_string(), Foto { both_e8: e8("1"), ..Foto::default() })]));
    assert!(tiene(&certificar_ciclos(&e), CausaCiclosNd::DescuadreFotoT1));
}

#[test]
fn demo_ciclos_07_reversion_con_un_fill() {
    // +2 -> -1 con una venta de 3 y UNA fila de comision: sin importes explicitos por tramo, N/D (D).
    let mut g = Gen::nuevo();
    g.ow("BTCUSDT", Lado::Compra, "2", "0");
    g.fill_con("BTCUSDT", Lado::Venta, Pierna::Both, "3", imp("0.4"), Some("0.4"), Some("-0.03"), CausaSalida::Normal);
    g.fill_con("BTCUSDT", Lado::Compra, Pierna::Both, "1", imp("0.1"), Some("0.1"), Some("0"), CausaSalida::Normal);
    let d = certificar_ciclos(&g.entrada(ModoPosicion::OneWay));
    assert_eq!(d.ciclos_estructurales, 2, "cierre de 2 y apertura de 1 con el mismo fill");
    assert!(tiene(&d, CausaCiclosNd::ReversionSinImportesExplicitos));
    assert_eq!((d.n_ciclos_validos, d.estado.as_str(), d.ev_neto_por_ciclo), (0, ESTADO_ND, None));
    let (ev, fase) = por_la_puerta("c07", &d);
    assert_eq!(ev, Err(MotivoSinEvidencia::Incompleto));
    assert_eq!(fase, SystemPhase::DemoVerify);
    // Opcion C: el venue entrega el importe de cada tramo; el tramo de cierre va al ciclo que cierra.
    let mut c = Gen::nuevo();
    c.ow("BTCUSDT", Lado::Compra, "2", "0");
    let id = c.fill_con("BTCUSDT", Lado::Venta, Pierna::Both, "3", imp("0.4"), Some("0.4"), None, CausaSalida::Normal);
    let t = c.t;
    c.caja_con("BTCUSDT", TipoCaja::Comision, Some(id), t, "USDT", imp("-0.02"), Some(Tramo::Cierre));
    c.caja_con("BTCUSDT", TipoCaja::Comision, Some(id), t, "USDT", imp("-0.01"), Some(Tramo::Apertura));
    c.fill_con("BTCUSDT", Lado::Compra, Pierna::Both, "1", imp("0.1"), Some("0.1"), Some("0"), CausaSalida::Normal);
    let ok = certificar_ciclos(&c.entrada(ModoPosicion::OneWay));
    assert_eq!((ok.n_ciclos_validos, ok.estado.as_str()), (2, ESTADO_VALIDO), "{:?}", ok.causas_nd);
    assert_eq!(ok.ciclos.iter().map(|x| x.neto_e8).collect::<Vec<_>>(), vec![Some(e8("0.38")), Some(e8("0.09"))]);
    assert_eq!(ok.neto_total_e8, e8("0.47"), "sin residuo perdido: -0,03 entero repartido");
}

#[test]
fn demo_ciclos_08_replay_orden_y_cobertura() {
    let g = diez_episodios();
    let base = certificar_ciclos(&g.entrada(ModoPosicion::OneWay));
    // Replay identico por WS y REST: el mismo resultado.
    let mut e = g.entrada(ModoPosicion::OneWay);
    e.fills.extend(g.fills.iter().cloned().map(|mut f| {
        f.canal = Canal::Rest;
        f
    }));
    e.caja.extend(g.caja.iter().cloned());
    let replay = certificar_ciclos(&e);
    assert_eq!((replay.n_ciclos_validos, replay.neto_total_e8, &replay.hash_ciclos), (10, e8("30"), &base.hash_ciclos));
    // Desorden recuperable: la misma evidencia en otro orden.
    let mut e = g.entrada(ModoPosicion::OneWay);
    e.fills.reverse();
    e.caja.reverse();
    assert_eq!(certificar_ciclos(&e).hash_ciclos, base.hash_ciclos);
    // Duplicado contradictorio.
    let mut e = g.entrada(ModoPosicion::OneWay);
    let mut x = g.fills[1].clone();
    x.qty_e8 = Some(e8("2"));
    x.canal = Canal::Rest;
    e.fills.push(x);
    let dup = certificar_ciclos(&e);
    assert!(tiene(&dup, CausaCiclosNd::DuplicadoContradictorio));
    assert_eq!(dup.ev_neto_por_ciclo, None);
    // Lado desconocido.
    let mut e = g.entrada(ModoPosicion::OneWay);
    e.fills[2].lado = None;
    assert!(tiene(&certificar_ciclos(&e), CausaCiclosNd::LadoOCantidadDesconocida));
    // Paginacion incompleta.
    let mut e = g.entrada(ModoPosicion::OneWay);
    e.cobertura_fills_completa = false;
    let pag = certificar_ciclos(&e);
    assert!(tiene(&pag, CausaCiclosNd::CoberturaFillsIncompleta));
    assert_eq!(pag.estado, ESTADO_ND);
}

#[test]
fn demo_ciclos_09_ev_firmado_y_atribucion() {
    // +0,50 - 0,40 + 0,05 + 0,20 - 0,10 = +0,25 (espec. §2.1).
    let caso = |rp_cierre: Result<ImporteVenue, CausaImporteNd>, fila_pnl: &str| {
        let mut g = Gen::nuevo();
        g.fill_con("BTCUSDT", Lado::Compra, Pierna::Both, "1", imp("0"), None, Some("-0.40"), CausaSalida::Normal);
        let t_abre = g.t;
        g.caja_con("BTCUSDT", TipoCaja::Funding, None, t_abre + 100, "USDT", imp("0.20"), None);
        g.caja_con("BTCUSDT", TipoCaja::Funding, None, t_abre + 200, "USDT", imp("-0.10"), None);
        g.fill_con("BTCUSDT", Lado::Venta, Pierna::Both, "1", rp_cierre, Some(fila_pnl), Some("0.05"), CausaSalida::Normal);
        g
    };
    let g = caso(imp("0.50"), "0.50");
    let cert = certificar_ciclos(&g.entrada(ModoPosicion::OneWay));
    assert_eq!(cert.ciclos.len(), 1);
    let c = &cert.ciclos[0];
    assert_eq!((c.bruto_e8, c.comision_e8, c.funding_e8, c.cargos_e8), (e8("0.50"), e8("-0.35"), e8("0.10"), 0), "rp e ingreso, una sola vez");
    assert_eq!(c.neto_e8, Some(e8("0.25")));
    assert_eq!((cert.ganadores, cert.ev_neto_por_ciclo), (1, Some(0.25)));
    // rp ausente o ilegible: N/D, nunca bruto cero.
    for causa in [CausaImporteNd::Ausente, CausaImporteNd::Formato] {
        let nd = certificar_ciclos(&caso(Err(causa), "0.50").entrada(ModoPosicion::OneWay));
        assert!(tiene(&nd, CausaCiclosNd::RpAusenteOIlegible), "{causa:?}");
        assert_eq!(nd.ev_neto_por_ciclo, None);
    }
    // Testigos contradictorios (rp 0,50 frente a REALIZED_PNL 0,49).
    let nd = certificar_ciclos(&caso(imp("0.50"), "0.49").entrada(ModoPosicion::OneWay));
    assert!(tiene(&nd, CausaCiclosNd::TestigosPnlContradictorios));
    // Comision cero solo con evidencia explicita: sin fila COMMISSION, N/D.
    let mut g = episodios(1, "1");
    g.caja.retain(|c| !(c.tipo == TipoCaja::Comision && c.trade_id == Some(1)));
    let sin_com = certificar_ciclos(&g.entrada(ModoPosicion::OneWay));
    assert!(tiene(&sin_com, CausaCiclosNd::ComisionSinEvidencia));
    assert_eq!(sin_com.n_ciclos_validos, 0);
    // Funding cero solo con cobertura comprobada: sin ella y sin filas, N/D.
    let mut e = episodios(1, "1").entrada(ModoPosicion::OneWay);
    e.cobertura_funding_completa = false;
    let sin_fund = certificar_ciclos(&e);
    assert!(tiene(&sin_fund, CausaCiclosNd::FundingSinCobertura));
    assert_eq!(sin_fund.n_ciclos_validos, 0);
    // Funding en el borde (mismo instante que el fill que cierra): no se adjudica.
    let mut g = episodios(1, "1");
    let t_cierre = g.fills[1].tiempo_ms;
    g.caja_con("BTCUSDT", TipoCaja::Funding, None, t_cierre, "USDT", imp("-0.01"), None);
    assert!(tiene(&certificar_ciclos(&g.entrada(ModoPosicion::OneWay)), CausaCiclosNd::FundingAmbiguo));
    // Conversion no demostrada (comision en BNB): N/D.
    let mut g = episodios(1, "1");
    g.caja.iter_mut().filter(|c| c.tipo == TipoCaja::Comision).for_each(|c| c.asset = "BNB".into());
    assert!(tiene(&certificar_ciclos(&g.entrada(ModoPosicion::OneWay)), CausaCiclosNd::MonedaNoUsdt));
    // Ganancia bruta con perdida neta: entra al conteo y no es ganador.
    let mut g = Gen::nuevo();
    g.fill_con("BTCUSDT", Lado::Compra, Pierna::Both, "1", imp("0"), None, Some("-0.15"), CausaSalida::Normal);
    g.fill_con("BTCUSDT", Lado::Venta, Pierna::Both, "1", imp("0.10"), Some("0.10"), Some("-0.05"), CausaSalida::Normal);
    let perd = certificar_ciclos(&g.entrada(ModoPosicion::OneWay));
    assert_eq!((perd.n_ciclos_validos, perd.ganadores, perd.neto_total_e8), (1, 0, e8("-0.10")));
}

#[test]
fn demo_ciclos_10_recibo_y_reinicio() {
    let valido = certificar_ciclos(&episodios(30, "1").entrada(ModoPosicion::OneWay));
    let d = Dir::nuevo("c10");
    let r = d.cert();
    assert_eq!(cargar_evidencia_ciclos_v2(&r, Some(&esperado())), Err(MotivoSinEvidencia::Ausente));
    assert_eq!(escribir_certificado_v2(&r, &valido), Ok(()));
    assert_eq!(cargar_evidencia_ciclos_v2(&r, Some(&esperado())).map(|e| e.cierres), Ok(30), "control: el v2 valido autoriza");
    assert_eq!(cargar_evidencia_ciclos_v2(&r, None), Err(MotivoSinEvidencia::AmbitoNoAcreditado));
    let con = |et: &str, f: &dyn Fn(&mut serde_json::Value)| {
        let mut v = serde_json::to_value(&valido).unwrap_or_default();
        f(&mut v);
        let d = Dir::nuevo(et);
        std::fs::write(d.cert(), v.to_string()).unwrap_or_else(|e| panic!("{e}"));
        cargar_evidencia_ciclos_v2(&d.cert(), Some(&esperado()))
    };
    assert_eq!(con("v3", &|v| v["version"] = 3.into()), Err(MotivoSinEvidencia::VersionDesconocida));
    assert_eq!(con("ndc", &|v| v["n_ciclos_validos"] = 31.into()), Err(MotivoSinEvidencia::Inconsistente));
    assert_eq!(con("hash", &|v| v["hash_ciclos"] = "".into()), Err(MotivoSinEvidencia::SinRespaldo));
    assert_eq!(con("est", &|v| v["estado"] = ESTADO_ND.into()), Err(MotivoSinEvidencia::Incompleto));
    assert_eq!(con("bruto", &|v| v["ciclos"][0]["neto_e8"] = 900_000_000i64.into()), Err(MotivoSinEvidencia::Inconsistente));
    let d2 = Dir::nuevo("c10x");
    std::fs::write(d2.cert(), "{").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(cargar_evidencia_ciclos_v2(&d2.cert(), Some(&esperado())), Err(MotivoSinEvidencia::Corrupto));
    // Otro entorno (Mainnet o paper): el productor los certifica igual, la puerta no.
    for entorno in ["mainnet", "paper"] {
        let mut g = episodios(30, "1");
        g.entorno = entorno.into();
        for f in g.fills.iter_mut() {
            f.libro.entorno = entorno.into();
        }
        for c in g.caja.iter_mut() {
            c.libro.entorno = entorno.into();
        }
        let cert = certificar_ciclos(&g.entrada(ModoPosicion::OneWay));
        assert_eq!(cert.n_ciclos_validos, 30, "{entorno}");
        let (ev, fase) = por_la_puerta(entorno, &cert);
        assert_eq!(ev, Err(MotivoSinEvidencia::OtroEntorno), "{entorno}");
        assert_eq!(fase, SystemPhase::DemoVerify);
    }
    // Cuenta, configuracion, binario o ventana discordantes.
    let mut otros = Vec::new();
    for campo in ["cuenta", "sha_config", "sha_binario", "sesion", "ventana"] {
        let mut esp = esperado();
        match campo {
            "cuenta" => esp.ambito.cuenta = "otra".into(),
            "sha_config" => esp.ambito.sha_config = "d".repeat(64),
            "sha_binario" => esp.ambito.sha_binario = "e".repeat(64),
            "sesion" => esp.ambito.sesion = "sesion-2".into(),
            _ => esp.ventana_hasta_ms = T1 + 1,
        }
        otros.push((campo, cargar_evidencia_ciclos_v2(&r, Some(&esp))));
    }
    assert!(otros.iter().all(|(_, x)| *x == Err(MotivoSinEvidencia::AmbitoDiscordante)), "{otros:?}");
    // Revocacion: persiste y sobrevive al reinicio (carga posterior desde disco).
    assert_eq!(revocar_certificado_v2(&r, "BREAKER_PROD_DEMO"), Ok(()));
    assert_eq!(cargar_evidencia_ciclos_v2(&r, Some(&esperado())), Err(MotivoSinEvidencia::Revocado));
    assert_eq!(fase_con(cargar_evidencia_ciclos_v2(&r, Some(&esperado())).ok(), true), SystemPhase::DemoVerify);
    // Reescribir el MISMO recibo (mismo hash) no lo rehabilita: la marca lo retiene.
    assert_eq!(escribir_certificado_v2(&r, &valido), Ok(()));
    assert_eq!(cargar_evidencia_ciclos_v2(&r, Some(&esperado())), Err(MotivoSinEvidencia::Revocado));
    // Una re-certificacion (otro hash) si vale.
    let nuevo = certificar_ciclos(&episodios(31, "1").entrada(ModoPosicion::OneWay));
    assert_eq!(escribir_certificado_v2(&r, &nuevo), Ok(()));
    assert_eq!(cargar_evidencia_ciclos_v2(&r, Some(&esperado())).map(|e| e.cierres), Ok(31));
    // Revocar un recibo ilegible retiene cualquier recibo posterior hasta que el operador decida.
    std::fs::write(&r, "{").unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(revocar_certificado_v2(&r, "BREAKER_PROD_DEMO"), Ok(()));
    assert_eq!(escribir_certificado_v2(&r, &nuevo), Ok(()));
    assert_eq!(cargar_evidencia_ciclos_v2(&r, Some(&esperado())), Err(MotivoSinEvidencia::Revocado));
}

// ---------------------------------------------------------------- regresiones (sin cambios)

fn fuente_bin() -> String {
    include_str!("../src/bin/god_engine.rs").replace("\r\n", "\n")
}

/// El freno de Produccion sigue en FILAS, con su minimo de 10; con 10 filas de perdida en 2 ciclos
/// dispara y baja a DemoVerify, y la revocacion del recibo v2 sobrevive a una carga posterior.
#[test]
fn demo_ciclos_r1_freno_por_filas_minimo_10_y_revocacion_persistente() {
    assert_eq!(PhaseOrchestrator::MIN_CIERRES_BREAKER_PROD, 10);
    assert_eq!(MIN_CIERRES_PARA_PROD, 30);
    let mut g = Gen::nuevo();
    for _ in 0..2 {
        g.ow("BTCUSDT", Lado::Compra, "5", "0");
        for _ in 0..5 {
            g.ow("BTCUSDT", Lado::Venta, "1", "-1");
        }
    }
    let cert = certificar_ciclos(&g.entrada(ModoPosicion::OneWay));
    assert_eq!((cert.n_ciclos_validos, cert.filas_realized_pnl), (2, 10));
    let filas: Vec<(u64, f64)> = g.caja.iter().filter(|c| c.tipo == TipoCaja::RealizedPnl).map(|c| (c.tiempo_ms, -1.0)).collect();
    let (bruto, flujo, cierres, _) = quantum_engine::host_oms::agregado_income_de_cuenta([(filas.as_slice(), 0.0, 0.0)]);
    assert_eq!(cierres, 10, "el freno recibe filas, no ciclos");
    let d = Dir::nuevo("r1");
    let valido = certificar_ciclos(&episodios(30, "1").entrada(ModoPosicion::OneWay));
    assert_eq!(escribir_certificado_v2(&d.cert(), &valido), Ok(()));
    let prod = || {
        let mut p = PhaseOrchestrator::new_with_oms(1, false, Arc::new(AtomicBool::new(true)), Arc::new(AtomicBool::new(true)));
        p.set_evidencia_demo(cargar_evidencia_ciclos_v2(&d.cert(), Some(&esperado())).ok());
        for _ in 0..5 {
            p.on_tick();
        }
        p
    };
    let mut p9 = prod();
    assert_eq!(p9.current_phase, SystemPhase::ProductionMainnet);
    assert!(!quantum_engine::host_oms::breaker_de_ingreso(&mut p9, 9, -9.0, 0.0), "con 9 filas no decide");
    let mut p = prod();
    assert!(quantum_engine::host_oms::breaker_de_ingreso(&mut p, cierres, bruto, flujo), "dispara en la decima fila");
    assert_eq!(p.current_phase, SystemPhase::DemoVerify);
    assert!(p.evidencia_demo.is_none(), "retira la autorizacion en memoria");
    // El host revoca el recibo v2 en disco al disparar (ver r4); tras el reinicio sigue cerrado.
    assert_eq!(revocar_certificado_v2(&d.cert(), "BREAKER_PROD_DEMO"), Ok(()));
    let reinicio = prod();
    assert_eq!(reinicio.current_phase, SystemPhase::DemoVerify);
    assert!(reinicio.evidencia_demo.is_none());
}

/// Profiler y panel siguen en filas: con filas y 0 ciclos acreditados el agregado y la seleccion
/// de fuente del panel no cambian.
#[test]
fn demo_ciclos_r2_profiler_y_panel_siguen_en_filas() {
    let g = diez_episodios();
    let mut e = g.entrada(ModoPosicion::OneWay);
    e.cobertura_funding_completa = false;
    let cert = certificar_ciclos(&e);
    assert_eq!((cert.n_ciclos_validos, cert.filas_realized_pnl), (0, 30));
    let filas: Vec<(u64, f64)> = g.caja.iter().filter(|c| c.tipo == TipoCaja::RealizedPnl).map(|c| (c.tiempo_ms, 1.0)).collect();
    assert_eq!(quantum_engine::host_oms::agregado_income_de_cuenta([(filas.as_slice(), 0.0, 0.0)]), (30.0, 0.0, 30, 30));
    let src = fuente_bin();
    assert!(src.contains("telemetry_server::profiler::record_exchange_income(\n                            x_gross, x_fees, x_closes, x_wins,"));
    assert!(src.contains("let (gross_display, net_display, wr_display) = if x_closes > 0"));
}

/// Freno estadistico por simbolo: misma agrupacion temporal (1 s), mismos datos, misma decision.
#[test]
fn demo_ciclos_r3_freno_por_simbolo_intacto() {
    let src = fuente_bin();
    for ancla in [
        "let cierres = agrupar_cierres(a.registros, 1_000);",
        "let net = bruto + a.funding - a.fees;",
        "let decision = decision_fee_breaker(\n                            &cierres,\n                            a.funding,\n                            a.fees,",
        "\"REALIZED_PNL\" => a.registros.push((e.time, e.income)),",
    ] {
        assert!(src.contains(ancla), "ancla del freno por simbolo: {ancla}");
    }
}

/// Cableado: el arranque de Produccion solo acepta el recibo v2 (el legado ya no autoriza) y el
/// disparo del freno revoca el recibo v2 en disco.
#[test]
fn demo_ciclos_r4_cableado_de_la_puerta() {
    let src = fuente_bin();
    assert!(src.contains("quantum_engine::ciclos_completos::cargar_evidencia_ciclos_v2("), "el arranque lee el recibo v2");
    assert!(src.contains("quantum_engine::ciclos_completos::AmbitoEsperado::del_arranque()"));
    assert!(!src.contains("serde_json::from_str::<god_engine_core::orchestrator::EvidenciaDemo>(&txt)"), "el legado ya no autoriza");
    let disparo = src.find("if disparo {").unwrap_or(usize::MAX);
    let revoca = src.find("quantum_engine::ciclos_completos::revocar_certificado_v2(").unwrap_or(0);
    assert!(disparo < revoca && revoca - disparo < 1_200, "el disparo del freno revoca el recibo v2 en disco");
    assert!(src.contains("registrar_breaker_ingreso(&mut *orch, x_closes, x_gross, x_fees)"), "el freno sigue en filas");
}
