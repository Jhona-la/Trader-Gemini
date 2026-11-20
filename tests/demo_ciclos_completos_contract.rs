//! CICLOS-2 (pedido 20261118-0535-cicloscompletos; espec. `ciclos_completos_v2.md` §3, matriz 01-10 y
//! regresiones; CICLOS-4: casos 11 y 12 de QODER-APP C8, 07b y r5). El productor real (`ciclos_completos::certificar_ciclos`) recibe fills, cashflows,
//! fotos de borde y cobertura inyectados en memoria; su recibo pasa por el escritor y el lector
//! reales (`escribir_certificado_v2` / `cargar_evidencia_ciclos_v2`, en un directorio temporal) y de
//! ahi a la puerta real (`PhaseOrchestrator`). Ningun contador se inyecta ya calculado.
//! Importes del venue como decimal exacto con signo sobre su texto (CICLOS-3, sin escala elegida):
//! la cuadratura es por igualdad exacta, sin tolerancia. Las cantidades siguen en 1e-8 (escala de
//! cantidad).
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use execution_engine::importe_venue::{importe_de_texto, CausaImporteNd, DecimalExacto, ImporteVenue};
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
/// Saldo USDT de borde con el texto dado.
fn saldo(t: &str) -> Option<SaldoVenue> {
    Some(SaldoVenue { asset: "USDT".into(), importe: imp(t) })
}
const SALDO_T0: &str = "1000";
/// Importe exacto del fixture.
fn dec(t: &str) -> DecimalExacto {
    match importe_de_texto(t) {
        Ok(i) => i.valor,
        Err(c) => panic!("fixture {t}: {c:?}"),
    }
}
/// Cantidad del fixture a la escala de cantidad (1e-8).
fn e8(t: &str) -> i64 {
    match dec(t).en_unidades(8).and_then(|x| i64::try_from(x).ok()) {
        Some(x) => x,
        None => panic!("cantidad {t} fuera de la escala de 1e-8"),
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
            saldo_t0: saldo(SALDO_T0),
            saldo_t1: saldo(&self.saldo_final().texto()),
        }
    }
    /// Saldo en T1 que cuadra exacto con las filas de caja legibles del fixture (CICLOS-4, C3).
    fn saldo_final(&self) -> DecimalExacto {
        let mut s = dec(SALDO_T0);
        for c in &self.caja {
            if let Ok(i) = &c.importe {
                s = s.checked_add(i.valor).unwrap_or_else(|| panic!("saldo del fixture desborda"));
            }
        }
        s
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
    assert_eq!(cert.neto_total, dec("30"));
    assert!(cert.ciclos.iter().all(|c| c.comision == DecimalExacto::CERO && c.funding == DecimalExacto::CERO && c.neto == Some(dec("3"))));
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
    assert_eq!(cerrado.neto_total, dec("4"));
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
    assert_eq!((dos.n_ciclos_validos, dos.neto_total), (2, dec("3")));
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
        assert_eq!(cert.ciclos[0].cargos, dec("-0.5"));
        assert_eq!(cert.neto_total, dec("-3.5"), "{causa:?}");
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
    assert_eq!(ok.ciclos.iter().map(|x| x.neto).collect::<Vec<_>>(), vec![Some(dec("0.38")), Some(dec("0.09"))]);
    assert_eq!(ok.neto_total, dec("0.47"), "sin residuo perdido: -0,03 entero repartido");
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
    assert_eq!((replay.n_ciclos_validos, replay.neto_total, &replay.hash_ciclos), (10, dec("30"), &base.hash_ciclos));
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
    assert_eq!((c.bruto, c.comision, c.funding, c.cargos), (dec("0.50"), dec("-0.35"), dec("0.10"), DecimalExacto::CERO), "rp e ingreso, una sola vez");
    assert_eq!(c.neto, Some(dec("0.25")));
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
    assert_eq!((perd.n_ciclos_validos, perd.ganadores, perd.neto_total), (1, 0, dec("-0.10")));
    // CICLOS-3 (sin escala elegida): una comision de 9 cifras decimales es exacta, no N/D ni
    // redondeada, y su signo de pago se conserva.
    let mut g = Gen::nuevo();
    g.fill_con("BTCUSDT", Lado::Compra, Pierna::Both, "1", imp("0"), None, Some("0"), CausaSalida::Normal);
    g.fill_con("BTCUSDT", Lado::Venta, Pierna::Both, "1", imp("0.25"), Some("0.25"), Some("-0.000000001"), CausaSalida::Normal);
    let fino = certificar_ciclos(&g.entrada(ModoPosicion::OneWay));
    assert_eq!((fino.n_ciclos_validos, fino.estado.as_str()), (1, ESTADO_VALIDO), "{:?} {:?}", fino.causas_nd, fino.ciclos);
    assert_eq!(fino.ciclos[0].comision, dec("-0.000000001"));
    assert_eq!(fino.neto_total, dec("0.249999999"));
    assert_ne!(fino.neto_total, dec("0.25"), "sin tolerancia: una unidad en la ultima cifra cuenta");
    let s = serde_json::to_value(&fino).unwrap_or_default();
    assert_eq!(s["neto_total"], serde_json::json!("0.249999999"), "el recibo lleva el texto exacto");
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
    assert_eq!(con("bruto", &|v| v["ciclos"][0]["neto"] = "9".into()), Err(MotivoSinEvidencia::Inconsistente));
    assert_eq!(con("neto_f", &|v| v["neto_total"] = 30.into()), Err(MotivoSinEvidencia::Corrupto), "un numero JSON no es un importe");
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

// ---------------------------------------------------------------- CICLOS-4 (adenda y N1 de QODER-APP)

/// C1/C8 caso 11 (denominador mixto, bloqueante): 30 ciclos acreditados con EV > 0 y UN ciclo
/// cerrado estructuralmente con neto N/D. El N/D no sale del denominador: bloquea el recibo.
#[test]
fn demo_ciclos_11_denominador_mixto_bloquea_el_recibo() {
    // Control: los 30 solos abren (la puerta muerde).
    let (ev30, f30) = por_la_puerta("c11c", &certificar_ciclos(&episodios(30, "1").entrada(ModoPosicion::OneWay)));
    assert_eq!(ev30.map(|e| e.cierres), Ok(30));
    assert_eq!(f30, SystemPhase::ProductionMainnet, "control: 30 acreditados con EV > 0 admiten");
    // (a) 31 estructurales, 30 acreditados, 1 N/D (rp ausente en su cierre).
    let mut g = episodios(30, "1");
    g.ow("BTCUSDT", Lado::Compra, "1", "0");
    g.ow("BTCUSDT", Lado::Venta, "1", "1");
    let ultimo = g.fills.len() - 1;
    g.fills[ultimo].rp = Err(CausaImporteNd::Ausente);
    let mixto = certificar_ciclos(&g.entrada(ModoPosicion::OneWay));
    assert_eq!((mixto.ciclos_estructurales, mixto.n_ciclos_validos, mixto.ciclos_nd), (31, 30, 1));
    assert!(tiene(&mixto, CausaCiclosNd::CicloNdEnDenominador), "{:?}", mixto.causas_nd);
    assert_eq!((mixto.estado.as_str(), mixto.ev_neto_por_ciclo), (ESTADO_ND, None), "el N/D bloquea, no se excluye");
    assert_eq!(mixto.neto_total, dec("30"), "la media de los observables es descripcion");
    let (ev, fase) = por_la_puerta("c11a", &mixto);
    assert_eq!(ev, Err(MotivoSinEvidencia::Incompleto));
    assert_eq!(fase, SystemPhase::DemoVerify);
    // (b) La reversion de un solo fill que C+D manda a N/D (dos ciclos estructurales N/D).
    let mut r = episodios(30, "1");
    r.ow("BTCUSDT", Lado::Compra, "2", "0");
    r.fill_con("BTCUSDT", Lado::Venta, Pierna::Both, "3", imp("0.4"), Some("0.4"), Some("-0.03"), CausaSalida::Normal);

    // Caso 11 de QODER: el tramo que abre con la reversion sigue abierto en T1.
    let mut entrada_reversion = r.entrada(ModoPosicion::OneWay);
    entrada_reversion.foto_t1 = Some(BTreeMap::from([(
        "BTCUSDT".into(),
        Foto { both_e8: e8("-1"), ..Foto::default() },
    )]));
    let reversion_abierta = certificar_ciclos(&entrada_reversion);
    assert_eq!(
        (reversion_abierta.ciclos_estructurales, reversion_abierta.n_ciclos_validos, reversion_abierta.ciclos_nd),
        (31, 30, 1)
    );
    assert_eq!((reversion_abierta.arrastres, reversion_abierta.abiertos), (0, 1));
    assert!(!tiene(&reversion_abierta, CausaCiclosNd::DescuadreFotoT1));
    assert!(tiene(&reversion_abierta, CausaCiclosNd::ReversionSinImportesExplicitos));
    assert!(tiene(&reversion_abierta, CausaCiclosNd::CicloNdEnDenominador));
    assert_eq!(reversion_abierta.neto_total, dec("30"));
    assert_eq!((reversion_abierta.estado.as_str(), reversion_abierta.ev_neto_por_ciclo), (ESTADO_ND, None));
    assert_eq!(
        por_la_puerta("c11g", &reversion_abierta),
        (Err(MotivoSinEvidencia::Incompleto), SystemPhase::DemoVerify)
    );

    r.fill_con("BTCUSDT", Lado::Compra, Pierna::Both, "1", imp("0.1"), Some("0.1"), Some("0"), CausaSalida::Normal);
    let rev = certificar_ciclos(&r.entrada(ModoPosicion::OneWay));
    assert_eq!((rev.ciclos_estructurales, rev.n_ciclos_validos, rev.ciclos_nd), (32, 30, 2));
    assert!(tiene(&rev, CausaCiclosNd::ReversionSinImportesExplicitos));
    assert!(tiene(&rev, CausaCiclosNd::CicloNdEnDenominador));
    assert_eq!((rev.estado.as_str(), rev.ev_neto_por_ciclo), (ESTADO_ND, None));
    assert_eq!(por_la_puerta("c11b", &rev).1, SystemPhase::DemoVerify);
    // (c) Un recibo retocado para "excluir" el N/D no autoriza.
    let con = |et: &str, f: &dyn Fn(&mut serde_json::Value)| {
        let mut v = serde_json::to_value(&mixto).unwrap_or_default();
        f(&mut v);
        let d = Dir::nuevo(et);
        std::fs::write(d.cert(), v.to_string()).unwrap_or_else(|e| panic!("{e}"));
        cargar_evidencia_ciclos_v2(&d.cert(), Some(&esperado()))
    };
    let valido = |v: &mut serde_json::Value| {
        v["estado"] = ESTADO_VALIDO.into();
        v["causas_nd"] = serde_json::json!([]);
        v["ev_neto_por_ciclo"] = 1.0.into();
    };
    assert_eq!(con("c11d", &|v| valido(v)), Err(MotivoSinEvidencia::Inconsistente), "ciclos_nd > 0");
    assert_eq!(
        con("c11e", &|v| {
            valido(v);
            v["ciclos_nd"] = 0.into();
        }),
        Err(MotivoSinEvidencia::Inconsistente),
        "30 validos de 31 estructurales"
    );
    assert_eq!(
        con("c11f", &|v| {
            valido(v);
            v["ciclos_nd"] = 0.into();
            v["ciclos_estructurales"] = 30.into();
            if let Some(a) = v["ciclos"].as_array_mut() {
                a.pop();
            }
        }),
        Err(MotivoSinEvidencia::Inconsistente),
        "quitar el ciclo N/D rompe el respaldo"
    );
}

/// C3/C8 caso 12 (cuadratura exacta): ventana USDT sin FX con delta de saldo EXACTAMENTE igual al
/// neto mas los flujos externos => pasa; una unidad en la ultima cifra => N/D y puerta cerrada.
#[test]
fn demo_ciclos_12_cuadratura_exacta_sin_tolerancia() {
    let g = episodios(30, "1");
    let base = g.entrada(ModoPosicion::OneWay);
    let ok = certificar_ciclos(&base);
    assert_eq!((ok.estado.as_str(), ok.saldo_t0, ok.saldo_t1), (ESTADO_VALIDO, Some(dec("1000")), Some(dec("1030"))), "{:?}", ok.causas_nd);
    assert_eq!(por_la_puerta("c12a", &ok).1, SystemPhase::ProductionMainnet, "control: cuadra exacto y abre");
    let con_saldo_t1 = |t: &str| {
        let mut e = base.clone();
        e.saldo_t1 = saldo(t);
        certificar_ciclos(&e)
    };
    for t in ["1030.00000001", "1029.99999999", "1030.000000000001", "1030.1"] {
        let c = con_saldo_t1(t);
        assert!(tiene(&c, CausaCiclosNd::CuadraturaDescuadrada), "{t}: {:?}", c.causas_nd);
        assert_eq!((c.estado.as_str(), c.ev_neto_por_ciclo), (ESTADO_ND, None), "{t}");
        let (ev, fase) = por_la_puerta("c12b", &c);
        assert_eq!((ev, fase), (Err(MotivoSinEvidencia::Incompleto), SystemPhase::DemoVerify), "{t}: sin tolerancia");
    }
    assert_eq!(con_saldo_t1("1030.000").estado, ESTADO_VALIDO, "mismo valor con otro texto cuadra");
    // Saldo ausente o ilegible: N/D.
    for s in [None, Some(SaldoVenue { asset: "USDT".into(), importe: Err(CausaImporteNd::NumeroJsonSinTexto) })] {
        let mut e = base.clone();
        e.saldo_t0 = s;
        assert!(tiene(&certificar_ciclos(&e), CausaCiclosNd::SaldoBordeAusente));
    }
    // Fuera de la cohorte USDT unico sin FX: N/D, nunca "igualdad con tolerancia".
    let mut e = base.clone();
    e.saldo_t1 = Some(SaldoVenue { asset: "BNB".into(), importe: imp("1030") });
    assert!(tiene(&certificar_ciclos(&e), CausaCiclosNd::CuadraturaFueraDeCohorte));
    let mut e = base.clone();
    if let Some(c) = e.caja.first_mut() {
        c.asset = String::new();
    }
    let sin_activo = certificar_ciclos(&e);
    assert!(tiene(&sin_activo, CausaCiclosNd::MonedaNoUsdt), "QODER-APP C7: activo ausente no es USDT");
    assert!(tiene(&sin_activo, CausaCiclosNd::CuadraturaFueraDeCohorte));
    // Flujo externo: cuenta en la cuadratura y no en los ciclos.
    let mut t = episodios(30, "1");
    t.caja_con("", TipoCaja::Externo, None, T0 + 500, "USDT", imp("100"), None);
    let tr = certificar_ciclos(&t.entrada(ModoPosicion::OneWay));
    assert_eq!((tr.estado.as_str(), tr.n_ciclos_validos, tr.neto_total, tr.externos), (ESTADO_VALIDO, 30, dec("30"), dec("100")), "{:?}", tr.causas_nd);
    let mut e = t.entrada(ModoPosicion::OneWay);
    e.caja.retain(|c| c.tipo != TipoCaja::Externo);
    assert!(tiene(&certificar_ciclos(&e), CausaCiclosNd::CuadraturaDescuadrada), "un deposito no declarado no cuadra");
    // Un recibo con el saldo retocado no autoriza.
    let mut v = serde_json::to_value(&ok).unwrap_or_default();
    v["saldo_t1"] = "1030.00000001".into();
    let d = Dir::nuevo("c12c");
    std::fs::write(d.cert(), v.to_string()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(cargar_evidencia_ciclos_v2(&d.cert(), Some(&esperado())), Err(MotivoSinEvidencia::Inconsistente));
}

#[test]
fn demo_ciclos_12_desborde_publica_causa_y_cierra_puerta() {
    let maximo = i128::MAX.to_string();
    let mut g = episodios(30, "1");
    let mut diferencia = g.entrada(ModoPosicion::OneWay);
    diferencia.saldo_t0 = saldo("-1");
    diferencia.saldo_t1 = saldo(&maximo);

    let mut suma = g.entrada(ModoPosicion::OneWay);
    g.caja_con("", TipoCaja::Externo, None, T0 + 500, "USDT", imp(&maximo), None);
    suma.caja = g.caja;

    let delta = certificar_ciclos(&diferencia);
    let rhs = certificar_ciclos(&suma);
    let causa = [format!("{:?}", CausaCiclosNd::Desbordamiento)];
    assert_eq!(
        [delta.causas_nd.as_slice(), rhs.causas_nd.as_slice()],
        [causa.as_slice(), causa.as_slice()],
        "desbordes en delta de saldo y en neto más flujos"
    );
    for (etiqueta, cert) in [("c12delta", &delta), ("c12suma", &rhs)] {
        assert_eq!((cert.ciclos_estructurales, cert.n_ciclos_validos, cert.ciclos_nd), (30, 30, 0));
        assert_eq!((cert.estado.as_str(), cert.ev_neto_por_ciclo), (ESTADO_ND, None));
        assert_eq!(
            por_la_puerta(etiqueta, cert),
            (Err(MotivoSinEvidencia::Incompleto), SystemPhase::DemoVerify)
        );
    }
}

/// C2: el adaptador conserva la granularidad POR FILA (tranId), no la suma por tradeId; con un
/// solo fill de reversion, dos filas sin tramo explicito siguen sin satisfacer C.
#[test]
fn demo_ciclos_07b_granularidad_por_fila() {
    let mut g = Gen::nuevo();
    g.ow("BTCUSDT", Lado::Compra, "1", "0");
    let id = g.fill_con("BTCUSDT", Lado::Venta, Pierna::Both, "1", imp("1"), Some("1"), Some("-0.02"), CausaSalida::Normal);
    let t = g.t;
    g.caja_con("BTCUSDT", TipoCaja::Comision, Some(id), t, "USDT", imp("-0.01"), None);
    let dos = certificar_ciclos(&g.entrada(ModoPosicion::OneWay));
    assert_eq!((dos.n_ciclos_validos, dos.estado.as_str()), (1, ESTADO_VALIDO), "{:?}", dos.causas_nd);
    assert_eq!(dos.ciclos[0].comision, dec("-0.03"), "dos filas del mismo tradeId, cada una por su tranId");
    // Reversion con dos filas de comision sin tramo: C no se satisface => D (N/D).
    let mut r = Gen::nuevo();
    r.ow("BTCUSDT", Lado::Compra, "2", "0");
    let id = r.fill_con("BTCUSDT", Lado::Venta, Pierna::Both, "3", imp("0.4"), Some("0.4"), Some("-0.02"), CausaSalida::Normal);
    let t = r.t;
    r.caja_con("BTCUSDT", TipoCaja::Comision, Some(id), t, "USDT", imp("-0.01"), None);
    r.fill_con("BTCUSDT", Lado::Compra, Pierna::Both, "1", imp("0.1"), Some("0.1"), Some("0"), CausaSalida::Normal);
    let nd = certificar_ciclos(&r.entrada(ModoPosicion::OneWay));
    assert!(tiene(&nd, CausaCiclosNd::ReversionSinImportesExplicitos));
    assert_eq!((nd.n_ciclos_validos, nd.ev_neto_por_ciclo), (0, None));
}

/// C2/C5/C6/C7: el codigo del certificado no reutiliza tolerancias ni lectores prohibidos.
#[test]
fn demo_ciclos_r5_sin_tolerancias_ni_lectores_prohibidos() {
    let src = include_str!("../src/ciclos_completos.rs").replace("\r\n", "\n");
    let codigo: String = src.lines().filter(|l| !l.trim_start().starts_with("//")).collect::<Vec<_>>().join("\n");
    assert!(codigo.contains("fn cuadra("), "control: el filtro deja el codigo");
    for prohibido in ["mismo_saldo", "state_validator", "1e-8", "1e-9", "abs() <=", "trade_ledger", "parse_income_jsonl", "decimal_e8", "f64::EPSILON"] {
        assert!(!codigo.contains(prohibido), "prohibido en el certificado: {prohibido}");
    }
}
