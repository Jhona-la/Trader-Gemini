//! LG-27 (Línea G; B4 de la revisión del PR #62 por la Línea C): el detector
//! e-CUSUM de pérdida de ventaja de cada moneda (LG-22, cableado GLM 121)
//! sobrevive al reinicio del host.
//!
//! RED (antes de LG-27). `MonedaGuardada` no guardaba el detector: cada
//! reinicio lo ponía a cero. Con la fuente del fixture (E[R] ≈ −0,19 R) y
//! ARL 20, el detector sin reinicios alarma en el cierre 567; reiniciado cada
//! 60 cierres no alarma nunca. Una mutación que quita la restauración deja
//! el contrato 1 sin alarmas.
//! GREEN. Con el detector guardado y restaurado, alarma en los mismos cierres.
//!
//! Fixture. Dos símbolos (BTCUSDT y ETHUSDT); la fuente es ETH (coin 1), una
//! orden de dos barreras con R máxima 2,25 que gana 1 de cada 4 cierres, más
//! un cierre fuera de garantía (R 6, anula la fracción mayor) justo antes de
//! un reinicio y otro fuera de cota (R 3).

use god_engine_core::estado_aprendido::{cargar, guardar, validar, Carga, EstadoAprendido};
use god_engine_core::resultado_orden::{DetectorGuardado, DetectorPerdidaDeVentaja};
use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering::Relaxed;

const R_MAX: f64 = 2.25;
const ARL: f64 = 20.0;
/// El host reinicia cada `CADENCIA` cierres.
const CADENCIA: u64 = 60;
const CIERRES: u64 = 1_100;

fn registrar_simbolos() {
    quantum_arena::symbol_registry::update_registry(
        ["BTCUSDT", "ETHUSDT"]
            .iter()
            .map(|s| quantum_arena::symbol_registry::SymbolSpec {
                symbol: (*s).into(),
                step_size: 0.001,
                tick_size: 0.01,
                min_qty: 0.001,
                min_notional: 5.0,
                max_leverage: 20,
                maker_fee: 0.0002,
                taker_fee: 0.0004,
                is_shadow: false,
            })
            .collect(),
    );
}

fn nucleo() -> GodEngineCore {
    GodEngineCore::new(GlobalArena::build_in_own_stack(1_000.0))
}

/// R neta del cierre `i`.
fn r_de(i: u64) -> f64 {
    match i {
        59 => 6.0,  // fuera de garantía: 6 · 0,4 > 2,25 ⇒ M de φ = 0,4 a cero
        130 => 3.0, // fuera de cota, dentro de garantía
        _ if i % 4 == 0 => R_MAX,
        _ => -1.0,
    }
}

/// Lo que hace el núcleo en cada cierre (lib.rs, LG-22): observar, mirar la
/// alarma y, si salta, reiniciar. Devuelve si alarmó.
fn cerrar(det: &mut DetectorPerdidaDeVentaja, i: u64) -> bool {
    det.observar(r_de(i), R_MAX);
    let alarma = det.alarma(ARL);
    if alarma {
        det.reiniciar();
    }
    alarma
}

struct Temporal(std::path::PathBuf);

impl Temporal {
    fn nuevo(nombre: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("lg27_{}_{}", nombre, std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
    fn ruta(&self) -> std::path::PathBuf {
        self.0.join("estado_aprendido.json")
    }
}

impl Drop for Temporal {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn leer(ruta: &std::path::Path) -> EstadoAprendido {
    match cargar(ruta) {
        Carga::Leido(e) => e,
        otra => panic!("el estado recién guardado se lee: {otra:?}"),
    }
}

/// El host reinicia: guarda el estado del núcleo viejo, lo lee del disco y lo
/// restaura en un núcleo nuevo.
fn reiniciar_host(viejo: &GodEngineCore, ruta: &std::path::Path) -> GodEngineCore {
    guardar(ruta, &viejo.estado_aprendido()).unwrap();
    let e = leer(ruta);
    let mut nuevo = nucleo();
    assert!(nuevo.restaurar_estado_aprendido(&e) >= 2);
    nuevo
}

fn casi_igual(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

#[test]
fn lg27_el_detector_alarma_en_los_mismos_cierres_aunque_el_host_reinicie() {
    registrar_simbolos();
    let tmp = Temporal::nuevo("mismos_cierres");

    // Sin reinicios: el detector de referencia.
    let mut referencia = DetectorPerdidaDeVentaja::nuevo();
    let alarmas_referencia: Vec<u64> = (0..CIERRES).filter(|&i| cerrar(&mut referencia, i)).collect();
    assert!(
        alarmas_referencia.first().is_some_and(|&a| a > CADENCIA),
        "la primera alarma llega después del primer reinicio: {alarmas_referencia:?}"
    );

    // Como era antes de LG-27: el detector vuelve a cero en cada reinicio.
    let mut sin_guardar = DetectorPerdidaDeVentaja::nuevo();
    let alarmas_sin_guardar: Vec<u64> = (0..CIERRES)
        .filter(|&i| {
            if i > 0 && i % CADENCIA == 0 {
                sin_guardar = DetectorPerdidaDeVentaja::nuevo();
            }
            cerrar(&mut sin_guardar, i)
        })
        .collect();
    assert!(alarmas_sin_guardar.is_empty(), "{alarmas_sin_guardar:?}");

    // Con LG-27: el host reinicia cada CADENCIA cierres y justo después de
    // cada alarma (detector recién reiniciado), y el detector sigue.
    let mut nucleo_vivo = nucleo();
    let mut alarmas = Vec::new();
    let mut tras_alarma = false;
    let mut reinicios = 0;
    for i in 0..CIERRES {
        if i > 0 && (i % CADENCIA == 0 || tras_alarma) {
            nucleo_vivo = reiniciar_host(&nucleo_vivo, &tmp.ruta());
            reinicios += 1;
            let det = &nucleo_vivo.lg22_detectores[1];
            let reg = &nucleo_vivo.arena.registry;
            assert_eq!(reg.get_for_coin_or(1, "lg22_cierres", -1.0), det.cierres_totales as f64);
            assert!(casi_igual(reg.get_for_coin_or(1, "lg22_estadistico", -1.0), det.estadistico()));
            assert_eq!(reg.get_for_coin_or(1, "lg22_detector_descartado", -1.0), 0.0);
            if i == CADENCIA {
                // El cierre 59 anuló la fracción mayor: M = 0 sólo en el sufijo.
                let guardado = det.guardado();
                assert!(guardado.log_cusum[..3].iter().all(Option::is_some), "{guardado:?}");
                assert_eq!(guardado.log_cusum[3], None);
            }
        }
        tras_alarma = cerrar(&mut nucleo_vivo.lg22_detectores[1], i);
        if tras_alarma {
            alarmas.push(i);
        }
    }
    assert!(reinicios > CIERRES / CADENCIA, "también se reinició tras las alarmas");
    assert_eq!(alarmas, alarmas_referencia, "mismas alarmas que sin reinicios");
    let det = &nucleo_vivo.lg22_detectores[1];
    assert_eq!(det.cierres_totales, referencia.cierres_totales);
    assert_eq!(det.cierres_desde_reinicio, referencia.cierres_desde_reinicio);
    assert_eq!((det.fuera_de_cota, det.fuera_de_garantia), (2, 1));
    assert_eq!(
        (det.fuera_de_cota, det.fuera_de_garantia),
        (referencia.fuera_de_cota, referencia.fuera_de_garantia)
    );
    // serde_json sin `float_roundtrip` puede mover el último ulp de ln M.
    assert!(casi_igual(det.estadistico(), referencia.estadistico()));
    // BTC (coin 0) no tuvo cierres y sigue sin ellos.
    assert_eq!(nucleo_vivo.lg22_detectores[0].cierres_totales, 0);
}

/// Un estado anterior a LG-27 (sin `detector_lg22`, o con `null`) se sigue
/// leyendo y restaurando entero; sólo el detector arranca de cero.
#[test]
fn lg27_un_estado_anterior_sin_detector_se_restaura() {
    registrar_simbolos();
    let tmp = Temporal::nuevo("legado");
    let mut viejo = nucleo();
    viejo.arena.coins[1].metrics.trade_count.store(7, Relaxed);
    for i in 0..30 {
        cerrar(&mut viejo.lg22_detectores[1], i);
    }
    for quitar in [true, false] {
        let mut json = serde_json::to_value(viejo.estado_aprendido()).unwrap();
        for moneda in json["monedas"].as_array_mut().unwrap() {
            let objeto = moneda.as_object_mut().unwrap();
            if quitar {
                assert!(objeto.remove("detector_lg22").is_some());
            } else {
                objeto.insert("detector_lg22".into(), serde_json::Value::Null);
            }
        }
        std::fs::write(tmp.ruta(), serde_json::to_string(&json).unwrap()).unwrap();
        let e = leer(&tmp.ruta());
        let mut nuevo = nucleo();
        assert!(nuevo.restaurar_estado_aprendido(&e) >= 2);
        assert_eq!(nuevo.arena.coins[1].metrics.trade_count.load(Relaxed), 7, "lo demás se restaura");
        assert_eq!(nuevo.lg22_detectores[1].cierres_totales, 0);
        assert_eq!(nuevo.lg22_detectores[1].estadistico(), 0.0);
    }
}

/// Un detector guardado que `observar`/`reiniciar` no pueden producir no
/// bloquea el estado: se valida, se escribe y se restaura todo menos ese
/// detector, que empieza de cero y queda marcado como descartado.
#[test]
fn lg27_un_detector_inservible_se_descarta_sin_bloquear_el_estado() {
    registrar_simbolos();
    let tmp = Temporal::nuevo("inservible");
    let mut c = nucleo();
    c.arena.coins[1].metrics.trade_count.store(9, Relaxed);
    for i in 0..30 {
        cerrar(&mut c.lg22_detectores[1], i);
    }
    let bueno = c.estado_aprendido();
    let mut referencia = bueno.monedas[1].detector_lg22.clone().unwrap();
    assert!(DetectorPerdidaDeVentaja::desde_guardado(&referencia).is_some());
    referencia.fuera_de_cota = 3;
    referencia.fuera_de_garantia = 1;
    assert!(DetectorPerdidaDeVentaja::desde_guardado(&referencia).is_some(), "contadores coherentes");

    let casos: [(&str, fn(&mut DetectorGuardado)); 9] = [
        ("más cierres desde el reinicio que totales", |d| {
            d.cierres_desde_reinicio = d.cierres_totales + 1
        }),
        ("más fuera de cota que totales", |d| d.fuera_de_cota = d.cierres_totales + 1),
        ("más fuera de garantía que fuera de cota", |d| {
            d.fuera_de_cota = 0;
            d.fuera_de_garantia = 1;
        }),
        ("ln M no finito", |d| d.log_cusum[0] = Some(f64::NAN)),
        ("otro número de fracciones", |d| d.log_cusum.push(None)),
        ("referencia negativa", |d| d.referencia = -0.1),
        ("referencia −0,0", |d| d.referencia = -0.0),
        ("M > 0 sin cierres desde el reinicio", |d| {
            d.cierres_desde_reinicio = 0;
            d.log_cusum[0] = Some(0.5);
        }),
        ("M = 0 fuera del sufijo", |d| {
            d.log_cusum[0] = None;
            d.log_cusum[3] = Some(0.1);
        }),
    ];
    for (caso, estropear) in casos {
        let mut malo = bueno.clone();
        estropear(malo.monedas[1].detector_lg22.as_mut().unwrap());
        assert!(
            DetectorPerdidaDeVentaja::desde_guardado(malo.monedas[1].detector_lg22.as_ref().unwrap())
                .is_none(),
            "{caso}: inalcanzable"
        );
        assert!(validar(&malo).is_ok(), "{caso}: no bloquea el estado");
        let mut nuevo = nucleo();
        assert!(nuevo.restaurar_estado_aprendido(&malo) >= 2, "{caso}");
        assert_eq!(nuevo.arena.coins[1].metrics.trade_count.load(Relaxed), 9, "{caso}: lo demás sí");
        assert_eq!(nuevo.lg22_detectores[1].cierres_totales, 0, "{caso}: detector de cero");
        assert_eq!(
            nuevo.arena.registry.get_for_coin_or(1, "lg22_detector_descartado", -1.0),
            1.0,
            "{caso}: queda constancia"
        );
    }

    // Un cambio de formato del detector pasa también por el disco: el
    // archivo sigue legible (no Ilegible ni Bloqueado).
    let mut otro_formato = serde_json::to_value(&bueno).unwrap();
    otro_formato["monedas"][1]["detector_lg22"]["log_cusum"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::Value::Null);
    std::fs::write(tmp.ruta(), serde_json::to_string(&otro_formato).unwrap()).unwrap();
    let e = leer(&tmp.ruta());
    let mut nuevo = nucleo();
    assert!(nuevo.restaurar_estado_aprendido(&e) >= 2);
    assert_eq!(nuevo.lg22_detectores[1].cierres_totales, 0);
}

/// La evidencia contra otra H0 (otra referencia μ) no se traslada; contra la
/// misma, sí, también con μ distinta de cero.
#[test]
fn lg27_el_detector_se_continua_solo_contra_la_misma_referencia() {
    registrar_simbolos();
    let mut viejo = nucleo();
    viejo.lg22_detectores[1] = DetectorPerdidaDeVentaja::con_referencia(0.30);
    for i in 0..30 {
        cerrar(&mut viejo.lg22_detectores[1], i);
    }
    let e = viejo.estado_aprendido();
    assert!((e.monedas[1].detector_lg22.as_ref().unwrap().referencia - 0.30).abs() < 1e-15);

    // El núcleo vivo mide contra μ = 0: no se continúa.
    let mut con_cero = nucleo();
    assert!(con_cero.restaurar_estado_aprendido(&e) >= 2);
    assert_eq!(con_cero.lg22_detectores[1].cierres_totales, 0);
    assert_eq!(con_cero.lg22_detectores[1].referencia(), 0.0);
    assert_eq!(con_cero.arena.registry.get_for_coin_or(1, "lg22_detector_descartado", -1.0), 1.0);

    // El núcleo vivo mide contra la misma μ = 0,30: se continúa.
    let mut con_030 = nucleo();
    con_030.lg22_detectores[1] = DetectorPerdidaDeVentaja::con_referencia(0.30);
    assert!(con_030.restaurar_estado_aprendido(&e) >= 2);
    assert_eq!(con_030.lg22_detectores[1].cierres_totales, 30);
    assert!(casi_igual(con_030.lg22_detectores[1].estadistico(), viejo.lg22_detectores[1].estadistico()));
}

/// LG-27c: replantar un universo (CL-134, otra hipótesis) reinicia también el
/// detector, como los demás frenos; conserva su μ.
#[test]
fn lg27c_reiniciar_los_frenos_reinicia_el_detector() {
    registrar_simbolos();
    let mut c = nucleo();
    c.lg22_detectores[0] = DetectorPerdidaDeVentaja::con_referencia(0.30);
    for i in 0..30 {
        cerrar(&mut c.lg22_detectores[0], i);
        cerrar(&mut c.lg22_detectores[1], i);
    }
    assert!(c.lg22_detectores[1].estadistico() > 0.0);
    c.reiniciar_frenos(1_000.0);
    for (coin, mu) in [(0, 0.30), (1, 0.0)] {
        let d = &c.lg22_detectores[coin];
        assert_eq!((d.cierres_totales, d.cierres_desde_reinicio), (0, 0), "coin {coin}");
        assert_eq!(d.estadistico(), 0.0, "coin {coin}");
        assert_eq!(d.referencia(), mu, "coin {coin}: conserva μ");
    }
}
