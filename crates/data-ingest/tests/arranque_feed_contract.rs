//! I-6 (Línea I) — contrato del orden de arranque del feed.
//!
//! Los lectores WS se lanzaban al arrancar mientras el hilo consumidor seguía
//! inicializándose (VirtualLock, sesión OMS, calentamiento, bosque sombra).
//! La cola acumulaba hasta 5000 eventos viejos que el bucle procesaba como
//! vivos: cada arranque daba un LATENCY_PANIC, un KILL_SWITCH y unos segundos
//! de entradas bloqueadas (hallazgos A4 y C3,
//! `docs/audit/linea_i/i1_verificacion_y_diseno.json`). Este contrato fija
//! que cada lector espera a que el consumidor suelte su emisor y que el
//! consumidor lo suelta justo antes de su primer `recv`.

use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

const HOST: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../src/bin/god_engine.rs"));

/// El host sin comentarios de línea y sin espacios.
fn compacto() -> String {
    HOST.lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<String>()
        .split_whitespace()
        .collect()
}

#[test]
fn i6_el_consumidor_suelta_el_emisor_justo_antes_de_su_primer_recv() {
    let c = compacto();
    assert_eq!(
        c.matches("let(consumidor_listo,consumidor_listo_rx)=crossbeam_channel::bounded::<()>(0);").count(),
        1
    );
    let fase6 = c.find("SystemDiagnostics::execute_phase_6_hft();").expect("fase 6 del bucle");
    let suelta = c.find("drop(consumidor_listo);").expect("el consumidor debe soltar el emisor");
    let recv = c.find("whileletOk(mensaje_ws)=rx_events.recv()").expect("bucle del consumidor");
    assert!(fase6 < suelta && suelta < recv, "el emisor se suelta tras la fase 6 y antes del primer recv");
    assert_eq!(c.matches("drop(consumidor_listo);").count(), 1);
    // El emisor sólo se crea y se suelta: un clon o un uso más lo mantendría
    // vivo y los lectores esperarían para siempre.
    assert_eq!(
        c.matches("consumidor_listo").count() - c.matches("consumidor_listo_rx").count(),
        2,
        "el emisor sólo aparece al crear el canal y al soltarlo"
    );
    // El canal se crea antes del hilo consumidor, que se lleva el emisor.
    let canal = c.find("let(consumidor_listo,consumidor_listo_rx)").unwrap_or(usize::MAX);
    let hilo = c.find("letunified_handle=std::thread::Builder::new()").expect("hilo consumidor");
    assert!(canal < hilo);
}

#[test]
fn i6_cada_lector_espera_al_consumidor_antes_de_conectar() {
    let c = compacto();
    assert_eq!(c.matches("bucle_lector_ws(").count(), 3, "ancla de I-1b: una definición y dos llamadas");
    for hilo in ["name(\"ws-reader\".to_string())", "name(\"ws-reader-mercado\".to_string())"] {
        let ini = c.find(hilo).unwrap_or_else(|| panic!("falta el hilo {hilo}"));
        let resto = &c[ini..];
        let espera = resto.find("esperar_consumidor(&consumidor_listo_rx);").unwrap_or(usize::MAX);
        let conecta = resto.find("bucle_lector_ws(").unwrap_or(0);
        assert!(espera < conecta, "{hilo}: debe esperar al consumidor antes de bucle_lector_ws");
    }
    assert!(c.contains("fnesperar_consumidor(listo:&crossbeam_channel::Receiver<()>){whilelisto.recv().is_ok(){}}"));
}

/// La semántica del canal en la que se apoya la espera: un `bounded(0)` sin
/// emisores vivos despierta a todos los receptores, también a los clones, y
/// mientras el emisor vive ninguno pasa.
#[test]
fn i6_soltar_el_emisor_despierta_a_todos_los_lectores() {
    let (listo, rx) = crossbeam::channel::bounded::<()>(0);
    let arrancados = Arc::new(Barrier::new(3));
    let lectores: Vec<_> = (0..2)
        .map(|_| {
            let rx = rx.clone();
            let arrancados = Arc::clone(&arrancados);
            std::thread::spawn(move || {
                arrancados.wait();
                while rx.recv().is_ok() {}
                Instant::now()
            })
        })
        .collect();
    drop(rx);
    // Los dos hilos ya pasaron la barrera; el sueño les da tiempo de entrar
    // en `recv` antes de comprobar que siguen esperando.
    arrancados.wait();
    std::thread::sleep(Duration::from_millis(50));
    assert!(lectores.iter().all(|h| !h.is_finished()), "con el emisor vivo nadie conecta");
    let soltado = Instant::now();
    drop(listo);
    for h in lectores {
        let despierta = h.join().expect("el lector termina");
        assert!(despierta >= soltado);
    }
}
