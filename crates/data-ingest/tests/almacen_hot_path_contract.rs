//! CONTRATO I-7 (Línea I, rescate GLM-170): cero E/S del almacén de
//! genomas en el hilo caliente del núcleo unificado.
//!
//! QUÉ: el borde de cada 5000 mensajes ya no ejecuta `metadata`,
//! `load_active` ni `promote` (fsync) — toda esa E/S vive en el worker
//! `almacen-genomas`; el borde pide sin bloquear y aplica la respuesta en
//! el borde SIGUIENTE (aplicación diferida ≤5000 mensajes).
//! POR QUÉ: un fsync en el bucle de ticks es un stall de 10-100+ ms con
//! riesgo de watchdog (docs/PLAN_LINEA_I_CLAUDE_2026-10-24.md, fila I-7;
//! premisa verificada en GLM-169).
//! PARA QUÉ: el arranque y la cosecha en vivo dejan de congelar el flujo
//! de eventos.
//! CÓMO: (1-3) verificación estructural del fuente del bin (la función
//! vive en `src/bin/god_engine.rs`, patrón de `arranque_feed_contract`);
//! (4) modelo conductual del diferimiento con canales reales.

use std::path::PathBuf;

fn fuente_god_engine() -> String {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let ruta = manifest.join("../../src/bin/god_engine.rs");
    std::fs::read_to_string(&ruta).unwrap_or_else(|e| panic!("no pude leer {:?}: {}", ruta, e))
}

/// El borde de 5000 del núcleo unificado: desde el `if` hasta el final
/// del bucle (marcador del log de terminación del núcleo).
fn borde_5000(fuente: &str) -> &str {
    let inicio = fuente
        .find("if msg_count.is_multiple_of(5000) {")
        .expect("el borde de 5000 debe existir");
    let fin = fuente[inicio..]
        .find("✅ [UNIFIED CORE]")
        .map(|i| inicio + i)
        .unwrap_or(fuente.len());
    &fuente[inicio..fin]
}

#[test]
fn i7_1_el_borde_de_5000_no_toca_el_almacen_de_genomas() {
    let fuente = fuente_god_engine();
    let borde = borde_5000(&fuente);
    assert!(
        !borde.contains("genome_store::"),
        "I-7: el borde de 5000 ejecuta E/S del almacén de genomas en el hilo caliente"
    );
    assert!(
        !borde.contains("std::fs::"),
        "I-7: el borde de 5000 ejecuta E/S de archivos en el hilo caliente"
    );
}

#[test]
fn i7_2_el_borde_pide_sin_bloquear_y_aplica_diferido() {
    let fuente = fuente_god_engine();
    let borde = borde_5000(&fuente);
    // Consumir primero (try_recv, jamás recv bloqueante)…
    assert!(
        borde.contains("try_recv"),
        "I-7: el borde debe consumir las respuestas del worker con try_recv"
    );
    assert!(
        !borde.contains("rx_desde_almacen.recv()"),
        "I-7: el borde NO debe bloquear esperando al worker (recv bloqueante)"
    );
    // …pedir después (fire-and-forget).
    assert!(
        borde.contains("tx_almacen.send(PedidoAlmacen::Sincronizar)"),
        "I-7: el borde debe pedir la sincronización al worker"
    );
    assert!(
        borde.contains("PedidoAlmacen::Promover"),
        "I-7: la cosecha debe pasar por el worker (promote con fsync fuera)"
    );
}

#[test]
fn i7_3_el_worker_propietario_existe_y_es_el_unico_dueno_de_la_escritura() {
    let fuente = fuente_god_engine();
    assert!(
        fuente.contains("\"almacen-genomas\""),
        "I-7: el worker del almacén debe existir como hilo nombrado"
    );
    // El único promote del núcleo unificado vive DENTRO del worker: el
    // llamado en el borde ya no existe (se busca la FORMA DE LLAMADA con
    // paréntesis; los comentarios que lo mencionan no cuentan).
    let borde = borde_5000(&fuente);
    assert!(
        !borde.contains("GenomeEnvelope::promote("),
        "I-7: promote (fsync) debe vivir en el worker, no en el borde"
    );
    // El espejo de mtime (CL-40) vive en el worker.
    let worker = fuente
        .find("\"almacen-genomas\"")
        .expect("worker presente");
    let fin_worker = fuente[worker..].find("});").map(|i| worker + i).unwrap_or(fuente.len());
    assert!(
        fuente[worker..fin_worker].contains("espejo"),
        "I-7: el espejo de mtime (CL-40) debe vivir dentro del worker"
    );
}

#[test]
fn i7_4_el_borde_nunca_espera_al_worker_aplicacion_un_borde_despues() {
    // Modelo conductual del contrato: el worker tarda (fsync); el borde
    // que pide NO espera (try_recv vacío e inmediato) y el borde
    // SIGUIENTE recibe la respuesta ya lista y la aplica.
    type Pedido = u32;
    type Respuesta = u32;
    let (tx_pedidos, rx_pedidos) = std::sync::mpsc::channel::<Pedido>();
    let (tx_resp, rx_resp) = std::sync::mpsc::channel::<Respuesta>();
    let hilo = std::thread::spawn(move || {
        while let Ok(pedido) = rx_pedidos.recv() {
            std::thread::sleep(std::time::Duration::from_millis(80)); // el "fsync"
            let _ = tx_resp.send(pedido * 10);
        }
    });

    // Borde N: pedir sin bloquear.
    let t0 = std::time::Instant::now();
    let _ = tx_pedidos.send(7);
    let temprano = rx_resp.try_recv();
    let espera_borde_n = t0.elapsed();
    assert!(
        matches!(temprano, Err(std::sync::mpsc::TryRecvError::Empty)),
        "borde N: la respuesta NO puede estar ya (el worker está en el fsync)"
    );
    assert!(
        espera_borde_n < std::time::Duration::from_millis(20),
        "borde N no espera al worker: pasó {:?}",
        espera_borde_n
    );

    // Borde N+1 (≈5000 mensajes después): la respuesta llega SIN QUE EL
    // BORDE BLOQUEE — try_recv en un bucle acotado (bajo carga del SO el
    // sleep del worker puede estirarse; el invariante es que el borde
    // jamás usa recv() bloqueante, no que llegue en un tiempo fijo).
    let limite = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        match rx_resp.try_recv() {
            Ok(aplicada) => {
                assert_eq!(aplicada, 70, "la respuesta del worker se aplica");
                break;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                panic!("borde N+1: el worker murió sin responder")
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                assert!(
                    std::time::Instant::now() < limite,
                    "borde N+1: la respuesta diferida no llegó en 10 s (¿worker muerto?)"
                );
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        }
    }
    drop(tx_pedidos);
    let _ = hilo.join();
}
