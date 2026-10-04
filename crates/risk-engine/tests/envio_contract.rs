//! CL-41 — el host y el replay deciden el apalancamiento de envío con la
//! misma función, que nunca supera el validado por el riesgo. Guardia sobre
//! las fuentes: el ajuste D-382 vivía copiado en ambos con el doble conteo
//! de la reserva propia y sin techo.

fn compacta(src: &str) -> String {
    src.split_whitespace().collect()
}

#[test]
fn cl41_host_y_replay_usan_la_funcion_de_envio() {
    let host = compacta(include_str!("../../../src/bin/god_engine.rs"));
    let replay = compacta(include_str!("../../backtest-engine/src/booktick_replay.rs"));
    for (nombre, src) in [("host", &host), ("replay", &replay)] {
        assert!(
            src.contains("risk_engine::envio::apalancamiento_de_envio("),
            "{nombre}: el apalancamiento de envío sale de risk_engine::envio"
        );
        assert!(
            src.contains("risk_engine::envio::margen_libre_sin_la_propia("),
            "{nombre}: el margen libre no descuenta la reserva propia"
        );
        let copia = ["let", "needed_leverage"].concat();
        assert!(!src.contains(&copia), "{nombre}: ajuste D-382 copiado en línea");
    }
}

/// CL-41c: el host decide el envío y reajusta la reserva en el hilo del
/// núcleo, antes del `spawn` de la E/S. Dentro del spawn, el reajuste corría
/// en un worker de tokio: la siguiente entrada podía validarse con el margen
/// viejo y un cierre del núcleo podía intercalarse con el reajuste.
#[test]
fn cl41c_el_host_decide_el_envio_antes_del_spawn() {
    let host = compacta(include_str!("../../../src/bin/god_engine.rs"));
    let guarda = "letSome(effective_leverage)=decision_envioelse{";
    let i = host.find(guarda).expect("el spawn de la entrada usa la decisión tomada fuera");
    let spawn = host[..i].rfind("rt_handle.spawn(asyncmove{").expect("spawn de la entrada");
    let decision = host[..spawn]
        .rfind("letdecision_envio:Option<u32>='envio:{")
        .expect("decisión de envío en el hilo del núcleo");
    let antes = &host[decision..spawn];
    for requerido in [
        "risk_engine::envio::apalancamiento_de_envio(",
        "reservation.reajustar_margen(",
        "is_symbol_suspended(",
    ] {
        assert!(antes.contains(requerido), "{requerido} debe ir antes del spawn");
    }
    let resto = &host[spawn..];
    let fin = resto.find("});").unwrap_or(resto.len()).min(20_000);
    assert!(
        !resto[..fin].contains("reajustar_margen("),
        "ningún reajuste de la reserva dentro del spawn"
    );
}
