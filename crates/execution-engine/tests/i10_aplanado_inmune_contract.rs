//! I-10 (Línea I, fila N19 del censo C-F7b): el aplanado del sistema inmune
//! que no se completó se reintenta.
//!
//! Antes el inmune se marcaba hecho (`latched`) aunque el aplanado devolviera
//! Err, y su cerrojo era `if latched { continue; }`: con el freno 429/418
//! activo el aplanado no enviaba nada, con un rechazo del exchange dejaba la
//! posición abierta, y nadie volvía a intentarlo. La política de reintento la
//! fijan las pruebas de `execution_engine::aplanado_inmune`, también sobre el
//! ejecutor real frenado; aquí, la decisión pública y su uso en el host.

use execution_engine::aplanado_inmune::{
    tras_aplanado, ReintentoAplanado, TrasAplanado, PERIODO_CICLO_INMUNE_MS, TECHO_ESPERA_MS,
};

const HOST: &str = include_str!("../../../src/bin/god_engine.rs");

/// La fuente del host sin comentarios de línea completa y sin espacios.
fn compacto() -> String {
    HOST.lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<String>()
        .split_whitespace()
        .collect()
}

/// Desde `desde` hasta `hasta` (excluido); vacío si falta un ancla.
fn tramo<'a>(h: &'a str, desde: &str, hasta: &str) -> &'a str {
    let Some(i) = h.find(desde) else {
        return "";
    };
    let Some(f) = h[i..].find(hasta) else {
        return "";
    };
    &h[i..i + f]
}

/// El ciclo del inmune: desde el rearme por archivo hasta el disparo.
fn ciclo_inmune(h: &str) -> &str {
    tramo(
        h,
        "letruta_rearme=",
        "letoperator_lock=operator_lock_exists();",
    )
}

#[test]
fn i10_decision_publica_reintenta_lo_frenado_cada_ciclo_y_lo_fallido_con_espera_doble() {
    let freno = "RATE_BRAKE: freno temporal activo hasta 2 (ahora 1)";
    let incompleto = "FLATTEN_INCOMPLETE: 1 posiciones no pudieron cerrarse: [\"ETHUSDT\"]";
    assert_eq!(
        tras_aplanado::<(usize, usize)>(&Ok((0, 1))),
        TrasAplanado::Cumplido
    );
    assert_eq!(
        tras_aplanado::<(usize, usize)>(&Err(freno.into())),
        TrasAplanado::Frenado
    );
    assert_eq!(
        tras_aplanado::<(usize, usize)>(&Err(incompleto.into())),
        TrasAplanado::Fallido
    );

    let mut r = ReintentoAplanado::nuevo();
    r.registrar::<(usize, usize)>(&Err(incompleto.into()), 0);
    assert_eq!(r.proximo_ms(), Some(PERIODO_CICLO_INMUNE_MS));
    r.registrar::<(usize, usize)>(&Err(freno.into()), 7_000);
    assert_eq!(r.proximo_ms(), Some(7_000 + PERIODO_CICLO_INMUNE_MS));
    assert!(TECHO_ESPERA_MS > PERIODO_CICLO_INMUNE_MS);
}

/// RED antes de I-10: el cerrojo era `if latched { continue; }`.
#[test]
fn i10_con_el_cerrojo_puesto_el_inmune_reintenta_el_aplanado_pendiente() {
    let h = compacto();
    let cerrojo = tramo(ciclo_inmune(&h), "iflatched{", "continue;}");
    assert!(!cerrojo.is_empty(), "cerrojo del inmune en el host");
    let toca = cerrojo
        .find("reintento_aplanado.toca(")
        .unwrap_or(usize::MAX);
    let aplana = cerrojo
        .find("aplanado_inmune(&executor,&arena_imm,&mutreintento_aplanado,")
        .unwrap_or(usize::MAX);
    assert!(
        toca < aplana && aplana != usize::MAX,
        "con el cerrojo puesto, el aplanado pendiente se reintenta cuando toca"
    );
}

/// RED antes de I-10: el resultado del aplanado sólo se registraba en el log.
#[test]
fn i10_cada_intento_del_aplanado_anota_su_resultado() {
    let h = compacto();
    let f = tramo(
        &h,
        "asyncfnaplanado_inmune(",
        "asyncfnensure_position_protected(",
    );
    assert!(!f.is_empty(), "función del aplanado inmune en el host");
    let aplana = f
        .find("executor.flatten_all_positions().await")
        .unwrap_or(usize::MAX);
    let anota = f
        .find("reintento.registrar(&resultado,ahora_ms)")
        .unwrap_or(usize::MAX);
    assert!(
        aplana < anota && anota != usize::MAX,
        "el resultado del aplanado se anota para decidir el reintento"
    );
    let disparo = tramo(&h, "[SISTEMAINMUNE]ACTIVADO", "latched=true;");
    assert!(
        disparo.contains("aplanado_inmune(&executor,&arena_imm,&mutreintento_aplanado,ahora_imm"),
        "el disparo del inmune aplana por la misma función"
    );
}

/// El rearme del operador abandona el aplanado pendiente, y el periodo del
/// reintento frenado es el ciclo real del inmune.
#[test]
fn i10_el_rearme_olvida_el_pendiente_y_el_ciclo_es_el_de_la_politica() {
    let h = compacto();
    let rearme = tramo(ciclo_inmune(&h), "ifruta_rearme.exists(){", "iflatched{");
    assert!(
        rearme.contains("reintento_aplanado.olvidar();"),
        "el rearme por archivo abandona el aplanado pendiente"
    );
    let inicio_ciclo = h.find("letruta_rearme=").unwrap_or(0);
    let espera = h[..inicio_ciclo].rfind(
        "sleep(std::time::Duration::from_millis(execution_engine::aplanado_inmune::PERIODO_CICLO_INMUNE_MS",
    );
    assert!(
        espera.is_some_and(|i| inicio_ciclo - i < 160),
        "el inmune duerme el periodo de la política justo antes de cada ciclo"
    );
}
