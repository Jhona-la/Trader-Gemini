//! I-3a / I-3b (Línea I) — contrato de fuente del sistema inmune.
//!
//! El inmune aplana todas las posiciones a mercado tras tres muestras de
//! latencia por encima del umbral. Contaba con la edad total del dato, que
//! incluye la espera en la cola local: un atraso del consumidor se tomaba por
//! red degradada. Ahora cuenta con la componente red+reloj (I-2b) mediante
//! `data_ingest::edad_feed::ciclo_strikes`: conserva los strikes cuando el
//! exceso de edad viene de la cola local o el feed no trae eventos públicos
//! nuevos, nunca suma uno que el inmune anterior no sumara, y el rearme los
//! reinicia. Los avisos de «conservados» sólo salen cuando de verdad se
//! conservan.
//!
//! La premisa de seguridad de I-3a es que el núcleo sigue bloqueando ENTRADAS
//! con la edad total: este contrato la fija también en el núcleo.

const HOST: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../src/bin/god_engine.rs"));
const NUCLEO: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../god-engine-core/src/lib.rs"));

fn compacto(src: &str) -> String {
    src.lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<String>()
        .split_whitespace()
        .collect()
}

fn tramo<'a>(h: &'a str, desde: &str, hasta: &str) -> &'a str {
    let i = h.find(desde).unwrap_or_else(|| panic!("ancla «{desde}» ausente"));
    let f = i + h[i..].find(hasta).unwrap_or_else(|| panic!("ancla «{hasta}» ausente tras «{desde}»"));
    &h[i..f]
}

#[test]
fn i3a_el_inmune_cuenta_strikes_con_red_y_reloj() {
    let h = compacto(HOST);
    let t = tramo(&h, "letstalled=quantum_arena::feed_health::is_stalled();", "letlat_breach=latency_strikes>=3;");
    assert!(
        t.contains("letciclo=data_ingest::edad_feed::ciclo_strikes(red,espera_local,lat_thresh,stalled,!feed_avanza,);"),
        "los strikes deben salir de ciclo_strikes con red+reloj, la espera local, el stall y la frescura del feed"
    );
    assert!(t.contains("letstrikes_previos=latency_strikes;latency_strikes=ciclo.aplicar(strikes_previos);"));
    assert!(t.contains("letred=arena_imm.last_ws_red_reloj_ms.load(Ordering::Relaxed);"));
    assert!(t.contains("letespera_local=arena_imm.last_ws_espera_local_ms.load(Ordering::Relaxed);"));
    assert!(t.contains("leteventos_publicos=arena_imm.ws_eventos_publicos.load(Ordering::Relaxed);"));
    assert!(
        t.contains("letfeed_avanza=eventos_publicos!=eventos_publicos_previos;eventos_publicos_previos=eventos_publicos;"),
        "sin actualizar el contador previo, el feed parecería avanzar siempre"
    );
    assert!(!t.contains("iflat>lat_thresh"), "la edad total ya no cuenta strikes");
    assert!(!t.contains("latency_strikes+=1"), "la regla vive en una sola función");
    let asignaciones = t.match_indices("latency_strikes=").filter(|(i, _)| !t[i + 16..].starts_with('=')).count();
    assert_eq!(asignaciones, 1, "latency_strikes sólo cambia por ciclo.aplicar en el tramo");
    // Los avisos de «conservados» sólo salen cuando el ciclo conserva.
    assert!(t.contains("letconserva=ciclo==data_ingest::edad_feed::CicloStrikes::Conserva;"));
    assert!(t.contains("ifconserva&&!feed_avanza&&!stalled&&eventos_publicos>0{"));
    assert!(t.contains("ifconserva&&feed_avanza&&!stalled{"));
}

#[test]
fn i3a_los_eventos_publicos_alimentan_la_frescura() {
    let h = compacto(HOST);
    let t = tramo(&h, "ifevent_time>0&&(is_trade||is_depth){", "quantum_arena::feed_health::clear();");
    assert!(t.contains(".ws_eventos_publicos.fetch_add(1,Ordering::Relaxed);"));
}

#[test]
fn i3a_premisa_el_nucleo_bloquea_entradas_con_la_edad_total() {
    // Host: la cifra de decisión es la edad total y es la que llega al arena.
    let h = compacto(HOST);
    assert!(h.contains("letlatency_ms=edad.total_ms;"));
    assert!(h.contains(".last_ws_latency_ms.store(latency_ms.max(0)asu64,Ordering::Relaxed);"));
    // Núcleo: el interlock lee ese campo y bloquea entradas con él.
    let n = compacto(NUCLEO);
    let t = tramo(&n, "letlatency_ms=self.arena.last_ws_latency_ms.load(Ordering::Relaxed);", "ifis_latency_panic{");
    assert!(t.contains("letis_latency_panic=latency_ms>latency_threshold_msasu64;"));
    let bloqueo = tramo(t, "letentries_blocked=", ";");
    assert!(bloqueo.contains("||is_latency_panic"), "la edad total debe seguir bloqueando entradas");
}

#[test]
fn i3b_el_rearme_reinicia_los_strikes_de_latencia() {
    let h = compacto(HOST);
    let t = tramo(&h, "ifruta_rearme.exists(){", "iflatched{continue;}");
    assert!(t.contains("latched=false;"));
    assert!(t.contains("latency_strikes=0;"), "el rearme debe reiniciar los strikes");
    assert!(t.contains("letks_armado=arena_imm.kill_switch_active.load(Ordering::Relaxed);"));
    assert!(t.contains("ifks_armado{\"Elkill-switch"), "el aviso sólo afirma un kill-switch armado si lo está");
}
