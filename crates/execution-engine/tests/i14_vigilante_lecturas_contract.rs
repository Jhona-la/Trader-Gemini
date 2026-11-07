//! I-14 (Línea I, filas N44, N34 y N42 del censo C-F7b): el vigilante de
//! protección del host sólo actúa sobre lo que leyó.
//!
//! Las decisiones las prueban las pruebas de `execution_engine::vigilante`;
//! aquí, su uso en el host (guardas sobre la fuente de `god_engine.rs`, que no
//! es testeable por unidad: sin exchange ni red) y la escalada B3.5b con el
//! acumulado real de `protection_health`.

use execution_engine::vigilante::{BaseDeRechazos, Cierre, Lecturas, ReintentoLectura};
use std::collections::HashSet;

/// La fuente del host sin los comentarios de línea completa, compacta.
fn host() -> String {
    include_str!("../../../src/bin/god_engine.rs")
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<String>()
        .split_whitespace()
        .collect()
}

/// El texto entre `desde` y la primera aparición de `hasta` tras él.
fn tramo<'a>(h: &'a str, desde: &str, hasta: &str) -> &'a str {
    let i = h
        .find(desde)
        .unwrap_or_else(|| panic!("ancla «{desde}» ausente"));
    let f = i + h[i..]
        .find(hasta)
        .unwrap_or_else(|| panic!("ancla «{hasta}» ausente tras «{desde}»"));
    &h[i..f]
}

fn vigilante(h: &str) -> &str {
    tramo(
        h,
        "[PROTECTION-WATCHDOG]Vigilantedeposicióndesnudaactivo",
        "let_msg_count:u64=0;",
    )
}

/// RED antes de I-14: las piernas se leían con `unwrap_or_default()` y un
/// error era «ninguna pierna»: se colocaban TP y SL encima de las vivas.
#[test]
fn i14_n44_la_proteccion_no_rellena_lo_que_no_leyo() {
    let h = host();
    let f = tramo(
        &h,
        "asyncfnensure_position_protected(",
        "staticSYMBOL_SUSPENDED_UNTIL",
    );
    assert!(
        !f.contains("fetch_open_algo_orders(symbol).await.unwrap_or_default()"),
        "un error de lectura no es «ninguna pierna»"
    );
    assert!(f.contains(")->Result<ProtectionAudit,String>{"));
    let lee = f
        .find(
            "vigilante::cobertura_leida(executor.fetch_open_algo_orders(symbol).await,is_long,)?;",
        )
        .unwrap_or(usize::MAX);
    // Sin el filtro no se coloca nada, pero los huecos leídos vuelven, como
    // antes de I-14: cuentan para la escalada.
    let filtro = f
        .find(
            "letf=matchexecutor.get_symbol_filter(symbol).await{Ok(f)=>f,Err(e)=>{returnOk(ProtectionAudit{tp_gap,sl_gap,sin_filtro:Some(e),})}};",
        )
        .unwrap_or(usize::MAX);
    let huecos = f
        .find("lettp_gap=(pos_qty-tp_covered).max(0.0);letsl_gap=(pos_qty-sl_covered).max(0.0);")
        .unwrap_or(usize::MAX);
    let coloca = f.find("place_algo_leg(").unwrap_or(0);
    assert!(
        lee < huecos && huecos < filtro && filtro < coloca && coloca != 0,
        "se leen las piernas, sus huecos y el filtro, con salida temprana, antes de colocar nada"
    );
    assert!(!f.contains("get_symbol_filter(symbol).await?"));
    assert_eq!(f.matches("sin_filtro:None").count(), 3);
    assert!(
        f.ends_with("Ok(audit)}"),
        "la auditoría completa devuelve Ok"
    );
}

/// El vigilante salta la posición sin leer: no coloca, no cuenta racha ni
/// toca la base de rechazos. Sin el filtro cuenta los huecos y relee. La
/// adopción del arranque lo dice y despierta al vigilante, en vez de
/// descartar el resultado con `let _ =`.
#[test]
fn i14_n44_quien_llama_no_toma_el_error_por_cobertura() {
    let h = host();
    let v = vigilante(&h);
    let llamada = tramo(
        v,
        "letprot=matchensure_position_protected(",
        "let(g_tp,g_sl)=",
    );
    assert!(
        llamada.contains("Err(e)=>{sin_leer+=1;lecturas.error(&p.symbol,&e);continue;}"),
        "sin lectura: ni piernas, ni racha, ni base"
    );
    assert!(
        llamada.ends_with("};ifletSome(e)=&prot.sin_filtro{lecturas.error(&p.symbol,e);}"),
        "sin el filtro, la auditoría no está completa"
    );
    assert!(!h.contains("let_=ensure_position_protected("));
    let adopcion = tramo(
        &h,
        "ifletOk(ProtectionAudit{sin_filtro:Some(e),..})|Err(e)=ensure_position_protected(",
        "all_positions_adopted=false;",
    );
    assert!(adopcion.contains("\"RESTORE\""));
    assert!(adopcion.contains("protection_health::mark_dirty();"));
}

/// RED antes de I-14 (N34): `if let Ok(all_legs)` se tragaba el error del
/// listado, y una huérfana sin cancelar sólo se reportaba.
#[test]
fn i14_n34_la_purga_de_huerfanas_dice_si_leyo_y_anota_lo_que_no_cancelo() {
    let h = host();
    let f = tramo(
        &h,
        "asyncfnpurge_orphan_legs(",
        "fncerrar_lecturas_del_vigilante(",
    );
    assert!(f.contains(")->Result<u64,String>{"));
    assert!(
        !f.contains("ifletOk(all_legs)"),
        "el error del listado no se traga"
    );
    assert!(
        f.contains("letall_legs=executor.fetch_all_open_algo_orders().await?;letmutmarcas=0u64;")
    );
    assert!(
        f.contains("if!es_huerfana(leg,positions){continue;}"),
        "la regla de D-698, en vigilante"
    );
    let fallo = tramo(f, "Err(e)=>{", "Ok(marcas)}");
    assert!(
        fallo
            .contains("ifanotar_huerfana_pendiente(&leg.symbol,lado_de_la_purga(leg)){marcas+=1;}"),
        "la huérfana viva bloquea la reentrada hasta purgarla (I-9)"
    );
    // RED en la segunda versión de I-14: `anotar_purga_pendiente` marca la
    // bandera siempre, y una huérfana que sigue sin cancelarse la marcaba en
    // cada auditoría.
    let anota = tramo(
        &h,
        "fnanotar_huerfana_pendiente(",
        "asyncfnreintentar_purgas_pendientes(",
    );
    assert!(anota.contains(
        "->bool{letnueva=execution_engine::purga_pendiente::PURGAS_PENDIENTES.anotar(symbol,is_long);ifnueva{quantum_arena::protection_health::mark_dirty();}nueva}"
    ));
    let v = vigilante(&h);
    assert!(v.contains(
        "matchpurge_orphan_legs(&executor,&positions).await{Ok(marcas)=>marcas_propias+=marcas,Err(e)=>lecturas.error(\"openAlgoOrders\",&e),}"
    ));
    let apagado = tramo(
        &h,
        "UnifiedEventLoopsafelyterminated",
        "std::process::exit(0)",
    );
    assert!(apagado.contains(
        "ifletErr(e)=rt_handle_for_thread.block_on(purge_orphan_legs(&executor,&positions))"
    ));
}

/// RED antes de I-14: la bandera sucia se limpiaba aunque la purga no
/// hubiera leído el listado, y una lectura fallida sin bandera esperaba a la
/// auditoría nominal de 60 s.
#[test]
fn i14_la_bandera_sucia_solo_se_limpia_con_las_lecturas_completas() {
    let h = host();
    let v = vigilante(&h);
    assert!(v.contains(
        "letnominal=ticks%execution_engine::vigilante::CICLOS_NOMINALES==0;if!nominal&&!reintento_lectura.toca(ticks,events_at_start,dirty){continue;}"
    ));
    // Un solo acumulador por auditoría, creado tras empezarla: el de antes
    // no se arrastra a la siguiente.
    assert_eq!(v.matches("lecturas=").count(), 1);
    assert_eq!(v.matches("marcas_propias=").count(), 1);
    let bucle = v.find("loop{").unwrap_or(usize::MAX);
    let empieza = v.find("try_begin_audit()").unwrap_or(usize::MAX);
    let crea = v
        .find("letmutlecturas=execution_engine::vigilante::Lecturas::new();letmutmarcas_propias=0u64;")
        .unwrap_or(0);
    assert!(bucle < empieza && empieza < crea && crea != 0);
    // Las dos llamadas de cierre pasan los eventos del inicio más las marcas
    // de la propia auditoría.
    assert_eq!(
        v.matches(
            "cerrar_lecturas_del_vigilante(&mutreintento_lectura,ticks,events_at_start+marcas_propias,&lecturas,)"
        )
        .count(),
        2
    );
    // Las dos purgas pendientes de la escalada son marcas propias.
    assert_eq!(
        v.matches("anotar_purga_pendiente(&close_sym,close_long);marcas_propias+=1;")
            .count(),
        2
    );
    assert_eq!(v.matches("anotar_purga_pendiente(").count(), 2);
    let posiciones = tramo(
        v,
        "letpositions=matchexecutor.fetch_position_risk().await{",
        "};",
    );
    assert!(
        posiciones.contains("lecturas.error(\"positionRisk\",&e);cerrar_lecturas_del_vigilante(")
    );
    assert!(posiciones.ends_with("continue;}"));
    let cierre = v
        .find("letcompleta=cerrar_lecturas_del_vigilante(")
        .unwrap_or(usize::MAX);
    assert!(
        v.rfind("lecturas.error(").unwrap_or(usize::MAX) < cierre,
        "se cierra tras anotar todas las lecturas"
    );
    let si_sucia = v[cierre.min(v.len())..]
        .find("ifdirty{")
        .map(|i| i + cierre);
    let limpia = v.find("ifcompleta&&quantum_arena::protection_health::terminal_events_seen()==events_at_start{quantum_arena::protection_health::clear_dirty();}");
    assert!(
        matches!((si_sucia, limpia), (Some(s), Some(l)) if cierre < s && s < l),
        "se cierran las lecturas, luego se mira la bandera, y sólo se limpia completa"
    );
    assert_eq!(
        h.matches("clear_dirty()").count(),
        1,
        "un solo sitio del host limpia la bandera"
    );
    let ayuda = tramo(
        &h,
        "fncerrar_lecturas_del_vigilante(",
        "asyncfnesperar_auditoria_en_curso(",
    );
    // RED en la primera versión de I-14: volvía a marcar la bandera, y
    // `mark_dirty` cuenta un evento terminal que adelantaba la relectura
    // cada 5 s.
    assert!(
        !ayuda.contains("mark_dirty"),
        "una auditoría incompleta no se adelanta a sí misma"
    );
    // RED en la segunda versión: tomaba los eventos al cerrar, y un evento
    // ajeno llegado durante la auditoría no adelantaba la relectura.
    assert!(!ayuda.contains("terminal_events_seen"));
    assert!(ayuda.contains(
        "ciclo:u32,eventos:u64,lecturas:&execution_engine::vigilante::Lecturas,)->bool{"
    ));
    assert!(ayuda.contains("letcierre=reintento.cerrar(ciclo,eventos,lecturas);"));
    assert!(ayuda.ends_with("cierre.completa()}"));
}

/// RED antes de I-14 (N42): la foto se borraba al quedar cubierto o cerrado
/// el símbolo.
#[test]
fn i14_n42_la_base_de_rechazos_no_se_borra() {
    let h = host();
    let v = vigilante(&h);
    assert!(!v.contains("naked_rejections.remove("));
    assert!(!v.contains("naked_rejections.retain("));
    assert!(
        v.contains("letmutnaked_rejections=execution_engine::vigilante::BaseDeRechazos::new();")
    );
    assert!(v.contains("letrejections_prev=naked_rejections.tomar(&p.symbol,rejections_now);ifrejections_now<=rejections_prev{"));
    assert!(v.contains("}else{naked_streak.remove(&p.symbol);naked_rejections.fijar(&p.symbol,quantum_arena::protection_health::rejections_of(&p.symbol),);}"));
    assert!(v.contains("naked_rejections.fijar_los_cerrados(&open_syms,quantum_arena::protection_health::rejections_of,);"));
    // Una posición sin leer sigue abierta: `fijar_los_cerrados` no la toma
    // por cerrada.
    let abierta = v
        .find("open_syms.insert(p.symbol.clone());")
        .unwrap_or(usize::MAX);
    let lee = v
        .find("letprot=matchensure_position_protected(")
        .unwrap_or(0);
    assert!(abierta < lee);
}

/// La escalada B3.5b de dos huecos de stop con el acumulado real de
/// `protection_health`. Con la base borrada al quedar cubierto (antes de
/// I-14), el segundo hueco empezaba con un paso de racha por los dos
/// rechazos del primero; ahora empieza en cero.
#[test]
fn i14_n42_dos_huecos_de_stop_con_el_acumulado_real() {
    use quantum_arena::protection_health::{note_rejection, rejections_of};
    const S: &str = "I14N42USDT";
    let mut base = BaseDeRechazos::new();
    let mut racha = 0u32;
    // La auditoría del vigilante sobre un hueco de stop: paso de racha sólo
    // con rechazos nuevos.
    let auditar_hueco = |base: &mut BaseDeRechazos, racha: &mut u32| {
        let ahora = rejections_of(S);
        let prev = base.tomar(S, ahora);
        if ahora > prev {
            *racha += 1;
        }
    };
    // Primer hueco: dos rechazos reales del exchange en dos auditorías.
    assert!(note_rejection(S, "-2021: Order would immediately trigger."));
    auditar_hueco(&mut base, &mut racha);
    assert!(note_rejection(S, "-2021: Order would immediately trigger."));
    auditar_hueco(&mut base, &mut racha);
    assert_eq!(racha, 2);
    // La pierna entra: cubierta. La racha vuelve a cero y la base se fija.
    racha = 0;
    base.fijar(S, rejections_of(S));
    // Segundo hueco, sin rechazos nuevos (un 5xx no cuenta como rechazo).
    assert!(!note_rejection(S, "HTTP 503 Service Unavailable"));
    auditar_hueco(&mut base, &mut racha);
    assert_eq!(
        racha, 0,
        "los rechazos del primer hueco no cuentan en el segundo"
    );
    // La posición cierra y vuelve a abrir: el cierre fija la base.
    let abiertos: HashSet<String> = HashSet::new();
    base.fijar_los_cerrados(&abiertos, rejections_of);
    assert!(note_rejection(S, "-2021: Order would immediately trigger."));
    auditar_hueco(&mut base, &mut racha);
    assert_eq!(racha, 1, "el rechazo de la reentrada sí cuenta");
}

/// La cadencia que el vigilante aplica, de punta a punta: una lectura que
/// falla tres veces, un freno 429/418 que se relee en el ciclo siguiente y
/// esa relectura, que completa.
#[test]
fn i14_relectura_tras_fallos_y_frenos() {
    let mut r = ReintentoLectura::new();
    let fallo = || {
        let mut l = Lecturas::new();
        l.error(
            "openAlgoOrders",
            "OPEN_ALGO_PARSE: expected value at line 1",
        );
        l
    };
    let freno = || {
        let mut l = Lecturas::new();
        l.error(
            "positionRisk",
            "RATE_BRAKE: freno temporal activo hasta 1700000000000 (ahora 1)",
        );
        l
    };
    let mut auditados = Vec::new();
    let lecturas_de = |ciclo: u32| match ciclo {
        1 | 2 | 4 => fallo(),
        8 => freno(),
        _ => Lecturas::new(),
    };
    // La bandera sigue sucia: una auditoría incompleta no la limpia.
    for ciclo in 1..=30u32 {
        let nominal = ciclo % 12 == 0;
        if !nominal && !r.toca(ciclo, 0, true) {
            continue;
        }
        auditados.push(ciclo);
        if r.cerrar(ciclo, 0, &lecturas_de(ciclo)).completa() {
            break;
        }
    }
    // 1 falla (espera 1), 2 falla (espera 2), 4 falla (espera 4), 8 frenada
    // (espera 1, sin doblar) y el 9 completa.
    assert_eq!(auditados, vec![1, 2, 4, 8, 9]);
    assert_eq!(r.incompletas(), 0);
}

/// Con el contador real de `protection_health`: la marca que hace la propia
/// auditoría (una purga pendiente nueva, `mark_dirty`, que cuenta un evento
/// terminal) no adelanta su relectura; un evento ajeno sí, llegue durante
/// la auditoría o después. Ninguna otra prueba de este binario toca la
/// bandera.
#[test]
fn i14_las_marcas_de_la_propia_auditoria_no_adelantan_la_relectura() {
    use quantum_arena::protection_health::{is_dirty, mark_dirty, terminal_events_seen};
    let mut r = ReintentoLectura::new();
    let mut fallo = Lecturas::new();
    fallo.error("openAlgoOrders", "OPEN_ALGO_PARSE: expected value");
    // Ciclo 1: la auditoría falla y anota una purga pendiente nueva.
    let inicio = terminal_events_seen();
    mark_dirty();
    let c = r.cerrar(1, inicio + 1, &fallo);
    assert!(matches!(
        c,
        Cierre::Fallida {
            espera_ciclos: 1,
            ..
        }
    ));
    assert!(!r.toca(1, terminal_events_seen(), is_dirty()));
    // Ciclo 2: toca (espera de un ciclo). Vuelve a fallar y a anotar.
    assert!(r.toca(2, terminal_events_seen(), is_dirty()));
    let inicio = terminal_events_seen();
    mark_dirty();
    r.cerrar(2, inicio + 1, &fallo);
    // Espera de dos ciclos: el ciclo 3 no audita aunque la bandera esté sucia
    // y el contador haya subido durante la auditoría del 2.
    assert!(is_dirty());
    assert!(
        !r.toca(3, terminal_events_seen(), is_dirty()),
        "su propia marca no adelanta la relectura"
    );
    assert!(r.toca(4, terminal_events_seen(), is_dirty()));
    // Ciclo 4: un ALGO_UPDATE ajeno llega durante la auditoría, que no
    // marca nada: la relectura se adelanta al ciclo siguiente aunque la
    // espera sea de cuatro.
    let inicio = terminal_events_seen();
    mark_dirty();
    assert!(matches!(
        r.cerrar(4, inicio, &fallo),
        Cierre::Fallida {
            espera_ciclos: 4,
            ..
        }
    ));
    assert!(
        r.toca(5, terminal_events_seen(), is_dirty()),
        "el evento ajeno llegado durante la auditoría la adelanta"
    );
    // Y uno que llega después del cierre, también.
    let fin = terminal_events_seen();
    r.cerrar(5, fin, &fallo);
    assert!(!r.toca(6, terminal_events_seen(), is_dirty()));
    mark_dirty();
    assert!(r.toca(6, terminal_events_seen(), is_dirty()));
}
