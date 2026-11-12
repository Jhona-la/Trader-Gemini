//! I-14 (Línea I, filas N44, N34 y N42 del censo C-F7b): el vigilante de
//! protección del host sólo actúa sobre lo que leyó.
//!
//! Las decisiones las prueban las pruebas de `execution_engine::vigilante`;
//! aquí, su uso en el host (guardas sobre la fuente de `god_engine.rs`, que no
//! es testeable por unidad: sin exchange ni red) y la escalada B3.5b con el
//! acumulado real de `protection_health`, por (símbolo, lado).

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
            "letf=matchexecutor.get_symbol_filter(symbol).await{Ok(f)=>f,Err(e)=>{returnOk(ProtectionAudit{tp_gap,sl_gap,sin_filtro:Some(e),colocacion_frenada:None,})}};",
        )
        .unwrap_or(usize::MAX);
    let huecos = f
        .find("lettp_gap=(pos_qty-tp_covered).max(0.0);letsl_gap=(pos_qty-sl_covered).max(0.0);")
        .unwrap_or(usize::MAX);
    // RED en la quinta versión de I-14: una posición cubierta del todo pedía
    // el filtro, y un símbolo cuyo filtro no se podía leer dejaba incompleta
    // cada auditoría.
    let cubierta = f
        .find(
            "iftp_gap<=0.0&&sl_gap<=0.0{returnOk(ProtectionAudit{tp_gap:0.0,sl_gap:0.0,sin_filtro:None,colocacion_frenada:None,});}",
        )
        .unwrap_or(usize::MAX);
    let coloca = f.find("place_algo_leg(").unwrap_or(0);
    assert!(
        lee < huecos && huecos < cubierta && cubierta < filtro && filtro < coloca && coloca != 0,
        "se leen las piernas y sus huecos; cubierta, sale sin el filtro; si no, el filtro, con salida temprana, antes de colocar nada"
    );
    assert!(!f.contains("get_symbol_filter(symbol).await?"));
    assert_eq!(f.matches("sin_filtro:None").count(), 4);
    assert_eq!(f.matches("colocacion_frenada:None").count(), 5);
    // Las dos piernas: el rechazo es de su lado, y la colocación frenada se
    // devuelve para que el vigilante relea en el ciclo siguiente.
    assert_eq!(f.matches("note_rejection(symbol,is_long,&e);").count(), 2);
    assert_eq!(
        f.matches("letfrenada=execution_engine::freno_temporal::es_freno_temporal(&e);")
            .count(),
        2
    );
    assert_eq!(
        f.matches("iffrenada&&audit.colocacion_frenada.is_none(){audit.colocacion_frenada=Some(e);}break;")
            .count(),
        2
    );
    assert!(
        f.ends_with("Ok(audit)}"),
        "la auditoría completa devuelve Ok"
    );
}

/// El vigilante salta la posición sin leer: no coloca ni cuenta racha, y su
/// base de rechazos sólo se siembra si era nueva. Sin el filtro cuenta los
/// huecos y relee; con una colocación frenada, relee en el ciclo siguiente.
/// Si no pudo leer las piernas o el filtro, o una colocación salió frenada,
/// la adopción del arranque lo dice y despierta al vigilante, en vez de
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
        llamada.ends_with(
            "};ifletSome(e)=&prot.sin_filtro{lecturas.error(&p.symbol,e);}ifprot.colocacion_frenada.is_some(){lecturas.colocacion_frenada();}"
        ),
        "sin el filtro, o con una colocación frenada, la auditoría no está completa"
    );
    assert!(!h.contains("let_=ensure_position_protected("));
    let adopcion = tramo(
        &h,
        "ifletOk(ProtectionAudit{sin_filtro:Some(e),..})|Ok(ProtectionAudit{colocacion_frenada:Some(e),..})|Err(e)=ensure_position_protected(",
        "all_positions_adopted=false;",
    );
    assert!(adopcion.contains("\"RESTORE\""));
    assert!(adopcion.contains("protection_health::mark_dirty();"));
}

/// RED antes de I-14 (N34): `if let Ok(all_legs)` se tragaba el error del
/// listado, y una huérfana sin cancelar sólo se reportaba. RED en la quinta
/// versión: el listado se juzgaba con posiciones leídas antes, y el bracket
/// que una entrada colocaba entre las dos lecturas salía huérfano. RED en la
/// sexta: el reintento de purgas pendientes sacaba del listado todo el
/// símbolo purgado (en hedge, también las huérfanas del otro lado), y una
/// huérfana que ya no existía quedaba como purga pendiente.
#[test]
fn i14_n34_la_purga_de_huerfanas_usa_el_listado_previo_y_anota_lo_que_sigue_abierto() {
    let h = host();
    let f = tramo(
        &h,
        "asyncfnpurge_orphan_legs(",
        "fncerrar_lecturas_del_vigilante(",
    );
    assert!(f.contains(
        "all_legs:&[execution_engine::order_types::OpenAlgoOrder],positions:&[execution_engine::PositionRiskEntry],)->u64{"
    ));
    assert!(
        !f.contains("fetch_all_open_algo_orders"),
        "el listado lo lee quien llama, antes que las posiciones"
    );
    assert!(
        f.contains("if!es_huerfana(leg,positions){continue;}"),
        "la regla de D-698, en vigilante"
    );
    let fallo = tramo(f, "Err(e)=>{", "marcas}");
    assert!(
        fallo
            .contains("ifanotar_huerfana_pendiente(&leg.symbol,lado_de_la_purga(leg)){marcas+=1;}"),
        "la huérfana viva bloquea la reentrada hasta purgarla (I-9)"
    );
    let relee = fallo
        .find("letreleida=executor.fetch_open_algo_orders(&leg.symbol).await;if!sigue_abierta(leg,&releida){")
        .unwrap_or(usize::MAX);
    let anota = fallo.find("ifanotar_huerfana_pendiente(").unwrap_or(0);
    assert!(
        relee < anota && anota != 0,
        "antes de anotarla se relee: la que ya no está no bloquea el símbolo"
    );
    assert!(
        tramo(
            fallo,
            "if!sigue_abierta(leg,&releida){",
            "telemetry_engine::telemetry_err!("
        )
        .ends_with("continue;}"),
        "la que ya no está no se anota"
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
    let lista = v
        .find("letlistado=executor.fetch_all_open_algo_orders().await;")
        .unwrap_or(usize::MAX);
    let posiciones = v
        .find("letpositions=matchexecutor.fetch_position_risk().await{")
        .unwrap_or(usize::MAX);
    let reintento = v
        .find("letpurgados=reintentar_purgas_pendientes(&executor,&positions).await;")
        .unwrap_or(usize::MAX);
    let purga = v
        .find(
            "matchlistado{Ok(mutpiernas)=>{execution_engine::vigilante::quitar_las_purgadas(&mutpiernas,&purgados,executor.is_hedge_mode.load(Ordering::SeqCst),);marcas_propias+=purge_orphan_legs(&executor,&piernas,&positions).await;}Err(e)=>lecturas.error(\"openAlgoOrders\",&e),}",
        )
        .unwrap_or(0);
    assert!(
        lista < posiciones && posiciones < reintento && reintento < purga && purga != 0,
        "listado, posiciones, reintento de purgas pendientes y purga de huérfanas, en ese orden"
    );
    assert_eq!(v.matches("fetch_all_open_algo_orders").count(), 1);
    let reintenta = tramo(
        &h,
        "asyncfnreintentar_purgas_pendientes(",
        "asyncfnpurge_orphan_legs(",
    );
    assert!(reintenta.contains(")->std::collections::HashSet<(String,bool)>{"));
    // Sólo la rama Purgada devuelve la posición purgada: una Aplazada o una
    // Fallida no canceló (todas) sus piernas.
    assert!(
        reintenta.contains("purgados.insert((symbol,is_long));}Reintento::Aplazada=>"),
        "la inserción, al final de la rama Purgada"
    );
    assert!(tramo(
        reintenta,
        "Reintento::Purgada(n)=>{",
        "Reintento::Aplazada=>"
    )
    .ends_with("purgados.insert((symbol,is_long));}"));
    assert_eq!(reintenta.matches("purgados.insert(").count(), 1);
    assert!(reintenta.ends_with("purgados}"));
    let apagado = tramo(
        &h,
        "UnifiedEventLoopsafelyterminated",
        "std::process::exit(0)",
    );
    assert!(apagado.contains(
        "letlistado=rt_handle_for_thread.block_on(executor.fetch_all_open_algo_orders());match(listado,rt_handle_for_thread.block_on(executor.fetch_position_risk())){(Ok(piernas),Ok(positions))=>{letnuevas=rt_handle_for_thread.block_on(purge_orphan_legs(&executor,&piernas,&positions));ifnuevas>0{"
    ));
}

/// RED en la quinta versión de I-14: con positionRisk sin la posición recién
/// llenada, la entrada se saltaba el bracket (OCO-SKIP) sin despertar al
/// vigilante, que no la veía hasta la auditoría nominal. Ahora la audita en
/// su disparo siguiente; si positionRisk sigue sin mostrarla entonces, la
/// protección espera igual a la nominal (N34c, abierto).
#[test]
fn i14_n34_la_entrada_que_salta_el_bracket_despierta_al_vigilante() {
    let h = host();
    let salto = tramo(&h, "if!still_open{", "oco_success=true;");
    assert!(
        salto.contains("quantum_arena::protection_health::mark_dirty();"),
        "la audita el vigilante"
    );
    assert!(
        !salto.contains("TPllenódurante"),
        "no hay pierna que haya llenado: el bracket no se colocó"
    );
    // F2A (main) añadió un segundo salto en la lectura de la cantidad del
    // bracket: cada OCO-SKIP despierta al vigilante, y ninguno afirma que
    // llenó un TP.
    let saltos: Vec<usize> = h.match_indices("[OCO-SKIP]").map(|(i, _)| i).collect();
    assert_eq!(saltos.len(), 2, "los dos OCO-SKIP del bracket");
    for i in saltos {
        let salto = tramo(&h[i..], "[OCO-SKIP]", "oco_success=true;");
        assert!(
            salto.contains("quantum_arena::protection_health::mark_dirty();"),
            "cada OCO-SKIP la audita el vigilante"
        );
    }
    assert!(
        !h.contains("TPllenódurante"),
        "no hay pierna que haya llenado: el bracket no se colocó"
    );
}

/// RED antes de I-14: la bandera sucia se limpiaba aunque la purga no
/// hubiera leído el listado, y una lectura fallida sin bandera esperaba a la
/// auditoría nominal de 60 s.
#[test]
fn i14_la_bandera_sucia_solo_se_limpia_con_las_lecturas_completas() {
    let h = host();
    let v = vigilante(&h);
    assert!(v.contains(
        "letnominal=ticks.is_multiple_of(execution_engine::vigilante::CICLOS_NOMINALES);if!nominal&&!reintento_lectura.toca(ticks,events_at_start,dirty){continue;}"
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
    assert!(posiciones.contains(
        "ifletErr(el)=&listado{lecturas.error(\"openAlgoOrders\",el);}lecturas.error(\"positionRisk\",&e);cerrar_lecturas_del_vigilante("
    ));
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
/// el símbolo. RED en la quinta versión: en hedge la racha y la foto eran
/// por símbolo y el lado cubierto borraba la racha del lado desnudo; y sin
/// sembrar la foto, la primera mirada contaba los rechazos de RESTORE.
#[test]
fn i14_n42_la_base_de_rechazos_no_se_borra() {
    let h = host();
    let v = vigilante(&h);
    assert!(!v.contains("naked_rejections.remove("));
    assert!(!v.contains("naked_rejections.retain("));
    assert!(
        v.contains("letmutnaked_rejections=execution_engine::vigilante::BaseDeRechazos::new();")
    );
    assert!(v.contains(
        "letmutnaked_streak:std::collections::HashMap<(String,bool),u32>=std::collections::HashMap::new();"
    ));
    assert!(v.contains("letrejections_now=quantum_arena::protection_health::rejections_of(&p.symbol,is_long);letrejections_prev=naked_rejections.tomar(&p.symbol,is_long,rejections_now);ifrejections_now<=rejections_prev{"));
    assert!(v.contains(".entry(clave.clone())"));
    assert!(v.contains("}else{naked_streak.remove(&clave);naked_rejections.fijar(&p.symbol,is_long,quantum_arena::protection_health::rejections_of(&p.symbol,is_long),);}"));
    assert_eq!(v.matches("naked_streak.remove(&clave);").count(), 3);
    assert_eq!(v.matches("naked_streak.remove(").count(), 3);
    assert!(v.contains("naked_streak.retain(|k,_|abiertas.contains(k));naked_rejections.fijar_los_cerrados(&abiertas,quantum_arena::protection_health::rejections_of,);"));
    assert!(!h.contains("rejections_of(&p.symbol)"));
    // Una posición sin leer sigue abierta: `fijar_los_cerrados` no la toma
    // por cerrada. Y se siembra antes de colocar nada.
    let abierta = v
        .find("letis_long=p.position_amt>0.0;letclave=(p.symbol.clone(),is_long);abiertas.insert(clave.clone());naked_rejections.sembrar(&p.symbol,is_long,quantum_arena::protection_health::rejections_of(&p.symbol,is_long),);")
        .unwrap_or(usize::MAX);
    let lee = v
        .find("letprot=matchensure_position_protected(")
        .unwrap_or(0);
    assert!(abierta < lee);
}

/// La escalada B3.5b de dos huecos de stop con el acumulado real de
/// `protection_health`. Con la base borrada al quedar cubierto (antes de
/// I-14), el segundo hueco empezaba con un paso de racha por los dos
/// rechazos del primero; ahora empieza en cero. Los rechazos del otro lado
/// no cuentan, y un rechazo que llega entre la última mirada y el cierre
/// tampoco cuenta para la reentrada.
#[test]
fn i14_n42_dos_huecos_de_stop_con_el_acumulado_real() {
    use quantum_arena::protection_health::{note_rejection, rejections_of};
    const S: &str = "I14N42USDT";
    let mut base = BaseDeRechazos::new();
    let mut racha = 0u32;
    // La auditoría del vigilante sobre un hueco de stop del largo: siembra
    // antes de colocar, y paso de racha sólo con rechazos nuevos.
    let mirar = |base: &mut BaseDeRechazos| base.sembrar(S, true, rejections_of(S, true));
    let auditar_hueco = |base: &mut BaseDeRechazos, racha: &mut u32| {
        let ahora = rejections_of(S, true);
        let prev = base.tomar(S, true, ahora);
        if ahora > prev {
            *racha += 1;
        }
    };
    // Primer hueco: dos rechazos reales del exchange en dos auditorías.
    mirar(&mut base);
    assert!(note_rejection(
        S,
        true,
        "-2021: Order would immediately trigger."
    ));
    auditar_hueco(&mut base, &mut racha);
    mirar(&mut base);
    assert!(note_rejection(
        S,
        true,
        "-2021: Order would immediately trigger."
    ));
    auditar_hueco(&mut base, &mut racha);
    assert_eq!(racha, 2);
    // El corto del mismo símbolo, cubierto, acumula rechazos de su TP: no
    // son de la racha del largo.
    assert!(note_rejection(S, false, "-4164: Notional too small"));
    mirar(&mut base);
    auditar_hueco(&mut base, &mut racha);
    assert_eq!(racha, 2, "los rechazos del corto no cuentan para el largo");
    // La pierna entra: cubierta. La racha vuelve a cero y la base se fija.
    racha = 0;
    base.fijar(S, true, rejections_of(S, true));
    // Segundo hueco, sin rechazos nuevos (un 5xx no cuenta como rechazo).
    mirar(&mut base);
    assert!(!note_rejection(S, true, "HTTP 503 Service Unavailable"));
    auditar_hueco(&mut base, &mut racha);
    assert_eq!(
        racha, 0,
        "los rechazos del primer hueco no cuentan en el segundo"
    );
    // Un rechazo llega tras la última mirada, la posición cierra, y la
    // auditoría que la ve cerrada fija la base: la reentrada no lo cuenta.
    assert!(note_rejection(
        S,
        true,
        "-2021: Order would immediately trigger."
    ));
    let abiertas: HashSet<(String, bool)> = HashSet::new();
    base.fijar_los_cerrados(&abiertas, rejections_of);
    mirar(&mut base);
    auditar_hueco(&mut base, &mut racha);
    assert_eq!(racha, 0, "el rechazo de antes del cierre no cuenta");
    // El rechazo de la reentrada sí cuenta.
    assert!(note_rejection(
        S,
        true,
        "-2021: Order would immediately trigger."
    ));
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
    let mut cierre = None;
    // La bandera sigue sucia: una auditoría incompleta no la limpia.
    for ciclo in 1..=30u32 {
        let nominal = ciclo % 12 == 0;
        if !nominal && !r.toca(ciclo, 0, true) {
            continue;
        }
        auditados.push(ciclo);
        let c = r.cerrar(ciclo, 0, &lecturas_de(ciclo));
        if c.completa() {
            cierre = Some(c);
            break;
        }
    }
    // 1 falla (espera 1), 2 falla (espera 2), 4 falla (espera 4), 8 frenada
    // (espera 1, sin doblar) y el 9 completa.
    assert_eq!(auditados, vec![1, 2, 4, 8, 9]);
    assert_eq!(
        cierre,
        Some(Cierre::Completa {
            incompletas_previas: 3,
            frenadas_previas: 1
        })
    );
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
