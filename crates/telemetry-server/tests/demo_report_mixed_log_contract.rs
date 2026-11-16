//! Sustituye la aserción de que M1 ya reconoce la línea humana FILL.
//! Incorporar en telemetry-server/src/report_demo_run.rs o como integración
//! telemetry-server/tests/demo_report_mixed_log_contract.rs.
use telemetry_server::report_demo_run::{build_report, parse_income_jsonl, parse_run_log, render_report_table};

const LEGACY: &str = concat!(
    "[ts=1000] DEMO SIGNAL symbol=VETUSDT side=SELL trade_id=701\n",
    "[ts=2000] DEMO ORDER symbol=VETUSDT side=SELL trade_id=701\n",
    "[ts=3000] DEMO FILL symbol=VETUSDT trade_id=701 price=0.00512\n",
    "[ts=4000] DEMO CLOSE symbol=VETUSDT trade_id=701 price=0.005072 reason=FLATTEN by=HOST\n",
);

const MIXED_EXTRA: &str = concat!(
    "Cayendo a DEMO\n",
    "[ts=900] [MODO WARMUP/DEMO ACTIVO]\n",
    "[ts=950] DEMO_TEL event=SYMBOL_SPEC symbol=VETUSDT min_notional=5.0000 tick_size=0.00000100 step_size=1.00000000\n",
    "[ts=1001] DEMO_TEL event=DECISION_MID ts=1000 signal_id=9 symbol=VETUSDT slot=0 dir=SHORT mid=0.005120 bid=0.005119 ask=0.005121\n",
    "[ts=2001] DEMO_TEL event=ORDER_SEND ts=2000 symbol=VETUSDT is_long=false qty=996.000000 client_order_id=cS_a lev=2 route=MakerChase\n",
    "💧 [FILL] VETUSDT SELL LIMIT cS_a @ 0.00512 (acum 996/996 status Filled)\n",
    "DEMO_TEL event=FILL ts=3000 symbol=VETUSDT side=SELL price=0.005120 qty=996.000000 cum_qty=996.000000 client_order_id=cS_a trade_id=701 is_maker=false status=Filled\n",
    "DEMO_TEL event=TRADE_FEE ts=3000 symbol=VETUSDT client_order_id=cS_a fee=0.00203981 fee_asset=USDT is_maker=false\n",
    "[ts=3100] DEMO_TEL event=CAPITAL_UPDATE ts=3099 wallet_balance=1655.8071\n",
    "[ts=3200] DEMO_TEL event=UNREALIZED_PNL ts=3199 total_upnl=-0.001000 open_positions=1\n",
    "[ts=3300] DEMO_TEL event=FUNDING ts=3299 symbol=VETUSDT income=-0.00010000 asset=USDT tran_id=801\n",
    "[ts=4001] DEMO_TEL event=TRADE_CLOSE ts=4000 close_id=10 symbol=VETUSDT slot=0 dir=SHORT entry_price=0.005120 exit_price=0.005072 gross_pnl=0.047808 reason=FLATTEN\n",
    "[ts=4500] DEMO_TEL event=IMMUNE_TRIGGER ts=4499 peak_capital=1655.8092 balance_rest=1655.8071 balance_stream=1655.8071 drop_pct=0.0001 threshold_pct=50.0000 symbol=VETUSDT freno_capital=1655.8071 freno_capital_plano=min(arena.unified_capital,arena.capital_exchange) peak_plano=registrar_pico(arena.pico_capital,freno_capital) balance_rest_plano=puente.equity_del_freno balance_stream_plano=arena.unified_capital\n",
    "[ts=5000] DEMO_TEL event=AUDIT_SUMMARY ts=4999 status=ORDERLY_SHUTDOWN closed_positions=1 cancelled_orders=2\n",
);

const INCOME: &str = concat!(
    "{\"symbol\":\"VETUSDT\",\"income_type\":\"REALIZED_PNL\",\"income\":0.047808,\"time_ms\":4000,\"trade_id\":\"701\",\"tran_id\":802,\"asset\":\"USDT\"}\n",
    "{\"symbol\":\"VETUSDT\",\"income_type\":\"COMMISSION\",\"income\":-0.00406,\"time_ms\":4000,\"trade_id\":\"701\",\"tran_id\":803,\"asset\":\"USDT\"}\n",
);

#[test]
fn m1_distingue_tokens_en_log_mixto_sin_perder_ni_duplicar_operaciones() -> Result<(), String> {
    let parse = |text: &str| parse_run_log(text).map_err(|error| format!("{error:?}"));
    let mixed = format!("{MIXED_EXTRA}{LEGACY}");
    let expected = parse(LEGACY)?;
    let actual = parse(&mixed)?;
    assert_eq!(expected.len(), 1);
    assert_eq!(actual.len(), 1);

    let income = parse_income_jsonl(INCOME);
    let before = build_report(&expected, &income);
    let after = build_report(&actual, &income);
    assert_eq!(before.len(), 1);
    assert_eq!(after.len(), 1);
    assert_eq!(render_report_table(&after), render_report_table(&before));
    assert_eq!(after[0].order_to_fill_ms, Some(1000));
    assert_eq!(after[0].entry_price, Some(0.00512));
    assert_eq!(after[0].exit_price, Some(0.005072));

    let humans = parse("💧 [FILL] VETUSDT SELL LIMIT cS_a @ 0.00512 (acum 996/996 status Filled)\n")?;
    assert!(humans.is_empty());
    let rows = build_report(&humans, &income);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].entry_price, None);
    assert_eq!(rows[0].order_to_fill_ms, None);
    Ok(())
}

#[test]
fn m1_rechaza_demo_estructurado_malformado_con_linea_exacta() {
    for event in [
        "DEMO FILL symbol=VETUSDT trade_id=701 price=0.00512",
        "DEMO FILL symbol=VETUSDT trade_id=701 price=0.00512 [ts=3000]",
        "[ts=x] DEMO FILL symbol=VETUSDT trade_id=701 price=0.00512",
        "[ts=3000] DEMO UNKNOWN symbol=VETUSDT trade_id=701",
    ] {
        let text = format!("[BOOT] listo\n{event}\n");
        assert!(matches!(parse_run_log(&text), Err(error) if error.line == 2), "{event}");
    }
}

// ===========================================================================
// Añadidas por el integrador (QRA-PARSER-M1, "Prueba A nueva", puntos 3 y 4).
// ===========================================================================
use telemetry_server::report_demo_run::{
    build_report_mixto, parse_demo_tel, parse_human_fills, unir_fills, Ambito, VentanaIncome,
};

/// Entrada parcial 400 + 596 bajo el mismo clientOrderId con tradeId distintos,
/// la versión humana intercalada, una repetición idéntica, el mismo tradeId en
/// otro símbolo, un conflicto de qty para el mismo tradeId, un truncado y un
/// cierre externo BUY con ID nuevo. Las DEMO_TEL de stdout van sin prefijo.
const PARCIAL: &str = concat!(
    "[ts=2001] DEMO_TEL event=ORDER_SEND tipo=INTENTO ts=2000 symbol=VETUSDT is_long=false qty=996.000000 client_order_id=cS_p lev=2 route=Ioc { price: 0.00511 }\n",
    "💧 [FILL] VETUSDT SELL LIMIT cS_p @ 0.00512 (acum 400/996 status PartiallyFilled)\n",
    "DEMO_TEL event=FILL ts=3000 symbol=VETUSDT side=SELL price=0.005120 qty=400.000000 cum_qty=400.000000 client_order_id=cS_p trade_id=711 is_maker=false status=PartiallyFilled\n",
    "💧 [FILL] VETUSDT SELL LIMIT cS_p @ 0.005119 (acum 996/996 status Filled)\n",
    "DEMO_TEL event=FILL ts=3005 symbol=VETUSDT side=SELL price=0.005119 qty=596.000000 cum_qty=996.000000 client_order_id=cS_p trade_id=712 is_maker=false status=Filled\n",
    "DEMO_TEL event=FILL ts=3000 symbol=VETUSDT side=SELL price=0.005120 qty=400.000000 cum_qty=400.000000 client_order_id=cS_p trade_id=711 is_maker=false status=PartiallyFilled\n",
    "DEMO_TEL event=FILL ts=3010 symbol=DOGEUSDT side=BUY price=0.150000 qty=10.000000 cum_qty=10.000000 client_order_id=cL_d trade_id=711 is_maker=true status=Filled\n",
    "DEMO_TEL event=FILL ts=3020 symbol=VETUSDT side=SELL price=0.005118 qty=10.000000 cum_qty=10.000000 client_order_id=cS_q trade_id=713 is_maker=false status=PartiallyFilled\n",
    "DEMO_TEL event=FILL ts=3020 symbol=VETUSDT side=SELL price=0.005118 qty=11.000000 cum_qty=11.000000 client_order_id=cS_q trade_id=713 is_maker=false status=PartiallyFilled\n",
    "DEMO_TEL event=FILL ts=3030 symbol=VETUSDT side=SELL price=0.0051\n",
    "[ts=4002] 💧 [FILL] VETUSDT BUY MARKET ext_9 @ 0.005072 (acum 996/996 status Filled)\n",
    "[ts=4003] DEMO_TEL event=FILL ts=4000 symbol=VETUSDT side=BUY price=0.005072 qty=996.000000 cum_qty=996.000000 client_order_id=ext_9 trade_id=720 is_maker=false status=Filled\n",
);
/// El fill nativo del ciclo legacy 701 (el mismo hecho que `DEMO FILL`).
const TEL_701: &str = "DEMO_TEL event=FILL ts=3000 symbol=VETUSDT side=SELL price=0.005120 qty=996.000000 cum_qty=996.000000 client_order_id=cS_a trade_id=701 is_maker=false status=Filled\n";
/// El mismo tradeId 711 en otra cuenta.
const OTRO_AMBITO: &str = "DEMO_TEL event=FILL ts=3000 symbol=VETUSDT side=SELL price=0.009999 qty=1.000000 cum_qty=1.000000 client_order_id=cS_x trade_id=711 is_maker=false status=Filled\n";
const INCOME_P: &str = concat!(
    "{\"symbol\":\"VETUSDT\",\"income_type\":\"COMMISSION\",\"income\":-0.00082,\"time_ms\":3000,\"trade_id\":\"711\",\"tran_id\":901,\"asset\":\"USDT\"}\n",
    "{\"symbol\":\"VETUSDT\",\"income_type\":\"COMMISSION\",\"income\":-0.00122,\"time_ms\":3005,\"trade_id\":\"712\",\"tran_id\":902,\"asset\":\"USDT\"}\n",
    "{\"symbol\":\"VETUSDT\",\"income_type\":\"COMMISSION\",\"income\":-0.00002,\"time_ms\":3020,\"trade_id\":\"713\",\"tran_id\":903,\"asset\":\"USDT\"}\n",
    "{\"symbol\":\"VETUSDT\",\"income_type\":\"REALIZED_PNL\",\"income\":0.047808,\"time_ms\":4000,\"trade_id\":\"720\",\"tran_id\":904,\"asset\":\"USDT\"}\n",
    "{\"symbol\":\"VETUSDT\",\"income_type\":\"COMMISSION\",\"income\":-0.00199,\"time_ms\":4000,\"trade_id\":\"720\",\"tran_id\":905,\"asset\":\"USDT\"}\n",
);
/// Líneas reales 8, 279 y 3171 de `logs/corrida_demo_20261112_094131.log` (humo).
const HUMO: &str = concat!(
    "[ts=1794494493189] 🔥 [MODO WARMUP/DEMO ACTIVO] El sistema inicia en Paper Trading para simulación.\n",
    "💧 [FILL] VETUSDT SELL LIMIT cS_01a1d036a8e978e29a360418357ab707 @ 0.00512 (acum 996/996 status Filled)\n",
    "💧 [FILL] VETUSDT BUY MARKET 01a1d067355e7f419557423e33d72b63 @ 0.005072 (acum 996/996 status Filled)\n",
);
const VENTANA: VentanaIncome = VentanaIncome { desde_ms: 0, hasta_ms: 10_000 };

fn acreditado() -> Ambito {
    Ambito::new("binance_usdm", "A")
}

fn suma_dinero(rows: &[telemetry_server::report_demo_run::DemoOperation]) -> f64 {
    rows.iter().map(|o| [o.gross, o.commission, o.funding].iter().filter(|x| x.is_finite()).sum::<f64>()).sum()
}

#[test]
fn m1_fill_tel_une_por_trade_nativo_y_no_duplica_legacy() -> Result<(), String> {
    let texto = format!("{PARCIAL}{LEGACY}{TEL_701}");
    let a = parse_demo_tel(&texto, "stdout", &acreditado());
    let b = parse_demo_tel(OTRO_AMBITO, "otra_cuenta", &Ambito::new("binance_usdm", "B"));

    // Truncado diagnosticado con su línea; la línea humana se cuenta, no se convierte.
    assert_eq!(a.diagnosticos.len(), 1, "{:?}", a.diagnosticos);
    assert_eq!(a.diagnosticos[0].linea, 10);
    assert_eq!(a.fills_humanos.len(), 3);
    assert!(a.fills_humanos.iter().all(|h| h.texto.starts_with("💧 [FILL]") || h.texto.starts_with("[ts=4002] 💧 [FILL]")));
    assert_eq!(parse_human_fills(&texto).len(), 3);

    // Unión por identidad nativa: repetición deduplicada, conflicto bloqueado.
    let u = unir_fills(&[a.clone(), b.clone()]);
    assert_eq!(u.repetidos, 1);
    assert_eq!(u.conflictos.len(), 1, "{:?}", u.conflictos);
    assert_eq!(u.conflictos[0].linea, 9);
    let k = |s: &str, t: &str, c: &str| ("binance_usdm".to_string(), c.to_string(), s.to_string(), t.to_string());
    assert!(u.claves_en_conflicto.contains(&k("VETUSDT", "713", "A")));
    assert!(!u.fills.contains_key(&k("VETUSDT", "713", "A")));
    let f711 = &u.fills[&k("VETUSDT", "711", "A")];
    let f712 = &u.fills[&k("VETUSDT", "712", "A")];
    assert_eq!((f711.qty, f711.ts_evento, f711.ts_recepcion), (400.0, 3000, None), "ts intrínseco sin prefijo");
    assert_eq!((f712.qty, f712.ts_evento, f712.cum_qty), (596.0, 3005, 996.0));
    assert_eq!(f711.client_order_id, f712.client_order_id, "mismo clientOrderId, dos fills");
    assert_eq!(u.fills[&k("DOGEUSDT", "711", "A")].qty, 10.0, "otro símbolo: otra identidad");
    assert_eq!(u.fills[&k("VETUSDT", "711", "B")].price, 0.009999, "otro ámbito: otra identidad");
    let f720 = &u.fills[&k("VETUSDT", "720", "A")];
    assert_eq!((f720.side.as_str(), f720.client_order_id.as_str(), f720.ts_recepcion), ("BUY", "ext_9", Some(4003)));

    // Informe: sólo los IDs acreditados enriquecen; el legacy no se duplica.
    let mut income = parse_income_jsonl(INCOME_P);
    income.extend(parse_income_jsonl(INCOME));
    let legacy = parse_run_log(&texto).map_err(|e| format!("{e:?}"))?;
    assert_eq!(legacy.len(), 1, "sólo el ciclo legacy 701 es una operación del parser DEMO");
    let informe = build_report_mixto(&legacy, &[a, b], &acreditado(), &income, Some(VENTANA));
    let fila = |t: &str| informe.filas.iter().find(|f| f.op.trade_id == t).unwrap_or_else(|| panic!("fila {t}"));
    let puro = build_report(&parse_run_log(LEGACY).map_err(|e| format!("{e:?}"))?, &income);
    let puro_701 = puro.iter().find(|o| o.trade_id == "701").unwrap();
    assert_eq!(fila("701").join, "LEGACY");
    assert_eq!(&fila("701").op, puro_701, "el ciclo legacy no cambia con su DEMO_TEL al lado");
    for (t, qty, ts) in [("711", 400.0, 3000u128), ("712", 596.0, 3005)] {
        let f = fila(t);
        assert_eq!(f.join, "FILL_NATIVO");
        assert_eq!(f.fill.as_ref().map(|x| x.qty), Some(qty));
        assert_eq!(f.op.fill_ts_ms, Some(ts));
        assert_eq!(f.op.entry_price, None, "un fill no es el precio de toda la posición");
        assert_eq!(f.op.order_to_fill_ms, None, "ORDER_SEND es intento: sin latencia");
        assert_eq!(f.op.exit_price, None);
    }
    assert_eq!(fila("713").join, "N/D", "el conflicto no enriquece");
    assert!(fila("713").fill.is_none());
    let c720 = fila("720");
    assert_eq!(c720.join, "FILL_NATIVO");
    assert_eq!(c720.fill.as_ref().map(|x| x.client_order_id.as_str()), Some("ext_9"));
    assert_eq!((c720.op.entry_price, c720.op.order_to_fill_ms), (None, None), "sin parentesco inventado");
    // Ningún caso agrega dinero: las filas suman exactamente el income.
    let filas: Vec<_> = informe.filas.iter().map(|f| f.op.clone()).collect();
    assert_eq!(suma_dinero(&filas).to_bits(), suma_dinero(&build_report(&legacy, &income)).to_bits());
    let total_income: f64 = income.iter().map(|r| r.income).sum();
    assert!((suma_dinero(&filas) - total_income).abs() < 1e-12);
    assert_eq!(informe.fills_fuera_de_ambito, 1);
    assert_eq!(informe.fills_sin_pareja, vec![k("DOGEUSDT", "711", "A")]);
    assert!(!informe.apto);

    // Humo real sin DEMO_TEL: evidencia incompleta, sin latencia ni monto.
    assert!(parse_run_log(HUMO).map_err(|e| format!("{e:?}"))?.is_empty());
    let h = parse_demo_tel(HUMO, "humo", &acreditado());
    assert!(h.eventos.is_empty() && h.diagnosticos.is_empty());
    assert_eq!(h.fills_humanos.len(), 2);
    assert_eq!(h.fills_humanos[0].texto, HUMO.lines().nth(1).unwrap(), "bytes de la línea humana");
    assert_eq!(h.fills_humanos[1].client_order_id, "01a1d067355e7f419557423e33d72b63");
    assert_eq!(h.fills_humanos[0].ts_recepcion, None, "sin reloj en su formato real");
    let vacio = build_report_mixto(&[], &[h.clone()], &acreditado(), &[], Some(VENTANA));
    assert!(vacio.filas.is_empty() && !vacio.apto);
    let solo_income = parse_income_jsonl(INCOME);
    let con_income = build_report_mixto(&[], &[h], &acreditado(), &solo_income, Some(VENTANA));
    assert_eq!(con_income.filas.len(), 1);
    assert_eq!(con_income.filas[0].join, "N/D");
    assert_eq!((con_income.filas[0].op.order_to_fill_ms, con_income.filas[0].op.entry_price), (None, None));
    let filas_humo: Vec<_> = con_income.filas.iter().map(|f| f.op.clone()).collect();
    assert_eq!(suma_dinero(&filas_humo).to_bits(), suma_dinero(&build_report(&[], &solo_income)).to_bits());
    assert!(!con_income.apto);
    Ok(())
}

struct Corrida {
    codigo: i32,
    salida: String,
    error: String,
}

fn correr_cli(args: &[&str]) -> Corrida {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_demo_report"))
        .args(args)
        .output()
        .expect("lanza demo_report");
    Corrida {
        codigo: out.status.code().unwrap_or(-1),
        salida: String::from_utf8_lossy(&out.stdout).into_owned(),
        error: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

#[test]
fn m1_cli_log_mixto_informa_cobertura_y_veta_apto_si_faltan_joins() -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!("m1_cli_{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let escribir = |nombre: &str, texto: &str| -> Result<String, String> {
        let p = dir.join(nombre);
        std::fs::write(&p, texto).map_err(|e| e.to_string())?;
        Ok(p.to_string_lossy().into_owned())
    };
    let legacy = escribir("legacy.log", &format!("{MIXED_EXTRA}{LEGACY}"))?;
    let income = escribir("income.jsonl", INCOME)?;

    // Soporte completo: la tabla legacy sale igual y el informe es apto.
    let r = correr_cli(&["--log", &legacy, "--income", &income, "--ventana", "0:10000"]);
    assert_eq!(r.codigo, 0, "{}", r.error);
    let esperado = render_report_table(&build_report(
        &parse_run_log(LEGACY).map_err(|e| format!("{e:?}"))?,
        &parse_income_jsonl(INCOME),
    ));
    assert!(r.salida.starts_with(&esperado), "la tabla legacy cambió:\n{}", r.salida);
    // T3 (T3-3, criterio :37): el resultado mixto es NO APTO antes de reemplazarlo.
    assert!(r.salida.contains("ACEPTACION: NO APTO"), "{}", r.salida);
    // MIXED_EXTRA trae el esquema DEMO_TEL de 515e3bf4 (ORDER_SEND sin `tipo`, ...):
    // queda diagnosticado y el informe oficial no es apto aunque el legacy una.
    let solo_legacy = escribir("solo_legacy.log", LEGACY)?;
    let r = correr_cli(&["--log", &solo_legacy, "--income", &income, "--ventana", "0:10000"]);
    assert_eq!(r.codigo, 0, "{}", r.error);
    assert!(r.salida.starts_with(&esperado));
    assert!(r.salida.contains("VETUSDT | 701 | LEGACY | 3000 |"), "{}", r.salida);
    // T3 (T3-3, criterio :37): la expectativa T2 de APTO con ventana sola se corrige: una
    // ventana declarada no certifica funding ni neto (oficiales N/D) y no hay APTO.
    assert!(r.salida.contains("VETUSDT | 701 | LEGACY | 3000 | N/D | N/D | 1000 | N/D | N/D"), "{}", r.salida);
    assert!(r.salida.contains("ACEPTACION: NO APTO"), "{}", r.salida);
    assert!(!r.salida.contains("ACEPTACION: APTO"), "{}", r.salida);

    // Cero sin respaldo: sin ventana, funding y neto N/D y no apto (la tabla legacy igual).
    let r = correr_cli(&["--log", &solo_legacy, "--income", &income]);
    assert!(r.salida.starts_with(&esperado));
    assert!(r.salida.contains("ventana income: N/D"));
    assert!(r.salida.contains("VETUSDT | 701 | LEGACY | 3000 | N/D | N/D | 1000 | N/D | N/D"), "{}", r.salida);
    assert!(r.salida.contains("ACEPTACION: NO APTO"));

    // Evidencia parcial en dos fuentes (stdout y segmento del logger): legible, N/D, no apto.
    let stdout = escribir("demo.stdout.log", PARCIAL)?;
    let segmento = escribir("demo.log", &format!("{MIXED_EXTRA}"))?;
    let mut inc = String::from(INCOME_P);
    inc.push_str(INCOME_P.lines().next().unwrap());
    inc.push('\n');
    inc.push_str("{\"symbol\":\"VETUSDT\",\"income_type\":\"COMMISSION\",\"income\":-0.5,\"time_ms\":99999,\"trade_id\":\"799\",\"tran_id\":999,\"asset\":\"USDT\"}\n");
    let income_p = escribir("income_p.jsonl", &inc)?;
    let r = correr_cli(&[
        "--log", &stdout, "--log", &segmento, "--income", &income_p, "--ventana", "0:10000", "--ambito", "binance_usdm:A",
    ]);
    assert_eq!(r.codigo, 0, "{}", r.error);
    for frase in [
        &format!("fuente {stdout}: 12 lineas"),
        &format!("fuente {segmento}: 14 lineas"),
        &"income fuera de ventana 1, duplicado 1".to_string(),
        &"VETUSDT | 711 | FILL_NATIVO | 3000 | 0.005120 | 400.000000 | N/D |".to_string(),
        &"VETUSDT | 713 | N/D | N/D | N/D | N/D | N/D |".to_string(),
        &"ACEPTACION: NO APTO".to_string(),
        &"sin join nativo ni legacy".to_string(),
    ] {
        assert!(r.salida.contains(frase.as_str()), "falta {frase:?} en:\n{}", r.salida);
    }
    let mixto = r.salida.split_once("COBERTURA M1").map(|(_, m)| m).unwrap_or("");
    assert!(!mixto.contains("| 799 |"), "la fila fuera de ventana no entra al informe mixto:\n{mixto}");

    // Errores del DEMO estructurado: código 2 con la línea exacta.
    let malo = escribir("malo.log", "[BOOT] listo\n[ts=3000] DEMO UNKNOWN symbol=VETUSDT trade_id=701\n")?;
    let r = correr_cli(&["--log", &malo, "--income", &income]);
    assert_eq!(r.codigo, 2);
    assert!(r.error.contains("línea 2"), "{}", r.error);
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}


// ===========================================================================
// DEMO_TEL T3 (qra_t3_criterios.md, T3-2, T3-3, T3-4 y T3-6). Oráculos escritos sobre
// la API pública que ya existía en bb2db3e4 (CLI `demo_report`, `parse_demo_tel`,
// `build_report_mixto`, `render_informe_mixto`): corren igual en la base y en la entrega.
// ===========================================================================
use telemetry_server::report_demo_run::render_informe_mixto;

/// sha256 hex (FIPS 180-4), propio de la prueba: el recibo del fixture se calcula aquí,
/// sin depender de la implementación bajo prueba.
fn sha256_prueba(datos: &[u8]) -> String {
    let k: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut m = datos.to_vec();
    let bits = (datos.len() as u64) * 8;
    m.push(0x80);
    while m.len() % 64 != 56 {
        m.push(0);
    }
    m.extend_from_slice(&bits.to_be_bytes());
    for b in m.chunks(64) {
        let mut w = [0u32; 64];
        for t in 0..16 {
            w[t] = u32::from_be_bytes([b[4 * t], b[4 * t + 1], b[4 * t + 2], b[4 * t + 3]]);
        }
        for t in 16..64 {
            let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
            let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16].wrapping_add(s0).wrapping_add(w[t - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for t in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(k[t]).wrapping_add(w[t]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            v = [t1.wrapping_add(s0.wrapping_add(maj)), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for (x, y) in h.iter_mut().zip(v) {
            *x = x.wrapping_add(y);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}

#[test]
fn t3_sha256_de_la_prueba_es_el_de_fips() {
    assert_eq!(sha256_prueba(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert_eq!(sha256_prueba(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

fn dir_prueba(nombre: &str) -> Result<std::path::PathBuf, String> {
    let dir = std::env::temp_dir().join(format!("t3_{nombre}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

fn escribir_en(dir: &std::path::Path, nombre: &str, texto: &str) -> Result<String, String> {
    let p = dir.join(nombre);
    std::fs::write(&p, texto).map_err(|e| e.to_string())?;
    Ok(p.to_string_lossy().into_owned())
}

/// La sección oficial (después de la tabla LEGACY).
fn seccion_oficial(salida: &str) -> &str {
    salida.split_once("COBERTURA M1").map(|(_, m)| m).unwrap_or("")
}

// --- T3-2 · El fallback tranId solo identifica la fila contable ----------------------------

const FILL_711: &str = "DEMO_TEL event=FILL ts=3000 symbol=VETUSDT side=SELL price=0.005120 qty=400.000000 cum_qty=400.000000 client_order_id=cS_t trade_id=711 is_maker=false status=PartiallyFilled\n";
/// Income sin tradeId (omitido) y con tradeId vacío, ambos con tranId=711.
const INC_FALLBACK_OMITIDO: &str = "{\"symbol\":\"VETUSDT\",\"income_type\":\"COMMISSION\",\"income\":-0.00082,\"time_ms\":3000,\"tran_id\":711,\"asset\":\"USDT\"}\n";
const INC_FALLBACK_VACIO: &str = "{\"symbol\":\"VETUSDT\",\"income_type\":\"COMMISSION\",\"income\":-0.00082,\"time_ms\":3000,\"trade_id\":\"\",\"tran_id\":711,\"asset\":\"USDT\"}\n";
/// Gemelo positivo: tradeId explícito 711 y tranId distinto.
const INC_EXPLICITO: &str = "{\"symbol\":\"VETUSDT\",\"income_type\":\"COMMISSION\",\"income\":-0.00082,\"time_ms\":3000,\"trade_id\":\"711\",\"tran_id\":905,\"asset\":\"USDT\"}\n";
/// Movimiento distinto con fallback del mismo número (funding firmado sin tradeId).
const INC_FUNDING_711: &str = "{\"symbol\":\"VETUSDT\",\"income_type\":\"FUNDING_FEE\",\"income\":-0.0001,\"time_ms\":3300,\"tran_id\":711,\"asset\":\"USDT\"}\n";

#[test]
fn t3_join_nativo_exige_trade_id_explicito_aunque_tran_id_coincida() {
    let a = || Ambito::new("binance_usdm", "A");
    let k = |c: &str, s: &str, t: &str| ("binance_usdm".to_string(), c.to_string(), s.to_string(), t.to_string());
    let tel = parse_demo_tel(FILL_711, "stdout", &a());
    assert_eq!((tel.eventos.len(), tel.diagnosticos.len()), (1, 0), "control: el fill nativo es válido");

    // Negativo, en sus dos variantes (tradeId omitido y vacío).
    for (variante, inc) in [("omitido", INC_FALLBACK_OMITIDO), ("vacío", INC_FALLBACK_VACIO)] {
        let income = parse_income_jsonl(inc);
        assert_eq!(income.len(), 1, "{variante}");
        let inf = build_report_mixto(&[], &[tel.clone()], &a(), &income, Some(VENTANA));
        assert_eq!(inf.filas.len(), 1, "{variante}: la fila contable se conserva");
        let f = &inf.filas[0];
        assert_eq!(f.op.trade_id, "711", "{variante}: identidad transaccional de la fila");
        assert_eq!(f.op.commission.to_bits(), (-0.00082f64).to_bits(), "{variante}: importe exacto");
        assert_eq!(f.join, "N/D", "{variante}: el tranId no es identidad nativa");
        assert!(f.fill.is_none(), "{variante}");
        assert_eq!((f.op.fill_ts_ms, f.op.entry_price, f.op.order_to_fill_ms), (None, None, None), "{variante}: nada prestado del fill");
        assert!(f.op.side.is_empty(), "{variante}: sin lado prestado");
        assert_eq!(inf.fills_nativos, 1, "{variante}");
        assert_eq!(inf.fills_sin_pareja, vec![k("A", "VETUSDT", "711")], "{variante}: el fill sigue sin pareja, una vez");
        assert!(!inf.apto, "{variante}");
        let r = render_informe_mixto(&inf);
        assert!(
            r.contains("VETUSDT | 711 | N/D | N/D | N/D | N/D | N/D | N/D | N/D | TRAN_ID | COMMISSION=-0.00082 USDT\n"),
            "{variante}: origen de la clave y fila exacta:\n{r}"
        );
        assert!(r.contains("sin pareja 1)"), "{variante}:\n{r}");
    }

    // Gemelo positivo: tradeId explícito 711, tranId distinto.
    let income = parse_income_jsonl(INC_EXPLICITO);
    let inf = build_report_mixto(&[], &[tel.clone()], &a(), &income, Some(VENTANA));
    assert_eq!(inf.filas.len(), 1);
    let f = &inf.filas[0];
    assert_eq!(f.join, "FILL_NATIVO");
    let fill = f.fill.as_ref().expect("fill del join");
    assert_eq!((fill.qty, fill.price, fill.ts_evento, fill.side.as_str()), (400.0, 0.00512, 3000, "SELL"));
    assert_eq!((f.op.fill_ts_ms, f.op.side.as_str()), (Some(3000), "SELL"));
    assert!(inf.fills_sin_pareja.is_empty());
    let r = render_informe_mixto(&inf);
    assert!(r.contains("VETUSDT | 711 | FILL_NATIVO | 3000 | 0.005120 | 400.000000 | N/D | N/D | N/D | TRADE_ID | COMMISSION=-0.00082 USDT\n"), "{r}");

    // Otro símbolo, otra cuenta u otro venue: no enlaza.
    for (lectura, que) in [
        (parse_demo_tel(&FILL_711.replace("VETUSDT", "DOGEUSDT"), "stdout", &a()), "símbolo"),
        (parse_demo_tel(FILL_711, "otra", &Ambito::new("binance_usdm", "B")), "cuenta"),
        (parse_demo_tel(FILL_711, "otra", &Ambito::new("bybit", "A")), "venue"),
    ] {
        let inf = build_report_mixto(&[], &[lectura], &a(), &income, Some(VENTANA));
        assert_eq!(inf.filas.len(), 1, "{que}");
        assert_eq!(inf.filas[0].join, "N/D", "otro {que}: no enlaza");
        assert!(inf.filas[0].fill.is_none(), "{que}");
    }

    // Coexistencia: comisión con tradeId explícito 711 y funding sin tradeId con tranId=711.
    let income = parse_income_jsonl(&format!("{INC_EXPLICITO}{INC_FUNDING_711}"));
    let inf = build_report_mixto(&[], &[tel.clone()], &a(), &income, Some(VENTANA));
    assert_eq!(inf.filas.len(), 2, "dos procedencias, dos filas: {:?}", inf.filas);
    let nativa: Vec<_> = inf.filas.iter().filter(|f| f.join == "FILL_NATIVO").collect();
    let nd: Vec<_> = inf.filas.iter().filter(|f| f.join == "N/D").collect();
    assert_eq!((nativa.len(), nd.len()), (1, 1));
    assert_eq!(nativa[0].op.commission.to_bits(), (-0.00082f64).to_bits());
    assert_eq!(nativa[0].op.funding, 0.0, "el funding sin tradeId no se imputa al fill");
    assert_eq!(nd[0].op.funding.to_bits(), (-0.0001f64).to_bits(), "el funding conserva su identidad e importe");
    assert!(nd[0].op.commission.is_nan());
    assert!(nd[0].fill.is_none() && nd[0].op.fill_ts_ms.is_none());
    assert!(inf.fills_sin_pareja.is_empty());
    let r = render_informe_mixto(&inf);
    assert!(r.contains("VETUSDT | 711 | N/D | N/D | N/D | N/D | N/D | N/D | N/D | TRAN_ID | FUNDING_FEE=-0.0001 USDT\n"), "{r}");
    // Ningún caso agrega dinero.
    let filas: Vec<_> = inf.filas.iter().map(|f| f.op.clone()).collect();
    let total: f64 = income.iter().map(|r| r.income).sum();
    assert!((suma_dinero(&filas) - total).abs() < 1e-15);
}

// --- T3-3 · Cobertura financiera verificable y compuerta de CLI ---------------------------

const INC_T33: &str = concat!(
    "{\"symbol\":\"VETUSDT\",\"income_type\":\"REALIZED_PNL\",\"income\":0.047808,\"time_ms\":4000,\"trade_id\":\"701\",\"tran_id\":802,\"asset\":\"USDT\"}\n",
    "{\"symbol\":\"VETUSDT\",\"income_type\":\"COMMISSION\",\"income\":-0.00406,\"time_ms\":4000,\"trade_id\":\"701\",\"tran_id\":803,\"asset\":\"USDT\"}\n",
    "{\"symbol\":\"VETUSDT\",\"income_type\":\"COMMISSION\",\"income\":-0.5,\"time_ms\":99999,\"trade_id\":\"799\",\"tran_id\":999,\"asset\":\"USDT\"}\n",
);

#[derive(Clone)]
struct Recibo {
    sha: String,
    venue: Option<&'static str>,
    cuenta: Option<&'static str>,
    activo: Option<&'static str>,
    paginas: Vec<(u64, u64, Option<&'static str>)>,
    cubierto: (u64, u64),
    extra: &'static str,
}

impl Recibo {
    fn json(&self) -> String {
        let campo = |k: &str, v: Option<&str>| v.map(|v| format!("\"{k}\":\"{v}\",")).unwrap_or_default();
        let paginas: Vec<String> = self
            .paginas
            .iter()
            .map(|(n, f, c)| {
                let cont = c.map(|c| format!("\"{c}\"")).unwrap_or_else(|| "null".to_string());
                format!("{{\"pagina\":{n},\"filas\":{f},\"continuacion\":{cont}}}")
            })
            .collect();
        format!(
            "{{\"archivo_sha256\":\"{}\",{}{}{}\"tipos\":[\"REALIZED_PNL\",\"COMMISSION\",\"FUNDING_FEE\"],\"intervalo_solicitado\":{{\"desde_ms\":0,\"hasta_ms\":10000}},\"intervalo_cubierto\":{{\"desde_ms\":{},\"hasta_ms\":{}}},\"paginas\":[{}],\"termino\":true{}}}",
            self.sha,
            campo("venue", self.venue),
            campo("cuenta", self.cuenta),
            campo("activo", self.activo),
            self.cubierto.0,
            self.cubierto.1,
            paginas.join(","),
            self.extra
        )
    }
}

fn recibo_completo(income: &str) -> Recibo {
    Recibo {
        sha: sha256_prueba(income.as_bytes()),
        venue: Some("binance_usdm"),
        cuenta: Some("A"),
        activo: Some("USDT"),
        paginas: vec![(1, income.lines().filter(|l| !l.trim().is_empty()).count() as u64, None)],
        cubierto: (0, 10_000),
        extra: "",
    }
}

#[test]
fn t3_cli_ventana_sola_no_certifica_funding_ni_neto() -> Result<(), String> {
    let dir = dir_prueba("t33")?;
    let log = escribir_en(&dir, "legacy.log", LEGACY)?;
    let legacy_de = |inc: &str| -> Result<String, String> {
        Ok(render_report_table(&build_report(&parse_run_log(LEGACY).map_err(|e| format!("{e:?}"))?, &parse_income_jsonl(inc))))
    };
    let fila_701 = "VETUSDT | 701 | LEGACY | 3000 | N/D | N/D | 1000 | N/D | N/D | TRADE_ID | COMMISSION=-0.00406 USDT; REALIZED_PNL=0.047808 USDT";
    let control = |nombre: &str, r: &Corrida, legacy: &str| -> String {
        assert_eq!(r.codigo, 0, "{nombre}: {}", r.error);
        assert!(r.salida.starts_with(legacy), "{nombre}: la tabla LEGACY cambió:\n{}", r.salida);
        let oficial = seccion_oficial(&r.salida);
        assert!(!oficial.is_empty(), "{nombre}: sin sección oficial delimitada:\n{}", r.salida);
        assert!(oficial.contains("ventana income: 0..10000 ms"), "{nombre}");
        assert!(
            oficial.contains("VETUSDT | 701 | LEGACY | 3000 | N/D | N/D | 1000 | N/D | N/D |"),
            "{nombre}: join verificado y funding/neto oficiales N/D:\n{oficial}"
        );
        assert!(oficial.contains("REALIZED_PNL=0.047808 USDT"), "{nombre}: el bruto observado se muestra como tal:\n{oficial}");
        assert!(oficial.contains("income fuera de ventana 1,"), "{nombre}: la fila exterior se contabiliza:\n{oficial}");
        assert!(!oficial.contains("| 799 |"), "{nombre}: la fila exterior se filtra:\n{oficial}");
        assert!(oficial.contains("ACEPTACION: NO APTO"), "{nombre}:\n{oficial}");
        assert!(!oficial.contains("ACEPTACION: APTO"), "{nombre}:\n{oficial}");
        oficial.to_string()
    };
    let correr = |nombre: &str, income: &str, recibo: Option<String>, extra: &[&str]| -> Result<Corrida, String> {
        let inc = escribir_en(&dir, &format!("{nombre}.jsonl"), income)?;
        let mut args: Vec<String> = vec!["--log".into(), log.clone(), "--income".into(), inc, "--ventana".into(), "0:10000".into()];
        if let Some(rc) = recibo {
            args.push("--recibo-income".into());
            args.push(escribir_en(&dir, &format!("{nombre}.recibo.json"), &rc)?);
        }
        args.extend(extra.iter().map(|s| s.to_string()));
        let a: Vec<&str> = args.iter().map(String::as_str).collect();
        Ok(correr_cli(&a))
    };
    let alcance: &[&str] = &["--ambito", "binance_usdm:A", "--activo", "USDT"];
    let esperado_legacy = legacy_de(INC_T33)?;

    // (0) Ventana sola: el caso que en T2 daba funding 0, neto finito y APTO.
    let r = correr("ventana_sola", INC_T33, None, &[])?;
    let of = control("ventana_sola", &r, &esperado_legacy);
    assert!(!of.contains("0.000000"), "ventana sola: funding cero sin respaldo:\n{of}");

    // (1) Gemelo con ventana y metadatos completos y coherentes, también anunciando
    // cobertura sin respaldo: N/D y NO APTO, con motivo de cobertura.
    for (nombre, extra) in [("gemelo", ""), ("anuncia_cobertura", ",\"cobertura\":\"COMPLETA\",\"certificado\":true")] {
        let mut rc = recibo_completo(INC_T33);
        rc.extra = extra;
        let r = correr(nombre, INC_T33, Some(rc.json()), alcance)?;
        let of = control(nombre, &r, &esperado_legacy);
        assert!(of.contains(fila_701), "{nombre}: fila exacta con join y observados:\n{of}");
        assert!(of.contains("motivo: cobertura: el recibo no acredita la procedencia"), "{nombre}:\n{of}");
        for categoria in ["motivo: ámbito:", "motivo: activo:", "integridad:", "motivo: paginación:", "intervalo incompleto"] {
            assert!(!of.contains(categoria), "{nombre}: motivo ajeno {categoria}:\n{of}");
        }
    }

    // (2) Una carencia por variante, con su motivo específico.
    let sin_asset = INC_T33.replace(",\"tran_id\":803,\"asset\":\"USDT\"", ",\"tran_id\":803");
    let asset_vacio = INC_T33.replace(",\"tran_id\":803,\"asset\":\"USDT\"", ",\"tran_id\":803,\"asset\":\"\"");
    let asset_tipo = INC_T33.replace(",\"tran_id\":803,\"asset\":\"USDT\"", ",\"tran_id\":803,\"asset\":5");
    let truncado = format!("{INC_T33}{{\"symbol\":\"VETUSDT\",\"income_ty\n");
    let base = || recibo_completo(INC_T33);
    let con = |f: &dyn Fn(&mut Recibo)| {
        let mut rc = base();
        f(&mut rc);
        Some(rc.json())
    };
    let casos: Vec<(&str, String, Option<String>, &[&str], &str)> = vec![
        ("ambito_ausente", INC_T33.into(), Some(base().json()), &["--activo", "USDT"], "motivo: ámbito: venue/cuenta del alcance no declarados"),
        ("cuenta_discordante", INC_T33.into(), con(&|r| r.cuenta = Some("B")), alcance, "motivo: ámbito: el recibo declara cuenta=B"),
        ("venue_discordante", INC_T33.into(), con(&|r| r.venue = Some("bybit")), alcance, "motivo: ámbito: el recibo declara venue=bybit"),
        ("cuenta_ausente_en_recibo", INC_T33.into(), con(&|r| r.cuenta = None), alcance, "motivo: ámbito: el recibo no declara cuenta"),
        ("activo_no_declarado", INC_T33.into(), Some(base().json()), &["--ambito", "binance_usdm:A"], "motivo: activo: numerario del alcance no declarado"),
        ("activo_fuera_de_alcance", INC_T33.into(), con(&|r| r.activo = Some("BNB")), alcance, "motivo: activo: el recibo declara activo=BNB"),
        ("asset_ausente", sin_asset.clone(), Some(recibo_completo(&sin_asset).json()), alcance, "asset AUSENTE; el sufijo del símbolo no acredita la moneda"),
        ("asset_vacio", asset_vacio.clone(), Some(recibo_completo(&asset_vacio).json()), alcance, "asset VACIO; el sufijo del símbolo no acredita la moneda"),
        ("asset_tipo_invalido", asset_tipo.clone(), Some(recibo_completo(&asset_tipo).json()), alcance, "asset INVALIDO; el sufijo del símbolo no acredita la moneda"),
        ("sin_recibo", INC_T33.into(), None, alcance, "motivo: cobertura: sin recibo de cobertura del income"),
        ("hash_discordante", INC_T33.into(), con(&|r| r.sha = sha256_prueba(b"otro archivo")), alcance, "motivo: integridad: el sha256 del recibo"),
        ("jsonl_truncado", truncado.clone(), Some(recibo_completo(&truncado).json()), alcance, "integridad: la línea no es un objeto JSON (truncada o corrupta)"),
        ("pagina_faltante", INC_T33.into(), con(&|r| r.paginas = vec![(1, 2, Some("c1")), (3, 1, None)]), alcance, "motivo: paginación: falta la página 2"),
        ("intervalo_incompleto", INC_T33.into(), con(&|r| r.cubierto = (0, 5_000)), alcance, "intervalo_cubierto 0..5000 no cubre la ventana 0..10000: intervalo incompleto"),
    ];
    for (nombre, inc, rc, extra, motivo) in &casos {
        let r = correr(nombre, inc, rc.clone(), extra)?;
        let of = control(nombre, &r, &legacy_de(inc)?);
        assert!(of.contains(motivo), "{nombre}: falta el motivo {motivo:?}:\n{of}");
    }
    // Un activo inválido se muestra con su marca, nunca como USDT por el sufijo del símbolo.
    let r = correr("asset_ausente_fila", &sin_asset, Some(recibo_completo(&sin_asset).json()), alcance)?;
    let of = seccion_oficial(&r.salida);
    assert!(of.contains("COMMISSION=-0.00406 AUSENTE; REALIZED_PNL=0.047808 USDT (INCOMPLETA)"), "{of}");
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

// --- T3-4 · Los activos no se suman entre sí -----------------------------------------------

const FILL_711_ENTERO: &str = "DEMO_TEL event=FILL ts=3000 symbol=VETUSDT side=SELL price=0.005120 qty=996.000000 cum_qty=996.000000 client_order_id=cS_m trade_id=711 is_maker=false status=Filled\n";
const INC_MEZCLA: &str = concat!(
    "{\"symbol\":\"VETUSDT\",\"income_type\":\"REALIZED_PNL\",\"income\":10,\"time_ms\":4000,\"trade_id\":\"711\",\"tran_id\":1001,\"asset\":\"USDT\"}\n",
    "{\"symbol\":\"VETUSDT\",\"income_type\":\"COMMISSION\",\"income\":-1,\"time_ms\":3000,\"trade_id\":\"711\",\"tran_id\":1002,\"asset\":\"USDT\"}\n",
    "{\"symbol\":\"VETUSDT\",\"income_type\":\"COMMISSION\",\"income\":-0.01,\"time_ms\":3000,\"trade_id\":\"711\",\"tran_id\":1003,\"asset\":\"BNB\"}\n",
    "{\"symbol\":\"VETUSDT\",\"income_type\":\"COMMISSION\",\"income\":-0.01,\"time_ms\":3000,\"trade_id\":\"711\",\"tran_id\":1003,\"asset\":\"BNB\"}\n",
);
const INC_SOLO_USDT: &str = concat!(
    "{\"symbol\":\"VETUSDT\",\"income_type\":\"REALIZED_PNL\",\"income\":10,\"time_ms\":4000,\"trade_id\":\"711\",\"tran_id\":1001,\"asset\":\"USDT\"}\n",
    "{\"symbol\":\"VETUSDT\",\"income_type\":\"COMMISSION\",\"income\":-1,\"time_ms\":3000,\"trade_id\":\"711\",\"tran_id\":1002,\"asset\":\"USDT\"}\n",
    "{\"symbol\":\"VETUSDT\",\"income_type\":\"FUNDING_FEE\",\"income\":0.25,\"time_ms\":3500,\"tran_id\":1004,\"asset\":\"USDT\"}\n",
);

#[test]
fn t3_income_no_mezcla_activos_en_la_misma_operacion() -> Result<(), String> {
    let dir = dir_prueba("t34")?;
    let log = escribir_en(&dir, "fill.log", FILL_711_ENTERO)?;
    let alcance = ["--ventana", "0:10000", "--ambito", "binance_usdm:A", "--activo", "USDT"];
    let correr = |nombre: &str, inc: &str| -> Result<Corrida, String> {
        let p = escribir_en(&dir, nombre, inc)?;
        let mut a = vec!["--log", log.as_str(), "--income", p.as_str()];
        a.extend(alcance);
        Ok(correr_cli(&a))
    };

    // Mezcla: +10 USDT, -1 USDT, -0,01 BNB y el replay idéntico de la comisión BNB.
    let r = correr("mezcla.jsonl", INC_MEZCLA)?;
    assert_eq!(r.codigo, 0, "{}", r.error);
    let legacy = render_report_table(&build_report(&[], &parse_income_jsonl(INC_MEZCLA)));
    assert!(r.salida.starts_with(&legacy), "la tabla LEGACY conserva su comportamiento:\n{}", r.salida);
    let of = seccion_oficial(&r.salida);
    assert!(of.contains("duplicado 1,"), "un replay contado:\n{of}");
    assert!(
        of.contains("VETUSDT | 711 | FILL_NATIVO | 3000 | 0.005120 | 996.000000 | N/D | N/D | N/D | TRADE_ID | COMMISSION=-0.01 BNB; COMMISSION=-1 USDT; REALIZED_PNL=10 USDT (INCOMPLETA)\n"),
        "tres movimientos, importes y unidades exactos, operación incompleta:\n{of}"
    );
    assert!(of.contains("motivo: activo: VETUSDT trade_id=711: importes fuera del numerario USDT (COMMISSION -0.01 BNB)"), "{of}");
    assert!(of.contains("LEGACY sin autoridad oficial: VETUSDT 711: la tabla LEGACY suma nominalmente BNB + USDT"), "{of}");
    for mixto in ["-1.01", "8.99", "-1.02", "8.98"] {
        assert!(!of.contains(mixto), "total mixto {mixto} en la salida oficial:\n{of}");
    }
    assert!(of.contains("ACEPTACION: NO APTO"));
    // La misma lectura por la API: un solo grupo, el replay contado, neto oficial N/D.
    let tel = parse_demo_tel(FILL_711_ENTERO, "stdout", &acreditado());
    let inf = build_report_mixto(&[], &[tel], &acreditado(), &parse_income_jsonl(INC_MEZCLA), Some(VENTANA));
    assert_eq!((inf.filas.len(), inf.income_duplicado), (1, 1));
    assert_eq!((inf.filas[0].funding_acreditado, inf.filas[0].neto_acreditado), (None, None));

    // Gemelo solo USDT: comisión observada -1 USDT exacta; sin cobertura acreditada el neto
    // oficial sigue N/D. El funding firmado conserva signo e identidad propia.
    let r = correr("usdt.jsonl", INC_SOLO_USDT)?;
    assert_eq!(r.codigo, 0, "{}", r.error);
    let of = seccion_oficial(&r.salida);
    assert!(
        of.contains("VETUSDT | 711 | FILL_NATIVO | 3000 | 0.005120 | 996.000000 | N/D | N/D | N/D | TRADE_ID | COMMISSION=-1 USDT; REALIZED_PNL=10 USDT\n"),
        "{of}"
    );
    assert!(of.contains("VETUSDT | 1004 | N/D | N/D | N/D | N/D | N/D | N/D | N/D | TRAN_ID | FUNDING_FEE=0.25 USDT\n"), "{of}");
    assert!(!of.contains("INCOMPLETA") && !of.contains("LEGACY sin autoridad"), "{of}");
    assert!(!of.contains("9.000000") && !of.contains("| 9 |"), "neto 9 sin cobertura acreditada:\n{of}");
    assert!(of.contains("motivo: cobertura: sin recibo de cobertura del income"), "{of}");
    assert!(of.contains("duplicado 0,"));
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

// --- T3-6 · Dominios de DEMO_TEL validados al leer -----------------------------------------

/// Una línea válida completa por emisor (esquema de la entrega T2).
const VALIDAS: &[(&str, &str)] = &[
    ("FILL", "DEMO_TEL event=FILL ts=3000 symbol=VETUSDT side=SELL price=0.005120 qty=400.000000 cum_qty=400.000000 client_order_id=cS_p trade_id=711 is_maker=false status=PartiallyFilled"),
    ("TRADE_FEE", "DEMO_TEL event=TRADE_FEE ts=3000 symbol=VETUSDT client_order_id=cS_p trade_id=711 fee=0.00082000 fee_asset=USDT is_maker=false alcance=ENTRADA"),
    ("ORDER_SEND", "DEMO_TEL event=ORDER_SEND tipo=INTENTO ts=2000 symbol=VETUSDT is_long=false qty=996.000000 client_order_id=cS_p lev=2 route=Ioc { price: 0.00511 }"),
    ("ORDER_RESULT", "DEMO_TEL event=ORDER_RESULT ts=2001 symbol=VETUSDT client_order_id=cS_p resultado=ENTREGADA"),
    ("IMMUNE_TRIGGER", "DEMO_TEL event=IMMUNE_TRIGGER ts=4499 motivo=DRAWDOWN alcance=CUENTA freno_capital=1655.8100 freno_capital_plano=FRENO peak_capital=7299.8800 peak_plano=PICO equity_del_freno=AUSENTE equity_del_freno_plano=PUENTE caida_capital_pct=77.3172 tope_capital_pct=50.0000 operator_lock=false insolvent=false dd_breach=true lat_breach=false latency_strikes=0 revision=N/D lecturas_posteriores=ninguna"),
];

/// (tipo, campo, valor válido actual, valor mutante).
const MUTANTES: &[(&str, &str, &str, &str)] = &[
    ("FILL", "side", "SELL", "XYZ"),
    ("FILL", "side", "SELL", "sell"),
    ("FILL", "side", "SELL", "1"),
    ("FILL", "status", "PartiallyFilled", "Arbitrario"),
    ("FILL", "status", "PartiallyFilled", "Canceled"),
    ("FILL", "status", "PartiallyFilled", "New"),
    ("FILL", "status", "PartiallyFilled", "Expired"),
    ("FILL", "status", "PartiallyFilled", "Unknown"),
    ("FILL", "is_maker", "false", "maybe"),
    ("FILL", "is_maker", "false", "1"),
    ("FILL", "is_maker", "false", "AUSENTE"),
    ("TRADE_FEE", "is_maker", "false", "0"),
    ("TRADE_FEE", "is_maker", "false", "AUSENTE"),
    ("ORDER_SEND", "is_long", "false", "yes"),
    ("ORDER_SEND", "is_long", "false", "1"),
    ("ORDER_SEND", "is_long", "false", "AUSENTE"),
    ("ORDER_SEND", "tipo", "INTENTO", "ENVIO"),
    ("IMMUNE_TRIGGER", "operator_lock", "false", "1"),
    ("IMMUNE_TRIGGER", "insolvent", "false", "AUSENTE"),
    ("IMMUNE_TRIGGER", "dd_breach", "true", "si"),
    ("IMMUNE_TRIGGER", "lat_breach", "false", "0"),
    ("ORDER_RESULT", "resultado", "ENTREGADA", "DESCONOCIDA"),
];

/// Variantes válidas del dominio (siguen siendo eventos con ese valor).
const VALIDOS_ALTERNOS: &[(&str, &str, &str, &str)] = &[
    ("FILL", "side", "SELL", "BUY"),
    ("FILL", "status", "PartiallyFilled", "Filled"),
    ("FILL", "is_maker", "false", "true"),
    ("TRADE_FEE", "is_maker", "false", "true"),
    ("ORDER_SEND", "is_long", "false", "true"),
    ("ORDER_RESULT", "resultado", "ENTREGADA", "RECHAZADA"),
    ("ORDER_RESULT", "resultado", "ENTREGADA", "AMBIGUA"),
    ("IMMUNE_TRIGGER", "operator_lock", "false", "true"),
];

fn linea_valida(tipo: &str) -> &'static str {
    VALIDAS.iter().find(|(t, _)| *t == tipo).map(|(_, l)| *l).expect("tipo con línea válida")
}

fn mutar(tipo: &str, k: &str, de: &str, a: &str) -> String {
    let linea = linea_valida(tipo);
    let viejo = format!(" {k}={de}");
    assert_eq!(linea.matches(&viejo).count(), 1, "{tipo}: {viejo}");
    linea.replacen(&viejo, &format!(" {k}={a}"), 1)
}

#[test]
fn t3_demo_tel_rechaza_dominios_invalidos_y_conserva_validos() {
    for prefijo in ["", "[ts=1784000000000] "] {
        let fuente = if prefijo.is_empty() { "stdout" } else { "segmento" };
        // Por cada mutante, su gemelo válido en la línea anterior.
        let mut texto = String::new();
        let mut esperados = Vec::new();
        for (tipo, k, de, a) in MUTANTES {
            texto.push_str(&format!("{prefijo}{}\n", linea_valida(tipo)));
            texto.push_str(&format!("{prefijo}{}\n", mutar(tipo, k, de, a)));
            esperados.push((texto.lines().count(), *tipo, *k, *a));
        }
        for (tipo, k, de, a) in VALIDOS_ALTERNOS {
            texto.push_str(&format!("{prefijo}{}\n", mutar(tipo, k, de, a)));
        }
        texto.push_str(&format!("{prefijo}DEMO_TEL event=FUTURO_X ts=1 a=1\n"));
        let l = parse_demo_tel(&texto, fuente, &acreditado());
        // Conteos exactos: gemelos y alternos son eventos; cada mutante, un diagnóstico.
        assert_eq!(l.eventos.len(), MUTANTES.len() + VALIDOS_ALTERNOS.len(), "{fuente}: {:?}", l.diagnosticos);
        assert_eq!(l.diagnosticos.len(), MUTANTES.len(), "{fuente}: {:#?}", l.diagnosticos);
        assert_eq!(l.tipos_desconocidos, vec![(texto.lines().count(), "FUTURO_X".to_string())], "{fuente}: futuros aparte");
        for ((linea, tipo, k, v), d) in esperados.iter().zip(&l.diagnosticos) {
            assert_eq!((d.fuente.as_str(), d.linea), (fuente, *linea), "{tipo} {k}={v}");
            assert!(d.mensaje.starts_with(tipo), "{tipo}: {}", d.mensaje);
            assert!(d.mensaje.contains(&format!("{k}={v}")), "diagnóstico sin campo y valor: {}", d.mensaje);
        }
        let unknown = esperados.iter().position(|e| e.3 == "Unknown").unwrap();
        assert!(l.diagnosticos[unknown].mensaje.contains("evidencia insuficiente"), "{}", l.diagnosticos[unknown].mensaje);
        for estado in ["Canceled", "New", "Expired"] {
            let i = esperados.iter().position(|e| e.3 == estado).unwrap();
            assert!(l.diagnosticos[i].mensaje.contains("no acredita"), "{}", l.diagnosticos[i].mensaje);
        }
        // Cada gemelo válido conserva identidad, campos y estado exactos.
        for (i, (tipo, _, _, _)) in MUTANTES.iter().enumerate() {
            let e = &l.eventos[i];
            assert_eq!((e.tipo.as_str(), e.linea), (*tipo, 2 * i + 1));
            assert_eq!(e.ts_recepcion, if prefijo.is_empty() { None } else { Some(1_784_000_000_000) });
            for par in linea_valida(tipo).split(' ').skip(2) {
                if let Some((k, v)) = par.split_once('=') {
                    if k != "route" {
                        assert_eq!(e.campos.get(k).map(String::as_str), Some(v), "{tipo}.{k}");
                    }
                }
            }
        }
        for (j, (tipo, k, _, a)) in VALIDOS_ALTERNOS.iter().enumerate() {
            let e = &l.eventos[MUTANTES.len() + j];
            assert_eq!((e.tipo.as_str(), e.campos.get(*k).map(String::as_str)), (*tipo, Some(*a)));
        }
        // LEGACY intacto: estas líneas no son del parser DEMO.
        assert_eq!(parse_run_log(&texto).map(|v| v.len()), Ok(0));
    }

    // Un FILL mutante queda fuera del join y deja el informe no apto; su gemelo válido une.
    let malo = mutar("FILL", "side", "SELL", "XYZ").replace("trade_id=711", "trade_id=712");
    let texto = format!("{}\n{malo}\n", linea_valida("FILL"));
    let l = parse_demo_tel(&texto, "stdout", &acreditado());
    let income = parse_income_jsonl(concat!(
        "{\"symbol\":\"VETUSDT\",\"income_type\":\"COMMISSION\",\"income\":-0.00082,\"time_ms\":3000,\"trade_id\":\"711\",\"tran_id\":901,\"asset\":\"USDT\"}\n",
        "{\"symbol\":\"VETUSDT\",\"income_type\":\"COMMISSION\",\"income\":-0.00122,\"time_ms\":3005,\"trade_id\":\"712\",\"tran_id\":902,\"asset\":\"USDT\"}\n",
    ));
    let inf = build_report_mixto(&[], &[l], &acreditado(), &income, Some(VENTANA));
    let fila = |t: &str| inf.filas.iter().find(|f| f.op.trade_id == t).unwrap();
    assert_eq!(fila("711").join, "FILL_NATIVO", "control positivo: el gemelo válido une");
    assert_eq!(fila("712").join, "N/D", "el mutante no une");
    assert_eq!(inf.diagnosticos.len(), 1);
    assert_eq!(inf.diagnosticos[0].linea, 2);
    assert!(!inf.apto);
    let r = render_informe_mixto(&inf);
    assert!(r.contains("diagnostico stdout:2: FILL: dominio: campo side=XYZ"), "{r}");
}
