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
    // MIXED_EXTRA trae el esquema DEMO_TEL de 515e3bf4 (ORDER_SEND sin `tipo`, ...):
    // queda diagnosticado y el informe oficial no es apto aunque el legacy una.
    let solo_legacy = escribir("solo_legacy.log", LEGACY)?;
    let r = correr_cli(&["--log", &solo_legacy, "--income", &income, "--ventana", "0:10000"]);
    assert_eq!(r.codigo, 0, "{}", r.error);
    assert!(r.salida.starts_with(&esperado));
    assert!(r.salida.contains("VETUSDT | 701 | LEGACY | 3000 |"), "{}", r.salida);
    assert!(r.salida.contains("ACEPTACION: APTO"), "{}", r.salida);

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