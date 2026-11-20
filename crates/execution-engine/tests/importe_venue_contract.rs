//! CICLOS-1 (pedido 20261118-0535-cicloscompletos), corregido por CICLOS-3 (adenda del coordinador,
//! punto 4; QODER-APP C6): importes de caja del venue como decimal EXACTO CON SIGNO sobre el texto
//! del venue, con la cadena original y SIN escala elegida (la de `decimal_e8`, 1e-8, es de
//! cantidad, sin signo y cortada en 8 cifras). Lo que no se puede representar exacto => N/D con
//! causa, nunca cero ni redondeo.
use execution_engine::importe_venue::{
    importe_de_texto, importe_venue, CausaImporteNd, DecimalExacto, FilaIncomeExacta, ImporteVenue, MAX_CIFRAS,
};
use serde_json::json;

fn ok(t: &str) -> ImporteVenue {
    match importe_de_texto(t) {
        Ok(i) => i,
        Err(c) => panic!("{t:?} debia leerse exacto, salio N/D {c:?}"),
    }
}
fn d(t: &str) -> DecimalExacto {
    ok(t).valor
}
fn par(t: &str) -> (i128, u32) {
    let v = d(t);
    (v.mantisa(), v.cifras())
}

#[test]
fn ciclos_a_01_exacto_sobre_el_texto_y_conserva_el_original() {
    let i = ok("0.12345678");
    assert_eq!(par("0.12345678"), (12_345_678, 8));
    assert_eq!(i.original, "0.12345678");
    assert_eq!(par("3"), (3, 0));
    assert_eq!(par("101.5"), (1_015, 1));
    // Ceros de cola: mismo valor, texto original intacto.
    let relleno = ok("0.123456780000");
    assert_eq!(relleno.valor, d("0.12345678"));
    assert_eq!(relleno.original, "0.123456780000", "la cadena original se conserva tal cual");
    assert_eq!(d("0.50"), d("0.5"), "la igualdad es de valor");
    assert_eq!(d("-0.015").texto(), "-0.015");
    assert_eq!(d("0.000001").texto(), "0.000001");
}

#[test]
fn ciclos_a_02_signo_del_flujo_de_caja() {
    assert_eq!(par("-0.015"), (-15, 3), "comision pagada: el signo se conserva (decimal_e8 lo rechaza)");
    assert_eq!(par("+0.05"), (5, 2), "rebate");
    assert_eq!(par("-0.10000000"), (-1, 1), "funding pagado");
    let cero = ok("0");
    assert_eq!(cero.valor, DecimalExacto::CERO, "un cero EXPLICITO del venue es cero acreditado");
    assert_eq!(cero.original, "0");
    assert_eq!(d("-0"), DecimalExacto::CERO);
    assert_eq!(d("-0.015").signo(), std::cmp::Ordering::Less);
}

/// Sin escala elegida: mas de 8 cifras decimales NO es N/D ni se redondea; es exacto.
#[test]
fn ciclos_a_03_sin_escala_elegida_mas_de_ocho_cifras_es_exacto() {
    assert_eq!(par("0.123456789"), (123_456_789, 9));
    assert_eq!(par("-0.000000001"), (-1, 9));
    assert_eq!(par("1.000000005"), (1_000_000_005, 9));
    assert_eq!(par("0.00000000001"), (1, 11));
    assert_eq!(d("0.123456789").en_unidades(8), None, "a la escala de cantidades no cabe: no se redondea");
    assert_eq!(d("0.12345678").en_unidades(8), Some(12_345_678));
    // Lo que no se puede representar exacto: N/D.
    let largo = format!("0.{}1", "0".repeat(MAX_CIFRAS as usize));
    assert_eq!(importe_de_texto(&largo), Err(CausaImporteNd::FueraDeRango));
    assert_eq!(importe_de_texto(&"9".repeat(40)), Err(CausaImporteNd::FueraDeRango));
}

#[test]
fn ciclos_a_04_no_finito_es_nd() {
    for t in ["NaN", "nan", "inf", "-inf", "Infinity", "-Infinity", "+INF"] {
        assert_eq!(importe_de_texto(t), Err(CausaImporteNd::NoFinito), "{t}");
    }
}

#[test]
fn ciclos_a_05_ausente_vacio_tipo_y_formato_son_nd_con_su_causa() {
    assert_eq!(importe_venue(None), Err(CausaImporteNd::Ausente));
    assert_eq!(importe_venue(Some(&json!(null))), Err(CausaImporteNd::TipoNoNumerico));
    assert_eq!(importe_venue(Some(&json!(true))), Err(CausaImporteNd::TipoNoNumerico));
    assert_eq!(importe_venue(Some(&json!(""))), Err(CausaImporteNd::Vacio));
    assert_eq!(importe_venue(Some(&json!("   "))), Err(CausaImporteNd::Vacio));
    for t in ["abc", "1e-8", "1E3", "0x10", "1.2.3", ".5", "5.", "-", "1,5", "- 1"] {
        assert_eq!(importe_de_texto(t), Err(CausaImporteNd::Formato), "{t}");
    }
    // Lo que antes desbordaba la escala de 1e-8 en i64 ahora es exacto.
    assert_eq!(par("92233720368.54775808"), (9_223_372_036_854_775_808, 8));
}

/// Un numero JSON con decimales ya paso por `f64` en serde_json: el texto del venue se perdio, N/D.
/// Un entero JSON es exacto.
#[test]
fn ciclos_a_06_numero_json_con_decimales_es_nd_entero_exacto() {
    assert_eq!(importe_venue(Some(&json!("-0.015"))).map(|i| i.valor), Ok(d("-0.015")));
    assert_eq!(importe_venue(Some(&json!(-0.015))), Err(CausaImporteNd::NumeroJsonSinTexto));
    assert_eq!(importe_venue(Some(&json!(0.1))), Err(CausaImporteNd::NumeroJsonSinTexto));
    assert_eq!(importe_venue(Some(&json!(3))).map(|i| i.valor), Ok(d("3")));
    assert_eq!(importe_venue(Some(&json!(-7))).map(|i| i.valor), Ok(d("-7")));
}

/// Hallazgo de la N1 de NVIDIA: con `f64`, 0,1 + 0,2 no es 0,3. En decimal exacto cuadra.
#[test]
fn ciclos_a_07_cuadratura_exacta_sin_tolerancia() {
    let f = |t: &str| t.parse::<f64>().unwrap_or(f64::NAN);
    assert_ne!(f("0.1") + f("0.2"), f("0.3"), "control: en f64 no cuadra");
    assert_eq!(d("0.1").checked_add(d("0.2")), Some(d("0.3")));
    // Bruto +0,50, comision -0,40, rebate +0,05, funding +0,20 y -0,10 => +0,25 exacto (espec. §2.1).
    let x = ["0.50", "-0.40", "0.05", "0.20", "-0.10"].iter().try_fold(DecimalExacto::CERO, |a, t| a.checked_add(d(t)));
    assert_eq!(x, Some(d("0.25")));
    // Cifras distintas se alinean sin perder nada; una unidad en la ultima cifra NO es igual.
    assert_eq!(d("0.25").checked_add(d("-0.000000001")), Some(d("0.249999999")));
    assert_ne!(d("0.249999999"), d("0.25"));
    assert_ne!(d("1000.000000000001"), d("1000"));
    assert_eq!(d("1000.000000000001").checked_sub(d("1000")), Some(d("0.000000000001")));
    // Serializa como texto canonico exacto y vuelve igual.
    let s = serde_json::to_string(&d("-0.249999999")).unwrap_or_default();
    assert_eq!(s, "\"-0.249999999\"");
    assert_eq!(serde_json::from_str::<DecimalExacto>(&s).ok(), Some(d("-0.249999999")));
    assert!(serde_json::from_str::<DecimalExacto>("\"0.50\"").is_err(), "solo texto canonico");
    assert!(serde_json::from_str::<DecimalExacto>("0.5").is_err(), "un numero JSON no");
}

#[test]
fn ciclos_a_08_fila_de_income_exacta_conserva_identidad_y_causa() {
    let v = json!({"symbol":"BTCUSDT","incomeType":"COMMISSION","income":"-0.012345678","asset":"USDT",
                   "time":1700000000000u64,"tranId":9001u64,"tradeId":"777"});
    let fila = FilaIncomeExacta::desde_valor(&v);
    assert_eq!(fila.symbol, "BTCUSDT");
    assert_eq!(fila.income_type, "COMMISSION");
    assert_eq!(fila.importe, Ok(ImporteVenue { valor: d("-0.012345678"), original: "-0.012345678".into() }));
    assert_eq!((fila.asset.as_str(), fila.time, fila.tran_id, fila.trade_id.as_str()), ("USDT", 1_700_000_000_000, 9001, "777"));
    let sin = FilaIncomeExacta::desde_valor(&json!({"symbol":"BTCUSDT","incomeType":"FUNDING_FEE","asset":"USDT"}));
    assert_eq!(sin.importe, Err(CausaImporteNd::Ausente), "sin income: N/D, nunca 0");
    // QODER-APP C7: activo ausente queda vacio, nunca "USDT" por defecto.
    let sin_activo = FilaIncomeExacta::desde_valor(&json!({"income":"-0.01","tranId":1u64}));
    assert_eq!(sin_activo.asset, "");
    let flotante = FilaIncomeExacta::desde_valor(&json!({"income":-0.01}));
    assert_eq!(flotante.importe, Err(CausaImporteNd::NumeroJsonSinTexto));
}
