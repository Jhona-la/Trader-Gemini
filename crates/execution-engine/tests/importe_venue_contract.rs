//! CICLOS-1 (pedido 20261118-0535-cicloscompletos): importes de caja del venue en punto fijo, a la
//! escala de `user_data_stream::decimal_e8` (1e-8), con la cadena original. Mas decimales de los que
//! admite la escala o valor no finito => N/D con causa exacta, nunca cero.
use execution_engine::importe_venue::{
    importe_de_texto, importe_venue_e8, CausaImporteNd, FilaIncomeExacta, ImporteVenue, UNIDADES_POR_ACTIVO,
};
use serde_json::json;

fn ok(t: &str) -> ImporteVenue {
    match importe_de_texto(t) {
        Ok(i) => i,
        Err(c) => panic!("{t:?} debia leerse exacto, salio N/D {c:?}"),
    }
}

#[test]
fn ciclos_a_01_misma_escala_que_decimal_e8_y_conserva_el_original() {
    assert_eq!(UNIDADES_POR_ACTIVO, 100_000_000);
    let i = ok("0.12345678");
    assert_eq!(i.e8, 12_345_678);
    assert_eq!(i.original, "0.12345678");
    assert_eq!(ok("3").e8, 300_000_000);
    assert_eq!(ok("101.5").e8, 10_150_000_000);
    // Ceros de relleno mas alla de la escala no cambian el valor (igual que decimal_e8).
    let relleno = ok("0.123456780000");
    assert_eq!(relleno.e8, 12_345_678);
    assert_eq!(relleno.original, "0.123456780000", "la cadena original se conserva tal cual");
}

#[test]
fn ciclos_a_02_signo_del_flujo_de_caja() {
    assert_eq!(ok("-0.015").e8, -1_500_000, "comision pagada");
    assert_eq!(ok("+0.05").e8, 5_000_000, "rebate");
    assert_eq!(ok("-0.10000000").e8, -10_000_000, "funding pagado");
    let cero = ok("0");
    assert_eq!(cero.e8, 0, "un cero EXPLICITO del venue es cero acreditado");
    assert_eq!(cero.original, "0");
}

#[test]
fn ciclos_a_03_mas_decimales_que_la_escala_es_nd_nunca_redondeo() {
    for t in ["0.123456789", "-0.000000001", "1.000000005", "0.00000000001"] {
        assert_eq!(importe_de_texto(t), Err(CausaImporteNd::MasDecimalesQueLaEscala), "{t}");
    }
}

#[test]
fn ciclos_a_04_no_finito_es_nd() {
    for t in ["NaN", "nan", "inf", "-inf", "Infinity", "-Infinity", "+INF"] {
        assert_eq!(importe_de_texto(t), Err(CausaImporteNd::NoFinito), "{t}");
    }
}

#[test]
fn ciclos_a_05_ausente_vacio_tipo_y_formato_son_nd_con_su_causa() {
    assert_eq!(importe_venue_e8(None), Err(CausaImporteNd::Ausente));
    assert_eq!(importe_venue_e8(Some(&json!(null))), Err(CausaImporteNd::TipoNoNumerico));
    assert_eq!(importe_venue_e8(Some(&json!(true))), Err(CausaImporteNd::TipoNoNumerico));
    assert_eq!(importe_venue_e8(Some(&json!(""))), Err(CausaImporteNd::Vacio));
    assert_eq!(importe_venue_e8(Some(&json!("   "))), Err(CausaImporteNd::Vacio));
    for t in ["abc", "1e-8", "1E3", "0x10", "1.2.3", ".5", "5.", "-", "1,5", "- 1"] {
        assert_eq!(importe_de_texto(t), Err(CausaImporteNd::Formato), "{t}");
    }
    assert_eq!(importe_de_texto("92233720368.54775808"), Err(CausaImporteNd::FueraDeRango));
    assert_eq!(importe_de_texto("99999999999999999999"), Err(CausaImporteNd::FueraDeRango));
}

#[test]
fn ciclos_a_06_numero_json_y_cadena_dan_el_mismo_entero() {
    assert_eq!(importe_venue_e8(Some(&json!("-0.015"))).map(|i| i.e8), Ok(-1_500_000));
    assert_eq!(importe_venue_e8(Some(&json!(-0.015))).map(|i| i.e8), Ok(-1_500_000));
    assert_eq!(importe_venue_e8(Some(&json!(3))).map(|i| i.e8), Ok(300_000_000));
}

/// Hallazgo de la N1 de NVIDIA: con `f64`, 0,1 + 0,2 no es 0,3 y la cuadratura por igualdad exacta
/// falla sin que haya descuadre real. En punto fijo cuadra exacta.
#[test]
fn ciclos_a_07_cuadratura_exacta_en_punto_fijo() {
    let f = |t: &str| t.parse::<f64>().unwrap_or(f64::NAN);
    assert_ne!(f("0.1") + f("0.2"), f("0.3"), "control: en f64 no cuadra");
    assert_eq!(ok("0.1").e8 + ok("0.2").e8, ok("0.3").e8);
    // Bruto +0,50, comision -0,40, rebate +0,05, funding +0,20 y -0,10 => +0,25 exacto (espec. §2.1).
    let x: i64 = ["0.50", "-0.40", "0.05", "0.20", "-0.10"].iter().map(|t| ok(t).e8).sum();
    assert_eq!(x, ok("0.25").e8);
}

#[test]
fn ciclos_a_08_fila_de_income_exacta_conserva_identidad_y_causa() {
    let v = json!({"symbol":"BTCUSDT","incomeType":"COMMISSION","income":"-0.01234567","asset":"USDT",
                   "time":1700000000000u64,"tranId":9001u64,"tradeId":"777"});
    let fila = FilaIncomeExacta::desde_valor(&v);
    assert_eq!(fila.symbol, "BTCUSDT");
    assert_eq!(fila.income_type, "COMMISSION");
    assert_eq!(fila.importe, Ok(ImporteVenue { e8: -1_234_567, original: "-0.01234567".into() }));
    assert_eq!((fila.asset.as_str(), fila.time, fila.tran_id, fila.trade_id.as_str()), ("USDT", 1_700_000_000_000, 9001, "777"));
    let sin = FilaIncomeExacta::desde_valor(&json!({"symbol":"BTCUSDT","incomeType":"FUNDING_FEE","asset":"USDT"}));
    assert_eq!(sin.importe, Err(CausaImporteNd::Ausente), "sin income: N/D, nunca 0");
    let largo = FilaIncomeExacta::desde_valor(&json!({"income":"0.123456789"}));
    assert_eq!(largo.importe, Err(CausaImporteNd::MasDecimalesQueLaEscala));
}
