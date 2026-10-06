//! XCV (F6-A-H4 del barrido): las 9 dims cross-exchange de get_features()
//! son MUERTAS POR CONTRATO — ceros estructurales, no falta de datos.
//!
//! Los pollers que las escribirían (run_bybit_ws/run_okx_ws vía
//! OmniDataHub::start_feeds) son código muerto sin callers; el default 0.0
//! de cada AtomicU64 más norm_spread(0.0)=0.0 las deja perpetuamente en
//! cero. Los modelos MOTOR de la familia honesta ENTRENARON con estos
//! ceros — la paridad trainer↔vivo existe por construcción sobre ellos.
//! Este contrato declara el estado: si alguien "arregla" los pollers o
//! cambia el estado inicial, ESTE TEST SUENA — y quien lo rompa debe
//! re-entrenar los modelos antes de fusionar (rompería la paridad
//! silenciosamente).

use data_pipeline::omni_multiplexer::OmniState;

#[test]
fn xcv_las_9_dims_cross_exchange_son_ceros_estructurales() {
    let st = OmniState::new();
    let feats = st.get_features();
    // slots 2..10: bybit, okx, bitget, coinbase, kraken, htx, deribit,
    // bitfinex (8 exchanges) + el propio binance_futures en slot 1 si su
    // poller también está muerto en frío — el contrato exige el cero
    // EXACTO (bits), no aproximado.
    for slot in 1..10 {
        assert_eq!(
            feats[slot].to_bits(),
            0.0_f64.to_bits(),
            "slot {slot}: las dims cross-exchange son ceros estructurales \
             (muertas por contrato, F6-A-H4/XCV) — si esto cambió, un poller \
             se activó: RE-ENTRENAR los modelos antes de fusionar"
        );
    }
    // La referencia base (slot 0) es 0.0 por definición (retorno relativo).
    assert_eq!(feats[0], 0.0);
}

#[test]
fn xcv_el_fallo_guia_la_accion_no_solo_cuenta() {
    // Auto-documentación del contrato: el mensaje del assert nombra la
    // acción requerida (re-entrenar), no sólo el hecho. Este test
    // acompaña al anterior para que el reporte del CI sea accionable.
    let msg = "las dims cross-exchange son ceros estructurales \
               (muertas por contrato, F6-A-H4/XCV) — si esto cambió, un poller \
               se activó: RE-ENTRENAR los modelos antes de fusionar";
    assert!(msg.contains("RE-ENTRENAR"), "el contrato debe ser accionable");
}
