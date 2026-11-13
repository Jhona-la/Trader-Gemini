//! Offline acceptance of the single USDT wallet measure, using the real adapters.
use execution_engine::client::BinanceClient;
use execution_engine::execution_evidence::{parse_account_capital, parse_account_lectura};
use execution_engine::puente_cuenta::PuenteCuenta;
use execution_engine::{OrderRegistry, UserDataStreamer};
use std::sync::{Arc, Mutex};

const FOTO: &str = r#"{"multiAssetsMargin":true,"totalWalletBalance":"150","availableBalance":"80","assets":[{"asset":"USDC","walletBalance":"50","updateTime":300},{"asset":"USDT","walletBalance":"100","availableBalance":"80","unrealizedProfit":"0","updateTime":100}],"positions":[]}"#;
const MOVIMIENTO: &[u8] = br#"{"e":"ACCOUNT_UPDATE","T":200,"E":200,"a":{"m":"ORDER","B":[{"a":"USDC","wb":"70","cw":"70","bc":"20"},{"a":"USDT","wb":"100","cw":"80","bc":"0"}],"P":[]}}"#;

#[test]
fn rest_y_ws_entregan_la_misma_magnitud_de_usdt_con_dos_activos() -> Result<(), String> {
    let resumen = parse_account_capital(FOTO)?;
    let (_, lectura) = parse_account_lectura(FOTO)?;
    assert_eq!(resumen.capital, 100.0);
    assert_eq!(lectura.saldo, 100.0);
    assert_eq!(lectura.tiempo_saldo_ms, Some(100));
    let aviso = resumen.aviso().ok_or("falta el desglose del saldo agregado")?;
    assert!(aviso.contains("150.0000") && aviso.contains("50.0000"));

    let puente = Arc::new(PuenteCuenta::con_sesion(1));
    let publicaciones = Arc::new(Mutex::new(Vec::new()));
    let recibos = Arc::clone(&publicaciones);
    puente.vincular_publicador(move |p| {
        if let Some(saldo) = p.vigente {
            recibos.lock().unwrap_or_else(|e| e.into_inner()).push(saldo);
        }
        !p.veta
    });

    let streamer = UserDataStreamer::new(
        BinanceClient::new(String::new(), true),
        Arc::new(OrderRegistry::new()),
    ).with_puente(Arc::clone(&puente));
    puente.abrir_socket(puente.era()).map_err(|e| e.to_string())?;
    let rest = puente.observar_rest(puente.ticket(), &lectura).map_err(|e| e.to_string())?;
    assert_eq!((rest.saldo, rest.equity), (100.0, 100.0));
    assert!(!puente.veta_entradas());

    streamer.dispatch(MOVIMIENTO);
    let ws = puente.observacion().ok_or("falta la observación del stream")?;
    assert_eq!((ws.saldo, ws.equity), (100.0, 100.0));
    assert!(!puente.veta_entradas());
    assert_eq!(
        publicaciones.lock().unwrap_or_else(|e| e.into_inner()).as_slice(),
        &[(100.0, Some(100)), (100.0, Some(200))],
    );
    Ok(())
}
