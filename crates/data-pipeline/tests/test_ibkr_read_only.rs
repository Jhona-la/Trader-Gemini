// IBKR-I3 (integrador, 2026-11-16): sonda de conectividad contra un IB Gateway
// REAL. Va con #[ignore]: nunca corre en CI ni en la compuerta. Solo a mano,
// con el Gateway ya configurado en Read-Only (pasos manuales del pedido v2):
//
//   cargo test -p data-pipeline --test test_ibkr_read_only -- --ignored
//
// Direccion: variable TG_IBKR_GATEWAY (host:puerto). Sin ella se usa el puerto
// de Paper Trading del pedido (paso manual 3: 4002), nunca el de Live. Ninguna
// credencial pasa por aqui: el usuario entra en el Gateway, no en el codigo.
//
// La sonda NEGATIVA de la especificacion v2 (seccion 6.1) intentaba una orden
// para ver el rechazo del modo Read-Only (Error 321). No se escribe: seria una
// ruta a ordenes contra la cuenta real si el Read-Only estuviera desmarcado, y
// la puerta tests/ibkr_solo_lectura_contract.rs la prohibe tambien en tests/.
// El Read-Only se comprueba en la pantalla del Gateway (paso manual 6).

use ibapi::market_data::MarketDataType;
use ibapi::Client;

/// Puerto de Paper Trading del IB Gateway (pedido v2, pasos manuales, paso 3).
const GATEWAY_PAPER_POR_DEFECTO: &str = "127.0.0.1:4002";

/// client_id de la sonda: N/D (IBKR solo exige que no lo use otra sesion
/// conectada al mismo Gateway). Se cambia con TG_IBKR_CLIENT_ID.
const CLIENT_ID_SONDA_POR_DEFECTO: i32 = 101;

fn client_id() -> i32 {
    std::env::var("TG_IBKR_CLIENT_ID").ok().and_then(|v| v.parse().ok()).unwrap_or(CLIENT_ID_SONDA_POR_DEFECTO)
}

fn direccion() -> String {
    std::env::var("TG_IBKR_GATEWAY").unwrap_or_else(|_| GATEWAY_PAPER_POR_DEFECTO.to_string())
}

#[tokio::test]
#[ignore = "necesita un IB Gateway real en Read-Only; ejecucion manual"]
async fn test_ibkr_read_only_connectivity_and_delayed_stream() {
    let dir = direccion();
    let cliente = Client::connect(&dir, client_id()).await.unwrap_or_else(|e| panic!("no conecta a {dir}: {e}"));
    let hora = cliente.server_time().await.expect("server_time");
    println!("IBKR server_time = {hora}");
    cliente
        .switch_market_data_type(MarketDataType::Delayed)
        .await
        .expect("switch_market_data_type(Delayed)");
}
