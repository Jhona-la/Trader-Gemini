use reqwest::{Client, ClientBuilder};
use std::sync::OnceLock;
use std::time::Duration;

pub static GLOBAL_HTTP_CLIENT: OnceLock<Client> = OnceLock::new();

pub fn get_global_client() -> &'static Client {
    GLOBAL_HTTP_CLIENT.get_or_init(|| {
        ClientBuilder::new()
            .tcp_nodelay(true)
            .tcp_keepalive(Duration::from_secs(15))
            .pool_idle_timeout(None)
            .pool_max_idle_per_host(50)
            .timeout(Duration::from_secs(5))
                        .build()
            .expect("Fallo al inicializar Quantum Global HTTP Client")
    })
}
