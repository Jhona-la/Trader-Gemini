//! I-10 (Línea I, fila N19 del censo C-F7b): el aplanado del sistema inmune
//! que no se completó.
//!
//! Cuando el sistema inmune del host se dispara (cerrojo del operador, cuenta
//! insolvente, freno duro por drawdown o tres strikes de latencia) arma el
//! kill-switch y aplana todo con `OrderExecutor::flatten_all_positions`. Antes
//! se marcaba hecho (`latched`) también si el aplanado devolvía Err, y no lo
//! volvía a intentar: las posiciones que no cerró quedaban abiertas, con sus
//! brackets (el vigilante, CL-44, las re-protege pero no las cierra), hasta
//! que tocaran TP o SL o alguien interviniera. Dos casos:
//!
//! - Con un 429/418 activo el aplanado no envía nada: devuelve el freno antes
//!   de firmar (N5/RV14). Ese freno caduca solo (M4-C02), y nadie reintentaba.
//! - Desde CL-178 un 429 a mitad del aplanado deja sin enviar los cierres
//!   restantes, y un rechazo del exchange deja su posición abierta: devuelve
//!   `FLATTEN_INCOMPLETE`.
//!
//! Aquí queda el aplanado pendiente mientras el inmune siga en su cerrojo, y
//! cuándo se reintenta. Un intento frenado no envía ningún cierre, así que se
//! repite cada ciclo del inmune y sale en cuanto acaba el freno. Tras
//! cualquier otro error (un rechazo del exchange, la lectura de posiciones
//! caída) la espera se dobla desde un ciclo hasta la cadencia nominal del
//! vigilante: cada intento lee el modo y las posiciones y vuelve a enviar los
//! cierres, y un rechazo que se repite no debe comerse la cuota de las
//! salidas. La espera cuenta desde el FINAL del intento, y los reintentos
//! usan `OrderExecutor::flatten_open_positions`, sin el barrido de órdenes
//! huérfanas que el primer intento ya hizo: un intento largo no encadena el
//! siguiente ni deja al vigilante sin auditar.
//!
//! Un cierre con llenado parcial tampoco cierra la posición: el aplanado lo
//! lleva a X-026 en vez de darlo por cerrado y purgar sus brackets.

use crate::freno_temporal::es_freno_temporal;

/// Periodo del ciclo del sistema inmune del host: evalúa sus disparadores
/// cada 5 s.
pub const PERIODO_CICLO_INMUNE_MS: u64 = 5_000;

/// Techo de la espera entre reintentos tras un error que no es freno: la
/// cadencia nominal del vigilante de protección (B1.3, 12 ciclos de 5 s), que
/// es la que audita esas mismas posiciones mientras siguen abiertas.
pub const TECHO_ESPERA_MS: u64 = 12 * PERIODO_CICLO_INMUNE_MS;

/// Qué se sabe del aplanado del inmune tras su último intento.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrasAplanado {
    /// Devolvió Ok: el aplanado cerró entera cada posición que vio.
    Cumplido,
    /// Lo rechazó el freno temporal: no salió ningún cierre. Se reintenta en
    /// el siguiente ciclo del inmune.
    Frenado,
    /// Otro error: algún cierre no aterrizó o no se pudieron leer las
    /// posiciones. Se reintenta tras la espera, que se dobla.
    Fallido,
}

/// Clasifica el resultado de `flatten_all_positions` o
/// `flatten_open_positions`.
pub fn tras_aplanado<T>(resultado: &Result<T, String>) -> TrasAplanado {
    match resultado {
        Ok(_) => TrasAplanado::Cumplido,
        Err(e) if es_freno_temporal(e) => TrasAplanado::Frenado,
        Err(_) => TrasAplanado::Fallido,
    }
}

/// El aplanado pendiente del sistema inmune y su calendario de reintentos.
///
/// Los instantes son milisegundos de un reloj monótono del host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReintentoAplanado {
    pendiente: bool,
    /// No se reintenta antes de este instante.
    proximo_ms: u64,
    /// La espera tras el siguiente error que no sea freno.
    espera_ms: u64,
    /// Intentos desde que el inmune se disparó.
    intentos: u32,
}

impl Default for ReintentoAplanado {
    fn default() -> Self {
        Self::nuevo()
    }
}

impl ReintentoAplanado {
    /// Sin aplanado pendiente.
    pub const fn nuevo() -> Self {
        Self {
            pendiente: false,
            proximo_ms: 0,
            espera_ms: PERIODO_CICLO_INMUNE_MS,
            intentos: 0,
        }
    }

    /// Anota el resultado de un intento del aplanado (el del disparo o un
    /// reintento) que terminó en `ahora_ms`, y devuelve su clasificación. La
    /// espera hasta el siguiente cuenta desde ese final.
    pub fn registrar<T>(&mut self, resultado: &Result<T, String>, ahora_ms: u64) -> TrasAplanado {
        self.intentos = self.intentos.saturating_add(1);
        let tras = tras_aplanado(resultado);
        match tras {
            TrasAplanado::Cumplido => {
                self.pendiente = false;
                self.espera_ms = PERIODO_CICLO_INMUNE_MS;
            }
            TrasAplanado::Frenado => {
                self.pendiente = true;
                self.proximo_ms = ahora_ms.saturating_add(PERIODO_CICLO_INMUNE_MS);
            }
            TrasAplanado::Fallido => {
                self.pendiente = true;
                self.proximo_ms = ahora_ms.saturating_add(self.espera_ms);
                self.espera_ms = self.espera_ms.saturating_mul(2).min(TECHO_ESPERA_MS);
            }
        }
        tras
    }

    /// Si toca reintentar el aplanado en `ahora_ms`.
    pub fn toca(&self, ahora_ms: u64) -> bool {
        self.pendiente && ahora_ms >= self.proximo_ms
    }

    /// Si el último intento no se completó.
    pub fn pendiente(&self) -> bool {
        self.pendiente
    }

    /// Cuándo es el próximo reintento, si hay aplanado pendiente.
    pub fn proximo_ms(&self) -> Option<u64> {
        self.pendiente.then_some(self.proximo_ms)
    }

    /// Intentos desde que el inmune se disparó.
    pub fn intentos(&self) -> u32 {
        self.intentos
    }

    /// El operador rearma el inmune (`rearme_frenos.txt`): deja de haber
    /// aplanado pendiente. Si el disparador sigue, el inmune vuelve a
    /// dispararse y a aplanar en ese mismo ciclo.
    pub fn olvidar(&mut self) {
        *self = Self::nuevo();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::OrderExecutor;

    const FRENO: &str = "RATE_LIMIT_COOLDOWN: activo hasta 2 (ahora 1)";
    const INCOMPLETO: &str = "FLATTEN_INCOMPLETE: 1 posiciones no pudieron cerrarse: [\"BTCUSDT\"]";

    fn err(e: &str) -> Result<(usize, usize), String> {
        Err(e.to_string())
    }

    #[test]
    fn i10_clasifica_el_resultado_del_aplanado() {
        assert_eq!(
            tras_aplanado::<(usize, usize)>(&Ok((2, 1))),
            TrasAplanado::Cumplido
        );
        assert_eq!(tras_aplanado(&err(FRENO)), TrasAplanado::Frenado);
        assert_eq!(
            tras_aplanado(&err("HTTP_418_IP_BANNED retry_after=300")),
            TrasAplanado::Frenado
        );
        assert_eq!(tras_aplanado(&err(INCOMPLETO)), TrasAplanado::Fallido);
        assert_eq!(
            tras_aplanado(&err("AMBIGUOUS: HTTP 503 unreadable response body")),
            TrasAplanado::Fallido
        );
    }

    #[test]
    fn i10_un_aplanado_cumplido_no_deja_nada_pendiente() {
        let mut r = ReintentoAplanado::nuevo();
        assert!(!r.pendiente() && !r.toca(0));
        assert_eq!(
            r.registrar::<(usize, usize)>(&Ok((0, 0)), 1_000),
            TrasAplanado::Cumplido
        );
        assert!(!r.pendiente());
        assert!(!r.toca(u64::MAX));
        assert_eq!(r.proximo_ms(), None);
        assert_eq!(r.intentos(), 1);
    }

    #[test]
    fn i10_un_aplanado_frenado_se_reintenta_cada_ciclo_sin_crecer_la_espera() {
        let mut r = ReintentoAplanado::nuevo();
        let mut t = 10_000;
        for _ in 0..20 {
            assert_eq!(r.registrar(&err(FRENO), t), TrasAplanado::Frenado);
            assert!(r.pendiente());
            assert!(!r.toca(t + PERIODO_CICLO_INMUNE_MS - 1));
            assert!(r.toca(t + PERIODO_CICLO_INMUNE_MS));
            t += PERIODO_CICLO_INMUNE_MS;
        }
        // Acabado el freno, el primer error real espera un solo ciclo.
        assert_eq!(r.registrar(&err(INCOMPLETO), t), TrasAplanado::Fallido);
        assert_eq!(r.proximo_ms(), Some(t + PERIODO_CICLO_INMUNE_MS));
    }

    #[test]
    fn i10_tras_un_error_real_la_espera_se_dobla_hasta_la_cadencia_del_vigilante() {
        let mut r = ReintentoAplanado::nuevo();
        let mut t = 0_u64;
        let mut esperas = Vec::new();
        for _ in 0..8 {
            r.registrar(&err(INCOMPLETO), t);
            let prox = r.proximo_ms().unwrap_or(0);
            esperas.push(prox - t);
            t = prox;
        }
        assert_eq!(
            esperas,
            [5_000, 10_000, 20_000, 40_000, 60_000, 60_000, 60_000, 60_000]
        );
        assert_eq!(TECHO_ESPERA_MS, 60_000);
        // Un freno entre medias no rebaja la espera acumulada ni la alarga.
        r.registrar(&err(FRENO), t);
        assert_eq!(r.proximo_ms(), Some(t + PERIODO_CICLO_INMUNE_MS));
        r.registrar(&err(INCOMPLETO), t + 5_000);
        assert_eq!(r.proximo_ms(), Some(t + 5_000 + TECHO_ESPERA_MS));
        // Un Ok cierra el pendiente y la espera vuelve a un ciclo.
        r.registrar::<(usize, usize)>(&Ok((1, 1)), t + 70_000);
        assert!(!r.pendiente());
        r.registrar(&err(INCOMPLETO), t + 80_000);
        assert_eq!(r.proximo_ms(), Some(t + 85_000));
    }

    #[test]
    fn i10_el_rearme_del_operador_olvida_el_pendiente() {
        let mut r = ReintentoAplanado::nuevo();
        r.registrar(&err(INCOMPLETO), 0);
        r.registrar(&err(INCOMPLETO), 5_000);
        assert!(r.pendiente());
        r.olvidar();
        assert_eq!(r, ReintentoAplanado::nuevo());
        assert!(!r.toca(u64::MAX));
    }

    #[test]
    fn i10_los_instantes_extremos_no_desbordan() {
        let mut r = ReintentoAplanado::nuevo();
        r.registrar(&err(INCOMPLETO), u64::MAX - 1);
        assert_eq!(r.proximo_ms(), Some(u64::MAX));
        r.registrar(&err(FRENO), u64::MAX);
        assert_eq!(r.proximo_ms(), Some(u64::MAX));
        assert!(r.toca(u64::MAX));
    }

    /// El aplanado del ejecutor REAL con el freno armado vuelve como freno
    /// temporal sin salir a la red: el inmune lo reintenta cada ciclo hasta
    /// que caduque, en vez de quedarse con las posiciones abiertas.
    #[tokio::test]
    async fn i10_el_aplanado_frenado_del_ejecutor_real_es_freno_temporal() {
        for armado in [
            "HTTP_429_RATE_LIMITED retry_after=30",
            "HTTP_418_IP_BANNED retry_after=300",
        ] {
            let ex = OrderExecutor::new("k".into(), "s".into(), true);
            ex.client().freno().registrar(armado);
            let r = ex.flatten_all_positions().await;
            assert!(
                matches!(&r, Err(e) if es_freno_temporal(e)),
                "{armado}: aplanado {r:?}"
            );
            let reintento_sin_barrido = ex.flatten_open_positions().await;
            assert!(
                matches!(&reintento_sin_barrido, Err(e) if es_freno_temporal(e)),
                "{armado}: aplanado sin barrido {reintento_sin_barrido:?}"
            );
            let mut reintento = ReintentoAplanado::nuevo();
            assert_eq!(reintento.registrar(&r, 0), TrasAplanado::Frenado);
            assert!(reintento.toca(PERIODO_CICLO_INMUNE_MS));
        }
    }
}
