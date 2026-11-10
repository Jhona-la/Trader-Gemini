//! NTP Synchronizer for live timestamp drift correction against Binance Futures.
//!
//! Implementación con Filtro de Kalman 1D y rechazo de outliers para eliminar
//! las oscilaciones y falsos picos de latencia en Windows.

use crate::client::BinanceClient;
use quantum_arena::GlobalArena;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Filtro de Kalman 1D para estimar de forma continua y suave el offset de reloj
/// entre el host local y los servidores de Binance Futures.
///
/// Estado $x$: Offset en milisegundos (`server_time - local_time`).
/// Varianza $P$: Incertidumbre de la estimación.
/// Ruido de proceso $Q$: Deriva temporal típica del oscilador de cristal en Windows (~0.5 ms² por ciclo).
/// Ruido de medición $R$: Incertidumbre del viaje de red RTT. $R = \max((RTT/2)^2, R_{\min})$.
///
/// Rechazo de outliers: Si $RTT > 350\text{ ms}$ y ya existe un prior convergente, la muestra
/// se descarta para evitar que una congestión de red transitoria contamine el reloj de trading.
#[derive(Debug, Clone, Copy)]
pub struct NtpKalmanFilter {
    pub offset_estimate: f64,
    pub error_covariance: f64,
    pub initialized: bool,
    pub process_noise_q: f64,
    pub sample_count: u64,
}

impl Default for NtpKalmanFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl NtpKalmanFilter {
    pub const DEFAULT_Q: f64 = 0.5; // (ms)² por ciclo de 15s
    pub const MIN_MEASUREMENT_VARIANCE: f64 = 4.0; // Mínimo 2ms de incertidumbre (2² = 4)
    pub const OUTLIER_RTT_THRESHOLD_MS: u64 = 350; // Descarte de picos de congestión

    pub const fn new() -> Self {
        Self {
            offset_estimate: 0.0,
            error_covariance: 10_000.0,
            initialized: false,
            process_noise_q: Self::DEFAULT_Q,
            sample_count: 0,
        }
    }

    /// I-5: como `update` pero con el offset crudo medido contra la
    /// PROYECCIÓN MONÓTONA (no contra la pared): el `raw_offset` lo calcula
    /// el llamador con `correccion_vs_proyeccion`. La puerta de outliers por
    /// RTT (>350 ms) se conserva.
    pub fn update_contra_proyeccion(&mut self, rtt_ms: u64, raw_offset: f64) -> Option<i64> {
        if !raw_offset.is_finite() {
            return None;
        }
        if self.initialized && rtt_ms > Self::OUTLIER_RTT_THRESHOLD_MS {
            return Some(self.offset_estimate.round() as i64);
        }
        let rtt_half = (rtt_ms as f64) / 2.0;
        if !self.initialized {
            self.offset_estimate = raw_offset;
            self.error_covariance = rtt_half.max(2.0).powi(2);
            self.initialized = true;
            self.sample_count = 1;
            return Some(raw_offset.round() as i64);
        }
        let p_prior = (self.error_covariance + self.process_noise_q).clamp(1e-4, 1e6);
        let r_meas = rtt_half.powi(2).max(Self::MIN_MEASUREMENT_VARIANCE);
        let k_gain = p_prior / (p_prior + r_meas);
        self.offset_estimate += k_gain * (raw_offset - self.offset_estimate);
        self.error_covariance = ((1.0 - k_gain) * p_prior).clamp(1e-4, 1e6);
        self.sample_count += 1;
        Some(self.offset_estimate.round() as i64)
    }

    /// Actualiza el filtro con una medición de sincronización NTP.
    /// Retorna el nuevo offset filtrado en milisegundos si la medición es válida.
    pub fn update(&mut self, t0: u64, t1: u64, server_time: u64) -> Option<i64> {
        let rtt = t1.saturating_sub(t0);

        // Descarte de outliers si ya estamos inicializados
        if self.initialized && rtt > Self::OUTLIER_RTT_THRESHOLD_MS {
            return Some(self.offset_estimate.round() as i64);
        }

        let rtt_half = (rtt as f64) / 2.0;
        let local_estimate = (t0 as f64) + rtt_half;
        let raw_offset = (server_time as f64) - local_estimate;

        if !raw_offset.is_finite() {
            return None;
        }

        if !self.initialized {
            self.offset_estimate = raw_offset;
            self.error_covariance = rtt_half.max(2.0).powi(2);
            self.initialized = true;
            self.sample_count = 1;
            return Some(raw_offset.round() as i64);
        }

        // 1. Predicción: P = P + Q
        let p_prior = (self.error_covariance + self.process_noise_q).clamp(1e-4, 1e6);

        // 2. Varianza de medición proporcional al RTT: R = max((RTT/2)², R_min)
        let r_meas = rtt_half.powi(2).max(Self::MIN_MEASUREMENT_VARIANCE);

        // 3. Ganancia de Kalman: K = P / (P + R)
        let k_gain = p_prior / (p_prior + r_meas);

        // 4. Actualización del estado e incertidumbre
        let innovation = raw_offset - self.offset_estimate;
        self.offset_estimate += k_gain * innovation;
        self.error_covariance = ((1.0 - k_gain) * p_prior).clamp(1e-4, 1e6);
        self.sample_count += 1;

        Some(self.offset_estimate.round() as i64)
    }

    #[inline(always)]
    pub fn current_offset(&self) -> Option<i64> {
        if self.initialized && self.offset_estimate.is_finite() {
            Some(self.offset_estimate.round() as i64)
        } else {
            None
        }
    }
}

/// I-5 (Línea I): proyección monótona de la hora del servidor. El ancla es
/// una estimación de servidor (`srv_ms`) tomada en un instante monótono
/// (`Instant`); a partir de ahí la proyección avanza con el reloj monótono
/// del proceso y NUNCA vuelve a leer el reloj de pared: una corrección de
/// w32time a mitad de sesión es estructuralmente invisible.
pub fn proyeccion_srv(ancla_srv_ms: i64, transcurrido: Duration) -> i64 {
    ancla_srv_ms.saturating_add(i64::try_from(transcurrido.as_millis()).unwrap_or(i64::MAX))
}

/// I-5: corrección medida contra la proyección monótona (no contra la pared).
/// Es lo único que puede entrar en la edad del dato como error de reloj:
/// deriva del oscilador y jitter, jamás un salto de w32time.
pub fn correccion_vs_proyeccion(server_ms: i64, proyeccion_ms: i64) -> i64 {
    server_ms.saturating_sub(proyeccion_ms)
}

/// Cadencia del sincronizador: una muestra de la hora del servidor cada 15 s.
pub const PERIODO_SINCRONIZACION: Duration = Duration::from_secs(15);

/// Muestras por hora a la cadencia de [`PERIODO_SINCRONIZACION`] (240).
/// Nunca 0: con un periodo de más de una hora queda en 1.
pub const MUESTRAS_POR_HORA: u64 = {
    let n = 3_600_000 / (PERIODO_SINCRONIZACION.as_millis() as u64);
    if n == 0 {
        1
    } else {
        n
    }
};

/// RB-N1: lo que el sincronizador publica de cada medición procesada; el
/// host lo usa para dejar traza de que el reloj está vivo y de cuánto corrige.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MuestraNtp {
    /// Mediciones aceptadas hasta ahora por el filtro del offset de pared
    /// (la primera fija el ancla de la proyección monótona).
    pub aceptadas: u64,
    /// `true` si ESTA medición entró en los filtros; `false` si la descartó la
    /// puerta de outliers por RTT (lo publicado no cambia).
    pub aceptada: bool,
    /// Mediciones descartadas seguidas (vuelve a 0 con la próxima aceptada).
    pub rechazadas_seguidas: u64,
    /// Ida y vuelta de la petición, ms.
    pub rtt_ms: u64,
    /// Offset filtrado servidor − reloj de pared local, ms (firma de órdenes).
    pub offset_pared_ms: i64,
    /// Corrección filtrada servidor − proyección monótona, ms (edad del dato).
    /// `None` mientras sólo exista el ancla.
    pub correccion_ms: Option<i64>,
}

/// RB-N1: estado del reloj que comparten TODAS las generaciones del
/// sincronizador: el filtro del offset de pared, el de la corrección de la
/// proyección monótona (I-5) y el ancla de esa proyección.
///
/// El ancla la fija el HOST ([`EstadoReloj::fijar_ancla`]) con la misma
/// pareja `(epoch_baseline_ms, instant_baseline)` con la que calcula la edad
/// del dato: la corrección es exactamente lo que le falta a ESA proyección
/// para dar la hora del servidor. Así absorbe también el error del offset de
/// arranque (una muestra cruda, o la hora local si la petición falló), y el
/// relevo de generación (transición demo→mainnet) no la reinicia: con un
/// ancla propia por generación, la nueva mediría la deriva sólo desde SU
/// primera muestra y la acumulada hasta entonces volvería de golpe a la edad
/// del dato, justo al pasar a Producción.
#[derive(Debug)]
pub struct EstadoReloj {
    pared: NtpKalmanFilter,
    proyeccion: NtpKalmanFilter,
    ancla: Option<(i64, Instant)>,
    rechazadas_seguidas: u64,
}

impl Default for EstadoReloj {
    fn default() -> Self {
        Self::new()
    }
}

impl EstadoReloj {
    pub const fn new() -> Self {
        Self {
            pared: NtpKalmanFilter::new(),
            proyeccion: NtpKalmanFilter::new(),
            ancla: None,
            rechazadas_seguidas: 0,
        }
    }

    /// Fija (o vuelve a fijar) el ancla contra la que se mide la corrección:
    /// la hora de servidor `ancla_srv_ms` que el host asigna al instante
    /// monótono `ancla_mono`. El filtro de la corrección vuelve a empezar y lo
    /// inicializa la próxima medición.
    pub fn fijar_ancla(&mut self, ancla_srv_ms: i64, ancla_mono: Instant) {
        self.ancla = Some((ancla_srv_ms, ancla_mono));
        self.proyeccion = NtpKalmanFilter::new();
    }

    /// Procesa una medición. `t0` y `t1` son el reloj de pared local (ms Unix)
    /// al pedir y al recibir la hora `servidor_ms`; `enviado` y `recibido`,
    /// los instantes monótonos de lo mismo. El servidor estampa su hora en el
    /// punto medio del viaje, así que la proyección se evalúa en el punto
    /// medio MONÓTONO (inmune a un salto de la pared durante la petición).
    /// Sin ancla del host no hay corrección que publicar. `None` si la
    /// medición no es representable.
    pub fn procesar(
        &mut self,
        t0: u64,
        t1: u64,
        servidor_ms: u64,
        enviado: Instant,
        recibido: Instant,
    ) -> Option<MuestraNtp> {
        let aceptadas_antes = self.pared.sample_count;
        let offset_pared_ms = self.pared.update(t0, t1, servidor_ms)?;
        let aceptada = self.pared.sample_count > aceptadas_antes;
        self.rechazadas_seguidas = if aceptada {
            0
        } else {
            self.rechazadas_seguidas.saturating_add(1)
        };
        let rtt_ms = t1.saturating_sub(t0);
        let correccion_ms = self.ancla.and_then(|(ancla_srv, ancla_mono)| {
            let viaje = recibido.saturating_duration_since(enviado);
            let medio = enviado.checked_add(viaje / 2).unwrap_or(recibido);
            let proyeccion = proyeccion_srv(ancla_srv, medio.saturating_duration_since(ancla_mono));
            let servidor = i64::try_from(servidor_ms).unwrap_or(i64::MAX);
            let cruda = correccion_vs_proyeccion(servidor, proyeccion) as f64;
            let rtt_monotono = u64::try_from(viaje.as_millis()).unwrap_or(u64::MAX);
            self.proyeccion
                .update_contra_proyeccion(rtt_monotono, cruda)
        });
        Some(MuestraNtp {
            aceptadas: self.pared.sample_count,
            aceptada,
            rechazadas_seguidas: self.rechazadas_seguidas,
            rtt_ms,
            offset_pared_ms,
            correccion_ms,
        })
    }
}

/// RB-N1 — HOST-016 corregido: UN solo sincronizador vivo… y que corra.
///
/// `reclamar` devuelve el número de la generación recién creada (1 la
/// primera, 2 la siguiente…), es decir, el valor del contador DESPUÉS del
/// incremento. Una generación sigue vigente mientras nadie reclame otra.
///
/// Corrige el defecto de 97996eba0 (2026-09-17): la guardia del host
/// guardaba lo que devuelve `fetch_add(1)` —el valor PREVIO— y salía cuando
/// `contador > previo`, condición verdadera desde su primer chequeo. Toda
/// generación, también la única, se declaraba reemplazada y el `select!`
/// cancelaba el bucle antes de su primera muestra: `server_time_offset_ms` y
/// `server_time_correction_ms` quedaban congelados en el arranque y la
/// deriva del reloj entraba entera en la edad del dato.
///
/// También guarda el [`EstadoReloj`] común a todas las generaciones.
#[derive(Debug, Default)]
pub struct SincronizadorUnico {
    generacion: AtomicU64,
    estado: Mutex<EstadoReloj>,
}

impl SincronizadorUnico {
    pub const fn new() -> Self {
        Self {
            generacion: AtomicU64::new(0),
            estado: Mutex::new(EstadoReloj::new()),
        }
    }

    /// Reclama una generación nueva; las anteriores dejan de estar vigentes.
    pub fn reclamar(&self) -> u64 {
        self.generacion
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1)
    }

    /// ¿Sigue vigente `generacion`? Sólo una reclamada después la releva.
    pub fn vigente(&self, generacion: u64) -> bool {
        self.generacion.load(Ordering::Acquire) == generacion
    }

    /// Estado del reloj que comparten todas las generaciones.
    pub fn estado(&self) -> &Mutex<EstadoReloj> {
        &self.estado
    }

    /// El host publica la pareja `(epoch_baseline_ms, instant_baseline)` de
    /// su proyección: desde aquí la corrección se mide contra ella.
    pub fn fijar_ancla(&self, ancla_srv_ms: i64, ancla_mono: Instant) {
        let mut reloj = match self.estado.lock() {
            Ok(guardia) => guardia,
            Err(envenenado) => envenenado.into_inner(),
        };
        reloj.fijar_ancla(ancla_srv_ms, ancla_mono);
    }
}

/// Cómo terminó un sincronizador gobernado por [`SincronizadorUnico`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinSincronizador {
    /// El bucle terminó por sí mismo.
    Terminado,
    /// Una generación posterior lo relevó.
    Relevado,
}

/// Corre `sync` mientras `generacion` siga vigente en `guardia`, revisando la
/// vigencia cada `sondeo` (como mínimo 1 ms, para no girar en vacío).
pub async fn correr_mientras_vigente<F>(
    guardia: &SincronizadorUnico,
    generacion: u64,
    sondeo: Duration,
    sync: F,
) -> FinSincronizador
where
    F: Future<Output = ()>,
{
    let sondeo = sondeo.max(Duration::from_millis(1));
    let relevo = async {
        while guardia.vigente(generacion) {
            tokio::time::sleep(sondeo).await;
        }
    };
    tokio::select! {
        _ = sync => FinSincronizador::Terminado,
        _ = relevo => FinSincronizador::Relevado,
    }
}

/// RB-N1: ¿merece esta muestra una línea en el log del host? Las dos
/// primeras aceptadas (el ancla y la primera corrección: prueban que el
/// bucle vive), después una por hora; y una por cada cuarto de hora seguido
/// en que la puerta de outliers lo descarte todo (el reloj no se corrige).
pub fn merece_aviso(m: &MuestraNtp) -> bool {
    let cuarto_de_hora = (MUESTRAS_POR_HORA / 4).max(1);
    if m.aceptada {
        m.aceptadas <= 2 || m.aceptadas.is_multiple_of(MUESTRAS_POR_HORA)
    } else {
        m.rechazadas_seguidas > 0 && m.rechazadas_seguidas.is_multiple_of(cuarto_de_hora)
    }
}

fn pared_ms() -> Option<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|d| u64::try_from(d.as_millis()).ok())
}

/// Bucle del sincronizador: cada [`PERIODO_SINCRONIZACION`] pide la hora del
/// servidor, la procesa sobre el estado COMPARTIDO del reloj y publica en la
/// arena el offset de pared (firma de órdenes) y la corrección de la
/// proyección monótona (edad del dato). `observador` recibe cada muestra
/// procesada, fuera de la ruta caliente (una vez por periodo).
pub async fn start_ntp_synchronizer<F>(
    client: Arc<BinanceClient>,
    arena: Arc<GlobalArena>,
    estado: &Mutex<EstadoReloj>,
    mut observador: F,
) where
    F: FnMut(MuestraNtp),
{
    let mut interval = tokio::time::interval(PERIODO_SINCRONIZACION);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        interval.tick().await;

        let Some(t0) = pared_ms() else {
            continue;
        };
        let enviado = Instant::now();

        match client.fetch_server_time().await {
            Ok(servidor_ms) => {
                let recibido = Instant::now();
                let Some(t1) = pared_ms() else {
                    continue;
                };
                let muestra = {
                    let mut reloj = match estado.lock() {
                        Ok(guardia) => guardia,
                        Err(envenenado) => envenenado.into_inner(),
                    };
                    reloj.procesar(t0, t1, servidor_ms, enviado, recibido)
                };
                if let Some(m) = muestra {
                    arena
                        .server_time_offset_ms
                        .store(m.offset_pared_ms, Ordering::Relaxed);
                    if let Some(correccion) = m.correccion_ms {
                        arena
                            .server_time_correction_ms
                            .store(correccion, Ordering::Relaxed);
                    }
                    observador(m);
                }
            }
            Err(e) => {
                eprintln!("⚠️ [NTP-SYNC] Server time fetch failed: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntp_kalman_initialization() {
        let mut kf = NtpKalmanFilter::new();
        assert!(!kf.initialized);
        assert_eq!(kf.current_offset(), None);

        // t0 = 1000, t1 = 1040 (RTT = 40ms, rtt_half = 20ms)
        // local_estimate = 1020
        // server_time = 1050 => raw_offset = +30ms
        let offset = kf.update(1000, 1040, 1050);
        assert_eq!(offset, Some(30));
        assert!(kf.initialized);
        assert_eq!(kf.sample_count, 1);
        assert_eq!(kf.current_offset(), Some(30));
    }

    #[test]
    fn test_ntp_kalman_smooth_filtering() {
        let mut kf = NtpKalmanFilter::new();
        // Inicializar con 50ms
        kf.update(1000, 1020, 1060); // t0=1000, t1=1020, rtt_half=10, server=1060 => offset = 50ms

        // Enviar mediciones con ruido simétrico de +- 6ms
        let offsets = [54, 46, 52, 48, 55, 45, 51, 49];
        for (i, &noisy_off) in offsets.iter().enumerate() {
            let t0 = 2000 + i as u64 * 1000;
            let t1 = t0 + 20; // RTT = 20ms
            let server_time = t0 + 10 + noisy_off as u64;
            let filtered = kf.update(t0, t1, server_time).expect("Valid update");
            // La estimación debe mantenerse suave alrededor de 50ms (+- 2ms)
            assert!(
                (filtered - 50).abs() <= 3,
                "Offset filtered {} divagó más de 3ms de 50",
                filtered
            );
        }
    }

    #[test]
    fn test_ntp_kalman_outlier_rejection() {
        let mut kf = NtpKalmanFilter::new();
        // Inicializar en 100ms con RTT sano de 20ms
        kf.update(1000, 1020, 1110);

        // Inyectar un spike de congestión severa: RTT = 600ms (t1 = t0 + 600)
        // server_time reporta un salto aparente de 400ms
        let outlier_offset = kf.update(2000, 2600, 2500);

        // El filtro debió rechazar la medición y mantener el estado previo (100ms)
        assert_eq!(outlier_offset, Some(100));
        assert_eq!(kf.current_offset(), Some(100));
    }
}
