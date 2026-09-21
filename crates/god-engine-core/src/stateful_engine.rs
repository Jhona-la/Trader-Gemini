use crate::math_kernels::{
    ContinuousVPIN, ExponentialDecayTensor, FundingRateElasticity, ObiAcceleration, RecursiveHurst,
    ShannonEntropy,
};
use feature_engine::OrderFlowTracker;
use std::sync::atomic::{AtomicUsize, Ordering};

pub static DROP_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// C-02 — índices del vector 34D (`get_universal_features`) cuya fuente de datos
/// está MUERTA en producción: hoy sólo [9] (dark_alpha / dex_severity, sin
/// productor MEV-DEX en vivo: `update_macro_features` la recibe como 0.0
/// constante). Contrato del trainer: estos índices van a 0.0 TAMBIÉN en
/// entrenamiento (train_forest) — el modelo no puede aprender a depender de
/// una columna que en serve es constante. Si un productor dex revive la
/// fuente, actualizar este slice y re-entrenar en el mismo cambio.
pub const FEATURES_DEAD_IN_SERVE: &[usize] = &[4, 5, 9, 10];
// B3.35: [4][5][10] (obi_accel) añadidas — el OBI del trainer (aggTrades
// sintético con is_buyer_maker) tiene DISTRIBUCIÓN INCOMPATIBLE con el OBI
// del vivo (depth L2 real). El modelo entrenado con OBI sintético predice
// ~0.503 en vivo (verificado en v41: señal muerta, cero entradas). Zerificar
// en AMBOS lados restaura la transferencia del modelo. Si en el futuro se
// calibra un OBI sintético que matchee la distribución del libro, retirar
// estos índices y re-entrenar.

#[derive(Debug, PartialEq, Clone, Copy, Default)]
/// U-6 (MOTOR UNIVERSAL CONTINUO): variantes Scalping/Swing extirpadas —
/// el régimen del motor continuo es Continuous/Neutral (el régimen MACRO
/// mayor vive en risk_engine::regime::MarketRegime, ortogonal).
pub enum MarketRegime {
    #[default]
    Continuous,
    Neutral,
}

/// Internal recursive state using strictly f64 (Double Precision)
#[repr(C, align(64))]
pub struct StatefulEngine {
    pub order_flow: OrderFlowTracker,
    pub ofi_model: feature_engine::OFIModel,
    pub rsi_rs: f64,
    pub last_price: f64,
    pub v_t: f64,
    pub a_t: f64,
    /// MOD2/7-031 — EMA del centroide espectral (adimensional, actualizada
    /// cada 64 ticks). ANTES vivía en `a_t` y era sobrescrita al tick
    /// siguiente por la aceleración cinemática ($/tick): escritor doble con
    /// un solo sobreviviente. Separada: `a_t` es SÓLO la cinemática
    /// instantánea que alimentan las features (norm_at) y el registry.
    pub a_t_spectral: f64,
    pub last_inst_v: f64,
    pub dir_velocity: f64,
    pub tick_count: u64,
    pub hurst: RecursiveHurst,
    pub obi_accel: ObiAcceleration,
    /// D-688: media y varianza exponenciales del OBI (ruido del libro).
    pub obi_noise: ObiNoise,
    pub fr_elasticity: FundingRateElasticity,
    pub cvpin: ContinuousVPIN,
    pub entropy: ShannonEntropy,
    pub dark_alpha: ExponentialDecayTensor,
    pub last_entropy: f64,
    // Add compatibility properties so swing engine isn't completely broken
    pub ema_fast: f64,
    pub ema_slow: f64,
    pub omni: feature_engine::OmniStrategyEngine,
    pub spectral: feature_engine::SpectralCycleEngine,
    pub multifractal: feature_engine::MultiScaleHurstConfluence,
    pub lead_lag: feature_engine::LeadLagAlphaEngine,
    pub regime: MarketRegime,
    // Native Kline Aggregator
    pub kline_start_ms: u64,
    pub kline_open: f64,
    pub kline_high: f64,
    pub kline_low: f64,
    pub kline_volume: f64,
    pub kline_ema_fast: f64,
    pub kline_ema_slow: f64,
    pub kline_ema_trend: f64,
    pub kline_ema_macro: f64,
    /// B2.3 — bloque espectral publicado (antes calculado y descartado):
    /// frecuencia dominante, potencia y centroide del FFT de 64 retornos;
    /// Hurst multifractal en 3 escalas. + momentum multiescala vía las EMAs
    /// de kline. Es el vector que el F8 dice que DECIDE, ahora alcanzable
    /// por el entrenamiento (train_forest) y por la inferencia, idéntico.
    pub spectral_bin: f32,
    pub spectral_power: f32,
    pub spectral_centroid: f32,
    pub hurst_micro: f32,
    pub hurst_meso: f32,
    pub hurst_macro: f32,
    pub last_scalp_exit_tick: u64,
    pub last_scalp_was_loss: bool,
    pub scalp_loss_streak: u32,
    pub scalp_short_loss_streak: u32,
    pub scalp_long_loss_streak: u32,
    pub last_trade_is_sell: bool,
    /// D-753 — CANTIDAD REAL DEL ÚLTIMO TRADE (activo base), o 0 si el evento
    /// en curso NO es un trade. La escribe `process_event` con el `trade_qty`
    /// que ya recibía de todos los llamadores (vivo, `booktick_replay`,
    /// forense) y la consume `process_tick_dual` para alimentar el VPIN.
    /// Antes se fabricaba un «volumen» del 0,5 % de la PROFUNDIDAD del libro
    /// acotado a [0,01; 10] — una cifra que no es el volumen negociado ni
    /// guarda relación monótona con él.
    pub ultima_cantidad_trade: f64,
    /// D-754 — RELOJ DEL ÚLTIMO CIERRE, en milisegundos de evento. El
    /// enfriamiento se medía en CUENTA DE TICKS: 600 ticks son segundos en un
    /// tape denso y horas en uno ralo, de modo que la misma regla significaba
    /// cosas distintas según el símbolo, la hora y el entorno (vivo vs
    /// forense). El enfriamiento es TIEMPO.
    pub last_scalp_exit_ms: u64,
    /// Reloj de evento más reciente visto por el motor de features. Permite
    /// medir el enfriamiento sin cambiar la firma pública de
    /// `can_open_position`.
    pub last_event_ms: u64,
    /// D-754 — HORIZONTE τ CON EL QUE SE DIMENSIONÓ LA POSICIÓN QUE ACABA DE
    /// CERRARSE. Es la base natural del enfriamiento: tras salir de una
    /// operación de horizonte τ, reentrar antes de que pase τ es reentrar
    /// DENTRO del mismo movimiento que se acaba de abandonar. 0 = todavía no
    /// hubo cierre.
    pub tau_ultimo_cierre_ms: u64,
    /// D-758 — DESVIACIÓN TÍPICA MEDIDA DEL RETORNO POR TICK.
    ///
    /// QUÉ FALTABA: el núcleo comparaba `micro_trend` —el diferencial relativo
    /// de las EMAs de 20 y 200 TICKS— contra fracciones crudas (0,00005;
    /// 0,00010; 0,00040; 0,015). Para tipificar ese diferencial hace falta la
    /// σ del retorno POR TICK, y el motor sólo tenía la σ por VELA DE 1 MINUTO
    /// (vía ATR). Usar la segunda para juzgar la primera es un error de
    /// unidades de varios órdenes de magnitud, así que el literal era la única
    /// salida disponible. Aquí se mide la que faltaba.
    ///
    /// `ObiNoise` es un estimador EWMA genérico de media y varianza (el nombre
    /// viene de su primer uso); se reutiliza tal cual, con su mismo
    /// calentamiento, en vez de duplicar la aritmética.
    pub ruido_retorno_tick: ObiNoise,
}

/// D-758 — PERIODOS DE LAS EMAs POR TICK, como constantes nombradas.
///
/// Son los 20/200 que `process_tick` ya usaba escritos a mano (ver la decisión
/// S-8 documentada ahí: están atados al contrato 34D del vector universal y no
/// se cablean a los genes sin reentrenar). Se nombran para que quien tipifique
/// `micro_trend` use EXACTAMENTE los periodos con los que se calculó, y no una
/// copia que pueda desincronizarse.
pub const TICK_EMA_FAST_BARS: f64 = 20.0;
/// Periodo lento de las EMAs por tick — ver [`TICK_EMA_FAST_BARS`].
pub const TICK_EMA_SLOW_BARS: f64 = 200.0;

impl Default for StatefulEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl StatefulEngine {
    pub fn new() -> Self {
        DROP_COUNTER.fetch_add(1, Ordering::SeqCst);
        Self {
            order_flow: OrderFlowTracker::new(),
            ofi_model: feature_engine::OFIModel::new(),
            rsi_rs: 0.0,
            last_price: 0.0,
            v_t: 0.0,
            a_t: 0.0,
            a_t_spectral: 0.0,
            last_inst_v: 0.0,
            dir_velocity: 0.0,
            tick_count: 0,
            hurst: RecursiveHurst::new(),
            obi_accel: ObiAcceleration::new(),
            obi_noise: ObiNoise::new(),
            fr_elasticity: FundingRateElasticity::new(),
            cvpin: ContinuousVPIN::new(10_000.0), // $10,000 USD rolling bucket size
            entropy: ShannonEntropy::new(),
            dark_alpha: ExponentialDecayTensor::new(10000.0), // 10s half-life
            last_entropy: 0.0,
            ema_fast: 0.0,
            ema_slow: 0.0,
            omni: feature_engine::OmniStrategyEngine::new(),
            spectral: feature_engine::SpectralCycleEngine::new(),
            multifractal: feature_engine::MultiScaleHurstConfluence::new(),
            lead_lag: feature_engine::LeadLagAlphaEngine::new(50),
            regime: MarketRegime::Neutral,
            kline_start_ms: 0,
            kline_open: 0.0,
            kline_high: 0.0,
            kline_low: 0.0,
            kline_volume: 0.0,
            kline_ema_fast: 0.0,
            kline_ema_slow: 0.0,
            kline_ema_trend: 0.0,
            kline_ema_macro: 0.0,
            spectral_bin: 0.0,
            spectral_power: 0.0,
            spectral_centroid: 0.0,
            hurst_micro: 0.5,
            hurst_meso: 0.5,
            hurst_macro: 0.5,
            last_scalp_exit_tick: 0,
            last_scalp_was_loss: false,
            scalp_loss_streak: 0,
            scalp_short_loss_streak: 0,
            scalp_long_loss_streak: 0,
            last_trade_is_sell: false,
            ultima_cantidad_trade: 0.0,
            last_scalp_exit_ms: 0,
            last_event_ms: 0,
            tau_ultimo_cierre_ms: 0,
            ruido_retorno_tick: ObiNoise::new(),
        }
    }

    /// D-758 — σ MEDIDA DEL RETORNO POR TICK, o `None` durante el
    /// calentamiento. Sin ella no hay forma honesta de tipificar `micro_trend`
    /// —que vive en la escala del TICK— y el núcleo tenía que compararlo
    /// contra fracciones escritas a mano.
    #[inline]
    pub fn sigma_retorno_tick(&self) -> Option<f64> {
        self.ruido_retorno_tick.sd().filter(|s| s.is_finite() && *s > 0.0)
    }

    /// D-754 — CUÁNTAS VECES HA CABIDO UNA DUPLICACIÓN DEL ENFRIAMIENTO BASE
    /// EN EL TIEMPO TRANSCURRIDO.
    ///
    /// Es la inversa del retroceso binario: si una racha de `k` exige esperar
    /// `base·2^k`, entonces haber esperado `base·2^m` amortiza `m` niveles de
    /// racha. No hay ventana de olvido inventada (antes: 7 200 y 18 000
    /// TICKS); el olvido es la propia escalera, leída al revés.
    #[inline(always)]
    fn niveles_amortizados(transcurrido_ms: f64, base_ms: f64) -> u32 {
        if !(transcurrido_ms > 0.0) || !(base_ms > 0.0) || !transcurrido_ms.is_finite() {
            return 0;
        }
        let m = (1.0 + transcurrido_ms / base_ms).log2().floor();
        if m.is_finite() && m > 0.0 {
            m.min(u32::MAX as f64) as u32
        } else {
            0
        }
    }

    /// Racha efectiva tras amortizar por el tiempo transcurrido desde el
    /// último cierre.
    #[inline(always)]
    fn racha_amortizada(&self, cruda: u32, base_ms: f64) -> u32 {
        if self.last_scalp_exit_ms == 0 {
            return 0;
        }
        let transcurrido =
            self.last_event_ms.saturating_sub(self.last_scalp_exit_ms) as f64;
        cruda.saturating_sub(Self::niveles_amortizados(transcurrido, base_ms))
    }

    /// D-754 — ENFRIAMIENTO MEDIDO EN TIEMPO, NO EN CUENTA DE EVENTOS.
    ///
    /// QUÉ ESTABA MAL: el enfriamiento se contaba en TICKS (600 de base, con
    /// ventanas de olvido de 7 200 y 18 000). En un tape denso —BTC en hora
    /// americana, o el forense leyendo aggTrades— 600 ticks son segundos; en
    /// uno ralo son horas. La MISMA regla producía enfriamientos que diferían
    /// en tres órdenes de magnitud según el símbolo, la hora y el entorno, y
    /// el backtest medía por tanto una política distinta de la que corre en
    /// vivo. Además, el escalón `v_t > 0,0015` comparaba un True Range en
    /// UNIDADES DE PRECIO contra una fracción: para cualquier activo de más de
    /// 1,5 USD era verdadero siempre, así que la rama «×2» no existía.
    ///
    /// QUÉ GARANTIZA: el enfriamiento es un múltiplo del horizonte dominante
    /// τ que el llamador mide —el tiempo que el propio mercado tarda en
    /// descorrelacionarse a la escala en la que el motor opera— y crece por
    /// retroceso binario (`base·2^racha`), que es la escalera canónica y no
    /// tiene escalones inventados. El techo es `TAU_ANCHOR_SLOW_MS`: más allá
    /// del horizonte más lento que el motor tiene permitido operar, esperar ya
    /// no es enfriarse sino estar apagado.
    #[inline(always)]
    pub fn can_open_position_ms(&self, enfriamiento_base_ms: f64) -> bool {
        if !enfriamiento_base_ms.is_finite() || enfriamiento_base_ms <= 0.0 {
            // Sin horizonte medido no hay enfriamiento que imponer: negar la
            // entrada sería inventar una regla con datos que no existen.
            return true;
        }
        if self.last_scalp_exit_ms == 0 {
            return true; // aún no hubo cierre del que enfriarse
        }
        let transcurrido =
            self.last_event_ms.saturating_sub(self.last_scalp_exit_ms) as f64;
        let racha = self.racha_amortizada(self.scalp_loss_streak, enfriamiento_base_ms);
        let requerido = (enfriamiento_base_ms * (racha as f64).exp2())
            .min(quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS);
        transcurrido >= requerido
    }

    /// Racha de pérdidas ACTIVA en una dirección, amortizada por el tiempo
    /// transcurrido (D-754: antes por cuenta de ticks, con el mismo defecto de
    /// escala que el enfriamiento).
    #[inline(always)]
    pub fn get_active_directional_streak_ms(
        &self,
        is_long: bool,
        enfriamiento_base_ms: f64,
    ) -> u32 {
        let cruda = if is_long {
            self.scalp_long_loss_streak
        } else {
            self.scalp_short_loss_streak
        };
        if !enfriamiento_base_ms.is_finite() || enfriamiento_base_ms <= 0.0 {
            return cruda;
        }
        self.racha_amortizada(cruda, enfriamiento_base_ms)
    }

    /// Centra las predicciones ML en 0.50 con rango [-1.0, 1.0] en O(1).
    /// MOD2/7-002 (INFORME DECIMOCUARTO): los campos `ml_prob_ewma`/`ml_prob_var`
    /// (la supuesta "normalización adaptativa") se eliminaron — declarados,
    /// inicializados y jamás leídos: estado fantasma con contrato falsamente
    /// documentado. Este mapeo es estático por diseño.
    #[inline(always)]
    pub fn update_ml_prediction(&mut self, ml_prob: f64) -> f64 {
        if !ml_prob.is_finite() || ml_prob < 0.0 || ml_prob > 1.0 {
            return 0.0;
        }
        ((ml_prob - 0.50) * 2.0).clamp(-1.0, 1.0)
    }

    /// Flushes all internal buffers. Used to auto-heal time-series glitches after network disconnects.
    pub fn reset(&mut self) {
        self.order_flow = OrderFlowTracker::new();
        self.ofi_model = feature_engine::OFIModel::new();
        self.rsi_rs = 0.0;
        self.last_price = 0.0;
        self.v_t = 0.0;
        self.a_t = 0.0;
        self.a_t_spectral = 0.0;
        self.tick_count = 0;
        self.hurst = RecursiveHurst::new();
        self.obi_accel = ObiAcceleration::new();
        self.obi_noise = ObiNoise::new();
        self.fr_elasticity = FundingRateElasticity::new();
        self.cvpin = ContinuousVPIN::new(10_000.0);
        self.entropy = ShannonEntropy::new();
        self.dark_alpha = ExponentialDecayTensor::new(10000.0);
        self.last_entropy = 0.0;
        self.ema_fast = 0.0;
        self.ema_slow = 0.0;
        self.omni = feature_engine::OmniStrategyEngine::new();
        self.spectral = feature_engine::SpectralCycleEngine::new();
        self.multifractal = feature_engine::MultiScaleHurstConfluence::new();
        self.lead_lag = feature_engine::LeadLagAlphaEngine::new(50);
        self.regime = MarketRegime::Neutral;
        self.kline_start_ms = 0;
        self.kline_open = 0.0;
        self.kline_high = 0.0;
        self.kline_low = 0.0;
        self.kline_volume = 0.0;
        self.kline_ema_fast = 0.0;
        self.kline_ema_slow = 0.0;
        self.kline_ema_trend = 0.0;
        self.kline_ema_macro = 0.0;
        // D7 / MOD6/8-004: el bloque espectral publicado también se limpia —
        // tras un reset (reconexión WS) los buffers del espectro quedaron
        // vacíos y servir los valores del período anterior es servir historia
        // congelada. Neutro hasta que 64 retornos nuevos llenen el anillo.
        self.spectral_bin = 0.0;
        self.spectral_power = 0.0;
        self.spectral_centroid = 0.0;
        self.hurst_micro = 0.5;
        self.hurst_meso = 0.5;
        self.hurst_macro = 0.5;
        // D-753/D-754: el volumen del último trade y los relojes de
        // enfriamiento también son estado del feed. Tras una reconexión no
        // hay trade reciente ni continuidad temporal que defender.
        self.ultima_cantidad_trade = 0.0;
        self.last_scalp_exit_ms = 0;
        self.last_event_ms = 0;
        self.tau_ultimo_cierre_ms = 0;
        // D-758: la σ por tick es una propiedad del feed vivo. Tras una
        // reconexión vuelve a calentarse desde cero, igual que el resto de
        // estimadores, para no tipificar con una escala de otro tramo.
        self.ruido_retorno_tick = ObiNoise::new();
    }

    /// Processes a new tick internally in f64
    pub fn process_tick(&mut self, price: f64, _volume: f64, event_time_ms: u64) {
        if price <= 0.0 || !price.is_finite() {
            return;
        }
        // D-754: el reloj del motor de features. El enfriamiento y el olvido
        // de rachas se miden contra ÉL, no contra `tick_count`.
        if event_time_ms > self.last_event_ms {
            self.last_event_ms = event_time_ms;
        }
        if self.last_price == 0.0 {
            self.ema_fast = price;
            self.ema_slow = price;
        } else {
            // S-8 — DECISIÓN DOCUMENTADA: los genes ema_fast_period /
            // ema_slow_period (~12.5/~25.1) NO se cablean aquí aunque
            // existan. Estos 20/200 alimentan la feature [0] del contrato
            // 34D del vector universal: cambiarlos SOLO en vivo rompería la
            // paridad train/serve que B3.30 y B3.35 pagaron caro por
            // restaurar (distribución de la feature distinta ⇒ el modelo
            // sirve ruido). Condición para activarlos: trainer leyendo el
            // MISMO genoma del símbolo + retrain completo del roster en el
            // MISMO cambio. Sustituir el ladder por signal_at(τ) del
            // espectro exige lo mismo.
            // D-758: los mismos 20/200, ahora nombrados, para que quien
            // tipifique `micro_trend` use los periodos REALES de estas EMAs.
            let alpha_fast = 2.0 / (TICK_EMA_FAST_BARS + 1.0);
            let alpha_slow = 2.0 / (TICK_EMA_SLOW_BARS + 1.0);

            self.ema_fast = (price - self.ema_fast) * alpha_fast + self.ema_fast;
            self.ema_slow = (price - self.ema_slow) * alpha_slow + self.ema_slow;

            let diff = (price - self.last_price).abs();
            let norm_return = (price - self.last_price) / self.last_price;
            // D-758: la σ del retorno POR TICK se mide aquí, sobre el mismo
            // retorno que alimenta la entropía. Es la escala que faltaba para
            // poder juzgar `micro_trend` sin literales.
            self.ruido_retorno_tick.update(norm_return);
            self.last_entropy = self.entropy.update(norm_return);
            // D7 / MOD6/8-004 (INFORME DECIMOCUARTO) — ESPECTRO VIVO EN
            // process_tick: en producción ESTE es el camino que corre
            // (process_event → process_tick_dual → process_tick);
            // process_kline sólo alimenta el warmup REST de arranque. El
            // SpectralCycleEngine acumula AQUÍ el RETORNO de cada tick y el
            // FFT Radix-2 se re-analiza cada 64 retornos (contador
            // tick_count); el multifractal se actualiza con el PRECIO de
            // cada tick. Las 6 features espectrales [0..6] del vector ML
            // (bin/potencia/centroide FFT + Hurst micro/meso/macro) viven
            // por esta vía — no dependen de klines cerrados.
            self.spectral.push(norm_return);
            // D-434: Invocar análisis espectral FFT Radix-2 periódicamente cada 64 ticks
            if self.tick_count % 64 == 0 {
                let (dominant_bin, max_power, centroid) = self.spectral.analyze_spectrum();
                if max_power > 0.0 && centroid.is_finite() {
                    // MOD2/7-031: el EMA del centroide espectral vive en su
                    // PROPIO campo — antes escribía `a_t` y la cinemática del
                    // tick siguiente lo borraba (escritor doble, un solo
                    // sobreviviente). Observabilidad del espectro; no toca el
                    // contrato 48D (que consume spectral_centroid directo).
                    self.a_t_spectral = self.a_t_spectral * 0.95 + (centroid * 0.001) * 0.05;
                }
                // B2.3: el espectro ya se calculaba aquí y se DESCARTABA
                // (capacidad fantasma). Ahora se publica para el vector ML
                // — mismas features en vivo y en entrenamiento (paridad 1:1).
                self.spectral_bin = dominant_bin as f32;
                self.spectral_power = max_power as f32;
                self.spectral_centroid = centroid as f32;
            }
            let (h_mic, h_mes, h_mac, _score, _micro_p, _macro_p) = self.multifractal.update(price);
            self.hurst_micro = h_mic as f32;
            self.hurst_meso = h_mes as f32;
            self.hurst_macro = h_mac as f32;
            self.regime = MarketRegime::Continuous;

            // Tick-level instantaneous velocity & acceleration
            let inst_v = diff;
            // FIX D-52: Aceleración cinemática dimensionalmente correcta a = (v_t - v_{t-1}) / dt con filtro EMA
            let raw_a = inst_v - self.last_inst_v;
            self.a_t = self.a_t * 0.70 + raw_a * 0.30;
            self.last_inst_v = inst_v;
            // Velocidad direccional suavizada (EMA de 10 ticks)
            self.dir_velocity = self.dir_velocity * 0.85 + inst_v * 0.15;
        }

        // D-615b: el Hurst YA NO se alimenta por evento (ver el cierre de la
        // vela interna de 1 minuto, más abajo).
        let notional_usd = if _volume > 0.0 && price > 0.0 {
            _volume * price
        } else {
            _volume
        };
        // D-110: Regla canónica Lee-Ready (1991): en ticks planos (price == last_price),
        // propagar la dirección del tick previo en lugar de falsear hacia compra sistemática
        let is_sell = if self.last_price > 0.0 {
            if price < self.last_price {
                self.last_trade_is_sell = true;
                true
            } else if price > self.last_price {
                self.last_trade_is_sell = false;
                false
            } else {
                self.last_trade_is_sell
            }
        } else {
            false
        };
        // D-753 — EL RELOJ DE VOLUMEN DEL VPIN SÓLO AVANZA CON VOLUMEN
        // NEGOCIADO. `_volume` es ahora la cantidad REAL del trade (0 cuando
        // el evento es un depth o un kline). Antes llegaba una cifra
        // fabricada —el 0,5 % de la PROFUNDIDAD del libro, acotada a
        // [0,01; 10]—, de modo que cada snapshot del libro cerraba buckets de
        // un VPIN que se supone construido sobre desequilibrio de flujo
        // NEGOCIADO: el indicador medía la frecuencia de actualización del
        // libro, no la toxicidad del flujo. Con volumen 0 el bucket no avanza
        // y el VPIN conserva su último valor, que es lo correcto: entre dos
        // trades no hay información nueva de flujo.
        if notional_usd > 0.0 {
            self.cvpin.update(notional_usd, is_sell);
        }

        if self.kline_start_ms == 0 {
            self.kline_start_ms = event_time_ms;
            self.kline_open = price;
            self.kline_high = price;
            self.kline_low = price;
            self.kline_volume = _volume;
            if self.v_t == 0.0 && price > 0.0 {
                self.v_t = price * 0.005; // Fallback inicial 50 bps True Range
            }
        } else {
            self.kline_high = self.kline_high.max(price);
            self.kline_low = self.kline_low.min(price);
            self.kline_volume += _volume;

            // FIX #1206: Actualizar features Omni en tiempo real en cada tick para eliminar desfase de 59s en la inferencia HFT
            self.omni.update(price, self.kline_high, self.kline_low);

            // D-435 & D-437: Actualización continua intra-vela del True Range (captura expansiones de volatilidad sin colapso a spread de tick)
            let intra_candle_tr = (self.kline_high - self.kline_low).max(price * 0.0010);
            if intra_candle_tr > self.v_t || self.v_t == 0.0 || !self.v_t.is_finite() {
                self.v_t = intra_candle_tr;
            }

            // Generate 1-minute Kline internally (60,000 ms) and update True Range EMA & Trend EMAs
            if event_time_ms.saturating_sub(self.kline_start_ms) >= 60000 {
                // D-615b (DÉCIMA OLA) — HURST MUESTREADO POR RELOJ.
                //
                // Se alimentaba en cada evento. Un estimador de memoria sobre
                // retornos por evento mide la microestructura del FEED, no la
                // del activo: en producción llegan eventos cada ~100 ms y en el
                // backtest forense barras de ~15 s, así que el mismo mercado
                // producía exponentes distintos en cada entorno (0,22 en el
                // forense). Y ese exponente gobierna la ley de escala del TP/SL,
                // el régimen, el Kelly, el apalancamiento, la duración y siete
                // umbrales de decisión.
                //
                // Se muestrea al cierre de la vela interna de 1 minuto: la
                // escala en la que se mide el ATR (v_t) y la de las velas REST
                // del calentamiento (`interval=1m`, vía `process_kline`). Así
                // el exponente describe la difusión entre 1 minuto y el
                // horizonte, que es exactamente lo que `tp_sl` necesita, y es
                // idéntico en producción y en backtest.
                self.hurst.update(price);
                // FIX #608: True Range robusto y no nulo para evitar distorsiones en SL dinámico
                let tr = (self.kline_high - self.kline_low).max(price * 0.0010);
                self.v_t = if self.v_t == 0.0 || !self.v_t.is_finite() {
                    tr
                } else {
                    (self.v_t * 0.85 + tr * 0.15).max(price * 0.0010)
                };

                // Actualizar EMAs de tendencia macro de 1 minuto (EMA 9 y EMA 21), tendencia intermedia (EMA 120 ~ 2 horas) y tendencia secular (EMA 720 ~ 12 horas)
                let alpha_k_fast = 2.0 / (9.0 + 1.0);
                let alpha_k_slow = 2.0 / (21.0 + 1.0);
                let alpha_k_trend = 2.0 / (120.0 + 1.0);
                let alpha_k_macro = 2.0 / (720.0 + 1.0);
                if self.kline_ema_fast == 0.0 {
                    self.kline_ema_fast = price;
                    self.kline_ema_slow = price;
                    self.kline_ema_trend = price;
                    self.kline_ema_macro = price;
                } else {
                    self.kline_ema_fast =
                        (price - self.kline_ema_fast) * alpha_k_fast + self.kline_ema_fast;
                    self.kline_ema_slow =
                        (price - self.kline_ema_slow) * alpha_k_slow + self.kline_ema_slow;
                    self.kline_ema_trend =
                        (price - self.kline_ema_trend) * alpha_k_trend + self.kline_ema_trend;
                    self.kline_ema_macro =
                        (price - self.kline_ema_macro) * alpha_k_macro + self.kline_ema_macro;
                }

                self.kline_start_ms = event_time_ms;
                self.kline_open = price;
                self.kline_high = price;
                self.kline_low = price;
                self.kline_volume = _volume;
            }
        }

        self.last_price = price;
        self.tick_count += 1;
    }

    pub fn update_trade_flow(&mut self, volume: f64, is_buyer_maker: bool) {
        self.order_flow.update(volume, is_buyer_maker);
    }

    /// Updates the Order Flow Imbalance (OFI) predictive model
    pub fn update_ofi(
        &mut self,
        bid_price: f64,
        ask_price: f64,
        bid_qty: f64,
        ask_qty: f64,
    ) -> f64 {
        self.ofi_model
            .update(bid_price, ask_price, bid_qty, ask_qty)
    }

    pub fn process_kline(&mut self, _open: f64, high: f64, low: f64, close: f64, _volume: f64) {
        // FIX #665: Descarte preventivo de klines con precios corruptos o no finitos
        // FIX: Erradicación del Feature Leakage (Ceguera Causal)
        // Usar los altos y bajos de la vela ANTERIOR para el cálculo actual de features de IA.
        // Si el modelo ve el high/low de esta misma vela, el backtest hace trampa leyendo el futuro.
        let prev_high = if self.kline_high > 0.0 {
            self.kline_high
        } else {
            high
        };
        let prev_low = if self.kline_low > 0.0 {
            self.kline_low
        } else {
            low
        };

        self.omni.update(close, prev_high, prev_low);

        // Guardar estado futuro para la próxima evaluación causal
        self.kline_high = high;
        self.kline_low = low;
        self.hurst.update(close);
        if self.last_price > 0.0 {
            self.spectral
                .push((close - self.last_price) / self.last_price);
        }
        let (_h_mic, _h_mes, _h_mac, _score, _micro_p, _macro_p) = self.multifractal.update(close);
        self.regime = MarketRegime::Continuous;
        // MOD2/7-032: `ema_fast`/`ema_slow` ya NO se escriben aquí. Tenían
        // DOS kernels: 20/200 (process_tick, el camino del trainer y del
        // backtest) y 12/26 (esta función, warmup REST 1m del vivo). El
        // estado era una quimera de dos escalas y una ruptura de paridad
        // train/serve en la feature (ema_fast−ema_slow)/ema_slow del vector
        // 34D: el vivo arrancaba con EMA de velas que el entrenamiento jamás
        // vio. Kernel ÚNICO 20/200 por ticks — el primer tick vivo siembra
        // (ema_fast==0 ⇒ =precio) y converge solo, idéntico a train.
        self.last_price = close;

        // FIX #624: True Range robusto y no nulo en kline processing
        let tr = (high - low).max(close * 0.0005);
        self.v_t = if self.v_t == 0.0 || !self.v_t.is_finite() {
            tr
        } else {
            (self.v_t * 0.8 + tr * 0.2).max(close * 0.0005)
        };
    }

    pub fn update_macro_features(
        &mut self,
        obi: f64,
        funding_rate: f64,
        dex_severity: f64,
        ts_ms: u64,
    ) {
        // FIX #665: Sanitizar macro features
        if !obi.is_finite() || !funding_rate.is_finite() || !dex_severity.is_finite() {
            return;
        }

        self.obi_accel.update(obi);
        self.obi_noise.update(obi);
        self.fr_elasticity.update(funding_rate, self.last_price);
        self.dark_alpha.apply_event(dex_severity, ts_ms);
    }

    pub fn update_macro_flow(
        &mut self,
        funding_rate: f64,
        dex_severity: f64,
        ts_ms: u64,
    ) {
        self.fr_elasticity.update(funding_rate, self.last_price);
        self.dark_alpha.apply_event(dex_severity, ts_ms);
    }

    pub fn get_market_regime(&self) -> MarketRegime {
        MarketRegime::Continuous
    }

    pub fn get_features(&self) -> [f32; 12] {
        let price_change = if self.last_price != 0.0 && self.ema_slow != 0.0 {
            (self.ema_fast - self.ema_slow) / self.ema_slow
        } else {
            0.0
        };

        let hurst = self.hurst.current();
        let ofi = self.ofi_model.ema_ofi;
        let vol_delta = self.order_flow.get_volume_delta_ratio();
        let norm_vt = if self.last_price > 0.0 {
            (self.v_t / self.last_price).clamp(0.0, 1.0)
        } else {
            0.005
        };
        let norm_at = if self.last_price > 0.0 {
            (self.a_t / self.last_price).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        [
            price_change as f32,
            hurst as f32,
            ofi as f32,
            norm_vt as f32,
            self.obi_accel.prev_obi_velocity as f32,
            self.obi_accel.prev_obi as f32,
            vol_delta as f32,
            (((self.cvpin.buy_volume - self.cvpin.sell_volume)
                / (self.cvpin.buy_volume + self.cvpin.sell_volume).max(1e-6))
            .clamp(-1.0, 1.0)) as f32,
            self.last_entropy as f32,
            self.dark_alpha.current_severity as f32,
            self.obi_accel.accel as f32,
            norm_at as f32,
        ]
    }

    /// C-02 (INFORME DECIMOCUARTO) — mapa vivo/muerto del contrato 34D:
    /// VIVAS en train y serve (microestructura/precio del PROPIO símbolo):
    /// [0..4], [6..9), [10..12) y las 22 omni [12..34] (RSI/MACD/BB/ATR/Fib
    /// de precio — NO del tensor 54D del omni_multiplexer).
    /// MUERTA en serve y en train: [9] dark_alpha (dex_severity=0.0, sin
    /// productor). VIVAS en serve (libro real vía update_macro_features),
    /// muertas en train salvo que el trainer llame update_macro_features:
    /// [4], [5], [10] (obi_accel). Ver FEATURES_DEAD_IN_SERVE.
    pub fn get_universal_features(&self) -> [f32; 34] {
        let micro = self.get_features();
        let omni_feats = self.omni.extract_features();

        // CERT-M2-H02: zerificar las dims MUERTAS en SERVE también (no
        // sólo en el trainer). B3.35 comentaba "zerificadas en AMBOS
        // lados" pero el serve seguía alimentando obi_accel vivo en dims
        // [4][5][10] mientras el trainer las zerificaba — cualquier
        // modelo pre-B3.35 con splits en esas dims servía una
        // distribución que nunca vio en entrenamiento. FEATURES_DEAD_IN_
        // SERVE es la fuente única de verdad del mapa vivo/muerto.
        let mut result = [0f32; 34];
        let mut raw: [f32; 34] = [
            micro[0],
            micro[1],
            micro[2],
            micro[3],
            micro[4],
            micro[5],
            micro[6],
            micro[7],
            micro[8],
            micro[9],
            micro[10],
            micro[11],
            // Omni Features (22 slots)
            omni_feats[0],
            omni_feats[1],
            omni_feats[2],
            omni_feats[3],
            omni_feats[4],
            omni_feats[5],
            omni_feats[6],
            omni_feats[7],
            omni_feats[8],
            omni_feats[9],
            omni_feats[10],
            omni_feats[11],
            omni_feats[12],
            omni_feats[13],
            omni_feats[14],
            omni_feats[15],
            omni_feats[16],
            omni_feats[17],
            omni_feats[18],
            omni_feats[19],
            omni_feats[20],
            omni_feats[21],
        ];
        // CERT-M2-H02: aplicar el zerificado del contrato EN SERVE.
        for &d in FEATURES_DEAD_IN_SERVE {
            if d < 34 {
                raw[d] = 0.0;
            }
        }
        result = raw;
        result
    }

    /// Returns ATR as a percentage of last price for Stop Loss scaling
    pub fn get_atr_pct(&self) -> f64 {
        if self.last_price > 0.0 {
            self.v_t / self.last_price
        } else {
            0.0
        }
    }

    /// B2.3 — bloque espectral (10D) para el vector ML. Se concatena a las
    /// 34 swing features TANTO en inferencia como en train_forest: la
    /// directriz F8 (el espectro DECIDE) entra así al aprendizaje. Los
    /// árboles existentes no se rompen: sus splits viven en índices <34 y
    /// los nuevos árboles pueden usar 34..44.
    ///
    /// [0..3] FFT de 64 retornos: bin dominante /32, ln(1+potencia),
    /// centroide /32 — frecuencia y energía del ciclo vivo.
    /// [3..6] Hurst multifractal micro(10)/meso(25)/macro(50) — persistencia
    /// por escala: >0.55 tendencial, <0.45 mean-reverting.
    /// [6..10] momentum multiescala: ln(p/EMA_kline)×100 en 4 escalas
    /// crecientes (posición del precio dentro de su tendencia por escala).
    pub fn get_spectral_ml_features(&self) -> [f32; 10] {
        let p = self.last_price;
        let dev = |ema: f64| -> f32 {
            if p > 0.0 && ema > 0.0 {
                (((p / ema).ln() * 100.0) as f32).clamp(-20.0, 20.0)
            } else {
                0.0
            }
        };
        [
            (self.spectral_bin / 32.0).clamp(0.0, 1.0),
            (1.0 + self.spectral_power as f64).ln() as f32,
            (self.spectral_centroid / 32.0).clamp(0.0, 1.0),
            self.hurst_micro.clamp(0.0, 1.0),
            self.hurst_meso.clamp(0.0, 1.0),
            self.hurst_macro.clamp(0.0, 1.0),
            dev(self.kline_ema_fast),
            dev(self.kline_ema_slow),
            dev(self.kline_ema_trend),
            dev(self.kline_ema_macro),
        ]
    }

    /// Retorna la pendiente relativa del micro-trend (Tick-level EMA 12 vs 26)
    #[inline(always)]
    pub fn get_micro_trend(&self) -> f64 {
        if self.ema_slow > 0.0 {
            (self.ema_fast - self.ema_slow) / self.ema_slow
        } else {
            0.0
        }
    }

    /// Retorna la pendiente relativa del macro-trend (1-Minute Kline EMA 9 vs 21)
    #[inline(always)]
    pub fn get_macro_trend(&self) -> f64 {
        if self.kline_ema_slow > 0.0 {
            (self.kline_ema_fast - self.kline_ema_slow) / self.kline_ema_slow
        } else if self.ema_slow > 0.0 {
            (self.ema_fast - self.ema_slow) / self.ema_slow
        } else {
            0.0
        }
    }

    /// Retorna la pendiente del macro-trend de orden superior (Price vs 2-Hour EMA 120)
    #[inline(always)]
    pub fn get_higher_trend(&self) -> f64 {
        if self.kline_ema_trend > 0.0 {
            (self.last_price - self.kline_ema_trend) / self.kline_ema_trend
        } else if self.kline_ema_slow > 0.0 {
            (self.last_price - self.kline_ema_slow) / self.kline_ema_slow
        } else {
            0.0
        }
    }

    /// Retorna la pendiente del macro-trend secular (Price vs 12-Hour EMA 720)
    #[inline(always)]
    pub fn get_secular_trend(&self) -> f64 {
        if self.kline_ema_macro > 0.0 {
            (self.last_price - self.kline_ema_macro) / self.kline_ema_macro
        } else if self.kline_ema_trend > 0.0 {
            (self.last_price - self.kline_ema_trend) / self.kline_ema_trend
        } else {
            0.0
        }
    }

    /// Determina si la estructura de mercado es inequívocamente bajista (Death Cross de orden superior y precio bajo EMA de tendencia)
    #[inline(always)]
    pub fn is_macro_bear(&self) -> bool {
        if self.kline_ema_fast > 0.0 && self.kline_ema_slow > 0.0 && self.kline_ema_fast > self.kline_ema_slow {
            return false; // El momentum rápido (EMA 9 > EMA 21) es alcista: régimen no bajista
        }
        if self.kline_ema_macro > 0.0 && self.kline_ema_trend > 0.0 {
            self.last_price < self.kline_ema_trend
                && self.kline_ema_slow < self.kline_ema_trend
                && (self.last_price < self.kline_ema_macro || self.kline_ema_trend < self.kline_ema_macro)
        } else if self.kline_ema_trend > 0.0 && self.kline_ema_slow > 0.0 {
            self.last_price < self.kline_ema_trend && self.kline_ema_slow < self.kline_ema_trend
        } else if self.kline_ema_slow > 0.0 {
            self.last_price < self.kline_ema_slow
        } else {
            false
        }
    }

    /// Determina si la estructura de mercado es inequívocamente alcista (Golden Cross y Precio sobre EMA 120 y 720)
    #[inline(always)]
    pub fn is_macro_bull(&self) -> bool {
        if self.kline_ema_fast > 0.0 && self.kline_ema_slow > 0.0 && self.kline_ema_fast < self.kline_ema_slow {
            return false; // El momentum rápido (EMA 9 < EMA 21) es bajista: régimen no alcista
        }
        if self.kline_ema_macro > 0.0 && self.kline_ema_trend > 0.0 {
            self.last_price > self.kline_ema_trend
                && self.kline_ema_slow > self.kline_ema_trend
                && (self.last_price > self.kline_ema_macro || self.kline_ema_trend > self.kline_ema_macro)
        } else if self.kline_ema_trend > 0.0 && self.kline_ema_slow > 0.0 {
            self.last_price > self.kline_ema_trend && self.kline_ema_slow > self.kline_ema_trend
        } else if self.kline_ema_slow > 0.0 {
            self.last_price > self.kline_ema_slow
        } else {
            false
        }
    }

    /// Projects the state out to the f32 barrier (576 bytes / 144 floats)
    pub fn export_f32(&mut self, out: &mut [f32; 144]) {
        out[0] = self.ema_fast as f32;
        out[1] = self.ema_slow as f32;
        out[2] = self.rsi_rs as f32;
        out[3] = self.last_price as f32;
        // Read Hurst for external observability — NO mutation (use current(), not update())
        out[4] = self.hurst.current() as f32;
    }
}

/// D-688 (DÉCIMA OLA) — RUIDO DEL DESEQUILIBRIO DEL LIBRO.
///
/// El escudo de libro L2 vetaba con `|OBI| > 0,10`: un umbral absoluto que
/// significa cosas distintas en un libro profundo y estable y en uno fino y
/// ruidoso. La desviación típica del OBI se estima con una media y una varianza
/// exponenciales sobre `OBI_NOISE_EVENTS` eventos —la misma escala que la EMA
/// lenta de ticks del motor (α = 2/201)— y el escudo exige que la presión sea
/// significativa frente a ese ruido.
#[derive(Debug, Clone, Copy)]
pub struct ObiNoise {
    mean: f64,
    var: f64,
    count: u32,
}

/// Eventos de la ventana exponencial y mínimo para considerar la estimación.
pub const OBI_NOISE_EVENTS: u32 = 200;

impl Default for ObiNoise {
    fn default() -> Self {
        Self::new()
    }
}

impl ObiNoise {
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            var: 0.0,
            count: 0,
        }
    }

    #[inline]
    pub fn update(&mut self, obi: f64) {
        if !obi.is_finite() {
            return;
        }
        if self.count == 0 {
            self.mean = obi;
            self.var = 0.0;
        } else {
            let alpha = 2.0 / (OBI_NOISE_EVENTS as f64 + 1.0);
            let delta = obi - self.mean;
            self.mean += alpha * delta;
            self.var = (1.0 - alpha) * (self.var + alpha * delta * delta);
        }
        self.count = self.count.saturating_add(1);
    }

    /// Desviación típica del OBI, o `None` durante el calentamiento.
    #[inline]
    pub fn sd(&self) -> Option<f64> {
        if self.count < OBI_NOISE_EVENTS {
            None
        } else {
            Some(self.var.max(0.0).sqrt())
        }
    }
}

impl Drop for StatefulEngine {
    fn drop(&mut self) {
        DROP_COUNTER.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// B2.3: el bloque espectral debe ser finito, estar en rango y VIVIR
    /// (cambiar con ticks con estructura cíclica) — la paranoia nace del
    /// diagnóstico "34 features congeladas sin edge".
    #[test]
    fn test_spectral_ml_features_finite_and_alive() {
        let mut engine = StatefulEngine::new();
        // Onda lenta + ruido: 8 minutos de ticks de 100ms con ciclo de 64s
        let mut ts: u64 = 1_789_000_000_000;
        for i in 0..4800 {
            let phase = (i as f64) * 0.1; // ~6.28 rad cada 63 ticks
            let price = 100.0 + (phase).sin() * 2.0;
            engine.process_tick(price, 1.0, ts);
            ts += 100;
        }
        let f = engine.get_spectral_ml_features();
        assert_eq!(f.len(), 10);
        for v in &f {
            assert!(v.is_finite(), "feature espectral no finita: {:?}", f);
        }
        assert!((f[3] >= 0.0) && (f[3] <= 1.0), "hurst micro fuera de rango");
        // El momentum multiescala debe haberse movido (EMAs de kline vivas):
        // en el pico de la onda el precio supera su EMA macro.
        assert!(
            f[6].abs() + f[7].abs() + f[8].abs() + f[9].abs() > 1e-3,
            "momentum multiescala muerto: {:?}",
            f
        );
    }

    /// D7 / MOD6/8-004 (INFORME DECIMOCUARTO): el espectro debe seguir VIVO
    /// tras el arranque con SOLO ticks — en vivo `process_kline` corre
    /// únicamente en el warmup REST inicial; todo lo demás es process_tick.
    /// Warmup con klines → tramo de ticks con ciclo rápido → las features
    /// espectrales deben REPUBLICARSE (moverse de sus valores post-warmup),
    /// y un cambio de régimen del ciclo bajo ticks debe volver a moverlas.
    #[test]
    fn d7_espectro_vivo_tras_warmup_solo_con_ticks() {
        let mut engine = StatefulEngine::new();
        // Warmup REST de arranque: 120 klines de 1m con ciclo lento (~51 velas).
        let mut ts: u64 = 1_789_000_000_000;
        for i in 0..120u64 {
            let o = 100.0 + ((i % 51) as f64).sin() * 3.0;
            let c = 100.0 + (((i + 1) % 51) as f64).sin() * 3.0;
            let h = o.max(c) + 0.4;
            let l = o.min(c) - 0.4;
            engine.process_kline(o, h, l, c, 50.0);
        }
        let post_warmup = engine.get_spectral_ml_features();

        // VIVO: SOLO process_tick. Ciclo rápido de 8 ticks (~2 ventanas de
        // FFT completas: 128 retornos nuevos).
        for i in 0..128 {
            let phase = (i as f64) * (std::f64::consts::TAU / 8.0);
            let price = 100.0 + phase.sin() * 2.0;
            engine.process_tick(price, 1.0, ts);
            ts += 100;
        }
        let post_live = engine.get_spectral_ml_features();

        // El bloque FFT [0..3] debe haberse republicado con el ciclo nuevo.
        let fft_changed = (0..3).any(|i| {
            (post_warmup[i] - post_live[i]).abs() > 1e-6
        });
        assert!(
            fft_changed,
            "FFT congelado post-arranque: {:?} vs {:?}",
            &post_warmup[0..3],
            &post_live[0..3]
        );
        // El bloque multifractal [3..6] debe estar en rango y vivir (alguna
        // escala movida respecto al warmup por tick, no por kline).
        let hurst_changed = (3..6).any(|i| {
            (post_warmup[i] - post_live[i]).abs() > 1e-6
        });
        assert!(
            hurst_changed,
            "Hurst multifractal congelado post-arranque: {:?} vs {:?}",
            &post_warmup[3..6],
            &post_live[3..6]
        );

        // Cambio de régimen EN VIVO (solo ticks): ciclo lento de 64 ticks —
        // el FFT debe volver a moverse, probando republicación continua.
        for i in 0..128 {
            let phase = (i as f64) * (std::f64::consts::TAU / 48.0);
            let price = 100.0 + phase.sin() * 2.0;
            engine.process_tick(price, 1.0, ts);
            ts += 100;
        }
        let post_regime = engine.get_spectral_ml_features();
        let fft_changed_again = (0..3).any(|i| {
            (post_live[i] - post_regime[i]).abs() > 1e-6
        });
        assert!(
            fft_changed_again,
            "FFT no se republica ante cambio de régimen: {:?} vs {:?}",
            &post_live[0..3],
            &post_regime[0..3]
        );
    }

    /// C-02 (INFORME DECIMOCUARTO): el mapa vivo/muerto del contrato 34D es
    /// verificable, no documentación muerta. [9] (dark_alpha) sirve 0.0
    /// constante en producción (dex_severity sin productor) — y el trainer
    /// la fuerza a 0 vía FEATURES_DEAD_IN_SERVE. Las dims de obi_accel
    /// [4],[5],[10] VIVEN en serve: se mueven con el obi real del libro que
    /// `update_macro_features` recibe por evento.
    #[test]
    fn c02_mapa_vivo_muerto_del_vector_34d() {
        // Contrato del slice: índices dentro de las 34, únicos y ordenados.
        let mut sorted = FEATURES_DEAD_IN_SERVE.to_vec();
        sorted.sort_unstable();
        assert!(
            sorted.windows(2).all(|w| w[0] < w[1]),
            "FEATURES_DEAD_IN_SERVE con índices repetidos: {:?}",
            sorted
        );
        assert!(
            sorted.iter().all(|&i| i < 34),
            "índice fuera del contrato 34D: {:?}",
            sorted
        );

        // Serve: sin productor dex, dim [9] sirve exactamente 0.0.
        let mut e = StatefulEngine::new();
        let mut ts: u64 = 1_789_000_000_000;
        for i in 0..600u64 {
            let p = 100.0 + ((i % 37) as f64).sin();
            e.process_tick(p, 1.0, ts);
            let obi = 0.3 * (((i % 11) as f64) - 5.0) / 5.0;
            e.update_macro_features(obi, 0.0, 0.0, ts);
            ts += 100;
        }
        // Determinismo: dos obis distintos y no nulos al final.
        e.update_macro_features(0.25, 0.0, 0.0, ts);
        e.update_macro_features(0.35, 0.0, 0.0, ts + 100);
        let f = e.get_universal_features();
        assert_eq!(f.len(), 34);
        // CERT-M2-H02: dims [4][5][9][10] AHORA zerificadas en AMBOS lados
        // (serve incluido). El test anterior verificaba que obi_accel
        // vivía en serve — eso era EXACTAMENTE el defecto: el trainer
        // zerificaba pero el serve no, rompiendo la paridad.
        assert_eq!(f[9], 0.0, "dark_alpha debe servir 0.0 sin productor dex");
        assert_eq!(f[4], 0.0, "obi_accel [4] zerificado en serve (paridad B3.35)");
        assert_eq!(f[5], 0.0, "obi_accel [5] zerificado en serve (paridad B3.35)");
        assert_eq!(f[10], 0.0, "obi_accel [10] zerificado en serve (paridad B3.35)");
    }

    #[test]
    fn test_stateful_engine_reset_and_feature_extraction() {
        let mut engine = StatefulEngine::new();
        engine.reset();

        let micro_feats = engine.get_features();
        assert_eq!(micro_feats.len(), 12);
        for f in &micro_feats {
            assert!(f.is_finite(), "Micro feature debe ser finita");
        }

        let swing_feats = engine.get_universal_features();
        assert_eq!(swing_feats.len(), 34);
        for f in &swing_feats {
            assert!(f.is_finite(), "Swing feature debe ser finita");
        }
    }

    #[test]
    fn test_stateful_engine_market_regime_classification() {
        let engine = StatefulEngine::new();
        let regime = engine.get_market_regime();
        // U-6: el continuo no tiene variantes Scalping/Swing.
        assert!(matches!(regime, MarketRegime::Continuous | MarketRegime::Neutral));

        let atr_pct = engine.get_atr_pct();
        assert!(atr_pct.is_finite());
    }

    /// D-615b: el Hurst depende del RELOJ, no de la cadencia del feed. La
    /// misma ventana de 3 min 5 s alimentada a 100 ms (producción) y a 15 s
    /// (backtest forense) debe entregar al estimador las mismas muestras.
    #[test]
    fn d615b_hurst_se_muestrea_por_reloj_no_por_evento() {
        let t0: u64 = 1_700_000_000_000;
        let precio = |i: u64| 50_000.0 + ((i % 7) as f64 - 3.0);

        let mut rapido = StatefulEngine::new();
        for i in 0..1_850u64 {
            rapido.process_tick(precio(i), 1.0, t0 + i * 100);
        }
        let mut lento = StatefulEngine::new();
        for i in 0..13u64 {
            lento.process_tick(precio(i), 1.0, t0 + i * 15_000);
        }

        assert_eq!(
            rapido.hurst.samples(),
            lento.hurst.samples(),
            "la cadencia del feed no debe cambiar la muestra del estimador"
        );
        assert!(
            rapido.hurst.samples() <= 3,
            "3 cierres de minuto no pueden producir {} retornos",
            rapido.hurst.samples()
        );
    }

    /// D-688: la desviación estimada del OBI converge a la del ruido real y
    /// no se publica antes del calentamiento.
    #[test]
    fn d688_ruido_del_obi_se_estima_y_espera_al_calentamiento() {
        let mut n = ObiNoise::new();
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        for i in 0..20_000u32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
            // Uniforme en [−0,3; 0,3]: σ = 0,6/√12 ≈ 0,1732.
            n.update(-0.3 + 0.6 * u);
            if i + 1 < OBI_NOISE_EVENTS {
                assert!(n.sd().is_none());
            }
        }
        let sd = n.sd().expect("calentado");
        assert!((sd - 0.1732).abs() < 0.03, "σ estimada {sd}");
    }

    #[test]
    fn test_stateful_engine_export_f32_stability() {
        let mut engine = StatefulEngine::new();
        let mut buffer = [0.0f32; 144];
        engine.export_f32(&mut buffer);

        for val in &buffer[0..5] {
            assert!(val.is_finite());
        }
    }
}

/// Tests de los defectos D-753 (volumen del VPIN) y D-754 (enfriamiento en
/// tiempo). Cada uno afirma la propiedad que el código ANTERIOR violaba, de
/// modo que un retroceso a la versión por cuenta de ticks o al «volumen»
/// fabricado los rompe.
#[cfg(test)]
mod tests_d753_d754 {
    use super::*;

    /// Alimenta `n` ticks separados `paso_ms`, con el mismo recorrido de
    /// precio y sin volumen negociado, partiendo de `t0`. Devuelve el reloj
    /// del último evento.
    fn alimentar(engine: &mut StatefulEngine, n: u64, paso_ms: u64, t0: u64) -> u64 {
        let mut ts = t0;
        for i in 0..n {
            // Precio con recorrido idéntico en ambas cadencias: lo único que
            // cambia entre los dos motores es CUÁNTOS eventos median.
            let p = 100.0 + ((i % 9) as f64 - 4.0) * 0.01;
            engine.process_tick(p, 0.0, ts);
            ts += paso_ms;
        }
        ts.saturating_sub(paso_ms)
    }

    /// D-754 — EL ENFRIAMIENTO NO PUEDE DEPENDER DE LA DENSIDAD DEL TAPE.
    ///
    /// QUÉ ESTABA MAL: `can_open_position(600)` contaba 600 EVENTOS. Dos
    /// motores que han visto pasar el MISMO tiempo de mercado —30 s— daban
    /// respuestas opuestas por el solo hecho de que uno recibía ticks cada
    /// 10 ms (tape denso: 3 000 eventos, «enfriado») y el otro cada segundo
    /// (tape ralo: 30 eventos, «en enfriamiento»). El backtest forense, que
    /// lee aggTrades a una cadencia distinta de la del WebSocket en vivo,
    /// medía por tanto una POLÍTICA DISTINTA de la que corre en producción.
    ///
    /// QUÉ GARANTIZA ESTE TEST: con el mismo tiempo transcurrido y la misma
    /// base de enfriamiento, la decisión es la misma cualquiera que sea la
    /// cadencia del feed. Con el código viejo `denso` devolvía `true` y
    /// `ralo` `false` para la misma ventana de 30 s.
    #[test]
    fn d754_enfriamiento_es_tiempo_y_no_cuenta_de_eventos() {
        let t0: u64 = 1_800_000_000_000;
        // Base de enfriamiento: un horizonte τ de 60 s. Los 30 s de mercado
        // transcurridos NO lo cubren, así que ambos motores deben negar.
        let base_ms = 60_000.0;

        // 30 s exactos de mercado en ambos, con cadencias que difieren ×100.
        let mut denso = StatefulEngine::new();
        denso.last_scalp_exit_ms = t0;
        alimentar(&mut denso, 3_001, 10, t0); // 30 s a 10 ms → 3 001 eventos

        let mut ralo = StatefulEngine::new();
        ralo.last_scalp_exit_ms = t0;
        alimentar(&mut ralo, 31, 1_000, t0); // 30 s a 1 s → 31 eventos

        assert_eq!(
            denso.last_event_ms.saturating_sub(denso.last_scalp_exit_ms),
            30_000,
            "el motor denso debe haber visto 30 s"
        );
        assert_eq!(
            denso.last_event_ms.saturating_sub(denso.last_scalp_exit_ms),
            ralo.last_event_ms.saturating_sub(ralo.last_scalp_exit_ms),
            "los dos motores deben haber visto el MISMO tiempo de mercado"
        );
        assert_ne!(
            denso.tick_count, ralo.tick_count,
            "el test carece de sentido si ambos vieron el mismo nº de eventos"
        );

        // LA PROPIEDAD: misma decisión, pese a 100× de diferencia en eventos.
        assert_eq!(
            denso.can_open_position_ms(base_ms),
            ralo.can_open_position_ms(base_ms),
            "la densidad del tape cambió la decisión de enfriamiento"
        );
        assert!(
            !denso.can_open_position_ms(base_ms),
            "30 s transcurridos no pueden cubrir un enfriamiento base de 60 s"
        );

        // Y con una base que SÍ cabe en lo transcurrido, ambos abren.
        assert!(denso.can_open_position_ms(10_000.0));
        assert!(ralo.can_open_position_ms(10_000.0));
    }

    /// D-754 — LA ESCALERA DE RACHA ES BINARIA Y SE AMORTIZA CON EL TIEMPO.
    ///
    /// El olvido de rachas tenía dos ventanas inventadas (7 200 y 18 000
    /// TICKS) que además sólo descontaban UN nivel. Ahora el olvido es la
    /// propia escalera leída al revés: haber esperado `base·(2^m − 1)`
    /// amortiza `m` niveles de racha, sin ninguna ventana aparte.
    #[test]
    fn d754_la_racha_exige_retroceso_binario_y_se_amortiza_sola() {
        let t0: u64 = 1_800_000_000_000;
        let base_ms = 1_000.0;

        // Racha de 3 recién cerrada: exige base·2³ = 8 s. Pero la espera que
        // transcurre AMORTIZA niveles mientras corre, así que el punto de
        // corte real es el primer instante en que lo esperado alcanza a lo
        // exigido por la racha que queda. A 2,5 s se han amortizado
        // log₂(1+2,5) = 1 nivel: quedan 2 y se exigen 4 s ⇒ todavía no.
        let mut e = StatefulEngine::new();
        e.scalp_loss_streak = 3;
        e.last_scalp_exit_ms = t0;
        e.last_event_ms = t0 + 2_500;
        assert!(
            !e.can_open_position_ms(base_ms),
            "2,5 s no cubren los 4 s que exige la racha aún no amortizada"
        );
        // A 3 s se amortizan log₂(1+3) = 2 niveles: queda 1 y se exigen 2 s.
        e.last_event_ms = t0 + 3_000;
        assert!(
            e.can_open_position_ms(base_ms),
            "3 s sí cubren la exigencia que queda tras amortizar 2 niveles"
        );

        // Monotonía en la racha: más pérdidas seguidas ⇒ más espera.
        let espera_minima = |racha: u32| -> u64 {
            let mut ms = 0u64;
            loop {
                let mut m = StatefulEngine::new();
                m.scalp_loss_streak = racha;
                m.last_scalp_exit_ms = t0;
                m.last_event_ms = t0 + ms;
                if m.can_open_position_ms(base_ms) {
                    return ms;
                }
                ms += 250;
                assert!(ms < 200_000, "racha {racha} sin convergencia");
            }
        };
        let (e1, e2, e3) = (espera_minima(1), espera_minima(2), espera_minima(3));
        assert!(
            e1 < e2 && e2 < e3,
            "la escalera de racha no es monótona: {e1} / {e2} / {e3}"
        );

        // Amortización: la racha DIRECCIONAL efectiva baja al pasar el tiempo,
        // sin ninguna ventana de olvido escrita a mano.
        let mut d = StatefulEngine::new();
        d.scalp_long_loss_streak = 3;
        d.last_scalp_exit_ms = t0;
        d.last_event_ms = t0 + 1; // nada transcurrido
        assert_eq!(d.get_active_directional_streak_ms(true, base_ms), 3);
        d.last_event_ms = t0 + 1_000; // 1·base ⇒ log₂(2) = 1 nivel amortizado
        assert_eq!(d.get_active_directional_streak_ms(true, base_ms), 2);
        d.last_event_ms = t0 + 7_000; // 7·base ⇒ log₂(8) = 3 niveles
        assert_eq!(d.get_active_directional_streak_ms(true, base_ms), 0);
    }

    /// D-754 — SIN CIERRE PREVIO NO HAY ENFRIAMIENTO QUE IMPONER, y una base
    /// no medida (τ ausente, NaN o cero) no puede inventar un veto.
    #[test]
    fn d754_sin_cierre_ni_base_medida_no_hay_veto() {
        let e = StatefulEngine::new();
        assert!(
            e.can_open_position_ms(60_000.0),
            "sin cierre previo no puede haber enfriamiento"
        );

        let mut c = StatefulEngine::new();
        c.last_scalp_exit_ms = 1_800_000_000_000;
        c.last_event_ms = c.last_scalp_exit_ms; // cero transcurrido
        assert!(
            c.can_open_position_ms(f64::NAN),
            "una base no medida no puede vetar"
        );
        assert!(c.can_open_position_ms(0.0), "base nula no puede vetar");
    }

    /// D-753 — EL RELOJ DE VOLUMEN DEL VPIN SÓLO AVANZA CON VOLUMEN NEGOCIADO.
    ///
    /// QUÉ ESTABA MAL: el camino per-tick fabricaba el «volumen» como el
    /// 0,5 % de la PROFUNDIDAD del libro acotado a [0,01; 10] y se lo pasaba
    /// al VPIN en CADA evento, snapshots de libro incluidos. El VPIN es un
    /// reloj de VOLUMEN NEGOCIADO: alimentarlo con la profundidad hacía que
    /// midiera la cadencia de actualización del libro, no la toxicidad del
    /// flujo. Y el VPIN gobierna el corte tóxico, el asiento causal del
    /// consejo y `vpin_risk` en el dimensionado.
    ///
    /// QUÉ GARANTIZA ESTE TEST: un evento sin cantidad negociada deja el
    /// estado del VPIN —incluido su calibrador de bucket— EXACTAMENTE igual;
    /// sólo un trade real lo mueve. Con el código viejo, `update` se llamaba
    /// igualmente y `ewma_tick_notional` quedaba contaminado.
    #[test]
    fn d753_el_vpin_no_avanza_sin_cantidad_negociada() {
        let t0: u64 = 1_800_000_000_000;
        let mut e = StatefulEngine::new();

        // 500 eventos de libro (cantidad negociada = 0).
        for i in 0..500u64 {
            e.process_tick(100.0 + ((i % 7) as f64 - 3.0) * 0.01, 0.0, t0 + i * 100);
        }
        assert_eq!(
            e.cvpin.buy_volume + e.cvpin.sell_volume,
            0.0,
            "un snapshot de libro no es volumen negociado"
        );
        assert_eq!(
            e.cvpin.ewma_tick_notional, 0.0,
            "el calibrador del bucket no puede contaminarse con eventos sin volumen"
        );

        // Un trade REAL sí mueve el reloj, y lo hace por su nocional.
        let ts = t0 + 500 * 100;
        e.process_tick(100.0, 3.0, ts);
        let total = e.cvpin.buy_volume + e.cvpin.sell_volume;
        assert!(
            (total - 300.0).abs() < 1e-9,
            "el VPIN debe recibir cantidad×precio = 300, recibió {total}"
        );
        assert!(
            (e.cvpin.ewma_tick_notional - 300.0).abs() < 1e-9,
            "el bucket debe calibrarse con el nocional del trade real"
        );
    }

    /// D-758 — LA ESCALA QUE FALTABA: σ DEL RETORNO POR TICK.
    ///
    /// Sin ella, `micro_trend` —un diferencial de EMAs por TICK— sólo podía
    /// compararse contra fracciones escritas a mano (0,00005 … 0,00040;
    /// 0,015), porque la única σ que el motor medía era la de la vela de
    /// 1 minuto, varios órdenes de magnitud mayor. Este test fija las dos
    /// propiedades del estimador: NO publica durante el calentamiento —
    /// tipificar con una σ a medio estimar es peor que no tipificar— y
    /// converge a la σ real del ruido que se le da.
    #[test]
    fn d758_la_sigma_por_tick_se_mide_y_espera_al_calentamiento() {
        let mut e = StatefulEngine::new();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut price = 100.0_f64;
        let mut ts: u64 = 1_800_000_000_000;

        e.process_tick(price, 0.0, ts);
        ts += 100;
        assert!(
            e.sigma_retorno_tick().is_none(),
            "no puede publicarse una σ con un solo tick"
        );

        for i in 0..8_000u32 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
            // Retorno uniforme en [−0,003; 0,003]: σ = 0,006/√12 ≈ 0,0017321.
            let r = -0.003 + 0.006 * u;
            price *= 1.0 + r;
            e.process_tick(price, 0.0, ts);
            ts += 100;
            if i + 2 < OBI_NOISE_EVENTS {
                assert!(
                    e.sigma_retorno_tick().is_none(),
                    "publicó σ antes de calentar, en el tick {i}"
                );
            }
        }

        let sd = e.sigma_retorno_tick().expect("calentada");
        assert!(
            (sd - 0.001_732_1).abs() < 3.5e-4,
            "σ por tick estimada {sd}, esperada ≈ 0,0017321"
        );
    }

    /// D-758 — LAS CONSTANTES NOMBRADAS SON LAS QUE `process_tick` USA.
    ///
    /// La tipificación de `micro_trend` depende de que los periodos con los
    /// que se calcula la escala sean EXACTAMENTE los de las EMAs que producen
    /// la magnitud. Si alguien cambia los 20/200 de `process_tick` sin tocar
    /// las constantes, la escala queda desincronizada y el z miente en
    /// silencio; este test lo impide.
    #[test]
    fn d758_las_constantes_de_ema_por_tick_son_las_que_el_motor_aplica() {
        let t0: u64 = 1_800_000_000_000;
        let mut e = StatefulEngine::new();
        e.process_tick(100.0, 0.0, t0); // siembra: ambas EMAs = 100
        e.process_tick(110.0, 0.0, t0 + 100); // salto de 10

        let alpha_fast = 2.0 / (TICK_EMA_FAST_BARS + 1.0);
        let alpha_slow = 2.0 / (TICK_EMA_SLOW_BARS + 1.0);
        assert!(
            (e.ema_fast - (100.0 + 10.0 * alpha_fast)).abs() < 1e-9,
            "TICK_EMA_FAST_BARS no describe la EMA rápida real: {}",
            e.ema_fast
        );
        assert!(
            (e.ema_slow - (100.0 + 10.0 * alpha_slow)).abs() < 1e-9,
            "TICK_EMA_SLOW_BARS no describe la EMA lenta real: {}",
            e.ema_slow
        );
    }

    /// D-753/D-754 — `reset` (reconexión del feed) borra también los relojes
    /// de enfriamiento y la última cantidad negociada: tras una desconexión no
    /// hay continuidad temporal ni trade reciente que defender.
    #[test]
    fn d753_d754_reset_borra_relojes_y_cantidad() {
        let mut e = StatefulEngine::new();
        e.ultima_cantidad_trade = 7.0;
        e.last_scalp_exit_ms = 123;
        e.last_event_ms = 456;
        e.tau_ultimo_cierre_ms = 789;
        for i in 0..(OBI_NOISE_EVENTS + 10) {
            e.process_tick(100.0 + (i % 5) as f64 * 0.1, 0.0, 1_000 + i as u64 * 100);
        }
        assert!(e.sigma_retorno_tick().is_some(), "premisa: σ calentada");
        e.reset();
        assert_eq!(e.ultima_cantidad_trade, 0.0);
        assert_eq!(e.last_scalp_exit_ms, 0);
        assert_eq!(e.last_event_ms, 0);
        assert_eq!(e.tau_ultimo_cierre_ms, 0);
        assert!(
            e.sigma_retorno_tick().is_none(),
            "la σ por tick debe volver a calentarse tras una reconexión"
        );
    }
}
