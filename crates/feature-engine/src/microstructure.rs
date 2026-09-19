/// Microestructura del Mercado (Order Book)
/// Axioma II: O(1) puro y #[inline(always)]

#[derive(Clone, Copy)]
pub struct OrderFlowTracker {
    pub cumulative_buy_vol: f64,
    pub cumulative_sell_vol: f64,
    pub short_ema_obi: f64,
    pub long_ema_obi: f64,
    pub velocity_obi: f64,
}

impl Default for OrderFlowTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl OrderFlowTracker {
    pub fn new() -> Self {
        Self {
            cumulative_buy_vol: 0.0,
            cumulative_sell_vol: 0.0,
            short_ema_obi: 0.0,
            long_ema_obi: 0.0,
            velocity_obi: 0.0,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, volume: f64, is_buyer_maker: bool) -> f64 {
        // En Binance: buyer_maker = true significa que el trade fue ejecutado contra el BID (Taker Sell)
        // buyer_maker = false significa que el trade fue ejecutado contra el ASK (Taker Buy)
        let v = if volume.is_finite() && volume > 0.0 {
            volume
        } else {
            0.0
        };
        let (buy_v, sell_v) = if is_buyer_maker { (0.0, v) } else { (v, 0.0) };

        self.cumulative_buy_vol += buy_v;
        self.cumulative_sell_vol += sell_v;

        // Instant Imbalance for this tick con clamping [-1.0, 1.0]
        let tick_imbalance = if v < 1e-12 {
            0.0
        } else {
            ((buy_v - sell_v) / v).clamp(-1.0, 1.0)
        };

        // EWMA update
        let short_alpha = 0.1; // Fast
        let long_alpha = 0.01; // Slow

        let prev_short = self.short_ema_obi;
        self.short_ema_obi =
            (tick_imbalance - self.short_ema_obi) * short_alpha + self.short_ema_obi;
        self.long_ema_obi = (tick_imbalance - self.long_ema_obi) * long_alpha + self.long_ema_obi;

        self.velocity_obi = self.short_ema_obi - prev_short;

        tick_imbalance
    }

    #[inline(always)]
    pub fn get_volume_delta_ratio(&self) -> f64 {
        let b = if self.cumulative_buy_vol.is_finite() && self.cumulative_buy_vol > 0.0 {
            self.cumulative_buy_vol
        } else {
            0.0
        };
        let s = if self.cumulative_sell_vol.is_finite() && self.cumulative_sell_vol > 0.0 {
            self.cumulative_sell_vol
        } else {
            0.0
        };
        let total = b + s;
        if total < 1e-12 {
            0.0
        } else {
            ((b - s) / total).clamp(-1.0, 1.0)
        }
    }
}

/// Cont et al. (2014) Level-1 Order Flow Imbalance (OFI)
/// Mide la presión direccional neta calculando el cambio neto en volumen
/// condicionado al movimiento del precio en el nivel BBO (Best Bid/Offer).
#[derive(Clone, Copy)]
pub struct OFIModel {
    pub prev_bid_price: f64,
    pub prev_bid_qty: f64,
    pub prev_ask_price: f64,
    pub prev_ask_qty: f64,
    pub prev_depth: f64,
    pub ema_ofi: f64,
}

impl Default for OFIModel {
    fn default() -> Self {
        Self::new()
    }
}

impl OFIModel {
    pub fn new() -> Self {
        Self {
            prev_bid_price: 0.0,
            prev_bid_qty: 0.0,
            prev_ask_price: 0.0,
            prev_ask_qty: 0.0,
            prev_depth: 0.0,
            ema_ofi: 0.0,
        }
    }

    #[inline(always)]
    pub fn update(&mut self, bid_price: f64, ask_price: f64, bid_qty: f64, ask_qty: f64) -> f64 {
        let b_p = if bid_price.is_finite() && bid_price > 0.0 {
            bid_price
        } else {
            self.prev_bid_price
        };
        let a_p = if ask_price.is_finite() && ask_price > 0.0 {
            ask_price
        } else {
            self.prev_ask_price
        };
        let b_qty = if bid_qty.is_finite() && bid_qty >= 0.0 {
            bid_qty
        } else {
            0.0
        };
        let a_qty = if ask_qty.is_finite() && ask_qty >= 0.0 {
            ask_qty
        } else {
            0.0
        };

        if self.prev_bid_price == 0.0 {
            self.prev_bid_price = b_p;
            self.prev_bid_qty = b_qty;
            self.prev_ask_price = a_p;
            self.prev_ask_qty = a_qty;
            return 0.0;
        }

        // Flujo neto en BID (e_bid)
        let e_bid = if b_p > self.prev_bid_price {
            b_qty
        } else if b_p == self.prev_bid_price {
            b_qty - self.prev_bid_qty
        } else {
            -self.prev_bid_qty
        };

        // Flujo neto en ASK (e_ask)
        let e_ask = if a_p < self.prev_ask_price {
            a_qty
        } else if a_p == self.prev_ask_price {
            a_qty - self.prev_ask_qty
        } else {
            -self.prev_ask_qty
        };

        // OFI Tick Normalizado por la liquidez estable (max de profundidad previa/actual)
        // FIX #900: Usar max(prev_depth, current_depth) para evitar spikes artificiales
        // cuando la profundidad colapsa abruptamente en flash crashes
        let current_depth = b_qty + a_qty;
        let stable_depth = current_depth.max(self.prev_depth).max(1e-6);
        let ofi_norm = ((e_bid - e_ask) / stable_depth).clamp(-10.0, 10.0);

        // Actualizar estados pasados
        self.prev_bid_price = bid_price;
        self.prev_bid_qty = b_qty;
        self.prev_ask_price = ask_price;
        self.prev_ask_qty = a_qty;
        self.prev_depth = current_depth;

        // Suavizado EWMA para evitar ruido (alpha = 0.1)
        self.ema_ofi = (ofi_norm - self.ema_ofi) * 0.1 + self.ema_ofi;

        self.ema_ofi
    }
}

/// Calcula el Order Book Imbalance (OBI).
#[inline(always)]
pub fn order_book_imbalance(bid_vol: f64, ask_vol: f64) -> f64 {
    let b = bid_vol.max(0.0);
    let a = ask_vol.max(0.0);
    let total_vol = b + a;
    if total_vol < 1e-12 {
        0.0
    } else {
        ((b - a) / total_vol).clamp(-1.0, 1.0)
    }
}

/// Aceleración de Liquidez (Derivada del OBI)
#[inline(always)]
pub fn obi_acceleration(current_obi: f64, previous_obi: f64) -> f64 {
    current_obi - previous_obi
}

/// 🚀 RASTREADOR DE VOLUMEN INSTITUCIONAL Y RÁFAGAS (INSTITUTIONAL VOLUME BURST TRACKER)
/// Utiliza el algoritmo Welford online para normalizar el flujo de volumen y detectar
/// ráfagas de liquidez institucional (Z-Score > 2.5) en O(1) sin jitter.
#[derive(Debug, Clone, Copy)]
pub struct InstitutionalVolumeTracker {
    pub welford_volume: crate::welford::WelfordOnline,
    pub prev_volume: f64,
    pub volume_acceleration: f64,
    pub burst_threshold_zscore: f64,
}

impl Default for InstitutionalVolumeTracker {
    fn default() -> Self {
        Self::new(2.5)
    }
}

impl InstitutionalVolumeTracker {
    pub fn new(burst_threshold_zscore: f64) -> Self {
        Self {
            welford_volume: crate::welford::WelfordOnline::new(),
            prev_volume: 0.0,
            volume_acceleration: 0.0,
            burst_threshold_zscore: burst_threshold_zscore.max(1.0),
        }
    }

    /// Actualiza el tracker con una nueva muestra de volumen y devuelve (z_score, acceleration, is_burst)
    #[inline(always)]
    pub fn update(&mut self, volume: f64) -> (f64, f64, bool) {
        if !volume.is_finite() {
            return (0.0, 0.0, false);
        }
        let v = volume.max(0.0);
        let accel = v - self.prev_volume;
        self.volume_acceleration = accel;
        self.prev_volume = v;

        self.welford_volume.update(v);
        let z_score = self.welford_volume.z_score(v);
        let is_burst = self.welford_volume.count >= 20.0 && z_score >= self.burst_threshold_zscore;

        (z_score, accel, is_burst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_flow_tracker() {
        let mut tracker = OrderFlowTracker::new();
        tracker.update(10.0, false); // Buy
        tracker.update(5.0, true); // Sell
        let delta = tracker.get_volume_delta_ratio();
        assert!((delta - (10.0 - 5.0) / 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_order_flow_tracker_nan_immunity() {
        let mut tracker = OrderFlowTracker::new();
        let imb1 = tracker.update(f64::NAN, false);
        assert!(imb1.is_finite());
        let imb2 = tracker.update(f64::INFINITY, true);
        assert!(imb2.is_finite());
        let ratio = tracker.get_volume_delta_ratio();
        assert!(ratio.is_finite());
    }

    #[test]
    fn test_ofi_model() {
        let mut ofi = OFIModel::new();
        let _ = ofi.update(100.0, 101.0, 5.0, 5.0);
        let val = ofi.update(100.5, 101.0, 10.0, 5.0); // Bid price rose
        assert!(val > 0.0, "OFI should be positive when bid price increases");
    }

    #[test]
    fn test_ofi_model_nan_immunity() {
        let mut ofi = OFIModel::new();
        let val1 = ofi.update(f64::NAN, 101.0, 5.0, 5.0);
        assert!(val1.is_finite());

        let val2 = ofi.update(100.0, f64::INFINITY, f64::NAN, 5.0);
        assert!(val2.is_finite());
    }

    #[test]
    fn test_institutional_volume_tracker() {
        let mut tracker = InstitutionalVolumeTracker::new(2.0);
        for _ in 0..50 {
            tracker.update(10.0);
        }
        // Sudden institutional spike
        let (z, accel, is_burst) = tracker.update(100.0);
        assert!(z > 2.0, "Z-Score should detect massive volume spike");
        assert!(accel > 50.0, "Acceleration should be positive and large");
        assert!(
            is_burst,
            "Burst flag must be triggered for large institutional volume"
        );
    }

    #[test]
    fn test_art2_microstructure_clustering() {
        let mut art2 = AdaptiveResonanceClustering::new(0.85, 0.1);
        let pattern1 = [1.0, 0.8, 0.2, 0.1];
        let (cat1, match1) = art2.classify_and_adapt(&pattern1);
        assert_eq!(cat1, 0);
        assert_eq!(match1, 1.0);

        // Patrón similar debe resonar en la misma categoría 0
        let pattern1_similar = [0.95, 0.85, 0.18, 0.12];
        let (cat2, match2) = art2.classify_and_adapt(&pattern1_similar);
        assert_eq!(cat2, 0);
        assert!(match2 >= 0.85);

        // Patrón completamente ortogonal debe crear una nueva categoría 1
        let pattern2 = [-0.9, -0.8, 0.9, 0.95];
        let (cat3, match3) = art2.classify_and_adapt(&pattern2);
        assert_eq!(cat3, 1);
        assert_eq!(match3, 1.0);
    }

    #[test]
    fn test_normalized_order_book_l2() {
        let mut ob = NormalizedOrderBookL2::new();
        let bids = vec![(60000.0, 1.0), (59990.0, 2.0)]; // 60k + 119.98k = 179.98k USD
        let asks = vec![(60010.0, 0.5), (60020.0, 0.5)]; // 30.005k + 30.01k = 60.015k USD
        let imb = ob.update_from_raw_depth(&bids, &asks);
        assert!(
            imb > 0.40,
            "Heavy bid side notional should yield strong positive imbalance"
        );
        assert!(ob.total_bid_usd > ob.total_ask_usd);
    }

    #[test]
    fn test_spoofing_detector() {
        let mut sp = SpoofingDetector::new(2.0);
        let risk0 = sp.evaluate_wall_decay(100.0, 10.0, 1000);
        assert_eq!(risk0, 0.0);

        // Cancelación abrupta del muro de 100 en 100ms
        let risk1 = sp.evaluate_wall_decay(20.0, 10.0, 1100);
        assert!(
            risk1 >= 0.50,
            "Sudden 80% bid wall drop in 100ms must trigger spoofing alert"
        );
    }

    #[test]
    fn test_normalized_order_book_inverted_spread_rejection() {
        let mut ob = NormalizedOrderBookL2::new();
        // Spread invertido: best bid 60050 >= best ask 60000 (Libro cruzado/anomalía)
        let inverted_bids = vec![(60050.0, 1.0)];
        let asks = vec![(60000.0, 1.0)];
        let res = ob.validate_and_update(&inverted_bids, &asks);
        assert_eq!(res, Err("INVERTED_CROSSED_SPREAD"));

        // Spread válido: best bid 59990 < best ask 60000
        let valid_bids = vec![(59990.0, 1.0)];
        let valid_res = ob.validate_and_update(&valid_bids, &asks);
        assert!(valid_res.is_ok());
    }
}

/// Red de Resonancia Adaptativa ART-2 para Clustering Online de Microestructura (#66-#72)
/// Clasifica en tiempo real vectores de 4 dimensiones (OFI, OBI, Hawkes Intensity, Volume Spike)
/// en prototipos de categorías resonantes estables sin olvido catastrófico.
#[derive(Debug, Clone)]
pub struct AdaptiveResonanceClustering {
    pub vigilance: f64,            // Parámetro de vigilancia $\rho \in [0.70, 0.99]$
    pub prototypes: [[f64; 4]; 8], // Hasta 8 prototipos de categorías de microestructura
    pub active_categories: usize,
    pub learning_rate: f64,
}

impl Default for AdaptiveResonanceClustering {
    fn default() -> Self {
        Self::new(0.85, 0.10)
    }
}

impl AdaptiveResonanceClustering {
    pub fn new(vigilance: f64, learning_rate: f64) -> Self {
        Self {
            vigilance: vigilance.clamp(0.50, 0.98),
            prototypes: [[0.0; 4]; 8],
            active_categories: 0,
            learning_rate: learning_rate.clamp(0.01, 0.50),
        }
    }

    /// Normaliza un vector de entrada a norma euclidiana unitaria
    #[inline(always)]
    fn normalize(input: &[f64; 4]) -> [f64; 4] {
        let v0 = if input[0].is_finite() { input[0] } else { 0.0 };
        let v1 = if input[1].is_finite() { input[1] } else { 0.0 };
        let v2 = if input[2].is_finite() { input[2] } else { 0.0 };
        let v3 = if input[3].is_finite() { input[3] } else { 0.0 };
        let norm = (v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3).sqrt();
        if norm < 1e-12 {
            [0.0; 4]
        } else {
            [v0 / norm, v1 / norm, v2 / norm, v3 / norm]
        }
    }

    /// Clasifica el vector y actualiza el prototipo si hay resonancia (o crea uno nuevo)
    /// Retorna: `(category_index, resonance_match_score)`
    #[inline(always)]
    pub fn classify_and_adapt(&mut self, input: &[f64; 4]) -> (usize, f64) {
        let normalized = Self::normalize(input);

        let mut best_category = 0;
        let mut best_match = -1.0;

        for i in 0..self.active_categories {
            // Similitud de coseno (producto punto entre vectores unitarios)
            let proto = &self.prototypes[i];
            let match_score = normalized[0] * proto[0]
                + normalized[1] * proto[1]
                + normalized[2] * proto[2]
                + normalized[3] * proto[3];
            if match_score > best_match {
                best_match = match_score;
                best_category = i;
            }
        }

        // Test de Vigilancia: ¿Resuena con la categoría ganadora?
        if best_match >= self.vigilance && self.active_categories > 0 {
            // Actualización del prototipo (Regla de aprendizaje ART-2)
            for (p_val, &n_val) in self.prototypes[best_category]
                .iter_mut()
                .zip(normalized.iter())
            {
                *p_val = (1.0 - self.learning_rate) * (*p_val) + self.learning_rate * n_val;
            }
            // Re-normalizar prototipo
            self.prototypes[best_category] = Self::normalize(&self.prototypes[best_category]);
            (best_category, best_match)
        } else if self.active_categories < 8 {
            // Asignar nueva categoría
            let new_cat = self.active_categories;
            self.prototypes[new_cat] = normalized;
            self.active_categories += 1;
            (new_cat, 1.0)
        } else {
            // Capacidad máxima de categorías: asignar a la más cercana sin actualizar prototipo (outlier no resonante si best_match < vigilance)
            (best_category, best_match)
        }
    }

    /// Determina si un score de match cumple el criterio de vigilancia resonante
    #[inline(always)]
    pub fn is_resonant(&self, match_score: f64) -> bool {
        match_score >= self.vigilance
    }

    pub fn save_to_disk<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let mut content = format!(
            "{}\n{}\n{}\n",
            self.vigilance, self.learning_rate, self.active_categories
        );
        for i in 0..8 {
            content.push_str(&format!(
                "{},{},{},{}\n",
                self.prototypes[i][0],
                self.prototypes[i][1],
                self.prototypes[i][2],
                self.prototypes[i][3]
            ));
        }
        std::fs::write(path, content)
    }

    pub fn load_from_disk<P: AsRef<std::path::Path>>(path: P) -> Option<Self> {
        let content = std::fs::read_to_string(path).ok()?;
        let mut lines = content.lines();
        let vigilance = lines.next()?.parse::<f64>().ok()?;
        let learning_rate = lines.next()?.parse::<f64>().ok()?;
        let active_categories = lines.next()?.parse::<usize>().ok()?;
        let mut prototypes = [[0.0; 4]; 8];
        for row in prototypes.iter_mut() {
            if let Some(l) = lines.next() {
                let mut parts = l.split(',');
                row[0] = parts.next()?.parse::<f64>().ok()?;
                row[1] = parts.next()?.parse::<f64>().ok()?;
                row[2] = parts.next()?.parse::<f64>().ok()?;
                row[3] = parts.next()?.parse::<f64>().ok()?;
            }
        }
        Some(Self {
            vigilance,
            prototypes,
            active_categories,
            learning_rate,
        })
    }
}

/// Libro de Órdenes L2 Normalizado por Nocional USD (Punto #018)
/// Convierte profundidades de unidades base (BTC, DOGE, SOL) a valor nocional USDT,
/// garantizando invarianza dimensional para las 30 monedas del universo.
#[derive(Debug, Clone)]
pub struct NormalizedOrderBookL2 {
    pub levels: usize,
    pub bid_notionals: [f64; 20],
    pub ask_notionals: [f64; 20],
    pub total_bid_usd: f64,
    pub total_ask_usd: f64,
}

impl Default for NormalizedOrderBookL2 {
    fn default() -> Self {
        Self::new()
    }
}

impl NormalizedOrderBookL2 {
    pub fn new() -> Self {
        Self {
            levels: 20,
            bid_notionals: [0.0; 20],
            ask_notionals: [0.0; 20],
            total_bid_usd: 0.0,
            total_ask_usd: 0.0,
        }
    }

    #[inline(always)]
    pub fn update_from_raw_depth(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)]) -> f64 {
        let mut sum_bid = 0.0;
        let mut sum_ask = 0.0;

        for (i, &(p, q)) in bids.iter().take(20).enumerate() {
            if p.is_finite() && q.is_finite() && p > 0.0 && q > 0.0 {
                let notional = p * q;
                self.bid_notionals[i] = notional;
                sum_bid += notional;
            } else {
                self.bid_notionals[i] = 0.0;
            }
        }

        for (i, &(p, q)) in asks.iter().take(20).enumerate() {
            if p.is_finite() && q.is_finite() && p > 0.0 && q > 0.0 {
                let notional = p * q;
                self.ask_notionals[i] = notional;
                sum_ask += notional;
            } else {
                self.ask_notionals[i] = 0.0;
            }
        }

        self.total_bid_usd = sum_bid;
        self.total_ask_usd = sum_ask;

        let total = sum_bid + sum_ask;
        if total > 1e-6 {
            ((sum_bid - sum_ask) / total).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    /// Actualiza el libro L2 con validación defensiva de spread no invertido (Punto #020)
    #[inline(always)]
    pub fn validate_and_update(
        &mut self,
        bids: &[(f64, f64)],
        asks: &[(f64, f64)],
    ) -> Result<f64, &'static str> {
        if bids.is_empty() || asks.is_empty() {
            return Err("EMPTY_DEPTH");
        }

        let best_bid = bids[0].0;
        let best_ask = asks[0].0;

        if !best_bid.is_finite() || !best_ask.is_finite() || best_bid <= 0.0 || best_ask <= 0.0 {
            return Err("NON_FINITE_TOP_OF_BOOK");
        }

        // Rechazar libros cruzados / spread invertido (Best Bid >= Best Ask)
        if best_bid >= best_ask {
            return Err("INVERTED_CROSSED_SPREAD");
        }

        Ok(self.update_from_raw_depth(bids, asks))
    }
}

/// Detector de Spoofing de Muros de Liquidez L2 (Punto #025)
/// Mide la tasa de cancelación rápida de muros de órdenes (<250ms) mediante decaimiento exponencial.
///
/// P-5-DECISIÓN (2026-09-18): SIGUE DESCONECTADO deliberadamente. Su API
/// (`update_from_raw_depth(bids: &[(f64,f64)], asks: &[(f64,f64)])`) exige
/// el libro L2 COMPLETO, pero el parser del stream @depth5 extrae sólo el
/// BEST level (bp/bq/ap/aq) — alimentarlo con un único nivel es ruido
/// disfrazado de señal (un "muro" de un nivel es el spread). PRERREQUISITO
/// para cablearlo: extender `parse_binance_depth` a los 5 niveles del
/// stream con buffers por símbolo fuera del camino caliente, y sólo
/// entonces instanciar por símbolo como InstitutionalVolumeTracker.
#[derive(Debug, Clone)]
pub struct SpoofingDetector {
    pub prev_max_bid_wall: f64,
    pub prev_max_ask_wall: f64,
    pub last_wall_ts: u64,
    pub spoofing_decay_lambda: f64,
    pub spoofing_risk_score: f64,
}

impl Default for SpoofingDetector {
    fn default() -> Self {
        Self::new(2.0)
    }
}

impl SpoofingDetector {
    pub fn new(spoofing_decay_lambda: f64) -> Self {
        Self {
            prev_max_bid_wall: 0.0,
            prev_max_ask_wall: 0.0,
            last_wall_ts: 0,
            spoofing_decay_lambda: spoofing_decay_lambda.clamp(0.01, 10.0),
            spoofing_risk_score: 0.0,
        }
    }

    #[inline(always)]
    pub fn evaluate_wall_decay(
        &mut self,
        current_max_bid: f64,
        current_max_ask: f64,
        timestamp_ms: u64,
    ) -> f64 {
        if !current_max_bid.is_finite() || !current_max_ask.is_finite() {
            return self.spoofing_risk_score;
        }

        if self.last_wall_ts > 0 && timestamp_ms > self.last_wall_ts {
            let dt_sec = (timestamp_ms - self.last_wall_ts) as f64 / 1000.0;
            let decay = (-self.spoofing_decay_lambda * dt_sec).exp();
            self.spoofing_risk_score *= decay;

            // Detección de caída abrupta de muro (>50% de evaporación en <250ms)
            if dt_sec < 0.25 {
                let bid_drop = (self.prev_max_bid_wall - current_max_bid).max(0.0);
                let ask_drop = (self.prev_max_ask_wall - current_max_ask).max(0.0);
                if self.prev_max_bid_wall > 0.0 && bid_drop / self.prev_max_bid_wall > 0.50 {
                    self.spoofing_risk_score = (self.spoofing_risk_score + 0.50).min(1.0);
                }
                if self.prev_max_ask_wall > 0.0 && ask_drop / self.prev_max_ask_wall > 0.50 {
                    self.spoofing_risk_score = (self.spoofing_risk_score + 0.50).min(1.0);
                }
            }
        }

        self.prev_max_bid_wall = current_max_bid;
        self.prev_max_ask_wall = current_max_ask;
        self.last_wall_ts = timestamp_ms;

        self.spoofing_risk_score
    }
}
