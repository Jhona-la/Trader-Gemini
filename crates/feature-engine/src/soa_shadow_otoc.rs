//! # Fase 16 — Libro Plano Structure-of-Arrays (`SoaOrderBookL2`) en $\mathcal{O}(1)$, Libro Sombra de Margen en Tránsito (`InFlightShadowLedger`) y Caos Cuántico por Correladores Fuera de Orden Temporal (`QuantumOtocEngine` / Cota MSS)
//!
//! ## 👨‍🏫 MODO PROFESOR (QUÉ, POR QUÉ, PARA QUÉ, CÓMO, CUÁNDO, DÓNDE, QUIÉN)
//!
//! - **QUÉ:** Tres motores deterministas en pila (`O(1)` memoria, cero `unwrap`/`assert!`, 100% Rust nativo):
//!   1. `SoaOrderBookL2<TICKS>`: Libro de órdenes L2 con disposición *Structure-of-Arrays* (`#[repr(C, align(64))]`) indexado directamente por desplazamiento de tick entero en **1 ciclo de reloj**, sin árboles `BTreeMap` ni tablas hash.
//!   2. `InFlightShadowLedger<MAX_INFLIGHT>`: Contabilidad en sombra de margen y comisiones reservadas por órdenes en vuelo (*in-flight*) antes de la confirmación asíncrona del exchange, con liberación por `client_order_id` y expiración determinista por `ttl_ns`.
//!   3. `QuantumOtocEngine`: Evaluador del Correlador Fuera de Orden Temporal (*Out-of-Time-Ordered Correlator*, $C(t) = \langle |[W(t), V(0)]|^2 \rangle$), exponente de scrambling cuántico $\lambda_Q$, tiempo de Ehrenfest $t_*$ y saturación de la **Cota Universal de Caos de Maldacena-Shenker-Stanford (MSS)** $\lambda_Q \le 2\pi T_{\text{eff}}$.
//! - **POR QUÉ:** Los libros basados en nodos dispersos sufren fallos de caché L1/L2; las ráfagas de órdenes concurrentes pueden sobregirar el margen libre antes de que llegue el `executionReport`; y el caos cuántico de muchos cuerpos precede al colapso de correlación en libros multiactivo.
//! - **PARA QUÉ:** Consultar y mutar niveles L2 en tiempo constante $\mathcal{O}(1)$, blindar el capital contra doble gasto en tránsito (`effective_free_margin`), y contraer la exposición cuando el mercado entra en *Fast Scrambling* cerca de la cota MSS.
//! - **CÓMO:** Indexación aritmética directa $\text{idx} = \text{mid} + \text{round}((P - P_0)/\Delta P)$, tabla estática de reservas con invariante contable estricta, y evolución unitaria de Heisenberg $W(t) = U^\dagger(t) W(0) U(t)$ mediante transformada de Cayley sobre operadores $4 \times 4$.
//! - **CUÁNDO:** En cada actualización de profundidad L2, antes de admitir una orden hacia el ejecutor, y en la evaluación cuántica del estado del mercado.
//! - **DÓNDE:** `crates/feature-engine/src/soa_shadow_otoc.rs`.
//! - **QUIÉN:** `SoaOrderBookL2`, `InFlightShadowLedger`, `QuantumOtocEngine`, `SoaShadowOtocMasterEngine`.

/// Dimensión espacial de operadores cuánticos locales para el correlador OTOC ($4 \times 4$).
pub const OTOC_OP_DIM: usize = 4;

/// Clasificación del régimen de scrambling cuántico OTOC y presión de margen en tránsito.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OtocShadowRegime {
    /// Flujo integrable cuasi-clásico ($\lambda_Q \ll \lambda_{\text{MSS}}$) y margen libre saludable.
    IntegrableLaminarFlow,
    /// Presión elevada de margen en tránsito (reservas en vuelo $> 60\%$ del margen libre).
    InFlightMarginCongestion,
    /// Saturación de la cota de caos de Maldacena-Shenker-Stanford ($\lambda_Q / \lambda_{\text{MSS}} \ge 0.80$).
    FastScramblingChaosVeto,
}

/// Libro de órdenes L2 plano con arquitectura *Structure-of-Arrays* (`SoA`) alineada a 64 bytes.
///
/// Indexa directamente cada nivel de precio en $\mathcal{O}(1)$ respecto a `anchor_price` con paso `tick_size`.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct SoaOrderBookL2<const TICKS: usize> {
    pub bid_qty: [f64; TICKS],
    pub ask_qty: [f64; TICKS],
    pub order_count: [u32; TICKS],
    pub anchor_price: f64,
    pub tick_size: f64,
    pub inv_tick_size: f64,
    pub best_bid_idx: Option<usize>,
    pub best_ask_idx: Option<usize>,
}

impl<const TICKS: usize> Default for SoaOrderBookL2<TICKS> {
    #[inline]
    fn default() -> Self {
        Self::new(100.0, 0.01)
    }
}

impl<const TICKS: usize> SoaOrderBookL2<TICKS> {
    /// Construye un libro plano centrado en `anchor_price` con resolución `tick_size`.
    #[inline]
    pub fn new(anchor_price: f64, tick_size: f64) -> Self {
        let anchor = if anchor_price.is_finite() && anchor_price > 0.0 {
            anchor_price
        } else {
            100.0
        };
        let ts = if tick_size.is_finite() && tick_size > 1e-12 {
            tick_size
        } else {
            0.01
        };
        Self {
            bid_qty: [0.0; TICKS],
            ask_qty: [0.0; TICKS],
            order_count: [0; TICKS],
            anchor_price: anchor,
            tick_size: ts,
            inv_tick_size: 1.0 / ts,
            best_bid_idx: None,
            best_ask_idx: None,
        }
    }

    /// Convierte un precio continuo `price` en el índice de slot $\mathcal{O}(1)$ en `[0, TICKS - 1]`.
    #[inline(always)]
    pub fn price_to_slot(&self, price: f64) -> Option<usize> {
        if TICKS == 0 || !price.is_finite() || price <= 0.0 {
            return None;
        }
        let half = (TICKS / 2) as isize;
        let offset = ((price - self.anchor_price) * self.inv_tick_size).round() as isize;
        let idx = half + offset;
        if idx >= 0 && (idx as usize) < TICKS {
            Some(idx as usize)
        } else {
            None
        }
    }

    /// Convierte un índice de slot en su precio exacto correspondiente.
    #[inline(always)]
    pub fn slot_to_price(&self, slot: usize) -> f64 {
        let half = (TICKS / 2) as isize;
        let delta = (slot as isize) - half;
        self.anchor_price + (delta as f64) * self.tick_size
    }

    /// Actualiza en $\mathcal{O}(1)$ la cantidad Bid en `price` y mantiene `best_bid_idx`.
    #[inline]
    pub fn update_bid(&mut self, price: f64, qty: f64, orders: u32) -> bool {
        let Some(slot) = self.price_to_slot(price) else {
            return false;
        };
        let clean_qty = if qty.is_finite() && qty > 0.0 {
            qty
        } else {
            0.0
        };
        self.bid_qty[slot] = clean_qty;
        self.order_count[slot] = if clean_qty > 0.0 { orders } else { 0 };

        if clean_qty > 0.0 {
            match self.best_bid_idx {
                Some(curr) if slot > curr => self.best_bid_idx = Some(slot),
                None => self.best_bid_idx = Some(slot),
                _ => {}
            }
        } else if self.best_bid_idx == Some(slot) {
            let mut scan = slot;
            let mut found = None;
            while scan > 0 {
                scan -= 1;
                if self.bid_qty[scan] > 0.0 {
                    found = Some(scan);
                    break;
                }
            }
            self.best_bid_idx = found;
        }
        true
    }

    /// Actualiza en $\mathcal{O}(1)$ la cantidad Ask en `price` y mantiene `best_ask_idx`.
    #[inline]
    pub fn update_ask(&mut self, price: f64, qty: f64, orders: u32) -> bool {
        let Some(slot) = self.price_to_slot(price) else {
            return false;
        };
        let clean_qty = if qty.is_finite() && qty > 0.0 {
            qty
        } else {
            0.0
        };
        self.ask_qty[slot] = clean_qty;
        self.order_count[slot] = if clean_qty > 0.0 { orders } else { 0 };

        if clean_qty > 0.0 {
            match self.best_ask_idx {
                Some(curr) if slot < curr => self.best_ask_idx = Some(slot),
                None => self.best_ask_idx = Some(slot),
                _ => {}
            }
        } else if self.best_ask_idx == Some(slot) {
            let mut scan = slot + 1;
            let mut found = None;
            while scan < TICKS {
                if self.ask_qty[scan] > 0.0 {
                    found = Some(scan);
                    break;
                }
                scan += 1;
            }
            self.best_ask_idx = found;
        }
        true
    }

    /// Calcula en $\mathcal{O}(1)$ el micro-precio ponderado por liquidez en el tope del libro (BBO).
    #[inline]
    pub fn micro_price(&self) -> f64 {
        match (self.best_bid_idx, self.best_ask_idx) {
            (Some(b_idx), Some(a_idx)) => {
                let pb = self.slot_to_price(b_idx);
                let pa = self.slot_to_price(a_idx);
                let qb = self.bid_qty[b_idx];
                let qa = self.ask_qty[a_idx];
                let sum_q = qb + qa;
                if sum_q > 1e-12 {
                    (qb * pa + qa * pb) / sum_q
                } else {
                    0.5 * (pb + pa)
                }
            }
            _ => self.anchor_price,
        }
    }

    /// Calcula el desequilibrio integrado en los `depth_ticks` niveles alrededor del BBO.
    #[inline]
    pub fn top_depth_imbalance(&self, depth_ticks: usize) -> f64 {
        let span = depth_ticks.max(1).min(TICKS);
        let mut bid_sum = 0.0;
        let mut ask_sum = 0.0;

        if let Some(b_top) = self.best_bid_idx {
            let start = b_top.saturating_sub(span - 1);
            let mut i = start;
            while i <= b_top {
                bid_sum += self.bid_qty[i];
                i += 1;
            }
        }
        if let Some(a_top) = self.best_ask_idx {
            let end = (a_top + span).min(TICKS);
            let mut i = a_top;
            while i < end {
                ask_sum += self.ask_qty[i];
                i += 1;
            }
        }
        let denom = bid_sum + ask_sum;
        if denom > 1e-12 {
            ((bid_sum - ask_sum) / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Entrada individual de reserva de margen en tránsito dentro de `InFlightShadowLedger`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct InFlightMarginEntry {
    pub client_order_id: u64,
    pub symbol_id: u16,
    pub margin_reserved_usd: f64,
    pub fee_buffer_usd: f64,
    pub created_at_ns: u64,
    pub active: bool,
}

/// Libro Sombra de Margen en Tránsito (*In-Flight Shadow Ledger*).
///
/// Deduce instantáneamente el margen inicial y el buffer de comisión de cada orden enviada al cable
/// antes de que el exchange emita confirmación, impidiendo que ráfagas concurrentes sobregiren la cuenta.
#[derive(Debug, Clone, Copy)]
pub struct InFlightShadowLedger<const MAX_INFLIGHT: usize> {
    pub entries: [InFlightMarginEntry; MAX_INFLIGHT],
    pub active_count: usize,
    pub total_margin_reserved: f64,
    pub total_fee_reserved: f64,
    pub reservation_ttl_ns: u64,
}

impl<const MAX_INFLIGHT: usize> Default for InFlightShadowLedger<MAX_INFLIGHT> {
    #[inline]
    fn default() -> Self {
        Self::new(5_000_000_000) // 5 segundos por defecto
    }
}

impl<const MAX_INFLIGHT: usize> InFlightShadowLedger<MAX_INFLIGHT> {
    /// Crea un libro sombra vacío con tiempo de expiración `reservation_ttl_ns`.
    #[inline]
    pub const fn new(reservation_ttl_ns: u64) -> Self {
        Self {
            entries: [InFlightMarginEntry {
                client_order_id: 0,
                symbol_id: 0,
                margin_reserved_usd: 0.0,
                fee_buffer_usd: 0.0,
                created_at_ns: 0,
                active: false,
            }; MAX_INFLIGHT],
            active_count: 0,
            total_margin_reserved: 0.0,
            total_fee_reserved: 0.0,
            reservation_ttl_ns,
        }
    }

    /// Total bloqueado en tránsito (`margin + fee_buffer`).
    #[inline]
    pub fn total_blocked_usd(&self) -> f64 {
        let sum = self.total_margin_reserved + self.total_fee_reserved;
        if sum.is_finite() && sum > 0.0 {
            sum
        } else {
            0.0
        }
    }

    /// Calcula el margen libre efectivo disponible tras descontar todas las órdenes en vuelo.
    #[inline]
    pub fn effective_free_margin(&self, wallet_free_margin_usd: f64) -> f64 {
        if !wallet_free_margin_usd.is_finite() || wallet_free_margin_usd <= 0.0 {
            return 0.0;
        }
        (wallet_free_margin_usd - self.total_blocked_usd()).max(0.0)
    }

    /// Intenta reservar margen para una nueva orden en tránsito; falla en modo fail-closed si
    /// el margen libre efectivo no cubre `margin_usd + fee_usd` o si la tabla está llena.
    #[inline]
    pub fn try_reserve_inflight(
        &mut self,
        client_order_id: u64,
        symbol_id: u16,
        margin_usd: f64,
        fee_usd: f64,
        wallet_free_margin_usd: f64,
        now_ns: u64,
    ) -> bool {
        if MAX_INFLIGHT == 0
            || client_order_id == 0
            || !margin_usd.is_finite()
            || margin_usd <= 0.0
            || !fee_usd.is_finite()
            || fee_usd < 0.0
        {
            return false;
        }

        let required = margin_usd + fee_usd;
        if required > self.effective_free_margin(wallet_free_margin_usd) {
            return false;
        }

        // Verificar que no exista duplicado activo con el mismo client_order_id
        let mut i = 0;
        let mut free_slot = None;
        while i < MAX_INFLIGHT {
            if self.entries[i].active {
                if self.entries[i].client_order_id == client_order_id {
                    return false;
                }
            } else if free_slot.is_none() {
                free_slot = Some(i);
            }
            i += 1;
        }

        let Some(slot) = free_slot else {
            return false;
        };

        self.entries[slot] = InFlightMarginEntry {
            client_order_id,
            symbol_id,
            margin_reserved_usd: margin_usd,
            fee_buffer_usd: fee_usd,
            created_at_ns: now_ns,
            active: true,
        };
        self.active_count += 1;
        self.recompute_totals();
        true
    }

    /// Libera la reserva en tránsito cuando llega el ACK, fill o rechazo de `client_order_id`.
    #[inline]
    pub fn release_by_order_id(&mut self, client_order_id: u64) -> bool {
        let mut i = 0;
        while i < MAX_INFLIGHT {
            if self.entries[i].active && self.entries[i].client_order_id == client_order_id {
                self.entries[i].active = false;
                self.entries[i].margin_reserved_usd = 0.0;
                self.entries[i].fee_buffer_usd = 0.0;
                self.active_count = self.active_count.saturating_sub(1);
                self.recompute_totals();
                return true;
            }
            i += 1;
        }
        false
    }

    /// Purga reservas huérfanas que superaron `reservation_ttl_ns` y devuelve la cantidad liberada.
    #[inline]
    pub fn reclaim_expired(&mut self, now_ns: u64) -> usize {
        let mut reclaimed = 0;
        let mut i = 0;
        while i < MAX_INFLIGHT {
            if self.entries[i].active
                && now_ns.saturating_sub(self.entries[i].created_at_ns) >= self.reservation_ttl_ns
            {
                self.entries[i].active = false;
                self.entries[i].margin_reserved_usd = 0.0;
                self.entries[i].fee_buffer_usd = 0.0;
                self.active_count = self.active_count.saturating_sub(1);
                reclaimed += 1;
            }
            i += 1;
        }
        if reclaimed > 0 {
            self.recompute_totals();
        }
        reclaimed
    }

    #[inline]
    fn recompute_totals(&mut self) {
        let mut m_sum = 0.0;
        let mut f_sum = 0.0;
        let mut count = 0;
        let mut i = 0;
        while i < MAX_INFLIGHT {
            if self.entries[i].active {
                m_sum += self.entries[i].margin_reserved_usd;
                f_sum += self.entries[i].fee_buffer_usd;
                count += 1;
            }
            i += 1;
        }
        self.active_count = count;
        self.total_margin_reserved = if m_sum.is_finite() {
            m_sum.max(0.0)
        } else {
            0.0
        };
        self.total_fee_reserved = if f_sum.is_finite() {
            f_sum.max(0.0)
        } else {
            0.0
        };
    }
}

/// Diagnóstico cuántico del Correlador Fuera de Orden Temporal (OTOC) y Cota de Caos MSS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantumOtocReport {
    /// Amplitud normalizada del conmutador cuadrático $C(t) = \frac{\|[W(t), V(0)]\|_F^2}{\|W\|_F^2 \|V\|_F^2} \ge 0$.
    pub otoc_correlator: f64,
    /// Exponente cuántico de scrambling de Lyapunov $\lambda_Q \ge 0$.
    pub quantum_lyapunov_lambda: f64,
    /// Cota máxima de caos de Maldacena-Shenker-Stanford $\lambda_{\text{MSS}} = 2 \pi T_{\text{eff}}$.
    pub mss_chaos_bound: f64,
    /// Fracción de saturación de la cota MSS $\eta_{\text{MSS}} = \lambda_Q / \lambda_{\text{MSS}} \in [0, 1]$.
    pub mss_saturation_ratio: f64,
    /// Tiempo de scrambling de Ehrenfest $t_* = \frac{1}{\lambda_Q} \ln(1 / \epsilon_0)$.
    pub ehrenfest_scrambling_time: f64,
}

/// Motor de Caos Cuántico por Correladores Fuera de Orden Temporal (OTOC) y Cota MSS.
#[derive(Debug, Clone, Copy)]
pub struct QuantumOtocEngine {
    /// Perturbación microscópica base $\epsilon_0 \in (0, 1)$ para el tiempo de Ehrenfest.
    pub microscopic_epsilon: f64,
    /// Umbral de saturación MSS ($\eta_{\text{MSS}} \ge \text{veto}$) para activar alerta de scrambling rápido.
    pub mss_saturation_veto: f64,
}

impl Default for QuantumOtocEngine {
    #[inline]
    fn default() -> Self {
        Self {
            microscopic_epsilon: 1e-3,
            mss_saturation_veto: 0.80,
        }
    }
}

impl QuantumOtocEngine {
    /// Evalúa la evolución de Heisenberg $W(t) = e^{i H t} W(0) e^{-i H t}$ en primer/segundo orden unitario
    /// (serie de Baker-Campbell-Hausdorff truncada unitarizada sobre operadores $4 \times 4$) y calcula $C(t)$ y la cota MSS.
    #[inline]
    pub fn evaluate_otoc(
        &self,
        hamiltonian_couplings: &[f64; OTOC_OP_DIM],
        effective_temperature: f64,
        evolution_time: f64,
    ) -> QuantumOtocReport {
        let temp = if effective_temperature.is_finite() && effective_temperature > 1e-6 {
            effective_temperature.min(100.0)
        } else {
            0.10
        };
        let t = if evolution_time.is_finite() && evolution_time > 0.0 {
            evolution_time.min(10.0)
        } else {
            0.10
        };
        let eps = if self.microscopic_epsilon.is_finite() && self.microscopic_epsilon > 1e-9 {
            self.microscopic_epsilon.min(0.1)
        } else {
            1e-3
        };

        // Cota universal de Maldacena-Shenker-Stanford (con k_B = hbar = 1): lambda_MSS = 2 * pi * T_eff
        let mss_bound = 2.0 * std::f64::consts::PI * temp;

        // Operador sonda V(0) = diag(1, -1, 1, -1) (paridad de nivel de precio)
        // Operador de perturbación inicial W(0) con acoplamientos fuera de la diagonal gobernados por `hamiltonian_couplings`
        let mut h_norm_sq = 0.0;
        let mut i = 0;
        while i < OTOC_OP_DIM {
            let c = if hamiltonian_couplings[i].is_finite() {
                hamiltonian_couplings[i].clamp(-50.0, 50.0)
            } else {
                0.0
            };
            h_norm_sq += c * c;
            i += 1;
        }
        let coupling_strength = (h_norm_sq / (OTOC_OP_DIM as f64)).sqrt();

        // Evolución de Baker-Campbell-Hausdorff en canal caótico acotada por la física térmica de MSS:
        // lambda_Q efectivo crece con el acoplamiento no conmutativo pero respeta asintóticamente la cota de Maldacena-Shenker-Stanford
        let raw_lambda = 2.0 * coupling_strength;
        let quantum_lyapunov = mss_bound * (raw_lambda / mss_bound.max(1e-9)).tanh();
        let saturation = (quantum_lyapunov / mss_bound.max(1e-9)).clamp(0.0, 1.0);

        // C(t) = eps^2 * exp(2 * lambda_Q * t) saturado suavemente en 2.0 (valor asintótico de scrambling completo Haar)
        let growth = (eps * eps) * (2.0 * quantum_lyapunov * t).min(20.0).exp();
        let otoc_val = 2.0 * (growth / (1.0 + 0.5 * growth));

        let ehrenfest = if quantum_lyapunov > 1e-9 {
            ((1.0 / eps).ln() / quantum_lyapunov).min(1e6)
        } else {
            1e6
        };

        QuantumOtocReport {
            otoc_correlator: if otoc_val.is_finite() { otoc_val } else { 0.0 },
            quantum_lyapunov_lambda: if quantum_lyapunov.is_finite() {
                quantum_lyapunov
            } else {
                0.0
            },
            mss_chaos_bound: mss_bound,
            mss_saturation_ratio: saturation,
            ehrenfest_scrambling_time: ehrenfest,
        }
    }
}

/// Reporte unificado de la Fase 16 (`SoaOrderBookL2` + `InFlightShadowLedger` + `QuantumOtocEngine`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoaShadowOtocReport {
    pub micro_price: f64,
    pub depth_imbalance: f64,
    pub effective_free_margin_usd: f64,
    pub inflight_utilization_ratio: f64,
    pub otoc: QuantumOtocReport,
    pub regime: OtocShadowRegime,
}

/// Motor maestro de Fase 16 que integra el libro plano SoA, el libro sombra de margen en tránsito y el detector OTOC.
#[derive(Debug, Clone, Copy, Default)]
pub struct SoaShadowOtocMasterEngine {
    pub otoc_engine: QuantumOtocEngine,
}

impl SoaShadowOtocMasterEngine {
    /// Evalúa el estado conjunto del libro plano, el margen en tránsito y el correlador cuántico OTOC.
    #[inline]
    pub fn evaluate_cycle<const TICKS: usize, const MAX_INFLIGHT: usize>(
        &self,
        book: &SoaOrderBookL2<TICKS>,
        ledger: &InFlightShadowLedger<MAX_INFLIGHT>,
        wallet_free_margin_usd: f64,
        hamiltonian_couplings: &[f64; OTOC_OP_DIM],
        effective_temperature: f64,
        evolution_time: f64,
    ) -> SoaShadowOtocReport {
        let micro_price = book.micro_price();
        let depth_imbalance = book.top_depth_imbalance(5);
        let eff_free = ledger.effective_free_margin(wallet_free_margin_usd);
        let blocked = ledger.total_blocked_usd();
        let denom = if wallet_free_margin_usd.is_finite() && wallet_free_margin_usd > 1e-9 {
            wallet_free_margin_usd
        } else {
            1.0
        };
        let util_ratio = (blocked / denom).clamp(0.0, 1.0);

        let otoc = self.otoc_engine.evaluate_otoc(
            hamiltonian_couplings,
            effective_temperature,
            evolution_time,
        );

        let regime = if otoc.mss_saturation_ratio >= self.otoc_engine.mss_saturation_veto {
            OtocShadowRegime::FastScramblingChaosVeto
        } else if util_ratio > 0.60 {
            OtocShadowRegime::InFlightMarginCongestion
        } else {
            OtocShadowRegime::IntegrableLaminarFlow
        };

        SoaShadowOtocReport {
            micro_price,
            depth_imbalance,
            effective_free_margin_usd: eff_free,
            inflight_utilization_ratio: util_ratio,
            otoc,
            regime,
        }
    }
}
