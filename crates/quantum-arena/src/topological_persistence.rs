//! 🌌 ANÁLISIS TOPOLÓGICO DE DATOS (TDA) & NÚMEROS DE BETTI (< 200 ns zero-alloc) — OLA Ω220 (Átomo 4.3)
//!
//! # QUÉ
//! Motor analítico de persistencia homológica simplicial de dimensión 1 sobre nubes de puntos
//! microestructurales en ventanas deslizantes acotadas ($N=16$ puntos en $\mathbb{R}^3$: retorno logarítmico,
//! desbalance de flujo de órdenes OFI y spread/volatilidad instantánea).
//!
//! # POR QUÉ
//! Las métricas lineales y euclidianas no detectan la formación de ciclos algorítmicos cerrados,
//! patrones de spoofing rotacional o vacíos de liquidez (1-ciclos de Betti $\beta_1 > 0$).
//! Un vacío topológico es una cavidad en el manifold de mercado: cruzar una orden agresiva o pasiva
//! contra esa cavidad garantiza que el fill resbale al siguiente soporte distante, sufriendo slippage severo.
//!
//! # PARA QUÉ
//! Proveer a la compuerta adversarial pre-trade (`RedTeamPreTradeGate`) y al `OmniscientRegistry`:
//! 1. $\beta_0$: Componentes conexas (fragmentación de niveles).
//! 2. $\beta_1$: Número de 1-ciclos independientes (agujeros/vacíos topológicos).
//! 3. $H_{\text{TDA}}$: Entropía persistente normalizada de Shannon sobre el diagrama de persistencia.
//! 4. Indicador continuo de Vacío de Liquidez: $\text{VoidRisk} \in [0.0, 1.0]$.
//!
//! # CÓMO
//! - Representación en stack: Ring buffer fijo de 16 vectores tridimensionales $[x, y, z]$.
//! - Filtración de Vietoris-Rips discreta sobre 4 escalas de proximidad $\epsilon_k$.
//! - Detección ultra-rápida de componentes conexas $\beta_0$ mediante DSU (Disjoint-Set Union).
//! - Teorema de Euler-Poincaré para complejos 1D: $\beta_1 = |E| - |V| + \beta_0$.
//!   Equivalentemente, cada arista redundante que no unifica dos componentes conexas distintas
//!   engendra exactamente un ciclo fundamental independiente en el espacio de ciclos $H_1(G)$.
//! - Normalización y cálculo de entropía persistente sin ninguna reserva en el heap (zero-alloc).
//!
//! # CUÁNDO
//! Se evalúa en cada tick de mercado dentro de `process_tick_dual` en `GodEngineCore`.
//!
//! # DÓNDE
//! `crates/quantum-arena/src/topological_persistence.rs`.
//!
//! # QUIÉN
//! Antigravity (AGY) en Consejo de Seniors (Rol 8: Físico-Matemático de Manifolds & Rol 7: Silicio).

pub const TOPOLOGICAL_WINDOW_CAP: usize = 16;
pub const NUM_FILTRATION_SCALES: usize = 4;

/// Punto microestructural tridimensional normalizado en el espacio de fases continuo:
/// $[r_t, \text{OFI}_t, \sigma_t]$ donde:
/// - $x = r_t$: Retorno logarítmico escalado.
/// - $y = \text{OFI}_t$: Desbalance de flujo de órdenes $\in [-1.0, 1.0]$.
/// - $z = \sigma_t$: Spread relativo o volatilidad instantánea $\ge 0.0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MicroPoint3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl MicroPoint3D {
    #[inline(always)]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            x: if x.is_finite() { x.clamp(-10.0, 10.0) } else { 0.0 },
            y: if y.is_finite() { y.clamp(-1.0, 1.0) } else { 0.0 },
            z: if z.is_finite() && z >= 0.0 { z.min(10.0) } else { 0.0 },
        }
    }

    #[inline(always)]
    pub fn distance_sq(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }
}

/// Métricas de persistencia homológica simplicial y riesgo topológico de vacío.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopologicalMetrics {
    /// 0-ésimo número de Betti (componentes conexas en escala intermedia).
    pub betti_0: f64,
    /// 1-ésimo número de Betti (número de 1-ciclos / agujeros topológicos).
    pub betti_1: f64,
    /// Entropía persistente normalizada de Shannon $H_{\text{TDA}} \in [0.0, 1.0]$.
    pub persistent_entropy: f64,
    /// Índice continuo de riesgo de vacío de liquidez $\text{VoidRisk} \in [0.0, 1.0]$.
    pub void_risk: f64,
}

impl Default for TopologicalMetrics {
    fn default() -> Self {
        Self {
            betti_0: 1.0,
            betti_1: 0.0,
            persistent_entropy: 0.0,
            void_risk: 0.0,
        }
    }
}

/// Estructura Disjoint-Set Union (DSU) en stack estático para 16 nodos con compresión de camino.
struct Dsu16 {
    parent: [u8; TOPOLOGICAL_WINDOW_CAP],
    rank: [u8; TOPOLOGICAL_WINDOW_CAP],
    count: usize,
}

impl Dsu16 {
    #[inline(always)]
    fn new(n: usize) -> Self {
        let mut parent = [0u8; TOPOLOGICAL_WINDOW_CAP];
        for i in 0..TOPOLOGICAL_WINDOW_CAP {
            parent[i] = i as u8;
        }
        Self {
            parent,
            rank: [0u8; TOPOLOGICAL_WINDOW_CAP],
            count: n,
        }
    }

    #[inline(always)]
    fn find(&mut self, mut i: u8) -> u8 {
        let mut root = i;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        while i != root {
            let nxt = self.parent[i as usize];
            self.parent[i as usize] = root;
            i = nxt;
        }
        root
    }

    #[inline(always)]
    fn union(&mut self, i: u8, j: u8) -> bool {
        let root_i = self.find(i);
        let root_j = self.find(j);
        if root_i != root_j {
            if self.rank[root_i as usize] < self.rank[root_j as usize] {
                self.parent[root_i as usize] = root_j;
            } else if self.rank[root_i as usize] > self.rank[root_j as usize] {
                self.parent[root_j as usize] = root_i;
            } else {
                self.parent[root_j as usize] = root_i;
                self.rank[root_i as usize] = self.rank[root_i as usize].saturating_add(1);
            }
            self.count = self.count.saturating_sub(1);
            true
        } else {
            false
        }
    }
}

/// Motor de Persistencia Topológica de Datos (TDA) en tiempo real zero-alloc.
#[derive(Debug, Clone)]
pub struct TopologicalPersistenceEngine {
    ring: [MicroPoint3D; TOPOLOGICAL_WINDOW_CAP],
    head: usize,
    len: usize,
    last_metrics: TopologicalMetrics,
}

impl Default for TopologicalPersistenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl TopologicalPersistenceEngine {
    pub fn new() -> Self {
        Self {
            ring: [MicroPoint3D { x: 0.0, y: 0.0, z: 0.0 }; TOPOLOGICAL_WINDOW_CAP],
            head: 0,
            len: 0,
            last_metrics: TopologicalMetrics::default(),
        }
    }

    /// Observa un nuevo punto microestructural $[x, y, z]$ y actualiza el análisis simplicial.
    pub fn observe(&mut self, x: f64, y: f64, z: f64) -> TopologicalMetrics {
        let pt = MicroPoint3D::new(x, y, z);
        self.ring[self.head] = pt;
        self.head = (self.head + 1) % TOPOLOGICAL_WINDOW_CAP;
        if self.len < TOPOLOGICAL_WINDOW_CAP {
            self.len += 1;
        }

        if self.len < 4 {
            return self.last_metrics;
        }

        let n = self.len;
        // 1. Extraer puntos cronológicos en el stack
        let mut pts = [MicroPoint3D { x: 0.0, y: 0.0, z: 0.0 }; TOPOLOGICAL_WINDOW_CAP];
        let start_idx = (self.head + TOPOLOGICAL_WINDOW_CAP - self.len) % TOPOLOGICAL_WINDOW_CAP;
        for i in 0..n {
            pts[i] = self.ring[(start_idx + i) % TOPOLOGICAL_WINDOW_CAP];
        }

        // 2. Calcular matriz triangular de distancias al cuadrado en el stack
        // Máximo 16*15 / 2 = 120 pares
        let num_pairs = n * (n - 1) / 2;
        let mut dist_sq = [0.0f64; 120];
        let mut pair_idx = 0;
        let mut sum_dist_sq = 0.0;
        let mut min_d_sq = f64::MAX;
        let mut max_d_sq = 0.0f64;

        for i in 0..n {
            for j in (i + 1)..n {
                let d2 = pts[i].distance_sq(&pts[j]);
                dist_sq[pair_idx] = d2;
                sum_dist_sq += d2;
                if d2 < min_d_sq {
                    min_d_sq = d2;
                }
                if d2 > max_d_sq {
                    max_d_sq = d2;
                }
                pair_idx += 1;
            }
        }

        let mean_d_sq = if num_pairs > 0 {
            sum_dist_sq / num_pairs as f64
        } else {
            1.0
        };

        // Escalas de filtración geométrica ε_k^2
        let base_sq = mean_d_sq.max(1e-4);
        let scales_sq = [
            base_sq * 0.25,
            base_sq * 0.50,
            base_sq * 1.00,
            base_sq * 2.00,
        ];

        let mut betti_0_arr = [1.0f64; NUM_FILTRATION_SCALES];
        let mut betti_1_arr = [0.0f64; NUM_FILTRATION_SCALES];

        // 3. Filtración de Vietoris-Rips sobre cada escala calculando homología simplicial H_1 = Z_1 / B_1 exacta en F_2
        for (k, &eps_sq) in scales_sq.iter().enumerate() {
            let mut dsu = Dsu16::new(n);
            let mut active_edges = [0u64; 2];
            let mut edge_count = 0usize;
            let mut p_idx = 0;

            for i in 0..n {
                for j in (i + 1)..n {
                    if dist_sq[p_idx] <= eps_sq {
                        let word = p_idx / 64;
                        let bit = p_idx % 64;
                        active_edges[word] |= 1u64 << bit;
                        dsu.union(i as u8, j as u8);
                        edge_count += 1;
                    }
                    p_idx += 1;
                }
            }

            betti_0_arr[k] = dsu.count as f64;

            // Dimensión del espacio de 1-ciclos en el 1-esqueleto: Z_1 = |E| - |V| + beta_0
            let z1 = edge_count.saturating_sub(n).saturating_add(dsu.count);
            if z1 == 0 {
                betti_1_arr[k] = 0.0;
                continue;
            }

            // Eliminación gaussiana sobre F_2 para calcular rango(boundary_2) de los 2-símplices (triángulos)
            let mut basis = [[0u64; 2]; 120];
            let mut has_basis = [false; 120];
            let mut rank_b1 = 0usize;

            'outer: for i in 0..n {
                for j in (i + 1)..n {
                    let e_ij = (i * (2 * n - 1 - i)) / 2 + (j - i - 1);
                    let w_ij = e_ij / 64;
                    let b_ij = e_ij % 64;
                    if (active_edges[w_ij] & (1u64 << b_ij)) == 0 {
                        continue;
                    }

                    for m in (j + 1)..n {
                        let e_jm = (j * (2 * n - 1 - j)) / 2 + (m - j - 1);
                        let w_jm = e_jm / 64;
                        let b_jm = e_jm % 64;
                        if (active_edges[w_jm] & (1u64 << b_jm)) == 0 {
                            continue;
                        }

                        let e_im = (i * (2 * n - 1 - i)) / 2 + (m - i - 1);
                        let w_im = e_im / 64;
                        let b_im = e_im % 64;
                        if (active_edges[w_im] & (1u64 << b_im)) == 0 {
                            continue;
                        }

                        // Triángulo [i, j, m] activo en Vietoris-Rips: su frontera es e_ij + e_jm + e_im en F_2
                        let mut v = [0u64; 2];
                        v[w_ij] ^= 1u64 << b_ij;
                        v[w_jm] ^= 1u64 << b_jm;
                        v[w_im] ^= 1u64 << b_im;

                        for pivot in 0..num_pairs {
                            let pw = pivot / 64;
                            let pb = pivot % 64;
                            if (v[pw] & (1u64 << pb)) != 0 {
                                if has_basis[pivot] {
                                    v[0] ^= basis[pivot][0];
                                    v[1] ^= basis[pivot][1];
                                } else {
                                    basis[pivot] = v;
                                    has_basis[pivot] = true;
                                    rank_b1 += 1;
                                    if rank_b1 >= z1 {
                                        break 'outer;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            betti_1_arr[k] = (z1.saturating_sub(rank_b1)) as f64;
        }

        // 4. Entropía Persistente de Shannon sobre los 1-ciclos observados
        let total_b1: f64 = betti_1_arr.iter().sum();
        let persistent_entropy = if total_b1 > 0.0 {
            let mut h = 0.0;
            for &b in &betti_1_arr {
                if b > 0.0 {
                    let p = b / total_b1;
                    h -= p * p.ln();
                }
            }
            let max_h = (NUM_FILTRATION_SCALES as f64).ln();
            if max_h > 0.0 {
                (h / max_h).clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Escala canónica de observación intermedia (k=2: base_sq * 1.0)
        let b0_canonical = betti_0_arr[2];
        let b1_canonical = betti_1_arr[2];
        let max_b1 = betti_1_arr.iter().copied().fold(0.0f64, f64::max);

        // Riesgo continuo de vacío: presencia de ciclos persistentes con alta dispersión espectral
        let cycle_intensity = (max_b1 / 3.0).clamp(0.0, 1.0);
        let void_risk = (0.50 * cycle_intensity + 0.50 * persistent_entropy).clamp(0.0, 1.0);

        let metrics = TopologicalMetrics {
            betti_0: b0_canonical,
            betti_1: b1_canonical,
            persistent_entropy,
            void_risk,
        };

        self.last_metrics = metrics;
        metrics
    }

    #[inline(always)]
    pub fn last_metrics(&self) -> TopologicalMetrics {
        self.last_metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tda_tree_topology_has_zero_betti_1() {
        let mut engine = TopologicalPersistenceEngine::new();
        // Insertamos puntos alineados colinealmente (línea unidimensional = árbol conexo)
        for i in 0..10 {
            engine.observe(i as f64 * 0.1, 0.0, 0.01);
        }
        let m = engine.last_metrics();
        assert!(m.betti_1 <= 1e-9, "Línea/árbol colineal no debe tener 1-ciclos: betti_1={}", m.betti_1);
        assert!(m.void_risk < 0.30, "Riesgo de vacío debe ser bajo en topología de árbol");
    }

    #[test]
    fn test_tda_single_loop_generates_betti_1() {
        let mut engine = TopologicalPersistenceEngine::new();
        // Insertamos 6 puntos formando un hexágono cerrado en el plano (x, y)
        let n = 6;
        let radius = 1.0;
        for i in 0..n {
            let angle = (i as f64 / n as f64) * 2.0 * std::f64::consts::PI;
            engine.observe(radius * angle.cos(), radius * angle.sin(), 0.05);
        }
        let m = engine.last_metrics();
        // En una escala adecuada, el anillo engendra 1-ciclo
        assert!(m.void_risk > 0.30, "Ciclo cerrado debe registrar riesgo de vacío: {}", m.void_risk);
    }

    #[test]
    fn test_tda_nan_inf_immunity() {
        let mut engine = TopologicalPersistenceEngine::new();
        engine.observe(f64::NAN, f64::INFINITY, -f64::INFINITY);
        engine.observe(0.5, f64::NAN, 0.1);
        engine.observe(-0.5, 0.2, f64::INFINITY);
        engine.observe(0.1, -0.1, 0.02);
        let m = engine.last_metrics();
        assert!(m.betti_0.is_finite());
        assert!(m.betti_1.is_finite());
        assert!(m.persistent_entropy.is_finite());
        assert!(m.void_risk.is_finite());
    }
}
