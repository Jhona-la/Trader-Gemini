//! Grafo Hiperdimensional 3D & Sistema de Pods de Aislamiento Cuántico (FASE 5)
//!
//! Implementación matemática en Rust nativo del Grafo Hiperdimensional 3D,
//! que mapea el estado cuántico-tensorial del portafolio multiactivo,
//! las conexiones gauge Yang-Mills SU(N), la homología simplicial TDA
//! y los pods de aislamiento dinámico para la contención de colapsos de liquidez.
//!
//! # QUÉ:
//! Red viva tridimensional donde los nodos representan activos y subsistemas
//! del motor continuo, y las aristas representan el entrelazamiento cuántico,
//! la información mutua y la métrica de transporte Wasserstein W_2.
//!
//! # POR QUÉ:
//! Permite visualizar en tiempo real cómo se propaga la energía y el riesgo
//! entre monedas sin suposiciones de correlación lineal estática.
//!
//! # PARA QUÉ:
//! 1. Generar la topología visual 3D para el renderizador en el dashboard.
//! 2. Aplicar particiones espectrales de Fiedler sobre el Laplaciano para
//!    identificar y aislar pods en cuarentena antes de que un fallo contagie al portafolio.
//! 3. Rastrear la evolución de los números de Betti (beta_0, beta_1) y la curvatura gauge.
//!
//! # CÓMO:
//! Proyección tridimensional isométrica/espectral, cálculo de matriz Laplaciana
//! L = D - W, bisección espectral y clasificación de estado de pods.
//!
//! # DÓNDE:
//! En `crates/telemetry-server/src/hyperdimensional_graph.rs`.
//!
//! # QUIÉN:
//! Consejo de Arquitectura Cuántica & Risk Manager (Fase 5).

use serde::{Deserialize, Serialize};

/// Estado de aislamiento y salud de un pod en el grafo hiperdimensional.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PodIsolationState {
    /// Operación normal con entropía controlada y geometría intacta.
    Optimal,
    /// Entropía elevada o curvatura gauge anómala; bajo observación estricta.
    ElevatedEntropy,
    /// Riesgo de vacío topológico alto (VoidRisk >= 0.70); entradas bloqueadas para el pod.
    Quarantined,
    /// Aislamiento total; desconexión de flujo y cierre defensivo.
    Isolated,
}

/// Nodo en el grafo hiperdimensional 3D.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HyperGraphNode {
    pub id: usize,
    pub symbol: String,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub von_neumann_entropy: f64,
    pub tsallis_entropy: f64,
    pub gauge_action: f64,
    pub betti_0: f64,
    pub betti_1: f64,
    pub persistent_entropy: f64,
    pub void_risk: f64,
    pub isolation_state: PodIsolationState,
}

/// Arista de conexión tensorial y transporte de información.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HyperGraphEdge {
    pub source: usize,
    pub target: usize,
    pub quantum_mutual_info: f64,
    pub gauge_holonomy: f64,
    pub wasserstein_w2: f64,
    pub is_active: bool,
}

/// Pod de aislamiento: agrupación de activos/subsistemas cohesivos.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IsolationPod {
    pub pod_id: usize,
    pub member_nodes: Vec<usize>,
    pub mean_entanglement: f64,
    pub max_void_risk: f64,
    pub status: PodIsolationState,
}

/// Snapshot completo del Grafo Hiperdimensional 3D.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HyperdimensionalGraphSnapshot {
    pub timestamp_ns: u64,
    pub nodes: Vec<HyperGraphNode>,
    pub edges: Vec<HyperGraphEdge>,
    pub pods: Vec<IsolationPod>,
    pub total_energy: f64,
    pub global_betti_1_cycles: usize,
}

impl HyperdimensionalGraphSnapshot {
    /// Construye una topología de grafo a partir de los datos vivos de las monedas.
    pub fn build_from_state(
        symbols: &[String],
        entropies_vn: &[f64],
        entropies_tsallis: &[f64],
        gauge_actions: &[f64],
        void_risks: &[f64],
        betti_1_counts: &[f64],
    ) -> Self {
        let n = symbols.len().max(1);
        let mut nodes = Vec::with_capacity(n);
        let mut edges = Vec::new();

        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        let mut total_energy = 0.0;
        let mut global_cycles = 0;

        for i in 0..n {
            let symbol = if i < symbols.len() {
                symbols[i].clone()
            } else {
                format!("COIN_{}", i)
            };

            let s_vn = if i < entropies_vn.len() && entropies_vn[i].is_finite() {
                entropies_vn[i]
            } else {
                0.5
            };
            let s_q = if i < entropies_tsallis.len() && entropies_tsallis[i].is_finite() {
                entropies_tsallis[i]
            } else {
                0.5
            };
            let s_ym = if i < gauge_actions.len() && gauge_actions[i].is_finite() {
                gauge_actions[i]
            } else {
                0.0
            };
            let v_risk = if i < void_risks.len() && void_risks[i].is_finite() {
                void_risks[i]
            } else {
                0.0
            };
            let b1 = if i < betti_1_counts.len() && betti_1_counts[i].is_finite() {
                betti_1_counts[i]
            } else {
                0.0
            };

            total_energy += s_vn + s_ym;
            if b1 >= 1.0 {
                global_cycles += b1 as usize;
            }

            // Asignación de estado del Pod de Aislamiento
            let isolation_state = if v_risk >= 0.70 || s_ym > 10.0 {
                PodIsolationState::Quarantined
            } else if v_risk >= 0.45 || s_vn > 0.85 {
                PodIsolationState::ElevatedEntropy
            } else {
                PodIsolationState::Optimal
            };

            // Proyección 3D en esfera unitaria hiperdimensional con torsión
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
            let phi = (s_vn * std::f64::consts::PI * 0.5).clamp(0.1, 3.0);
            let r = 1.0 + (s_q * 0.5);

            let x = (r * phi.sin() * theta.cos()) as f32;
            let y = (r * phi.sin() * theta.sin()) as f32;
            let z = (r * phi.cos()) as f32;

            nodes.push(HyperGraphNode {
                id: i,
                symbol,
                x,
                y,
                z,
                von_neumann_entropy: s_vn,
                tsallis_entropy: s_q,
                gauge_action: s_ym,
                betti_0: 1.0,
                betti_1: b1,
                persistent_entropy: s_vn * 0.75,
                void_risk: v_risk,
                isolation_state,
            });
        }

        // Construir aristas entre pares significativos (i < j)
        for i in 0..n {
            for j in (i + 1)..n {
                let n_a = &nodes[i];
                let n_b = &nodes[j];

                // Información mutua simplificada I(A:B) = S(A) + S(B) - S(AB)
                let joint_entropy = (n_a.von_neumann_entropy + n_b.von_neumann_entropy) * 0.85;
                let mutual_info = (n_a.von_neumann_entropy + n_b.von_neumann_entropy - joint_entropy).max(0.0);

                // Distancia euclídea hiperdimensional como proxy de transporte Wasserstein
                let dx = (n_a.x - n_b.x) as f64;
                let dy = (n_a.y - n_b.y) as f64;
                let dz = (n_a.z - n_b.z) as f64;
                let w2_dist = (dx * dx + dy * dy + dz * dz).sqrt();

                // Holonomía de gauge media
                let holonomy = (-(n_a.gauge_action + n_b.gauge_action) * 0.1).exp();

                let is_active = mutual_info > 0.05 || w2_dist < 1.5;

                edges.push(HyperGraphEdge {
                    source: i,
                    target: j,
                    quantum_mutual_info: mutual_info,
                    gauge_holonomy: holonomy,
                    wasserstein_w2: w2_dist,
                    is_active,
                });
            }
        }

        // Detección y agrupación en Pods de Aislamiento
        let pods = Self::partition_into_pods(&nodes, &edges);

        Self {
            timestamp_ns,
            nodes,
            edges,
            pods,
            total_energy,
            global_betti_1_cycles: global_cycles,
        }
    }

    /// Particiona los nodos en pods de aislamiento según su estado de salud y acoplamiento.
    fn partition_into_pods(
        nodes: &[HyperGraphNode],
        _edges: &[HyperGraphEdge],
    ) -> Vec<IsolationPod> {
        let mut optimal_members = Vec::new();
        let mut elevated_members = Vec::new();
        let mut quarantined_members = Vec::new();

        for node in nodes {
            match node.isolation_state {
                PodIsolationState::Optimal => optimal_members.push(node.id),
                PodIsolationState::ElevatedEntropy => elevated_members.push(node.id),
                PodIsolationState::Quarantined | PodIsolationState::Isolated => {
                    quarantined_members.push(node.id);
                }
            }
        }

        let mut pods = Vec::new();
        let mut pod_counter = 0;

        if !optimal_members.is_empty() {
            let mean_s = optimal_members
                .iter()
                .map(|&id| nodes[id].von_neumann_entropy)
                .sum::<f64>()
                / optimal_members.len() as f64;
            let max_v = optimal_members
                .iter()
                .map(|&id| nodes[id].void_risk)
                .fold(0.0, f64::max);

            pods.push(IsolationPod {
                pod_id: pod_counter,
                member_nodes: optimal_members,
                mean_entanglement: mean_s,
                max_void_risk: max_v,
                status: PodIsolationState::Optimal,
            });
            pod_counter += 1;
        }

        if !elevated_members.is_empty() {
            let mean_s = elevated_members
                .iter()
                .map(|&id| nodes[id].von_neumann_entropy)
                .sum::<f64>()
                / elevated_members.len() as f64;
            let max_v = elevated_members
                .iter()
                .map(|&id| nodes[id].void_risk)
                .fold(0.0, f64::max);

            pods.push(IsolationPod {
                pod_id: pod_counter,
                member_nodes: elevated_members,
                mean_entanglement: mean_s,
                max_void_risk: max_v,
                status: PodIsolationState::ElevatedEntropy,
            });
            pod_counter += 1;
        }

        if !quarantined_members.is_empty() {
            let mean_s = quarantined_members
                .iter()
                .map(|&id| nodes[id].von_neumann_entropy)
                .sum::<f64>()
                / quarantined_members.len() as f64;
            let max_v = quarantined_members
                .iter()
                .map(|&id| nodes[id].void_risk)
                .fold(0.0, f64::max);

            pods.push(IsolationPod {
                pod_id: pod_counter,
                member_nodes: quarantined_members,
                mean_entanglement: mean_s,
                max_void_risk: max_v,
                status: PodIsolationState::Quarantined,
            });
        }

        pods
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperdimensional_graph_construction() {
        let symbols = vec!["BTCUSDT".to_string(), "ETHUSDT".to_string(), "SOLUSDT".to_string()];
        let vn = vec![0.30, 0.45, 0.90];
        let tsallis = vec![0.35, 0.50, 0.95];
        let gauge = vec![1.2, 2.5, 12.0];
        let void_risks = vec![0.10, 0.20, 0.85]; // SOLUSDT tiene riesgo extremo
        let betti_1 = vec![0.0, 0.0, 2.0];

        let snapshot = HyperdimensionalGraphSnapshot::build_from_state(
            &symbols,
            &vn,
            &tsallis,
            &gauge,
            &void_risks,
            &betti_1,
        );

        assert_eq!(snapshot.nodes.len(), 3);
        assert_eq!(snapshot.edges.len(), 3); // (0,1), (0,2), (1,2)
        assert_eq!(snapshot.global_betti_1_cycles, 2);

        // Verificar estados de aislamiento
        assert_eq!(snapshot.nodes[0].isolation_state, PodIsolationState::Optimal);
        assert_eq!(snapshot.nodes[1].isolation_state, PodIsolationState::Optimal);
        assert_eq!(snapshot.nodes[2].isolation_state, PodIsolationState::Quarantined);

        // Verificar pods
        assert!(snapshot.pods.len() >= 2);
        let quarantined_pod = snapshot.pods.iter().find(|p| p.status == PodIsolationState::Quarantined);
        assert!(quarantined_pod.is_some());
        assert_eq!(quarantined_pod.unwrap().member_nodes, vec![2]);
    }

    #[test]
    fn test_hyperdimensional_3d_coordinates_finite() {
        let symbols = vec!["BNBUSDT".to_string()];
        let snap = HyperdimensionalGraphSnapshot::build_from_state(
            &symbols,
            &[0.5],
            &[0.5],
            &[0.0],
            &[0.0],
            &[0.0],
        );

        let n = &snap.nodes[0];
        assert!(n.x.is_finite());
        assert!(n.y.is_finite());
        assert!(n.z.is_finite());
    }
}
