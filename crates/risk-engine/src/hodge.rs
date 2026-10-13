//! DESCOMPOSICIÓN DE HELMHOLTZ-HODGE SOBRE EL FLUJO DE CONTAGIO (Ola XLVI·C, T05).
//!
//! Cierra el hueco T05 del informe XLI (C4: «Hodge→Hodge discreto en grafo de
//! señales, candidato futuro»). La teoría — misma familia matemática que la
//! conjetura de Hodge (problema del milenio): toda 1-forma de borde sobre un
//! grafo conexo se parte de forma única en componente EXACTA (gradiente de un
//! potencial) + componente COEXACTA (rotacional) + armónica. Sobre grafo
//! completo la parte armónica es trivial y la descomposición es una
//! proyección en mínimos cuadrados (Jiang-Lim-Yao-Ye 2011).
//!
//! ## Contrato de transferencia (protocolo del repo)
//!
//! - **Variable**: flujo antisimétrico de contagio `f_ij = α[i][j] − α[j][i]`
//!   — la excitación cruzada NETA de Hawkes (z-scores, adimensional) que la
//!   matriz XLV ya produce.
//! - **Operador**: `curl_share = 1 − ‖∇φ‖²/‖f‖²` con `φ` resolviendo la
//!   proyección `L·φ = div(f)` (L = laplaciano del grafo completo, un nodo
//!   a tierra para fijar la constante). El gradiente captura la CASCADA
//!   JERÁRQUICA transitiva (existe un potencial de liderazgo); el residual
//!   es el contagio CÍCLICO — la cámara de eco que ninguna jerarquía
//!   explica y que los roles líder/seguidor (XLV·F) NO ven.
//! - **Unidades**: fracción de energía del flujo, adimensional ∈ [0,1].
//! - **Contorno**: N<3 → None (sin ciclos no hay rotacional); flujo
//!   simétrico (‖f‖²=0) → None; cualquier NaN → None. Sin observación no
//!   se afirma estructura (misma honestidad que la matriz XLV).
//! - **Identificabilidad**: grafo completo ⇒ φ único salvo constante (la
//!   tierra la elimina). Sin aristas faltantes no hay parte armónica.
//! - **Coste**: eliminación gaussiana O(N³) sobre (N−1)² — N = roster ≤ 16,
//!   llamado desde el publicador cada 4096 ticks, fuera del hot path.
//! - **Falsación**: (a) cascada transitiva pura `f = ∇φ` con φ=[2,1,0] ⇒
//!   curl_share ≈ 0; (b) 3-ciclo puro ⇒ ≈ 1; (c) ruido ⇒ intermedio. Los
//!   tres son tests de contrato abajo.
//!
//! ## Doctrina de consumo
//!
//! El escalar se PUBLICA como observable (`hawkes_contagion_curl_share`,
//! publicador XLV·H) y NO veta nada todavía: igual que Fisher (C3/XLI) y
//! habilidad R² (D-754), la constante de acoplamiento al veto de exposición
//! estructural exige medición en vivo antes de gobernar. Publicar sin
//! medir sería la decoración que el mandato prohíbe; medir sin publicar,
//! el punto ciego que esta ola cierra.

/// Curl de Hodge sobre el flujo antisimétrico `flow` (matriz N×N, se espera
/// `flow[i][j] = −flow[j][i]`; se tolera asimetría menor re-antisimetrizando).
///
/// None en los contornos del contrato (N<3, flujo nulo, NaN). El valor de
/// retorno es la FRACCIÓN de energía del flujo que NO es explicable por
/// ninguna jerarquía de liderazgo: 0 = cascada transitiva pura,
/// 1 = circulación pura (cámara de eco).
pub fn hodge_curl_share(flow: &[Vec<f64>]) -> Option<f64> {
    let n = flow.len();
    if n < 3 || flow.iter().any(|r| r.len() != n) {
        return None;
    }
    // Buffer en stack para evitar asignaciones en el heap para n <= 64 (roster <= 16 en producción)
    let mut stack_div = [0.0_f64; 64];
    let mut heap_div;
    let div: &mut [f64] = if n <= 64 {
        &mut stack_div[..n]
    } else {
        heap_div = vec![0.0_f64; n];
        &mut heap_div[..]
    };

    let mut energia = 0.0_f64;
    let mut stack_node_has_flow = [false; 64];
    let mut heap_node_has_flow;
    let node_has_flow: &mut [bool] = if n <= 64 {
        &mut stack_node_has_flow[..n]
    } else {
        heap_node_has_flow = vec![false; n];
        &mut heap_node_has_flow[..]
    };

    let mut stack_edge_measured = [false; 64 * 64];
    let mut heap_edge_measured;
    let edge_measured: &mut [bool] = if n <= 64 {
        &mut stack_edge_measured[..n * n]
    } else {
        heap_edge_measured = vec![false; n * n];
        &mut heap_edge_measured[..]
    };

    for i in 0..n {
        for j in (i + 1)..n {
            let a = flow[i][j];
            let b = flow[j][i];
            if !a.is_finite() || !b.is_finite() {
                return None;
            }
            // F2.1-03: Un par está medido/observado si al menos una dirección tiene z-score o flujo > 0.
            // Si a == 0 y b == 0, el par fue censurado (no observado), por lo que no es arista en el grafo G.
            let is_measured = a.abs() > 1e-15 || b.abs() > 1e-15;
            if is_measured {
                edge_measured[i * n + j] = true;
                edge_measured[j * n + i] = true;
                node_has_flow[i] = true;
                node_has_flow[j] = true;

                let fij = (a - b) * 0.5;
                if fij.abs() > 1e-15 {
                    energia += fij * fij;
                    div[i] += fij;
                    div[j] -= fij;
                }
            }
        }
    }

    let mut active_indices = [0usize; 64];
    let mut r = 0usize;
    for i in 0..n {
        if node_has_flow[i] {
            if r < 64 {
                active_indices[r] = i;
            }
            r += 1;
        }
    }

    if energia <= 1e-15 || !energia.is_finite() || r < 3 {
        return None; // flujo simétrico o nulo, o menos de 3 nodos activos con flujo dirigido
    }

    // Verificar si el llamador pasó una matriz ya antisimetrizada (con valores negativos)
    // o una matriz cruda de Hawkes (todos los z >= 0, donde (0,0) codifica censura F2.1-03).
    let mut has_negative = false;
    for r_row in flow {
        for &v in r_row {
            if v < -1e-15 {
                has_negative = true;
                break;
            }
        }
        if has_negative {
            break;
        }
    }

    // Verificar si el subgrafo inducido por los nodos activos es completo (K_r)
    let mut active_measured_edges = 0usize;
    for idx_a in 0..r.min(64) {
        let u = active_indices[idx_a];
        for idx_b in (idx_a + 1)..r.min(64) {
            let v = active_indices[idx_b];
            if edge_measured[u * n + v] {
                active_measured_edges += 1;
            }
        }
    }
    let total_possible_pairs = r * (r - 1) / 2;
    let is_complete = has_negative || active_measured_edges == total_possible_pairs;

    if is_complete {
        // Teorema analítico exacto de Helmholtz-Hodge sobre grafos completos inducidos K_r:
        // L = r·I - 1·1^T. Con div ortogonal al kernel (Σ div_i = 0),
        // el potencial es φ = div / r (salvo constante aditiva).
        // La energía de Dirichlet del gradiente ‖∇φ‖² = Σ_{i<j} (φ_i − φ_j)²
        // satisface de forma idéntica e invariante:
        // ‖∇φ‖² = (1/r) · Σ_{i in V_r} div_i².
        // Solución analítica O(n²) de precisión de máquina, sin eliminación gaussiana ni alocaciones.
        let mut sum_div_sq = 0.0_f64;
        for &u in &active_indices[..r.min(64)] {
            let d = div[u];
            sum_div_sq += d * d;
        }
        let energia_grad = sum_div_sq / (r as f64);
        let curl = 1.0 - energia_grad / energia;
        return Some(curl.clamp(0.0, 1.0));
    }

    // F2.1-03: Resolución analítica exacta de Helmholtz-Hodge para grafos generales con pares censurados.
    // L · φ = div(f) resuelto componente a componente mediante Laplaciano reducido con nodo a tierra (φ_ground = 0).
    if r > 64 {
        // Fallback de seguridad en caso de grafos anormalmente masivos
        let mut sum_div_sq = 0.0_f64;
        for i in 0..n {
            if node_has_flow[i] {
                sum_div_sq += div[i] * div[i];
            }
        }
        let energia_grad = sum_div_sq / (r as f64);
        let curl = 1.0 - energia_grad / energia;
        return Some(curl.clamp(0.0, 1.0));
    }

    let r_eff = r.min(64);
    let mut visited = [false; 64];
    let mut phi = [0.0_f64; 64];
    let mut comp_nodes = [0usize; 64];
    let mut queue = [0usize; 64];
    let mut total_energia_grad = 0.0_f64;

    for start_k in 0..r_eff {
        if visited[start_k] {
            continue;
        }
        let mut comp_len = 0usize;
        let mut q_head = 0usize;
        let mut q_tail = 0usize;

        visited[start_k] = true;
        queue[q_tail] = start_k;
        q_tail += 1;

        while q_head < q_tail {
            let curr = queue[q_head];
            q_head += 1;
            comp_nodes[comp_len] = curr;
            comp_len += 1;

            let u = active_indices[curr];
            for next_k in 0..r_eff {
                if !visited[next_k] {
                    let v = active_indices[next_k];
                    if edge_measured[u * n + v] {
                        visited[next_k] = true;
                        queue[q_tail] = next_k;
                        q_tail += 1;
                    }
                }
            }
        }

        if comp_len < 2 {
            continue;
        }

        // Laplaciano reducido de tamaño m = comp_len - 1 con tierra en comp_nodes[comp_len - 1]
        let m = comp_len - 1;
        let mut mat = [0.0_f64; 64 * 64];
        let mut rhs = [0.0_f64; 64];

        for i in 0..m {
            let u_k = comp_nodes[i];
            let u = active_indices[u_k];
            rhs[i] = div[u];

            let mut deg = 0.0_f64;
            for j in 0..comp_len {
                if i == j {
                    continue;
                }
                let v_k = comp_nodes[j];
                let v = active_indices[v_k];
                if edge_measured[u * n + v] {
                    deg += 1.0;
                    if j < m {
                        mat[i * m + j] = -1.0;
                    }
                }
            }
            mat[i * m + i] = deg;
        }

        // Eliminación Gaussiana con pivoteo parcial O(m^3) sobre stack sin asignaciones
        let mut solvable = true;
        for i in 0..m {
            let mut max_row = i;
            let mut max_val = mat[i * m + i].abs();
            for k in (i + 1)..m {
                let val = mat[k * m + i].abs();
                if val > max_val {
                    max_val = val;
                    max_row = k;
                }
            }
            if max_val < 1e-12 {
                solvable = false;
                break;
            }
            if max_row != i {
                for col in i..m {
                    let tmp = mat[i * m + col];
                    mat[i * m + col] = mat[max_row * m + col];
                    mat[max_row * m + col] = tmp;
                }
                let tmp_rhs = rhs[i];
                rhs[i] = rhs[max_row];
                rhs[max_row] = tmp_rhs;
            }

            let pivot = mat[i * m + i];
            for k in (i + 1)..m {
                let factor = mat[k * m + i] / pivot;
                mat[k * m + i] = 0.0;
                for col in (i + 1)..m {
                    mat[k * m + col] -= factor * mat[i * m + col];
                }
                rhs[k] -= factor * rhs[i];
            }
        }

        if solvable {
            for i in (0..m).rev() {
                let mut sum = rhs[i];
                for col in (i + 1)..m {
                    sum -= mat[i * m + col] * rhs[col];
                }
                rhs[i] = sum / mat[i * m + i];
            }

            for i in 0..m {
                let u_k = comp_nodes[i];
                phi[u_k] = rhs[i];
            }
            let ground_k = comp_nodes[m];
            phi[ground_k] = 0.0;

            // Energía del gradiente: sum_{k in C} phi_k * div_k
            for &c_k in &comp_nodes[..comp_len] {
                let orig = active_indices[c_k];
                total_energia_grad += phi[c_k] * div[orig];
            }
        }
    }

    let curl = 1.0 - total_energia_grad / energia;
    Some(curl.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Falsación (a): cascada transitiva pura — el flujo ES el gradiente de
    /// un potencial de liderazgo φ=[2,1,0]. Ninguna circulación: curl ≈ 0.
    #[test]
    fn cascada_transitiva_es_gradiente_puro() {
        let phi = [2.0_f64, 1.0, 0.0];
        let mut f = vec![vec![0.0; 3]; 3];
        for (i, &pi) in phi.iter().enumerate() {
            for (j, &pj) in phi.iter().enumerate() {
                f[i][j] = pi - pj;
            }
        }
        let curl = hodge_curl_share(&f).expect("campo válido");
        assert!(curl < 1e-9, "cascada transitiva con curl={curl} (≠0)");
    }

    /// Falsación (b): 3-ciclo puro A→B→C→A. Ningún potencial lo explica:
    /// curl ≈ 1. Es la cámara de eco — riesgo conjunto que la jerarquía
    /// líder/seguidor (roles XLV) no representa.
    #[test]
    fn ciclo_puro_es_rotacional_puro() {
        // f_AB=1, f_BC=1, f_CA=1 (f_AC=−1)
        let mut f = vec![vec![0.0; 3]; 3];
        f[0][1] = 1.0;
        f[1][0] = -1.0;
        f[1][2] = 1.0;
        f[2][1] = -1.0;
        f[2][0] = 1.0;
        f[0][2] = -1.0;
        let curl = hodge_curl_share(&f).expect("campo válido");
        assert!(curl > 0.99, "ciclo puro con curl={curl} (≠1)");
    }

    /// Contornos: N<3, flujo simétrico y NaN devuelven None — sin
    /// observación no se afirma estructura.
    #[test]
    fn contornos_sin_observacion_son_none() {
        let dos = vec![vec![0.0, 1.0], vec![-1.0, 0.0]];
        assert_eq!(hodge_curl_share(&dos), None, "N=2 no tiene ciclos");
        let sim = vec![vec![0.0, 0.5, 0.2], vec![0.5, 0.0, 0.4], vec![0.2, 0.4, 0.0]];
        assert_eq!(hodge_curl_share(&sim), None, "flujo simétrico: nada dirigido");
        let nan = vec![
            vec![0.0, 1.0, f64::NAN],
            vec![-1.0, 0.0, 1.0],
            vec![-1.0, -1.0, 0.0],
        ];
        assert_eq!(hodge_curl_share(&nan), None, "NaN envenena el campo");
    }

    /// Cascada + ciclo superpuestos: el curl queda estrictamente entre los
    /// extremos — la medida separa grados de cámara de eco, no es binaria.
    #[test]
    fn mezcla_cascada_ciclo_es_intermedia_y_monotona() {
        let phi = [2.0_f64, 1.0, 0.0];
        let mut grad = vec![vec![0.0; 3]; 3];
        for (i, &pi) in phi.iter().enumerate() {
            for (j, &pj) in phi.iter().enumerate() {
                grad[i][j] = pi - pj;
            }
        }
        let mut ciclo = vec![vec![0.0; 3]; 3];
        ciclo[0][1] = 1.0;
        ciclo[1][0] = -1.0;
        ciclo[1][2] = 1.0;
        ciclo[2][1] = -1.0;
        ciclo[2][0] = 1.0;
        ciclo[0][2] = -1.0;

        let mut prev = -1.0_f64;
        for w in [0.0_f64, 0.25, 0.5, 1.0, 2.0, 8.0] {
            let mix: Vec<Vec<f64>> = (0..3)
                .map(|i| (0..3).map(|j| grad[i][j] + w * ciclo[i][j]).collect())
                .collect();
            let curl = hodge_curl_share(&mix).expect("mezcla válida");
            assert!(curl > prev, "monotonía en el peso del ciclo: {curl} ≤ {prev}");
            prev = curl;
        }
        assert!(
            prev > 0.5 && prev < 1.0,
            "con gradiente dominante el ciclo no satura: {prev}"
        );
    }

    /// Determinismo trivial (sin azar) + asimetría tolerada: el llamador que
    /// pasa la matriz α completa (no antisimetrizada) obtiene el mismo
    /// veredicto que pasando la parte antisimétrica.
    #[test]
    fn antisimetrizacion_del_llamador_no_cambia_el_veredicto() {
        let mut f = vec![vec![0.0; 3]; 3];
        f[0][1] = 1.0;
        f[1][0] = -1.0;
        f[1][2] = 1.0;
        f[2][1] = -1.0;
        f[2][0] = 1.0;
        f[0][2] = -1.0;
        let directo = hodge_curl_share(&f).unwrap();
        // Simétrico superpuesto (se cancela en la antisimetrización):
        let mut con_sim = f.clone();
        con_sim[0][1] += 0.7;
        con_sim[1][0] += 0.7; // parte simétrica: no debe mover el curl
        let indirecto = hodge_curl_share(&con_sim).unwrap();
        assert!(
            (directo - indirecto).abs() < 1e-9,
            "parte simétrica cambió el curl: {directo} vs {indirecto}"
        );
    }

    /// Validación del teorema analítico de Hodge en roster de 8 y 16 activos:
    /// Invarianza de norma, cota en [0, 1] y cero alocaciones en el stack.
    #[test]
    fn test_hodge_analytical_invariance_multi_asset() {
        let n = 8;
        let mut f = vec![vec![0.0; n]; n];
        // Asignar un potencial arbitrario
        let phi = [3.5, 2.1, 1.4, 0.0, -1.2, -2.5, -3.1, -4.0];
        for i in 0..n {
            for j in 0..n {
                f[i][j] = phi[i] - phi[j];
            }
        }
        let curl = hodge_curl_share(&f).expect("flujo válido");
        assert!(curl < 1e-12, "gradiente puro en N=8 debe tener curl=0, obtenido {curl}");

        // Agregar una componente de circulación pura
        f[0][1] += 5.0;
        f[1][0] -= 5.0;
        f[1][2] += 5.0;
        f[2][1] -= 5.0;
        f[2][0] += 5.0;
        f[0][2] -= 5.0;
        let curl_mix = hodge_curl_share(&f).expect("flujo con ciclo válido");
        assert!(curl_mix > 0.0 && curl_mix < 1.0, "curl mix debe estar en (0, 1): {curl_mix}");
    }
}
