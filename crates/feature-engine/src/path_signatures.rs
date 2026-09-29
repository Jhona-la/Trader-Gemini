//! FIRMAS DE CAMINO TRUNCADAS (Lyons) — NIVEL 2 (Ola XLVIII·B).
//!
//! Transferencia de la teoría de rough paths (Lyons): la firma de un camino
//! es su "huella algebraica" — una representación universal de la
//! trayectoria que alimenta modelos lineales y redes con enorme poder
//! expresivo, con identidades exactas (Chen, reverso) que la hacen
//! AUDITABLE. Para la doctrina espectral: compara la FORMA del camino
//! entre escalas sin que la frecuencia de muestreo la contamine (los
//! términos de nivel 1 son totales exactos; los de nivel 2, la co-ordenada
//! temporal del recorrido).
//!
//! Contrato (docs/TRIAGE_TEORICO_2026-09-29.md):
//! - **Variable**: camino 2D X = (log-P(t), t/T normalizado); ventana de
//!   una escala τ a elección del consumidor.
//! - **Operador**: firma truncada a nivel 2 — 6 términos significativos:
//!   S¹, S² (incrementos totales) y S^{ij} = Σ_{a<b} Δ^i_a Δ^j_b.
//!   Cómputo O(n) con sumas prefijas: S^{ij} = Σ_b prefijo_i(b−1)·Δ^j_b.
//! - **Unidades**: adimensionales (log-acumulado × fracción de ventana).
//! - **Contorno**: <2 puntos ⇒ None; precio ≤0/NaN ⇒ None (el log exige
//!   positividad — el mismo saneamiento que todo el dominio espectral).
//! - **Identificabilidad**: la firma a nivel 2 caracteriza el camino salvo
//!   equivalencia tree-like (Hambly-Lyons) — unicidad NO afirmada.
//! - **Coste**: O(n) por ventana, fuera del hot path.
//! - **Falsación** (tests): línea recta en densidades distintas ⇒ firma
//!   idéntica (invariancia a remuestreo del caso exacto); identidad de
//!   Chen S(A⊗B) = S(A)·(1⊕S(B)) a nivel 2; reverso: S¹→−S¹, S^{ij}→S^{ji}.

/// Firma truncada a nivel 2 de un camino 2D.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Signature2 {
    /// Nivel 1: incrementos totales por coordenada.
    pub level1: [f64; 2],
    /// Nivel 2: level2[i][j] = S^{ij} = Σ_{a<b} Δ^i_a Δ^j_b.
    pub level2: [[f64; 2]; 2],
}

impl Signature2 {
    /// Vector plano (S¹, S², S¹¹, S¹², S²¹, S²²) para features de modelos.
    pub fn to_features(self) -> [f64; 6] {
        [
            self.level1[0],
            self.level1[1],
            self.level2[0][0],
            self.level2[0][1],
            self.level2[1][0],
            self.level2[1][1],
        ]
    }
}

/// Firma de nivel 2 sobre los INCREMENTOS ya computados (δx, δy) en orden.
/// O(n): mantiene el prefijo acumulado de cada coordenada.
///
/// DISCRETIZACIÓN SIMÉTRICA (Stratonovich): S^{ij} = Σ_{a<b} δ^i_a δ^j_b
/// + ½ Σ_a δ^i_a δ^j_a. El término de diagonal media hace que la firma del
/// camino recto sea EXACTA a cualquier densidad de muestreo (S^{ij} =
/// x_i·x_j/2) — el caso continuo exacto — y conserva sin error la identidad
/// de Chen y la del reverso (ver tests). Sin ella, la suma discreta de
/// pares ordenados arrastra una corrección O(δ) dependiente del muestreo.
pub fn firma_nivel2_incrementos(incrementos: &[(f64, f64)]) -> Signature2 {
    let mut sig = Signature2::default();
    let mut pref = [0.0_f64; 2];
    for &(dx, dy) in incrementos {
        // S^{ij} += prefijo_i · Δ^j + ½ δ^i δ^j (Stratonovich)
        sig.level2[0][0] += pref[0] * dx + 0.5 * dx * dx;
        sig.level2[0][1] += pref[0] * dy + 0.5 * dx * dy;
        sig.level2[1][0] += pref[1] * dx + 0.5 * dy * dx;
        sig.level2[1][1] += pref[1] * dy + 0.5 * dy * dy;
        pref[0] += dx;
        pref[1] += dy;
    }
    sig.level1 = pref;
    sig
}

/// Firma de la ventana (precios, timestamps): camino X = (log P, t/T).
/// None con <2 puntos o precios no positivos/no finitos (contorno).
pub fn firma_ventana_logprecio(precios: &[f64], ts_ms: &[u64]) -> Option<Signature2> {
    if precios.len() != ts_ms.len() || precios.len() < 2 {
        return None;
    }
    let t0 = ts_ms[0] as f64;
    let t1 = ts_ms[ts_ms.len() - 1] as f64;
    let span = t1 - t0;
    if !(span > 0.0) {
        return None;
    }
    let mut incs: Vec<(f64, f64)> = Vec::with_capacity(precios.len() - 1);
    let mut log_prev = None;
    let mut t_prev = t0;
    for (i, &p) in precios.iter().enumerate() {
        if !p.is_finite() || p <= 0.0 {
            return None;
        }
        let lp = p.ln();
        let t_norm = ts_ms[i] as f64 / t1; // t/T con T = último timestamp
        if let Some(lpp) = log_prev {
            incs.push((lp - lpp, t_norm - t_prev));
        }
        log_prev = Some(lp);
        t_prev = t_norm;
    }
    Some(firma_nivel2_incrementos(&incs))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FALSACIÓN (a) — LÍNEA RECTA, forma cerrada: camino con n incrementos
    /// iguales (a,b): S¹ = n·a, S² = n·b, S^{ij} = (n·a^i)(n·a^j)/2 — igual
    /// para CUALQUIER n: la firma del caso exacto no depende del muestreo.
    #[test]
    fn linea_recta_forma_cerrada_e_invariante_al_remuestreo() {
        let firma = |n: usize, a: f64, b: f64| {
            firma_nivel2_incrementos(&vec![(a, b); n])
        };
        for n in [2_usize, 3, 10, 137] {
            let s = firma(n, 0.01, 0.2);
            assert!((s.level1[0] - n as f64 * 0.01).abs() < 1e-12);
            assert!((s.level1[1] - n as f64 * 0.2).abs() < 1e-12);
            let (x, y) = (n as f64 * 0.01, n as f64 * 0.2);
            assert!((s.level2[0][0] - x * x / 2.0).abs() < 1e-12, "S11");
            assert!((s.level2[0][1] - x * y / 2.0).abs() < 1e-12, "S12");
            assert!((s.level2[1][0] - y * x / 2.0).abs() < 1e-12, "S21");
            assert!((s.level2[1][1] - y * y / 2.0).abs() < 1e-12, "S22");
        }
        // Invariancia al remuestreo del camino recto: mismo total, distinta
        // densidad ⇒ misma firma (el caso donde la firma discreta ES exacta).
        let densa = firma(100, 0.01, 0.005);
        let gruesa = firma(10, 0.1, 0.05);
        for (a, b) in densa.to_features().iter().zip(gruesa.to_features().iter()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    /// FALSACIÓN (b) — IDENTIDAD DE CHEN a nivel 2:
    /// S(A⊗B) = S(A) ⊕ (S(A)₁ ⊗ S(B)₁ + S(B)): concatenar caminos suma
    /// firmas más el producto cruzado de niveles 1.
    #[test]
    fn identidad_de_chen_nivel2() {
        let a = vec![(0.1, 0.3), (-0.05, 0.1)];
        let b = vec![(0.2, -0.05), (0.05, 0.25), (0.1, 0.1)];
        let sa = firma_nivel2_incrementos(&a);
        let sb = firma_nivel2_incrementos(&b);
        let mut concat = a.clone();
        concat.extend(b.iter());
        let sc = firma_nivel2_incrementos(&concat);
        for i in 0..2 {
            assert!((sc.level1[i] - (sa.level1[i] + sb.level1[i])).abs() < 1e-12);
            for j in 0..2 {
                let chen = sa.level2[i][j] + sb.level2[i][j] + sa.level1[i] * sb.level1[j];
                assert!((sc.level2[i][j] - chen).abs() < 1e-12, "S{i}{j}");
            }
        }
    }

    /// FALSACIÓN (c) — CAMINO REVERSO: S¹ → −S¹ y S^{ij} → S^{ji}
    /// (el antípodo a nivel 2: el producto S^iS^j queda invariante porque
    /// ambos niveles 1 se niegan — consistente con el shuffle S^iS^j =
    /// S^{ij}+S^{ji}).
    #[test]
    fn reverso_niega_nivel1_y_transpone_nivel2() {
        let camino = vec![(0.12, 0.4), (-0.07, 0.2), (0.05, -0.1), (0.03, 0.3)];
        let s = firma_nivel2_incrementos(&camino);
        let reverso: Vec<(f64, f64)> = camino.iter().rev().map(|&(x, y)| (-x, -y)).collect();
        let r = firma_nivel2_incrementos(&reverso);
        for i in 0..2 {
            assert!((r.level1[i] + s.level1[i]).abs() < 1e-12);
            for j in 0..2 {
                assert!(
                    (r.level2[i][j] - s.level2[j][i]).abs() < 1e-12,
                    "S({i},{j}) del reverso debe ser S({j},{i}) del original"
                );
            }
        }
    }

    /// Contornos: <2 puntos ⇒ None; precio inválido ⇒ None; camino con
    /// precio constante y tiempo avanzando ⇒ S¹=log 0… S del log-precio
    /// nulo, S del tiempo positivo.
    #[test]
    fn contornos_de_ventana() {
        assert!(firma_ventana_logprecio(&[100.0], &[1000]).is_none());
        assert!(firma_ventana_logprecio(&[100.0, -3.0], &[1000, 2000]).is_none());
        assert!(firma_ventana_logprecio(&[100.0, 101.0], &[1000, 1000]).is_none());
        // Precio constante: nivel 1 del log-precio = 0; el tiempo avanza.
        let s = firma_ventana_logprecio(&[100.0, 100.0, 100.0], &[1000, 2000, 3000])
            .expect("ventana válida");
        assert!(s.level1[0].abs() < 1e-15);
        assert!(s.level1[1] > 0.0);
        assert!(s.level2[0][0].abs() < 1e-15);
    }
}
