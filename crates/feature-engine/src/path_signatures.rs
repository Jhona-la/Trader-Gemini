//! FIRMAS DE CAMINO TRUNCADAS (Lyons) — NIVEL 2 (Ola XLVIII·B).
//!
//! Transferencia de la teoría de rough paths (Lyons): la firma de un camino
//! es su "huella algebraica" — una representación universal de la
//! trayectoria. Aquí sólo se retienen dos niveles: un resumen NO inyectivo,
//! sin garantía de poder predictivo. Las identidades de Chen y del reverso
//! permiten auditar la fórmula (con tolerancia de redondeo en f64).
//! Subdividir segmentos del MISMO camino lineal por partes conserva su
//! firma; remuestrear una trayectoria distinta puede perder excursiones.
//!
//! Contrato (docs/TRIAGE_TEORICO_2026-09-29.md):
//! - **Variable**: camino 2D X = (log-P(t), (t-t0)/(tN-t0)); interpolación
//!   lineal por partes sobre una escala τ a elección del consumidor.
//!   El consumidor debe conservar τ y la frescura por separado: normalizar
//!   el tiempo elimina la duración absoluta, no identifica todos los horizontes.
//! - **Operador**: firma truncada a nivel 2 — 6 términos significativos:
//!   S¹, S² (incrementos totales) y
//!   S^{ij} = Σ_{a<b} Δ^i_a Δ^j_b + ½ Σ_a Δ^i_a Δ^j_a.
//!   No son seis grados de libertad: S^{ij}+S^{ji}=S^i S^j.
//! - **Unidades**: adimensionales (log-acumulado × fracción de ventana).
//! - **Contorno**: <2 puntos ⇒ None; precio ≤0/NaN ⇒ None (el log exige
//!   positividad); timestamps no decrecientes y duración positiva.
//! - **Identificabilidad**: Hambly-Lyons trata la firma COMPLETA de caminos
//!   de variación acotada, no esta truncación. Caminos diferentes, incluso
//!   aumentados con tiempo estrictamente creciente, pueden compartir nivel 2.
//!   Fuentes: https://arxiv.org/abs/math/0507536 y https://arxiv.org/abs/1603.03788.
//! - **Coste**: O(n) por ventana, fuera del hot path.
//! - **Falsación** (tests): línea recta en densidades distintas ⇒ firma
//!   idéntica (invariancia a remuestreo del caso exacto); identidad de
//!   Chen S(A*B) = S(A)⊗S(B); reverso: S¹→−S¹, S^{ij}→S^{ji}.

/// Firma truncada a nivel 2 de un camino 2D.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Signature2 {
    /// Nivel 1: incrementos totales por coordenada.
    pub level1: [f64; 2],
    /// Nivel 2: S^{ij} = Σ_{a<b} Δ^i_a Δ^j_b + ½ Σ_a Δ^i_a Δ^j_a.
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
/// Firma geométrica del camino lineal por partes: S^{ij} = Σ_{a<b} δ^i_a δ^j_b
/// + ½ Σ_a δ^i_a δ^j_a. El término de diagonal media hace que la firma del
/// camino recto sea exacta en aritmética real a cualquier subdivisión (S^{ij} =
/// x_i·x_j/2) y conserva Chen y reverso salvo redondeo numérico (ver tests).
/// Sin ella, la suma discreta de
/// pares ordenados arrastra una corrección O(δ) dependiente del muestreo.
///
/// API algebraica SIN validación: el llamador debe asegurar incrementos e
/// intermedios representables y finitos. No recorta overflow/NaN ni certifica
/// una interpretación estocástica de Stratonovich para cualquier feed.
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

/// Firma de la ventana: camino X = (log P, (t-t0)/(tN-t0)).
/// None si longitudes distintas, <2 puntos, precios no positivos/no finitos,
/// reloj decreciente, duración nula o resultado no finito. Timestamps iguales
/// conservan el orden del feed; no se inventan tiempos ni se ordenan precios.
/// Devuelve seis coordenadas adimensionales, NO duración ni edad del dato.
pub fn firma_ventana_logprecio(precios: &[f64], ts_ms: &[u64]) -> Option<Signature2> {
    if precios.len() != ts_ms.len() || precios.len() < 2 {
        return None;
    }
    // Restar en el dominio entero ANTES de convertir evita perder un delta
    // pequeño por redondeo del epoch absoluto (por ejemplo 2^53 y 2^53+1).
    let span = ts_ms[ts_ms.len() - 1].checked_sub(ts_ms[0])?;
    if span == 0 || !precios[0].is_finite() || precios[0] <= 0.0 {
        return None;
    }
    let mut incs: Vec<(f64, f64)> = Vec::with_capacity(precios.len() - 1);
    let mut p_prev = precios[0];
    let mut t_prev = ts_ms[0];
    for (&p, &t) in precios.iter().zip(ts_ms).skip(1) {
        if !p.is_finite() || p <= 0.0 {
            return None;
        }
        let dt = t.checked_sub(t_prev)?;
        // ln(1+r) mantiene movimientos pequeños que ln(p)-ln(p_prev) cancela.
        // Cerca (factor 2), la resta satisface la condición de Sterbenz;
        // lejos, restar logaritmos evita r redondeado cerca de -1. La frontera
        // es numérica, no un filtro económico ni un recorte de rendimientos.
        let log_return = if p >= 0.5 * p_prev && 0.5 * p <= p_prev {
            ((p - p_prev) / p_prev).ln_1p()
        } else {
            p.ln() - p_prev.ln()
        };
        incs.push((log_return, dt as f64 / span as f64));
        p_prev = p;
        t_prev = t;
    }
    let signature = firma_nivel2_incrementos(&incs);
    signature
        .to_features()
        .into_iter()
        .all(f64::is_finite)
        .then_some(signature)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FALSACIÓN (a) — LÍNEA RECTA, forma cerrada: camino con n incrementos
    /// iguales (a,b): S¹ = n·a, S² = n·b, S^{ij} = (n·a^i)(n·a^j)/2 — igual
    /// para CUALQUIER n: la firma del caso exacto no depende del muestreo.
    #[test]
    fn linea_recta_forma_cerrada_e_invariante_al_remuestreo() {
        let firma = |n: usize, a: f64, b: f64| firma_nivel2_incrementos(&vec![(a, b); n]);
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
