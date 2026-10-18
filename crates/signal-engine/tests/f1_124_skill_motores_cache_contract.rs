//! Contract test for F1-124 (Ola Ω156):
//!
//! Contract verifying that `SkillMotores::pesos(&self)` and `SkillMotores::diagnostico_banda(&self)`
//! are perfectly cached in O(1) time and strictly match the dynamically evaluated `peso(m, k)`.
//!
//! Hot-path depth events in `TensorVoteOrchestrator` consume `pesos()` on every tick.
//! Prior to F1-124, evaluating 480 pairs (15 motors × 32 scales) with sqrt/div on every tick
//! introduced significant CPU overhead. F1-124 caches `cached_pesos` and `cached_diagnostico`
//! with dirty-updates exclusively upon block maturity (`observar_maduracion*`).

use signal_engine::skill_motores::{SkillMotores, MOTORES, PISO_EXPLORACION};
use signal_engine::voto_espectral::{VotoEspectral, ESCALAS_VOTO};

#[test]
fn test_f1_124_skill_motores_cache_contract() {
    let mut skill = SkillMotores::new();

    // 1. In cold start, all weights must equal PISO_EXPLORACION (0.15)
    let pesos_inicial = skill.pesos();
    let diag_inicial = skill.diagnostico_banda();

    assert_eq!(
        diag_inicial.0, 0,
        "Cold start must report 0 outstanding motor-scale pairs"
    );
    assert!(
        (diag_inicial.1 - PISO_EXPLORACION).abs() < 1e-12,
        "Cold start max weight must equal PISO_EXPLORACION"
    );

    for m in 0..MOTORES {
        for k in 0..ESCALAS_VOTO {
            assert!(
                (pesos_inicial[m][k] - PISO_EXPLORACION).abs() < 1e-12,
                "Cold weight for motor {} scale {} must be PISO_EXPLORACION",
                m,
                k
            );
            assert!(
                (pesos_inicial[m][k] - skill.peso(m, k)).abs() < 1e-12,
                "Cached weight must identically match on-demand evaluation"
            );
        }
    }

    // 2. Simulate block maturity across multiple scales with returns and votes
    let dummy_votos: Vec<VotoEspectral> = (0..MOTORES)
        .map(|m| {
            let val = if m % 2 == 0 { 0.8 } else { -0.5 };
            VotoEspectral::desde_escalar(val)
        })
        .collect();

    // Observe maturity on scale 3 with return +0.02
    skill.observar_maduracion(3, 1000, 0.02, &dummy_votos);

    // Verify cache coherency after maturity
    let pesos_post = skill.pesos();
    for m in 0..MOTORES {
        for k in 0..ESCALAS_VOTO {
            let expected_peso = skill.peso(m, k);
            assert!(
                (pesos_post[m][k] - expected_peso).abs() < 1e-12,
                "Post-maturity cached weight for motor {} scale {} must match direct eval: cached={}, eval={}",
                m,
                k,
                pesos_post[m][k],
                expected_peso
            );
        }
    }

    // 3. Observe maturity without rearm on scale 5
    skill.observar_maduracion_sin_rearmar(5, 2000, -0.015);

    let pesos_post2 = skill.pesos();
    let diag_post2 = skill.diagnostico_banda();

    let mut manual_sobresalientes = 0usize;
    let mut manual_max = PISO_EXPLORACION;
    for m in 0..MOTORES {
        for k in 0..ESCALAS_VOTO {
            let p = skill.peso(m, k);
            assert!(
                (pesos_post2[m][k] - p).abs() < 1e-12,
                "Cached weight must remain consistent after observar_maduracion_sin_rearmar"
            );
            if p > PISO_EXPLORACION + 1e-9 {
                manual_sobresalientes += 1;
                manual_max = manual_max.max(p);
            }
        }
    }

    assert_eq!(
        diag_post2.0, manual_sobresalientes,
        "Diagnostico sobresalientes must match manual count"
    );
    assert!(
        (diag_post2.1 - manual_max).abs() < 1e-12,
        "Diagnostico max weight must match manual maximum"
    );
}
