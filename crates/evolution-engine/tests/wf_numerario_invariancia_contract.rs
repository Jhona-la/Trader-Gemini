//! GLM 145 — CERTIFICACIÓN de la invariancia del numeraire del pre-screen.
//!
//! El comentario de `WF_INITIAL_CAPITAL_DEFAULT` reclamaba que la simulación
//! es puramente multiplicativa y que el fitness (cociente en logaritmo) no
//! depende del capital inicial. Este contrato LO DEMUESTRA llamando a la
//! función REAL del daemon (`prescreen_evaluate`, extraída para este fin):
//! cualquier capital > 0 produce el MISMO fitness — el 13.0 no es un sesgo,
//! es un numeraire certificado, y `TG_WF_CAPITAL` permite re-certificar sin
//! recompilar (precedente TG_SA_* de F1-122).
use evolution_engine::online_daemon::{prescreen_evaluate, wf_initial_capital};

/// Serie determinista de retornos con momento alterno (entra en cada barra
/// con umbrales 0.5/0.5: prev_sigma>0 ⇒ +1, <0 ⇒ -1).
fn serie_momento(n: usize, amp: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            if i % 2 == 0 {
                amp
            } else {
                -amp * 0.6
            }
        })
        .collect()
}

fn evaluar_con(capital: f64, n_barras: usize) -> f64 {
    let series = vec![serie_momento(n_barras, 0.004), serie_momento(n_barras, 0.002)];
    prescreen_evaluate(
        0.02,  // tp
        0.01,  // sl
        0.20,  // kelly
        0.5,   // ml_thr_long (gate 0 ⇒ entra con cualquier sigma > 0)
        0.5,   // ml_thr_short
        &series,
        0.003, // volatility
        0.0004, // roundtrip_fee
        capital,
    )
}

/// INVARIANCIA DEL NUMERARIO: mismo fitness para cualquier capital > 0,
/// tanto en el régimen de pocas operaciones (rama bayesiana) como en el de
/// muchas (rama normal, ≥ WF_MIN_TRADES).
#[test]
fn el_fitness_del_prescreen_es_invariante_al_numerario() {
    for capital in [13.0, 100.0, 1_000.0, 1.0e6] {
        // Régimen de muchas operaciones (2 monedas × 40% OOS de 200 = 160 trades).
        let f_amplio = evaluar_con(capital, 200);
        let f_ref = evaluar_con(13.0, 200);
        assert!(
            (f_amplio - f_ref).abs() < 1e-9,
            "régimen amplio: fitness con capital {capital} ({f_amplio}) difiere del de 13.0 ({f_ref})"
        );
        // Régimen de pocas operaciones (rama bayesiana por < WF_MIN_TRADES).
        let f_corto = evaluar_con(capital, 20);
        let f_corto_ref = evaluar_con(13.0, 20);
        assert!(
            (f_corto - f_corto_ref).abs() < 1e-9,
            "régimen corto: fitness con capital {capital} ({f_corto}) difiere del de 13.0 ({f_corto_ref})"
        );
    }
    // El fitness no es trivialmente constante: series distintas puntúan distinto.
    let f_a = evaluar_con(13.0, 200);
    let f_b = prescreen_evaluate(
        0.02, 0.01, 0.20, 0.5, 0.5,
        &[serie_momento(200, 0.001)], 0.003, 0.0004, 13.0,
    );
    assert!((f_a - f_b).abs() > 1e-6, "sanity: series distintas ⇒ fitness distinto");
}

/// Override TG_WF_CAPITAL con guardas CL-15: válido se usa, inválido/ausente
/// cae al default documentado — jamás pánico, jamás 0 silencioso.
#[test]
fn override_entorno_tg_wf_capital_con_guardas() {
    struct EnvGuard(&'static str);
    impl EnvGuard {
        fn set(k: &'static str, v: &str) -> Self {
            // SAFETY: test de integración monohilo sobre una variable propia
            // del proyecto (TG_WF_CAPITAL); no hay otros lectores en el proceso.
            unsafe { std::env::set_var(k, v) };
            Self(k)
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: ver arriba; restaura el entorno al salir del scope.
            unsafe { std::env::remove_var(self.0) };
        }
    }

    let base = wf_initial_capital(); // sin env ⇒ default 1.0 (unidad neutra)
    assert!((base - 1.0).abs() < 1e-12, "sin env ⇒ default documentado");

    {
        let _g = EnvGuard::set("TG_WF_CAPITAL", "500.0");
        assert!((wf_initial_capital() - 500.0).abs() < 1e-12, "válido ⇒ se usa");
    }
    {
        let _g = EnvGuard::set("TG_WF_CAPITAL", "-5");
        assert!((wf_initial_capital() - 1.0).abs() < 1e-12, "≤0 ⇒ default");
    }
    {
        let _g = EnvGuard::set("TG_WF_CAPITAL", "no-un-numero");
        assert!((wf_initial_capital() - 1.0).abs() < 1e-12, "NaN ⇒ default");
    }
    assert!((wf_initial_capital() - 1.0).abs() < 1e-12, "limpio ⇒ default");
}
