//! ALMACÉN DE GENOMAS UNIFICADO (F4.3) — UNA sola verdad con linaje.
//!
//! Problema de auditoría: el genoma activo se escribía desde múltiples sitios
//! (evolver, polars_evolver) directamente a active_genome.json, sin versión,
//! sin historia, sin saber quién promovió qué ni por qué — imposible auditar
//! o revertir un genoma malo.
//!
//! Ahora TODO pasa por GenomeEnvelope::promote():
//!   config_dir/genomes/active.json            ← la verdad (envelope versionado)
//!   config_dir/genomes/history/gen_000042.json ← linaje completo inmutable
//!   config_dir/genotypes/active_genome.json    ← espejo legacy (compat hot-reload)
//!
//! Cada promoción registra: generación monotónica, fuente, padre, motivo con
//! métricas y timestamp. rollback() re-promociona una generación anterior.

use serde::{Deserialize, Serialize};
use std::io;

use crate::genome::SuperGenotype;

pub const SCHEMA_VERSION: u32 = 1;

/// E3 — SEPARACIÓN DE ENTORNOS DEL ALMACÉN DE GENOMAS (endurecida en D-651).
///
/// Variable TG_GENOME_ENV: "backtest" | "demo" | "prod". **Obligatoria.**
///
/// D-651 (DÉCIMA OLA) — LA BARRERA YA NO SE ATRAVIESA POR OMISIÓN. Antes:
///   - `env_root()` asumía `demo` cuando la variable no estaba definida, de
///     modo que cualquier evolver lanzado sin entorno escribía en demo; y
///   - `load_active()` heredaba AUTOMÁTICAMENTE backtest→demo→prod, de forma
///     que el campeón sobreajustado del backtest llegaba a operar capital
///     real sin intervención humana — exactamente lo que el docstring de
///     este módulo declaraba impedir.
/// Ahora el entorno es explícito o el proceso no arranca, y la ausencia de
/// `active.json` en el entorno es un ERROR, no una invitación a copiar.
///
/// Entornos reconocidos (lista cerrada: un typo ya no crea un silo nuevo).
pub const KNOWN_ENVS: [&str; 3] = ["backtest", "demo", "prod"];

/// Entorno activo. Devuelve `Err` si la variable falta o no es reconocida:
/// ningún camino del sistema puede inventar un entorno por defecto.
pub fn current_env() -> Result<String, String> {
    let raw = std::env::var("TG_GENOME_ENV").map_err(|_| {
        "TG_GENOME_ENV no está definida. El almacén de genomas exige entorno          EXPLÍCITO (backtest | demo | prod) — D-651: el default silencioso a          'demo' permitía que un evolver sin entorno contaminara la cadena de          herencia hacia producción."
            .to_string()
    })?;
    let env = raw.trim().to_lowercase();
    if !KNOWN_ENVS.contains(&env.as_str()) {
        return Err(format!(
            "TG_GENOME_ENV='{}' no es un entorno reconocido. Válidos: {:?}",
            raw, KNOWN_ENVS
        ));
    }
    Ok(env)
}

fn env_root() -> String {
    match current_env() {
        Ok(env) => format!("config_dir/genomes/{}", env),
        // Un caller que ignore el error de entorno no debe poder escribir en
        // NINGÚN silo real: se le da una ruta inválida que falla al abrir.
        Err(_) => "config_dir/genomes/__UNSET__".to_string(),
    }
}

pub fn active_json_path() -> String {
    active_path()
}

fn active_path() -> String {
    format!("{}/active.json", env_root())
}

fn history_dir() -> String {
    format!("{}/history", env_root())
}

/// Espejo legacy: el loader viejo lee esta ruta en arranques fríos.
/// X-001 (REHAB-2): se escribe SIEMPRE — es una VISTA de compatibilidad, no
/// un linaje. Antes solo en entorno compartido: con default demo, los
/// loaders legacy habrían quedado congelados en el último estado compartido.
fn legacy_mirror() -> Option<String> {
    Some("config_dir/genotypes/active_genome.json".to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomeEnvelope {
    pub schema_version: u32,
    /// Generación monotónica global — 1, 2, 3...
    pub generation: u64,
    pub created_ms: u64,
    /// Quién promovió: "ga_evolver" | "polars_evolver" | "online_daemon" | "manual" | ...
    pub source: String,
    /// Generación de la que desciende (0 = baseline).
    pub parent_generation: u64,
    /// POR QUÉ se promovió: métricas de validación del momento.
    pub promotion_reason: String,
    pub genome: SuperGenotype,
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

impl GenomeEnvelope {
    /// Aplica el genoma a la arena y registra atómicamente la generación aplicada (checkpointing de linaje)
    pub fn apply_to_arena(&self, arena: &crate::GlobalArena) {
        self.genome.apply_to_arena(arena);
        arena
            .applied_generation
            .store(self.generation, std::sync::atomic::Ordering::Release);
    }

    /// Carga el genoma activo del entorno ACTUAL. Sin herencia automática.
    ///
    /// D-651 (DÉCIMA OLA) — SE ELIMINARON LAS DOS RAMAS DE HERENCIA:
    ///   1. `prod` sin active.json heredaba de `demo` ("demo_heritage").
    ///   2. Todo entorno != backtest heredaba del campeón de backtest
    ///      ("backtest_heritage"), lo que incluye explícitamente a `prod`.
    /// La cadena resultante era backtest → demo → prod SIN intervención
    /// humana, sin validación de desempeño y sin confirmación: el genoma
    /// sobreajustado del backtest acababa operando capital real. Esto
    /// contradecía de forma directa el docstring del módulo, que declara que
    /// producción exige promoción EXPLÍCITA humana.
    ///
    /// Ahora: se carga el active.json del entorno, o NADA. Cruzar la frontera
    /// entre entornos requiere `promote_across_env()`, que exige armado
    /// humano vía TG_GENOME_PROMOTE_ARMED.
    pub fn load_active() -> Option<GenomeEnvelope> {
        let env = match current_env() {
            Ok(e) => e,
            Err(msg) => {
                eprintln!("🚨 [GENOME-STORE] {msg}");
                return None;
            }
        };
        match std::fs::read_to_string(active_path()) {
            Ok(data) => match serde_json::from_str::<GenomeEnvelope>(&data) {
                Ok(envelope) => Some(envelope),
                Err(e) => {
                    // R-05: un genoma que existe pero no parsea es un evento
                    // crítico de linaje. NO se cae a ningún fallback: caer
                    // silenciosamente a baseline (o peor, a otro entorno) es
                    // cómo se pierde la trazabilidad de qué está operando.
                    eprintln!(
                        "🚨 [GENOME-STORE] {} existe pero NO parsea ({e}).                          El entorno '{env}' queda SIN genoma activo — migra o                          regenera el linaje. No se hereda de otro entorno.",
                        active_path()
                    );
                    None
                }
            },
            Err(_) => {
                eprintln!(
                    "🚨 [GENOME-STORE] el entorno '{env}' no tiene {} — sin                      genoma activo. D-651: la herencia automática entre                      entornos fue eliminada; usa promote_across_env() con                      TG_GENOME_PROMOTE_ARMED=1 para cruzar la frontera.",
                    active_path()
                );
                None
            }
        }
    }

    /// Única vía para cruzar la frontera entre entornos. Exige armado humano
    /// explícito — misma filosofía que MAINNET_ARMED (D-651).
    ///
    /// `TG_GENOME_PROMOTE_ARMED` debe valer "1" y `TG_GENOME_PROMOTE_OPERATOR`
    /// debe identificar a quien autoriza: ambos quedan escritos en el motivo
    /// de promoción, de modo que el linaje registra QUIÉN cruzó la frontera.
    pub fn promote_across_env(
        from_env: &str,
        reason: &str,
    ) -> Result<GenomeEnvelope, String> {
        let to_env = current_env()?;
        let from = from_env.trim().to_lowercase();
        if !KNOWN_ENVS.contains(&from.as_str()) {
            return Err(format!("entorno origen '{from_env}' no reconocido"));
        }
        if from == to_env {
            return Err(format!("origen y destino son el mismo entorno ('{from}')"));
        }
        if std::env::var("TG_GENOME_PROMOTE_ARMED").unwrap_or_default().trim() != "1" {
            return Err(format!(
                "promoción {from} → {to_env} BLOQUEADA: exporta                  TG_GENOME_PROMOTE_ARMED=1 para autorizarla explícitamente."
            ));
        }
        let operator = std::env::var("TG_GENOME_PROMOTE_OPERATOR")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .ok_or_else(|| {
                "promoción BLOQUEADA: TG_GENOME_PROMOTE_OPERATOR debe                  identificar a quien autoriza el cruce de entorno."
                    .to_string()
            })?;

        let src = format!("config_dir/genomes/{from}/active.json");
        let data = std::fs::read_to_string(&src)
            .map_err(|e| format!("no se puede leer {src}: {e}"))?;
        let src_env: GenomeEnvelope = serde_json::from_str(&data)
            .map_err(|e| format!("{src} no parsea: {e}"))?;

        Self::promote(
            src_env.genome,
            &format!("cross_env:{from}->{to_env}"),
            &format!(
                "promoción manual {from} → {to_env} autorizada por '{operator}'                  (gen origen {}) — motivo: {reason}",
                src_env.generation
            ),
        )
        .map_err(|e| format!("promoción rechazada por el gate de validación: {e}"))
    }

    /// FASE 3 — Gate de validación pre-promoción. Rechaza genomas que no
    /// podrían operar: dimensionalidad rota, genes no finitos o fuera de los
    /// bounds evolutivos, o invariantes de riesgo violados (TP/SL asimétrico).
    /// Es un filtro de sanidad, no de desempeño: la promoción destruye el
    /// genoma activo en disco y ningún caller debe poder escribir basura.
    fn validate(genome: &SuperGenotype) -> Result<(), String> {
        let vec = genome.to_vector();
        if vec.len() != SuperGenotype::DIMENSION {
            return Err(format!(
                "dimensionalidad rota: to_vector dio {} genes, se esperaban {}",
                vec.len(),
                SuperGenotype::DIMENSION
            ));
        }
        let lower = SuperGenotype::get_lower_bounds();
        let upper = SuperGenotype::get_upper_bounds();
        for (i, &gene) in vec.iter().enumerate() {
            if !gene.is_finite() {
                return Err(format!("gen {} no finito ({})", i, gene));
            }
            if gene < lower[i] || gene > upper[i] {
                return Err(format!(
                    "gen {} fuera de bounds evolutivos: {} no está en [{}, {}]",
                    i, gene, lower[i], upper[i]
                ));
            }
        }
        // D-636 + D-608 (DÉCIMA OLA) — INVARIANTE RR SOBRE LA BANDA OPERABLE
        // COMPLETA, con el mínimo DEPENDIENTE DEL NIVEL DE SL.
        //
        // Antes: se comparaba contra la constante 1,5 (el equilibrio SIN
        // comisiones) y SOLO en las dos anclas legacy (30 s y 12 h), dejando
        // 14 de las 19 escalas del espectro sin protección alguna.
        //
        // Ahora: la banda operable se DERIVA de la fricción (ver
        // `SuperGenotype::tradeable_band_ms`) y, como `RR(τ) = TP(τ)/SL(τ)`
        // es monótona en `ln τ` —ambas curvas son log-lineales—, verificar
        // los DOS EXTREMOS de esa banda es NECESARIO Y SUFICIENTE para todas
        // las escalas contenidas en ella. Mismo coste, cobertura completa.
        let fee = SuperGenotype::REFERENCE_ROUNDTRIP_FEE;
        let (lo_tau, hi_tau) = genome.tradeable_band_ms(fee).ok_or_else(|| {
            format!(
                "genoma sin banda operable: su curva de SL nunca alcanza el                  mínimo viable {:.6} ({:.1} bps) impuesto por la fricción de                  {:.4}. Ninguna escala del espectro puede producir EV positivo.",
                SuperGenotype::min_viable_sl(fee),
                SuperGenotype::min_viable_sl(fee) * 1e4,
                fee
            )
        })?;
        for &tau in [lo_tau, hi_tau].iter() {
            let tp = genome.tp_horizon_curve.eval(tau);
            let sl = genome.sl_horizon_curve.eval(tau);
            if !tp.is_finite() || !sl.is_finite() || sl <= 0.0 {
                return Err(format!(
                    "curvas no finitas en tau={tau} ms: tp={tp}, sl={sl}"
                ));
            }
            let required =
                SuperGenotype::min_rr_for(SuperGenotype::WORST_TOLERATED_WR, fee, sl);
            if tp < sl * required {
                return Err(format!(
                    "invariante RR violada en tau={:.0} ms (banda operable                      {:.0}..{:.0} ms): TP {:.6} < {:.3} x SL {:.6} — EV negativo                      tras friccion {:.4}",
                    tau, lo_tau, hi_tau, tp, required, sl, fee
                ));
            }
        }
        Ok(())
    }

    /// Promueve un genoma: escribe historia inmutable + active atómico +
    /// espejo legacy. Único embudo de promoción del sistema.
    pub fn promote(
        genome: SuperGenotype,
        source: &str,
        reason: &str,
    ) -> io::Result<GenomeEnvelope> {
        // FASE 3: todo genoma pasa por el gate ANTES de tocar disco. Un error
        // aquí es promoción rechazada, nunca promoción parcial.
        if let Err(violation) = Self::validate(&genome) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "[GENOME-GATE] promoción de '{}' rechazada: {}",
                    source, violation
                ),
            ));
        }
        let parent = if let Ok(data) = std::fs::read_to_string(active_path()) {
            serde_json::from_str::<GenomeEnvelope>(&data)
                .map(|e| e.generation)
                .unwrap_or(0)
        } else {
            0
        };
        let envelope = GenomeEnvelope {
            schema_version: SCHEMA_VERSION,
            generation: parent + 1,
            created_ms: now_ms(),
            source: source.to_string(),
            parent_generation: parent,
            promotion_reason: reason.to_string(),
            genome,
        };

        std::fs::create_dir_all(history_dir())?;

        // 1) Historia inmutable (append-only por nombre de generación).
        let hist_path = format!("{}/gen_{:06}.json", history_dir(), envelope.generation);
        let hist_json = serde_json::to_string_pretty(&envelope)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        atomic_write(&hist_path, &hist_json)?;

        // 2) Active atómico (tmp + rename — jamás un active a medias).
        let active_json = serde_json::to_string_pretty(&envelope)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        atomic_write(&active_path(), &active_json)?;

        // 3) Espejo legacy: solo el genoma crudo (compat con loaders/watchers viejos).
        let legacy_json = serde_json::to_string_pretty(&envelope.genome)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        {
            // Espejo legacy solo en entorno compartido (ver legacy_mirror).
            if let Some(mirror) = legacy_mirror() {
                atomic_write(&mirror, &legacy_json)?;
            }
        }

        Ok(envelope)
    }

    /// Rollback: re-promociona la generación `target` como nueva generación
    /// (la historia es append-only; revertir también queda auditado).
    pub fn rollback(target_generation: u64) -> io::Result<GenomeEnvelope> {
        let hist_path = format!("{}/gen_{:06}.json", history_dir(), target_generation);
        let data = std::fs::read_to_string(&hist_path)?;
        let previous: GenomeEnvelope = serde_json::from_str(&data)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Self::promote(
            previous.genome,
            "rollback",
            &format!(
                "rollback a generación {} (fuente original: {})",
                target_generation, previous.source
            ),
        )
    }

    /// Últimas N generaciones para inspección/telemetría.
    pub fn recent_history(n: usize) -> Vec<(u64, String, String)> {
        let mut out = Vec::new();
        let active_gen = if let Ok(data) = std::fs::read_to_string(active_path()) {
            serde_json::from_str::<GenomeEnvelope>(&data)
                .map(|e| e.generation)
                .unwrap_or(0)
        } else {
            0
        };
        let mut g = active_gen;
        while g > 0 && out.len() < n {
            let hist_path = format!("{}/gen_{:06}.json", history_dir(), g);
            if let Ok(data) = std::fs::read_to_string(&hist_path) {
                if let Ok(e) = serde_json::from_str::<GenomeEnvelope>(&data) {
                    out.push((e.generation, e.source, e.promotion_reason));
                }
            }
            g -= 1;
        }
        out
    }
}

fn atomic_write(path: &str, contents: &str) -> io::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let tmp = format!("{}.tmp", path);
    std::fs::write(&tmp, contents)?;
    if std::path::Path::new(path).exists() {
        let _ = std::fs::remove_file(path);
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    /// T-2 (DÉCIMA OLA) — SIMETRÍA ENTRE ARRANQUE EN FRÍO Y HOT-SWAP (D-650).
    ///
    /// El arena se puede poblar por dos caminos: `QuantumConfig::from_genome()`
    /// en el arranque y `SuperGenotype::apply_to_arena()` en cada promoción
    /// evolutiva. Si divergen —y divergían en 12 genes—, el organismo vivo tras
    /// un hot-swap es una quimera que ninguna aptitud evaluó: N genes del
    /// genoma nuevo y M del de arranque.
    ///
    /// Este test compara gen a gen, mediante `current_from_arena`, el estado
    /// que produce cada camino. Es el contrato que impide que la exhaustividad
    /// de `apply_to_arena` vuelva a depender de que alguien se acuerde.
    /// D-649b: el gen zombie entra acotado a [4 h, 8 h]; el valor del genoma de
    /// producción (35 min) queda en el suelo, que reproduce la escala previa.
    #[test]
    fn d649b_zombie_se_acota_a_la_banda_de_diseno() {
        use std::sync::atomic::Ordering;
        let mut g = SuperGenotype::new_baseline(0.0002, 0.0005);
        assert_eq!(g.to_vector()[SuperGenotype::SLOT_ZOMBIE_TIMEOUT], g.zombie_timeout_ms);
        g.zombie_timeout_ms = 2_100_140.98;
        let arena = crate::state::GlobalArena::from_genome(13.0, &g);
        assert_eq!(arena.config.zombie_timeout_ms.load(Ordering::Relaxed), 14_400_000.0);
    }

    /// D-680: el prior del win rate es el de diseño y una operación no lo destruye.
    #[test]
    fn d680_prior_del_win_rate_y_media_posterior() {
        use std::sync::atomic::Ordering;
        let g = SuperGenotype::new_baseline(0.0002, 0.0005);
        let arena = crate::state::GlobalArena::from_genome(13.0, &g);
        let w0 = arena.coins[0].metrics.win_rate.load(Ordering::Relaxed);
        assert_eq!(w0, SuperGenotype::WORST_TOLERATED_WR);
        let tras_perdida = SuperGenotype::posterior_win_rate(w0, 0.0, false);
        assert!(
            tras_perdida > 0.25 && tras_perdida < w0,
            "una pérdida no puede llevarlo a 0: {tras_perdida}"
        );
        let mut w = w0;
        for i in 0..10_000u32 {
            w = SuperGenotype::posterior_win_rate(w, i as f64, i % 4 == 0);
        }
        assert!((w - 0.25).abs() < 0.01, "debe converger a la frecuencia observada, dio {w}");
    }

    /// D-683: un genoma cargado desde disco sin curvas continuas (serde las
    /// rellena con literales) entra al arena con las curvas DERIVADAS de sus
    /// genes, igual que el que sale de `from_vector`. Arranque en frío y
    /// hot-swap coinciden.
    #[test]
    fn d683_curvas_continuas_se_derivan_de_los_genes_al_entrar_al_arena() {
        use crate::temporal_spectrum::{HorizonCurve, TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS};
        let mut g = SuperGenotype::new_baseline(0.0002, 0.0005);
        g.scalp_obi_threshold = 0.45;
        g.swing_obi_threshold = 0.30;
        g.scalp_kelly_fraction = 0.157;
        g.swing_kelly_fraction = 0.112;
        g.scalp_trail_atr_mult_base = 2.2;
        g.swing_trail_atr_mult_base = 4.1;
        // Lo que serde deja en un genoma serializado sin curvas.
        g.obi_horizon_curve =
            HorizonCurve::through_two_points(TAU_ANCHOR_FAST_MS, 0.25, TAU_ANCHOR_SLOW_MS, 0.40);
        g.kelly_horizon_curve =
            HorizonCurve::through_two_points(TAU_ANCHOR_FAST_MS, 0.20, TAU_ANCHOR_SLOW_MS, 0.15);
        g.trail_mult_horizon_curve =
            HorizonCurve::through_two_points(TAU_ANCHOR_FAST_MS, 2.5, TAU_ANCHOR_SLOW_MS, 3.5);

        let frio = crate::state::GlobalArena::from_genome(13.0, &g);
        let caliente = crate::state::GlobalArena::new(13.0);
        g.apply_to_arena(&caliente);
        for (nombre, arena) in [("frío", &frio), ("hot-swap", &caliente)] {
            let c = &arena.config;
            assert!((c.obi_threshold_at_tau(TAU_ANCHOR_FAST_MS) - 0.45).abs() < 1e-9, "{nombre}: OBI rápido");
            assert!((c.obi_threshold_at_tau(TAU_ANCHOR_SLOW_MS) - 0.30).abs() < 1e-9, "{nombre}: OBI lento");
            assert!((c.kelly_at_tau(TAU_ANCHOR_SLOW_MS) - 0.112).abs() < 1e-9, "{nombre}: Kelly lento");
            assert!((c.trail_params_at_tau(TAU_ANCHOR_FAST_MS).0 - 2.2).abs() < 1e-9, "{nombre}: trailing rápido");
        }
    }

    #[test]
    fn t2_simetria_from_genome_vs_apply_to_arena() {
        use crate::GlobalArena;

        // Genoma de prueba con valores distinguibles del baseline, para que un
        // gen no refrescado se note.
        let mut g = SuperGenotype::new_baseline(0.0002, 0.0005);
        g = g.mutate_cmaes_seeded(0.45, 20260910);
        let g = SuperGenotype::from_vector(&g.to_vector());

        // Camino A: arranque en frío.
        let arena_frio = GlobalArena::from_genome(13.0, &g);
        let visto_frio = SuperGenotype::current_from_arena(&arena_frio);

        // Camino B: arena con OTRO genoma y luego hot-swap al nuestro.
        let otro = SuperGenotype::new_baseline(0.0004, 0.0009);
        let arena_swap = GlobalArena::from_genome(13.0, &otro);
        g.apply_to_arena(&arena_swap);
        let visto_swap = SuperGenotype::current_from_arena(&arena_swap);

        let a = visto_frio.to_vector();
        let b = visto_swap.to_vector();
        assert_eq!(a.len(), b.len(), "dimensionalidad divergente");

        let mut divergentes: Vec<(usize, f64, f64)> = Vec::new();
        for i in 0..a.len() {
            let (x, y) = (a[i], b[i]);
            let tol = 1e-9 * x.abs().max(1.0);
            if !(x - y).abs().le(&tol) {
                divergentes.push((i, x, y));
            }
        }
        assert!(
            divergentes.is_empty(),
            "hot-swap NO refresca {} gen(es): {:?}. Un gen que se inicializa en              frío pero no se refresca deja el arena con una mezcla de dos              genomas tras cada promoción (D-650).",
            divergentes.len(),
            divergentes
        );
    }

    use super::*;

    #[test]
    fn envelope_roundtrip_schema() {
        let env = GenomeEnvelope {
            schema_version: SCHEMA_VERSION,
            generation: 7,
            created_ms: 1_700_000_000_000,
            source: "ga_evolver".into(),
            parent_generation: 6,
            promotion_reason: "val_bce 0.543→0.521, oos +2.1%".into(),
            genome: SuperGenotype::new_baseline(0.0002, 0.0005),
        };
        let json = serde_json::to_string(&env).expect("serialize");
        let back: GenomeEnvelope = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.generation, 7);
        assert_eq!(back.source, "ga_evolver");
        assert_eq!(back.schema_version, SCHEMA_VERSION);
        assert_eq!(back.parent_generation, 6);
        assert!(back.promotion_reason.contains("val_bce"));
    }

    #[test]
    fn test_r11_evolution_pipeline_never_blocked_by_gate() {
        // R1.1 — certificación extremo a extremo: la cadena completa de la
        // evolución versionada (mutate -> to_vector -> from_vector -> gate)
        // debe tener éxito de forma sistemática. Antes de R1.1, los clamps
        // inline de from_vector eran disjuntos de los bounds de validate()
        // en 4 genes de TP/SL y TODA mutación CMA-ES era rechazada.
        let base = SuperGenotype::new_baseline(0.0002, 0.0005);
        let mut rejected = 0usize;
        for i in 0..1000 {
            let rate = 0.05 + (i % 10) as f64 * 0.05; // 0.05..0.50
            let mutant = base.mutate_cmaes(rate);
            let roundtrip = SuperGenotype::from_vector(&mutant.to_vector());
            if GenomeEnvelope::validate(&roundtrip).is_err() {
                rejected += 1;
            }
        }
        assert_eq!(
            rejected, 0,
            "ningún mutante round-trip debe ser rechazado por el gate de promoción"
        );
    }

    #[test]
    fn test_r11_bounds_are_structurally_sound() {
        // R1.1 — invariantes de la fuente única de verdad, agnósticos al
        // calibrado fino de cada gen (que evoluciona con el desarrollo):
        // (a) todo bound inferior es estrictamente menor que su superior;
        // (b) from_vector clampa activamente contra los bounds (extremos
        //     del vector aplanan exactamente en las cotas);
        // (c) la caja de TP/SL admite genomas que cumplen la invariante RR
        //     >= 1.5 exigida por el gate (hi_tp >= 1.5 * lo_sl).
        let lo = SuperGenotype::get_lower_bounds();
        let hi = SuperGenotype::get_upper_bounds();
        for i in 0..SuperGenotype::DIMENSION {
            assert!(
                lo[i] < hi[i],
                "bounds degenerados en gen {}: {} >= {}",
                i,
                lo[i],
                hi[i]
            );
        }
        let n = SuperGenotype::DIMENSION;
        let floor_g = SuperGenotype::from_vector(&vec![f64::NEG_INFINITY; n]);
        let ceil_g = SuperGenotype::from_vector(&vec![f64::INFINITY; n]);
        let v_floor = floor_g.to_vector();
        let v_ceil = ceil_g.to_vector();
        // N-02: los genes de TP (13, 15) tienen un floor efectivo mayor que
        // X-004 (REHAB-1): los genes 13-16 (anclas tp/sl) son VISTAS derivadas
        // de las curvas — ya no reciben reparación de ancla ni clamp exacto.
        //
        // D-636/D-608 (DÉCIMA OLA): los genes 140-143 (coeficientes de las
        // curvas) pasan a la MISMA categoría. Antes el contrato era «clampan
        // exactamente a su bound», que solo se sostenía porque la invariante
        // RR se verificaba en dos anclas fijas y casi nunca se activaba. Con
        // el RR correcto —dependiente del nivel de SL y verificado en toda la
        // banda operable— `enforce_curve_rr()` SÍ reposiciona los coeficientes
        // cuando la fricción lo exige. Ese es el comportamiento buscado: un
        // genoma en el borde de sus bandas evolutivas debe salir del gate con
        // EV no negativo, aunque para ello deba moverse dentro de bounds.
        //
        // Contrato vigente para 13-16 y 140-143: finitos, dentro de bounds y
        // con la invariante RR satisfecha. Contrato para el resto: clamp exacto.
        let derived = |i: usize| (13..=16).contains(&i) || (140..=143).contains(&i);
        for i in 0..n {
            if derived(i) {
                assert!(
                    v_floor[i].is_finite() && v_floor[i] >= lo[i] && v_floor[i] <= hi[i],
                    "vista {} fuera de bounds en floor: {}",
                    i,
                    v_floor[i]
                );
            } else {
                assert!(
                    (v_floor[i] - lo[i]).abs() < 1e-9,
                    "gen {} no clampa al floor esperado",
                    i
                );
            }
            if derived(i) {
                assert!(
                    v_ceil[i].is_finite() && v_ceil[i] >= lo[i] && v_ceil[i] <= hi[i],
                    "vista {} fuera de bounds en ceil: {}",
                    i,
                    v_ceil[i]
                );
            } else {
                assert!(
                    (v_ceil[i] - hi[i]).abs() < 1e-12,
                    "gen {} no clampa al techo",
                    i
                );
            }
        }
        // La invariante RR se preserva incluso en los extremos del box
        // (las vistas la heredan de las curvas reparadas).
        assert!(v_floor[13] >= v_floor[14] * SuperGenotype::MIN_RR_GATE);
        assert!(v_floor[15] >= v_floor[16] * SuperGenotype::MIN_RR_GATE);
        // Caja RR factible (indices: 13=scalp_tp, 14=scalp_sl, 15=swing_tp, 16=swing_sl)
        // N-06 — el baseline génesis debe ser ESTABLE bajo round-trip:
        // cualquier gen fuera de bounds se reescribe silenciosamente x2-x100.
        let baseline = SuperGenotype::new_baseline(0.0002, 0.0005);
        let v1 = baseline.to_vector();
        let v2 = SuperGenotype::from_vector(&v1).to_vector();
        for i in 0..SuperGenotype::DIMENSION {
            assert!(
                (v1[i] - v2[i]).abs() <= v1[i].abs() * 1e-9,
                "gen {} inestable bajo round-trip: {} -> {}",
                i,
                v1[i],
                v2[i]
            );
        }
        assert!(hi[13] >= 1.5 * lo[14], "caja scalp RR infactible");
        assert!(hi[15] >= 1.5 * lo[16], "caja swing RR infactible");
    }

    #[test]
    fn test_n06_baseline_passes_promotion_gate() {
        let base = SuperGenotype::new_baseline(0.0002, 0.0005);
        let validation = GenomeEnvelope::validate(&base);
        assert!(
            validation.is_ok(),
            "new_baseline debe pasar el gate de promoción: {:?}",
            validation.err()
        );
        let roundtrip = SuperGenotype::from_vector(&base.to_vector());
        let roundtrip_val = GenomeEnvelope::validate(&roundtrip);
        assert!(
            roundtrip_val.is_ok(),
            "roundtrip de baseline debe pasar el gate: {:?}",
            roundtrip_val.err()
        );
    }

    #[test]
    fn test_quantum_champion_passes_validation() {
        let path = "../../config_dir/genotypes/quantum_champion.json";
        let alt_path = "config_dir/genotypes/quantum_champion.json";
        let data = std::fs::read_to_string(path)
            .or_else(|_| std::fs::read_to_string(alt_path))
            .expect("quantum_champion.json debe existir");
        let g: SuperGenotype = serde_json::from_str(&data)
            .expect("quantum_champion.json debe deserializarse como SuperGenotype");
        let sanitized = SuperGenotype::from_vector(&g.to_vector());
        let validation = GenomeEnvelope::validate(&sanitized);
        assert!(
            validation.is_ok(),
            "quantum_champion.json debe pasar el gate de promoción: {:?}",
            validation.err()
        );
    }
}
