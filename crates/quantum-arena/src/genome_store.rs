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

/// E3 — SEPARACIÓN DE ENTORNOS DEL ALMACÉN DE GENOMAS. Sin esto, un backtest
/// que termina promueve su overfit al MISMO active.json del que bootea
/// producción (contaminación bidireccional silenciosa).
///
/// Variable TG_GENOME_ENV: "backtest" | "demo" | "prod". Por defecto (ausente
/// o vacía) se conserva la ruta compartida histórica para no romper estados
/// existentes — los lanzadores DEBEN fijarla.
///
/// Promoción cruzada deliberada (p.ej. promover el campeón de backtest a
/// producción): exportar TG_GENOME_ENV=prod en el proceso promotor, o copiar
/// el envelope con `promote` desde el entorno destino — nunca implícitamente.
fn env_root() -> String {
    match std::env::var("TG_GENOME_ENV")
        .ok()
        .filter(|v| !v.trim().is_empty())
    {
        Some(env) => format!("config_dir/genomes/{}", env.trim().to_lowercase()),
        None => "config_dir/genomes".to_string(),
    }
}

fn active_path() -> String {
    format!("{}/active.json", env_root())
}

fn history_dir() -> String {
    format!("{}/history", env_root())
}

/// Espejo legacy: el loader viejo y el watcher leen esta ruta. Solo se
/// escribe/lee en el entorno compartido (sin TG_GENOME_ENV) para no cruzar
/// linajes entre entornos.
fn legacy_mirror() -> Option<String> {
    if std::env::var("TG_GENOME_ENV")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .is_some()
    {
        None
    } else {
        Some("config_dir/genotypes/active_genome.json".to_string())
    }
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

    /// Carga el genoma activo desde el envelope versionado con resolución resiliente multi-ruta:
    /// 1. config_dir/genomes/active.json (Envelope oficial versionado)
    /// 2. config_dir/genotypes/active_genome.json (Genoma activo legacy)
    /// 3. config_dir/genotypes/quantum_champion.json (Genoma campeón guardado)
    pub fn load_active() -> Option<GenomeEnvelope> {
        // 1. Intentar cargar desde el envelope oficial
        if let Ok(data) = std::fs::read_to_string(active_path()) {
            if let Ok(env) = serde_json::from_str::<GenomeEnvelope>(&data) {
                return Some(env);
            } else {
                // R-05: un genoma activo que existe pero NO parsea (p.ej.
                // schema viejo sin genes nuevos) es un evento crítico de
                // linaje — antes se descartaba EN SILENCIO y el sistema
                // reiniciaba en baseline sin que nadie lo supiera.
                eprintln!(
                    "🚨 [GENOME-STORE] {} EXISTE pero falla el parseo — se descarta y cae al fallback. Migra el genoma o regenéralo.",
                    active_path()
                );
            }
        }
        // T-09 / L-0 — MIGRACIÓN RESILIENTE Y HERENCIA DEL CAMPEÓN:
        // Si el entorno ({env}) no tiene active.json, buscar en orden de prioridad:
        // 1. config_dir/genomes/backtest/active.json (campeón de backtest)
        // 2. config_dir/genomes/active.json (linaje compartido)
        // 3. config_dir/genotypes/quantum_champion.json (campeón guardado)
        // 4. config_dir/genotypes/active_genome.json (legacy activo)
        let env_tag = std::env::var("TG_GENOME_ENV").unwrap_or_else(|_| "default".to_string());

        // 1. Intentar campeón de backtest si estamos en demo/prod
        if env_tag.trim().to_lowercase() != "backtest" {
            let bt_path = "config_dir/genomes/backtest/active.json";
            if let Ok(data) = std::fs::read_to_string(bt_path) {
                if let Ok(bt_env) = serde_json::from_str::<GenomeEnvelope>(&data) {
                    let sanitized = SuperGenotype::from_vector(&bt_env.genome.to_vector());
                    if let Ok(env) = Self::promote(
                        sanitized,
                        "backtest_heritage",
                        &format!(
                            "herencia automática del campeón de backtest al entorno {}",
                            env_tag.trim()
                        ),
                    ) {
                        eprintln!(
                            "🧬 [L-0] Campeón de backtest heredado exitosamente al entorno '{}' (generación {}).",
                            env_tag.trim(),
                            env.generation
                        );
                        return Some(env);
                    }
                }
            }
        }

        // 2. Intentar linaje compartido
        let shared = "config_dir/genomes/active.json";
        if let Ok(data) = std::fs::read_to_string(shared) {
            if let Ok(shared_env) = serde_json::from_str::<GenomeEnvelope>(&data) {
                let sanitized = SuperGenotype::from_vector(&shared_env.genome.to_vector());
                if let Ok(env) = Self::promote(
                    sanitized,
                    "shared_migration",
                    &format!(
                        "migración del linaje compartido al entorno {}",
                        env_tag.trim()
                    ),
                ) {
                    eprintln!(
                        "🧬 [L-0] Linaje de la era compartida migrado al entorno '{}' (generación {}).",
                        env_tag.trim(),
                        env.generation
                    );
                    return Some(env);
                }
            }
        }

        // 3. Fallback: quantum_champion.json
        let champ_path = "config_dir/genotypes/quantum_champion.json";
        if let Ok(data) = std::fs::read_to_string(champ_path) {
            if let Ok(g) = serde_json::from_str::<SuperGenotype>(&data) {
                let sanitized = SuperGenotype::from_vector(&g.to_vector());
                if let Ok(env) = Self::promote(
                    sanitized,
                    "champion_bootstrap",
                    &format!(
                        "bootstrap resiliente desde quantum_champion.json para {}",
                        env_tag.trim()
                    ),
                ) {
                    eprintln!(
                        "🧬 [L-0] Genoma campeón (quantum_champion.json) promovido al entorno '{}' (generación {}).",
                        env_tag.trim(),
                        env.generation
                    );
                    return Some(env);
                }
            }
        }

        // 4. Fallback: active_genome.json
        let legacy_path = "config_dir/genotypes/active_genome.json";
        if let Ok(data) = std::fs::read_to_string(legacy_path) {
            if let Ok(g) = serde_json::from_str::<SuperGenotype>(&data) {
                let sanitized = SuperGenotype::from_vector(&g.to_vector());
                if let Ok(env) = Self::promote(
                    sanitized,
                    "legacy_bootstrap",
                    &format!(
                        "bootstrap resiliente desde active_genome.json para {}",
                        env_tag.trim()
                    ),
                ) {
                    eprintln!(
                        "🧬 [L-0] Genoma activo legacy promovido al entorno '{}' (generación {}).",
                        env_tag.trim(),
                        env.generation
                    );
                    return Some(env);
                }
            }
        }

        None
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
        // R1.2 — INVARIANTE RR UNIFICADA: misma constante que mutate_cmaes
        // (SuperGenotype::MIN_RR_GATE), derivada de fees y del peor WR
        // tolerado — ver la documentación de la constante en genome.rs.
        // Antes convivían cuatro estándares (1.5x gate / 1.8x-2.0x mutación /
        // 2.2x-3.5x reparación).
        if genome.scalp_tp_base < genome.scalp_sl_base * SuperGenotype::MIN_RR_GATE {
            return Err(format!(
                "invariante RR violada: scalp_tp_base {} < {:.2} x scalp_sl_base {}",
                genome.scalp_tp_base,
                SuperGenotype::MIN_RR_GATE,
                genome.scalp_sl_base
            ));
        }
        if genome.swing_tp_base < genome.swing_sl_base * SuperGenotype::MIN_RR_GATE {
            return Err(format!(
                "invariante RR violada: swing_tp_base {} < {:.2} x swing_sl_base {}",
                genome.swing_tp_base,
                SuperGenotype::MIN_RR_GATE,
                genome.swing_sl_base
            ));
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
        // de las curvas — ya no reciben reparación de ancla ni clamp exacto:
        // valen lo que valgan las curvas clamped en sus propios genes
        // (140-143). Se verifica que sean finitos, dentro de bounds, y que la
        // invariante RR (ahora sobre CURVAS) se cumpla en las vistas.
        for i in 0..n {
            if (13..=16).contains(&i) {
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
            if (13..=16).contains(&i) {
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
