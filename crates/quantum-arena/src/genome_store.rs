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
const ACTIVE_PATH: &str = "config_dir/genomes/active.json";
const HISTORY_DIR: &str = "config_dir/genomes/history";
/// Espejo legacy: el loader viejo y el watcher leen esta ruta.
const LEGACY_MIRROR: &str = "config_dir/genotypes/active_genome.json";

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
    /// Carga el genoma activo desde el envelope versionado con resolución resiliente multi-ruta:
    /// 1. config_dir/genomes/active.json (Envelope oficial versionado)
    /// 2. config_dir/genotypes/active_genome.json (Genoma activo legacy)
    /// 3. config_dir/genotypes/quantum_champion.json (Genoma campeón guardado)
    pub fn load_active() -> Option<GenomeEnvelope> {
        // 1. Intentar cargar desde el envelope oficial
        if let Ok(data) = std::fs::read_to_string(ACTIVE_PATH) {
            if let Ok(env) = serde_json::from_str::<GenomeEnvelope>(&data) {
                return Some(env);
            }
        }
        // 2. Fallback resiliente: cargar genoma raw de LEGACY_MIRROR
        if let Ok(data) = std::fs::read_to_string(LEGACY_MIRROR) {
            if let Ok(g) = serde_json::from_str::<SuperGenotype>(&data) {
                if let Ok(env) = Self::promote(g, "legacy_bootstrap", "Migración automática desde active_genome.json") {
                    return Some(env);
                }
            }
        }
        // 3. Fallback resiliente: quantum_champion.json
        let champ_path = "config_dir/genotypes/quantum_champion.json";
        if let Ok(data) = std::fs::read_to_string(champ_path) {
            if let Ok(g) = serde_json::from_str::<SuperGenotype>(&data) {
                if let Ok(env) = Self::promote(g, "champion_bootstrap", "Migración automática desde quantum_champion.json") {
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
        // INVARIANTE DE RIESGO (mismo axioma que mutate_cmaes): asimetría
        // ganadora obligatoria — un genoma con SL >= TP matemáticamente
        // pierde ante fees.
        if genome.scalp_tp_base < genome.scalp_sl_base * 1.5 {
            return Err(format!(
                "invariante RR violada: scalp_tp_base {} < 1.5 x scalp_sl_base {}",
                genome.scalp_tp_base, genome.scalp_sl_base
            ));
        }
        if genome.swing_tp_base < genome.swing_sl_base * 1.5 {
            return Err(format!(
                "invariante RR violada: swing_tp_base {} < 1.5 x swing_sl_base {}",
                genome.swing_tp_base, genome.swing_sl_base
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
                format!("[GENOME-GATE] promoción de '{}' rechazada: {}", source, violation),
            ));
        }
        let parent = if let Ok(data) = std::fs::read_to_string(ACTIVE_PATH) {
            serde_json::from_str::<GenomeEnvelope>(&data).map(|e| e.generation).unwrap_or(0)
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

        std::fs::create_dir_all(HISTORY_DIR)?;

        // 1) Historia inmutable (append-only por nombre de generación).
        let hist_path = format!("{}/gen_{:06}.json", HISTORY_DIR, envelope.generation);
        let hist_json = serde_json::to_string_pretty(&envelope)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        atomic_write(&hist_path, &hist_json)?;

        // 2) Active atómico (tmp + rename — jamás un active a medias).
        let active_json = serde_json::to_string_pretty(&envelope)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        atomic_write(ACTIVE_PATH, &active_json)?;

        // 3) Espejo legacy: solo el genoma crudo (compat con loaders/watchers viejos).
        let legacy_json = serde_json::to_string_pretty(&envelope.genome)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        atomic_write(LEGACY_MIRROR, &legacy_json)?;

        Ok(envelope)
    }

    /// Rollback: re-promociona la generación `target` como nueva generación
    /// (la historia es append-only; revertir también queda auditado).
    pub fn rollback(target_generation: u64) -> io::Result<GenomeEnvelope> {
        let hist_path = format!("{}/gen_{:06}.json", HISTORY_DIR, target_generation);
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
        let active_gen = if let Ok(data) = std::fs::read_to_string(ACTIVE_PATH) {
            serde_json::from_str::<GenomeEnvelope>(&data).map(|e| e.generation).unwrap_or(0)
        } else {
            0
        };
        let mut g = active_gen;
        while g > 0 && out.len() < n {
            let hist_path = format!("{}/gen_{:06}.json", HISTORY_DIR, g);
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
}
