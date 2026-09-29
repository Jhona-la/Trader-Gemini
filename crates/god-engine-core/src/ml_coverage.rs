//! DIAGNÓSTICO DE COBERTURA DE MODELOS DEL ROSTER (Ola XLVII·C).
//!
//! La medición de brecha contra la meta (docs/BRECHA_META_2026-09-29.md)
//! cuantificó el cuello de botella del volumen: los símbolos SIN modelo
//! promovido quedan en la sonda única de B3.25 (veto permanente tras la
//! primera operación). El circuito de desbloqueo existe y está cerrado a
//! nivel host —
//!
//!   sonda → evidencia (outcomes) → trainer FMT (gates honestos) →
//!   models/{SYM}_MOTOR.json → watcher hot-reload (10 s) →
//!   `load_global` → `has_roster_model` = true → B3.25 desbloqueado.
//!
//! — pero era INVISIBLE en operación: nadie reporta qué símbolos están
//! bloqueados por cobertura. Este módulo lo hace medible.
//!
//! Reglas de cobertura (mismas que el watcher del host):
//! - Clave del roster: `{SYM}_MOTOR` (U-7).
//! - Cargable: `{SYM}_MOTOR.json`, o `.bin` SIN hermano `.json` (el loader
//!   deriva la pareja y regenera la caché cuando el .json está presente).
//! - `_CANDIDATE` NO cuenta: un candidato no ha pasado el embudo de
//!   promoción y el roster no lo carga.

/// Cobertura de modelos del roster frente a un directorio `models/`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RosterCoverage {
    /// Símbolos con modelo promovido cargable ({SYM}_MOTOR.json|bin).
    pub covered: Vec<String>,
    /// Símbolos del roster sin modelo promovido — sonda única + veto B3.25.
    pub missing: Vec<String>,
}

impl RosterCoverage {
    /// Fracción cubierta [0,1] del roster.
    pub fn frac(&self) -> f64 {
        let total = self.covered.len() + self.missing.len();
        if total == 0 {
            return 1.0;
        }
        self.covered.len() as f64 / total as f64
    }

    /// Línea de telemetría de una sola línea, sin estados intermedios.
    pub fn telemetry_line(&self) -> String {
        format!(
            "📊 [ROSTER-COBERTURA] {}/{} símbolos con modelo promovido ({:.0}%) — sonda-bloqueados: {}",
            self.covered.len(),
            self.covered.len() + self.missing.len(),
            self.frac() * 100.0,
            self.missing.join(", ")
        )
    }
}

/// Escanea `models_dir` y clasifica el roster por clave `{SYM}_MOTOR`.
/// Los archivos `_CANDIDATE` y las claves que no están en el roster se
/// ignoran (la cobertura se mide DESDE el roster, no desde el disco).
pub fn roster_coverage<S: AsRef<str>>(
    roster: &[S],
    models_dir: &std::path::Path,
) -> RosterCoverage {
    // Un .json cargable POR CLAVE; un .bin sólo si no tiene hermano .json.
    let mut json_keys = std::collections::HashSet::new();
    let mut bin_only_keys = std::collections::HashSet::new();
    if let Ok(entries) = std::fs::read_dir(models_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if stem.ends_with("_CANDIDATE") {
                continue; // no promovido: el roster no lo carga
            }
            match path.extension().and_then(|e| e.to_str()) {
                Some("json") => {
                    json_keys.insert(stem.to_string());
                }
                Some("bin") => {
                    bin_only_keys.insert(stem.to_string());
                }
                _ => {}
            }
        }
    }
    let mut covered = Vec::new();
    let mut missing = Vec::new();
    for sym in roster {
        let key = format!("{}_MOTOR", sym.as_ref());
        let has = json_keys.contains(&key)
            || (bin_only_keys.contains(&key) && !json_keys.contains(&key));
        if has {
            covered.push(sym.as_ref().to_string());
        } else {
            missing.push(sym.as_ref().to_string());
        }
    }
    RosterCoverage { covered, missing }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn setup(dir: &std::path::Path, files: &[&str]) {
        fs::create_dir_all(dir).unwrap();
        for f in files {
            fs::write(dir.join(f), "{}").unwrap();
        }
    }

    #[test]
    fn clasifica_promovidos_candidatos_y_ausentes() {
        let tmp = std::env::temp_dir().join(format!("tg_cov_{:x}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        setup(&tmp, &[
            "LTCUSDT_MOTOR.json",        // promovido
            "ADAUSDT_MOTOR.bin",         // bin sin hermano json: cargable
            "LINKUSDT_MOTOR_CANDIDATE.json", // candidato: NO cuenta
            "OTRO.bin",                  // fuera del roster
        ]);
        let roster = ["LTCUSDT", "ADAUSDT", "LINKUSDT", "THETAUSDT"];
        let cov = roster_coverage(&roster, &tmp);
        assert_eq!(cov.covered, vec!["LTCUSDT", "ADAUSDT"]);
        assert_eq!(cov.missing, vec!["LINKUSDT", "THETAUSDT"]);
        assert!((cov.frac() - 0.5).abs() < 1e-12);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn bin_con_hermano_json_no_duplica() {
        let tmp = std::env::temp_dir().join(format!("tg_cov2_{:x}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        setup(&tmp, &["BTCUSDT_MOTOR.json", "BTCUSDT_MOTOR.bin"]);
        let cov = roster_coverage(&["BTCUSDT"], &tmp);
        assert_eq!(cov.covered.len(), 1);
        assert!(cov.missing.is_empty());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn directorio_ausente_es_cero_cobertura_sin_panico() {
        let cov = roster_coverage(
            &["LTCUSDT"],
            std::path::Path::new("/no/existe/models"),
        );
        assert!(cov.covered.is_empty());
        assert_eq!(cov.missing, vec!["LTCUSDT"]);
        assert_eq!(cov.frac(), 0.0);
        assert!(cov.telemetry_line().contains("0/1"));
    }

    #[test]
    fn linea_de_telemetria_menciona_los_bloqueados() {
        let cov = RosterCoverage {
            covered: vec!["BTCUSDT".into()],
            missing: vec!["LTCUSDT".into(), "ADAUSDT".into()],
        };
        let line = cov.telemetry_line();
        assert!(line.contains("1/3"));
        assert!(line.contains("LTCUSDT, ADAUSDT"));
    }
}
