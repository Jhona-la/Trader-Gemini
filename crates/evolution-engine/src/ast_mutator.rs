use serde_json::Value;
use std::fs;
use std::path::Path;

/// Demonio Autoevolutivo (AST-Modifier)
/// Este mutador tiene permisos para reescribir lógica en .rs y archivos .json
pub struct ASTMutator;

impl ASTMutator {
    pub fn new() -> Self {
        Self
    }

    /// Mutates a JSON configuration file dynamically.
    pub fn mutate_json_config<P: AsRef<Path>>(
        &self,
        path: P,
        key: &str,
        new_value: Value,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.as_ref();
        if !path.exists() {
            return Err("Configuration file does not exist".into());
        }

        let content = fs::read_to_string(path)?;
        let mut json: Value = serde_json::from_str(&content)?;

        // Simple nested key updater (e.g., "Risk.MAX_DRAWDOWN")
        let mut current = &mut json;
        let keys: Vec<&str> = key.split('.').collect();
        for (i, k) in keys.iter().enumerate() {
            if i == keys.len() - 1 {
                if let Some(obj) = current.as_object_mut() {
                    obj.insert(k.to_string(), new_value.clone());
                }
            } else {
                current = current.get_mut(*k).ok_or("Key path not found")?;
            }
        }

        let new_content = serde_json::to_string_pretty(&json)?;
        let tmp_path = format!("{}.tmp", path.display());
        fs::write(&tmp_path, new_content)?;
        let _ = fs::remove_file(path);
        fs::rename(&tmp_path, path)?;
        println!(
            "🧬 [AST Mutator] Evolved {} at {} to {}",
            path.display(),
            key,
            new_value
        );
        Ok(())
    }

    /// Mutates a hardcoded constant in a Rust file.
    /// Uses regex to find `const CONST_NAME: Type = Value;` and updates the Value.
    pub fn mutate_rs_constant<P: AsRef<Path>>(
        &self,
        path: P,
        const_name: &str,
        new_value: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let content = fs::read_to_string(path)?;

        let pattern = format!(
            r"(?m)(^\s*pub\s*const\s*{}\s*:\s*[a-zA-Z0-9_]+\s*=\s*)([^;]+)(;)",
            const_name
        );
        let regex = regex::Regex::new(&pattern)?;

        if regex.is_match(&content) {
            let replaced = regex.replace(&content, format!("${{1}}{}$3", new_value));
            let tmp_path = format!("{}.tmp", path.display());
            fs::write(&tmp_path, replaced.to_string())?;
            let _ = fs::remove_file(path);
            fs::rename(&tmp_path, path)?;
            println!(
                "🧬 [AST Mutator] Evolved .rs constant {} to {}",
                const_name, new_value
            );
        } else {
            return Err("Constant not found or malformed in the source file".into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_mutator_mutate_json_config() {
        let temp_dir = std::env::temp_dir();
        let json_file = temp_dir.join("test_ast_config.json");
        let initial_json = r#"{"Risk": {"MAX_DRAWDOWN": 0.05, "LEVERAGE": 5}}"#;
        fs::write(&json_file, initial_json).expect("write initial json");

        let mutator = ASTMutator::new();
        mutator
            .mutate_json_config(&json_file, "Risk.MAX_DRAWDOWN", serde_json::json!(0.08))
            .expect("mutation should succeed");

        let updated_content = fs::read_to_string(&json_file).expect("read updated json");
        assert!(updated_content.contains("0.08"));

        let _ = fs::remove_file(&json_file);
    }

    #[test]
    fn test_ast_mutator_mutate_rs_constant() {
        let temp_dir = std::env::temp_dir();
        let rs_file = temp_dir.join("test_ast_constant.rs");
        let initial_rs =
            "pub const KELLY_FRACTION: f64 = 0.25;\npub const MAX_SLIPPAGE: f64 = 0.001;\n";
        fs::write(&rs_file, initial_rs).expect("write initial rs");

        let mutator = ASTMutator::new();
        mutator
            .mutate_rs_constant(&rs_file, "KELLY_FRACTION", "0.50")
            .expect("constant mutation should succeed");

        let updated_content = fs::read_to_string(&rs_file).expect("read updated rs");
        assert!(updated_content.contains("pub const KELLY_FRACTION: f64 = 0.50;"));

        let _ = fs::remove_file(&rs_file);
    }
}
