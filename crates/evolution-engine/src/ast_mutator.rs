use std::fs;
use std::path::Path;
use serde_json::Value;

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
        fs::write(path, new_content)?;
        println!("🧬 [AST Mutator] Evolved {} at {} to {}", path.display(), key, new_value);
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
        
        let pattern = format!(r"(?m)(^\s*pub\s*const\s*{}\s*:\s*[a-zA-Z0-9_]+\s*=\s*)([^;]+)(;)", const_name);
        let regex = regex::Regex::new(&pattern)?;
        
        if regex.is_match(&content) {
            let replaced = regex.replace(&content, format!("${{1}}{}$3", new_value));
            fs::write(path, replaced.to_string())?;
            println!("🧬 [AST Mutator] Evolved .rs constant {} to {}", const_name, new_value);
        } else {
            return Err("Constant not found or malformed in the source file".into());
        }
        
        Ok(())
    }
}
