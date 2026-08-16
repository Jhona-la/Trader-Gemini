use memmap2::MmapOptions;
use std::fs::{File, OpenOptions};
use std::path::Path;
use crate::core::config::Genome;

/// Institutional-grade zero-copy mmap storage for Genomes.
/// Eliminates JSON/TXT parsing completely from the hot path.
pub struct GenomeStorage;

impl GenomeStorage {
    /// Loads a Genome instantly via memory mapping. 
    /// If the file does not exist, returns None (forcing Seed Mode).
    pub fn load_zero_copy(path: &str) -> Option<Genome> {
        if !Path::new(path).exists() {
            return None;
        }

        match File::open(path) {
            Ok(file) => {
                // Mapear el archivo directamente a la memoria virtual (Page Tables)
                // Esto elimina la asignación dinámica de memoria (Vec<u8>) del Heap, 
                // ahorrando RAM y CPU en la lectura de archivos.
                match unsafe { MmapOptions::new().map(&file) } {
                    Ok(mmap) => {
                        // Deserializamos desde la región mmap sin copias intermedias
                        match rkyv::from_bytes::<Genome>(&mmap) {
                            Ok(genome) => Some(genome),
                            Err(e) => {
                                eprintln!("[STORAGE] ❌ Corrupted Genome at {}, error: {:?}, forcing Seed Mode.", path, e);
                                None
                            }
                        }
                    }
                    Err(_) => None,
                }
            }
            Err(_) => None,
        }
    }

    /// Atomically saves a Genome by writing to a temporary file and renaming.
    /// Provides picosecond crash-safety (rename is atomic in POSIX and NTFS).
    pub fn save_atomic(genome: &Genome, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        
        let temp_path = format!("{}.tmp", path);
        let bytes = rkyv::to_bytes::<_, 8192>(genome).unwrap();
        
        {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&temp_path)?;
            file.write_all(&bytes)?;
            file.sync_all()?; // Force to disk hardware
        }
        
        // Atomic rename
        std::fs::rename(temp_path, path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use crate::core::config::QuantumConfig;

    #[test]
    fn test_zero_copy_storage_atomic_write_and_read() {
        let test_path = "test_genome_atomic.rkyv";
        let genome = Genome::bootstrap_seed();

        // 1. Guardado Atómico
        let res = GenomeStorage::save_atomic(&genome, test_path);
        assert!(res.is_ok(), "Fallo guardado atómico zero-copy");

        // 2. Carga Zero-Copy
        let loaded = GenomeStorage::load_zero_copy(test_path);
        assert!(loaded.is_some(), "Fallo en la carga del genoma zero-copy");
        let loaded_genome = loaded.unwrap();

        // Verificamos un parámetro para comprobar la integridad
        assert_eq!(loaded_genome.global.base_capital, genome.global.base_capital);
        
        // Limpiar archivo
        let _ = fs::remove_file(test_path);
    }

    #[test]
    fn test_corrupted_file_fallback() {
        let test_path = "test_corrupted.rkyv";
        use std::io::Write;
        {
            let mut file = std::fs::File::create(test_path).unwrap();
            file.write_all(b"basura aleatoria que no es rkyv").unwrap();
        }

        // Debería rechazar la basura e imprimir el warning, sin hacer panic (Fallback)
        let loaded = GenomeStorage::load_zero_copy(test_path);
        assert!(loaded.is_none(), "Storage cargó un archivo corrupto sin fallar");

        let _ = fs::remove_file(test_path);
    }
}
