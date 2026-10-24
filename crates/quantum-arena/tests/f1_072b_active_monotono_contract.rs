//! SOL (2026-10-20): F1-072b — PUBLICACIÓN MONÓTONA DE `active.json`.
//!
//! El contrato F1-072 ya certificaba que las promociones concurrentes reclaman
//! generaciones ÚNICAS (O_EXCL sobre `gen_{N}.json`). Faltaba la otra mitad:
//! `active.json` se escribía sin orden y un promotor con generación MENOR podía
//! sobrescribir a uno mayor — el contrato veía `active = 5` con 8 generaciones
//! en el historial (3 fallos en CI, no flaky).
//!
//! Invariante que este contrato certifica:
//!   tras cualquier conjunto de promociones (secuenciales o concurrentes),
//!   `active.generation == max(generación del historial)`.
//!
//! Cifras: 6 hilos × 6 rondas ⇒ 36 generaciones únicas 1..=36; en CADA ronda
//! `active` debe ser el máximo del historial, no el de la última escritura.

use quantum_arena::genome::SuperGenotype;
use quantum_arena::genome_store::GenomeEnvelope;
use std::sync::Mutex;
use std::thread;

static ENV_MUTEX: Mutex<()> = Mutex::new(());

fn raiz() -> String {
    let temp_root = std::env::temp_dir().join(format!(
        "tg_test_f1072b_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    ));
    let _ = std::fs::create_dir_all(&temp_root);
    let s = temp_root.to_str().expect("utf8").to_string();
    unsafe {
        std::env::set_var("TG_CONFIG_ROOT", &s);
        std::env::set_var("TG_GENOME_ENV", "demo");
    }
    s
}

/// Máximo de generación presente en el historial del entorno demo.
fn max_historial(raiz: &str) -> u64 {
    let dir = std::path::Path::new(raiz).join("genomes/demo/history");
    let mut max = 0u64;
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            if let Some(num) = n
                .strip_prefix("gen_")
                .and_then(|r| r.strip_suffix(".json"))
                .and_then(|r| r.parse::<u64>().ok())
            {
                max = max.max(num);
            }
        }
    }
    max
}

fn active_generation(raiz: &str) -> u64 {
    let p = std::path::Path::new(raiz).join("genomes/demo/active.json");
    std::fs::read_to_string(&p)
        .ok()
        .and_then(|d| serde_json::from_str::<GenomeEnvelope>(&d).ok())
        .map(|e| e.generation)
        .unwrap_or(0)
}

#[test]
fn active_no_retrocede_en_promociones_secuenciales() {
    let _g = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let raiz = raiz();
    for i in 0..4 {
        let env = GenomeEnvelope::promote(
            SuperGenotype::new_baseline(0.0002, 0.0005),
            "seq",
            &format!("secuencial {i}"),
        )
        .expect("promote secuencial");
        assert_eq!(env.generation, i + 1, "generación secuencial monótona");
        assert_eq!(
            active_generation(&raiz),
            i + 1,
            "active sigue a la promoción secuencial"
        );
    }
    let _ = std::fs::remove_dir_all(&raiz);
}

#[test]
fn active_es_el_maximo_del_historial_en_cada_ronda_concurrente() {
    let _g = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let raiz = raiz();
    let hilos = 6;
    let rondas = 6;
    for ronda in 0..rondas {
        let mut handles = Vec::with_capacity(hilos);
        for i in 0..hilos {
            let r = ronda;
            handles.push(thread::spawn(move || {
                GenomeEnvelope::promote(
                    SuperGenotype::new_baseline(0.0002, 0.0005),
                    &format!("w{r}_{i}"),
                    &format!("ronda {r} hilo {i}"),
                )
                .expect("promote concurrente")
            }));
        }
        let mut gens = Vec::with_capacity(hilos);
        for h in handles {
            gens.push(h.join().expect("hilo sin panic").generation);
        }
        gens.sort_unstable();
        gens.dedup();
        assert_eq!(gens.len(), hilos, "generaciones únicas por ronda (F1-072)");

        let max_hist = max_historial(&raiz);
        let active = active_generation(&raiz);
        assert_eq!(
            active, max_hist,
            "ronda {ronda}: active ({active}) DEBE ser el máximo del historial ({max_hist}); \
             un promotor con generación menor no puede sobrescribir a uno mayor"
        );
        assert_eq!(
            active,
            *gens.last().expect("al menos una generación"),
            "ronda {ronda}: active coincide con la mayor generación promocionada"
        );
    }
    assert_eq!(
        max_historial(&raiz),
        (hilos * rondas) as u64,
        "sin huecos ni duplicados en {rondas} rondas × {hilos} hilos"
    );
    let _ = std::fs::remove_dir_all(&raiz);
}
