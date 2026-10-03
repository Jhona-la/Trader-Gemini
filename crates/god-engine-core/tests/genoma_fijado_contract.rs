//! CL-40 — un núcleo que evalúa un genoma concreto no lo sustituye por el
//! activo del almacén. `refresh_models` (primer evento y cada 1000) aplicaba
//! `config_dir/genomes/<env>/active.json` a CUALQUIER núcleo del proceso: en
//! el host (`TG_GENOME_ENV` definida) el examen del demonio, los universos
//! del bosque sombra y los mutantes de los backtests de evolución acababan
//! juzgando el genoma activo, no el candidato. Sólo el núcleo conectado a la
//! ejecución sigue al almacén. Binario propio: cambia el directorio de
//! trabajo y variables de entorno del proceso.
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

use god_engine_core::outcome_context::OutcomeContext;
use god_engine_core::GodEngineCore;
use quantum_arena::genome::SuperGenotype;
use quantum_arena::genome_store::GenomeEnvelope;
use quantum_arena::GlobalArena;

static PROCESO: Mutex<()> = Mutex::new(());

const UMBRAL_ACTIVO: f64 = 0.25;
const UMBRAL_CANDIDATO: f64 = 0.29;
const GENERACION_ACTIVA: u64 = 7;

/// Almacén temporal con un `demo/active.json` y el entorno del host; al
/// soltarse restaura el directorio, las variables y borra el almacén.
struct AlmacenTemporal {
    raiz: std::path::PathBuf,
    cwd_previo: std::path::PathBuf,
    env_previo: Option<std::ffi::OsString>,
    apagado_previo: Option<std::ffi::OsString>,
}

impl AlmacenTemporal {
    fn nuevo(etiqueta: &str) -> Self {
        let raiz = std::env::temp_dir().join(format!("cl40_{}_{}", etiqueta, std::process::id()));
        let silo = raiz.join("config_dir").join("genomes").join("demo");
        std::fs::create_dir_all(&silo).unwrap();
        let mut genoma = SuperGenotype::new_baseline(0.0002, 0.0005);
        genoma.tech_threshold = UMBRAL_ACTIVO;
        let sobre = GenomeEnvelope {
            schema_version: 1,
            generation: GENERACION_ACTIVA,
            created_ms: 0,
            source: "manual".into(),
            parent_generation: 0,
            promotion_reason: "CL-40 test".into(),
            genome: genoma,
        };
        std::fs::write(silo.join("active.json"), serde_json::to_string(&sobre).unwrap()).unwrap();
        let cwd_previo = std::env::current_dir().unwrap();
        let env_previo = std::env::var_os("TG_GENOME_ENV");
        let apagado_previo = std::env::var_os("GOD_NO_HOT_RELOAD");
        std::env::set_current_dir(&raiz).unwrap();
        // SAFETY: binario propio y `PROCESO` serializa los tests; ningún otro
        // hilo lee el entorno mientras se cambia.
        unsafe {
            std::env::set_var("TG_GENOME_ENV", "demo");
            std::env::remove_var("GOD_NO_HOT_RELOAD");
        }
        Self { raiz, cwd_previo, env_previo, apagado_previo }
    }
}

impl Drop for AlmacenTemporal {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.cwd_previo);
        // SAFETY: mismo razonamiento que en `nuevo`.
        unsafe {
            match &self.env_previo {
                Some(v) => std::env::set_var("TG_GENOME_ENV", v),
                None => std::env::remove_var("TG_GENOME_ENV"),
            }
            if let Some(v) = &self.apagado_previo {
                std::env::set_var("GOD_NO_HOT_RELOAD", v);
            }
        }
        let _ = std::fs::remove_dir_all(&self.raiz);
    }
}

fn nucleo_con_candidato(contexto: OutcomeContext) -> (GodEngineCore, Arc<GlobalArena>) {
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut candidato = SuperGenotype::new_baseline(0.0002, 0.0005);
    candidato.tech_threshold = UMBRAL_CANDIDATO;
    candidato.apply_to_arena(&arena);
    let mut core = GodEngineCore::new_with_outcome_context(arena.clone(), contexto);
    core.swing_nn = None;
    core.scalp_forest = None;
    (core, arena)
}

fn un_evento(core: &mut GodEngineCore) {
    // El primer evento unificado (tick 0) dispara `refresh_models`.
    let _ = core.process_event(
        0, true, false, false, 100.0, 1.0, 100.0, 100.01, 5.0, 5.0, 0.0, 0.0, 10_000, false,
        &[0.0; 54], false,
    );
}

#[test]
fn cl40_el_nucleo_aislado_evalua_su_candidato_y_no_el_genoma_activo() {
    let _g = PROCESO.lock().unwrap_or_else(|e| e.into_inner());
    let _almacen = AlmacenTemporal::nuevo("aislado");
    let (mut core, arena) = nucleo_con_candidato(OutcomeContext::IsolatedSimulation);
    un_evento(&mut core);
    let leido = arena.config.tech_threshold.load(Ordering::Relaxed);
    assert!(
        (leido - UMBRAL_CANDIDATO).abs() < 1e-12,
        "el examen debe juzgar al candidato (tech_threshold {UMBRAL_CANDIDATO}); \
         leído {leido} = el genoma activo del almacén"
    );
    assert_eq!(core.applied_generation.load(Ordering::SeqCst), 0);
}

#[test]
fn cl40_el_nucleo_conectado_a_la_ejecucion_sigue_al_almacen() {
    let _g = PROCESO.lock().unwrap_or_else(|e| e.into_inner());
    let _almacen = AlmacenTemporal::nuevo("vivo");
    let (mut core, arena) = nucleo_con_candidato(OutcomeContext::ExchangeLocalEstimate);
    un_evento(&mut core);
    let leido = arena.config.tech_threshold.load(Ordering::Relaxed);
    assert!(
        (leido - UMBRAL_ACTIVO).abs() < 1e-12,
        "el núcleo vivo adopta la generación sancionada; leído {leido}"
    );
    assert_eq!(core.applied_generation.load(Ordering::SeqCst), GENERACION_ACTIVA);
}

#[test]
fn cl40_el_host_replanta_el_bosque_por_generacion() {
    // Guardia sobre la fuente: el bosque sombra ya no adopta el genoma activo
    // por su cuenta, así que el host debe replantarlo por generación (tras la
    // cosecha y cuando el núcleo vivo aplica una nueva), nunca a ciegas.
    let host = include_str!("../../../src/bin/god_engine.rs");
    let replantar = ["shadow_forest", ".replant("].concat();
    assert!(!host.contains(&replantar), "el host replanta sin anotar la generación");
    let seguir = ["shadow_forest", ".seguir_generacion("].concat();
    assert!(
        host.matches(&seguir).count() >= 2,
        "el host debe seguir la generación tras la cosecha y antes de cosechar"
    );
}
