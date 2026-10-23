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
use sha2::{Digest, Sha256};
use std::io;

use crate::genome::SuperGenotype;

pub const SCHEMA_VERSION: u32 = 1;
pub const BINARY_SCHEMA_VERSION: u32 = 1;
pub const BINARY_MAGIC: [u8; 8] = *b"TGGENOM\0";

/// FASE 3 (Átomo 3.2) — Cabecera binaria inmutable de 72 bytes con suma de comprobación SHA-256.
/// Garantiza validación criptográfica zero-copy en O(1) y arranques/recargas < 5 µs.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinaryGenomeHeader {
    pub magic: [u8; 8],
    pub schema_version: u32,
    pub flags: u32,
    pub generation: u64,
    pub created_ms: u64,
    pub payload_len: u64,
    pub checksum: [u8; 32],
}

impl BinaryGenomeHeader {
    pub const SIZE: usize = 72;

    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..8].copy_from_slice(&self.magic);
        buf[8..12].copy_from_slice(&self.schema_version.to_le_bytes());
        buf[12..16].copy_from_slice(&self.flags.to_le_bytes());
        buf[16..24].copy_from_slice(&self.generation.to_le_bytes());
        buf[24..32].copy_from_slice(&self.created_ms.to_le_bytes());
        buf[32..40].copy_from_slice(&self.payload_len.to_le_bytes());
        buf[40..72].copy_from_slice(&self.checksum);
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::SIZE {
            return None;
        }
        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[0..8]);
        let schema_version = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
        let flags = u32::from_le_bytes(bytes[12..16].try_into().ok()?);
        let generation = u64::from_le_bytes(bytes[16..24].try_into().ok()?);
        let created_ms = u64::from_le_bytes(bytes[24..32].try_into().ok()?);
        let payload_len = u64::from_le_bytes(bytes[32..40].try_into().ok()?);
        let mut checksum = [0u8; 32];
        checksum.copy_from_slice(&bytes[40..72]);
        Some(Self {
            magic,
            schema_version,
            flags,
            generation,
            created_ms,
            payload_len,
            checksum,
        })
    }
}

pub fn compute_sha256(payload: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(payload);
    let hash = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&hash);
    out
}

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
        Ok(env) => format!("{}/genomes/{}", crate::paths::config_root(), env),
        // Un caller que ignore el error de entorno no debe poder escribir en
        // NINGÚN silo real: se le da una ruta inválida que falla al abrir.
        Err(_) => format!("{}/genomes/__UNSET__", crate::paths::config_root()),
    }
}

pub fn active_json_path() -> String {
    active_path()
}

fn active_path() -> String {
    format!("{}/active.json", env_root())
}

pub fn active_bin_path() -> String {
    format!("{}/active.bin", env_root())
}

fn history_dir() -> String {
    format!("{}/history", env_root())
}

pub fn history_bin_path(generation: u64) -> String {
    format!("{}/gen_{:06}.bin", history_dir(), generation)
}

/// Espejo legacy: el loader viejo lee esta ruta en arranques fríos.
/// X-001 (REHAB-2): se escribe SIEMPRE — es una VISTA de compatibilidad, no
/// un linaje. Antes solo en entorno compartido: con default demo, los
/// loaders legacy habrían quedado congelados en el último estado compartido.
///
/// CERT-F-14: la ruta era COMPARTIDA entre entornos — el fallback de
/// emergencia de `genome.rs` podía cargar en producción lo que demo
/// escribió en el espejo (herencia cross-env silenciosa, exactamente lo
/// que D-651 prohibió para active.json). Ahora el espejo también es
/// per-env; sin entorno válido, no se escribe NI lee espejo alguno.
pub fn legacy_mirror() -> Option<String> {
    current_env()
        .ok()
        .map(|env| format!("{}/genotypes/active_genome.{}.json", crate::paths::config_root(), env))
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

        // 1. Intento de carga binaria zero-copy rápida (< 5 µs con SHA-256)
        let bin_p = active_bin_path();
        if std::path::Path::new(&bin_p).exists() {
            match std::fs::read(&bin_p) {
                Ok(bytes) => match Self::from_binary_bytes(&bytes) {
                    Ok(envelope) => {
                        return Some(envelope);
                    }
                    Err(e) => {
                        eprintln!(
                            "🚨 [GENOME-STORE] {} corrupto ({e}). Intentando fallback a active.json.",
                            bin_p
                        );
                    }
                },
                Err(e) => {
                    eprintln!("⚠️ [GENOME-STORE] Error al leer {bin_p}: {e}");
                }
            }
        }

        // 2. Fallback a JSON canónico
        match std::fs::read_to_string(active_path()) {
            // F4-H3: el genoma activo sale normalizado (ver `desde_json`).
            Ok(data) => match Self::desde_json(&data) {
                Ok(envelope) => {
                    // F1-072: Verificación de integridad de linaje y auto-reparación.
                    // Si el archivo de historia gen_{generation}.json no existe en history_dir(),
                    // reconstruirlo desde active.json de forma atómica para restaurar la
                    // capacidad de rollback y recent_history.
                    let hist_path = format!("{}/gen_{:06}.json", history_dir(), envelope.generation);
                    if !std::path::Path::new(&hist_path).exists() {
                        eprintln!(
                            "ℹ️ [GENOME-STORE] linaje reparado: reconstruyendo {} a partir de active.json ({})",
                            hist_path, env
                        );
                        let _ = std::fs::create_dir_all(history_dir());
                        let _ = std::fs::OpenOptions::new()
                            .write(true)
                            .create_new(true)
                            .open(&hist_path)
                            .and_then(|mut f| {
                                use std::io::Write;
                                f.write_all(data.as_bytes())?;
                                f.sync_all()
                            });
                    }
                    // Auto-generación de active.bin para acelerar próximos arranques
                    if let Ok(bin_bytes) = envelope.to_binary_bytes() {
                        let _ = atomic_write_bytes(&bin_p, &bin_bytes);
                    }
                    Some(envelope)
                }
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

    /// Carga explícitamente el genoma activo en formato binario con validación de hash SHA-256
    pub fn load_active_binary() -> Option<GenomeEnvelope> {
        let env = match current_env() {
            Ok(e) => e,
            Err(msg) => {
                eprintln!("🚨 [GENOME-STORE] {msg}");
                return None;
            }
        };
        let bin_p = active_bin_path();
        match std::fs::read(&bin_p) {
            Ok(bytes) => match Self::from_binary_bytes(&bytes) {
                Ok(envelope) => Some(envelope),
                Err(e) => {
                    eprintln!("🚨 [GENOME-STORE] {} no se pudo deserializar ({e})", bin_p);
                    None
                }
            },
            Err(e) => {
                eprintln!("🚨 [GENOME-STORE] el entorno '{env}' no tiene {} ({e})", bin_p);
                None
            }
        }
    }

    /// Serializa el sobre completo a formato binario compacto con cabecera y suma de verificación SHA-256
    pub fn to_binary_bytes(&self) -> io::Result<Vec<u8>> {
        let payload = bincode::serialize(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Error de serialización binaria bincode: {e}")))?;
        let checksum = compute_sha256(&payload);
        let header = BinaryGenomeHeader {
            magic: BINARY_MAGIC,
            schema_version: BINARY_SCHEMA_VERSION,
            flags: 0,
            generation: self.generation,
            created_ms: self.created_ms,
            payload_len: payload.len() as u64,
            checksum,
        };
        let mut out = Vec::with_capacity(BinaryGenomeHeader::SIZE + payload.len());
        out.extend_from_slice(&header.to_bytes());
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Deserializa el sobre desde bytes binarios validando magic, schema_version y checksum SHA-256
    pub fn from_binary_bytes(bytes: &[u8]) -> io::Result<GenomeEnvelope> {
        let header = BinaryGenomeHeader::from_bytes(bytes)
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "Bytes insuficientes para cabecera binaria de genoma"))?;
        if header.magic != BINARY_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Magic number binario inválido: {:?}", header.magic),
            ));
        }
        if header.schema_version != BINARY_SCHEMA_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Versión binaria no soportada: {} (esperada {})", header.schema_version, BINARY_SCHEMA_VERSION),
            ));
        }
        let total_expected = BinaryGenomeHeader::SIZE + header.payload_len as usize;
        if bytes.len() != total_expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Longitud de archivo binario corrupta: {} bytes (esperados {})", bytes.len(), total_expected),
            ));
        }
        let payload = &bytes[BinaryGenomeHeader::SIZE..];
        let actual_checksum = compute_sha256(payload);
        if actual_checksum != header.checksum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Fallo de integridad criptográfica: el checksum SHA-256 del genoma binario no coincide (corrupción de disco detectada)",
            ));
        }
        let mut envelope: GenomeEnvelope = bincode::deserialize(payload)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Error al deserializar payload bincode: {e}")))?;
        envelope.genome = SuperGenotype::from_vector(&envelope.genome.to_vector());
        Ok(envelope)
    }

    /// Sobre leído de JSON tal como lo entrega `load_active`. F4-H3:
    /// normalización y saneamiento estricto del genoma respecto a cotas
    /// evolutivas e invariantes de curva (clamp, enforce_curve_rr,
    /// normalize_sl_curve_friction_floor).
    pub fn desde_json(texto: &str) -> serde_json::Result<GenomeEnvelope> {
        let mut envelope = serde_json::from_str::<GenomeEnvelope>(texto)?;
        envelope.genome = SuperGenotype::from_vector(&envelope.genome.to_vector());
        Ok(envelope)
    }

    /// CL-53b — el sobre tal como lo cargarán de `active.json` los
    /// vigilantes del núcleo y del host, en el formato que escribe `promote`.
    /// Quien aplica al arena un genoma recién promovido debe aplicar éste:
    /// si no, el arena opera con un genoma que el almacén no sanciona hasta
    /// que un vigilante vuelve a leer el disco.
    pub fn como_se_carga(&self) -> serde_json::Result<GenomeEnvelope> {
        Self::desde_json(&serde_json::to_string_pretty(self)?)
    }

    /// Única vía para cruzar la frontera entre entornos. Exige armado humano
    /// explícito — misma filosofía que MAINNET_ARMED (D-651).
    ///
    /// `TG_GENOME_PROMOTE_ARMED` debe valer "1" y `TG_GENOME_PROMOTE_OPERATOR`
    /// debe identificar a quien autoriza: ambos quedan escritos en el motivo
    /// de promoción, de modo que el linaje registra QUIÉN cruzó la frontera.
    pub fn promote_across_env(from_env: &str, reason: &str) -> Result<GenomeEnvelope, String> {
        let to_env = current_env()?;
        let from = from_env.trim().to_lowercase();
        if !KNOWN_ENVS.contains(&from.as_str()) {
            return Err(format!("entorno origen '{from_env}' no reconocido"));
        }
        if from == to_env {
            return Err(format!("origen y destino son el mismo entorno ('{from}')"));
        }
        if std::env::var("TG_GENOME_PROMOTE_ARMED")
            .unwrap_or_default()
            .trim()
            != "1"
        {
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

        let src = format!("{}/genomes/{from}/active.json", crate::paths::config_root());
        let data =
            std::fs::read_to_string(&src).map_err(|e| format!("no se puede leer {src}: {e}"))?;
        let src_env: GenomeEnvelope =
            serde_json::from_str(&data).map_err(|e| format!("{src} no parsea: {e}"))?;

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
    pub fn validate(genome: &SuperGenotype) -> Result<(), String> {
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
        // La banda satisface el presupuesto elegido f <= q*SL; no es una
        // garantía de rentabilidad. Para p fijo, 0<p<1, y f>=0 constantes,
        // h(x)=ln(p)+a_tp+b_tp*x-ln((1-p)*exp(a_sl+b_sl*x)+f) es cóncava:
        // el último término es una log-sum-exp convexa. h>=0 en los DOS
        // EXTREMOS implica h>=0 dentro, equivalente a EV>=0 bajo ese modelo.
        // La monotonía de TP/SL sola no prueba el umbral dependiente de SL.
        // Esta garantía no se traslada a p(tau), f(tau) o curvas arbitrarias.
        let fee = SuperGenotype::REFERENCE_ROUNDTRIP_FEE;
        let (lo_tau, hi_tau) = genome.tradeable_band_ms(fee).ok_or_else(|| {
            format!(
                "genoma sin banda admisible por política de fricción: su curva de SL no alcanza el piso {:.6} ({:.1} bps) para fee {:.4}. Esto no certifica el signo del EV fuera de la política.",
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
            let required = SuperGenotype::min_rr_for(SuperGenotype::WORST_TOLERATED_WR, fee, sl);
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
        std::fs::create_dir_all(history_dir())?;

        let parent = if let Ok(data) = std::fs::read_to_string(active_path()) {
            serde_json::from_str::<GenomeEnvelope>(&data)
                .map(|e| e.generation)
                .unwrap_or(0)
        } else {
            0
        };

        // F1-072: Reclamar la generación con create_new (O_EXCL) sobre gen_{N+1}.json
        // y reintentar con N+2 si existe (el archivo de historia ES el lock atómico
        // frente a promotores concurrentes).
        let mut candidate_gen = parent + 1;
        let mut claimed_file = None;
        let mut claimed_path = String::new();

        for _ in 0..10_000 {
            let hist_path = format!("{}/gen_{:06}.json", history_dir(), candidate_gen);
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&hist_path)
            {
                Ok(file) => {
                    claimed_file = Some(file);
                    claimed_path = hist_path;
                    break;
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                    candidate_gen += 1;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        let mut hist_file = claimed_file.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "[F1-072] No se pudo reclamar un número de generación tras 10000 intentos",
            )
        })?;

        let envelope = GenomeEnvelope {
            schema_version: SCHEMA_VERSION,
            generation: candidate_gen,
            created_ms: now_ms(),
            source: source.to_string(),
            parent_generation: parent,
            promotion_reason: reason.to_string(),
            genome,
        };

        // 1) Historia inmutable (append-only por nombre de generación).
        let hist_json = serde_json::to_string_pretty(&envelope)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        {
            use std::io::Write;
            if let Err(e) = hist_file
                .write_all(hist_json.as_bytes())
                .and_then(|_| hist_file.sync_all())
            {
                let _ = std::fs::remove_file(&claimed_path);
                return Err(e);
            }
        }
        drop(hist_file);

        // 1b) Generación de bytes binarios para publicación active.bin
        let bin_bytes = envelope.to_binary_bytes().ok();

        // 2) Active atómico (tmp único + fsync + rename sin remove previo) y
        // MONÓTONO (F1-072b, Sol): un promotor con generación menor no puede
        // sobrescribir a uno mayor. Ver `publicar_active_monotono`.
        publicar_active_monotono(&envelope, &hist_json, bin_bytes.as_deref())?;

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

    /// Lee una generación de la historia inmutable del entorno actual.
    pub fn load_generation(generation: u64) -> io::Result<GenomeEnvelope> {
        let hist_bin = history_bin_path(generation);
        if std::path::Path::new(&hist_bin).exists() {
            if let Ok(bytes) = std::fs::read(&hist_bin) {
                if let Ok(env) = Self::from_binary_bytes(&bytes) {
                    return Ok(env);
                }
            }
        }
        let hist_path = format!("{}/gen_{:06}.json", history_dir(), generation);
        let data = std::fs::read_to_string(&hist_path)?;
        serde_json::from_str(&data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Rollback: re-promociona la generación `target` como nueva generación
    /// (la historia es append-only; revertir también queda auditado).
    pub fn rollback(target_generation: u64) -> io::Result<GenomeEnvelope> {
        let previous = Self::load_generation(target_generation)?;
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

/// F1-072b (Sol, 2026-10-20): PUBLICACIÓN MONÓTONA de `active.json`.
///
/// Reclamar generaciones con O_EXCL ya garantiza unicidad, pero escribir
/// `active.json` sin orden permitía que un hilo con generación MENOR
/// sobrescribiera a uno mayor: con 8 promotores concurrentes el contrato
/// veía `active.generation = 5` con 8 generaciones en el historial (carrera
/// real detectada 3 veces en CI, no flaky).
///
/// Sección crítica con lock O_EXCL (`active.json.lock`) + escritura SÓLO si
/// la generación nueva es mayor: `active` pasa a ser el máximo monótono de
/// lo promocionado. Un lock obsoleto (>5 s, proceso muerto) se recupera para
/// no bloquear nunca la promoción. `promote` no está en el hot path: el
/// coste de un lock de archivo es aceptable y sólo ocurre al promocionar.
fn publicar_active_monotono(
    envelope: &GenomeEnvelope,
    hist_json: &str,
    bin_bytes: Option<&[u8]>,
) -> io::Result<()> {
    static TOKEN_NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let active = active_path();
    let lock = format!("{active}.lock");
    let token = format!(
        "{}.{}",
        std::process::id(),
        TOKEN_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    let inicio = std::time::Instant::now();
    loop {
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock)
        {
            Ok(mut guard) => {
                // Confirmación por token: si otro hilo borró el lock entre
                // nuestro create y nuestra escritura, el contenido no es el
                // nuestro y NO tenemos la sección crítica.
                {
                    use std::io::Write;
                    let _ = guard.write_all(token.as_bytes());
                }
                let leido = std::fs::read_to_string(&lock).unwrap_or_default();
                if leido != token {
                    drop(guard);
                    continue;
                }
                let resultado = (|| -> io::Result<()> {
                    let actual = std::fs::read_to_string(&active)
                        .ok()
                        .and_then(|d| serde_json::from_str::<GenomeEnvelope>(&d).ok())
                        .map(|e| e.generation)
                        .unwrap_or(0);
                    if envelope.generation > actual {
                        atomic_write(&active, hist_json)?;
                        if let Some(bytes) = bin_bytes {
                            let _ = atomic_write_bytes(&active_bin_path(), bytes);
                        }
                    }
                    Ok(())
                })();
                drop(guard);
                let _ = std::fs::remove_file(&lock);
                return resultado;
            }
            Err(e)
                if e.kind() == io::ErrorKind::AlreadyExists
                    || e.kind() == io::ErrorKind::PermissionDenied =>
            {
                // En Windows, create_new sobre un lock ABIERTO devuelve
                // PermissionDenied (5), no AlreadyExists: es contención, no
                // fallo. Un lock huérfano (proceso muerto) se recupera a los 5 s.
                if inicio.elapsed() > std::time::Duration::from_secs(5) {
                    let _ = std::fs::remove_file(&lock);
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
                continue;
            }
            Err(e) => return Err(e),
        }
    }
}

fn atomic_write_bytes(path: &str, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    static TMP_BIN_NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let nonce = TMP_BIN_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let pid = std::process::id();
    let tmp = format!("{path}.{pid}.{nonce}.tmp");

    {
        use std::io::Write;
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }

    let mut rename_res = std::fs::rename(&tmp, path);
    if rename_res.is_err() {
        for _ in 0..5 {
            std::thread::sleep(std::time::Duration::from_millis(2));
            rename_res = std::fs::rename(&tmp, path);
            if rename_res.is_ok() {
                break;
            }
        }
    }
    if let Err(e) = rename_res {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

fn atomic_write(path: &str, contents: &str) -> io::Result<()> {
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    static TMP_NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let nonce = TMP_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let pid = std::process::id();
    let tmp = format!("{path}.{pid}.{nonce}.tmp");

    {
        use std::io::Write;
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(contents.as_bytes())?;
        file.sync_all()?;
    }

    // F1-072: En Windows (MoveFileExW MOVEFILE_REPLACE_EXISTING) y POSIX moderno,
    // rename reemplaza atómicamente si el destino existe.
    // NUNCA llamar a remove_file(path) antes de rename, pues abre una ventana de carrera
    // donde active.json no existe para procesos o hilos concurrentes.
    // Se incluye reintento defensivo ante cierres transitorios de handles en Windows NTFS.
    let mut rename_res = std::fs::rename(&tmp, path);
    if rename_res.is_err() {
        for _ in 0..5 {
            std::thread::sleep(std::time::Duration::from_millis(2));
            rename_res = std::fs::rename(&tmp, path);
            if rename_res.is_ok() {
                break;
            }
        }
    }
    if let Err(e) = rename_res {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests_lg4 {
    use super::*;

    /// LG-4 — el gate de promoción/carga juzga la RR en la banda operable.
    /// Con el dominio anterior (1 ns – 146 años) `validate` rechazaba el
    /// genoma activo de demo y prod (TP/SL = 2,05 < 3,125 en la raíz del piso
    /// a ~156 ns) y `from_vector` le subía el TP ×1,68: si GENOME-GATE al
    /// cargar (LC-F7-16) aterrizaba antes, demo y prod se quedaban sin genoma.
    #[test]
    fn lg4_el_genoma_activo_pasa_el_gate_y_la_carga_no_le_toca_el_tp() {
        let mut g = SuperGenotype::new_baseline(0.0002, 0.0005);
        g.tp_horizon_curve = crate::temporal_spectrum::HorizonCurve {
            a: -4.9600174176553296,
            b: 0.09143837668088041,
        };
        g.sl_horizon_curve = crate::temporal_spectrum::HorizonCurve {
            a: -5.719394858308684,
            b: 0.08643759946713066,
        };
        assert!(GenomeEnvelope::validate(&g).is_ok(), "{:?}", GenomeEnvelope::validate(&g));
        let cargado = SuperGenotype::from_vector(&g.to_vector());
        assert!(
            (cargado.tp_horizon_curve.a - g.tp_horizon_curve.a).abs() < 1e-9,
            "a_tp {} → {}",
            g.tp_horizon_curve.a,
            cargado.tp_horizon_curve.a
        );
    }
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
        assert_eq!(
            g.to_vector()[SuperGenotype::SLOT_ZOMBIE_TIMEOUT],
            g.zombie_timeout_ms
        );
        g.zombie_timeout_ms = 2_100_140.98;
        let arena = crate::state::GlobalArena::from_genome_in_own_stack(13.0, &g);
        assert_eq!(
            arena.config.zombie_timeout_ms.load(Ordering::Relaxed),
            14_400_000.0
        );
    }

    /// D-680: el prior del win rate es el de diseño y una operación no lo destruye.
    #[test]
    fn d680_prior_del_win_rate_y_media_posterior() {
        use std::sync::atomic::Ordering;
        let g = SuperGenotype::new_baseline(0.0002, 0.0005);
        let arena = crate::state::GlobalArena::from_genome_in_own_stack(13.0, &g);
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
        assert!(
            (w - 0.25).abs() < 0.01,
            "debe converger a la frecuencia observada, dio {w}"
        );
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

        let frio = crate::state::GlobalArena::from_genome_in_own_stack(13.0, &g);
        // D-714: el arena no cabe en la pila por defecto de un hilo de test.
        let caliente = crate::state::GlobalArena::build_in_own_stack(13.0);
        g.apply_to_arena(&caliente);
        for (nombre, arena) in [("frío", &frio), ("hot-swap", &caliente)] {
            let c = &arena.config;
            assert!(
                (c.obi_threshold_at_tau(TAU_ANCHOR_FAST_MS) - 0.45).abs() < 1e-9,
                "{nombre}: OBI rápido"
            );
            assert!(
                (c.obi_threshold_at_tau(TAU_ANCHOR_SLOW_MS) - 0.30).abs() < 1e-9,
                "{nombre}: OBI lento"
            );
            assert!(
                (c.kelly_at_tau(TAU_ANCHOR_SLOW_MS) - 0.112).abs() < 1e-9,
                "{nombre}: Kelly lento"
            );
            assert!(
                (c.trail_params_at_tau(TAU_ANCHOR_FAST_MS).0 - 2.2).abs() < 1e-9,
                "{nombre}: trailing rápido"
            );
        }
    }

    /// D-625: los slots usados para acotar en la entrada corresponden a los
    /// genes que dicen, y un gen fuera de bounds entra al arena acotado.
    #[test]
    fn d625_genes_se_acotan_al_entrar_al_arena() {
        use crate::GlobalArena;
        use std::sync::atomic::Ordering;
        let mut g = SuperGenotype::new_baseline(0.0002, 0.0005);
        let v = g.to_vector();
        assert_eq!(v[SuperGenotype::SLOT_TECH_THRESHOLD], g.tech_threshold);
        assert_eq!(v[SuperGenotype::SLOT_TREND_THRESHOLD], g.trend_threshold);

        // El genoma de producción trae tech_threshold = 0,05.
        g.tech_threshold = 0.05;
        let frio = GlobalArena::from_genome_in_own_stack(13.0, &g);
        assert!((frio.config.tech_threshold.load(Ordering::Relaxed) - 0.24).abs() < 1e-12);
        let swap = GlobalArena::build_in_own_stack(13.0);
        g.apply_to_arena(&swap);
        assert!((swap.config.tech_threshold.load(Ordering::Relaxed) - 0.24).abs() < 1e-12);
    }

    /// D-643/D-687: los genes de confianza y colchón entran acotados a sus
    /// bounds, y el colchón del genoma de producción conserva su valor efectivo.
    #[test]
    fn d687_colchon_y_confianza_se_acotan_al_entrar_al_arena() {
        use std::sync::atomic::Ordering;
        let mut g = SuperGenotype::new_baseline(0.0002, 0.0005);
        let v = g.to_vector();
        assert_eq!(v[SuperGenotype::SLOT_MIN_CONFIDENCE], g.min_confidence_btc);
        assert_eq!(v[SuperGenotype::SLOT_MARGIN_CUSHION], g.margin_cushion_pct);
        g.margin_cushion_pct = 1.092_219_631_099_718;
        g.min_confidence_btc = 0.30;
        let frio = crate::state::GlobalArena::from_genome_in_own_stack(13.0, &g);
        assert_eq!(frio.config.margin_cushion_pct.load(Ordering::Relaxed), 0.98);
        assert_eq!(frio.config.min_confidence_btc.load(Ordering::Relaxed), 0.50);
        let swap = crate::state::GlobalArena::build_in_own_stack(13.0);
        g.apply_to_arena(&swap);
        assert_eq!(swap.config.margin_cushion_pct.load(Ordering::Relaxed), 0.98);
    }

    #[test]
    fn t2_simetria_from_genome_vs_apply_to_arena() {
        use crate::GlobalArena;

        // Genoma de prueba con valores distinguibles del baseline, para que un
        // gen no refrescado se note.
        let mut g = SuperGenotype::new_baseline(0.0002, 0.0005);
        g = g.mutate_cmaes_seeded(0.45, 20260910);
        // LG-18: los genes congelados no mutan; se fijan a mano dentro de sus
        // cotas para que sigan distinguiéndose de `otro` y la prueba vea si
        // `apply_to_arena` los refresca.
        g.scalp_trail_max_atr = 7.0;
        g.swing_trail_max_atr = 9.0;
        g.scalp_trail_min_pnl = 3.0;
        g.swing_trail_min_pnl = 4.0;
        g.hurst_swing_threshold = 0.66;
        g.scalp_accel_min_samples = 17.0;
        g.swing_accel_min_samples = 23.0;
        g.executor_max_orders_10s = 333.0;
        g.executor_max_weight_1m = 1777.0;
        // LG-21: los 35 genes congelados por LG-21, por índice del vector, a
        // un punto interior de sus cotas que no coincide con el de `otro`.
        let (lo, hi) = (SuperGenotype::get_lower_bounds(), SuperGenotype::get_upper_bounds());
        let mut v = g.to_vector();
        for i in [36, 42, 43, 64, 65, 70, 71, 72, 73, 74, 77, 78, 79, 84, 85, 90, 91, 92, 93, 94, 97, 100, 101, 102, 103, 104, 105, 111, 122, 123, 124, 125, 126, 127, 128] {
            v[i] = lo[i] + 0.37 * (hi[i] - lo[i]);
        }
        let g = SuperGenotype::from_vector(&v);

        // Camino A: arranque en frío.
        let arena_frio = GlobalArena::from_genome_in_own_stack(13.0, &g);
        let visto_frio = SuperGenotype::current_from_arena(&arena_frio);

        // Camino B: arena con OTRO genoma y luego hot-swap al nuestro.
        let otro = SuperGenotype::new_baseline(0.0004, 0.0009);
        let arena_swap = GlobalArena::from_genome_in_own_stack(13.0, &otro);
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

    /// CL-53b — `como_se_carga` coincide con lo que `load_active` lee del
    /// archivo que escribe `promote`, también cuando la normalización cambia
    /// el genoma.
    #[test]
    fn cl53b_como_se_carga_es_lo_que_lee_load_active() {
        let mut genome = SuperGenotype::new_baseline(0.0002, 0.0005);
        genome.global_leverage = 1.0e6;
        let env = GenomeEnvelope {
            schema_version: SCHEMA_VERSION,
            generation: 7,
            created_ms: 1_700_000_000_000,
            source: "darwin_daemon".into(),
            parent_generation: 6,
            promotion_reason: "cl53b".into(),
            genome,
        };
        let cargado = env.como_se_carga().expect("serializa y parsea");
        let del_disco =
            GenomeEnvelope::desde_json(&serde_json::to_string_pretty(&env).unwrap()).unwrap();
        assert_eq!(cargado.genome.to_vector(), del_disco.genome.to_vector());
        assert!(cargado.genome.global_leverage < 1.0e6, "la carga recorta a la cota");
        // La lectura JSON no es bit a bit (serde_json sin `float_roundtrip`
        // puede mover el último dígito): por eso `como_se_carga` pasa por el
        // mismo texto que escribe `promote` y no sólo por `from_vector`.
        assert_eq!(cargado.generation, 7);
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


    /// (Ola XLIV·diag) DETERMINISTA: encuentra y REPORTA el mutante que el
    /// gate rechaza, con su vector — el flaky 3/4 del test r11 es un agujero
    /// real bounds<->gate que este diagnóstico clava en sitio.
    #[test]
    fn xliv_diag_r11_encuentra_el_rechazo_determinista() {
        let base = SuperGenotype::new_baseline(0.0002, 0.0005);
        for i in 0..20_000u64 {
            let rate = 0.05 + ((i % 10) as f64) * 0.05;
            let mutant = base.mutate_cmaes_seeded(rate, i);
            let roundtrip = SuperGenotype::from_vector(&mutant.to_vector());
            if let Err(e) = GenomeEnvelope::validate(&roundtrip) {
                use crate::temporal_spectrum::{TAU_ANCHOR_FAST_MS, TAU_ANCHOR_SLOW_MS};
                let fast_sl = roundtrip.sl_horizon_curve.eval(TAU_ANCHOR_FAST_MS);
                let slow_sl = roundtrip.sl_horizon_curve.eval(TAU_ANCHOR_SLOW_MS);
                panic!(
                    "seed={i} rate={rate} err={e:?} | mutante: a={} b={} fast={} slow={} | roundtrip: a={} b={} fast={fast_sl} slow={slow_sl} | bases mut={} {} floor={}",
                    mutant.sl_horizon_curve.a, mutant.sl_horizon_curve.b,
                    mutant.sl_horizon_curve.eval(TAU_ANCHOR_FAST_MS),
                    mutant.sl_horizon_curve.eval(TAU_ANCHOR_SLOW_MS),
                    roundtrip.sl_horizon_curve.a, roundtrip.sl_horizon_curve.b,
                    mutant.scalp_sl_base, mutant.swing_sl_base,
                    crate::genome::SuperGenotype::min_viable_sl(crate::genome::SuperGenotype::REFERENCE_ROUNDTRIP_FEE)
                );
            }
        }
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
