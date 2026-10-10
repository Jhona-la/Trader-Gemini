//! QS-2 — LINAJE DE MODELOS Y LIBRO DE USOS DEL HOLDOUT FINAL.
//!
//! # Qué faltaba
//!
//! `config_dir/models_manifest.json` (XLVIII·G / MR) es un INVENTARIO: hash y
//! estructura de lo que hay en `models/`. No dice con qué tapes, qué
//! argumentos, qué versión del código ni qué evidencia de gate se produjo
//! cada modelo. `docs/audit/PROTOCOLO_OOS_R4_2026-10-07.md` §3 lo deja
//! abierto: «no registra hashes de tapes, fechas de entrenamiento/selección/
//! test ni decisiones de promoción … se requiere ledger de usos/consultas,
//! intervalo final no usado, hashes, cobertura temporal por activo». Y el
//! propio entrenador lo declara: «esto no controla reutilización del holdout
//! entre ejecuciones».
//!
//! # Qué garantiza
//!
//! 1. **Manifiesto por modelo** (`{KEY}.manifest`, JSON) escrito por el
//!    entrenador junto al modelo, con el sha256 de los bytes EXACTOS del
//!    modelo, cada tape de entrada (rol, ruta, tamaño, sha256, cobertura
//!    temporal de las muestras), la transformación (argumentos, etiqueta,
//!    horizonte, dimensión, árboles), el commit del código, la frontera de
//!    evidencia y la evidencia del gate (selección y test). La extensión NO
//!    es `.json` a propósito: el recargador del host
//!    (`model_reload`), la cobertura del roster (`ml_coverage`) y el
//!    inventario (`ml_registry`) sólo leen `*.json` / `*.bin`.
//!    [`verify_model_lineage`] detecta un modelo editado tras entrenarse.
//!
//! 2. **Libro de usos del holdout** (JSONL de sólo-añadir): cada vez que un
//!    tape de test JUZGA un artefacto se anota (símbolo, objetivo, sha256 del
//!    test, sha256 del modelo, veredicto). Un holdout sólo es «final» para
//!    UNA decisión: si ya juzgó OTRO artefacto del mismo símbolo y objetivo,
//!    lo que se ve en el test ya informó la elección (otro horizonte, otros
//!    hiperparámetros, otro tape de train) y deja de ser fuera de muestra
//!    para la promoción. [`check_holdout_fresh`] lo rechaza salvo motivo
//!    explícito, que queda escrito en el manifiesto. Re-juzgar el MISMO
//!    artefacto (mismo sha256: el entrenador es determinista, semilla fija)
//!    no añade selección y se permite — es el flujo «candidato, luego
//!    `--promote`».
//!
//! # Qué NO garantiza
//!
//! Que el modelo tenga edge: se registra la evidencia, no se certifica. El
//! libro sólo ve las corridas que pasaron por él (un test consultado a mano
//! fuera del entrenador no queda anotado). Los modelos anteriores no tienen
//! manifiesto y se reportan como [`Lineage::Missing`], no como fallo.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Versión del esquema del manifiesto.
pub const MANIFEST_SCHEMA: u32 = 1;
/// Extensión del manifiesto (NO `.json`: ver el doc del módulo).
pub const MANIFEST_EXT: &str = "manifest";
/// Ruta por defecto del libro de usos del holdout (versionable).
pub const DEFAULT_HOLDOUT_LEDGER: &str = "config_dir/holdout_ledger.jsonl";

/// Un fichero de entrada del entrenamiento.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct InputRecord {
    /// `train`, `train+selection` (split interno), `selection` o `test`.
    pub role: String,
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
    /// Primera y última marca de tiempo (ms) de las muestras extraídas.
    pub first_ts_ms: Option<u64>,
    pub last_ts_ms: Option<u64>,
    /// Muestras extraídas del tape (antes de purgar).
    pub samples: Option<usize>,
}

/// Evidencia del gate con la que se decidió el destino del modelo.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GateRecord {
    /// `logloss` (clasificación) o `MSE` (regresión).
    pub metric: String,
    pub margin: f64,
    pub selection_loss: f64,
    /// Baseline del gate: tasa base (clasificación) o PERSISTENCIA (regresión).
    pub selection_baseline: f64,
    pub selection_pass: bool,
    pub test_loss: Option<f64>,
    pub test_baseline: Option<f64>,
    pub test_pass: Option<bool>,
}

/// Manifiesto de linaje de un modelo.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ModelManifest {
    pub schema: u32,
    /// Nombre del fichero del modelo (sin directorio).
    pub model_file: String,
    /// sha256 (hex) de los bytes exactos del modelo.
    pub model_sha256: String,
    pub symbol: String,
    pub label_mode: String,
    pub horizon_ms: u64,
    pub feature_dim: usize,
    pub n_trees: usize,
    pub trainer: String,
    pub trainer_args: Vec<String>,
    /// Commit de git del código que entrenó (`+dirty` si había cambios).
    pub code_version: Option<String>,
    /// Fecha UTC RFC 3339.
    pub created_utc: String,
    pub inputs: Vec<InputRecord>,
    /// Filas de train tras purgar el solape con la selección.
    pub train_rows: usize,
    pub selection_rows: usize,
    /// Fin de la información usada para ajustar y seleccionar: última
    /// muestra de selección + τ. El test debe empezar DESPUÉS.
    pub evidence_end_ms: u64,
    pub gate: GateRecord,
    /// Veces que el test ya había juzgado OTROS artefactos del mismo
    /// símbolo y objetivo antes de esta corrida (0 = holdout fresco).
    pub holdout_prior_other_artifacts: usize,
    /// Motivo declarado para promover pese a un holdout ya usado.
    pub holdout_reuse_reason: Option<String>,
    pub promoted: bool,
}

/// Resultado de comparar un modelo con su manifiesto.
#[derive(Debug, Clone, PartialEq)]
pub enum Lineage {
    /// El manifiesto existe y el sha256 del modelo coincide.
    Verified(ModelManifest),
    /// No hay manifiesto (modelo anterior a QS-2 o escrito a mano).
    Missing,
    /// El modelo no es el que se entrenó: su contenido cambió.
    Mismatch { expected: String, actual: String },
    /// El manifiesto o el modelo no se pudieron leer o interpretar.
    Invalid(String),
}

/// Una consulta del holdout final: qué artefacto juzgó y con qué veredicto.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HoldoutUse {
    pub utc: String,
    pub symbol: String,
    pub label_mode: String,
    pub test_sha256: String,
    pub model_sha256: String,
    pub test_pass: bool,
    pub promoted: bool,
    pub code_version: Option<String>,
}

/// Ruta del manifiesto de un modelo: `models/X.json` → `models/X.manifest`.
pub fn manifest_path_for(model_path: &Path) -> PathBuf {
    model_path.with_extension(MANIFEST_EXT)
}

/// sha256 en hexadecimal de unos bytes.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Tamaño y sha256 de un fichero, leído por bloques (los tapes pesan GB).
pub fn sha256_file(path: &Path) -> std::io::Result<(u64, String)> {
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((total, hex::encode(hasher.finalize())))
}

/// Registro de una entrada: hash del fichero y cobertura de sus muestras.
pub fn input_record(role: &str, path: &str, sample_ts: &[u64]) -> std::io::Result<InputRecord> {
    let (bytes, sha256) = sha256_file(Path::new(path))?;
    Ok(InputRecord {
        role: role.to_string(),
        path: path.to_string(),
        bytes,
        sha256,
        first_ts_ms: sample_ts.first().copied(),
        last_ts_ms: sample_ts.last().copied(),
        samples: Some(sample_ts.len()),
    })
}

/// Commit del código: `GIT_COMMIT` si está definido; si no, `git rev-parse
/// HEAD` con sufijo `+dirty` cuando el árbol tiene cambios. `None` si no hay
/// git (el manifiesto lo deja explícito en vez de inventarlo).
pub fn code_version() -> Option<String> {
    if let Ok(c) = std::env::var("GIT_COMMIT") {
        let c = c.trim();
        if !c.is_empty() {
            return Some(c.to_string());
        }
    }
    let head = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let sha = String::from_utf8_lossy(&head.stdout).trim().to_string();
    if sha.is_empty() {
        return None;
    }
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    Some(if dirty { format!("{sha}+dirty") } else { sha })
}

/// Escribe `bytes` en `path` de forma atómica (temporal + `sync_all` +
/// rename): en disco está siempre el fichero viejo completo o el nuevo, y
/// un lector concurrente (el watcher del host) nunca ve uno a medias.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

/// Escribe el modelo y DESPUÉS su manifiesto, con el sha256 de los bytes
/// realmente escritos (los `schema`, `model_file` y `model_sha256` que
/// traiga `manifest` se sustituyen).
pub fn write_model_with_manifest(
    model_path: &Path,
    model_bytes: &[u8],
    mut manifest: ModelManifest,
) -> std::io::Result<ModelManifest> {
    write_atomic(model_path, model_bytes)?;
    manifest.schema = MANIFEST_SCHEMA;
    manifest.model_file = model_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string();
    manifest.model_sha256 = sha256_hex(model_bytes);
    let json = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    write_atomic(&manifest_path_for(model_path), &json)?;
    Ok(manifest)
}

/// Compara el modelo presente en `model_path` con su manifiesto.
pub fn verify_model_lineage(model_path: &Path) -> Lineage {
    let mpath = manifest_path_for(model_path);
    let raw = match std::fs::read(&mpath) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Lineage::Missing,
        Err(e) => return Lineage::Invalid(format!("{}: {e}", mpath.display())),
    };
    let manifest: ModelManifest = match serde_json::from_slice(&raw) {
        Ok(m) => m,
        Err(e) => return Lineage::Invalid(format!("{}: {e}", mpath.display())),
    };
    if manifest.schema != MANIFEST_SCHEMA {
        return Lineage::Invalid(format!(
            "{}: esquema {} desconocido (se esperaba {MANIFEST_SCHEMA})",
            mpath.display(),
            manifest.schema
        ));
    }
    let model = match std::fs::read(model_path) {
        Ok(b) => b,
        Err(e) => return Lineage::Invalid(format!("{}: {e}", model_path.display())),
    };
    let actual = sha256_hex(&model);
    if actual != manifest.model_sha256 {
        return Lineage::Mismatch {
            expected: manifest.model_sha256,
            actual,
        };
    }
    Lineage::Verified(manifest)
}

/// Lee el libro de usos del holdout. Sin fichero ⇒ libro vacío. Una línea
/// ilegible es un error (con su número): un libro corrupto no puede
/// certificar que un holdout está fresco.
pub fn read_ledger(path: &Path) -> Result<Vec<HoldoutUse>, String> {
    let raw = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    raw.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            serde_json::from_str::<HoldoutUse>(l)
                .map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))
        })
        .collect()
}

/// Añade una consulta al libro (una línea JSON, `sync_all`).
pub fn append_ledger(path: &Path, entry: &HoldoutUse) -> std::io::Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let mut line = serde_json::to_string(entry)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    line.push('\n');
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    f.write_all(line.as_bytes())?;
    f.sync_all()
}

/// Cuántas veces este test ya juzgó OTROS artefactos de la misma decisión
/// (símbolo + objetivo). El horizonte NO entra en la clave: probar otro τ
/// contra el mismo test es justamente elegir con el test.
pub fn prior_other_artifacts(
    ledger: &[HoldoutUse],
    symbol: &str,
    label_mode: &str,
    test_sha256: &str,
    model_sha256: &str,
) -> usize {
    ledger
        .iter()
        .filter(|u| {
            u.symbol == symbol
                && u.label_mode == label_mode
                && u.test_sha256 == test_sha256
                && u.model_sha256 != model_sha256
        })
        .count()
}

/// Contrato de promoción sobre el holdout: fresco (0 artefactos previos) o
/// motivo explícito no vacío. Devuelve el motivo aceptado, si lo hubo.
pub fn check_holdout_fresh(prior: usize, reuse_reason: &str) -> Result<Option<String>, String> {
    if prior == 0 {
        return Ok(None);
    }
    let reason = reuse_reason.trim();
    if reason.is_empty() {
        return Err(format!(
            "el test ya juzgó {prior} artefacto(s) distinto(s) del mismo símbolo y objetivo: \
             dejó de ser fuera de muestra para esta promoción. Usa un tramo posterior nuevo, \
             o declara el motivo con --test-reuse \"<motivo>\" (queda en el manifiesto)"
        ));
    }
    Ok(Some(reason.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir_temporal(nombre: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let d = std::env::temp_dir().join(format!("qs2_{nombre}_{}_{nanos}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn manifiesto_base() -> ModelManifest {
        ModelManifest {
            schema: 0,
            model_file: String::new(),
            model_sha256: String::new(),
            symbol: "ATOMUSDT".into(),
            label_mode: "dir".into(),
            horizon_ms: 300_000,
            feature_dim: 48,
            n_trees: 3,
            trainer: "train_forest".into(),
            trainer_args: vec!["ATOMUSDT".into(), "--promote".into()],
            code_version: Some("abc123".into()),
            created_utc: "2026-10-10T00:00:00Z".into(),
            inputs: vec![],
            train_rows: 100,
            selection_rows: 25,
            evidence_end_ms: 1_000,
            gate: GateRecord {
                metric: "logloss".into(),
                margin: 0.001,
                selection_loss: 0.60,
                selection_baseline: 0.61,
                selection_pass: true,
                test_loss: Some(0.605),
                test_baseline: Some(0.61),
                test_pass: Some(true),
            },
            holdout_prior_other_artifacts: 0,
            holdout_reuse_reason: None,
            promoted: true,
        }
    }

    fn uso(symbol: &str, label: &str, test: &str, model: &str) -> HoldoutUse {
        HoldoutUse {
            utc: "2026-10-10T00:00:00Z".into(),
            symbol: symbol.into(),
            label_mode: label.into(),
            test_sha256: test.into(),
            model_sha256: model.into(),
            test_pass: false,
            promoted: false,
            code_version: None,
        }
    }

    /// El manifiesto va junto al modelo con una extensión que ninguno de los
    /// tres lectores de `models/` (recarga, roster, inventario) toma por modelo.
    #[test]
    fn qs2_el_manifiesto_no_es_un_modelo_cargable() {
        let p = manifest_path_for(Path::new("models/ATOMUSDT_MOTOR.json"));
        assert_eq!(p, PathBuf::from("models/ATOMUSDT_MOTOR.manifest"));
        let ext = p.extension().and_then(|s| s.to_str());
        assert!(ext != Some("json") && ext != Some("bin"));
        let c = manifest_path_for(Path::new("models/ATOMUSDT_MOTOR_CANDIDATE.json"));
        assert_eq!(c, PathBuf::from("models/ATOMUSDT_MOTOR_CANDIDATE.manifest"));
    }

    /// Escribir y verificar: el hash es el de los bytes escritos, el
    /// manifiesto sobrevive el viaje de ida y vuelta y no quedan temporales.
    #[test]
    fn qs2_escribir_y_verificar() {
        let d = dir_temporal("roundtrip");
        let modelo = d.join("models").join("ATOMUSDT_MOTOR.json");
        let bytes = br#"{"init_score":0.0}"#;
        let escrito = write_model_with_manifest(&modelo, bytes, manifiesto_base()).unwrap();
        assert_eq!(escrito.schema, MANIFEST_SCHEMA);
        assert_eq!(escrito.model_file, "ATOMUSDT_MOTOR.json");
        assert_eq!(escrito.model_sha256, sha256_hex(bytes));
        assert_eq!(std::fs::read(&modelo).unwrap(), bytes);
        match verify_model_lineage(&modelo) {
            Lineage::Verified(m) => assert_eq!(m, escrito),
            otro => panic!("se esperaba Verified, salió {otro:?}"),
        }
        let restos: Vec<_> = std::fs::read_dir(d.join("models"))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(restos.is_empty(), "quedaron temporales: {restos:?}");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Un modelo modificado después de entrenarse deja de coincidir.
    #[test]
    fn qs2_un_modelo_editado_no_coincide() {
        let d = dir_temporal("mismatch");
        let modelo = d.join("ATOMUSDT_MOTOR.json");
        write_model_with_manifest(&modelo, b"{\"a\":1}", manifiesto_base()).unwrap();
        std::fs::write(&modelo, b"{\"a\":2}").unwrap();
        match verify_model_lineage(&modelo) {
            Lineage::Mismatch { expected, actual } => {
                assert_eq!(expected, sha256_hex(b"{\"a\":1}"));
                assert_eq!(actual, sha256_hex(b"{\"a\":2}"));
            }
            otro => panic!("se esperaba Mismatch, salió {otro:?}"),
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Sin manifiesto: modelo legado, se reporta como ausente, no como fallo.
    /// Manifiesto corrupto o de otro esquema: inválido.
    #[test]
    fn qs2_ausente_e_invalido() {
        let d = dir_temporal("missing");
        let modelo = d.join("X.json");
        std::fs::write(&modelo, b"{}").unwrap();
        assert_eq!(verify_model_lineage(&modelo), Lineage::Missing);
        std::fs::write(manifest_path_for(&modelo), b"no es json").unwrap();
        assert!(matches!(verify_model_lineage(&modelo), Lineage::Invalid(_)));
        let mut m = manifiesto_base();
        m.schema = 99;
        m.model_sha256 = sha256_hex(b"{}");
        std::fs::write(manifest_path_for(&modelo), serde_json::to_vec(&m).unwrap()).unwrap();
        assert!(matches!(verify_model_lineage(&modelo), Lineage::Invalid(_)));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// El hash por bloques (más de un bloque de 1 MiB) coincide con el de
    /// sus bytes, y el registro lleva rol, tamaño y cobertura temporal.
    #[test]
    fn qs2_hash_de_fichero_por_bloques_y_cobertura() {
        let d = dir_temporal("hash");
        let p = d.join("tape.bin");
        let datos: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&p, &datos).unwrap();
        let (n, h) = sha256_file(&p).unwrap();
        assert_eq!(n, datos.len() as u64);
        assert_eq!(h, sha256_hex(&datos));
        let r = input_record("test", p.to_str().unwrap(), &[10, 20, 30]).unwrap();
        assert_eq!((r.role.as_str(), r.bytes), ("test", datos.len() as u64));
        assert_eq!((r.first_ts_ms, r.last_ts_ms, r.samples), (Some(10), Some(30), Some(3)));
        let vacio = input_record("train", p.to_str().unwrap(), &[]).unwrap();
        assert_eq!((vacio.first_ts_ms, vacio.last_ts_ms, vacio.samples), (None, None, Some(0)));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// El libro: sin fichero está vacío; las consultas se añaden y se leen en
    /// orden; una línea corrupta es un error con su número de línea.
    #[test]
    fn qs2_libro_de_usos_ida_y_vuelta() {
        let d = dir_temporal("ledger");
        let libro = d.join("config_dir").join("holdout_ledger.jsonl");
        assert_eq!(read_ledger(&libro).unwrap(), vec![]);
        let a = uso("ATOMUSDT", "dir", "t1", "m1");
        let b = uso("ATOMUSDT", "dir", "t1", "m2");
        append_ledger(&libro, &a).unwrap();
        append_ledger(&libro, &b).unwrap();
        assert_eq!(read_ledger(&libro).unwrap(), vec![a, b]);
        let mut f = std::fs::OpenOptions::new().append(true).open(&libro).unwrap();
        f.write_all(b"{roto\n").unwrap();
        let err = read_ledger(&libro).unwrap_err();
        assert!(err.contains("holdout_ledger.jsonl:3:"), "{err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// La clave de la decisión es símbolo + objetivo + test. Re-juzgar el
    /// MISMO artefacto no cuenta; otro artefacto sí, aunque cambie τ; otro
    /// símbolo, otro objetivo u otro test no cuentan.
    #[test]
    fn qs2_reuso_del_holdout_por_decision() {
        let libro = vec![
            uso("ATOMUSDT", "dir", "t1", "m1"),
            uso("ATOMUSDT", "dir", "t1", "m1"),
            uso("ATOMUSDT", "dir", "t1", "m2"),
            uso("BNBUSDT", "dir", "t1", "m3"),
            uso("ATOMUSDT", "vol", "t1", "m4"),
            uso("ATOMUSDT", "dir", "t2", "m5"),
        ];
        assert_eq!(prior_other_artifacts(&libro, "ATOMUSDT", "dir", "t1", "m1"), 1);
        assert_eq!(prior_other_artifacts(&libro, "ATOMUSDT", "dir", "t1", "m2"), 2);
        assert_eq!(prior_other_artifacts(&libro, "ATOMUSDT", "dir", "t1", "m9"), 3);
        assert_eq!(prior_other_artifacts(&libro, "ATOMUSDT", "dir", "t3", "m9"), 0);
        assert_eq!(prior_other_artifacts(&libro, "ATOMUSDT", "vol", "t1", "m4"), 0);
    }

    /// Promoción: holdout fresco pasa; usado exige motivo no vacío, que se
    /// devuelve para quedar escrito en el manifiesto.
    #[test]
    fn qs2_contrato_de_promocion_sobre_el_holdout() {
        assert_eq!(check_holdout_fresh(0, ""), Ok(None));
        assert_eq!(check_holdout_fresh(0, "ignorado"), Ok(None));
        assert!(check_holdout_fresh(1, "").is_err());
        assert!(check_holdout_fresh(2, "   ").is_err());
        assert_eq!(
            check_holdout_fresh(1, " sólo cambió la serialización "),
            Ok(Some("sólo cambió la serialización".into()))
        );
    }
}
