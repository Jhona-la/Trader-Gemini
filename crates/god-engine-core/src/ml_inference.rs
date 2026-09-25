use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

lazy_static::lazy_static! {
    pub static ref GLOBAL_FORESTS: ArcSwap<HashMap<String, Arc<NanoForest>>> = ArcSwap::from_pointee(HashMap::new());
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NanoForestData {
    pub children_left: Vec<i32>,
    pub children_right: Vec<i32>,
    pub feature: Vec<i32>,
    pub threshold: Vec<f32>,
    pub value: Vec<f32>,
    pub tree_offsets: Vec<i32>,
    pub init_score: f32,
}

#[derive(Clone)]
pub struct NanoForest {
    data: NanoForestData,
    required_features: usize,
}

impl NanoForest {
    /// B3.9-aud — el contrato de dimensión aplica a TODA construcción, no
    /// sólo a `load_model`: `from_data` es la vía de tests y de trainers
    /// in-proc; un modelo más ancho que el binario vivo haría OOB en
    /// `x[feature]`. También se valida topología y finitud antes de activar;
    /// una entrada insuficiente produce ausencia de predicción, no neutralidad.
    pub fn from_data(data: NanoForestData) -> Result<Self, String> {
        let required_features = Self::validate_dim_contract(&data, "<from_data>")?;
        Ok(NanoForest {
            data,
            required_features,
        })
    }

    /// B3.9 — CONTRATO DE DIMENSIÓN del vector ML de inferencia. Un modelo
    /// entrenado con un vector MÁS ANCHO que el de este binario (p.ej.
    /// 48D con splits en dims 44-47 corriendo en un binario 44D) haría
    /// out-of-bounds en `x[feature]`. El cargador RECHAZA cualquier modelo
    /// que parta por una dim ≥ este contrato. El rechazo no acredita
    /// compatibilidad semántica ni autoriza un fallback entre instrumentos.
    pub const ML_VECTOR_DIM: usize = 48;

    /// B3.36 — tasa base del PROPIO modelo: sigmoid(init_score). Con el
    /// etiquetado honesto (HOST-010: SL −sl_pct vs TP +tp_pct, RR≥2) la
    /// base ya NO es ~50% sino ~30% (NEAR ago→sep: base 0.232) — los gates
    /// absolutos (≥0.50 long) quedaban inalcanzables y los espejos del
    /// short siempre abiertos. Todo gate de entrada se expresa ahora como
    /// LIFT sobre ESTA base: p ≥ base+lift (largo), p ≤ base−lift (corto).
    pub fn base_prob(&self) -> f64 {
        1.0 / (1.0 + (-(self.data.init_score as f64)).exp())
    }

    /// P-1b — valor CRUDO del init_score: para modelos de REGRESIÓN
    /// ({SYM}_VOL, {SYM}_VOLU) el init es la MEDIA del label del mes de
    /// entrenamiento (no una probabilidad — sigmoid NO aplica). El freno
    /// de sizing compara el pronóstico contra esta base: unidades exactas.
    pub fn init_value(&self) -> f64 {
        self.data.init_score as f64
    }

    /// Validate topology once, before activation or cache writes. Child indices
    /// are GLOBAL array indices and must stay inside their tree's offset range.
    /// This is structural validity, not a feature-schema or calibration certificate.
    fn validate_dim_contract(data: &NanoForestData, origin: &str) -> Result<usize, String> {
        let fail = |reason: &str| format!("modelo {origin}: {reason}");
        let n = data.value.len();
        if [
            data.children_left.len(),
            data.children_right.len(),
            data.feature.len(),
            data.threshold.len(),
        ]
        .iter()
        .any(|&len| len != n)
        {
            return Err(fail("parallel node arrays have different lengths"));
        }
        if !data.init_score.is_finite()
            || data
                .value
                .iter()
                .chain(&data.threshold)
                .any(|x| !x.is_finite())
        {
            return Err(fail("non-finite model parameter"));
        }
        // Explicit legacy bias-only representation, used by measurement oracles.
        // No tree is traversed. Other empty/degenerate encodings are rejected.
        if n == 0 && data.tree_offsets == [0, 0] {
            return Ok(0);
        }
        if n > i32::MAX as usize
            || data.tree_offsets.len() < 2
            || data.tree_offsets.first() != Some(&0)
            || data.tree_offsets.last().copied() != Some(n as i32)
            || data
                .tree_offsets
                .windows(2)
                .any(|w| w[0] < 0 || w[1] <= w[0])
        {
            return Err(fail("invalid tree offsets"));
        }
        let mut required = 0;
        let mut color = vec![0_u8; n];
        for bounds in data.tree_offsets.windows(2) {
            let (start, end) = (bounds[0] as usize, bounds[1] as usize);
            if end > n {
                return Err(fail("tree offset outside node arrays"));
            }
            for node in start..end {
                let (left, right) = (data.children_left[node], data.children_right[node]);
                if left == -1 && right == -1 {
                    continue;
                }
                if left < 0
                    || right < 0
                    || [left, right]
                        .iter()
                        .any(|&c| (c as usize) < start || (c as usize) >= end)
                {
                    return Err(fail("split has missing or cross-tree child"));
                }
                let feature = data.feature[node];
                if feature < 0 || feature as usize >= Self::ML_VECTOR_DIM {
                    return Err(fail("split feature outside ML_VECTOR_DIM"));
                }
                required = required.max(feature as usize + 1);
            }
            // Iterative DFS avoids recursion-stack overflow and also checks
            // unreachable nodes. Shared subtrees are allowed, cycles are not.
            for root in start..end {
                if color[root] != 0 {
                    continue;
                }
                let mut stack = vec![(root, false)];
                while let Some((node, leaving)) = stack.pop() {
                    if leaving {
                        color[node] = 2;
                        continue;
                    }
                    if color[node] == 1 {
                        return Err(fail("cycle in tree"));
                    }
                    if color[node] == 2 {
                        continue;
                    }
                    color[node] = 1;
                    stack.push((node, true));
                    if data.children_left[node] != -1 {
                        stack.push((data.children_right[node] as usize, false));
                        stack.push((data.children_left[node] as usize, false));
                    }
                }
            }
        }
        Ok(required)
    }

    /// Parsea el JSON fuente y (best-effort) recompila el .bin de caché.
    /// B3.9-aud: la compilación del .bin la decide el CALLER, después de
    /// validar el contrato — un modelo rechazado no debe envenenar la caché.
    fn parse_json(json_path: &str) -> Result<NanoForestData, Box<dyn std::error::Error>> {
        let file = File::open(json_path)?;
        let reader = BufReader::new(file);
        Ok(serde_json::from_reader(reader)?)
    }

    pub fn load_model(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // B3.9-aud — PAREJA (json, bin) EXPLÍCITA. El host (god_engine.rs)
        // carga del directorio models/ TANTO el .json como el .bin (clave =
        // file stem). Si el caller pasa el .bin directamente, el
        // `path.replace(".json", ".bin")` anterior era un no-op y la prueba
        // de frescura comparaba el archivo consigo mismo (nunca stale): un
        // .bin obsoleto podía PISAR la recarga de un .json más nuevo según
        // el orden de read_dir. Ahora el .bin pasado como path se valida
        // contra su .json hermano.
        let (json_path, bin_path) = if path.ends_with(".json") {
            (path.to_string(), path.replace(".json", ".bin"))
        } else if path.ends_with(".bin") {
            (path.replace(".bin", ".json"), path.to_string())
        } else {
            (path.to_string(), path.replace(".json", ".bin"))
        };

        // Verificar frescura: si el JSON es más nuevo que el BIN, el BIN es obsoleto
        let is_stale = match (std::fs::metadata(&json_path), std::fs::metadata(&bin_path)) {
            (Ok(m_json), Ok(m_bin)) => {
                let t_json = m_json
                    .modified()
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                let t_bin = m_bin
                    .modified()
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                t_json > t_bin
            }
            _ => false,
        };

        let (data, from_bin) = if !is_stale && std::path::Path::new(&bin_path).exists() {
            match std::fs::read(&bin_path)
                .map_err(|e| -> Box<dyn std::error::Error> { e.into() })
                .and_then(|bin_data| {
                    bincode::deserialize(&bin_data)
                        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })
                }) {
                Ok(parsed) => (parsed, true),
                Err(_) => (Self::parse_json(&json_path)?, false),
            }
        } else {
            // Fallback to JSON (fresh compile below)
            (Self::parse_json(&json_path)?, false)
        };
        // B3.9 — contrato de dimensión: el modelo debe vivir dentro del
        // vector que ESTE binario construye. Rechazo ruidoso, no silencio.
        // Aplica IGUAL al camino del .bin (caché) — y el .bin sólo se
        // escribe DESPUÉS de validar: un modelo rechazado no contamina la
        // caché para el próximo arranque.
        let required_features = Self::validate_dim_contract(&data, path)
            .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        if !from_bin {
            if let Ok(encoded) = bincode::serialize(&data) {
                let _ = std::fs::write(&bin_path, encoded);
            }
        }
        Ok(NanoForest {
            data,
            required_features,
        })
    }

    /// Loads the forest into the global static cache under a specific key
    pub fn load_global(key: &str, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let forest = Self::load_model(path)?;
        let current_map = crate::ml_inference::GLOBAL_FORESTS.load();
        let mut new_map = (**current_map).clone();
        new_map.insert(key.to_string(), Arc::new(forest));
        crate::ml_inference::GLOBAL_FORESTS.store(Arc::new(new_map));
        Ok(())
    }

    /// Inserta un forest YA CONSTRUIDO en el caché global (mismo contrato de
    /// dimensión que load_model). Uso: oráculos de medición que necesitan un
    /// predictor sintético (t1: siempre-confiado, para medir expresividad
    /// genética condicional a la cooperación de la predicción).
    pub fn store_global(key: &str, forest: NanoForest) {
        let current_map = crate::ml_inference::GLOBAL_FORESTS.load();
        let mut new_map = (**current_map).clone();
        new_map.insert(key.to_string(), Arc::new(forest));
        crate::ml_inference::GLOBAL_FORESTS.store(Arc::new(new_map));
    }

    /// Predicts using a specific global forest.
    /// F5.4: firma Option ⇒ comportamiento Option. El panic anterior mataba el
    /// proceso (panic=abort) con posiciones abiertas si un modelo no estaba
    /// cargado. Sin modelo: None ⇒ el caller decide (neutral 0.5 o no-trade).
    pub fn predict_global(key: &str, features: &[f32]) -> Option<f32> {
        let map = crate::ml_inference::GLOBAL_FORESTS.load();
        map.get(key).and_then(|forest| forest.predict(features))
    }

    /// Fetches a clone of the global forest for hot-path use without RwLock
    pub fn get_global(key: &str) -> Option<Arc<Self>> {
        let map = crate::ml_inference::GLOBAL_FORESTS.load();
        if let Some(forest) = map.get(key) {
            return Some(Arc::clone(forest));
        }
        None
    }

    /// Bounded traversal of a validated tree. No allocation on the prediction path.
    #[inline(always)]
    fn evaluate_tree(&self, features: &[f32], tree_idx: usize) -> Option<f32> {
        let start = *self.data.tree_offsets.get(tree_idx)? as usize;
        let end = *self.data.tree_offsets.get(tree_idx + 1)? as usize;
        let mut node = start;
        for _ in start..end {
            if node < start || node >= end {
                return None;
            }
            let left = self.data.children_left[node];
            let right = self.data.children_right[node];
            if left == -1 && right == -1 {
                return Some(self.data.value[node]);
            }
            let feature = usize::try_from(self.data.feature[node]).ok()?;
            node = usize::try_from(if *features.get(feature)? <= self.data.threshold[node] {
                left
            } else {
                right
            })
            .ok()?;
        }
        None
    }

    /// A valid number is not evidence when required features are absent.
    pub fn predict(&self, features: &[f32]) -> Option<f32> {
        self.predict_raw_checked(features)
            .map(|(_, probability)| probability)
    }

    /// Unified input/numerical contract for classification and regression.
    /// The second component is a logistic transform, not a calibrated
    /// probability for regression targets.
    pub fn predict_raw_checked(&self, features: &[f32]) -> Option<(f32, f32)> {
        if features.is_empty()
            || features.len() < self.required_features
            || features.iter().any(|f| !f.is_finite())
        {
            return None;
        }
        let mut sum = self.data.init_score as f64;
        if !self.data.value.is_empty() {
            for tree in 0..self.data.tree_offsets.len() - 1 {
                sum += self.evaluate_tree(features, tree)? as f64;
            }
        }
        let raw = sum as f32;
        if !sum.is_finite() || !raw.is_finite() {
            return None;
        }
        let probability = (1.0 / (1.0 + (-sum).clamp(-50.0, 50.0).exp())) as f32;
        Some((raw, probability))
    }

    /// Compatibility wrapper. Invalid input is explicitly non-finite, never a
    /// fabricated 0.5 or zero-return forecast. New consumers should use checked.
    pub fn predict_raw(&self, features: &[f32]) -> (f32, f32) {
        self.predict_raw_checked(features)
            .unwrap_or((f32::NAN, f32::NAN))
    }
}

/// B3.4 — BLOQUE MACRO del vector ML (dims 44..48, tras swing 34 ⊕ espectral 10).
///
/// Contrato ÚNICO entre inferencia viva y train_forest: los niveles FRED que
/// el feed vivo (`macro_feed.rs`) ya publica en `omni_features` —
/// DXY=DTWEXBGS [21], SP500 [22], NASDAQ=NASDAQCOM [23], VIX=VIXCLS [24] —
/// y que el trainer junta as-of (cierre t-1) desde `data/macro/*.csv`.
///
/// Las constantes son de UNIDADES (centro/escala típicos de cada serie),
/// no de información: un GBDT es invariante a transformaciones afines
/// monótonas de sus features — los splits se adaptan. Feed ausente o no
/// finito ⇒ 0.0 neutro (misma semántica que el saneo B2.5 del bloque 44D).
pub fn macro_ml_features(omni: &[f64; 54]) -> [f32; 4] {
    let aff = |x: f64, center: f64, scale: f64| -> f32 {
        if x.is_finite() && x > 0.0 {
            ((x - center) / scale) as f32
        } else {
            0.0
        }
    };
    [
        aff(omni[24], 20.0, 10.0),    // VIXCLS — nivel de miedo
        aff(omni[22], 5000.0, 500.0), // SP500 — nivel riesgo global
        // B3.23: ICE DXY (DX-Y.NYB ~99-105) en trainer y vivo — la serie
        // Fed DTWEXBGS (~120) quedó fuera (FRED bloquea la red); paridad
        // por MISMA SERIE en ambos lados del contrato.
        aff(omni[21], 100.0, 5.0),      // ICE DXY — nivel dólar
        aff(omni[23], 18000.0, 2000.0), // NASDAQCOM — nivel tech
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nano_forest_synthetic_prediction() {
        let data = NanoForestData {
            children_left: vec![1, -1, -1],
            children_right: vec![2, -1, -1],
            feature: vec![0, -1, -1],
            threshold: vec![0.5, 0.0, 0.0],
            value: vec![0.0, -0.5, 0.8],
            tree_offsets: vec![0, 3],
            init_score: 0.0,
        };
        let forest = NanoForest::from_data(data).unwrap();

        // Feature 0 <= 0.5 -> leaf value -0.5 -> prob < 0.50
        let prob_low = forest.predict(&[0.2]).unwrap();
        assert!(prob_low < 0.50);

        // Feature 0 > 0.5 -> leaf value 0.8 -> prob > 0.50
        let prob_high = forest.predict(&[0.9]).unwrap();
        assert!(prob_high > 0.50);

        // Empty features -> None
        assert!(forest.predict(&[]).is_none());
    }

    #[test]
    fn test_nano_forest_nan_and_out_of_bounds_features() {
        let data = NanoForestData {
            children_left: vec![1, -1, -1],
            children_right: vec![2, -1, -1],
            feature: vec![5, -1, -1], // Index 5 out of bounds for 1-element input
            threshold: vec![0.5, 0.0, 0.0],
            value: vec![0.0, -0.5, 0.8],
            tree_offsets: vec![0, 3],
            init_score: 0.0,
        };
        let forest = NanoForest::from_data(data).unwrap();

        // NaN feature -> None
        assert!(forest.predict(&[f32::NAN]).is_none());

        // Missing required input is unavailable, not a neutral vote.
        let prob = forest.predict(&[0.2]);
        assert!(prob.is_none());
    }

    #[test]
    fn test_nano_forest_global_cache_and_clone() {
        let data = NanoForestData {
            children_left: vec![1, -1, -1],
            children_right: vec![2, -1, -1],
            feature: vec![0, -1, -1],
            threshold: vec![0.5, 0.0, 0.0],
            value: vec![0.0, -0.5, 0.8],
            tree_offsets: vec![0, 3],
            init_score: 0.0,
        };
        let forest = NanoForest::from_data(data).unwrap();

        // Store directly in GLOBAL_FORESTS
        let current_map = GLOBAL_FORESTS.load();
        let mut new_map = (**current_map).clone();
        new_map.insert("TEST_MODEL".to_string(), Arc::new(forest));
        GLOBAL_FORESTS.store(Arc::new(new_map));

        let retrieved = NanoForest::get_global("TEST_MODEL");
        assert!(retrieved.is_some());

        let prob = NanoForest::predict_global("TEST_MODEL", &[0.8]);
        assert!(prob.is_some());
        assert!(prob.unwrap() > 0.50);

        let non_existent = NanoForest::predict_global("NON_EXISTENT", &[0.8]);
        assert!(non_existent.is_none());
    }

    // ── B3.9-aud: contrato ML_VECTOR_DIM en TODAS las vías ──────────────

    fn synthetic(leaf: f32) -> NanoForestData {
        NanoForestData {
            children_left: vec![1, -1, -1],
            children_right: vec![2, -1, -1],
            feature: vec![0, -1, -1],
            threshold: vec![0.5, 0.0, 0.0],
            value: vec![0.0, leaf, -leaf],
            tree_offsets: vec![0, 3],
            init_score: 0.0,
        }
    }

    fn temp_pair(tag: &str) -> (String, String) {
        let base = std::env::temp_dir()
            .join(format!(
                "tg_b39_{}_{}_{}",
                tag,
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ))
            .to_string_lossy()
            .to_string();
        (format!("{base}.json"), format!("{base}.bin"))
    }

    /// Un modelo que parte por dim ≥ ML_VECTOR_DIM se rechaza SIEMPRE:
    /// from_data, load_model(.json) y load_model(.bin) — incluida la caché.
    #[test]
    fn b39_rechaza_dim_fuera_de_contrato_en_todas_las_vias() {
        let mut wide = synthetic(1.0);
        wide.feature = vec![NanoForest::ML_VECTOR_DIM as i32, -1, -1];

        // from_data (vía de tests/trainers in-proc)
        assert!(NanoForest::from_data(wide.clone()).is_err());

        let (json, bin) = temp_pair("wide");
        std::fs::write(&json, serde_json::to_string(&wide).unwrap()).unwrap();

        // load_model(.json): rechazado Y sin envenenar la caché .bin
        assert!(NanoForest::load_model(&json).is_err());
        assert!(
            !std::path::Path::new(&bin).exists(),
            "un modelo rechazado no debe compilar .bin"
        );

        // Caché .bin manual (simula un bin viejo de un binario anterior):
        // el camino del .bin TAMBIÉN valida el contrato.
        std::fs::write(&bin, bincode::serialize(&wide).unwrap()).unwrap();
        assert!(NanoForest::load_model(&bin).is_err());
        assert!(NanoForest::load_model(&json).is_err());

        // Dim ML_VECTOR_DIM-1: dentro del contrato, aceptado en ambas vías.
        let mut edge = synthetic(1.0);
        edge.feature = vec![(NanoForest::ML_VECTOR_DIM - 1) as i32, -1, -1];
        assert!(NanoForest::from_data(edge.clone()).is_ok());
        let (json_ok, _bin_ok) = temp_pair("edge");
        std::fs::write(&json_ok, serde_json::to_string(&edge).unwrap()).unwrap();
        assert!(NanoForest::load_model(&json_ok).is_ok());
        // y el .bin compilado por esa carga valida en recargas posteriores
        let bin_ok = json_ok.replace(".json", ".bin");
        assert!(NanoForest::load_model(&bin_ok).is_ok());
    }

    /// Leaves may use negative feature sentinels; splits cannot.
    #[test]
    fn b39_leaf_sentinels_are_valid_but_missing_split_metadata_is_rejected() {
        let mut leaf = synthetic(1.0);
        leaf.children_left = vec![-1; 3];
        leaf.children_right = vec![-1; 3];
        leaf.feature = vec![-7; 3];
        assert!(NanoForest::from_data(leaf).is_ok());

        let mut empty = synthetic(1.0);
        empty.feature.clear();
        assert!(NanoForest::from_data(empty).is_err());
        let mut negative = synthetic(1.0);
        negative.feature[0] = -7;
        assert!(NanoForest::from_data(negative).is_err());
    }

    /// Frescura real de la caché: .json más nuevo que el .bin ⇒ se sirve el
    /// .json y se recompila el .bin — incluso cuando el caller pasa el
    /// .bin directo (lo que hace god_engine.rs con models/*.bin).
    #[test]
    fn b39_bin_obsoleto_se_recompila_del_json_nuevo() {
        let (json, bin) = temp_pair("stale");

        // v1: hoja izquierda +2 (p>0.5). Carga ⇒ compila .bin v1.
        std::fs::write(&json, serde_json::to_string(&synthetic(2.0)).unwrap()).unwrap();
        let v1 = NanoForest::load_model(&json).unwrap();
        assert!(v1.predict(&[0.1]).unwrap() > 0.5);
        assert!(std::path::Path::new(&bin).exists());

        // v2 llega después (mtime mayor): hoja izquierda -2 (p<0.5).
        std::thread::sleep(std::time::Duration::from_millis(25));
        std::fs::write(&json, serde_json::to_string(&synthetic(-2.0)).unwrap()).unwrap();

        // Pasando el .json: sirve v2 y recompila el .bin.
        let via_json = NanoForest::load_model(&json).unwrap();
        assert!(
            via_json.predict(&[0.1]).unwrap() < 0.5,
            "json nuevo debe ganar"
        );

        // Pasando el .BIN directo (caso del host): el hermano .json más
        // nuevo también gana — antes el .bin se comparaba consigo mismo y
        // servía v1 obsoleto para siempre.
        std::thread::sleep(std::time::Duration::from_millis(25));
        std::fs::write(&json, serde_json::to_string(&synthetic(2.0)).unwrap()).unwrap();
        let via_bin = NanoForest::load_model(&bin).unwrap();
        assert!(
            via_bin.predict(&[0.1]).unwrap() > 0.5,
            ".bin directo debe respetar el .json hermano más nuevo"
        );
    }

    /// B3.4: el bloque MACRO del vector 48D mapea el omni vivo con la
    /// semántica neutra documentada (sin feed ⇒ 0.0, no NaN).
    #[test]
    fn b34_macro_ml_features_mapea_el_omni_vivo() {
        let mut omni = [0.0f64; 54];
        omni[24] = 30.0; // VIXCLS
        omni[21] = 105.0; // ICE DXY (DX-Y.NYB — B3.23)
        omni[22] = 5500.0; // SP500
        omni[23] = 20000.0; // NASDAQCOM
        let m = macro_ml_features(&omni);
        assert!((m[0] - 1.0).abs() < 1e-6, "VIX (30-20)/10 = +1");
        assert!((m[1] - 1.0).abs() < 1e-6, "SP500 (5500-5000)/500 = +1");
        assert!((m[2] - 1.0).abs() < 1e-6, "ICE DXY (105-100)/5 = +1");
        assert!((m[3] - 1.0).abs() < 1e-6, "NASDAQ (20000-18000)/2000 = +1");

        // Feed ausente / degenerado ⇒ neutro 0.0 (mismo saneo del bloque 44D)
        let cold = macro_ml_features(&[0.0; 54]);
        assert_eq!(cold, [0.0; 4]);
        let mut nan_omni = [0.0f64; 54];
        nan_omni[24] = f64::NAN;
        assert_eq!(macro_ml_features(&nan_omni)[0], 0.0);
    }
}
