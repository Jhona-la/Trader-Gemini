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
}

impl NanoForest {
    /// B3.9-aud — el contrato de dimensión aplica a TODA construcción, no
    /// sólo a `load_model`: `from_data` es la vía de tests y de trainers
    /// in-proc; un modelo más ancho que el binario vivo haría OOB en
    /// `x[feature]` (la protección de `evaluate_tree` lo neutraliza a 0.0,
    /// pero mejor rechazar en la frontera que silenciar en el hot loop).
    pub fn from_data(data: NanoForestData) -> Result<Self, String> {
        Self::validate_dim_contract(&data, "<from_data>")?;
        Ok(NanoForest { data })
    }

    /// B3.9 — CONTRATO DE DIMENSIÓN del vector ML de inferencia. Un modelo
    /// entrenado con un vector MÁS ANCHO que el de este binario (p.ej.
    /// 48D con splits en dims 44-47 corriendo en un binario 44D) haría
    /// out-of-bounds en `x[feature]`. El cargador RECHAZA cualquier modelo
    /// que parta por una dim ≥ este contrato — el fallback a BTCUSDT_SCALP
    /// mantiene el motor vivo. La regresión de binario queda segura.
    pub const ML_VECTOR_DIM: usize = 48;

    /// Valida que ningún split del modelo parta por una dimensión fuera del
    /// vector que ESTE binario construye. Índices negativos (hojas: -1/-2)
    /// y un vector `feature` vacío (modelo sin splits) no violan el
    /// contrato: `evaluate_tree` los trata como hoja, sin acceso a memoria.
    fn validate_dim_contract(data: &NanoForestData, origen: &str) -> Result<(), String> {
        if let Some(&max_feat) = data.feature.iter().filter(|f| **f >= 0).max() {
            if max_feat as usize >= Self::ML_VECTOR_DIM {
                return Err(format!(
                    "modelo {origen} parte por dim {max_feat} ≥ contrato ML_VECTOR_DIM={} — binario obsoleto para este modelo; re-compilar",
                    Self::ML_VECTOR_DIM
                ));
            }
        }
        Ok(())
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
        Self::validate_dim_contract(&data, path).map_err(|e| -> Box<dyn std::error::Error> {
            e.into()
        })?;
        if !from_bin {
            if let Ok(encoded) = bincode::serialize(&data) {
                let _ = std::fs::write(&bin_path, encoded);
            }
        }
        Ok(NanoForest { data })
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

    /// Evaluates a single tree. Returns the leaf value with bounds protection.
    #[inline(always)]
    fn evaluate_tree(&self, features: &[f32], tree_idx: usize) -> f32 {
        let start_node = self.data.tree_offsets[tree_idx] as usize;
        let mut current_node = start_node;

        loop {
            let left_child = self
                .data
                .children_left
                .get(current_node)
                .copied()
                .unwrap_or(-1);
            let right_child = self
                .data
                .children_right
                .get(current_node)
                .copied()
                .unwrap_or(-1);

            if left_child == -1 && right_child == -1 {
                // Leaf node
                return self.data.value.get(current_node).copied().unwrap_or(0.0);
            }

            let feat_idx = self.data.feature.get(current_node).copied().unwrap_or(-1);
            if feat_idx < 0 || (feat_idx as usize) >= features.len() {
                // Out of bounds feature protection: fallback to left leaf or 0.0
                return 0.0;
            }
            let threshold = self
                .data
                .threshold
                .get(current_node)
                .copied()
                .unwrap_or(0.0);

            if features[feat_idx as usize] <= threshold {
                if left_child < 0 {
                    return 0.0;
                }
                current_node = left_child as usize;
            } else {
                if right_child < 0 {
                    return 0.0;
                }
                current_node = right_child as usize;
            }
        }
    }

    /// Predicts the probability for the given features.
    pub fn predict(&self, features: &[f32]) -> Option<f32> {
        if self.data.tree_offsets.len() <= 1 || features.is_empty() {
            return None;
        }
        // FIX #688: Validar finitud de todos los features
        for &f in features {
            if !f.is_finite() {
                return None;
            }
        }

        let n_trees = self.data.tree_offsets.len() - 1;
        let mut sum = self.data.init_score;

        for i in 0..n_trees {
            sum += self.evaluate_tree(features, i);
        }

        // Apply sigmoid with clamping for numerical stability [-50, +50]
        let safe_sum = if sum.is_finite() { sum } else { 0.0 };
        let clamped_sum = (-safe_sum).clamp(-50.0, 50.0);
        let prob = 1.0 / (1.0 + clamped_sum.exp());
        Some(if prob.is_finite() {
            prob.clamp(0.0, 1.0)
        } else {
            0.5
        })
    }

    pub fn predict_raw(&self, features: &[f32]) -> (f32, f32) {
        let n_trees = self.data.tree_offsets.len().saturating_sub(1);
        let mut sum = self.data.init_score;
        for i in 0..n_trees {
            sum += self.evaluate_tree(features, i);
        }
        let safe_sum = if sum.is_finite() { sum } else { 0.0 };
        let clamped_sum = (-safe_sum).clamp(-50.0, 50.0);
        let prob = 1.0 / (1.0 + clamped_sum.exp());
        (sum, prob)
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
        aff(omni[24], 20.0, 10.0), // VIXCLS — nivel de miedo
        aff(omni[22], 5000.0, 500.0), // SP500 — nivel riesgo global
        aff(omni[21], 120.0, 10.0), // DTWEXBGS — nivel dólar
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

        // Out of bounds feature index falls back safely
        let prob = forest.predict(&[0.2]);
        assert!(prob.is_some());
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

    /// Sin features (o sólo índices de hoja negativos) no hay violación de
    /// contrato: son modelos triviales/all-leaves, seguros en evaluate_tree.
    #[test]
    fn b39_modelo_sin_splits_o_con_hojas_negativas_es_valido() {
        let mut no_split = synthetic(1.0);
        no_split.feature = vec![-1, -1, -1];
        let f = NanoForest::from_data(no_split).unwrap();
        assert!(f.predict(&[0.3]).is_some());

        let mut empty = synthetic(1.0);
        empty.feature = Vec::new();
        let f2 = NanoForest::from_data(empty).unwrap();
        // feature vacío ⇒ feat_idx OOB se lee como -1 ⇒ hoja segura.
        assert!(f2.predict(&[0.3]).is_some());

        let mut negative = synthetic(1.0);
        negative.feature = vec![-7, -1, -1];
        assert!(NanoForest::from_data(negative).is_ok());
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
        assert!(via_json.predict(&[0.1]).unwrap() < 0.5, "json nuevo debe ganar");

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
        omni[21] = 130.0; // DTWEXBGS
        omni[22] = 5500.0; // SP500
        omni[23] = 20000.0; // NASDAQCOM
        let m = macro_ml_features(&omni);
        assert!((m[0] - 1.0).abs() < 1e-6, "VIX (30-20)/10 = +1");
        assert!((m[1] - 1.0).abs() < 1e-6, "SP500 (5500-5000)/500 = +1");
        assert!((m[2] - 1.0).abs() < 1e-6, "DXY (130-120)/10 = +1");
        assert!((m[3] - 1.0).abs() < 1e-6, "NASDAQ (20000-18000)/2000 = +1");

        // Feed ausente / degenerado ⇒ neutro 0.0 (mismo saneo del bloque 44D)
        let cold = macro_ml_features(&[0.0; 54]);
        assert_eq!(cold, [0.0; 4]);
        let mut nan_omni = [0.0f64; 54];
        nan_omni[24] = f64::NAN;
        assert_eq!(macro_ml_features(&nan_omni)[0], 0.0);
    }
}
