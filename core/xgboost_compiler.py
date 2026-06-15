import numpy as np
import xgboost as xgb
from numba import njit, prange

def compile_xgb_to_arrays(booster: xgb.Booster):
    """
    Convierte un modelo XGBoost en arrays planos 1D listos para Numba JIT.
    Retorna: (features, thresholds, left_children, right_children, missing_nodes, values, tree_offsets, base_score)
    """
    df = booster.trees_to_dataframe()
    # Identificadores unicos de arboles
    trees = df['Tree'].unique()
    n_trees = len(trees)
    
    # Asignaremos identificadores globales a cada nodo para aplanar todo
    # La logica original de XGBoost usa ID como string "Tree-Node" ej "0-0"
    # df ya viene ordenado
    
    total_nodes = len(df)
    features = np.zeros(total_nodes, dtype=np.int32)
    thresholds = np.zeros(total_nodes, dtype=np.float32)
    left_children = np.full(total_nodes, -1, dtype=np.int32)
    right_children = np.full(total_nodes, -1, dtype=np.int32)
    missing_nodes = np.full(total_nodes, -1, dtype=np.int32)
    values = np.zeros(total_nodes, dtype=np.float32)
    tree_offsets = np.zeros(n_trees + 1, dtype=np.int32)
    
    node_id_to_idx = {}
    idx = 0
    
    # Primera pasada: Mapear IDs a indices e iterar arboles
    current_tree = -1
    for i, row in df.iterrows():
        node_id_to_idx[row['ID']] = idx
        if row['Tree'] != current_tree:
            current_tree = row['Tree']
            tree_offsets[current_tree] = idx
        idx += 1
    tree_offsets[n_trees] = total_nodes
    
    # Segunda pasada: Llenar arreglos
    # Necesitamos un mapeo del nombre de la feature (ej. "f0", "f1" o el nombre original) a enteros
    feature_names = booster.feature_names
    feat_map = {name: i for i, name in enumerate(feature_names)} if feature_names else {}
    
    for i, row in df.iterrows():
        idx = node_id_to_idx[row['ID']]
        if row['Feature'] == 'Leaf':
            features[idx] = -1
            values[idx] = row['Gain']
        else:
            # Parse feature name
            fname = row['Feature']
            features[idx] = feat_map.get(fname, int(fname.replace('f', '')) if isinstance(fname, str) and fname.startswith('f') else 0)
            thresholds[idx] = row['Split']
            
            left_id = row.get('Yes')
            right_id = row.get('No')
            missing_id = row.get('Missing')
            
            if left_id in node_id_to_idx: left_children[idx] = node_id_to_idx[left_id]
            if right_id in node_id_to_idx: right_children[idx] = node_id_to_idx[right_id]
            if missing_id in node_id_to_idx: missing_nodes[idx] = node_id_to_idx[missing_id]
            
    # Base score es usualmente 0.5 (convertido a logit = 0) o 0
    # Se extrae de config si es posible
    try:
        import json
        config = json.loads(booster.save_config())
        base_score = float(config['learner']['learner_model_param']['base_score'])
    except:
        base_score = 0.5
        
    return features, thresholds, left_children, right_children, missing_nodes, values, tree_offsets, base_score

@njit(fastmath=True, cache=True)
def predict_xgb_jit_single(
    X: np.ndarray,
    features: np.ndarray,
    thresholds: np.ndarray,
    left_children: np.ndarray,
    right_children: np.ndarray,
    missing_nodes: np.ndarray,
    values: np.ndarray,
    tree_offsets: np.ndarray,
    base_score: float
) -> float:
    """
    Inferencia de 1 tick a nivel nanosegundo.
    """
    n_trees = len(tree_offsets) - 1
    # Convertimos base_score a log-odds si no es 0. 
    # Usualmente XGBoost suma base_margin. Asumimos log-odds o base margin directo.
    # En clasificacion binaria, base_score = 0.5 -> logit = 0.0
    # Pero XGBoost a partir de la version 2 almacena logit directamente. Lo manejaremos como suma simple.
    
    score = 0.0
    if base_score != 0.5:
        # Convert to log-odds if it's a probability (XGBoost 1.x)
        # But for safety, we assume score summation first.
        # En XGBoost la inferencia basica es suma de hojas + base_margin.
        pass
        
    for i in range(n_trees):
        node = tree_offsets[i]
        
        while features[node] != -1: # No es hoja
            f_idx = features[node]
            val = X[f_idx]
            
            if np.isnan(val):
                node = missing_nodes[node]
            elif val < thresholds[node]:
                node = left_children[node]
            else:
                node = right_children[node]
                
        score += values[node]
        
    # Sigmoid para prob
    if base_score == 0.5:
        # Logit default
        pass
    else:
        # XGB >= 2.0 uses logit in base_score config often. We'll add base_margin.
        score += np.log(base_score / (1.0 - base_score)) if 0 < base_score < 1 else base_score
        
    if score >= 0:
        return 1.0 / (1.0 + np.exp(-score))
    else:
        exp_s = np.exp(score)
        return exp_s / (1.0 + exp_s)

@njit(fastmath=True, parallel=True, cache=True)
def predict_xgb_jit_batch(
    X_matrix: np.ndarray, # Shape: (n_samples, n_features)
    features: np.ndarray,
    thresholds: np.ndarray,
    left_children: np.ndarray,
    right_children: np.ndarray,
    missing_nodes: np.ndarray,
    values: np.ndarray,
    tree_offsets: np.ndarray,
    base_score: float
) -> np.ndarray:
    """
    Inferencia ultra-masiva vectorizada y paralela para Backtests.
    Procesa millones de filas en milisegundos.
    """
    n_samples = X_matrix.shape[0]
    n_trees = len(tree_offsets) - 1
    
    out = np.zeros(n_samples, dtype=np.float32)
    
    base_margin = 0.0
    if base_score != 0.5:
        if 0 < base_score < 1:
            base_margin = np.log(base_score / (1.0 - base_score))
        else:
            base_margin = base_score
            
    for j in prange(n_samples):
        score = base_margin
        X = X_matrix[j]
        
        for i in range(n_trees):
            node = tree_offsets[i]
            
            while features[node] != -1:
                f_idx = features[node]
                val = X[f_idx]
                
                if np.isnan(val):
                    node = missing_nodes[node]
                elif val < thresholds[node]:
                    node = left_children[node]
                else:
                    node = right_children[node]
                    
            score += values[node]
            
        if score >= 0:
            out[j] = 1.0 / (1.0 + np.exp(-score))
        else:
            exp_s = np.exp(score)
            out[j] = exp_s / (1.0 + exp_s)
            
    return out
