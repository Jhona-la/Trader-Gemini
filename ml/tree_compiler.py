import numpy as np

def compile_rf_to_numpy_batch(rf_model):
    """
    Extracs Scikit-learn RandomForestClassifier trees to flat NumPy arrays
    suitable for nano-latency Numba JIT traversal.
    """
    n_trees = len(rf_model.estimators_)
    
    # Pre-allocate lists
    children_left_list = []
    children_right_list = []
    feature_list = []
    threshold_list = []
    value_list = []
    tree_offsets = np.zeros(n_trees + 1, dtype=np.int32)
    
    current_offset = 0
    for i, estimator in enumerate(rf_model.estimators_):
        tree = estimator.tree_
        
        # Offset children indices so they align in the concatenated array
        c_left = tree.children_left.copy()
        c_right = tree.children_right.copy()
        
        c_left[c_left != -1] += current_offset
        c_right[c_right != -1] += current_offset
        
        children_left_list.append(c_left)
        children_right_list.append(c_right)
        feature_list.append(tree.feature)
        threshold_list.append(tree.threshold)
        
        vals = tree.value[:, 0, :]
        row_sums = vals.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1 # prevent division by zero
        probs = vals / row_sums
        
        if probs.shape[1] > 1:
            value_list.append(probs[:, 1].astype(np.float32)) # Class 1 (positive) prob
        else:
            value_list.append(probs[:, 0].astype(np.float32))

        tree_offsets[i] = current_offset
        current_offset += tree.node_count
        
    tree_offsets[n_trees] = current_offset
    
    return {
        'children_left': np.concatenate(children_left_list).astype(np.int32),
        'children_right': np.concatenate(children_right_list).astype(np.int32),
        'feature': np.concatenate(feature_list).astype(np.int32),
        'threshold': np.concatenate(threshold_list).astype(np.float32),
        'value': np.concatenate(value_list).astype(np.float32),
        'tree_offsets': tree_offsets
    }

def compile_gb_to_numpy_batch(gb_model):
    """
    Extracs Scikit-learn GradientBoostingClassifier trees to flat NumPy arrays.
    """
    n_estimators = gb_model.n_estimators
    init_estimator = gb_model.init_
    
    # Determine the initial score from the prior/init estimator
    if hasattr(init_estimator, 'class_prior_'):
        prior = init_estimator.class_prior_[1]  # positive class
        init_score = np.log(prior / (1.0 - prior))
    elif hasattr(init_estimator, 'prior'):
        prior = init_estimator.prior
        init_score = np.log(prior / (1.0 - prior))
    else:
        init_score = 0.0

    learning_rate = gb_model.learning_rate
    
    # Pre-allocate lists
    children_left_list = []
    children_right_list = []
    feature_list = []
    threshold_list = []
    value_list = []
    tree_offsets = np.zeros(n_estimators + 1, dtype=np.int32)
    
    current_offset = 0
    # GB trees for binary classification are typically in .estimators_[:, 0]
    for i in range(n_estimators):
        estimator = gb_model.estimators_[i, 0]
        tree = estimator.tree_
        
        c_left = tree.children_left.copy()
        c_right = tree.children_right.copy()
        
        c_left[c_left != -1] += current_offset
        c_right[c_right != -1] += current_offset
        
        children_left_list.append(c_left)
        children_right_list.append(c_right)
        feature_list.append(tree.feature)
        threshold_list.append(tree.threshold)

        # Scikit-learn GradientBoostingRegressor trees store the value directly in tree.value[:, 0, 0]
        # and we multiply by the learning rate.
        val = tree.value[:, 0, 0] * learning_rate
        value_list.append(val.astype(np.float32))

        tree_offsets[i] = current_offset
        current_offset += tree.node_count
        
    tree_offsets[n_estimators] = current_offset
    
    return {
        'children_left': np.concatenate(children_left_list).astype(np.int32),
        'children_right': np.concatenate(children_right_list).astype(np.int32),
        'feature': np.concatenate(feature_list).astype(np.int32),
        'threshold': np.concatenate(threshold_list).astype(np.float32),
        'value': np.concatenate(value_list).astype(np.float32),
        'tree_offsets': tree_offsets,
        'init_score': np.float32(init_score)
    }

def export_forest_to_json(numpy_batch, filepath):
    """
    Exports the compiled numpy arrays into a JSON file for the Rust ML Inference Engine.
    """
    import json
    import os
    
    data = {
        'children_left': numpy_batch['children_left'].tolist(),
        'children_right': numpy_batch['children_right'].tolist(),
        'feature': numpy_batch['feature'].tolist(),
        'threshold': numpy_batch['threshold'].tolist(),
        'value': numpy_batch['value'].tolist(),
        'tree_offsets': numpy_batch['tree_offsets'].tolist(),
        'init_score': float(numpy_batch.get('init_score', 0.0))
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f)
    
    print(f"[AOT Compiler] Exported {len(numpy_batch['tree_offsets']) - 1} trees to {filepath}")
