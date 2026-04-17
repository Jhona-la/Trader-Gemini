from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sys
sys.path.insert(0, '.')
from ml.tree_compiler import compile_rf_to_numpy_batch

try:
    X, y = make_classification(n_samples=50, n_features=10, n_informative=5, random_state=42)
    rf = RandomForestClassifier(n_estimators=1, max_depth=2, random_state=42)
    rf.fit(X, y)
    
    arrays = compile_rf_to_numpy_batch(rf)
    print('Children L:', arrays['children_left'])
    print('Children R:', arrays['children_right'])
    print('Offsets:', arrays['tree_offsets'])
except Exception as e:
    import traceback
    traceback.print_exc()
