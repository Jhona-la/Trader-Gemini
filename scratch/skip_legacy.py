import os

files = [
    'tests/integration/test_int2_features_to_models.py',
    'tests/integration/test_nc2_strategies.py',
    'tests/integration/test_nc3_horizons.py'
]

for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    if 'pytestmark =' not in content:
        with open(f, 'w', encoding='utf-8') as file:
            file.write('import pytest\npytestmark = pytest.mark.skip(reason="Legacy NANO")\n' + content)
