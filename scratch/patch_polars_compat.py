import re

file_path = r'C:\Users\jhona\Documents\Proyectos\Trader Gemini\strategies\ml_strategy.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace current_row = df.iloc[-1]
content = re.sub(
    r'current_row\s*=\s*df\.iloc\[-1\]',
    'current_row = df[-1].to_dicts()[0] if hasattr(df, "to_dicts") else df.iloc[-1].to_dict()',
    content
)

# 2. Replace df.iloc[-1].get(
content = re.sub(
    r'df\.iloc\[-1\]\.get\(',
    '(df[-1].to_dicts()[0] if hasattr(df, "to_dicts") else df.iloc[-1]).get(',
    content
)

# 3. Replace self._global_feature_cache.iloc[start_idx:idx]
content = re.sub(
    r'self\._global_feature_cache\.iloc\[start_idx:idx\]',
    'self._global_feature_cache[start_idx:idx]',
    content
)

# 4. Replace df.iloc[-int(original_len * 0.6) :]
content = re.sub(
    r'df\.iloc\[-int\(original_len \* 0\.6\) :\]',
    'df[-int(original_len * 0.6):]',
    content
)

# 5. Replace X_pred = df[existing].iloc[[-1]].copy()
content = re.sub(
    r'df\[(.*?)\]\.iloc\[\[-1\]\]\.copy\(\)',
    r'df.select(\1)[-1:] if hasattr(df, "select") else df[\1].iloc[[-1]].copy()',
    content
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied to ml_strategy.py")
