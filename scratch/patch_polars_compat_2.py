import re

file_path = r'C:\Users\jhona\Documents\Proyectos\Trader Gemini\strategies\ml_strategy.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

content = re.sub(
    r'X\.iloc\[train_idx\]',
    'X[train_idx] if hasattr(X, "select") else X.iloc[train_idx]',
    content
)
content = re.sub(
    r'X\.iloc\[test_idx\]',
    'X[test_idx] if hasattr(X, "select") else X.iloc[test_idx]',
    content
)
content = re.sub(
    r'y\.iloc\[train_idx\]',
    'y[train_idx] if hasattr(y, "to_list") else y.iloc[train_idx]',
    content
)
content = re.sub(
    r'y\.iloc\[test_idx\]',
    'y[test_idx] if hasattr(y, "to_list") else y.iloc[test_idx]',
    content
)
content = re.sub(
    r'X_train\.iloc\[0\]\.values',
    'X_train.to_numpy()[0] if hasattr(X_train, "select") else X_train.iloc[0].values',
    content
)
content = re.sub(
    r'df\[available_cols\]\.iloc\[-60:\]\.values',
    'df.select(available_cols)[-60:].to_numpy() if hasattr(df, "select") else df[available_cols].iloc[-60:].values',
    content
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch 2 applied to ml_strategy.py")
