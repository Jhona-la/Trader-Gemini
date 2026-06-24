import os

cargo_path = 'core/rust_engine/Cargo.toml'
with open(cargo_path, 'r', encoding='utf-8') as f:
    text = f.read()

deps = '''hmac = "0.12"
sha2 = "0.10"
hex = "0.4"'''

if 'hmac' not in text:
    text += f"\n{deps}\n"
    with open(cargo_path, 'w', encoding='utf-8') as f:
        f.write(text)
