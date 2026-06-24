import os
import sys

# Add the Rust target/release folder so Python can load quantum_engine.dll
rust_dll_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rust_engine", "target", "release"))
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(rust_dll_dir)
else:
    os.environ["PATH"] = rust_dll_dir + os.pathsep + os.environ["PATH"]
