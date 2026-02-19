"""
🕵️ ARCHITECTURE AUDIT TOOL: Circular Dependency Detector
Scans project for import cycles.
Usage: python scripts/detect_cycles.py
"""
import ast
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

TARGET_DIRS = ['core', 'strategies', 'execution', 'risk', 'utils', 'data']

def get_module_name(file_path):
    rel_path = os.path.relpath(file_path, PROJECT_ROOT)
    return rel_path.replace(os.path.sep, '.').replace('.py', '')

def get_imports(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=file_path)
        except SyntaxError:
            return []

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports

def build_graph():
    graph = defaultdict(set)
    for d in TARGET_DIRS:
        path = os.path.join(PROJECT_ROOT, d)
        if not os.path.exists(path): continue
        
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.py') and file != '__init__.py':
                    full_path = os.path.join(root, file)
                    module = get_module_name(full_path)
                    
                    for imp in get_imports(full_path):
                        # Filter only internal imports
                        parts = imp.split('.')
                        if parts[0] in TARGET_DIRS:
                            graph[module].add(imp)
    return graph

def find_cycles(graph):
    visited = set()
    stack = []
    cycles = []

    def dfs(node, path):
        if node in path:
            cycle = path[path.index(node):] + [node]
            cycles.append(cycle)
            return
        
        if node in visited:
            return

        visited.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, []):
            # Check if neighbor is a known internal module (simplified)
            # Graph keys are full module paths. Imports might be packages.
            # We match if import is prefix of key or key is prefix of import?
            # Exact match is hard. Let's try exact match first.
            if neighbor in graph:
                dfs(neighbor, path)
            else:
                # Try simple matching
                pass 
                
        path.pop()

    for node in list(graph.keys()):
        dfs(node, [])
        
    return cycles

if __name__ == '__main__':
    print("🔍 Scanning for Circular Dependencies...")
    graph = build_graph()
    print(f"   Analyzed {len(graph)} modules.")
    
    cycles = find_cycles(graph)
    if cycles:
        print(f"\nBS!! Found {len(cycles)} potential cycles (showing top 5):")
        for i, c in enumerate(cycles[:5]):
            print(f"   Cycle {i+1}: {' -> '.join(c)}")
    else:
        print("\n✅ No simple import cycles found.")
