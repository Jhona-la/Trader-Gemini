use std::path::Path;
use std::fs;

fn main() {
    println!("============================================================");
    println!("🌌 TRADER GEMINI V5 - 4D GRAPH ARCHITECTURE RENDERER");
    println!("============================================================");
    
    let root = Path::new("crates");
    
    if !root.exists() {
        eprintln!("❌ Error: Directorio 'crates' no encontrado.");
        return;
    }
    
    println!("🔍 Analizando AST (Abstract Syntax Tree) de Rust...");
    
    let graph = graph_architecture::scan_workspace(root);
    
    println!("✅ AST escaneado con éxito.");
    
    fs::create_dir_all("dashboard").unwrap();
    let out_path = "dashboard/graph.json";
    
    let json = serde_json::to_string_pretty(&graph).unwrap();
    fs::write(out_path, json).unwrap();
    
    println!("📂 Topología del sistema exportada a: {}", out_path);
    println!("📊 Nodos detectados: {}", graph.nodes.len());
    println!("🔗 Conexiones detectadas: {}", graph.edges.len());
    println!("🚀 Listo para visualización Web/WASM.");
}
