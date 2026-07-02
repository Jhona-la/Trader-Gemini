use graph_architecture::scan_workspace;
use std::path::Path;
use warp::Filter;

#[tokio::main]
async fn main() {
    println!("🌌 Iniciando Sistema de Visualización de Grafos 4D...");

    // 1. Escanear el workspace
    let root_dir = Path::new("../../crates"); // Asumiendo que se ejecuta desde crates/graph-architecture
    
    // Ruta alternativa si se ejecuta desde el root de Trader Gemini
    let scan_dir = if Path::new("./crates").exists() {
        Path::new("./crates")
    } else {
        root_dir
    };

    println!("🔍 Escaneando AST en: {:?}", scan_dir);
    let graph = scan_workspace(scan_dir);
    
    let total_nodes = graph.nodes.len();
    let total_edges = graph.edges.len();
    println!("✅ Escaneo completado. Nodos: {} | Aristas: {}", total_nodes, total_edges);

    // 2. Definir la ruta API que retorna el JSON
    let graph_filter = warp::any().map(move || graph.clone());
    let api_route = warp::path("api")
        .and(warp::path("graph"))
        .and(graph_filter)
        .map(|g| warp::reply::json(&g));

    // 3. Servir el index.html estático en la raíz
    // Determinamos donde está el index.html
    let index_path = if Path::new("crates/graph-architecture/index.html").exists() {
        "crates/graph-architecture/index.html"
    } else if Path::new("index.html").exists() {
        "index.html"
    } else {
        // Fallback
        "../../crates/graph-architecture/index.html"
    };

    let static_route = warp::path::end()
        .and(warp::fs::file(index_path));

    let routes = api_route.or(static_route);

    println!("🚀 Servidor 4D Graph corriendo en http://localhost:3030");
    warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;
}
