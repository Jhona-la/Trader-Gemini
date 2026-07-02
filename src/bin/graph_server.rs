use warp::Filter;
use std::path::Path;
use graph_architecture::scan_workspace;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    println!("============================================================");
    println!("👁️ EL PANÓPTICO CUÁNTICO (GRAFO 4D) - INICIANDO...");
    println!("============================================================");

    let root_path = Path::new("crates");
    println!("🔍 Escaneando AST del workspace en {:?}...", root_path);
    let graph = scan_workspace(root_path);
    let graph_arc = Arc::new(graph);

    let api_graph = graph_arc.clone();
    let api_route = warp::path!("api" / "graph")
        .map(move || {
            let mut current_graph = (*api_graph).clone();
            let aggregator = telemetry_server::profiler::GLOBAL_AGGREGATOR.load();
            let averages = aggregator.get_averages();
            for node in current_graph.nodes.values_mut() {
                // Try to match node label or id with telemetry keys
                if let Some(&lat) = averages.get(node.label.as_str()) {
                    node.average_latency_ns = lat;
                }
            }
            warp::reply::json(&current_graph)
        });

    let html_content = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Panóptico Cuántico - Trader Gemini V5</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
    <style>
        body { margin: 0; padding: 0; background-color: #0f172a; color: #f8fafc; font-family: 'Inter', sans-serif; }
        #cy { width: 100vw; height: 100vh; display: block; }
        #info { position: absolute; top: 10px; left: 10px; background: rgba(15, 23, 42, 0.9); padding: 15px; border-radius: 8px; border: 1px solid #334155; pointer-events: none; z-index: 10; }
        h1 { margin: 0 0 10px 0; font-size: 1.2rem; color: #38bdf8; }
        .stat { font-size: 0.9rem; margin-bottom: 5px; color: #94a3b8; }
    </style>
</head>
<body>
    <div id="info">
        <h1>Panóptico Cuántico (AST)</h1>
        <div class="stat">Cargando Grafo Vivo...</div>
    </div>
    <div id="cy"></div>
    <script>
        fetch('/api/graph')
            .then(res => res.json())
            .then(data => {
                const elements = [];
                let orphCount = 0;
                
                // Nodos
                for (const [id, node] of Object.entries(data.nodes)) {
                    let color = '#3b82f6'; // Azul = módulo/función sana
                    if (node.is_orphan) {
                        color = '#ef4444'; // Rojo = Zombie / Huérfano
                        orphCount++;
                    }
                    if (node.node_type === 'workspace') color = '#10b981'; // Verde = Root
                    if (node.node_type === 'struct') color = '#f59e0b'; // Naranja = Estado

                    let labelText = node.label;
                    if (node.average_latency_ns > 0) {
                        labelText += ` (${node.average_latency_ns}ns)`;
                        if (node.average_latency_ns > 50000) {
                            color = '#b91c1c'; // Rojo oscuro = Cuello de botella severo
                        } else if (node.average_latency_ns > 5000) {
                            color = '#ea580c'; // Naranja oscuro = Advertencia de latencia
                        }
                    }
                    
                    elements.push({
                        data: { 
                            id: node.id, 
                            label: labelText, 
                            file_path: node.file_path, 
                            line_number: node.line_number,
                            latency: node.average_latency_ns
                        },
                        style: {
                            'background-color': color,
                            'label': 'data(label)'
                        }
                    });
                }
                
                // Aristas
                data.edges.forEach(edge => {
                    elements.push({
                        data: { source: edge.source, target: edge.target }
                    });
                });

                document.querySelector('.stat').innerHTML = `
                    Nodos de Código: ${Object.keys(data.nodes).length}<br>
                    Conexiones (Flujos): ${data.edges.length}<br>
                    <span style="color:#ef4444">Nodos Desconectados (Huérfanos): ${orphCount}</span>
                `;

                var cy = cytoscape({
                    container: document.getElementById('cy'),
                    elements: elements,
                    style: [
                        {
                            selector: 'node',
                            style: {
                                'label': 'data(label)',
                                'color': '#fff',
                                'text-valign': 'center',
                                'text-halign': 'right',
                                'text-margin-x': 10,
                                'font-size': '12px'
                            }
                        },
                        {
                            selector: 'edge',
                            style: {
                                'width': 2,
                                'line-color': '#334155',
                                'target-arrow-color': '#334155',
                                'target-arrow-shape': 'triangle',
                                'curve-style': 'bezier',
                                'opacity': 0.6
                            }
                        }
                    ],
                    layout: {
                        name: 'cose',
                        animate: false,
                        nodeOverlap: 20
                    }
                });

                // Drill-down (Abrir info)
                cy.on('tap', 'node', function(evt){
                    var node = evt.target;
                    var path = node.data('file_path');
                    var line = node.data('line_number');
                    if(path && path !== "") {
                        alert(`Componente: ${node.data('label')}\nArchivo: ${path}\nLínea: ${line}\n\nUsa VS Code: code -g "${path}:${line}"`);
                    }
                });
            });
    </script>
</body>
</html>"#;

    let index_route = warp::path::end()
        .map(move || warp::reply::html(html_content));

    let routes = index_route.or(api_route);

    println!("✅ Servidor Web interactivo listo en http://127.0.0.1:3030");
    warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;
}
