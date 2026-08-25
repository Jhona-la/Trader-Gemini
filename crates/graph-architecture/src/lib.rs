use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use syn::visit::{self, Visit};
use walkdir::WalkDir;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Node {
    pub id: String,
    pub label: String,
    pub node_type: String, // "crate", "module", "struct", "function", "feature"
    pub file_path: String,
    pub line_number: usize,
    pub is_orphan: bool,
    pub average_latency_ns: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Edge {
    pub source: String,
    pub target: String,
    pub edge_type: String, // "contains", "calls", "uses"
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct Graph4D {
    pub nodes: HashMap<String, Node>,
    pub edges: Vec<Edge>,
}

pub struct AstVisitor {
    pub current_file: String,
    pub current_module: String,
    pub current_fn: Option<String>,
    pub graph: Graph4D,
}

impl Default for AstVisitor {
    fn default() -> Self {
        Self::new()
    }
}

impl AstVisitor {
    pub fn new() -> Self {
        Self {
            current_file: String::new(),
            current_module: String::new(),
            current_fn: None,
            graph: Graph4D::default(),
        }
    }
}

impl<'ast> Visit<'ast> for AstVisitor {
    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        let fn_name = i.sig.ident.to_string();
        let id = format!("{}::{}", self.current_module, fn_name);

        let is_feature = fn_name.contains("feature") || fn_name.contains("update_ofi");
        let node_type = if is_feature { "feature" } else { "function" };

        self.graph.nodes.insert(
            id.clone(),
            Node {
                id: id.clone(),
                label: fn_name,
                node_type: node_type.to_string(),
                file_path: self.current_file.clone(),
                line_number: i.sig.ident.span().start().line,
                is_orphan: false,
                average_latency_ns: 0,
            },
        );

        self.graph.edges.push(Edge {
            source: self.current_module.clone(),
            target: id.clone(),
            edge_type: "contains".to_string(),
        });

        // Visitar cuerpo de la función registrando el frame de llamada activo
        let prev_fn = self.current_fn.take();
        self.current_fn = Some(id);
        visit::visit_item_fn(self, i);
        self.current_fn = prev_fn;
    }

    fn visit_expr_call(&mut self, i: &'ast syn::ExprCall) {
        // Extraer llamadas a funciones
        if let syn::Expr::Path(expr_path) = &*i.func {
            if let Some(segment) = expr_path.path.segments.last() {
                let target_name = segment.ident.to_string();

                let target_id = format!("{}::{}", self.current_module, target_name);
                let source_id = self
                    .current_fn
                    .clone()
                    .unwrap_or_else(|| self.current_module.clone());

                self.graph.edges.push(Edge {
                    source: source_id,
                    target: target_id,
                    edge_type: "calls".to_string(),
                });
            }
        }
        visit::visit_expr_call(self, i);
    }

    fn visit_expr_method_call(&mut self, i: &'ast syn::ExprMethodCall) {
        let target_name = i.method.to_string();
        let target_id = format!("{}::{}", self.current_module, target_name);
        let source_id = self
            .current_fn
            .clone()
            .unwrap_or_else(|| self.current_module.clone());

        self.graph.edges.push(Edge {
            source: source_id,
            target: target_id,
            edge_type: "calls".to_string(),
        });

        visit::visit_expr_method_call(self, i);
    }

    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        let struct_name = i.ident.to_string();
        let id = format!("{}::{}", self.current_module, struct_name);

        self.graph.nodes.insert(
            id.clone(),
            Node {
                id: id.clone(),
                label: struct_name,
                node_type: "struct".to_string(),
                file_path: self.current_file.clone(),
                line_number: i.ident.span().start().line,
                is_orphan: false,
                average_latency_ns: 0,
            },
        );

        self.graph.edges.push(Edge {
            source: self.current_module.clone(),
            target: id,
            edge_type: "contains".to_string(),
        });

        visit::visit_item_struct(self, i);
    }
}

pub fn scan_workspace(root_path: &Path) -> Graph4D {
    let mut visitor = AstVisitor::new();

    // Register root
    visitor.graph.nodes.insert(
        "trader_gemini".to_string(),
        Node {
            id: "trader_gemini".to_string(),
            label: "Trader Gemini V5".to_string(),
            node_type: "workspace".to_string(),
            file_path: "".to_string(),
            line_number: 0,
            is_orphan: false,
            average_latency_ns: 0,
        },
    );

    for entry in WalkDir::new(root_path) {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        if entry.path().extension().is_some_and(|ext| ext == "rs") {
            let path_str = entry.path().to_string_lossy().to_string();
            // Evitar escanear target/ o .rustup
            if path_str.contains("target") || path_str.contains(".cargo") {
                continue;
            }

            let code = match fs::read_to_string(entry.path()) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let syntax = match syn::parse_file(&code) {
                Ok(s) => s,
                Err(_) => continue,
            };

            // Determinar modulo por path
            let relative_path = entry.path().strip_prefix(root_path).unwrap_or(entry.path());
            let module_name = relative_path
                .with_extension("")
                .to_string_lossy()
                .replace("\\", "::")
                .replace("/", "::");

            visitor.graph.nodes.insert(
                module_name.clone(),
                Node {
                    id: module_name.clone(),
                    label: relative_path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string(),
                    node_type: "module".to_string(),
                    file_path: path_str.clone(),
                    line_number: 1,
                    is_orphan: false,
                    average_latency_ns: 0,
                },
            );

            visitor.graph.edges.push(Edge {
                source: "trader_gemini".to_string(),
                target: module_name.clone(),
                edge_type: "contains".to_string(),
            });

            visitor.current_file = path_str;
            visitor.current_module = module_name;
            visitor.visit_file(&syntax);
        }
    }

    // Detección Topológica de Nodos Huérfanos
    // Un nodo es huérfano si tiene 0 aristas entrantes de llamada ("calls") y no es el workspace root.
    let mut call_targets: HashMap<String, usize> = HashMap::new();
    for edge in &visitor.graph.edges {
        if edge.edge_type == "calls" {
            *call_targets.entry(edge.target.clone()).or_insert(0) += 1;
        }
    }
    for (id, node) in visitor.graph.nodes.iter_mut() {
        if id != "trader_gemini" && (node.node_type == "feature" || node.node_type == "function") {
            let calls = call_targets.get(id).copied().unwrap_or(0);
            node.is_orphan = calls == 0;
        }
    }

    visitor.graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_visitor_function_and_call_extraction() {
        let code = r#"
            fn compute_alpha() {
                update_ofi();
            }
            fn update_ofi() {}
        "#;
        let syntax: syn::File = syn::parse_str(code).unwrap();
        let mut visitor = AstVisitor::new();
        visitor.current_file = "test.rs".to_string();
        visitor.current_module = "test_mod".to_string();
        visitor.visit_file(&syntax);

        assert!(visitor.graph.nodes.contains_key("test_mod::compute_alpha"));
        assert!(visitor.graph.nodes.contains_key("test_mod::update_ofi"));
        assert_eq!(visitor.graph.nodes.get("test_mod::update_ofi").unwrap().node_type, "feature");
    }

    #[test]
    fn test_graph4d_serialization() {
        let mut graph = Graph4D::default();
        graph.nodes.insert(
            "root".to_string(),
            Node {
                id: "root".to_string(),
                label: "Root Node".to_string(),
                node_type: "crate".to_string(),
                file_path: "lib.rs".to_string(),
                line_number: 1,
                is_orphan: false,
                average_latency_ns: 10,
            },
        );
        let json = serde_json::to_string(&graph).unwrap();
        assert!(json.contains("Root Node"));
    }

    #[test]
    fn test_graph4d_edge_addition() {
        let mut graph = Graph4D::default();
        graph.edges.push(Edge {
            source: "A".to_string(),
            target: "B".to_string(),
            edge_type: "calls".to_string(),
        });
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0].edge_type, "calls");
    }

    #[test]
    fn test_graph4d_multiple_nodes_and_edges() {
        let mut graph = Graph4D::default();
        graph.nodes.insert("Engine".to_string(), Node {
            id: "Engine".to_string(),
            label: "Engine".to_string(),
            node_type: "module".to_string(),
            file_path: "engine.rs".to_string(),
            line_number: 10,
            is_orphan: false,
            average_latency_ns: 150,
        });
        graph.nodes.insert("OrderBook".to_string(), Node {
            id: "OrderBook".to_string(),
            label: "OrderBook".to_string(),
            node_type: "struct".to_string(),
            file_path: "orderbook.rs".to_string(),
            line_number: 20,
            is_orphan: false,
            average_latency_ns: 25,
        });
        graph.edges.push(Edge {
            source: "Engine".to_string(),
            target: "OrderBook".to_string(),
            edge_type: "uses".to_string(),
        });
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.nodes.get("Engine").unwrap().average_latency_ns, 150);
    }

    #[test]
    fn test_ast_visitor_struct_and_impl_detection() {
        let code = r#"
            pub struct SignalMatrix {
                pub weights: [f64; 4],
            }
            impl SignalMatrix {
                pub fn evaluate(&self) {
                    self.normalize();
                }
                pub fn normalize(&self) {}
            }
        "#;
        let syntax: syn::File = syn::parse_str(code).unwrap();
        let mut visitor = AstVisitor::new();
        visitor.current_file = "signals.rs".to_string();
        visitor.current_module = "signal_mod".to_string();
        visitor.visit_file(&syntax);

        assert!(visitor.graph.nodes.contains_key("signal_mod::SignalMatrix"));
        assert_eq!(visitor.graph.nodes.get("signal_mod::SignalMatrix").unwrap().node_type, "struct");
        assert!(visitor.graph.edges.iter().any(|e| e.edge_type == "calls" && e.target.contains("normalize")));
    }
}


