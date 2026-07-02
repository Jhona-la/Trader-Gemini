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

        self.graph.nodes.insert(id.clone(), Node {
            id: id.clone(),
            label: fn_name,
            node_type: node_type.to_string(),
            file_path: self.current_file.clone(),
            line_number: i.sig.ident.span().start().line,
            is_orphan: false,
            average_latency_ns: 0,
        });

        self.graph.edges.push(Edge {
            source: self.current_module.clone(),
            target: id.clone(),
            edge_type: "contains".to_string(),
        });

        // Visitar cuerpo de la función para extraer llamadas
        visit::visit_item_fn(self, i);
    }

    fn visit_expr_call(&mut self, i: &'ast syn::ExprCall) {
        // Extraer llamadas a funciones
        if let syn::Expr::Path(expr_path) = &*i.func {
            if let Some(segment) = expr_path.path.segments.last() {
                let target_name = segment.ident.to_string();
                
                // Tratar de estimar el target (heurística básica, asumiendo misma u otra module_path)
                let target_id = format!("{}::{}", self.current_module, target_name);
                
                // Evitamos crear auto-llamadas u overhead masivo en este prototipo, pero
                // las registramos como "calls". Asumimos que el source es el current_module.
                // Idealmente el source sería la función contenedora, pero requiere trackear la pila.
                
                self.graph.edges.push(Edge {
                    source: self.current_module.clone(), // Por ahora lo atamos al módulo
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
        
        self.graph.edges.push(Edge {
            source: self.current_module.clone(),
            target: target_id,
            edge_type: "calls".to_string(),
        });
        
        visit::visit_expr_method_call(self, i);
    }

    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        let struct_name = i.ident.to_string();
        let id = format!("{}::{}", self.current_module, struct_name);

        self.graph.nodes.insert(id.clone(), Node {
            id: id.clone(),
            label: struct_name,
            node_type: "struct".to_string(),
            file_path: self.current_file.clone(),
            line_number: i.ident.span().start().line,
            is_orphan: false,
            average_latency_ns: 0,
        });

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
    visitor.graph.nodes.insert("trader_gemini".to_string(), Node {
        id: "trader_gemini".to_string(),
        label: "Trader Gemini V5".to_string(),
        node_type: "workspace".to_string(),
        file_path: "".to_string(),
        line_number: 0,
        is_orphan: false,
        average_latency_ns: 0,
    });

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
            let module_name = relative_path.with_extension("").to_string_lossy().replace("\\", "::").replace("/", "::");

            visitor.graph.nodes.insert(module_name.clone(), Node {
                id: module_name.clone(),
                label: relative_path.file_name().unwrap_or_default().to_string_lossy().to_string(),
                node_type: "module".to_string(),
                file_path: path_str.clone(),
                line_number: 1,
                is_orphan: false,
                average_latency_ns: 0,
            });

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

    // Heuristica: Detección de Nodos Huérfanos
    // (Simplificada: Si es un "feature" y no hay otro nodo que contenga su nombre en su código, es huérfano)
    // Para propósitos de Fase 1, marcamos algunos al azar si no están conectados (lógica avanzada luego)
    for node in visitor.graph.nodes.values_mut() {
        if node.node_type == "feature" {
            // Placeholder: Check if it's orphaned (En una versión real, escaneamos uso de la función en todo el AST)
            node.is_orphan = false; 
        }
    }

    visitor.graph
}
