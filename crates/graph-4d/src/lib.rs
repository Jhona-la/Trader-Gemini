use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;
use syn::{Item, File};
use std::fs;
use std::path::Path;

pub struct GraphNode {
    pub name: String,
    pub kind: String,
}

pub struct SystemGraph {
    pub graph: DiGraph<GraphNode, String>, // nodes, edge weights (dependency type)
    pub node_indices: HashMap<String, NodeIndex>,
}

impl SystemGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_indices: HashMap::new(),
        }
    }

    pub fn parse_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let content = fs::read_to_string(&path)?;
        let ast: File = syn::parse_file(&content)?;

        let module_name = path.as_ref().file_stem().unwrap().to_str().unwrap().to_string();
        let module_idx = self.get_or_add_node(&module_name, "Module");

        for item in ast.items {
            match item {
                Item::Struct(s) => {
                    let struct_name = s.ident.to_string();
                    let struct_idx = self.get_or_add_node(&struct_name, "Struct");
                    self.graph.add_edge(module_idx, struct_idx, "contains".to_string());
                },
                Item::Fn(f) => {
                    let fn_name = f.sig.ident.to_string();
                    let fn_idx = self.get_or_add_node(&fn_name, "Function");
                    self.graph.add_edge(module_idx, fn_idx, "contains".to_string());
                },
                Item::Enum(e) => {
                    let enum_name = e.ident.to_string();
                    let enum_idx = self.get_or_add_node(&enum_name, "Enum");
                    self.graph.add_edge(module_idx, enum_idx, "contains".to_string());
                },
                Item::Use(u) => {
                    // Could extract dependencies from `use` statements here
                },
                _ => {}
            }
        }
        Ok(())
    }

    fn get_or_add_node(&mut self, name: &str, kind: &str) -> NodeIndex {
        if let Some(idx) = self.node_indices.get(name) {
            *idx
        } else {
            let idx = self.graph.add_node(GraphNode {
                name: name.to_string(),
                kind: kind.to_string(),
            });
            self.node_indices.insert(name.to_string(), idx);
            idx
        }
    }
}

impl Default for SystemGraph {
    fn default() -> Self {
        Self::new()
    }
}
