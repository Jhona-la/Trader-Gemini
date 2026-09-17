use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use syn::{File, Item};

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

    pub fn parse_file<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let content = fs::read_to_string(&path)?;
        let ast: File = syn::parse_file(&content)?;

        // FIX #1451: Extracción resiliente de module_name sin unwrap frágil
        let module_name = path
            .as_ref()
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "anonymous_module".to_string());
        let module_idx = self.get_or_add_node(&module_name, "Module");

        for item in ast.items {
            match item {
                Item::Struct(s) => {
                    let struct_name = format!("{}::{}", module_name, s.ident);
                    let struct_idx = self.get_or_add_node(&struct_name, "Struct");
                    self.graph
                        .add_edge(module_idx, struct_idx, "contains".to_string());
                }
                Item::Fn(f) => {
                    let fn_name = format!("{}::{}", module_name, f.sig.ident);
                    let fn_idx = self.get_or_add_node(&fn_name, "Function");
                    self.graph
                        .add_edge(module_idx, fn_idx, "contains".to_string());
                }
                Item::Enum(e) => {
                    let enum_name = format!("{}::{}", module_name, e.ident);
                    let enum_idx = self.get_or_add_node(&enum_name, "Enum");
                    self.graph
                        .add_edge(module_idx, enum_idx, "contains".to_string());
                }
                Item::Impl(i) => {
                    let type_name = match &*i.self_ty {
                        syn::Type::Path(tp) => tp
                            .path
                            .segments
                            .last()
                            .map(|s| s.ident.to_string())
                            .unwrap_or_else(|| "Unknown".to_string()),
                        _ => "Type".to_string(),
                    };
                    let impl_name = format!("{}::impl::{}", module_name, type_name);
                    let impl_idx = self.get_or_add_node(&impl_name, "Impl");
                    self.graph
                        .add_edge(module_idx, impl_idx, "contains".to_string());
                    for impl_item in i.items {
                        if let syn::ImplItem::Fn(m) = impl_item {
                            let method_name =
                                format!("{}::{}::{}", module_name, type_name, m.sig.ident);
                            let method_idx = self.get_or_add_node(&method_name, "Method");
                            self.graph
                                .add_edge(impl_idx, method_idx, "implements".to_string());
                        }
                    }
                }
                Item::Trait(t) => {
                    let trait_name = format!("{}::{}", module_name, t.ident);
                    let trait_idx = self.get_or_add_node(&trait_name, "Trait");
                    self.graph
                        .add_edge(module_idx, trait_idx, "contains".to_string());
                }
                Item::Use(_u) => {
                    // Could extract dependencies from `use` statements here
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_graph_creation_and_node_lookup() {
        let mut sg = SystemGraph::new();
        let idx1 = sg.get_or_add_node("Engine", "Module");
        let idx2 = sg.get_or_add_node("Engine", "Module");
        assert_eq!(idx1, idx2);

        let idx3 = sg.get_or_add_node("State", "Struct");
        assert_ne!(idx1, idx3);
        assert_eq!(sg.graph.node_count(), 2);
    }

    #[test]
    fn test_system_graph_parse_file_resilience() {
        let mut sg = SystemGraph::default();
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_syn_graph.rs");
        std::fs::write(&file_path, "pub struct TestStruct; pub fn test_fn() {}").unwrap();
        assert!(sg.parse_file(&file_path).is_ok());
        assert!(sg.graph.node_count() >= 2);
        let _ = std::fs::remove_file(file_path);
    }

    #[test]
    fn test_system_graph_edges_and_dependencies() {
        let mut sg = SystemGraph::new();
        let m_idx = sg.get_or_add_node("EngineModule", "Module");
        let s_idx = sg.get_or_add_node("EngineModule::Core", "Struct");
        let edge_idx = sg.graph.add_edge(m_idx, s_idx, "contains".to_string());
        assert_eq!(sg.graph.edge_weight(edge_idx).unwrap(), "contains");
        assert_eq!(sg.graph.edge_count(), 1);
    }

    #[test]
    fn test_system_graph_enum_trait_impl_parsing() {
        let mut sg = SystemGraph::default();
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_complex_syn_graph.rs");
        let code = r#"
            pub enum Action { Buy, Sell }
            pub trait Signal { fn eval(&self) -> bool; }
            pub struct Engine;
            impl Engine {
                pub fn process(&self) {}
            }
        "#;
        std::fs::write(&file_path, code).unwrap();
        assert!(sg.parse_file(&file_path).is_ok());
        assert!(sg.graph.node_count() >= 4);
        let _ = std::fs::remove_file(file_path);
    }
}
