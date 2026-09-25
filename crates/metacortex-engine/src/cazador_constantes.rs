//! # Constant Hunter (Cazador de Constantes)
//!
//! Experimental AST rewrite of selected float literals into global Epigenoma lookups.
//! Keys use file_tag + traversal ordinal, NOT stable semantic identity. This is
//! not a type/unit-aware gene extractor or a compilation check. The underlying
//! store is process-global atomics, not mmap. Const functions/blocks and macro
//! token streams require explicit handling before this can be a safe source transform.

use quote::quote;
use syn::visit_mut::{self, VisitMut};
use syn::{parse_file, Expr, ExprLit, Lit};

pub struct CazadorConstantesVisitor {
    pub file_tag: String,
    pub constants_replaced: usize,
    pub counter: usize,
}

impl CazadorConstantesVisitor {
    pub fn new() -> Self {
        Self {
            file_tag: "global".to_string(),
            constants_replaced: 0,
            counter: 0,
        }
    }

    pub fn with_file_tag(file_tag: impl Into<String>) -> Self {
        Self {
            file_tag: file_tag.into(),
            constants_replaced: 0,
            counter: 0,
        }
    }
}

impl VisitMut for CazadorConstantesVisitor {
    fn visit_item_const_mut(&mut self, _node: &mut syn::ItemConst) {
        // Ignorar expresiones dentro de constantes (const X: f64 = ...) para evitar E0015
    }

    fn visit_item_static_mut(&mut self, _node: &mut syn::ItemStatic) {
        // Ignorar expresiones dentro de estáticos para evitar E0015
    }

    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Expr::Lit(ExprLit { lit, .. }) = expr {
            match lit {
                Lit::Float(f) => {
                    let val: f64 = f.base10_parse().unwrap_or(0.0);
                    if !val.is_finite() {
                        return;
                    }
                    // FIX #620: Omitir 0.0, 1.0, 0.5, 2.0 y constantes matemáticas universales invariantes
                    let is_trivial = (val.abs() - 0.0).abs() < 1e-9
                        || (val.abs() - 1.0).abs() < 1e-9
                        || (val.abs() - 0.5).abs() < 1e-9
                        || (val.abs() - 2.0).abs() < 1e-9
                        || (val - std::f64::consts::PI).abs() < 1e-6
                        || (val - std::f64::consts::E).abs() < 1e-6;

                    if !is_trivial {
                        self.counter += 1;
                        let unique_key = format!("gene_{}_idx_{}", self.file_tag, self.counter);

                        let replacement = quote! {
                            (metacortex_engine::read_epigenoma_gene(#unique_key, #val) as _)
                        };
                        if let Ok(parsed) = syn::parse2::<Expr>(replacement) {
                            *expr = parsed;
                            self.constants_replaced += 1;
                            return; // Prevent recursive re-traversal of newly generated node!
                        }
                    }
                }
                Lit::Int(_) => {
                    // INTENCIONALMENTE SALTADO:
                    // Mutar enteros (`i64`/`usize`) corromperá el tamaño estático de arreglos
                    // en RAM y los bounds de bucles for, causando Kernel Panics.
                    // Solo mutamos coma flotante (f64) para pesos genéticos.
                }
                _ => {}
            }
        }

        visit_mut::visit_expr_mut(self, expr);
    }
}

pub struct CazadorConstantes;

impl CazadorConstantes {
    /// Scans a Rust source code string and replaces numeric literals with unique Epigenoma lookups
    pub fn strip_constants(rust_code: &str) -> Result<(String, usize), String> {
        Self::strip_constants_with_tag(rust_code, "global")
    }

    /// Scans a Rust source code string and replaces numeric literals with file-namespaced Epigenoma lookups
    pub fn strip_constants_with_tag(
        rust_code: &str,
        file_tag: &str,
    ) -> Result<(String, usize), String> {
        let mut file_ast = parse_file(rust_code).map_err(|e| format!("AST parse error: {}", e))?;
        let mut visitor = CazadorConstantesVisitor::with_file_tag(file_tag);
        visitor.visit_file_mut(&mut file_ast);
        let transformed_code = quote!(#file_ast).to_string();
        Ok((transformed_code, visitor.constants_replaced))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cazador_constantes_strip_floats() {
        let code = r#"
            fn evaluate_trade(price: f64) -> f64 {
                let threshold = 0.0543;
                let zero = 0.0;
                let one = 1.0;
                price * threshold + zero + one
            }
        "#;

        let (transformed, count) =
            CazadorConstantes::strip_constants(code).expect("Failed to transform code");
        assert_eq!(count, 1, "Only non-trivial float 0.0543 should be replaced");
        assert!(transformed.contains("read_epigenoma_gene"));
    }

    #[test]
    fn test_cazador_constantes_preserves_const_declarations() {
        let code = r#"
            const MAX_LEVERAGE: f64 = 20.0;
            static BASE_FEE: f64 = 0.0005;
        "#;

        let (transformed, count) =
            CazadorConstantes::strip_constants(code).expect("Failed to transform code");
        assert_eq!(
            count, 0,
            "Const and static items must not be replaced to avoid E0015"
        );
        assert!(!transformed.contains("read_epigenoma_gene"));
    }
}
