//! # Constant Hunter (Cazador de Constantes)
//!
//! AST scanner using `syn::visit_mut` that strips hardcoded magic numbers from Rust code
//! and replaces them with dynamic unique `Epigenoma` mmap state lookups (`gene_line{X}_col{Y}`).

use syn::visit_mut::{self, VisitMut};
use syn::{Expr, ExprLit, Lit, parse_file};
use quote::quote;

pub struct CazadorConstantesVisitor {
    pub constants_replaced: usize,
    pub counter: usize,
}

impl CazadorConstantesVisitor {
    pub fn new() -> Self {
        Self {
            constants_replaced: 0,
            counter: 0,
        }
    }
}

impl VisitMut for CazadorConstantesVisitor {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Expr::Lit(ExprLit { lit, .. }) = expr {
            match lit {
                Lit::Float(f) => {
                    let val: f64 = f.base10_parse().unwrap_or(0.0);
                    // Skip 0.0, 1.0, -1.0 trivial constants
                    if (val.abs() - 0.0).abs() > 1e-9 && (val.abs() - 1.0).abs() > 1e-9 {
                        self.counter += 1;
                        let unique_key = format!("gene_float_idx_{}", self.counter);

                        let replacement = quote! {
                            metacortex_engine::read_epigenoma_gene(#unique_key, #val)
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
        let mut file_ast = parse_file(rust_code).map_err(|e| format!("AST parse error: {}", e))?;
        let mut visitor = CazadorConstantesVisitor::new();
        visitor.visit_file_mut(&mut file_ast);
        let transformed_code = quote!(#file_ast).to_string();
        Ok((transformed_code, visitor.constants_replaced))
    }
}
