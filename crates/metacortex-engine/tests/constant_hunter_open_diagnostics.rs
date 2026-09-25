use metacortex_engine::CazadorConstantes;

#[test]
fn open_ordinal_gene_identity_moves_after_an_unrelated_literal_is_inserted() {
    let (a, _) =
        CazadorConstantes::strip_constants_with_tag("fn f()->f64 { 0.0543 }", "same_file").unwrap();
    let (b, _) = CazadorConstantes::strip_constants_with_tag(
        "fn f()->f64 { let unrelated=0.077; 0.0543 }",
        "same_file",
    )
    .unwrap();
    assert!(a.contains("\"gene_same_file_idx_1\" , 0.0543"));
    assert!(b.contains("\"gene_same_file_idx_1\" , 0.077"));
    assert!(b.contains("\"gene_same_file_idx_2\" , 0.0543"));
}

#[test]
fn open_const_function_and_inline_const_are_rewritten_with_runtime_calls() {
    for source in [
        "const fn coefficient()->f64 { 0.0543 }",
        "fn coefficient()->f64 { const {0.0543} }",
    ] {
        let (text, count) = CazadorConstantes::strip_constants(source).unwrap();
        assert_eq!(count, 1);
        assert!(text.contains("read_epigenoma_gene")); // Syntactic success is not const-evaluation validity.
    }
}

#[test]
fn open_macro_literals_are_not_visited_and_equal_semantics_have_different_mutability() {
    let (_, plain) = CazadorConstantes::strip_constants("fn f()->f64 {0.0543}").unwrap();
    let (_, wrapped) =
        CazadorConstantes::strip_constants("fn f(){ println!(\"{}\",0.0543); }").unwrap();
    assert_eq!((plain, wrapped), (1, 0));
}

#[test]
fn open_default_namespace_reuses_gene_keys_across_distinct_sources() {
    let (a, _) = CazadorConstantes::strip_constants("fn sl()->f64 {0.0543}").unwrap();
    let (b, _) = CazadorConstantes::strip_constants("fn fee()->f64 {0.077}").unwrap();
    assert!(a.contains("gene_global_idx_1"));
    assert!(b.contains("gene_global_idx_1"));
}
