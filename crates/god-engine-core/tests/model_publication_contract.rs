//! MP: synthetic publication contracts; never inspect or activate real models.
use god_engine_core::ml_inference::{GLOBAL_FORESTS, NanoForest, NanoForestData};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};

fn model(bias: f32) -> NanoForestData {
    NanoForestData {
        children_left: vec![-1],
        children_right: vec![-1],
        feature: vec![-1],
        threshold: vec![0.0],
        value: vec![0.0],
        tree_offsets: vec![0, 1],
        init_score: bias,
    }
}

struct Fixture(PathBuf);
impl Fixture {
    fn new(tag: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "tg_mp_{}_{}_{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn json(&self, name: &str, bias: f32) -> PathBuf {
        let path = self.0.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, serde_json::to_vec(&model(bias)).unwrap()).unwrap();
        path
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        // Only our uniquely created fixture subtree, including on assertion panic.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn load(path: &Path) -> NanoForest {
    NanoForest::load_model(path.to_str().unwrap()).unwrap()
}

#[test]
fn mp_concurrent_store_preserves_every_distinct_asset() {
    const WRITERS: usize = 32;
    const ROUNDS: usize = 8;
    let barrier = Arc::new(Barrier::new(WRITERS));
    std::thread::scope(|scope| {
        for writer in 0..WRITERS {
            let barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                for round in 0..ROUNDS {
                    let forest = NanoForest::from_data(model(writer as f32)).unwrap();
                    barrier.wait();
                    NanoForest::store_global(&format!("MP_STORE_{round}_{writer}"), forest);
                    barrier.wait();
                }
            });
        }
    });
    let missing: Vec<_> = (0..ROUNDS)
        .flat_map(|round| (0..WRITERS).map(move |writer| (round, writer)))
        .filter(|(round, writer)| {
            NanoForest::get_global(&format!("MP_STORE_{round}_{writer}"))
                .is_none_or(|f| f.init_value() != *writer as f64)
        })
        .collect();
    assert!(
        missing.is_empty(),
        "lost {} of {} publications: {missing:?}",
        missing.len(),
        WRITERS * ROUNDS
    );
}

#[test]
fn mp_concurrent_load_preserves_every_distinct_asset() {
    const WRITERS: usize = 24;
    let fixture = Fixture::new("parallel_load");
    let paths: Vec<_> = (0..WRITERS)
        .map(|i| {
            let path = fixture.json(&format!("asset_{i}.json"), i as f32);
            // Prebuild cache outside contention; JSON/bin are private per writer.
            load(&path);
            path
        })
        .collect();
    let barrier = Arc::new(Barrier::new(WRITERS));
    std::thread::scope(|scope| {
        for (i, path) in paths.iter().enumerate() {
            let barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                barrier.wait();
                NanoForest::load_global(&format!("MP_LOAD_{i}"), path.to_str().unwrap()).unwrap();
            });
        }
    });
    let missing: Vec<_> = (0..WRITERS)
        .filter(|i| {
            NanoForest::get_global(&format!("MP_LOAD_{i}"))
                .is_none_or(|f| f.init_value() != *i as f64)
        })
        .collect();
    assert!(
        missing.is_empty(),
        "lost {} of {WRITERS} loads: {missing:?}",
        missing.len()
    );
}

#[test]
fn mp_replacement_keeps_other_assets_and_reader_snapshot() {
    NanoForest::store_global("MP_REPLACE_A", NanoForest::from_data(model(1.0)).unwrap());
    NanoForest::store_global("MP_REPLACE_B", NanoForest::from_data(model(2.0)).unwrap());
    let old_map = GLOBAL_FORESTS.load_full();
    let old_a = NanoForest::get_global("MP_REPLACE_A").unwrap();
    NanoForest::store_global("MP_REPLACE_A", NanoForest::from_data(model(3.0)).unwrap());
    assert_eq!(old_map["MP_REPLACE_A"].init_value(), 1.0);
    assert_eq!(old_a.init_value(), 1.0);
    assert_eq!(
        NanoForest::get_global("MP_REPLACE_A").unwrap().init_value(),
        3.0
    );
    assert_eq!(
        NanoForest::get_global("MP_REPLACE_B").unwrap().init_value(),
        2.0
    );
}

#[test]
fn mp_failed_load_does_not_replace_last_valid_model() {
    let fixture = Fixture::new("rejected");
    let path = fixture.json("invalid.json", 0.0);
    let mut invalid = model(0.0);
    invalid.children_left[0] = 0;
    invalid.children_right[0] = 0;
    invalid.feature[0] = 0;
    std::fs::write(&path, serde_json::to_vec(&invalid).unwrap()).unwrap();
    NanoForest::store_global("MP_REJECT", NanoForest::from_data(model(7.0)).unwrap());
    assert!(NanoForest::load_global("MP_REJECT", path.to_str().unwrap()).is_err());
    assert_eq!(
        NanoForest::get_global("MP_REJECT").unwrap().init_value(),
        7.0
    );
    assert!(!path.with_extension("bin").exists());
}

#[test]
fn mp_json_cache_path_changes_only_final_extension() {
    let fixture = Fixture::new("json_parent");
    let path = fixture.json("archive.json/asset.json.v2.json", 3.0);
    let source = std::fs::read(&path).unwrap();
    assert_eq!(load(&path).init_value(), 3.0);
    assert!(
        path.with_extension("bin").exists(),
        "cache must be the true sibling"
    );
    assert_eq!(std::fs::read(&path).unwrap(), source);
    assert!(!fixture.0.join("archive.bin").exists());
}

#[test]
fn mp_bin_path_uses_real_sibling_json_even_with_bin_in_parent() {
    let fixture = Fixture::new("bin_parent");
    let json = fixture.json("archive.bin/asset.bin.v2.json", -2.0);
    let bin = json.with_extension("bin");
    std::fs::write(&bin, bincode::serialize(&model(2.0)).unwrap()).unwrap();
    let epoch = std::time::SystemTime::UNIX_EPOCH;
    std::fs::File::options()
        .write(true)
        .open(&bin)
        .unwrap()
        .set_times(
            std::fs::FileTimes::new().set_modified(epoch + std::time::Duration::from_secs(10_000)),
        )
        .unwrap();
    std::fs::File::options()
        .write(true)
        .open(&json)
        .unwrap()
        .set_times(
            std::fs::FileTimes::new().set_modified(epoch + std::time::Duration::from_secs(20_000)),
        )
        .unwrap();
    assert_eq!(load(&bin).init_value(), -2.0, "newer sibling JSON must win");
}

#[test]
fn mp_extensionless_source_is_not_overwritten_as_binary_cache() {
    let fixture = Fixture::new("no_extension");
    let path = fixture.json("model", 4.0);
    let source = std::fs::read(&path).unwrap();
    assert_eq!(load(&path).init_value(), 4.0);
    assert_eq!(
        std::fs::read(&path).unwrap(),
        source,
        "source bytes must remain JSON"
    );
}

#[test]
fn mp_standalone_binary_remains_supported_and_validated() {
    let fixture = Fixture::new("standalone");
    let bin = fixture.0.join("model.bin");
    std::fs::write(&bin, bincode::serialize(&model(5.0)).unwrap()).unwrap();
    assert_eq!(load(&bin).init_value(), 5.0);
    let mut invalid = model(5.0);
    invalid.value[0] = f32::INFINITY;
    std::fs::write(&bin, bincode::serialize(&invalid).unwrap()).unwrap();
    assert!(NanoForest::load_model(bin.to_str().unwrap()).is_err());
}

#[test]
fn mp_nonstandard_json_names_are_read_only_sources() {
    let fixture = Fixture::new("custom_names");
    for name in ["model.weights", "model.JSON", "archive.json/model.payload"] {
        let path = fixture.json(name, 6.0);
        let source = std::fs::read(&path).unwrap();
        assert_eq!(load(&path).init_value(), 6.0);
        assert_eq!(std::fs::read(&path).unwrap(), source);
        assert!(!path.with_extension("bin").exists());
    }
}

#[test]
fn mp_corrupt_binary_cache_falls_back_to_valid_json() {
    let fixture = Fixture::new("corrupt_cache");
    let json = fixture.json("model.json", 8.0);
    let bin = json.with_extension("bin");
    std::fs::write(&bin, b"not a bincode model").unwrap();
    make_cache_newer(&json, &bin);
    assert_eq!(load(&bin).init_value(), 8.0);
    let rebuilt: NanoForestData = bincode::deserialize(&std::fs::read(&bin).unwrap()).unwrap();
    assert_eq!(rebuilt.init_score, 8.0);
}

fn make_cache_newer(json: &Path, bin: &Path) {
    // Explicit ordering; never rely on filesystem timestamp resolution/sleeps.
    for (path, seconds) in [(json, 10_000), (bin, 20_000)] {
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_times(
                std::fs::FileTimes::new()
                    .set_modified(std::time::UNIX_EPOCH + std::time::Duration::from_secs(seconds)),
            )
            .unwrap();
    }
}

#[test]
fn mp_structurally_invalid_cache_falls_back_to_valid_json() {
    // Deserialization succeeds in every case. Rejection is structural, not I/O.
    for kind in ["cycle", "dimension", "length", "offset", "nan", "infinity"] {
        for extension in ["json", "bin"] {
            let fixture = Fixture::new(&format!("semantic_{kind}_{extension}"));
            let json = fixture.json("model.json", 9.0);
            let bin = json.with_extension("bin");
            let source = std::fs::read(&json).unwrap();
            let mut invalid = model(-9.0);
            match kind {
                "cycle" | "dimension" => {
                    invalid.children_left[0] = 0;
                    invalid.children_right[0] = 0;
                    invalid.feature[0] = if kind == "dimension" { 48 } else { 0 };
                }
                "length" => invalid.threshold.clear(),
                "offset" => invalid.tree_offsets[1] = 2,
                "nan" => invalid.init_score = f32::NAN,
                "infinity" => invalid.value[0] = f32::INFINITY,
                _ => unreachable!(),
            }
            let encoded = bincode::serialize(&invalid).unwrap();
            let decoded: NanoForestData = bincode::deserialize(&encoded).unwrap();
            assert!(NanoForest::from_data(decoded).is_err());
            std::fs::write(&bin, encoded).unwrap();
            make_cache_newer(&json, &bin);
            let request = json.with_extension(extension);
            let forest = NanoForest::load_model(request.to_str().unwrap())
                .unwrap_or_else(|e| panic!("{kind}/{extension}: valid JSON was blocked: {e}"));
            assert_eq!(forest.init_value(), 9.0);
            assert_eq!(std::fs::read(&json).unwrap(), source);
            let rebuilt: NanoForestData =
                bincode::deserialize(&std::fs::read(&bin).unwrap()).unwrap();
            assert_eq!(NanoForest::from_data(rebuilt).unwrap().init_value(), 9.0);
        }
    }
}

#[test]
fn mp_invalid_cache_and_source_preserve_files_and_last_valid_model() {
    let fixture = Fixture::new("both_invalid");
    let json = fixture.json("model.json", 0.0);
    let bin = json.with_extension("bin");
    let mut invalid = model(0.0);
    invalid.threshold.clear();
    let cache = bincode::serialize(&invalid).unwrap();
    NanoForest::store_global(
        "MP_BOTH_INVALID",
        NanoForest::from_data(model(7.0)).unwrap(),
    );
    for source in [
        b"invalid JSON".to_vec(),
        serde_json::to_vec(&invalid).unwrap(),
    ] {
        std::fs::write(&json, &source).unwrap();
        std::fs::write(&bin, &cache).unwrap();
        make_cache_newer(&json, &bin);
        for request in [&json, &bin] {
            let error = NanoForest::load_global("MP_BOTH_INVALID", request.to_str().unwrap())
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("cache ") && error.contains("JSON source "),
                "{error}"
            );
            assert_eq!(
                NanoForest::get_global("MP_BOTH_INVALID")
                    .unwrap()
                    .init_value(),
                7.0
            );
            assert_eq!(std::fs::read(&json).unwrap(), source);
            assert_eq!(std::fs::read(&bin).unwrap(), cache);
        }
    }
}

#[test]
fn mp_invalid_standalone_cache_does_not_publish_or_rewrite() {
    let fixture = Fixture::new("invalid_standalone");
    let bin = fixture.0.join("model.bin");
    let mut invalid = model(0.0);
    invalid.threshold.clear();
    let cache = bincode::serialize(&invalid).unwrap();
    std::fs::write(&bin, &cache).unwrap();
    assert!(NanoForest::load_global("MP_NO_SOURCE", bin.to_str().unwrap()).is_err());
    assert!(NanoForest::get_global("MP_NO_SOURCE").is_none());
    assert!(!bin.with_extension("json").exists());
    assert_eq!(std::fs::read(&bin).unwrap(), cache);
}

#[test]
fn mp_newer_invalid_json_must_not_fall_back_to_older_valid_cache() {
    let fixture = Fixture::new("invalid_new_source");
    let json = fixture.json("model.json", 0.0);
    let bin = json.with_extension("bin");
    let cache = bincode::serialize(&model(6.0)).unwrap();
    let mut invalid = model(0.0);
    invalid.threshold.clear();
    let source = serde_json::to_vec(&invalid).unwrap();
    std::fs::write(&json, &source).unwrap();
    std::fs::write(&bin, &cache).unwrap();
    // Invert the helper inputs to make the source strictly newer.
    make_cache_newer(&bin, &json);
    for request in [&json, &bin] {
        assert!(NanoForest::load_global("MP_STALE_VALID", request.to_str().unwrap()).is_err());
        assert!(NanoForest::get_global("MP_STALE_VALID").is_none());
        assert_eq!(std::fs::read(&json).unwrap(), source);
        assert_eq!(std::fs::read(&bin).unwrap(), cache);
    }
}
