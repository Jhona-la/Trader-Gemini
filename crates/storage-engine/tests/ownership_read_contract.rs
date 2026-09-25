//! Read-side contracts only. No asynchronous ledger worker or operational DB.
use rusqlite::{params, Connection};
use std::path::PathBuf;
use storage_engine::PositionLedger;

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("tg-xxx-ownership-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn path(&self) -> String {
        self.0.join("ledger.db").to_str().unwrap().to_owned()
    }
    fn db(&self) -> Connection {
        let db = Connection::open(self.path()).unwrap();
        db.execute_batch("CREATE TABLE position_ownership(symbol TEXT, position_side TEXT, strategy TEXT, qty REAL, entry_price REAL);").unwrap();
        db
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let root = self.0.canonicalize().unwrap();
        assert!(root.starts_with(std::env::temp_dir().canonicalize().unwrap()));
        assert!(root
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("tg-xxx-ownership-"));
        std::fs::remove_dir_all(root).unwrap();
    }
}

fn insert(db: &Connection, label: &str, qty: f64, price: f64) {
    db.execute(
        "INSERT INTO position_ownership VALUES ('AAAUSDT','LONG',?1,?2,?3)",
        params![label, qty, price],
    )
    .unwrap();
}

#[test]
fn legacy_tuple_must_not_report_zero_for_continuous_ownership() {
    let f = Fixture::new();
    let db = f.db();
    insert(&db, "continuous", 0.25, 100.0);
    assert!(
        PositionLedger::get_ownership(&f.path(), "AAAUSDT", "LONG").is_none(),
        "legacy tuple cannot represent a continuous row; must fail, not claim flat"
    );
}

#[test]
fn malformed_quantity_must_not_become_valid_zero() {
    let f = Fixture::new();
    let db = f.db();
    db.execute(
        "INSERT INTO position_ownership VALUES ('AAAUSDT','LONG','scalp','corrupt',100)",
        [],
    )
    .unwrap();
    assert!(PositionLedger::get_ownership(&f.path(), "AAAUSDT", "LONG").is_none());
}

#[test]
fn read_of_missing_database_does_not_create_it() {
    let f = Fixture::new();
    assert!(PositionLedger::get_ownership(&f.path(), "AAAUSDT", "LONG").is_none());
    assert!(
        !std::path::Path::new(&f.path()).exists(),
        "a query is not authorization to create a database"
    );
}

#[test]
fn legacy_known_labels_remain_compatible() {
    let f = Fixture::new();
    let db = f.db();
    insert(&db, "scalp", 0.25, 100.0);
    insert(&db, "swing", 0.5, 110.0);
    assert_eq!(
        PositionLedger::get_ownership(&f.path(), "AAAUSDT", "LONG"),
        Some((0.25, 100.0, 0.5, 110.0))
    );
}

#[test]
fn unified_read_preserves_every_label_without_making_them_engines() {
    let f = Fixture::new();
    let db = f.db();
    for label in ["continuous", "scalp", "swing", "research/run-42"] {
        insert(&db, label, 0.25, 100.0);
    }
    let rows = PositionLedger::read_ownership(&f.path(), "AAAUSDT", "LONG").unwrap();
    assert_eq!(rows.len(), 4);
    assert_eq!(rows.iter().map(|r| r.quantity).sum::<f64>(), 1.0);
    assert_eq!(
        rows.iter()
            .map(|r| r.provenance_label.as_str())
            .collect::<Vec<_>>(),
        vec!["continuous", "research/run-42", "scalp", "swing"]
    );
    assert!(PositionLedger::get_ownership(&f.path(), "AAAUSDT", "LONG").is_none());
}

#[test]
fn read_side_is_scoped_by_asset_and_position_side() {
    let f = Fixture::new();
    let db = f.db();
    insert(&db, "continuous", 0.25, 100.0);
    db.execute(
        "INSERT INTO position_ownership VALUES ('BBBUSDT','LONG','continuous',9,10)",
        [],
    )
    .unwrap();
    db.execute(
        "INSERT INTO position_ownership VALUES ('AAAUSDT','SHORT','continuous',8,20)",
        [],
    )
    .unwrap();
    let rows = PositionLedger::read_ownership(&f.path(), "AAAUSDT", "LONG").unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].quantity, 0.25);
}

#[test]
fn corrupt_row_rejects_the_whole_query_not_a_partial_portfolio() {
    let f = Fixture::new();
    let db = f.db();
    insert(&db, "continuous", 0.25, 100.0);
    db.execute(
        "INSERT INTO position_ownership VALUES ('AAAUSDT','LONG','z-corrupt','bad',100)",
        [],
    )
    .unwrap();
    assert!(PositionLedger::read_ownership(&f.path(), "AAAUSDT", "LONG").is_err());
}

#[test]
fn invalid_domains_are_errors_not_zero_imputations() {
    for (qty, price) in [
        (-1.0, 100.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (f64::INFINITY, 100.0),
        (1.0, f64::INFINITY),
        (f64::NAN, 100.0),
    ] {
        let f = Fixture::new();
        let db = f.db();
        insert(&db, "continuous", qty, price);
        assert!(PositionLedger::read_ownership(&f.path(), "AAAUSDT", "LONG").is_err());
    }
}

#[test]
fn exact_small_nonzero_exposure_survives_the_read_contract() {
    let f = Fixture::new();
    let db = f.db();
    insert(&db, "continuous", 1e-15, 1e8);
    let rows = PositionLedger::read_ownership(&f.path(), "AAAUSDT", "LONG").unwrap();
    assert_eq!(rows[0].quantity, 1e-15);
}

#[test]
fn observed_empty_and_missing_schema_are_different() {
    let f = Fixture::new();
    assert!(PositionLedger::read_ownership(&f.path(), "AAAUSDT", "LONG").is_err());
    assert!(!std::path::Path::new(&f.path()).exists());
    let db = Connection::open(f.path()).unwrap();
    assert!(PositionLedger::read_ownership(&f.path(), "AAAUSDT", "LONG").is_err());
    drop(db);
    let _db = f.db();
    assert!(PositionLedger::read_ownership(&f.path(), "AAAUSDT", "LONG")
        .unwrap()
        .is_empty());
}

#[test]
fn legacy_duplicate_label_cannot_silently_overwrite_ownership() {
    let f = Fixture::new();
    let db = f.db();
    insert(&db, "scalp", 0.25, 100.0);
    insert(&db, "scalp", 0.5, 200.0);
    assert!(PositionLedger::get_ownership(&f.path(), "AAAUSDT", "LONG").is_none());
}
