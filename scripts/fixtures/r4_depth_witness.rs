#![allow(dead_code)]
#[path = "parser.rs"] mod parser;
use parser::DepthEvent;

fn walls(wire: &[u8]) -> Option<(f64, f64)> {
    DepthEvent::parse_from_json(wire).map(|e| (e.bid_wall, e.ask_wall))
}

fn main() {
    let nominal = br#"{"bids":[["10","2"]],"asks":[["11","3"]]}"#;
    let bid_inner_space = br#"{"bids":[["10","2"] ],"asks":[["11","3"]]}"#;
    let mixed_key_space = br#"{"bids":[["10","2"] ],"asks" : [["11","3"]]}"#;
    let empty_bid = br#"{"bids":[],"asks":[["11","3"]]}"#;
    let reversed_ask_space = br#"{"asks":[["11","3"] ],"bids":[["10","2"]]}"#;
    let finite_side_overflow = br#"{"bids":[["10","1e308"] ],"asks":[["11","1e308"]]}"#;
    for wire in [nominal.as_slice(), bid_inner_space.as_slice(), mixed_key_space.as_slice(),
                 empty_bid.as_slice(), reversed_ask_space.as_slice(), finite_side_overflow.as_slice()] {
        let json: serde_json::Value = serde_json::from_slice(wire).expect("all fixtures are valid JSON");
        let reference = |key: &str| -> f64 {
            json[key].as_array().unwrap().iter()
                .map(|level| level[1].as_str().unwrap().parse::<f64>().unwrap()).sum()
        };
        println!("wire={} reference=({},{}) observed={:?}", String::from_utf8_lossy(wire),
                 reference("bids"), reference("asks"), walls(wire));
    }
    assert_eq!(walls(nominal), Some((2.0, 3.0)));
    assert_eq!(walls(mixed_key_space), Some((2.0, 3.0)));
    assert_eq!(walls(bid_inner_space), Some((5.0, 3.0)));
    assert_eq!(walls(empty_bid), Some((3.0, 3.0)));
    assert_eq!(walls(reversed_ask_space), Some((2.0, 5.0)));
    assert_eq!(walls(finite_side_overflow), None);
    println!("RESULT=CHARACTERIZED_EXISTING_DEPTH_BUG; not repaired, no live/replay/economic execution");
}
