// [SIN CONSUMIDOR — censo GLM F5-A H1, 2026-10-05] `start_evolution_loop`
// NO tiene callers (reemplazado por LiveEvolutionDaemon en el host). Su gate
// de promoción promueve basura (1 trade + PnL>0, sin DSR/incumbente/OOS) y
// su `frozen_macro` inyecta literales de 2024 — NO cablear sin rediseño con
// el embudo del daemon vivo (WF_MIN_TRADES=30, DSR D-746, incumbente D-740).
// Isla extendida del censo qo-605 (ahora 6 módulos: +lib.rs +polars_evolver).

// [CENSO CERRADO — Sol F9, 2026-10-17] La isla muerta de 2024 fue RETIRADA:
// `start_evolution_loop` + `EvolutionEngine` + `frozen_macro`. Tres motivos:
// (1) CENSO: sin un solo llamador en todo el repo (git grep 2026-10-17; el
//     host usa `LiveEvolutionDaemon` de `online_daemon`);
// (2) El censo GLM F5-A H1 (2026-10-05) ya dictaminó «NO cablear sin rediseño
//     con el embudo del daemon vivo»: su gate promovía con 1 trade + PnL>0,
//     sin DSR/incumbente/OOS — cablearla habría revivido una puerta de basura;
// (3) La macro del tensor 54D (slots 21-29: DXY/SP500/NASDAQ/VIX/US10Y/GOLD/
//     OIL/FED_RATE) la sirve el camino vivo con productores reales y centinela
//     honesto en `god-engine-core` (Yahoo B3.23 para 21-24, PAXG para 26;
//     cero incondicional para 25/27/29 sin productor) — los literales 2024
//     eran placeholders prohibidos por el mandato del operador.
// El backlog GLM-133 «cablear Arc<OmniState> al loop de evolución» queda
// CERRADO por esta vía: el loop de evolución vivo NO necesita esos literales
// (su omni se construye causal en online_daemon) y la isla muerta no se
// reactiva. Hallazgo derivado documentado en el buzón: el omni del daemon
// vivo se construye internamente (macro=0) mientras god-engine recibe el
// OmniState del host — skew de examen<->vivo a investigar como átomo propio.

pub mod fitness;
// G1-5 (Ola Ω11): unificar selection_stats en risk_engine como fuente canónica única (DRY).
pub use risk_engine::selection_stats;
pub mod score_retention;
pub mod sa_selection;
pub mod anti_bias_governor;
pub mod ast_mutator;
pub mod cma_es;
pub mod crossover_cauchy;
pub mod entropy_fitness;
pub mod meta;
pub mod moe_neat_arena;
pub mod neat;
pub mod online_daemon;
pub mod return_evidence;
pub use return_evidence::{SequentialVilleEvidence, VigilanciaDegradacion};
pub mod online_random_forest;
pub mod lock_seguro;
pub mod online_newton_gbdt;
pub use online_newton_gbdt::OnlineNewtonGbdt;
pub mod polars_evolver;
pub mod random_forest;
pub mod stanford_discovery;

pub use crossover_cauchy::EvolutionaryOperators;
pub use moe_neat_arena::{ParetoCandidate, fast_non_dominated_sort};
pub use stanford_discovery::{
    CypherValidationResult, CypherVerdict, DistilledChampion, QuantitativeHypothesis,
    SandboxEvaluationMetrics, StanfordDiscoveryEngine, PILLAR_COUNT,
};
