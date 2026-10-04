# ADR-0012 — Sólo el núcleo de ejecución sigue al almacén de genomas

- **Estado**: Aceptado
- **Fecha**: 2026-10-03
- **Responsables**: Claude (ciclo 8, CL-40); revisión cruzada en el PR del
  ciclo 8

## Contexto

`GodEngineCore::refresh_models` (primer evento y cada 1000) aplicaba
`config_dir/genomes/<env>/active.json` a cualquier núcleo cuya generación
aplicada fuera menor, y todos nacen en 0. En el host (`TG_GENOME_ENV`
definida) también corren núcleos que EVALÚAN un genoma: el examen
walk-forward del demonio, los universos del bosque sombra y el SA de la
evolución; en los backtests de evolución, los mutantes. Todos sustituían su
candidato por el activo en su primer evento: el examen juzgaba al genoma
activo frente a sí mismo. `train_forest` y `audit_forensic_backtest` ya lo
sorteaban a mano (`applied_generation = u64::MAX`).

## Decisión

1. La política de recarga se deriva del contexto del núcleo
   (`RecargaGenoma::para(OutcomeContext)`): sólo `ExchangeLocalEstimate`
   (el núcleo conectado a la ejecución) sigue al almacén;
   `IsolatedSimulation` conserva el genoma que se le aplicó.
   `GOD_NO_HOT_RELOAD` sigue apagando la recarga del núcleo vivo.
2. El bosque sombra anota la generación de la que se plantó
   (`generacion_base`) y el host lo replanta con `seguir_generacion`
   ANTES de cosechar cuando el almacén sanciona una generación más nueva,
   y tras su propia cosecha. Un mutante del genoma anterior no puede
   sustituir a la generación sancionada. El host mira el almacén (por la
   fecha de `active.json`, como `refresh_models`) y no el contador
   `applied_generation` del núcleo: el demonio aplica sus promociones y
   rollbacks directo al arena y ese contador no se mueve. Re-registrar el
   mismo genoma en otra generación no replanta (conserva capital y
   mutantes).

## Consecuencias

- Examen, bosque sombra, SA y backtests juzgan al candidato que se les
  dio (contrato `genoma_fijado_contract`).
- Un núcleo nuevo que deba seguir al almacén tiene que construirse con
  `new_with_outcome_context(.., ExchangeLocalEstimate)`; construir con
  `new` es evaluar un genoma fijo.
