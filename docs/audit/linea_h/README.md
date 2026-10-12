# Datos de la Línea H (Claude, LH-n) — 2026-10-12

Instrumento: `crates/backtest-engine/tests/t3_ventaja_conocida.rs` (T-3).
Plan y lectura de los resultados: `docs/PLAN_LINEA_H_CLAUDE_2026-10-12.md` §6.

| Archivo | Qué es | Cómo se regenera |
|---|---|---|
| `t3_escalera_13usd.tsv` | La escalera completa (8 peldaños) con 13 USD; una fila por peldaño, cabecera con el nombre de cada columna | `cargo test --release -p backtest-engine --test t3_ventaja_conocida -- --ignored --nocapture --test-threads=1 t3_escalera_de_ventaja_conocida` y filtrar `[T3-TSV]` |
| `t3_lineas_13usd.txt` | Las mismas filas en la forma legible `[T3] …` | ídem, filtrar `[T3]` |
| `t3_trazas_13usd.txt` | Contadores del runner nativo por peldaño (`opened`, `council_vetoes`, `ml_vetoes`, `closes`) | ídem, filtrar `[T1-DIAG]` (el test fija `TG_TRACE_NATIVO=1`) |
| `t3_escalera_capital.tsv`, `t3_trazas_capital.txt` | Los dos peldaños fuertes con 100 y 1 000 USD | `… t3_escalera_por_capital` (`T3_CAPITALES=100,1000`) |
| `t3_capital100_max_fee_035.tsv`, `t3_trazas_capital100_max_fee_035.txt` | Intervención: 100 USD con el gen `max_fee_pct` en 3,5 % | `T3_CAPITALES=100 T3_MAX_FEE_PCT=0.035 … t3_escalera_por_capital` |
| `t3_lado_vs_estado.txt`, `lado_vs_estado.py` | Fracción de aperturas trazadas del lado del estado oculto, τ mediana y rama por peldaño (sólo las 100 primeras aperturas que el núcleo traza) | `python -I lado_vs_estado.py <log de la corrida>`; el script replica la LCG del generador y es sólo diagnóstico (base `e0cd88b7`) |
| `lectores.tsv`, `predicciones.tsv`, `lectores_resumen.md`, `lectores_raw.json` | Los tres lectores de la cadena meta → orden (H1 tamaño, H2 evidencia/EV, H3 dirección/τ): hallazgos con cita, predicciones falsables y resúmenes | workflow `linea-h-lectores` (tres `Plan` agents de sólo lectura) |
| `verificacion.tsv`, `escepticos_raw.json` | Veredicto de un escéptico por lente sobre cada hallazgo y cada predicción, con la escalera medida en la mano | workflow `linea-h-escepticos` |

Todas las columnas con unidades: `*_bps` en puntos básicos, `d_pct`/`sigma_pct`
en %, `pnl_neto`/`capital_final` en USD, `dd_max` en fracción, `rechazos` como
`nombre=diferencia` de `risk_engine::REJECT_COUNTERS` (una evaluación de
intención por tick, 30 ticks por vela).
