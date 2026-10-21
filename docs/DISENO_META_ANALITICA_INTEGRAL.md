# Diseño del Sistema Integral de Informes, Perfilado y Meta-Analítica

> GLM 141 (2026-10-21). Mandato del operador: «un sistema integral de
> informes, perfilado y meta-analítica que documente las estadísticas
> actuales e históricas de todas las operaciones, cálculos y del
> rendimiento global… desglosando métricas por área, archivo, función,
> modelo e integración… midiendo el impacto real —matemático,
> computacional y financiero— de cada cambio… construyendo una serie
> histórica evolutiva… logrando que el propio análisis estadístico
> impulse una evolución continua».

## 1. Diagnóstico: qué existe y qué falta

El repo YA tiene las piezas dispersas — el sistema integral las TEJE:

| Pieza | Qué cubre | Cadencia | Fuente |
|---|---|---|---|
| `meta_analitica.py` (RB-A1a+Sol) | Impacto por CAMBIO: líneas/tests/commits por agente y área, pánicos, coste de build, cobertura T-1 | por merge | `docs/audit/meta_analitica/*.tsv` |
| Sombras del auditor (Sol PR#86/#89) | Atribución por COMPONENTE: `sombra_wr_disc`, `sombra_friccion_real_menos_modelada_bp` por moneda, mitades OOS | por corrida | `crates/audit-engine/src/behavioral_auditor.rs` |
| Censo de voces (GLM 129/130) | Salud del ENSAMBLE: tasa de voto por fuente (14/19 hablantes), puertas del confluence | bajo demanda `#[ignore]` | `tests/censo_mudos_medicion.rs` |
| Oráculo T-1 + testigo | GATE de conducta: cobertura genética + binario verificado | por cambio de conducta | `outputs/oracle_witness_*.log` |
| Perfil camino caliente | LATENCIA: 11 asignaciones/evento, claves de texto | por frente de rendimiento | `scripts/medir_camino_caliente.sh` |
| Telemetría (flight_recorder, mmap) | OPERACIONES en vivo: PnL, vetos, señales | continua | `data/telemetry_*.bin|.mmap` |
| Dashboard | Vista en vivo | tiempo real | `src/dashboard.rs` |
| Informes históricos (R2_30D, aseguramiento) | Certificaciones puntuales | manual | `docs/*.md` |

**El hueco** (identificado en el sondeo del 141): ninguna pieza UNA
las estadísticas de operaciones con la atribución por componente y la
meta-analítica de código en UNA serie histórica evolutiva comparable
entre sí. Hoy no se puede responder «¿cuánto aportó el hedge vs. el
peldaño LH-06 vs. Ω145-hablantes al PnL de octubre?» en una sola
consulta.

## 2. Arquitectura: tres capas sobre las piezas

**Capa 1 — Registro uniforme (el dato).** Un esquema común
`MetricaRegistro` (TSV/JSONL, append-only, rotación): `fecha`,
`fuente` (que pieza la produjo), `área` (crate/módulo), `componente`
(motor/función/índice), `métrica` (nombre), `valor`, `unidades`,
`contexto` (commit, tape, genoma). Cada pieza existente se adapta con
un EMISOR (función que traduce su salida al esquema) — sin reescribir
las piezas.

**Capa 2 — Recopilador (el tejido).** `scripts/informe_integral.py`:
ejecuta (o lee lo último de) los emisores y produce el informe
periódico `docs/audit/informe_integral/informe_<YYYY-MM-DD>.md` con
secciones fijas: (a) operaciones del período (de telemetría/informes),
(b) atribución por componente (sombras + censo), (c) meta-analítica de
cambios (libro_impacto), (d) latencia y presupuesto (perfil +
semaforo.log), (e) gates (oráculos corridos, verdes/rojos), (f)
comparativa con el informe anterior (la SERIE HISTÓRICA: Δ por
métrica). Determinista; corre como job de CI junto a meta-analitica.

**Capa 3 — Auditoría estadística (el impulso).** Sobre la serie
acumulada: tendencias (¿la fricción modelada converge a la real?),
cuellos de botella emergentes (¿qué componente pierde más WR?),
atribución acumulada (contribución de cada familia de cambios al
crecimiento, con las sombras como evidencia) y ALERTAS (métrica que
cruza umbral histórico). Esta capa es incremental: empieza con Δ
simple y crece hacia regresión/causalidad según acumulemos serie.

## 3. Primer átomo (este ciclo)

`scripts/informe_integral.py` con TRES emisores iniciales (los que ya
tienen datos concretos y estables): meta-analítica de código,
oráculos con testigo (parsers de oracle_witness_*.log) y censo de
voces del ensamble (el test `#[ignore]` ya produce el recibo). Produce
el primer `informe_2026-10-21.md` — la semilla de la serie histórica.
Los emisores de telemetría-bin y sombras se añaden en átomos
siguientes (requieren deserializar los .bin y leer los TSV de sombras
de Sol — coordinado con él).

## 4. Reglas del sistema (del mandato)

- Serie histórica INMUTABLE: los informes pasados no se regeneran; la
  comparativa vive en el informe nuevo.
- Cada métrica con unidades y commit de origen — sin números huérfanos.
- La auditoría estadística NEVER sobrescribe el juicio del oráculo T-1:
  es observación, no gate.
- Cadencia: por merge (CI) con informe diario consolidado.
