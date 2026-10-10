# Preparación del examen económico OOS — R4

Fecha: 2026-10-07. Autor: Codex / subagente fundamentos.
Base inspeccionada: `639c3e0d7c5f0dc8c376ac04af2a2fdc7a5bc5a9`, worktree
`quant-foundations-2026-10-07`. Revisión estática de código/documentación y
metadatos de archivos; **sin Cargo, tests, replay, entrenamiento ni lectura
de filas de datasets o logs privados**. No es un resultado económico.

## Dictamen acotado

El plan `PLAN_REVISION_ARCHIVO_POR_ARCHIVO_2026-10-07.md:19–49,207–219`
define correctamente la meta aspiracional, patrimonio MTM, separación de
aportes/retiros, costes y dependencia entre ventanas solapadas. Hace falta
convertir esos requisitos en un protocolo congelado y recibos verificables
antes de llamar a una corrida «examen final sellado». Se puede preparar el
instrumental; esta revisión no acredita que el examen esté listo.

«Consistente» aún necesita criterio previo: población de periodos, frecuencia
tolerada de incumplimiento, severidad, incertidumbre y restricciones de riesgo.
Un equivalente geométrico de 72 h calculado entre dos extremos no demuestra
que cada ventana de 72 h duplique el patrimonio. La insolvencia se conserva
como resultado adverso; no se elimina ni se transforma en dato ausente.

## Bloqueos y evidencia observada

1. **Medición de cartera pendiente.** `booktick_replay.rs:243–244` declara un
   símbolo; `576–590` calcula DD/final con `unified_capital`, y `609` pasa el
   span completo del tape al panel, incluido el prefijo de warm-up. No entrega
   una curva de patrimonio multiactivo MTM en reloj común. Su Sharpe es PnL
   por trade (`213–214,591–599`), distinción explícita también en
   `metrics.rs:12–17`. No sustituye retornos de cartera ni incertidumbre sobre
   periodos de 72 h. Recibo faltante: inicio económico tras warm-up, saldo +
   todos los slots abiertos − costes pendientes, precios/as-of, posiciones
   terminales y reconciliación con fills y flujos externos.
2. **Paridad de costes sin certificar.** El replay tiene `shift_atr_frac=0.10`
   por defecto (`booktick_replay.rs:176–198`); su propia documentación explica
   ensanchamiento del libro y doble conteo de slippage frente al vivo. No se
   propone cambiarlo en esta ola. Falta fijar configuración, física y fuente
   de cada coste y probar reconciliación: spread una vez, fees por lado,
   slippage, funding liquidado, parciales, rechazos, margen y liquidación.
   Tener funding como feature no acredita que se haya debitado su coste.
3. **Sellado/procedencia no acreditados.** `models_manifest.json` inventaría
   20 fuentes (generado 2026-10-02T05:55:12.066Z) con hashes y estructura.
   Su esquema `ml_registry.rs:34–78` no registra hashes de tapes, fechas de
   entrenamiento/selección/test ni decisiones de promoción. ADR-0010:129–174
   ya publica resultados ago/sep-14 y decisiones posteriores de calibración;
   esos cortes no son un examen final nuevo para decisiones informadas por
   ellos. ADR-0008 registra modelos promovidos con tapes hasta sep-14/17.
   Se requiere ledger de usos/consultas, intervalo final no usado, hashes,
   cobertura temporal por activo y purga por extremo de etiqueta, no sólo
   nombres distintos de archivo o un split 70/30.
4. **Vigencia y alcance distintos de integridad.** El manifest de cópulas
   tiene `mes=2026-08`; ADR-0009 prescribe el último mes completo para el mes
   operativo. No es recibo vigente de octubre. Para replay histórico también
   debe acreditarse que la estimación usa exclusivamente información disponible antes de cada decisión evaluada.
   L2 v1 sigue bloqueado en ADR-0010 por falta de transferencia de calibración;
   eso condiciona habilitar L2, no impide examinar un incumbente congelado que
   lo mantenga deshabilitado. Deben quedar explícitos módulos, flags y roster.
5. **Incertidumbre y selección abiertas.** Ventanas disjuntas de 72 h eliminan
   el solape mecánico, pero no garantizan independencia. Falta preregistrar
   reloj, anclas, tratamiento de huecos/periodos sin operar, estimador e
   intervalos que preserven dependencia y suficiente soporte temporal. Las
   ventanas móviles pueden ser diagnósticas; no se cuentan como nuevos
   ensayos independientes. `selection_stats.rs` conserva las limitaciones
   de dependencia/multiplicidad documentadas en `FUNDAMENTOS_R4_2026-10-07.md`;
   sustituir n por un escalar «efectivo» sin validar el estimador no las cierra.

Metadatos locales: se listaron únicamente nombres/tamaños/fechas de archivos
en `C:/Users/jhona/Documents/Proyectos/Trader Gemini/data`. Entre los
`*REAL*.bin` directamente bajo esa carpeta hay nombres junio–agosto,
sep-14 y sep-17; no apareció un nombre de septiembre completo u octubre.
También existen subdirectorios históricos/raw/vision. **No se inspeccionó
su contenido ni se certificó ausencia de datos posteriores**; el nombre y
mtime tampoco certifican la cobertura real. No se hashearon tapes grandes.

Otro harness, `continuous_evolution_backtest.rs:675–711`, sí suma posiciones
abiertas al cierre diario con una hipótesis de comisión de salida. Esto evita
afirmar ausencia universal de MTM, pero no acredita la curva, paridad ni el
examen sellado exigidos arriba; requiere su propio recibo económico.

## Próximo lote concreto y recibos de salida

| Fase | Archivos prioritarios | Recibo requerido antes del examen |
|---|---|---|
| R1 | `crates/backtest-engine/src/booktick_replay.rs`; `src/bin/god_engine.rs`; `crates/god-engine-core/src/lib.rs` | Productor→consumidor de eventos, posiciones, saldo y marcas por activo; ruta replay/sombra/vivo, fuentes presentes/ausentes y reloj común. |
| R2 | `crates/backtest-engine/src/metrics.rs`; `crates/risk-engine/src/selection_stats.rs`; `docs/adr/ADR-0003-doctrina-meta-geometrica.md` | Definición firmada del estimando 72 h, hipótesis/criterio de aceptación, dependencia, riesgo, familia de ensayos y presupuesto de consultas; resultado adverso y no identificado diferenciados. |
| R5 | `booktick_replay.rs`; `crates/god-engine-core/src/lib.rs`; `crates/backtest-engine/src/bin/continuous_evolution_backtest.rs`; módulos de fills/fees alcanzados desde ellos | Curva de equity reproducible, conciliación contable por evento y terminal, costes/estrés declarados, sizing/capacidad y paridad comprobados. |
| R6 | `src/bin/evolution.rs`; `src/bin/train_forest.rs`; `crates/god-engine-core/src/ml_registry.rs`; `config_dir/models_manifest.json`; `config_dir/copulas_manifest.json` | Identidad efectiva de modelos/genoma/configuración; genealogía y ledger de todos los usos de datos; train/selección/final separados; incumbente congelado, reglas de adaptación preregistradas y estado inicial identificado, promoción/rollback sin consultar el examen para retocar. |

El paquete previo mínimo contiene: SHA integrado y blobs; toolchain/semillas;
hashes y disponibilidad causal de datos/modelos/macro; universo y ventanas
UTC; manifiesto de costes y recursos; protocolo/criterios congelados; ledger
de consultas/ensayos; y recibo del harness con fixtures contables conocidas.
Después se ejecuta según protocolo preregistrado, conservando salidas completas,
fallos, ventanas sin operar y desviaciones. Réplicas y semillas deben estar
previstas; una reejecución idéntica comprueba reproducibilidad y no aporta
evidencia independiente. Un resultado inesperado se conserva como resultado
del examen. Si se usa para retocar decisiones o seleccionar candidatos, ese
uso convierte el examen en evidencia de selección y exige una reserva nueva
para el examen final.

ADR-0003 y el encabezado de `metrics.rs` conservan la doctrina histórica
«maximizar crecimiento sujeto a riesgo» y una afirmación categórica sobre
insostenibilidad. El plan actual sí representa la solicitud reciente como
aspiración a medir. Conviene enlazar ambos con fecha/precedencia, sin convertir
esa afirmación histórica ni su opuesta en teorema demostrado.

## Alcance de integración y hashes

La base anterior precede la recuperación OOS/SA/forest. Inspección adicional
del worktree de integración con HEAD
`3e82e64d6f62d9d3d93c1cd0a4d3e988208faf41`, aún con merge de feature-clock en
curso: el diff contra `639c3e0d` de
`crates/backtest-engine/src/booktick_replay.rs` y
`crates/backtest-engine/src/metrics.rs` no mostró delta;
`src/bin/evolution.rs` sí incorpora recuperación OOS/SA. Su frontera y
warm-up corregidos no equivalen a sellado, MTM o prueba económica. Forest
mejora evaluación local de drawdown; esta nota no reutiliza una prueba suya
como actual. **No se certifica el árbol final**: verificar estos consumidores
otra vez después de todos los merges. No se ejecutaron pruebas nuevas aquí.

Git blobs exactos de la base inspeccionada (SHA-1 de objeto Git, no SHA-256):

| Ruta | Blob |
|---|---|
| `docs/PLAN_REVISION_ARCHIVO_POR_ARCHIVO_2026-10-07.md` | `4d2902323ae179ab64d6adf061a1db44379c3315` |
| `crates/backtest-engine/src/booktick_replay.rs` | `1ed78cc9673f1300b2c44a418360f53e9e5d86f5` |
| `crates/backtest-engine/src/metrics.rs` | `bf99c0bc722e9efda25c5769ee7f0690a1b4e52e` |
| `crates/backtest-engine/src/bin/continuous_evolution_backtest.rs` | `114ab1a47db5d8c16796962a845f16f0d3d29730` |
| `crates/risk-engine/src/selection_stats.rs` | `2bb2fa0d0a413a3d386852341318497959324298` |
| `crates/god-engine-core/src/ml_registry.rs` | `a6d3f6945f24a3d3f4e49f57c1bfaca0afdcd055` |
| `src/bin/evolution.rs` | `44d7d0f58a4cea8efcaf1cc788ad69094a47c7b9` |
| `src/bin/train_forest.rs` | `ff7ebe4ea364d28083fda6f1a32e85d5f0d4f777` |
| `config_dir/models_manifest.json` | `4884fbc9bd4e69cc10b9c2defa39f45f131e790d` |
| `config_dir/copulas_manifest.json` | `e6e0001d01b66eec3ebc1968d9638dea2804d2ce` |
| `docs/adr/ADR-0003-doctrina-meta-geometrica.md` | `84f78fc952c9ba5bfa34714366c1878f83410757` |
| `docs/adr/ADR-0008-reloj-revalidacion-modelos.md` | `d6a71902f9f7ff3d875cacfdc0c91ab1cece4a9c` |
| `docs/adr/ADR-0009-regeneracion-mensual-copulas.md` | `468056f9821464ccbe03ff42d04f6f71a0b63d85` |
| `docs/adr/ADR-0010-arquitectura-dl-modular.md` | `7ba801f417e4e31bff7de6d2a29caf6093ce557c` |

## Revalidación de alcance sobre el candidato integrado

El candidato de código `f3f86960190d99aeacdb75b14cdfab0adceaa186` conserva
13 de los 14 blobs anteriores. El único delta es `src/bin/evolution.rs`,
ahora blob `30f34f1b205edf7929fbcfa8ace181d5a546e1d4`:

- OOS valida segmentos antes de activar modelos y usa el contexto IS
  disponible como warm-up. No cambia el span completo del panel de
  `booktick_replay` ni acredita sellado de datos.
- SA usa selección explícita, penalización con signo y campeón finito;
  deja de presentar una utilidad interna como crecimiento de tres días.
  No añade equity MTM, costes o intervalos económicos.

Los cinco bloques de recibos faltantes permanecen abiertos. La comparación
de objetos no es revisión de todas las dependencias ni del motor vivo.
El plan recibe después un cierre documental y el ledger actualizado; esa
edición de documentación no añade una corrida económica a esta revisión.

## Evaluación de una política que aprende

Congelar la política de evaluación no exige impedir las actualizaciones
causales prescritas por esa política. Antes del examen deben quedar fijas
las reglas de aprendizaje, maduración de etiquetas, frecuencia, selección,
límites, promoción y reversión, junto con el presupuesto de consultas y
ensayos. Durante la corrida se registran las actualizaciones y su conjunto
de información disponible; una decisión no puede usar una etiqueta que
aún no había madurado. Retocar esas reglas después de consultar resultados
convierte el examen en evidencia de selección y exige un examen nuevo.

Comparar una política adaptativa con un incumbente congelado, y reportar
tanto su ruta de estados como su patrimonio, permite estudiar adaptación
sin confundir causalidad de las decisiones con una mera partición de filas.
Este apartado es una especificación pendiente de implementación y pruebas.

## Revalidación del 8 de octubre

La [adenda OOS actual](REVALIDACION_OOS_R4_2026-10-08.md) compara 14 blobs
entre base639, remoto18bb y árbol integrado f26: cinco cambian y nueve
permanecen iguales. Reconoce el avance SOL-R5-01 de reporting diario
telescópico; no presenta la medición anterior como estado actual. Permanecen
abiertos patrimonio/DD terminal de replay, duración procesada, costes,
procedencia de modelos/datasets y custodia del examen. La fecha de creación
del artefacto no sustituye disponibilidad causal de sus datos y etiquetas.
