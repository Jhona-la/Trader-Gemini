# Línea H — cierre del instrumento T-3 v2, 15Oct2026

El cambio implementa un generador gaussiano y lectores de referencia, sin
modificar producción, sizing, vetos ni política de riesgo. Corrige una
refactorización incompleta y varias interpretaciones de resultados históricos.

## Evidencia verificable de esta reanudación

- Modelo estándar compartido `tests/support/t3_synthetic.rs`: **10 passed, 0 failed**
  con rustc nightly fijado, un proceso, sin Cargo ni dependencias Polars.
- Contratos: enumeración exacta de caminos HMM; odds gaussianas extremas;
  dirección anti y umbral de comisiones; causalidad de prefijo y reinicio por
  tramo; TP/SL/vencimientos/censura; controles deterministas V1/V2.
- Huellas de fuentes y comando en `linea_h/validation_2026-10-15.json`.
- Cargo `--workspace --all-targets` e integración con motor: **pendientes** de
  ventana coordinada. Este registro no certifica el motor actual como verde.

## Qué cambia en la interpretación

El posterior HMM es causal respecto de cierres y conoce parámetros y cambios
del generador. Su retorno por señal no acota estrategias selectivas, horarios
de salida diferentes, carteras con operaciones no solapadas o el replay OHLC
intravela. Las variantes siempre activa y con abstención publican cobertura.
Los retornos son logarítmicos por señal; los brackets usan retornos simples,
fills exactos en barreras y SL primero ante ambigüedad OHLC. No son métricas
de ejecución equivalentes a las del motor.

El promedio de R del bracket incorpora vencimientos al cierre de horizonte,
descarta entradas censuradas y evita cruzar uniones de tramo. Sin observaciones
se publica NaN. El EV condicional legado permanece identificado por separado.

Los `ola3_*` recuperados carecen de binario/fuente exactos verificables y son
evidencia histórica, inventariada en `ola3_historical_manifest.json`. No se
sobrescriben ni se atribuyen a Ω128. La diferencia V1/V2 mezcla cambios de
colas, mechas y volumen. Un corte por cierres necesita validar el prefijo y
no atribuye por sí mismo posiciones abiertas entre tramos. Un cuartil no
identifica aprendizaje, inversión ni crecimiento compuesto sostenible.

## Integración

Base remota preparada: `310f729f`; rama de entrega `codex/t3-v2-2026-10-15`.
La preparación histórica encontró un conflicto append del buzón; ese árbol
se conserva. Esta rama parte directamente de main y copia sólo archivos H
verificados byte a byte; sus entradas compartidas son nuevas y acotadas.
Contra main, los únicos cambios en crates son el T-3 y su módulo de apoyo.
CI incorpora el instrumento a Replay contracts y un job estándar pequeño
independiente. Se abrirá un PR listo para revisión antes del merge.
