# SI — Evidencia espectral y memoria causal de riesgo

Fecha: 2026-10-11. Base: `1d1f8bdd`. Rama: `codex/spectral-evidence-1011`.
Consejo ejecutor: revisión cuantitativa, riesgo, integración y revisión adversarial independientes.
Alcance: `quantum-arena/spectral_tape.rs`, feedback epigenético de `state.rs` el consumidor spectral_bench y sus contratos/CI. No es una certificación de todo el repositorio.

## 1. Qué se está corrigiendo y por qué

El motor puede adaptar parámetros y aun medir mal su propia habilidad. Para que la adaptación tenga evidencia utilizable, cada pronóstico debe quedar ligado a la información disponible cuando se emitió, y un dato rechazado no puede modificar ni su reloj ni sus estadísticas. La memoria de riesgo debe evolucionar desde el estado efectivamente vigente, sin restaurar penalizaciones ya disipadas.

| ID | Hallazgo | Efecto | Estado inicial de esta pasada |
|---|---|---|---|
| SI-01 | Predicción RLS y climatología recalculadas al madurar | Usa pesos aprendidos después de la emisión y ventanas parcialmente solapadas | Reproducir y congelar ambas referencias |
| SI-02 | Warmup decidido al recibir la etiqueta | Puntúa pronósticos emitidos antes de tener la madurez requerida | Congelar elegibilidad al emitir |
| SI-03 | Banco madura antes de validar el trade | NaN, cantidad cero o reloj regresivo pueden mutar aprendizaje y agenda | Prevalidar toda transición |
| SI-04 | ln(0) durante precio constante | Puede contaminar permanentemente climatología y SSE de varianza | No registrar features/objetivos no finitos |
| SI-05 | Mejor nulo ignora SSE cero o inválida | Un modelo peor que un predictor perfecto puede obtener habilidad positiva | Comparar contra ambos nulos válidos |
| SI-06 | Sumas de timestamps sin comprobar overflow | Panic en debug o vencimiento incorrecto en release | Aritmética temporal comprobada |
| SI-07 | Feedback suma al umbral almacenado sin decaer | Resucita castigo o descuento antiguo al cerrar una operación | Aplicar cambio sobre estado decaído |
| SI-08 | Feedback admite PnL NaN/Inf y tiempo anterior | Modifica sesgo, umbral, relojes y bandera con evidencia inválida | Rechazo previo sin mutación |

Las pruebas de septiembre no se recuperaron: el checkout antiguo y su rama ya no existen. Ningún resultado de aquella compilación se presenta aquí como ejecutado. Se reconstruyen contratos contra la base actual, conservando las mejoras posteriores (SSE con olvido y gate contra dos referencias).

## 2. Contrato matemático de la predicción con etiquetas demoradas

Para una emisión en t y horizonte h:

`p_t = persist_t + theta_t' x_t`, `c_t = climatologia_t`, `y_t = objetivo(t,t+h]`.

La pérdida correcta al recibir la etiqueta es `(y_t - p_t)^2`, con `p_t` congelada. Se conserva el olvido existente: `SSE <- lambda*SSE + perdida`. También se congelan `c_t` y la elegibilidad de puntuación. El modelo puede seguir aprendiendo de etiquetas maduras entre emisión y evaluación; ese aprendizaje no reescribe pronósticos anteriores.

Antes, con stride h/4, una muestra emitida en t=25 con h=100 podía puntuarse usando el resultado de (0,100], que contiene información posterior a t=25. Puntuar antes del update de esa MISMA muestra no elimina la contaminación por otras muestras solapadas.

La prueba mantiene un ledger independiente de predicciones emitidas y reconstruye SSE exponencial desde ese ledger. Otra prueba madura juntas muestras emitidas antes del warmup: deben entrenar, pero no contar como pronósticos emitidos con un modelo maduro.

Esto no convierte etiquetas solapadas en independientes. Treinta scores no son treinta apuestas independientes ni una prueba de significancia. También quedan abiertos: estimación de tamaño efectivo, intervalos de habilidad, multiplicidad entre horizontes, selección de modelos y partición temporal externa.

## 3. Gate contra referencias y dominio de datos

El criterio existente declara habilidad positiva sólo si el modelo bate climatología Y persistencia. Para pérdidas cuadráticas válidas: `skill = 1 - SSE_model/min(SSE_clim,SSE_persist)`. Si un nulo tiene error cero, ningún error no negativo puede batirlo estrictamente. Ignorarlo cambia el contrato. SSE negativas o no finitas no acreditan habilidad; retornos y predicciones firmados pueden ser negativos válidos.

Una serie inicialmente plana puede no tener aún escala positiva para transformar varianza con logaritmo. `ln(0)` no se debe convertir en una observación ni en evidencia de volatilidad cero. Se espera soporte finito y se reanuda el aprendizaje cuando aparece. No se inventa un epsilon ni una varianza inicial para hacer pasar el gate.

La validación del banco precede a maduración, ingestión y nueva emisión. El rechazo conserva el estado, incluido reloj y muestras pendientes. Los extremos numéricos sintéticos prueban el dominio del contrato; no se afirma que sean precios realistas del exchange.

## 4. Memoria epigenética

Se conserva la política de relajación actual, `T=900000 ms`:

`q(t) = 1 + (q(t0)-1) exp(-(t-t0)/T)`.

El nuevo cierre debe producir `q_nuevo = clamp(q(t)+delta,0.8,1.25)`. Sumar delta a `q(t0)` y renovar t0 restaura el pasado. Por ejemplo, castigo 1.25 tras 10T está prácticamente en 1; una victoria con delta=-0.019333 debe dar aproximadamente 0.980678, no 1.230667.

Se preservan coeficientes, límites, fallback legacy de timestamp0 y dos cierres distintos en el mismo timestamp. No se declara transacción atómica entre escritores concurrentes: varios campos siguen siendo atomics separados. Otros escritores de last_close_ts en el núcleo requieren su propio contrato de orden e identidad.

## 5. Auditoría de teoría e integraciones avanzadas

Modelo de trabajo recomendado para las siguientes fases: estado por activo y tiempo de evento, observables por log-horizonte con soporte explícito, pronóstico emitido/versionado, resultado posterior y exposición monetaria de cartera. Interpolar escalas no prueba resolución bajo la frecuencia de muestreo, causalidad entre activos ni habilidad predictiva.

| Familia | Uso verificable | Condición antes de integrarla al riesgo |
|---|---|---|
| Aprendizaje con feedback demorado | Separar emisión, etiqueta y update | Predicciones congeladas y pérdidas causales: esta pasada |
| Aprendizaje no estacionario | Adaptar memoria y comparar expertos por horizonte | Validación temporal, referencias competitivas y control de selección |
| Control estocástico / crecimiento log | Relacionar exposición, fricción y riesgo de ruina | Distribución neta fuera de muestra, restricciones monetarias y costes |
| Rough paths, espectros y operadores | Representar dependencia multiescala | Identidad de unidades, soporte, baseline y ablación |
| Hodge / redes multiactivo | Separar componentes de flujos de un grafo definido | Operadores, orientación, producto interno y observaciones justificadas |
| Navier–Stokes / Yang–Mills / analogías cuánticas | Hipótesis de representación que deben especificarse | Variables financieras, unidades, condiciones y predicción contrastable |

Las familias no forman una escalera en la que un nombre más avanzado sustituye automáticamente a otro. Una analogía física no transporta sus teoremas a datos de mercado. Los problemas del milenio plantean problemas matemáticos específicos; su formulación no es evidencia de alpha ni de computación cuántica ejecutada en esta CPU.

Fuentes primarias consultadas: [Joulani, Gyorgy y Szepesvari, feedback demorado](https://proceedings.mlr.press/v28/joulani13.html), que estudia cómo el retraso cambia el regret; [Kuznetsov y Mohri, series temporales y aprendizaje online](https://proceedings.mlr.press/v49/kuznetsov16.html), con garantías bajo su formulación de procesos no estacionarios; [Kelly (1956)](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1956.tb03809.x), crecimiento e información; [formulación Navier–Stokes del Clay Mathematics Institute](https://www.claymath.org/millennium/Navier-Stokes-Equation/). La aplicación propuesta aquí es una inferencia de ingeniería, no una garantía heredada de esos trabajos.

## 6. La meta financiera como contraste

Duplicar cada 72 h exige `g=ln(2)/3=0.231049` de crecimiento log por día, equivalente a t=25.9921% compuesto diario. Son identidades aritméticas, no rendimientos medidos. Esperanza de retorno positiva tampoco basta para crecimiento log positivo: retornos de +50% con probabilidad 0.6 y -50% con 0.4 tienen media +10%, pero crecimiento log esperado≈-0.03398 por apuesta.

Criterio de evidencia: estimar crecimiento log NETO y su incertidumbre por bloques temporales, evaluar bloques temporales fuera de muestra definidos antes de seleccionar modelos, supervivencia/ruina, fricción, capacidad y cartera conjunta. Mantener la brecha a la meta separada de la autorización de operar. Esta pasada no ejecuta trading, promoción, entrenamiento ni calibración de parámetros para alcanzar una rentabilidad prefijada.

## 7. Pendientes cuantitativos distintos

La auditoría inicial señaló dos contratos adicionales de `temporal_spectrum`: retención W1 (lag>=256 puede aliasar historia) y segundo momento anunciado frente a cuadrado del primer momento absoluto. Revalidación actual: existe raw_dev_s2 y S2 usa raw_dev_s2/masa; W1 usa lag físico y mass_history_before con retención acotada. Los hallazgos originales han quedado superados por cambios ajenos y NO se cuentan como abiertos nuevos ni reparaciones SI. No se modifica ese archivo.

Permanecen cuestiones de diseño: gates por habilidad individual frente a promedio entre anclas, soporte de horizontes lentos, unidades y precios sintéticos, feedback seleccionado sólo por operaciones ejecutadas, relojes escritos por varios productores y errores RLS ante dimensiones/estados extremos. No se cuentan como reparaciones de esta rama.

## 8. Integración y validación

Al corte inicial GO24 y MP25 están integrados (merges22bf9d27 y b56aa118; heads34324749 y ef2afefb ancestros de1d1f8bdd). La desaparición de una rama no prueba integración: los objetos históricos26afeb79/48421129/6209704a/f67a23e9 no son ancestros de main. Se preservan worktrees activos y ramas de otros agentes.

Este informe y su JSON son un corte versionado previo al cierre de validación. La [PR #39](https://github.com/Jhona-la/Trader-Gemini/pull/39) conserva el recibo terminal de pruebas/CI, SHA revisado y merge; no se presume aquí el éxito de comprobaciones todavía en curso. Los contratos de arena quedan cubiertos por `unit-suites (espectro-y-consejo)`; se añade `--bin spectral_bench` al job root existente para ejecutar también las regresiones del consumidor. No se rebaja el umbral T1, no se sustituye su fixture y no se afirma que una compilación equivalga a ejecutarlo.

### Evidencia local obtenida

- SI espectral: RED 0/7 sobre base; GREEN 12/0 (siete nuevos más cinco existentes), seguido por dos contratos adicionales de frontera/primer evento. Nueve nuevos finales.
- SI-EPI: RED 2/4; GREEN 6/0. No se cambiaron expectativas existentes para convertir fallos en éxitos.
- Quantum-arena completo `--lib --tests --locked -- --test-threads=1`: 238 aprobadas, 0 fallidas, 0 ignoradas en 21 bloques. Incluye ambas familias una sola vez.
- Commits funcionales: 8404bbdf (espectral), c9d877da (epigenética) y 06ce88ef (consumidor/CI). Primer workspace/all-targets aprobado (exit 0, 13 min 42 s); repetición final tras extender SI-O01 y consumidores/replay aún en curso.
- El witness del ledger NO demuestra que la contaminación siempre infle la habilidad: en esta fixture SSE_model antes 1.68303669 frente a 1.13275649 del ledger; SSE_clim antes 0.79254367 frente a 0.54177094. El defecto es puntuar una predicción diferente, sea más favorable o menos favorable.
- Overflow temporal reproducido como maduración indebida (2 frente a 0); no se afirma que esta configuración del compilador haya producido un panic.

### Límites confirmados en revisión adversarial

SI-O01: se amplía SI-03 al consumidor `src/bin/spectral_bench.rs`. Su helper de ingestión consulta el mismo preflight inmutable antes de `before_trade -> tape.on_trade -> after_trade`; preserva sus cinco objetivos y seis horizontes. Dos contratos ejecutan ese helper: rechazo sin mutación y continuación idéntica al control; ledger de predicciones emitidas con frontera temporal inclusiva. El preflight no reserva una transición ni garantiza concurrencia. No se certifica la población de los cálculos de fusión/IC anteriores a ese bloque. Ejecución final pendiente en este corte.

SI-O02 (P1 científico, abierto): `realized` consulta el suelo de varianza/nocional al madurar; `sigma_at` y `notional_at` invierten con `sqrt(exp(z))` y `exp(z)` sin retirar el desplazamiento de `log(Y+c)`. Son defectos distintos de SI-01 y de la retrans­formación de Jensen. La sigma sí llega al núcleo/riesgo; no se ha medido su impacto empírico aquí.

Contraejemplo verificable: con suelo nocional 100 al emitir, continuaciones de trades `{100,100}` y `{50,150}` tienen total físico 200 pero etiquetas actuales `ln(300)` y `ln(250)`. Con varianza persistente y suelo ambos `1e-4`, RLS sin corrección publica `sqrt(2e-4)` en vez de `sqrt(1e-4)`. Son derivaciones del código, no experimentos de mercado ejecutados.

Contrato propuesto para investigar: `T_s(Y)=ln((Y+c_s)/u_s)` con transformación y unidades versionadas al emitir; `decode(z)=u_s*exp(z)-c_s`, y sólo después raíz para sigma. Congelar `c_s` por Pending no basta: `clim_sum` mezcla logs con desplazamientos distintos y no permite reconversión exacta. Hace falta una transformación fija por generación o resultados físicos suficientes para expresar el baseline en la misma coordenada. Una inversión negativa debe declararse inválida; cero físico y ausencia de evidencia deben distinguirse. Las anclas con transformaciones diferentes requieren reconciliación antes de interpolar. La inversa de un predictor óptimo en log tampoco identifica automáticamente la media física.

Pruebas pendientes de ese rediseño: igualdad de etiquetas para las dos continuaciones de total 200; identidad encode/decode incluido cero; salida física igual a persistencia cuando RLS es cero; invariancia/covariancia de unidades; baseline en la misma transformación y dominio inválido explícito. Se comunicó el hallazgo para coordinar con C-F2; no se modifica su scope en esta rama.

SI-O03: el gate usa habilidad media de anclas maduras para habilitar publicación, sin demostrar habilidad positiva de cada ancla, incertidumbre ni control de múltiples pruebas. SI-O04: no se mide aquí latencia de producción ni coste por evento; el preflight evita alocaciones adicionales pero no implica tiempo garantizado.

SI-EPI usa un escritor serializado como antes. El núcleo escribe también `last_close_ts` fuera de este método; rechazar un evento regresivo local no certifica causalidad de todos esos escritores. La constante 15 min sigue siendo política heredada, no una ley universal del espectro; se preserva para aislar el defecto de composición.

### Precisión sobre el continuo y las unidades del plan

En esta base, la constante ejecutada `SPECTRUM_SCALES_MS` es `10^-6 ms * 4^i, i=0..31`: 1 ns a ≈146 años. El plan rector de 2026-10-11 dice 1 µs a 208 días; esa descripción no coincide con la malla del código (208 días es otro nodo). El reloj del feed sigue siendo entero en milisegundos: ninguno de esos nombres crea observaciones submilisegundo ni siglos de historia. La interpolación logarítmica proporciona consultas continuas sobre una malla finita, no conocimiento completo de todas las frecuencias.

Siguiente orden de investigación, sin activar nuevas apuestas: (1) resolver SI-O02 y validar la transformación del objetivo para que el benchmark mida el mismo contrato; (2) evaluar skill por horizonte con incertidumbre y corrección de selección sobre bloques OOS; (3) resultados de abstenciones con etiquetas observables, distinguiendo predicción de fills contrafactuales; (4) exposición monetaria conjunta con covarianza, soporte y reloj por par; (5) someter cualquier nueva analogía física a dimensiones, nulos y ablación. No se reemplaza una prueba empírica por el nombre de una teoría.

Publicación comprobada de los tres commits funcionales: rama remota y PR #39, head 06ce88efd3fd05638eea2b7cdf0698dcf235e066. Los cinco jobs remotos están en curso y la PR es borrador en este corte. Compilador local: rustc 1.98.0-nightly (096694416 2026-06-29). El recibo terminal se registra en la PR para identificar el SHA exacto sin una cadena de commits de recibos autorreferenciales.
