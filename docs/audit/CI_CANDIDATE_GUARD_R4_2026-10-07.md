# R4-CI-01: validar el candidato original sin mover HEAD

Fecha: 2026-10-07, America/Bogota. Auditor: Codex / `system_inventory`.
Base: `42652f70badadb9be66c0fb138b88918c8cc54dc`.
Severidad: MED. Estado: **CORREGIDO_Y_VERIFICADO_EN_FIXTURE_GIT**;
fix incorporado al candidato `7acf36aa`; CI remoto del conjunto aún pendiente.

## Defecto y cambio

El workflow normalizaba finales de línea y creaba un commit local antes de
ejecutar `git diff --check HEAD^ HEAD`. Si la normalización modificaba algún
archivo, `HEAD` dejaba de ser el candidato del checkout. El guard pasaba a
examinar sólo la normalización, omitiendo marcadores de conflicto añadidos por
el candidato original. En documentos Markdown, esos marcadores tampoco tienen
por qué ser detectados por la compilación de Rust.

Se elimina íntegramente el paso normalizador, incluidas las escrituras, cambios
de configuración Git, staging y commit. El guard existente queda inmediatamente
después del checkout:

```powershell
git -c core.whitespace=-blank-at-eof,-blank-at-eol diff --check HEAD^ HEAD
```

Esta configuración ya tolera blancos cosméticos al final del archivo/línea.
El guard inspecciona el candidato original y no modifica ni el árbol de trabajo
ni `HEAD`. Se conservan el checkout fijado por SHA, `fetch-depth: 2`, el compiler
`nightly-2026-06-30`, las suites, concurrencia y límites existentes.

Las suites de parser, riesgo no finito y frontera OOS de la rama `root-audit`
todavía no están en esta base. La eliminación está limitada al paso normalizador:
**esas suites deben conservarse al integrar la otra rama**.

## Prueba de comportamiento, sin Cargo ni red

Se crearon dos repositorios Git temporales independientes con identidad local
de fixture, `commit.gpgsign=false` y `core.autocrlf=false`. No se tocó el historial
del proyecto. El comando completo de fixture terminó con **exit 0** después de
comprobar mediante aserciones todos los resultados siguientes.

| Escenario | Comando / revisión | Exit observado | Resultado |
|---|---|---:|---|
| Candidato añade tres marcadores y un blanco EOF | Guard directamente sobre candidato `7619eb07` | **2** | Detecta los tres marcadores |
| Paso antiguo normaliza ese candidato y crea commit `0ed0a10f` | Mismo guard sobre nuevo `HEAD^ HEAD` | **0** | Falso verde; los marcadores siguen en `case.md` |
| Candidato independiente añade únicamente blanco EOF | Guard sobre candidato `de972e67` | **0** | Cosmética aceptada sin normalizar |

El guard directo conservó tanto el SHA del candidato como el árbol limpio.
El paso antiguo sí desplazó `HEAD`. Salida del rechazo correcto:

```text
case.md:3: leftover conflict marker
case.md:5: leftover conflict marker
case.md:7: leftover conflict marker
```

Fixture conservado para inspección local:

```text
C:/Users/jhona/AppData/Local/Temp/trader-gemini-ci-guard-d7711d6fa35f492aa7f29dc69df5e213/
  marker-and-blank-eof/
  cosmetic-only/
  proof.json
```

Identificadores completos:

- Candidato con marcadores: `7619eb07b6869cc96aa3064f3f6899248bb51fcc`.
- Commit del normalizador antiguo: `0ed0a10f1f8caf735182a17b273ae29831926cf9`.
- Candidato cosmético: `de972e678bfb9dfeda0c821b69023ac74293124a`.

Comandos para repetir los tres guardas contra las revisiones observadas, incluso
después de que el normalizador haya movido `HEAD` del primer repositorio:

```powershell
$fixture = 'C:/Users/jhona/AppData/Local/Temp/trader-gemini-ci-guard-d7711d6fa35f492aa7f29dc69df5e213'
git -C "$fixture/marker-and-blank-eof" -c core.whitespace=-blank-at-eof,-blank-at-eol diff --check '7619eb07^' 7619eb07
# $LASTEXITCODE = 2
git -C "$fixture/marker-and-blank-eof" -c core.whitespace=-blank-at-eof,-blank-at-eol diff --check '0ed0a10f^' 0ed0a10f
# $LASTEXITCODE = 0, aunque case.md conserva los marcadores
git -C "$fixture/cosmetic-only" -c core.whitespace=-blank-at-eof,-blank-at-eol diff --check 'de972e67^' de972e67
# $LASTEXITCODE = 0
```

Para reconstruir la prueba desde cero: crear un `case.md` base con dos líneas,
commitear; añadir un bloque de conflicto de tres marcadores seguido de dos
saltos de línea y commitear. Ejecutar el guard: rechaza. Aplicar la expresión
PowerShell usada por el paso retirado:

```powershell
$normalized = $content.TrimEnd("`r`n") + "`n"
```

Guardar y commitear esa normalización y repetir el guard: acepta incorrectamente
ese último diff. En el segundo repositorio, añadir sólo el salto de línea final
extra: el guard acepta directamente.

## Alcance de la evidencia

La prueba demuestra la semántica Git del bypass y su eliminación. No es una
ejecución de GitHub Actions ni una certificación de todas las suites Rust. El
guard sólo examina cambios del candidato contra su primer padre, como antes;
no se lo presenta como un escáner de todo el historial.

## Composición sobre el candidato local del 8 de octubre

El candidato `7acf36aa` conserva checkout, compiler fijado, concurrencia,
all-targets y todas las suites del padre `18bbd1d9`; añade los contratos
recuperados de parser, riesgo/C07, OOS, SA, reloj, evidencia y MW. La
comparación contra ambos padres mantuvo la retirada del normalizador:
el guard observa el candidato original. El guard con blancos cosméticos
tolerados pasó en todo el rango `18bbd1d9..7acf36aa`.

Estos son recibos locales y una revisión del workflow. El último corte
remoto consultado registraba Replay contracts run 37822504981 sobre
`18bbd1d9` en progreso; no es un éxito de esta recuperación. Consultar el
recibo de publicación para el estado de CI del SHA efectivamente publicado.
El guard sigue limitado al primer padre del candidato; no analiza todo
el historial ni una sucesión de commits ocultada en un push.
