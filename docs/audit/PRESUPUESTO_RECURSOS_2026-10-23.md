# Presupuesto de recursos medido — 2026-10-23 09:00–09:40 Bogotá

Medición directa sobre la máquina del operador (Windows, 23,4 GB visibles).
**Nada de lo que sigue es estimación: cada cifra tiene su comando.** El objetivo
es responder a la pregunta del operador («el proyecto está ocupando mucha
memoria y la va a desbordar») distinguiendo **qué recurso** se está desbordando,
porque la respuesta cambia por completo la política que deben seguir los agentes.

## 1. Veredicto

| Recurso | Medido | Límite del mandato | Estado |
|---|---|---|---|
| RAM disponible | **1,6–1,8 GB** de 23,4 GB | 16 GB de consumo total | **EN CRISIS** |
| Memoria comprometida | **54,2 GB** (contra 23,4 GB físicos) | — | **EN CRISIS** |
| Pagefile | 38,1 GB asignados, 17 % usado, pico 9,1 GB | — | paginando |
| Fallos de página | **120.074 /s**; 2.692 páginas/s a disco | — | **tormenta de paginación** |
| Pool paginado del kernel | **9,6 GB** | (<1 GB típico) | **anómalo** |
| Procesos vivos | **669**; 19,6 GB de working set | — | flota de IDEs/agentes |
| Disco del proyecto | **~71 GB** (target 41,3 + data 23,9 + .git 3,3) | 300 GB | **holgado** |
| Disco libre en C: | 335 GB | — | holgado |
| `target/` | 41,3 GB en **141.747 archivos** | — | un solo target (correcto) |

**El recurso que se desborda es la RAM, no el SSD.** El proyecto usa el 24 % del
presupuesto de disco; le sobran 229 GB. En cambio no le queda ni el 8 % de la RAM.

## 2. El síntoma que explica la lentitud de todo

Coste real de **crear un proceso** en esta máquina (medido con `Stopwatch` sobre
ejecutables triviales, fuera de MSYS):

| Proceso | Coste por arranque |
|---|---|
| `hostname.exe` (imprime el nombre y sale) | **520 ms** |
| `where.exe` | 476 ms |
| `git.exe --version` | **963 ms** |
| `bash -c 'exit 0'` (fork MSYS) | **1.510 ms** |
| `date +%s` ×5 desde bash | 4.932 ms (**~1 s cada uno**) |

Lo normal en Windows es 20–50 ms. Estamos **10–25× por encima**. Con ese coste:

- el semáforo de compilación tardaba **19–33 s sólo en decidir** (lanza ~30 procesos);
- `cargo` paga ese peaje en cada `rustc`, cada `link.exe` y cada script de build;
- un `cargo check` de workspace multiplica por decenas su tiempo de pared sin que
  el CPU de cómputo sea el culpable.

La causa es coherente con los números de arriba: con 1,6 GB disponibles, cada
arranque de imagen obliga a **traer páginas del pagefile** (120.074 fallos/s), y
MsMpEng (Defender) añade un escaneo por cada ejecutable que se carga.

## 3. Consumidores

Working set por proceso (los 15 mayores, `Get-Process`): Hermes 0,96 GB, WorkBuddyAI
0,80, codex 0,78, language_server 0,65, ChatGPT 0,66 + 0,40, msedgewebview2 0,53,
claude 0,36, python 0,50 + 0,63 (privada), ZCode 0,46 + 0,32, Antigravity 0,28,
MsMpEng 0,46. **Ninguno es un derroche individual**: el problema es la suma (669
procesos) sobre una máquina de 23,4 GB.

Hallazgo aparte: **dos `find /` huérfanos de Hermes** (`C:\Users\jhona\AppData\Local\hermes\tools\git-…\usr\bin\find.exe`,
PIDs **174452** y **100964**) llevan desde las **01:13** recorriendo la raíz del
disco buscando `job-watch-state*` y `magneto_empleos_pages*`. Siete horas
golpeando el disco con 141.747+ archivos en `target/`. Consumen 12 % de CPU y son
la peor fuente de I/O del sistema ahora mismo.

## 4. Lo que NO hay que hacer (y por qué la política actual está invertida)

`scripts/higiene_recursos.sh` y `scripts/barrido_target.sh` están diseñados para
**liberar SSD**. Hoy eso es optimizar el recurso abundante:

- barrer `target/` ahora borraría 30 GB de artefactos **para forzar una
  recompilación completa** que esta máquina no puede pagar (cada spawn cuesta 1 s
  y no hay RAM para 2 `rustc`). Se cambia un recurso holgado por el recurso
  escaso: **no se barre**.
- `scripts/disk_audit.sh` recorre los 141.747 archivos con `find`/`du` sin
  acotar. Medido hoy: dos invocaciones mías fueron terminadas por tiempo de
  espera y **dejaron los `find`/`du` hijos huérfanos** consumiendo CPU y disco.
  Medir el disco no puede costar más que el problema que mide.

## 5. Política que deben seguir todos los agentes (ordenada por efecto/riesgo)

1. **Matar los dos `find /` huérfanos** (PIDs 174452, 100964). Efecto inmediato
   sobre I/O; riesgo nulo (nadie espera un `find` de siete horas).
2. **Cerrar los hosts de agente inactivos** (ChatGPT ×2, claude, ZCode ×2,
   Antigravity). Recupera varios GB de working set sin tocar el repositorio.
3. **Excluir del escaneo en tiempo real de Defender** la carpeta del proyecto y
   `target/`, más `cargo.exe`, `rustc.exe`, `link.exe`. Ataca a la vez el coste
   por spawn y el pool paginado del kernel.
4. **Reiniciar** cuando se pueda: los 9,6 GB de pool paginado del kernel no se
   liberan de otra forma, y son 9,6 GB que la RAM no puede usar.
5. **No medir el disco mientras se compila.** Si hace falta el mapa, medir por
   directorio de primer nivel con presupuesto de tiempo, nunca la raíz entera.
6. **La puerta de admisión se mide sobre memoria DISPONIBLE**, no sobre memoria
   libre (ver §6). Un `cargo` sólo entra si hay ≥4 GB disponibles.

## 6. Cambio de código de este ciclo (F1-097)

`scripts/cargo_semaforo.sh` — la puerta que gobierna **todos** los builds del
proyecto medía dos magnitudes distintas según la plataforma:

- Linux: `MemAvailable` (libre + reclamable). Correcto.
- Windows: `FreePhysicalMemory` (sólo páginas libres). **Incorrecto**: deja fuera
  la lista de standby, que en una máquina de builds es la mayor parte de lo
  reclamable. El semáforo anunciaba «2 GB» donde el Administrador de tareas
  mostraba 9, y los agentes aprendían a compilar **por fuera del cerrojo** — que
  es exactamente el fallo de coordinación que el semáforo existía para evitar.

Ahora ambos sistemas miden lo mismo (`AvailableMBytes` de
`Win32_PerfRawData_PerfOS_Memory`, propiedades CIM en inglés, así que no depende
del idioma de Windows). Además, una medición ilegible se declara
`ram_libre=desconocida` en la bitácora en vez de coercionarse a `0` (un valor
ilegible no es cero) y el mensaje de bloqueo dice por qué.

Contratos nuevos en `scripts/tests/test_cargo_semaforo.py`: la medición real no
tenía **ni una** prueba — todas inyectaban `SEMAFORO_RAM_GB` y saltaban justo el
camino que estaba mal. Ahora se fija la magnitud pedida (`AvailableMBytes`, nunca
`FreePhysicalMemory`), el valor medido que decide, y que una lectura ilegible no
se convierte en cero ni habilita saltarse la paciencia.

## 7. Cómo reproducir

```
Get-Counter '\Memory\Available MBytes','\Memory\Page Faults/sec','\Memory\Pages/sec'
(Get-CimInstance Win32_OperatingSystem) | Select FreePhysicalMemory,TotalVisibleMemorySize
Get-CimInstance Win32_PageFileUsage | Select AllocatedBaseSize,PeakUsage
Get-CimInstance Win32_PerfRawData_PerfOS_Memory | Select AvailableMBytes
Get-Process | Measure-Object WorkingSet64 -Sum
Get-ChildItem <raíz> -Recurse -Force | Measure-Object Length -Sum
```
