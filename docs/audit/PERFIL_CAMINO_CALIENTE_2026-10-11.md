# Perfil del camino caliente (ciclo 12, Línea C, 2026-10-11)

> Dónde gasta el núcleo su tiempo y de dónde salen sus asignaciones por
> evento, medido con las herramientas de CL-63 sobre el medidor de CL-60.
> Es la evidencia de la petición del dueño «más potencia con menos impacto
> en nuestros recursos, conservando las velocidades de nanosegundos».
> Ningún número de aquí es de producción: ver §6 antes de usarlos.

## 1. Qué se midió

- Binario: el medidor `crates/god-engine-core/tests/medidor_camino_caliente.rs`
  (CL-60) compilado sobre `c4e8e285` (main con el registro de #720) en
  `release`, `x86_64-pc-windows-gnu`, `CARGO_PROFILE_RELEASE_DEBUG=line-tables-only`.
- Carga: el flujo sintético del medidor, 4 monedas, con el genoma base del
  T-1 y un bosque sintético por moneda (intenciones y cierres reales, como
  muestran las trazas `OPEN/CLOSE TRACE`).
- Entorno: wine en la nube (4 CPU), `MEDIDOR_EVENTOS=3 000 000`.
- Muestras: 300 de CPU (pila completa del hilo que trabaja, a intervalos al
  azar) y 400 de asignación (pila en cada entrada a `__rust_alloc`,
  `__rust_realloc` o `__rust_alloc_zeroed`).
- Error de muestreo: con 300 muestras, una fracción del 36 % tiene un
  intervalo del 95 % de ±5,5 pp (binomial); con 400, una del 33 % tiene ±4,6 pp.

## 2. Método (CL-63)

Bajo wine no hay `perf`, y ni `winedbg` ni `gdb` desenrollan un binario de
Rust para `windows-gnu`: no trae CFI DWARF, sólo la información de
desenrollado de Windows (`.pdata`/`.xdata`).

- `scripts/muestrear_pila_wine.sh PID TID N CARPETA` detiene el proceso con
  gdb, guarda los registros del hilo y 64 KiB de su pila, y lo suelta.
- `scripts/perfil_pila_pe.py MAPS CARPETA EXE` desenrolla cada muestra con
  la `.pdata`/`.xdata` de cada módulo PE mapeado (el exe y las DLL de wine),
  como `RtlVirtualUnwind`, y simboliza con `addr2line -i` (cuenta las
  funciones inlineadas).
- `scripts/tests/test_perfil_pila_pe.py` (7 tests) fija el desenrollado:
  prólogo clásico, prólogo a medias, registro de marco con `alloca`
  dinámico, marco fijado antes de reservar (este caso falló con la primera
  versión, que aplicaba el marco antes de los códigos), función hoja sin
  `.pdata`, parada fuera de los módulos y tamaño de cada código.
- Para las asignaciones se usó el mismo desenrollador con puntos de ruptura
  de gdb en las tres entradas del asignador, sólo en el hilo que trabaja.

## 3. CPU por categoría

| Categoría | Fracción |
|---|---|
| Registro (`omniscient-registry`: búsqueda en el skiplist, `memcmp` de claves, reloj por escritura) | 36,3 % |
| libm de mingw (`exp`, `ln`, `tanh`, `powf`) | 29,3 % |
| Resto del núcleo | 22,7 % |
| msvcrt fuera del registro | 5,7 % |
| Lado unix de wine | 4,7 % |

Inclusivo por función (muestras en cuya pila aparece):

| Función | Fracción |
|---|---|
| `process_tick_dual` | 57,7 % |
| `OmniscientRegistry::get` | 32,3 % |
| `set_for_coin` (incluye su búsqueda) | 29 % |
| `{closure#23}` de `process_tick_dual` | 22,7 % |
| `exp` | 20,3 % |
| `pesos_observables` / `pesos_espectrales` | 14,3 % |
| `spectral_field` | 10 % |
| `ln` | 8 % |
| `chrono::Utc::now` (dentro de `set_value`) | 3,3 % |

Casi todo el msvcrt que aparece como hoja es el `memcmp` de
`above_lower_bound<str>`: el skiplist compara claves `String` en cada
nivel. `set_value` lee `chrono::Utc::now()` en cada escritura.

## 4. Coste de la libm por llamada

Microbanco (`f64::exp` y compañía sobre 5 millones de argumentos, misma
máquina; código en el apéndice):

| Función | windows-gnu bajo wine | glibc nativo |
|---|---|---|
| `exp` | 48 ns | 7,6 ns |
| `ln` | 30 ns | 6,6 ns |
| `tanh` | 34 ns | 18 ns |
| `powf` | 118 ns | 22 ns |

El `exp` de mingw pasa por `__expl_internal` (x87, precisión extendida).
Producción corre `x86_64-pc-windows-msvc` con la UCRT, otra libm: **esta
fracción no se traslada a producción**. Además gnu y msvc pueden diferir en
el último bit, que es la brecha de certificación del punto 1 del contrato
C-F1 (CI, la nube y el T-1 corren gnu; producción, msvc).

## 5. Origen de las asignaciones

Fracción de las 400 entradas al asignador, por el primer marco del
proyecto en la pila (fuentes de `c4e8e285`; las mismas líneas siguen en
`main` 8632ca3d):

| Sitio | Qué asigna | Fracción |
|---|---|---|
| `god-engine-core/src/lib.rs:4660`, `:4687`, `:4726` | `format!("{}_MOTOR")`, `"{}_VOL"`, `"{}_OI"` por evento | 33 % (11 % cada una) |
| `quantum-arena/src/symbol_registry.rs:140` (`try_symbol`) | clona el `String` del símbolo en cada llamada (12 sitios la llaman) | 18 % |
| `signal-engine/src/flow_impulse.rs:305` | dos `format!` de clave por parámetro leído | 15 % |
| `quantum-arena/src/symbol_registry.rs` (`try_spec`) | clona el `SymbolSpec` entero (su `String`) | 10,8 % |
| `signal-engine/src/renyi_tsallis_entropy.rs:247` | dos `format!` de clave por parámetro leído | 10 % |
| `god-engine-core/src/lib.rs:5278`, `:5298`, `:5299` | clona `latest_prices`, `latest_ofis`, `latest_returns` | 7,5 % |
| `quantum-arena/src/spectral_tape.rs:534`, `:755` | vectores de rasgos y de pronosticadores maduros | 4,8 % |
| `quantum-arena/src/state.rs:312` (`sigma_forecast_at`) | vector de σ | 1 % |

Dos tercios de las asignaciones son **claves de texto construidas en cada
evento** para preguntar por algo que no cambia entre eventos (el nombre de
la moneda y sus claves en el registro). Con genoma y bosques el medidor da
19,4 asignaciones por evento (trade 26,2, libro 11,0, vela 24,3; §5b); la
primera lectura de CL-60, sin genoma ni bosques, dio 21,8.

## 5b. Puerta del medidor: PR #38 y coste de #720

Medido con la máquina quieta (sin T-1 ni compilaciones), 5 repeticiones
de 200 000 eventos por corrida, base, rama y base otra vez, con los
binarios de CL-60c (huella `f554aecc59d8`):

| Comparación | Base | Rama | Ruido de la base | Asignaciones por evento | Veredicto |
|---|---|---|---|---|---|
| PR #38: 4bffdd8a frente a 3b5e9050 | 238,5 µs | 236,0 µs | 9,7 µs | 19,424 en las dos | pasa |
| #720: 11cdd14b frente a c4e8e285 | 222,6 µs | 237,8 µs | 5,9 µs | 19,424 en las dos | no pasa |

Las dos corridas dan 42 intenciones y 44 cierres por repetición: la cinta
llega al riesgo y a la gestión de posiciones.

**#720 cuesta 15,2 µs por evento (+6,8 %) sin asignar más.** El
microbanco del medidor lo explica: `set_for_coin` pasa de 161,6 ns a
234,4 ns (+73 ns por escritura) y `chrono::Utc::now` cuesta 58–60 ns bajo
wine. 15,2 µs entre 73 ns da unas 208 escrituras al registro por evento,
dentro de la estimación de 135–280 que se hizo leyendo el código. Por
tipo de evento (primera corrida de la base frente a la rama), los trades
suben de 235,9 a 249,9 µs y el libro de 206,1 a 219,9 µs. En Windows nativo el reloj preciso cuesta menos que bajo wine,
así que la cifra de producción será menor; la medida en el PC del
operador (P4) la dará.

Hoy nadie lee ese sello en producción. `Parameter::get_timestamp`
(crates/omniscient-registry/src/lib.rs:73) sólo lo llaman tests;
`take_snapshot` lo guarda (:394) y `restore_from_snapshot` lo ignora
(:400-408). El grep de `get_timestamp()` y `.timestamp.load` fuera del
registro y de los tests sólo da el anillo de telemetría, que es otra
estructura (crates/telemetry-server/src/tensor_telemetry.rs:238). P1 puede
quitar la lectura del reloj sin perder nada vivo; si un lector de frescura
llega a necesitarlo, el sello debe ser el del evento, que el núcleo ya
tiene y que en un replay es el único reloj coherente con la cinta.

## 6. Límites de estas medidas

- Wine no es Windows y gnu no es msvc: el reparto entre registro y núcleo
  debería sostenerse (es código Rust), la fracción de libm no.
- El flujo es sintético; una cinta real cambia la mezcla de ramas.
- Cada muestra detiene el proceso unos 0,7 s con ptrace; el intervalo al
  azar evita sincronizarse con el trabajo, pero las cachés se enfrían.
- `line-tables-only` y el inlining reparten el tiempo de una función
  inlineada entre ella y su llamadora: el perfil inclusivo cuenta ambas.

## 7. Propuestas (cada una con su dueño y su prueba)

| Id | Propuesta | Dueño | Prueba que la acepta |
|---|---|---|---|
| P1 | Registro con asas pre-resueltas: cada lector resuelve su clave una vez y guarda un índice a un `AtomicU64`; la escritura deja de leer el reloj del sistema (usa el ts del evento o un reloj grueso). Ataca hasta un tercio de la CPU. | AGY (#720, registro) | Contrato de #720 intacto (desconocido no es cero, aislamiento por activo) y la puerta de rendimiento del medidor. |
| P2 | Pesos espectrales una vez por evento: `pesos_observables` recalcula 32 `exp` en cada lector (11 sitios de llamada en `temporal_spectrum.rs` y el núcleo). `scales` es público y los tests lo mutan sin `update`, así que una caché interna se quedaría vieja; la vía segura es una instantánea de pesos por evento que el núcleo pasa a los lectores. | Línea C (C-F6) | Igualdad bit a bit de todos los lectores contra la versión actual sobre el flujo del medidor, golden intacto y puerta de rendimiento. |
| P3 | libm en Rust puro (el crate `libm` ya está en `Cargo.lock`): mismo resultado en gnu y msvc, y más rápida que la de mingw. Cambia la numérica de producción, así que re-certifica goldens. | Decisión del dueño | Goldens re-certificados con su diff explicado y T-1 ≥ trinquete. |
| P4 | Medir en el PC del operador con msvc: el mismo medidor (`scripts/medir_camino_caliente.sh` o `cargo test --release -p god-engine-core --test medidor_camino_caliente -- --nocapture`). | Operador | Tabla `MEDIDOR` de producción en el buzón. |
| P5 | Claves y nombres por moneda construidos una vez: `try_symbol` y `try_spec` devuelven una referencia compartida (`Arc<str>` o `Arc<SymbolSpec>`), las claves `{sym}_MOTOR/_VOL/_OI` y las de los motores de señal se precalculan al registrar el universo, y los vectores `latest_*` reutilizan su búfer. | Línea C propone; cada dueño de archivo decide | La puerta del medidor exige que la asignación por evento no suba; ésta debe bajarla, con las mismas intenciones y cierres. |

## 8. Reproducir

```bash
# 1. Medidor con símbolos de línea.
CARGO_PROFILE_RELEASE_DEBUG=line-tables-only CARGO_PROFILE_RELEASE_STRIP=none \
  cargo test --release --locked --target x86_64-pc-windows-gnu \
  -p god-engine-core --test medidor_camino_caliente --no-run
# 2. Correrlo bajo wine con muchos eventos y buscar el hilo que trabaja.
MEDIDOR_EVENTOS=3000000 wine <exe> --nocapture &
ps -L -o tid,pcpu,comm -p <pid>
# 3. Muestras de CPU y perfil.
scripts/muestrear_pila_wine.sh <pid> <tid> 300 m1
scripts/perfil_pila_pe.py m1/maps.txt m1 <nombre del exe>
```

## Apéndice: microbanco de la libm

```rust
use std::hint::black_box;
use std::time::Instant;
fn medir(nombre: &str, f: impl Fn(f64) -> f64) {
    let n = 5_000_000u32;
    let t = Instant::now();
    let mut acc = 0.0;
    for i in 0..n {
        acc += f(black_box(-(i as f64) * 1e-6));
    }
    black_box(acc);
    println!("{nombre}\t{:.1} ns", t.elapsed().as_nanos() as f64 / n as f64);
}
fn main() {
    medir("vacío", |x| x);
    medir("f64::exp", f64::exp);
    medir("f64::ln", |x| (1.0 - x).ln());
    medir("f64::tanh", f64::tanh);
    medir("f64::powf", |x| (1.0 - x).powf(0.7));
}
```
