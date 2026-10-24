# F1-024 — vigencia del análisis temporal forest6

Fecha: 20 de octubre de 2026, America/Bogota. Dueño del contrato: AGY. Fuente pública fija: `597de4af0efae7c64c34b618651c357d406d38c3`. Comparación contra `a3e3ad84707dcaa8bf60492deb116412185aa04e`. Revisión estática, sin ejecución Rust.

El filtro efectivo de NaN/Inf permanece: `get_for_coin_or` devuelve scoped sólo si es finito; después global finito y por último default. Core usa ese getter con defaults timestamp 0 / probabilidad 0.5. Se retira la sospecha de bypass directo no finito. Ausencia global y selección de global finito son casos distintos.

| Bloque contrastado | Líneas en a3e3 | Líneas en597 | Resultado |
| --- | --- | --- | --- |
| Core, lectura/edad/voto/recorte F1-024 |8374–8413|8436–8475|1991 bytes idénticos|
| Registry, get_for_coin_or |309–322|321–334|470 bytes idénticos|
| Daemon, SystemTime Unixms |1078–1081|1062–1065|189 bytes idénticos|
| Daemon, publicación prob/acc/ts |1209–1211|1193–1195|241 bytes idénticos|
| Contrato f1_024_025_026 completo |archivo completo|archivo completo|6450 bytes idénticos|

El host conserva E del mensaje del exchange y lo entrega a Core. El daemon obtiene hora de pared local al comienzo de cada ciclo; el código no usa un reloj monotónico para ese sello. Un sello finito posterior al evento produce edad saturada 0; evento 0 omite la edad si el sello elegido es positivo. Son comportamientos del predicado, sin defecto productivo ni incidencia funcional demostrados. El retraso normal del mensaje puede dar una diferencia entre ambos orígenes temporales; rechazar futuros requiere primero fijar el contrato de disponibilidad/as-of.

Stale neutraliza la probabilidad a 0.5 y el voto del bosque. No rechaza automáticamente una orden. El recorte exige además intención no Flat, cota Wilson > 0.5 y desacuerdo; las demás puertas conservan su autoridad.

La prueba F1-024 (`crates/evolution-engine/tests/f1_024_025_026_daemon_hygiene_contract.rs:114–162`) sigue recalculando localmente el predicado para timestamp 0, edad 16 s y edad 2 s. No invoca Core ni observa su recorte. La identidad del archivo acredita continuidad de ese alcance; no es una búsqueda exhaustiva de toda posible prueba del repositorio.

Pendiente AGY: definir origen/as-of y semántica de evento 0; después contrastar `process_event` real con estados deterministas y controles neutro/fresco-contrario/stale. Primero acreditar que el fresco contrario tiene un efecto observable y que todos los casos alcanzan la etapa medida. Cubrir igualdad 15000 ms / 15001 ms, futuro finito, origen0 y fallback scoped/global/default. No saltar EV/riesgo para forzar una orden ni copiar la fórmula como sustituto de un test del consumidor.

Pines Git actuales:

| Fuente | Blob |
| --- | --- |
| crates/god-engine-core/src/lib.rs |79903563b8c602ed9346374c635f9924dc7ba584|
| crates/omniscient-registry/src/lib.rs |dec997797936b64e9c6569926ddd8852da3da611|
| crates/evolution-engine/src/online_daemon.rs |f8a5a1decf2bbc7353538aebcc7b6ae2ec7b45a5|
| crates/evolution-engine/tests/f1_024_025_026_daemon_hygiene_contract.rs |db456f318c68d039e32319972f0493449d9422f1|
| src/bin/god_engine.rs |2ff210bd397a9dc4be6892c50fecb8dd52eedf23|
| src/parsers.rs |87dc9dbef5528b9d14a73ec99b129c9be2ae624d|

Expediente original externo del 18 de octubre: `base15-f1024-forest6-temporal-ro-2026-10-18-91f3dc25-da49-40d4-a3ea-d1a425faf98b`, manifiesto SHA256 `7d7b3891787d2024808c90866e8ee1f0c6f54d79aae4b528551b14882d4b8455` (7 archivos / 53090 bytes). Esta nota no altera su sello ni atribuye una prueba ejecutada. No modifica estados del censo b9, hallazgos Fable ni los cierres privados ROOT.
