# ADR-0009: Regeneración mensual del manifest de cópulas

- **Estado**: Aceptado (GLM, LXXV, 2026-10-02)
- **Contexto**: la medición de estabilidad LXXV (144 pares en ≥2 meses)
  demostró que λ̂ deriva: mediana 0.106, 53% de pares >0.10, el peor par
  cambia de identidad cada mes. El veto same-bet consume
  `config_dir/copulas_manifest.json`; usar λ̂ de un mes anterior es un
  error sistemático de inflado de cola.
- **Decisión**: el manifest se REGENERARÁ con el tape del mes más
  reciente COMPLETO al iniciar cada mes de operación (misma cadencia que
  la revalidación de modelos, ADR-0008): `copulas_manifest <mes>` +
  commit del diff (el diff es el changelog de dependencia de cola).
  El campo `mes` del manifest ES la vigencia: un mes de desfase es
  deuda visible.
- **No decisión (por ahora)**: ventanas rodantes en vivo / EWMA de λ̂ —
  la deriva justifica el diseño, pero la forma correcta de la ventana
  exige su propia ola con oráculo. La regla mensual es el mínimo honesto
  que elimina el error sistemático.
- **Consecuencias**: 1 corrida de ~10 min por mes; el veto queda con λ̂
  del mes calendario anterior completo (el sesgo residual es intra-mes,
  acotado por ago-sep r=0.893).
- **Deuda menor**: flag `--sin-escribir` en el bin para mediciones sin
  mutar el insumo vivo (incidente LXXV: la medición de sep sobrescribió
  el manifest; restaurado de git).

Relacionado: ADR-0008 (misma cadencia), TRIAGE_TEORICO §LXXI/LXXV.

## Renovación del 2026-10-12 (CL-105, Línea C)

- **Fecha:** 2026-10-12. Decisión del dueño: «Regenerar ya» y «Descargar».
- **Datos:**
  - Periodo: septiembre de 2026 completo, del 2026-09-01 00:00 al 2026-09-30 23:59 UTC.
  - Origen: aggTrades diarios de futuros USDⓈ-M, de data.binance.vision, bajados con `binance_vision_sync --aggtrades --daily`.
  - Cobertura: las 26 monedas del roster, sin días omitidos.
  - El recibo está en `docs/audit/COPULAS_2026-09_2026-10-12.md`.
- **Resultado:**
  - 323 pares con λ̂ ≥ 0,05; antes había 153, de 18 monedas.
  - Los 153 pares de agosto siguen presentes.
  - En ellos λ̂ baja una mediana de 0,045 y ρ sube 0,054: colas algo más finas con más correlación.
  - 44 pares cambian λ̂ en más de 0,10.
- **Arreglos del generador:**
  - ETH entra en agosto: su cinta tiene el nombre histórico `ETHUSDT_AUG_REAL.bin`, y antes sólo se reconocía ese nombre para BTC.
  - SHIB se lee de `1000SHIBUSDT`, su símbolo en futuros.
- **Próximo vencimiento:** al empezar noviembre de 2026, con las cintas de octubre completo.
- **Hallazgo para el roster (no corregido aquí):**
  - El bootloader del host suscribe `shibusdt`, que no existe en futuros USDⓈ-M; el símbolo es `1000SHIBUSDT`.
  - Esa moneda no recibe datos en vivo y sus pares del manifiesto no se usan hasta que el roster la nombre bien.
