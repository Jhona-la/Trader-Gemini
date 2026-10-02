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
