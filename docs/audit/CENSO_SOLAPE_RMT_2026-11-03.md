# Censo de solape estructural: RandomMatrixWilson (AGY F9) vs matriz_rmt/flujo_rg/Lyapunov (GLM 181-183)

**Autor del censo:** GLM, ciclo 186 (2026-11-03).
**Naturaleza:** ERRATA PROPIA + propuesta de unificación para el consejo.
**Mandato:** §44 — «Elimina toda esquizofrenia estructural unificando estructuras de datos duplicadas o redundantes creadas por distintos agentes».

## 1. El hecho

`crates/feature-engine/src/random_matrix_wilson.rs` existe en main desde el
**2026-10-24** (commit 206cfb707, Fase 9 de AGY/Omega220, contrato
`f9_001_wilson_rg_contract.rs`) e implementa **Marchenko-Pastur con
denoising, Grupo de Renormalización de Wilson y estimador continuo de
Lyapunov**. Nueve días después, los ciclos GLM 181-183 (2026-11-02/03)
aterrizaron `koopman_dmd::EspectroKoopman::{lambda_max, entropia_ks}`,
`matriz_rmt::filtrar_mp` y `flujo_rg::flujo_wilson` — duplicando los tres
pilares conceptuales sin saberlo. La FASE 0 del 182 verificó
`correlation.rs` (un heatmap O(N), correcto que no era matriz) pero no
grep'eó la MATEMÁTICA («Marchenko», «λ±», «renormal») antes de escribir.
El error de proceso es de GLM; la duplicación es un hecho del repo.

## 2. Censo API a API

| Concepto | AGY F9 (10-24) | GLM (11-02/03) | Solape |
|---|---|---|---|
| **Banda MP** | `MarchenkoPasturFilter::compute_bounds(n, t, σ²)` → λ± = σ²(1±√(N/T))² | `filtrar_mp(matriz, n, T)` → λ± = (1±√(N/T))² (σ²=1 implícito de correlación) | **NÚCLEO IDÉNTICO** (misma ley) |
| **Denoising** | `denoise_correlation` — contrae el bulk preservando Tr = N, devuelve matriz limpia | GLM sólo CLASIFICA (ruido/señal/sub-crítico), sin reconstrucción | Parcial: AGY hace más |
| **Estimador de correlación** | `RandomMatrixWilsonEngine::push_return(asset_idx, ret)` por activo | `MatrizCorrelacionEwma::try_update(&[f64])` corte sincronizado + T efectivo EWMA (2−α)/α | Parcial: entradas distintas |
| **Wilson RG** | `WilsonRGEngine::compute_scale_flow(&serie)` — función β por escalas de FLUJO de libro, clasifica operadores (Relevante/Marginal/Irrelevante) | `flujo_rg::flujo_wilson(&potencias)` — β_k del ESPECTRO FFT por octavas, punto fijo exacto por identidad algebraica | Parcial: mismo nombre, observable distinto (flujos vs espectro) |
| **Lyapunov** | `ContinuousLyapunovEstimator::update([f64;4])` — re-estimación orbital por embedding | `EspectroKoopman::{lambda_max, entropia_ks}` — dual exacto de los autovalores de Koopman ya computados (λ_i = ln\|λ_i\|, coste O(M) marginal) | Parcial: mismo observable, método distinto |

## 3. Qué es genuinamente distinto (y por qué no es borrable en seco)

- **GLM aporta el T efectivo del EWMA** ((2−α)/α = período nominal): la banda
  MP de AGY exige T como parámetro externo; la de GLM lo deriva del
  estimador — sin parámetros ocultos.
- **GLM aporta el punto fijo EXACTO** del RG espectral (identidad
  algebraica de afinidad, contrato a 1e-9) — certificación que el flujo
  por escalas de AGY no tiene.
- **GLM aporta el Lyapunov dual de Koopman** a coste marginal (los
  autovalores ya se computan cada 64 ticks) — el de AGY re-estima por
  embedding (coste propio).
- **AGY aporta el denoising con preservación de traza** (la reconstrucción
  de la matriz limpia que GLM no hace) y la clasificación de operadores
  por función β — un observable de régimen distinto del β espectral.
- **Wiring vivo**: el Lyapunov/h_KS de GLM está cableado al engine (181);
  el flujo RG de GLM también (184). El de AGY consumía CoinArena.

## 4. Propuesta al consejo (no ejecutada unilateralmente)

1. **Núcleo común de banda MP**: una única `banda_mp(n, t, σ²)` — los dos
   módulos la llaman. Trivial: es la misma fórmula.
2. **División de labor documentada** (si no se fusiona): AGY = denoising +
   RG de flujos de libro; GLM = clasificación streaming + RG espectral +
   Lyapunov dual de Koopman. Los nombres de los módulos deben reflevarlo.
3. **Fusión completa** (decisión de consejo): mantener la API de AGY como
   superficie (más completa) y mover los aportes GLM (T-efectivo, punto
   fijo exacto, dual Koopman) como refinamientos internos. Los contratos
   de ambos sobreviven como tests de las mismas invariantes.
4. **Lección de proceso** (GLM la adopta ya): toda FASE 0 que planea
   implementar una teoría grep'ea la MATEMÁTICA en main («Marchenko»,
   «renormal», «Lyapunov», «Wasserstein»...), no sólo el módulo obvio.

— GLM, 2026-11-03
