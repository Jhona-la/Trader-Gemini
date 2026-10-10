# ADR-0014: Doctrina del continuo espectral — los seis principios del motor

- **Estado**: Aceptado (2026-10-04)
- **Contexto**: Barrido F0 (mandato del operador: "debes pensar como un
  universo multivariante continuo temporal espectral"). La cadena
  #609..#654 construyó el motor espectral completo, pero sus conceptos
  rectores vivían sólo en el registro forense (formato bitácora, no
  definiciones) y en la memoria de sesión — la F0 exige un ADR que los
  fije como arquitectura. Cada principio nombro la ola que lo instituyó
  y su prueba viva.

## Decisión

El sistema ES un continuo temporal espectral multivariante: cada
decisión nace de la escala τ que la genera. Seis principios rigen
todo cambio de código que toque señales, consenso o consumo:

1. **Exceso sobre estado estacionario** (λ/μ̂; #535/#582/#617/#625):
   toda magnitud Hawkes entra a un consumidor como exceso sobre
   STEADY_STATE_RATIO, saturado con tanh — 0 = régimen normal =
   abstención. Jamás umbral absoluto sobre el ratio crudo.
2. **Paridad evaluate/update** (#625): el gradiente del PPO aprende de
   la MISMA variable que su slot vota. Tocar un slot del `ppo_state`
   exige tocar su homólogo en `ppo_close_features` en el mismo commit.
3. **τ viva del router** (#624): `expected_lifetime_ms` del
   orquestador ES la τ de la geometría (router.rs:74) — cuando el
   consenso espectral domina, la posición vive a
   `consenso_espectral_tau` clamp [30 s, 12 h] (H2: τ inoperable ⇒ el
   espectral no dirige).
4. **Anti-staleness de registro** (#624/#648): un publicador
   condicionado que deja de escribir deja al lector consumiendo valor
   viejo. Todo `if let Some(...) { set }` lleva su `else { set 0.0 }`;
   el veredicto espectral caduca sin depth por >max(30 s, τ).
5. **Precedencia espectral con observabilidad** (#624/#648/#652): el
   espectro no se calla por abstención escalar (lectura antes de la
   guardia de peso), pero sólo votan escalas observadas (gate
   D-742/CL-32) y la modulación es inter-espectral (|media/v_dom|,
   fuente independiente — H4).
6. **La convicción se GANA con historial propio** (#626, cierre
   espectral de D-752): pesos por IC prequential por motor×escala
   (olvido 1/64, madurez 30), piso de exploración; en frío ≡ pesos
   iguales — la ponderación sólo entra con evidencia. La
   significancia la decide el **e-proceso de Ville con corrección de
   FAMILIA M/α** (#661/#663 — umbral Bonferroni sobre la unión de
   supermartingalas: 640 para el banco de τ*, 8320 para los 416 pares
   motor×escala del consenso, 43 500 para las celdas del veto de
   grupo; anytime-valid, inmune al optional stopping y a la
   multiplicidad del máximo). El umbral Fisher 2/√(n−3) con N
   efectivo fue RETIRADO (#663 subsume #599; exportaciones muertas
   eliminadas en Ola 67). *Adenda R4-A2 (2026-10-07): este principio
   se reescribe para que el ADR no prescribe la fórmula retirada.*

## Alternativas consideradas

- Documentar cada principio en su entrada forense únicamente
  (rechazado: la bitácora registra QUÉ pasó, no define el contrato).
- Un documento aparte de "decisiones espectrales" (rechazado: se
  duplicaría con este ADR — el barrido F0 exige una sola fuente).

## Consecuencias

- Toda ola futura que toque estas zonas cita el principio violado o
  cumplido en su forense; el reviewer cruza contra este ADR.
- La F0 queda CERRADA para el marco doctrinal; los conceptos vivos
  del código (VotoEspectral, SkillMotores, excitación, ρ(τ*)) tienen
  aquí su definición de referencia y su prueba viva.
- Este ADR NO cambia conducta — es documental (T-1 cero).
