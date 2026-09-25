# Reglas de Proyecto: Trader Gemini

> Estado vivo del proyecto (qué está integrado, decisiones vigentes, pendientes,
> cómo compilar/probar desde Linux): **`.agents/MEMORIA.md`**. Léelo antes de
> empezar y actualízalo al cerrar cada ola.

<RULE[growth_over_wr]>
- NO ES NECESARIO UN WIN RATE (WR) DEL 100%.
- El objetivo principal, único y fundamental que debe superar el 100% es el **crecimiento exponencial y el interés compuesto cada 3 días**.
- El sistema debe estar optimizado para maximizar la curva de capital y lograr duplicar la cuenta (13 USD) exponencialmente, aceptando pérdidas y asumiendo un Win Rate realista (ej. 60-80%), siempre y cuando la esperanza matemática y el crecimiento compuesto resulten en la meta financiera.
</RULE[growth_over_wr]>

<RULE[sesiones_concurrentes]>
- Este repo se edita en PARALELO por varias sesiones de agente. Cuatro episodios
  de pérdida de trabajo lo demostraron (V7, plan-maestro, heredoc/git-checkout,
  REHAB). Protocolo obligatorio:
- `git status` ANTES de cada commit. Si hay archivos modificados que NO son de
  tu cambio, NO los toques: haz `git add` SOLO de tus archivos. `git add -A`
  está PROHIBIDO — en REHAB arrastró un test de otra sesión.
- Antes de cablear cualquier ancla (nombre de función, campo, binario), RE-GREP
  el símbolo: otra sesión pudo moverla o renombrarla entre tu lectura y tu edit.
- Antes de matar/relanzar `god_engine.exe`, verifica el PID Y StartTime del
  proceso (otra sesión pudo relanzarlo con un binario distinto al tuyo).
- Commits atómicos, un bloque funcional por commit, prefijo del bloque
  (B3.7, F8, etc.) en el mensaje.
- OPS build: `Stop-Process god_engine` ANTES de `cargo build` (exe bloqueado =
  link fallido y el binario VIEJO se relanza en silencio). Al relanzar con
  PowerShell SIEMPRE `-RedirectStandardOutput logs/demo_vNN.log` — sin eso el
  motor queda ciego de log (45 min de sesión perdidos el 2026-09-15).
- MERGE ENTRE SESIONES: un auto-merge "limpio" NO es un merge correcto. Tras
  integrar ramas de sesiones paralelas, haz diff del resultado contra CADA
  padre y compila con `--all-targets` antes de commitear: el 2026-09-25 dos
  choques semánticos (Swing/Scalping ya erradicados por U-ERR-5; F-009
  editado en un bloque que la otra rama movió a `puertas_del_continuo`)
  pasaron sin marcas de conflicto.
</RULE[sesiones_concurrentes]>
