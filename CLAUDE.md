# Trader Gemini — contexto para Claude Code

Reglas del proyecto y memoria compartida entre sesiones (se cargan siempre):

@.agents/AGENTS.md

`.agents/MEMORIA.md` (1,17 MB) es el tablero histórico: NO se auto-carga (cargado entero, el arranque supera la ventana de contexto). Léelo por partes con offset/limit o `rg`, nunca entero.
