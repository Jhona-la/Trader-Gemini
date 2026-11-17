//! tick-tape: bytes VERIFICADOS de una cinta de ticks, venga como `.bin` o como contenedor `.bin.tgb`.
//!
//! PROCEDENCIA DEL FORMATO TGB00001. El formato y la herramienta `tgb` los escribio el coordinador para este
//! proyecto: fuente `C:\Users\jhona\Desktop\PC-Turbo\analisis\tgb\src\main.rs`, SHA-256
//! `4858b7f4179b8735d814ed6db9b07eb855e661c671446d06f808472123c0b110`. Tiene la misma licencia que este
//! repositorio. Este crate NO copia su codigo: es un decodificador nuevo, escrito contra el contrato del formato
//! (pedido CODEX-CLI `codex_cli_pedido_lector_tgb_v1.md`, Parte 2) y probado contra contenedores que produjo esa
//! herramienta (`tests/fixtures`).
//!
//! Contrato del contenedor (todos los enteros u64 little-endian):
//! - inicio: magia `TGB00001`, largo del original, SHA-256 del original (32 B), modo (u8);
//! - modo 0: largo comprimido y un frame Zstandard del original entero;
//! - modo 1: cabecera original de 8 B, total de registros y bloques de columnas;
//! - modo 2: total de registros y bloques de columnas, sin cabecera;
//! - cada bloque: cantidad de registros y, por campo (tiempo, bid, ask, bid_qty, ask_qty), largo comprimido y
//!   bytes. El tiempo va como diferencia con el anterior en zigzag, y el anterior vuelve a 0 en cada bloque;
//! - final: SHA-256 de todos los bytes anteriores del contenedor (el sello).

pub mod logica;
pub mod procedencia;
pub mod tgb;
