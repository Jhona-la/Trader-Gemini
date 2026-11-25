# Procedencia del sujeto base del colector captura_v2 (SALIDAFALLO-BASE)

Fecha de la lectura: 22 nov 2026, integrador. Comandos: `Get-FileHash -Algorithm SHA256`, `Get-Content`.
Paquete (P): `C:\Users\jhona\Desktop\PC-Turbo\releases\captura_v2_r3b\paquete_20261118-212417-015-8428-0c42b301e8e94d188dda7fe98c0d2586`.
Instalado (I): `C:\Users\jhona\Desktop\PC-Turbo\captura_v2\bin\captura_v2.exe`.

## Cadena documental

| Eslabon | Evidencia | Valor observado |
| --- | --- | --- |
| Bytes instalados | `Get-FileHash I` | `E3DDBF183803BD90B33839955A05FE6BD7D26C74C7B41E18904BE6B3A5D247B6` |
| Salida del build del paquete | `Get-FileHash P\target\release\captura_v2.exe` y `P\bin\captura_v2.exe` | el mismo SHA |
| Recibo del build | `P\RELEASE.json` (copia byte a byte en `r3b/RELEASE.json`): `exe_sha256`, `fuentes{}`, `compilacion`, `cargo`, `rustc`, `exit_code` | `exe_sha256` = el mismo SHA; `cargo build --release --offline --locked`; rustc 1.96.0 (ac68faa20 2026-05-25), host x86_64-pc-windows-msvc, LLVM 22.1.2; exit 0 |
| Fuentes que leyo el compilador | `P\target\release\captura_v2.d` (copia en `r3b/captura_v2.d`) | `src_r3b\src\{codec,gate,l2,main,public_l2,store}.rs` |
| Fuentes de este commit | `Get-FileHash` de `herramientas/captura_v2/**` frente a `RELEASE.json.fuentes` | las 10 coinciden (Cargo.toml, Cargo.lock, supervisor_v2.ps1 y los 7 .rs) |
| Flags | `P\target\release\.fingerprint\{captura_v2,tokio}-*\*.json` | `"rustflags":[]` en captura_v2 y tokio |
| Configuracion de cargo | `Test-Path` de `.cargo\config(.toml)` en src_r3b y en cada ancestro hasta `C:\`, `%USERPROFILE%\.cargo\config(.toml)`, `P\cargo_home\config(.toml)`, `rust-toolchain(.toml)` en src_r3b | ninguno existe |
| Perfil | `Cargo.toml` de este commit: `[profile.release] opt-level = 3`, sin `panic` ni `debug-assertions` | por defecto de Cargo para release: panic=unwind, debug-assertions=false |
| Tokio | `Cargo.lock`: tokio 1.53.1; copia leida en `P\cargo_home\registry\src\index.crates.io-1949cf8c6b5b557f\tokio-1.53.1`; features en el fingerprint: bytes, default, io-util, libc, macros, mio, net, rt, rt-multi-thread, signal, signal-hook-registry, socket2, sync, time, tokio-macros, windows-sys | |

## Limites (N/D)

- El recibo lo escribio el propio guion de empaquetado; no hay firma independiente. Los hashes de hoy coinciden con los del
  recibo, pero eso no prueba por si solo que nadie edito y restauro las fuentes entre el build y el recibo.
- No hay log `-v` del build historico: las invocaciones efectivas de rustc son N/D. El perfil se deduce del manifiesto,
  de la ausencia de configuracion y de `rustflags: []`; no se observo `-C debug-assertions` en una invocacion.
- No se reprodujo el build: en x86_64-pc-windows-msvc rustc siempre enlaza con `/DEBUG` (PDB con GUID/fecha), asi que un
  rebuild no daria los mismos bytes y no acreditaria identidad.
- Las fuentes vivas de la raiz del repo (`src/main.rs` 1D0FE496..., `src/store.rs` D72287FF..., `src/public_l2.rs`
  7CEFF7DE..., `Cargo.toml` 735E900B...) NO son el sujeto del instalado: son mas viejas (15-17 nov) y difieren de r3b
  (1257 lineas anadidas, 110 quitadas en 5 archivos). Las lineas citadas por el pedido SALIDAFALLO v2 y sus N1/N2 son de
  esas fuentes vivas; la respuesta del integrador las traduce a este sujeto.
