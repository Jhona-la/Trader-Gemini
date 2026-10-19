//! RB-P1 (Codex RB, 2026-10-19) — TRINQUETE DE PÁNICOS EN CÓDIGO DE PRODUCCIÓN.
//!
//! El perfil release usa `panic = "abort"` (LG-28): un `.unwrap()`,
//! `.expect(..)`, `.unwrap_err()`, `.expect_err(..)`, `panic!`,
//! `unreachable!`, `todo!`, `unimplemented!` o `assert!`/`assert_eq!`/
//! `assert_ne!` alcanzable en producción mata el proceso con posiciones
//! abiertas. Mandato del operador: prohibidos en todo el código Rust. LG-28
//! arregló siete pánicos alcanzables del camino vivo, pero nada impedía que
//! volvieran. Este contrato:
//!
//! 1. recorre el ÁRBOL DE MÓDULOS REAL de cada paquete del workspace (desde
//!    `src/lib.rs`, `src/main.rs`, `src/bin/*` y las rutas `[lib]`/`[[bin]]`
//!    del manifiesto, siguiendo `mod x;`, módulos en línea y `#[path]`): no
//!    cuenta archivos huérfanos que no se compilan, ni ítems o módulos de
//!    prueba (`#[cfg(test)]`, `#[test]`, `#[tokio::test]`, `#![cfg(test)]`), ni
//!    comentarios o literales (cadenas, crudas, de bytes y caracteres);
//! 2. compara la suma por paquete y por clase con la línea base versionada
//!    `docs/audit/trinquete_panicos_produccion.tsv`: ningún paquete puede
//!    SUBIR (lo nuevo se maneja en caliente con `Result`/`Option` antes de
//!    fusionar); al bajar, se regenera la línea base y el trinquete aprieta;
//! 3. con `TRINQUETE_ESCRIBIR=1` reescribe la línea base con el recuento
//!    actual; el historial git de ese TSV es la serie histórica de la quema.
//!
//! El conteo es léxico: `.map(Option::unwrap)` o un `.expect(` de un método
//! propio no se distinguen. Los contratos de abajo certifican el lexer y la
//! resolución de módulos con casos de solución conocida.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::fmt::Write as _;
use std::fs;
use std::path::{Component, Path, PathBuf};

const RUTA_LINEA_BASE: &str = "docs/audit/trinquete_panicos_produccion.tsv";

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
struct Cuenta {
    unwrap: u32,
    expect: u32,
    panico: u32,
    aserto: u32,
}

impl Cuenta {
    fn total(&self) -> u32 {
        self.unwrap + self.expect + self.panico + self.aserto
    }

    fn sumar(&mut self, o: &Cuenta) {
        self.unwrap += o.unwrap;
        self.expect += o.expect;
        self.panico += o.panico;
        self.aserto += o.aserto;
    }
}

/// Declaración `mod nombre;` de producción hallada en un archivo.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DeclMod {
    nombre: String,
    /// Módulos en línea (`mod x { … }`) que la envuelven, de fuera adentro.
    en_linea: Vec<String>,
    /// Valor de `#[path = "…"]`, si lo hay.
    ruta: Option<String>,
}

#[derive(Debug, Default)]
struct Analisis {
    cuenta: Cuenta,
    mods: Vec<DeclMod>,
    solo_pruebas: bool,
}

fn es_ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Copia del fuente con comentarios y contenidos de literales en blanco;
/// conserva los índices y los saltos de línea. Los lifetimes quedan intactos.
fn despojar(c: &[char]) -> Vec<char> {
    let n = c.len();
    let mut out = c.to_vec();
    fn blanquear(out: &mut [char], a: usize, b: usize) {
        let b = b.min(out.len());
        for x in out.iter_mut().take(b).skip(a) {
            if *x != '\n' {
                *x = ' ';
            }
        }
    }
    let frontera = |i: usize| i == 0 || !es_ident(c[i - 1]);
    let mut i = 0;
    while i < n {
        let ch = c[i];
        if ch == '/' && i + 1 < n && c[i + 1] == '/' {
            let s = i;
            while i < n && c[i] != '\n' {
                i += 1;
            }
            blanquear(&mut out, s, i);
            continue;
        }
        if ch == '/' && i + 1 < n && c[i + 1] == '*' {
            let s = i;
            let mut prof = 0usize;
            while i < n {
                if c[i] == '/' && i + 1 < n && c[i + 1] == '*' {
                    prof += 1;
                    i += 2;
                } else if c[i] == '*' && i + 1 < n && c[i + 1] == '/' {
                    prof = prof.saturating_sub(1);
                    i += 2;
                    if prof == 0 {
                        break;
                    }
                } else {
                    i += 1;
                }
            }
            blanquear(&mut out, s, i);
            continue;
        }
        // Cadenas crudas: r"…", r#"…"#, br"…", br#"…"#.
        if (ch == 'r' || (ch == 'b' && i + 1 < n && c[i + 1] == 'r')) && frontera(i) {
            let mut j = i + if ch == 'b' { 2 } else { 1 };
            let mut almohadillas = 0usize;
            while j < n && c[j] == '#' {
                almohadillas += 1;
                j += 1;
            }
            if j < n && c[j] == '"' {
                let s = i;
                j += 1;
                'cruda: while j < n {
                    if c[j] == '"' {
                        let mut k = 0;
                        while k < almohadillas && j + 1 + k < n && c[j + 1 + k] == '#' {
                            k += 1;
                        }
                        if k == almohadillas {
                            j += 1 + almohadillas;
                            break 'cruda;
                        }
                    }
                    j += 1;
                }
                blanquear(&mut out, s, j);
                i = j;
                continue;
            }
        }
        // Cadenas "…" y b"…" (con escapes).
        if ch == '"' || (ch == 'b' && i + 1 < n && c[i + 1] == '"' && frontera(i)) {
            let s = i;
            let mut j = if ch == 'b' { i + 2 } else { i + 1 };
            while j < n {
                if c[j] == '\\' {
                    j += 2;
                    continue;
                }
                if c[j] == '"' {
                    j += 1;
                    break;
                }
                j += 1;
            }
            blanquear(&mut out, s, j);
            i = j;
            continue;
        }
        // Literales de carácter ('x', '\n', '\'', '\u{..}') frente a lifetimes.
        if ch == '\'' {
            if i + 1 < n && c[i + 1] == '\\' {
                let s = i;
                let mut j = i + 3;
                while j < n && c[j] != '\'' {
                    j += 1;
                }
                blanquear(&mut out, s, j + 1);
                i = j + 1;
                continue;
            }
            if i + 2 < n && c[i + 2] == '\'' {
                blanquear(&mut out, i, i + 3);
                i += 3;
                continue;
            }
        }
        i += 1;
    }
    out
}

fn siguiente_no_blanco(t: &[char], mut j: usize) -> Option<(usize, char)> {
    while j < t.len() {
        if !t[j].is_whitespace() {
            return Some((j, t[j]));
        }
        j += 1;
    }
    None
}

fn anterior_no_blanco(t: &[char], i: usize) -> Option<char> {
    let mut j = i;
    while j > 0 {
        j -= 1;
        if !t[j].is_whitespace() {
            return Some(t[j]);
        }
    }
    None
}

/// Índice del delimitador `b` que cierra el `a` situado en `abre`.
fn cierre(t: &[char], abre: usize, a: char, b: char) -> usize {
    let mut prof = 0i64;
    let mut j = abre;
    while j < t.len() {
        if t[j] == a {
            prof += 1;
        } else if t[j] == b {
            prof -= 1;
            if prof == 0 {
                return j;
            }
        }
        j += 1;
    }
    t.len()
}

/// Fin del ítem que empieza en `j`: el primer `;` o `,` fuera de paréntesis,
/// corchetes y genéricos; el `}` que cierra el primer `{`; o, sin consumirlo,
/// el `}` del bloque que lo contiene.
fn fin_de_item(t: &[char], mut j: usize) -> usize {
    let mut prof = 0i64;
    let mut ang = 0i64;
    while j < t.len() {
        match t[j] {
            '(' | '[' => prof += 1,
            ')' | ']' => prof -= 1,
            '<' => ang += 1,
            '>' => {
                let previo = if j > 0 { t[j - 1] } else { ' ' };
                if previo != '-' && previo != '=' && ang > 0 {
                    ang -= 1;
                }
            }
            ';' if prof <= 0 => return j + 1,
            ',' if prof <= 0 && ang <= 0 => return j + 1,
            '{' if prof <= 0 => return cierre(t, j, '{', '}') + 1,
            '}' if prof <= 0 => return j,
            _ => {}
        }
        j += 1;
    }
    t.len()
}

fn sin_blancos(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

fn es_marca_de_prueba(contenido_atributo: &str) -> bool {
    let s = sin_blancos(contenido_atributo);
    s == "test"
        || s == "cfg(test)"
        || s.starts_with("cfg(all(test,")
        || s.ends_with("::test")
        || (s.contains("::test(") && !s.starts_with("cfg"))
}

/// Cuenta los sitios de pánico de producción de un fuente y lista sus
/// declaraciones `mod x;` de producción.
fn analizar(fuente: &str) -> Analisis {
    let orig: Vec<char> = fuente.chars().collect();
    let t = despojar(&orig);
    let n = t.len();
    let mut a = Analisis::default();
    let mut pila: Vec<Option<String>> = Vec::new();
    let mut mod_pendiente: Option<String> = None;
    let mut ruta_attr: Option<String> = None;
    let mut i = 0usize;
    while i < n {
        let ch = t[i];
        if ch == '#' {
            let atributo = match siguiente_no_blanco(&t, i + 1) {
                Some((j, '[')) => Some((false, j)),
                Some((j, '!')) => match siguiente_no_blanco(&t, j + 1) {
                    Some((k, '[')) => Some((true, k)),
                    _ => None,
                },
                _ => None,
            };
            if let Some((interno, abre)) = atributo {
                let fin = cierre(&t, abre, '[', ']');
                let contenido: String = orig[(abre + 1).min(n)..fin.min(n)].iter().collect();
                if es_marca_de_prueba(&contenido) {
                    if interno {
                        return Analisis {
                            solo_pruebas: true,
                            ..Analisis::default()
                        };
                    }
                    // Salta los atributos que siguen y el ítem completo.
                    let mut k = fin + 1;
                    while let Some((p, '#')) = siguiente_no_blanco(&t, k) {
                        match siguiente_no_blanco(&t, p + 1) {
                            Some((q, '[')) => k = cierre(&t, q, '[', ']') + 1,
                            _ => break,
                        }
                    }
                    i = fin_de_item(&t, k);
                    ruta_attr = None;
                    continue;
                }
                if let Some(resto) = sin_blancos(&contenido).strip_prefix("path=") {
                    ruta_attr = Some(resto.trim_matches('"').to_string());
                }
                i = fin + 1;
                continue;
            }
        }
        if ch == '{' {
            pila.push(mod_pendiente.take());
            i += 1;
            continue;
        }
        if ch == '}' {
            pila.pop();
            i += 1;
            continue;
        }
        if (ch.is_alphabetic() || ch == '_') && (i == 0 || !es_ident(t[i - 1])) {
            let s = i;
            while i < n && es_ident(t[i]) {
                i += 1;
            }
            let id: String = t[s..i].iter().collect();
            match id.as_str() {
                "unwrap" | "unwrap_err" => {
                    if anterior_no_blanco(&t, s) == Some('.') {
                        if let Some((p, '(')) = siguiente_no_blanco(&t, i) {
                            if let Some((_, ')')) = siguiente_no_blanco(&t, p + 1) {
                                a.cuenta.unwrap += 1;
                            }
                        }
                    }
                }
                "expect" | "expect_err" => {
                    if anterior_no_blanco(&t, s) == Some('.')
                        && matches!(siguiente_no_blanco(&t, i), Some((_, '(')))
                    {
                        a.cuenta.expect += 1;
                    }
                }
                "panic" | "unreachable" | "todo" | "unimplemented" | "assert" | "assert_eq"
                | "assert_ne" => {
                    if let Some((p, '!')) = siguiente_no_blanco(&t, i) {
                        if matches!(siguiente_no_blanco(&t, p + 1), Some((_, '(' | '[' | '{'))) {
                            if id.starts_with("assert") {
                                a.cuenta.aserto += 1;
                            } else {
                                a.cuenta.panico += 1;
                            }
                        }
                    }
                }
                "mod" => {
                    if let Some((p, c0)) = siguiente_no_blanco(&t, i) {
                        if c0.is_alphabetic() || c0 == '_' {
                            let mut q = p;
                            while q < n && es_ident(t[q]) {
                                q += 1;
                            }
                            let nombre: String = t[p..q].iter().collect();
                            match siguiente_no_blanco(&t, q) {
                                Some((r, ';')) => {
                                    let en_linea = pila.iter().flatten().cloned().collect();
                                    a.mods.push(DeclMod {
                                        nombre,
                                        en_linea,
                                        ruta: ruta_attr.take(),
                                    });
                                    i = r + 1;
                                }
                                Some((_, '{')) => {
                                    mod_pendiente = Some(nombre);
                                    ruta_attr = None;
                                    i = q;
                                }
                                _ => {}
                            }
                        }
                    }
                }
                _ => {}
            }
            continue;
        }
        i += 1;
    }
    a
}

/// Resuelve `..` y `.` léxicamente para que un mismo archivo tenga una sola
/// ruta, venga por donde venga.
fn normalizar(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            otro => out.push(otro.as_os_str()),
        }
    }
    out
}

/// Archivos candidatos de una declaración `mod x;` hallada en `f`.
fn candidatos(f: &Path, d: &DeclMod, mod_rs: bool) -> Vec<PathBuf> {
    let dir = f.parent().map(Path::to_path_buf).unwrap_or_default();
    let raiz_hijos = if mod_rs {
        dir.clone()
    } else {
        let stem = f
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        dir.join(stem)
    };
    match &d.ruta {
        Some(r) if d.en_linea.is_empty() => vec![normalizar(&dir.join(r))],
        Some(r) => {
            let mut b = raiz_hijos;
            for m in &d.en_linea {
                b = b.join(m);
            }
            vec![normalizar(&b.join(r))]
        }
        None => {
            let mut b = raiz_hijos;
            for m in &d.en_linea {
                b = b.join(m);
            }
            vec![
                normalizar(&b.join(format!("{}.rs", d.nombre))),
                normalizar(&b.join(&d.nombre).join("mod.rs")),
            ]
        }
    }
}

/// (nombre del paquete, rutas de `[lib]`/`[[bin]]`) de un Cargo.toml.
fn leer_manifiesto(texto: &str) -> (Option<String>, Vec<String>) {
    let mut seccion = String::new();
    let mut nombre = None;
    let mut rutas = Vec::new();
    for linea in texto.lines() {
        let l = linea.split('#').next().unwrap_or("").trim();
        if l.starts_with('[') {
            seccion = l.to_string();
            continue;
        }
        let Some((k, v)) = l.split_once('=') else {
            continue;
        };
        let (k, v) = (k.trim(), v.trim().trim_matches('"'));
        if seccion == "[package]" && k == "name" && nombre.is_none() {
            nombre = Some(v.to_string());
        }
        if (seccion == "[lib]" || seccion == "[[bin]]") && k == "path" {
            rutas.push(v.to_string());
        }
    }
    (nombre, rutas)
}

/// Rutas de `[workspace] members` de un Cargo.toml.
fn miembros(texto: &str) -> Vec<String> {
    let mut dentro = false;
    let mut out = Vec::new();
    for linea in texto.lines() {
        let l = linea.split('#').next().unwrap_or("").trim();
        if !dentro {
            if l.starts_with("members") {
                dentro = true;
            } else {
                continue;
            }
        }
        let mut resto = l;
        while let Some(a) = resto.find('"') {
            let tras = &resto[a + 1..];
            match tras.find('"') {
                Some(b) => {
                    out.push(tras[..b].to_string());
                    resto = &tras[b + 1..];
                }
                None => break,
            }
        }
        if l.contains(']') {
            break;
        }
    }
    out
}

/// Raíces de compilación de un paquete: lib, main, bins autodescubiertos y
/// rutas explícitas del manifiesto.
fn raices_de(dir: &Path, rutas_manifiesto: &[String]) -> Vec<PathBuf> {
    let src = dir.join("src");
    let mut v: Vec<PathBuf> = Vec::new();
    for f in ["lib.rs", "main.rs"] {
        let p = src.join(f);
        if p.is_file() {
            v.push(normalizar(&p));
        }
    }
    if let Ok(rd) = fs::read_dir(src.join("bin")) {
        let mut entradas: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        entradas.sort();
        for p in entradas {
            if p.is_file() && p.extension().is_some_and(|e| e == "rs") {
                v.push(normalizar(&p));
            } else if p.is_dir() {
                let m = p.join("main.rs");
                if m.is_file() {
                    v.push(normalizar(&m));
                }
            }
        }
    }
    for r in rutas_manifiesto {
        let p = normalizar(&dir.join(r));
        if p.is_file() && !v.contains(&p) {
            v.push(p);
        }
    }
    v
}

/// Recorre el árbol de módulos de producción desde las raíces y cuenta cada
/// archivo alcanzado (con cero incluido).
fn recorrer(raices: &[PathBuf]) -> Vec<(PathBuf, Cuenta)> {
    let conjunto_raices: HashSet<PathBuf> = raices.iter().cloned().collect();
    let mut vistos: HashSet<PathBuf> = HashSet::new();
    let mut cola: VecDeque<PathBuf> = raices.iter().cloned().collect();
    let mut res = Vec::new();
    while let Some(f) = cola.pop_front() {
        if !vistos.insert(f.clone()) {
            continue;
        }
        let Ok(fuente) = fs::read_to_string(&f) else {
            continue;
        };
        let an = analizar(&fuente);
        if an.solo_pruebas {
            continue;
        }
        let mod_rs = conjunto_raices.contains(&f)
            || matches!(
                f.file_name().and_then(|s| s.to_str()),
                Some("lib.rs" | "main.rs" | "mod.rs")
            );
        for d in &an.mods {
            if let Some(c) = candidatos(&f, d, mod_rs).into_iter().find(|c| c.is_file()) {
                cola.push_back(c);
            }
        }
        res.push((f, an.cuenta));
    }
    res
}

#[derive(Debug, Clone)]
struct Fila {
    paquete: String,
    archivo: String,
    cuenta: Cuenta,
}

fn raiz_workspace() -> PathBuf {
    normalizar(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

/// Censo de producción de todo el workspace: una fila por archivo alcanzado.
fn censo(raiz: &Path) -> Vec<Fila> {
    let manifiesto = fs::read_to_string(raiz.join("Cargo.toml")).unwrap_or_default();
    let mut dirs = vec![raiz.to_path_buf()];
    for m in miembros(&manifiesto) {
        dirs.push(normalizar(&raiz.join(m)));
    }
    let mut filas = Vec::new();
    let mut vistos: HashSet<PathBuf> = HashSet::new();
    for dir in dirs {
        let Ok(texto) = fs::read_to_string(dir.join("Cargo.toml")) else {
            continue;
        };
        let (Some(paquete), rutas) = leer_manifiesto(&texto) else {
            continue;
        };
        for (f, cuenta) in recorrer(&raices_de(&dir, &rutas)) {
            if !vistos.insert(f.clone()) {
                continue;
            }
            let archivo = f
                .strip_prefix(raiz)
                .unwrap_or(&f)
                .to_string_lossy()
                .replace('\\', "/");
            filas.push(Fila {
                paquete: paquete.clone(),
                archivo,
                cuenta,
            });
        }
    }
    filas
}

fn por_paquete(filas: &[Fila]) -> BTreeMap<String, Cuenta> {
    let mut m: BTreeMap<String, Cuenta> = BTreeMap::new();
    for f in filas {
        m.entry(f.paquete.clone()).or_default().sumar(&f.cuenta);
    }
    m
}

fn leer_linea_base(texto: &str) -> BTreeMap<String, Cuenta> {
    let mut m: BTreeMap<String, Cuenta> = BTreeMap::new();
    for l in texto.lines() {
        if l.starts_with('#') || l.starts_with("paquete\t") || l.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = l.split('\t').collect();
        if c.len() < 6 {
            continue;
        }
        let num = |s: &str| s.trim().parse::<u32>().unwrap_or(0);
        m.entry(c[0].to_string()).or_default().sumar(&Cuenta {
            unwrap: num(c[2]),
            expect: num(c[3]),
            panico: num(c[4]),
            aserto: num(c[5]),
        });
    }
    m
}

fn texto_linea_base(filas: &[Fila]) -> String {
    let mut s = String::new();
    s.push_str("# RB-P1 (Codex RB) — trinquete de pánicos en código de producción.\n");
    s.push_str("# Regenerar: TRINQUETE_ESCRIBIR=1 cargo test -p quantum-arena --test rb_p1_trinquete_panicos_contract\n");
    s.push_str("# Ningún paquete puede SUBIR su suma por clase. Al bajarla, regenera este archivo: el trinquete aprieta.\n");
    s.push_str("paquete\tarchivo\tunwrap\texpect\tpanico\taserto\n");
    let mut v: Vec<&Fila> = filas.iter().filter(|f| f.cuenta.total() > 0).collect();
    v.sort_by(|a, b| (&a.paquete, &a.archivo).cmp(&(&b.paquete, &b.archivo)));
    for f in v {
        let _ = writeln!(
            s,
            "{}\t{}\t{}\t{}\t{}\t{}",
            f.paquete, f.archivo, f.cuenta.unwrap, f.cuenta.expect, f.cuenta.panico, f.cuenta.aserto
        );
    }
    s
}

#[test]
fn rb_p1_el_lexer_ignora_comentarios_literales_y_pruebas() {
    let fuente = r##"
// x.unwrap() en comentario de línea
/* bloque /* anidado */ y.unwrap() panic!("no") */
/// doc: z.expect("no")
fn f(o: Option<u8>) -> u8 {
    let s = "z.unwrap() en cadena \" con comilla escapada panic!()";
    let r = r#"w.expect("x") cruda"#;
    let b = b"v.unwrap()";
    let c = '}';
    let e = '\'';
    let l: &'static str = "a";
    debug_assert!(true);
    let v = o.unwrap_or(0) + o.unwrap_or_default();
    o.unwrap() + o . unwrap ( ) + o.expect("boom")
}
fn g<'a>(r: Result<u8, u8>) {
    assert!(1 == 1);
    assert_eq!(1, 1);
    panic!("x");
    unreachable!();
    todo!();
    r.unwrap_err();
    r.expect_err("y");
}
#[cfg(test)]
mod tests {
    fn t() { None::<u8>.unwrap(); panic!(); }
}
#[test]
fn suelto() { None::<u8>.unwrap(); }
#[tokio::test(flavor = "current_thread")]
async fn asincrono() { None::<u8>.unwrap(); }
#[cfg(test)]
impl<A, B> Algo<A, B> { fn x() { None::<u8>.unwrap(); } }
#[cfg(not(test))]
fn solo_produccion() { None::<u8>.unwrap(); }
#[cfg_attr(test, allow(dead_code))]
fn tambien_produccion() { Some(1).expect("y"); }
"##;
    let a = analizar(fuente);
    assert_eq!(
        a.cuenta,
        Cuenta { unwrap: 4, expect: 3, panico: 3, aserto: 2 },
        "unwrap: 2 en f + unwrap_err + cfg(not(test)); expect: f + expect_err + cfg_attr; \
         pánicos: panic/unreachable/todo; asertos: assert/assert_eq"
    );
    assert!(!a.solo_pruebas);
    assert!(a.mods.is_empty());

    let solo_pruebas = analizar("#![cfg(test)]\nfn x() { None::<u8>.unwrap(); }\nmod y;\n");
    assert!(solo_pruebas.solo_pruebas, "un archivo con #![cfg(test)] no es producción");
    assert_eq!(solo_pruebas.cuenta, Cuenta::default());
}

#[test]
fn rb_p1_match_y_campos_con_cfg_test_no_se_comen_el_resto() {
    let fuente = r#"
fn f(x: u8) -> u8 {
    match x {
        #[cfg(test)]
        0 => None::<u8>.unwrap(),
        1 => { 7 }
        _ => Some(2).unwrap(),
    }
}
struct S {
    #[cfg(test)]
    solo_test: u8,
    vivo: u8,
}
fn h() { Some(3).unwrap(); }
"#;
    let a = analizar(fuente);
    assert_eq!(a.cuenta.unwrap, 2, "cuenta el brazo `_` y h(); no el brazo de prueba");
}

#[test]
fn rb_p1_resuelve_el_arbol_de_modulos_real() {
    let base = std::env::temp_dir().join(format!(
        "rb_p1_arbol_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let archivos: &[(&str, &str)] = &[
        (
            "src/lib.rs",
            "pub mod a;\n#[cfg(test)]\nmod t;\nmod interno { pub mod b; }\n#[path = \"otro/c_ruta.rs\"]\nmod c;\npub fn l() { None::<u8>.unwrap(); }\n",
        ),
        ("src/a.rs", "pub fn a() { None::<u8>.expect(\"a\"); }\npub mod sub;\n"),
        ("src/a/sub.rs", "pub fn s() { panic!(\"s\"); }\n"),
        ("src/t.rs", "pub fn t() { None::<u8>.unwrap(); None::<u8>.unwrap(); }\n"),
        ("src/interno/b.rs", "pub fn b() { assert!(true); }\n"),
        ("src/otro/c_ruta.rs", "pub fn c() { None::<u8>.unwrap(); }\n"),
        ("src/huerfano.rs", "pub fn h() { None::<u8>.unwrap(); }\n"),
        ("src/bin/herramienta/main.rs", "mod util;\nfn main() { util::u(); }\n"),
        ("src/bin/herramienta/util.rs", "pub fn u() { None::<u8>.unwrap(); }\n"),
        ("src/bin/simple.rs", "fn main() { todo!() }\n"),
    ];
    for (rel, contenido) in archivos {
        let p = base.join(rel);
        if let Some(dir) = p.parent() {
            let _ = fs::create_dir_all(dir);
        }
        let _ = fs::write(&p, contenido);
    }
    let raices = raices_de(&base, &[]);
    let res = recorrer(&raices);
    let _ = fs::remove_dir_all(&base);

    let mut total = Cuenta::default();
    let mut nombres: Vec<String> = Vec::new();
    for (f, c) in &res {
        total.sumar(c);
        let rel = f
            .strip_prefix(normalizar(&base))
            .unwrap_or(f)
            .to_string_lossy()
            .replace('\\', "/");
        nombres.push(rel);
    }
    nombres.sort();
    assert_eq!(
        nombres,
        vec![
            "src/a.rs",
            "src/a/sub.rs",
            "src/bin/herramienta/main.rs",
            "src/bin/herramienta/util.rs",
            "src/bin/simple.rs",
            "src/interno/b.rs",
            "src/lib.rs",
            "src/otro/c_ruta.rs",
        ],
        "ni el módulo de prueba ni el huérfano entran al censo"
    );
    assert_eq!(total, Cuenta { unwrap: 3, expect: 1, panico: 2, aserto: 1 });
}

#[test]
fn rb_p1_lee_manifiestos_y_miembros() {
    let raiz = r#"
[workspace]
members = [
    "crates/a", # comentario con "comillas"
    "crates/b",
]
[package]
name = "raiz-demo"
[[bin]]
name = "x"
path = "src/bin/x.rs"
[lib]
name = "l"
path = "src/lib.rs"
[dependencies]
path = "no-cuenta"
"#;
    assert_eq!(miembros(raiz), vec!["crates/a".to_string(), "crates/b".to_string()]);
    let (nombre, rutas) = leer_manifiesto(raiz);
    assert_eq!(nombre.as_deref(), Some("raiz-demo"));
    assert_eq!(rutas, vec!["src/bin/x.rs".to_string(), "src/lib.rs".to_string()]);
    assert_eq!(leer_manifiesto("[workspace]\nmembers = [\"a\"]\n").0, None);
}

#[test]
fn rb_p1_ningun_paquete_sube_su_cuenta_de_panicos() {
    let raiz = raiz_workspace();
    let filas = censo(&raiz);
    let actual = por_paquete(&filas);

    // Cobertura real: el censo no puede pasar en vacío y alcanza el núcleo
    // vivo (biblioteca del motor y binario host).
    let archivos: HashSet<&str> = filas.iter().map(|f| f.archivo.as_str()).collect();
    assert!(filas.len() > 150, "árbol de módulos sospechosamente pequeño: {}", filas.len());
    assert!(actual.len() >= 15, "paquetes alcanzados: {}", actual.len());
    for clave in [
        "crates/god-engine-core/src/lib.rs",
        "src/bin/god_engine.rs",
        "crates/risk-engine/src/lib.rs",
        "crates/quantum-arena/src/temporal_spectrum.rs",
    ] {
        assert!(archivos.contains(clave), "el censo no alcanzó {clave}");
    }

    let mut informe = String::new();
    let mut total = Cuenta::default();
    let _ = writeln!(informe, "{:<22} {:>6} {:>6} {:>6} {:>6} {:>6}", "paquete", "unwrap", "expect", "panico", "aserto", "total");
    for (p, c) in &actual {
        total.sumar(c);
        let _ = writeln!(informe, "{:<22} {:>6} {:>6} {:>6} {:>6} {:>6}", p, c.unwrap, c.expect, c.panico, c.aserto, c.total());
    }
    let _ = writeln!(informe, "{:<22} {:>6} {:>6} {:>6} {:>6} {:>6}", "TOTAL", total.unwrap, total.expect, total.panico, total.aserto, total.total());
    let mut top: Vec<&Fila> = filas.iter().filter(|f| f.cuenta.total() > 0).collect();
    top.sort_by(|a, b| b.cuenta.total().cmp(&a.cuenta.total()).then(a.archivo.cmp(&b.archivo)));
    let _ = writeln!(informe, "--- archivos con más sitios (de {} analizados) ---", filas.len());
    for f in top.iter().take(25) {
        let _ = writeln!(informe, "{:>4}  {}  {:?}", f.cuenta.total(), f.archivo, f.cuenta);
    }
    println!("{informe}");

    let ruta = raiz.join(RUTA_LINEA_BASE);
    if std::env::var("TRINQUETE_ESCRIBIR").as_deref() == Ok("1") {
        if let Err(e) = fs::write(&ruta, texto_linea_base(&filas)) {
            panic!("no se pudo escribir {}: {e}", ruta.display());
        }
        println!("línea base reescrita: {}", ruta.display());
        return;
    }
    let texto = match fs::read_to_string(&ruta) {
        Ok(t) => t,
        Err(e) => panic!(
            "falta la línea base {} ({e}); genérala con TRINQUETE_ESCRIBIR=1",
            ruta.display()
        ),
    };
    let base = leer_linea_base(&texto);
    let mut violaciones = Vec::new();
    let mut bajadas = Vec::new();
    for (p, c) in &actual {
        let b = base.get(p).copied().unwrap_or_default();
        for (clase, v, tope) in [
            ("unwrap", c.unwrap, b.unwrap),
            ("expect", c.expect, b.expect),
            ("pánico", c.panico, b.panico),
            ("aserto", c.aserto, b.aserto),
        ] {
            if v > tope {
                violaciones.push(format!("{p}: {clase} {v} > línea base {tope}"));
            } else if v < tope {
                bajadas.push(format!("{p}: {clase} {v} < línea base {tope}"));
            }
        }
    }
    if !bajadas.is_empty() {
        println!(
            "Bajaron (regenera la línea base con TRINQUETE_ESCRIBIR=1 para que el trinquete apriete):\n{}",
            bajadas.join("\n")
        );
    }
    assert!(
        violaciones.is_empty(),
        "RB-P1: sitios de pánico NUEVOS en producción (release = panic=abort):\n{}\n\
         Manéjalos en caliente con Result/Option (nada de unwrap/expect/panic!/assert! en \
         producción). Informe:\n{informe}",
        violaciones.join("\n")
    );
}
