param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("InitWorkspace", "PurgeLegacyBranches", "StageFiles", "Validate", "CommitAndMerge", "RescueStaleAgents", "StartGroupPushDaemon")]
    [string]$Action,

    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]${AgentId},

    [Parameter(Mandatory=$false)]
    [string]$Files = "",

    [Parameter(Mandatory=$false)]
    [string]$CratePath = ".",

    [Parameter(Mandatory=$false)]
    [string]$Message = "feat(quant): atomic zero-wait speculative update"
)

$ErrorActionPreference = "Stop"
$RepoRoot = [IO.Path]::GetFullPath((git rev-parse --show-toplevel).Trim())
Set-Location $RepoRoot

$TxRoot        = Join-Path $RepoRoot ".agent_tx"
$StagingDir    = Join-Path $TxRoot "staging\${AgentId}"
$DiagDir       = Join-Path $TxRoot "diagnostics"
$RescueDir     = Join-Path $TxRoot "rescued"
$HeartbeatFile = Join-Path $TxRoot "heartbeats\${AgentId}.json"
$CiAlertsFile  = Join-Path $TxRoot "ci_alerts.json"

foreach ($dir in @($TxRoot, $StagingDir, $DiagDir, $RescueDir, (Join-Path $TxRoot "heartbeats"), (Join-Path $TxRoot "candidates"))) {
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
}

function Write-JsonAtomic([string]$Path, $Value) {
    $tmp = "$Path.$([Guid]::NewGuid().ToString('N')).tmp"
    $Value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $tmp -Encoding UTF8
    Move-Item -LiteralPath $tmp -Destination $Path -Force
}

function Update-AgentHeartbeat([string]$Status) {
    Write-JsonAtomic $HeartbeatFile @{
        agent_id  = ${AgentId}
        timestamp = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        iso_utc   = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        files     = $Files
        status    = $Status
    }
}

function Assert-NoReparse([string]$Path) {
    $current = Get-Item -LiteralPath $Path -Force
    while ($null -ne $current) {
        if (($current.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Reparse point no admitido: $($current.FullName)"
        }
        if ($current.FullName -eq $RepoRoot) { break }
        $current = $current.Parent
    }
}

function Resolve-RelativeFiles {
    $result = @()
    foreach (${relPath} in ($Files -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })) {
        if ([IO.Path]::IsPathRooted(${relPath}) -or ${relPath}.Contains("..")) {
            throw "Ruta no relativa o no confinada: ${relPath}"
        }
        $full = [IO.Path]::GetFullPath((Join-Path $RepoRoot ${relPath}))
        $prefix = $RepoRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
        if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Ruta fuera del repositorio: ${relPath}"
        }
        Assert-NoReparse (Split-Path $full -Parent)
        $normalized = ${relPath}.Replace('\', '/')
        if ($result -contains $normalized) { throw "Ruta duplicada: $normalized" }
        $result += $normalized
    }
    return $result
}

function Assert-ProcessSuccess([string]$Name) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Name falló con código $LASTEXITCODE"
    }
}

function New-CandidateSnapshot([string]$Tree, [bool]$IncludeStaging) {
    $candidate = Join-Path $TxRoot "candidates\${AgentId}-$([Guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $candidate -Force | Out-Null
    Assert-NoReparse (Split-Path $candidate -Parent)
    $archive = Join-Path $TxRoot "candidates\${AgentId}-$([Guid]::NewGuid().ToString('N')).zip"
    if ($Tree -eq "HEAD") {
        git archive --format=zip HEAD -o $archive
    } else {
        git archive --format=zip $Tree -o $archive
    }
    Assert-ProcessSuccess "git archive $Tree"
    Expand-Archive -LiteralPath $archive -DestinationPath $candidate -Force
    Remove-Item -LiteralPath $archive -Force
    if ($IncludeStaging) {
        foreach (${relPath} in (Resolve-RelativeFiles)) {
            $source = Join-Path $StagingDir ${relPath}
            if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
                throw "Candidato staged ausente: ${relPath}"
            }
            $destination = Join-Path $candidate ${relPath}
            $parent = Split-Path $destination -Parent
            if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
            Copy-Item -LiteralPath $source -Destination $destination -Force
        }
    }
    return $candidate
}

function Resolve-CandidateManifest([string]$CandidateDir) {
    $candidateRoot = [IO.Path]::GetFullPath($CandidateDir).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $crateDir = [IO.Path]::GetFullPath((Join-Path $candidateRoot $CratePath))
    $prefix = $candidateRoot + [IO.Path]::DirectorySeparatorChar
    if ($crateDir -ne $candidateRoot -and -not $crateDir.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "CratePath fuera del candidato: $CratePath"
    }
    Assert-NoReparse $crateDir
    $resolved = (Resolve-Path -LiteralPath $crateDir).Path
    $manifest = [IO.Path]::GetFullPath((Join-Path $resolved "Cargo.toml"))
    if (-not $manifest.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Manifest fuera del candidato: $manifest"
    }
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) { throw "Manifest ausente: $manifest" }
    return $manifest
}

function Get-CandidateFileEvidence([string]$CandidateDir) {
    $evidence = @{}
    foreach (${relPath} in (Resolve-RelativeFiles)) {
        $path = Join-Path $CandidateDir ${relPath}
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Archivo del candidato ausente: ${relPath}" }
        $blob = (git --work-tree $CandidateDir hash-object -w --path=${relPath} $path).Trim()
        Assert-ProcessSuccess "git hash-object ${relPath}"
        $evidence[${relPath}] = @{
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash
            blob = $blob
        }
    }
    return $evidence
}

function Assert-CandidateTree([string]$CandidateDir, [string]$ExpectedTree) {
    $previousIndex = $env:GIT_INDEX_FILE
    $env:GIT_INDEX_FILE = Join-Path $RepoRoot ".git\index_${AgentId}_$([Guid]::NewGuid().ToString('N'))"
    try {
        git read-tree $ExpectedTree
        Assert-ProcessSuccess "git read-tree $ExpectedTree"
        git --work-tree $CandidateDir diff --quiet --no-ext-diff
        Assert-ProcessSuccess "el candidato difiere del árbol certificado"
    } finally {
        $env:GIT_INDEX_FILE = $previousIndex
    }
}

function New-CandidateTree([string]$BaseCommit, [string]$CandidateDir) {
    $previousIndex = $env:GIT_INDEX_FILE
    $env:GIT_INDEX_FILE = Join-Path $RepoRoot ".git\index_${AgentId}_$([Guid]::NewGuid().ToString('N'))"
    try {
        git read-tree $BaseCommit
        Assert-ProcessSuccess "git read-tree $BaseCommit"
        foreach (${relPath} in (Resolve-RelativeFiles)) {
            $existing = git ls-files --stage -- ${relPath}
            $mode = if ($existing) { ($existing -split '\s+')[0] } else { "100644" }
            if ($mode -notin @("100644", "100755")) { throw "Modo no admitido para ${relPath}: $mode" }
            $blob = (git --work-tree $CandidateDir hash-object -w --path=${relPath} (Join-Path $CandidateDir ${relPath})).Trim()
            Assert-ProcessSuccess "git hash-object ${relPath}"
            git update-index --add --cacheinfo "$mode,$blob,${relPath}"
            Assert-ProcessSuccess "git update-index ${relPath}"
        }
        $tree = (git write-tree).Trim()
        Assert-ProcessSuccess "git write-tree"
        return $tree
    } finally {
        $env:GIT_INDEX_FILE = $previousIndex
    }
}

function Save-BuildEnvironment {
    $saved = @{}
    foreach ($variable in Get-ChildItem Env:) {
        if ($variable.Name -match '^(CARGO_|RUST|RUSTUP_)' -and $variable.Name -notin @("CARGO_HOME", "RUSTUP_HOME")) {
            $saved[$variable.Name] = $variable.Value
            Remove-Item -LiteralPath "Env:\$($variable.Name)"
        }
    }
    return $saved
}

function Restore-BuildEnvironment([hashtable]$Saved) {
    foreach ($variable in Get-ChildItem Env:) {
        if ($variable.Name -match '^(CARGO_|RUST|RUSTUP_)' -and $variable.Name -notin @("CARGO_HOME", "RUSTUP_HOME")) {
            Remove-Item -LiteralPath "Env:\$($variable.Name)"
        }
    }
    foreach ($name in $Saved.Keys) { Set-Item -LiteralPath "Env:\$name" -Value $Saved[$name] }
}

function Get-CandidateCargoConfig([string]$CandidateDir) {
    $modern = Join-Path $CandidateDir ".cargo\config.toml"
    $legacy = Join-Path $CandidateDir ".cargo\config"
    $hasModern = Test-Path -LiteralPath $modern -PathType Leaf
    $hasLegacy = Test-Path -LiteralPath $legacy -PathType Leaf
    if ($hasModern -and $hasLegacy) { throw "Configuración Cargo duplicada en el candidato." }
    $path = if ($hasModern) { $modern } elseif ($hasLegacy) { $legacy } else { "" }
    if ($path) {
        $content = Get-Content -LiteralPath $path -Raw
        if ($content -match '(?im)^\s*runner\s*=') { throw "Runner de tests no admitido en la configuración candidata." }
        $wrappers = [regex]::Matches($content, '(?im)^\s*(?:rustc-wrapper|rustc-workspace-wrapper)\s*=\s*["'']([^"'']+)["'']')
        foreach ($wrapper in $wrappers) {
            if ($wrapper.Groups[1].Value -cne "sccache") {
                throw "Wrapper rustc no revisado en la configuración candidata."
            }
        }
        return @("--config", $path)
    }
    return @()
}

function Get-CandidateToolchain([string]$CandidateDir) {
    $modern = Join-Path $CandidateDir "rust-toolchain.toml"
    $legacy = Join-Path $CandidateDir "rust-toolchain"
    $hasModern = Test-Path -LiteralPath $modern -PathType Leaf
    $hasLegacy = Test-Path -LiteralPath $legacy -PathType Leaf
    if ($hasModern -and $hasLegacy) { throw "Toolchain Rust duplicado en el candidato." }
    if ($hasModern -or $hasLegacy) {
        $path = if ($hasModern) { $modern } else { $legacy }
        $content = Get-Content -LiteralPath $path -Raw
        $channel = [regex]::Match($content, '(?m)^\s*channel\s*=\s*["'']([^"'']+)["'']')
        if ($channel.Success) { return $channel.Groups[1].Value }
        $line = @($content -split '\r?\n' | ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not $_.StartsWith("#") } | Select-Object -First 1)
        if ($line.Count -eq 1) { return $line[0] }
        throw "Toolchain Rust inválido en el candidato."
    }
    $active = (& rustup show active-toolchain 2>$null | Select-Object -First 1)
    Assert-ProcessSuccess "rustup show active-toolchain"
    return ($active -split '\s+')[0]
}

function Invoke-CandidateValidation([string]$BaseCommit, [string]$CandidateDir, [string]$ExpectedTree, [bool]$Workspace) {
    $fileList = @(Resolve-RelativeFiles)
    if ($fileList.Count -eq 0) { throw "Validate requiere una lista no vacía de archivos." }
    $diagPath = Join-Path $DiagDir "${AgentId}.json"
    Write-JsonAtomic $diagPath @{ status = "VALIDATING"; agent = ${AgentId}; candidate = $CandidateDir; base_commit = $BaseCommit; candidate_tree = $ExpectedTree }
    $manifest = Resolve-CandidateManifest $CandidateDir
    $before = Get-CandidateFileEvidence $CandidateDir
    Assert-CandidateTree $CandidateDir $ExpectedTree
    if ((git rev-parse refs/heads/main).Trim() -ne $BaseCommit) { throw "main cambió antes de validar el candidato." }
    if (-not (Get-Command sccache -ErrorAction SilentlyContinue)) { throw "sccache es obligatorio para validar." }

    $toolchain = Get-CandidateToolchain $CandidateDir
    $previousEnvironment = Save-BuildEnvironment
    $locationPushed = $false
    try {
        $env:RUSTUP_TOOLCHAIN = $toolchain
        $env:CARGO_TARGET_DIR = Join-Path $RepoRoot "target"
        $env:RUSTC_WRAPPER = "sccache"
        $cargo = (Get-Command cargo -CommandType Application).Source
        $configArgs = @(Get-CandidateCargoConfig $CandidateDir)
        Push-Location ([IO.Path]::GetTempPath())
        $locationPushed = $true
        $targetConfigOut = & $cargo @configArgs -Z unstable-options config get target 2>&1
        $targetConfigText = $targetConfigOut | Out-String
        if ($LASTEXITCODE -eq 0 -and $targetConfigText -match '\brunner\s*[:=]') {
            throw "Runner de tests detectado en la configuración Cargo efectiva."
        }
        if ($LASTEXITCODE -ne 0 -and $targetConfigText -notmatch 'not found|not set') {
            throw "No se pudo inspeccionar la configuración Cargo efectiva: $targetConfigText"
        }
        $compilerConfigOut = & $cargo @configArgs -Z unstable-options config get build.rustc 2>&1
        if ($LASTEXITCODE -eq 0) { throw "Compilador rustc sustituido en la configuración Cargo efectiva." }
        foreach ($wrapperKey in @("build.rustc-wrapper", "build.rustc-workspace-wrapper")) {
            $wrapperOut = & $cargo @configArgs -Z unstable-options config get $wrapperKey 2>&1
            $wrapperText = $wrapperOut | Out-String
            if ($LASTEXITCODE -eq 0) {
                $approved = '(?m)^' + [regex]::Escape($wrapperKey) + '\s*=\s*["'']sccache(?:\.exe)?["'']\s*$'
                if ($wrapperText -notmatch $approved) { throw "Wrapper Cargo no revisado: $wrapperText" }
            } elseif ($wrapperText -notmatch 'not found|not set') {
                throw "No se pudo inspeccionar ${wrapperKey}: $wrapperText"
            }
        }
        $fmtOut = & $cargo @configArgs fmt --manifest-path $manifest -- --check 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-JsonAtomic $diagPath @{ status = "FAIL_FMT"; agent = ${AgentId}; candidate = $CandidateDir; output = ($fmtOut | Out-String) }
            throw "Fallo en cargo fmt --check. Revisa .agent_tx/diagnostics/${AgentId}.json"
        }

        $scope = @()
        if ($Workspace) { $scope += "--workspace" }
        $clippyOut = & $cargo @configArgs clippy --manifest-path $manifest @scope --all-targets --locked -- -D warnings -D clippy::unwrap_used -D clippy::expect_used -D clippy::panic 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-JsonAtomic $diagPath @{ status = "FAIL_CLIPPY"; agent = ${AgentId}; candidate = $CandidateDir; output = ($clippyOut | Out-String) }
            throw "Fallo en cargo clippy estricto. Revisa .agent_tx/diagnostics/${AgentId}.json"
        }

        $testOut = & $cargo @configArgs test --manifest-path $manifest @scope --lib --locked 2>&1
        $testText = $testOut | Out-String
        if ($LASTEXITCODE -ne 0) {
            Write-JsonAtomic $diagPath @{ status = "FAIL_TESTS"; agent = ${AgentId}; candidate = $CandidateDir; output = $testText }
            throw "Fallo en tests de invariantes analíticos. Revisa .agent_tx/diagnostics/${AgentId}.json"
        }
        $testMatches = [regex]::Matches($testText, '(?m)^test result: ok\. (\d+) passed; (\d+) failed;')
        $testsPassed = [int64]0
        foreach ($match in $testMatches) { $testsPassed += [int64]$match.Groups[1].Value }
        if ($testMatches.Count -eq 0 -or $testsPassed -le 0) {
            Write-JsonAtomic $diagPath @{ status = "FAIL_TEST_EVIDENCE"; agent = ${AgentId}; candidate = $CandidateDir; output = $testText }
            throw "Cargo no produjo evidencia terminal de tests ejecutados."
        }
    } finally {
        if ($locationPushed) { Pop-Location }
        Restore-BuildEnvironment $previousEnvironment
    }

    $after = Get-CandidateFileEvidence $CandidateDir
    foreach (${relPath} in $fileList) {
        if ($after[${relPath}].sha256 -cne $before[${relPath}].sha256 -or $after[${relPath}].blob -cne $before[${relPath}].blob) {
            Write-JsonAtomic $diagPath @{ status = "FAIL_CANDIDATE_MUTATION"; agent = ${AgentId}; candidate = $CandidateDir }
            throw "El candidato cambió durante la validación: ${relPath}"
        }
    }
    Assert-CandidateTree $CandidateDir $ExpectedTree
    if ((git rev-parse refs/heads/main).Trim() -ne $BaseCommit) { throw "main cambió durante la validación; repetir sobre la nueva base." }
    Write-JsonAtomic $diagPath @{
        status = "OK_CANDIDATE_CHECKS"
        agent = ${AgentId}
        candidate = $CandidateDir
        base_commit = $BaseCommit
        candidate_tree = $ExpectedTree
        crate_path = $CratePath
        workspace = $Workspace
        files = $after
        commands = @("cargo fmt --check", "cargo clippy --all-targets --locked -D warnings --workspace=$Workspace", "cargo test --lib --locked --workspace=$Workspace")
        cargo_path = $cargo
        cargo_config = $configArgs
        rustup_toolchain = $toolchain
        target_config_inspected = $true
        test_suites = $testMatches.Count
        tests_passed = $testsPassed
        timestamp = (Get-Date).ToUniversalTime().ToString("o")
    }
}

function Convert-GitHubGlob([string]$Pattern) {
    $prefix = "^"
    if ($Pattern.StartsWith("**/")) {
        $prefix = "^(.*/)?"
        $Pattern = $Pattern.Substring(3)
    } elseif (-not $Pattern.Contains("/")) {
        $prefix = "(^|/)"
    }
    if ($Pattern.EndsWith("/")) { $Pattern += "**" }
    if ($Pattern.Contains("[")) { throw "Glob de clase no soportado: $Pattern" }
    $escaped = [regex]::Escape($Pattern).Replace('\*\*', '.*').Replace('\*', '[^/]*').Replace('\?', '[^/]')
    return "$prefix$escaped$"
}

function Test-GitHubPatternSet([string]$Path, [string[]]$Patterns) {
    $matched = $false
    foreach ($pattern in $Patterns) {
        $negated = $pattern.StartsWith("!")
        $glob = if ($negated) { $pattern.Substring(1) } else { $pattern }
        if ([regex]::IsMatch($Path, (Convert-GitHubGlob $glob))) { $matched = -not $negated }
    }
    return $matched
}

function Get-YamlList([string]$Block, [string]$Key) {
    $inline = [regex]::Match($Block, "(?m)^\s{4}$([regex]::Escape($Key)):\s*\[([^\]]*)\]")
    if ($inline.Success) {
        $values = @($inline.Groups[1].Value -split "," | ForEach-Object { $_.Trim().Trim("'").Trim('"') } | Where-Object { $_ })
        return @{ present = $true; patterns = $values }
    }
    $marker = [regex]::Match($Block, "(?m)^\s{4}$([regex]::Escape($Key)):\s*$")
    if (-not $marker.Success) { return @{ present = $false; patterns = @() } }
    $lines = $Block -split '\r?\n'
    $values = @()
    $collecting = $false
    foreach ($line in $lines) {
        if ($line -match "^\s{4}$([regex]::Escape($Key)):\s*$") { $collecting = $true; continue }
        if ($collecting -and $line -match '^\s{4}\S') { break }
        if ($collecting -and $line -match '^\s*-\s+(.+?)\s*$') {
            $values += $Matches[1].Trim().Trim("'").Trim('"')
        }
    }
    return @{ present = $true; patterns = $values }
}

function Get-JsonList($Node, [string]$Key) {
    if ($null -eq $Node) { return @{ present = $false; patterns = @() } }
    $property = $Node.PSObject.Properties[$Key]
    if ($null -eq $property) { return @{ present = $false; patterns = @() } }
    return @{ present = $true; patterns = @($property.Value) }
}

function Get-PushWorkflowScope([string]$BaseSha, [string]$Sha) {
    $changed = @(git diff --name-only $BaseSha $sha)
    Assert-ProcessSuccess "diff base-head para CI"
    $workflowPaths = @(git ls-tree -r --name-only $sha -- .github/workflows | Where-Object { $_ -match '\.ya?ml$' })
    Assert-ProcessSuccess "inventario de workflows"
    $required = @()
    foreach ($path in $workflowPaths) {
        $content = (git show "${Sha}:${path}" | Out-String)
        Assert-ProcessSuccess "leer workflow $path"
        $name = ""
        if ($content.TrimStart().StartsWith("{")) {
            $spec = $content | ConvertFrom-Json
            $name = [string]$spec.name
            $onObject = $spec.PSObject.Properties['on'].Value
            if ($null -eq $onObject) { throw "Workflow JSON sin trigger on: $path" }
            $pushNode = $onObject.PSObject.Properties['push'].Value
            if ($null -eq $pushNode) { continue }
            $branches = Get-JsonList $pushNode "branches"
            $branchesIgnore = Get-JsonList $pushNode "branches-ignore"
            $paths = Get-JsonList $pushNode "paths"
            $pathsIgnore = Get-JsonList $pushNode "paths-ignore"
        } else {
            $on = [regex]::Match($content, '(?m)^(?:"on"|on):[ \t]*(.*)$')
            if (-not $on.Success) { throw "Workflow sin trigger on: $path" }
            $pushBlock = ""
            if ($on.Groups[1].Value.Trim()) {
                $triggerLine = $on.Groups[1].Value.Trim()
                if ($triggerLine -notmatch '\bpush\b') { continue }
            } else {
                $push = [regex]::Match($content, '(?m)^\s{2}push:\s*(.*)$')
                if (-not $push.Success) { continue }
                $lines = $content -split '\r?\n'
                $collecting = $false
                foreach ($line in $lines) {
                    if ($line -match '^\s{2}push:\s*(.*)$') { $collecting = $true; continue }
                    if ($collecting -and $line -match '^\s{0,2}\S') { break }
                    if ($collecting) { $pushBlock += $line + "`n" }
                }
            }
            $branches = Get-YamlList $pushBlock "branches"
            $branchesIgnore = Get-YamlList $pushBlock "branches-ignore"
            $paths = Get-YamlList $pushBlock "paths"
            $pathsIgnore = Get-YamlList $pushBlock "paths-ignore"
            $name = [regex]::Match($content, '(?m)^name:\s*(.+?)\s*$').Groups[1].Value.Trim().Trim("'").Trim('"')
        }
        if ($branches.present -and -not (Test-GitHubPatternSet "main" $branches.patterns)) { continue }
        if ($branchesIgnore.present -and (Test-GitHubPatternSet "main" $branchesIgnore.patterns)) { continue }
        if ($paths.present) {
            $accepted = @($changed | Where-Object { Test-GitHubPatternSet $_ $paths.patterns })
            if ($accepted.Count -eq 0) { continue }
        }
        if ($pathsIgnore.present) {
            $remaining = @($changed | Where-Object { -not (Test-GitHubPatternSet $_ $pathsIgnore.patterns) })
            if ($remaining.Count -eq 0) { continue }
        }
        if (-not $name) { $name = [IO.Path]::GetFileNameWithoutExtension($path) }
        if ($required -contains $name) { throw "Nombre de workflow duplicado: $name" }
        $required += $name
    }
    return $required
}

function Enter-AgentTransaction {
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $digest = -join ($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($RepoRoot)) | ForEach-Object { $_.ToString("x2") })
    } finally {
        $sha.Dispose()
    }
    $mutex = [Threading.Mutex]::new($false, "Local\TraderGeminiZeroWait_${digest}_${AgentId}")
    $acquired = $false
    try {
        $acquired = $mutex.WaitOne(0)
    } catch [Threading.AbandonedMutexException] {
        $acquired = $true
    }
    if (-not $acquired) {
        $mutex.Dispose()
        throw "Otra transacción de ${AgentId} está activa; no se esperará en cola."
    }
    return $mutex
}

function Get-FreshOriginMain {
    $remote = @(git ls-remote --heads origin refs/heads/main)
    Assert-ProcessSuccess "ls-remote de origin/main"
    if ($remote.Count -ne 1) { throw "origin/main no tiene exactamente una punta remota." }
    $parts = @($remote[0] -split '\s+')
    if ($parts.Count -ne 2 -or $parts[0] -notmatch '^(?:[0-9a-f]{40}|[0-9a-f]{64})$' -or $parts[1] -cne "refs/heads/main") {
        throw "Respuesta inválida para origin/main."
    }
    $remoteHead = $parts[0]
    $trackingHead = (git rev-parse refs/remotes/origin/main).Trim()
    Assert-ProcessSuccess "resolver origin/main fresco"
    if ($trackingHead -cne $remoteHead) { throw "origin/main local no coincide con la punta remota; HEAD local no sustituye esta verificación." }
    git cat-file -e "$remoteHead^{commit}"
    Assert-ProcessSuccess "objeto de origin/main"
    return $remoteHead
}

function Test-WorktreeReservation([string]$Branch, [string]$Tip, [string[]]$Worktrees) {
    $branchPattern = "(?m)^branch refs/heads/$([regex]::Escape($Branch))$"
    $tipPattern = "(?m)^HEAD $([regex]::Escape($Tip))$"
    return [bool](($Worktrees -match $branchPattern) -or ($Worktrees -match $tipPattern))
}

function Test-LeaseOrReservation([string]$Branch) {
    $syncPath = Join-Path $RepoRoot "AGENT_SYNC.md"
    if (-not (Test-Path -LiteralPath $syncPath -PathType Leaf)) { throw "No se puede comprobar leases sin AGENT_SYNC.md." }
    $syncText = Get-Content -LiteralPath $syncPath -Raw -ErrorAction Stop
    $agentTable = [regex]::Match($syncText, '(?ms)^## 1\..*?^## 2\.').Value
    if ([string]::IsNullOrWhiteSpace($agentTable)) { throw "No se pudo aislar el registro vigente de agentes." }
    if ($agentTable -match [regex]::Escape($Branch)) { return $true }
    $leaseFiles = @(Get-ChildItem -LiteralPath $TxRoot -Recurse -File -ErrorAction Stop | Where-Object { $_.Name -match '(?i)(lease|reserva|reservation)' })
    foreach ($file in $leaseFiles) {
        $content = Get-Content -LiteralPath $file.FullName -Raw -ErrorAction Stop
        if ($content -match [regex]::Escape($Branch)) { return $true }
    }
    return $false
}

function Test-MergedBranchEvidence([string]$Branch, [string]$Tip, [string]$FreshOriginMain) {
    $mergeShas = @(git log --merges --format=%H "$Tip..$FreshOriginMain")
    Assert-ProcessSuccess "buscar merge SHA de $Branch"
    foreach ($mergeSha in $mergeShas) {
        $parents = @((git show -s --format=%P $mergeSha).Trim() -split '\s+')
        Assert-ProcessSuccess "padres del merge $mergeSha"
        if ($parents -ccontains $Tip) {
            git merge-base --is-ancestor $mergeSha $FreshOriginMain 2>$null
            if ($LASTEXITCODE -eq 0) { return $true }
        }
    }
    if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
        Write-Warning "$Branch se conserva: no hay merge SHA verificado ni cliente gh para comprobar PR MERGED."
        return $false
    }
    $prJson = gh pr list --head $Branch --state all --limit 100 --json state,headRefOid,mergeCommit
    Assert-ProcessSuccess "consultar PR de $Branch"
    $prs = $prJson | ConvertFrom-Json
    foreach ($pr in @($prs)) {
        if ([string]$pr.state -cne "MERGED" -or [string]$pr.headRefOid -cne $Tip) { continue }
        $mergeSha = [string]$pr.mergeCommit.oid
        if ($mergeSha -notmatch '^(?:[0-9a-f]{40}|[0-9a-f]{64})$') { continue }
        git cat-file -e "$mergeSha^{commit}" 2>$null
        if ($LASTEXITCODE -ne 0) { continue }
        git merge-base --is-ancestor $mergeSha $FreshOriginMain 2>$null
        if ($LASTEXITCODE -eq 0) { return $true }
    }
    return $false
}

function Test-BranchPurgeAdmission([string]$Branch, [string]$Tip, [string]$FreshOriginMain, [string[]]$Worktrees) {
    if ($Tip -notmatch '^(?:[0-9a-f]{40}|[0-9a-f]{64})$') { throw "Tip inválido para $Branch." }
    if (Test-WorktreeReservation $Branch $Tip $Worktrees) { return $false }
    if (Test-LeaseOrReservation $Branch) { return $false }
    git merge-base --is-ancestor $Tip $FreshOriginMain 2>$null
    if ($LASTEXITCODE -ne 0) { return $false }
    $exclusiveText = (git rev-list --count "${FreshOriginMain}..${Tip}").Trim()
    Assert-ProcessSuccess "commits exclusivos de $Branch"
    [int64]$exclusive = 0
    if (-not [int64]::TryParse($exclusiveText, [ref]$exclusive)) { throw "Conteo exclusivo inválido para ${Branch}: $exclusiveText" }
    if ($exclusive -ne 0) { return $false }
    return Test-MergedBranchEvidence $Branch $Tip $FreshOriginMain
}

$agentMutex = $null
if ($Action -in @("StageFiles", "Validate", "CommitAndMerge")) {
    $agentMutex = Enter-AgentTransaction
}
try {
switch ($Action) {

    "InitWorkspace" {
        $ignoreEntries = @(".agent_tx/", ".git/index_*", "target/", "*.parquet", "*.bin", "*.mmap", "*.dmp", ".env")
        $gitignorePath = Join-Path $RepoRoot ".gitignore"
        $currentIgnore = if (Test-Path -LiteralPath $gitignorePath) { Get-Content -LiteralPath $gitignorePath } else { @() }
        foreach ($entry in $ignoreEntries) {
            if ($currentIgnore -notcontains $entry) { Add-Content -LiteralPath $gitignorePath -Value $entry }
        }
        $env:SCCACHE_CACHE_SIZE = "100G"
        if (Get-Command sccache -ErrorAction SilentlyContinue) {
            sccache --start-server 2>$null | Out-Null
            sccache --show-stats 2>$null | Out-Null
            Assert-ProcessSuccess "sccache disponible"
            Update-AgentHeartbeat "INITIALIZED_SCCACHE"
        } else {
            Update-AgentHeartbeat "SCCACHE_MISSING"
            throw "sccache es obligatorio para el despliegue completo."
        }
        Write-Output "[OK] Bus Especulativo Zero-Wait inicializado para ${AgentId}."
    }

    "PurgeLegacyBranches" {
        git fetch --all --prune
        Assert-ProcessSuccess "git fetch --all --prune"
        $freshOriginMain = Get-FreshOriginMain
        $worktrees = @(git worktree list --porcelain)
        Assert-ProcessSuccess "inventario de worktrees"
        $deleted = @()
        $preserved = @()
        $localBranches = @(git branch --format="%(refname:short)" | Where-Object { $_ -ne "main" })
        Assert-ProcessSuccess "inventario de ramas locales"
        foreach ($br in $localBranches) {
            $tip = (git rev-parse "$br^{commit}").Trim()
            Assert-ProcessSuccess "resolver rama $br"
            $admission = Test-BranchPurgeAdmission -Branch $br -Tip $tip -FreshOriginMain $freshOriginMain -Worktrees $worktrees
            if ($admission) {
                if ((Get-FreshOriginMain) -cne $freshOriginMain) { throw "origin/main cambió antes de borrar $br." }
                git update-ref -d "refs/heads/$br" $tip
                Assert-ProcessSuccess "borrado CAS local $br"
                $deleted += $br
            } else {
                $preserved += $br
            }
        }
        $remoteRows = @(git ls-remote --heads origin)
        Assert-ProcessSuccess "inventario de ramas remotas"
        $remoteBranches = @()
        foreach ($row in $remoteRows) {
            $parts = @($row -split '\s+')
            if ($parts.Count -ne 2 -or $parts[0] -notmatch '^(?:[0-9a-f]{40}|[0-9a-f]{64})$' -or $parts[1] -notmatch '^refs/heads/.+$') {
                throw "Fila remota inválida durante la purga."
            }
            $name = $parts[1] -replace '^refs/heads/', ''
            if ($name -cne "main") { $remoteBranches += [pscustomobject]@{ Name = $name; Tip = $parts[0] } }
        }
        foreach ($remote in $remoteBranches) {
            $rbr = $remote.Name
            $tip = $remote.Tip
            git fetch origin "$rbr"
            Assert-ProcessSuccess "fetch de $rbr"
            git cat-file -e "$tip^{commit}"
            Assert-ProcessSuccess "objeto remoto $rbr"
            $admission = Test-BranchPurgeAdmission -Branch $rbr -Tip $tip -FreshOriginMain $freshOriginMain -Worktrees $worktrees
            if ($admission) {
                if ((Get-FreshOriginMain) -cne $freshOriginMain) { throw "origin/main cambió antes de borrar origin/$rbr." }
                $currentRemote = @(git ls-remote --heads origin "refs/heads/$rbr")
                Assert-ProcessSuccess "revalidar punta remota $rbr"
                if ($currentRemote.Count -ne 1 -or ($currentRemote[0] -split '\s+')[0] -cne $tip) { throw "La rama origin/$rbr cambió antes del borrado." }
                git push --force-with-lease="refs/heads/${rbr}:${tip}" origin --delete "$rbr"
                Assert-ProcessSuccess "eliminación remota $rbr"
                $deleted += "origin/$rbr"
            } else {
                $preserved += "origin/$rbr"
            }
        }
        git fetch --prune
        Assert-ProcessSuccess "fetch final de la purga"
        Write-Output "[OK] Purga segura: eliminadas=$($deleted -join ','); preservadas=$($preserved -join ',')."
    }

    "StageFiles" {
        Update-AgentHeartbeat "STAGING"
        foreach (${relPath} in (Resolve-RelativeFiles)) {
            $srcFile = Join-Path $RepoRoot ${relPath}
            if (-not (Test-Path -LiteralPath $srcFile -PathType Leaf)) { throw "Archivo fuente ausente: ${relPath}" }
            $dstFile = Join-Path $StagingDir ${relPath}
            $dstParent = Split-Path $dstFile -Parent
            if (-not (Test-Path -LiteralPath $dstParent)) { New-Item -ItemType Directory -Path $dstParent -Force | Out-Null }
            $tmp = "$dstFile.$([Guid]::NewGuid().ToString('N')).tmp"
            Copy-Item -LiteralPath $srcFile -Destination $tmp -Force
            Move-Item -LiteralPath $tmp -Destination $dstFile -Force
        }
        Write-Output "[OK] Archivos preparados en el Staging Overlay aislado: $StagingDir"
    }

    "Validate" {
        if ([string]::IsNullOrWhiteSpace($Files)) {
            throw "Validate requiere -Files con la lista exacta del candidato."
        }
        Update-AgentHeartbeat "VALIDATING_ZERO_WAIT"
        $baseCommit = (git rev-parse refs/heads/main).Trim()
        Assert-ProcessSuccess "resolver refs/heads/main"
        $candidate = New-CandidateSnapshot $baseCommit $true
        $candidateTree = New-CandidateTree $baseCommit $candidate
        Invoke-CandidateValidation $baseCommit $candidate $candidateTree $false
        Update-AgentHeartbeat "VALIDATED_OK"
        Write-Output "[OK] Validación Zero-Wait superada para ${AgentId}: $candidate"
    }

    "CommitAndMerge" {
        Update-AgentHeartbeat "MERGING_IN_MEMORY"
        $diagPath = Join-Path $DiagDir "${AgentId}.json"
        if (-not (Test-Path -LiteralPath $diagPath -PathType Leaf)) { throw "No hay certificado previo para ${AgentId}." }
        $diag = Get-Content -LiteralPath $diagPath | ConvertFrom-Json
        if ($diag.status -ne "OK_CANDIDATE_CHECKS") { throw "El candidato no tiene OK_CANDIDATE_CHECKS." }
        $fileList = @(Resolve-RelativeFiles)
        if ($fileList.Count -eq 0) { throw "No hay archivos para fusionar." }
        if ($diag.crate_path -cne $CratePath) { throw "CratePath difiere del candidato validado." }
        $certifiedNames = @($diag.files.PSObject.Properties.Name | Sort-Object)
        if (($certifiedNames -join "`n") -cne (($fileList | Sort-Object) -join "`n")) {
            throw "El conjunto de archivos difiere del candidato validado."
        }
        if ([string]::IsNullOrWhiteSpace([string]$diag.candidate)) { throw "Falta la ruta del candidato validado." }
        $candidateRoot = [IO.Path]::GetFullPath((Join-Path $TxRoot "candidates")).TrimEnd([IO.Path]::DirectorySeparatorChar)
        $candidate = [IO.Path]::GetFullPath([string]$diag.candidate)
        if (-not $candidate.StartsWith($candidateRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Candidato fuera del almacén confinado."
        }
        Assert-NoReparse $candidate
        $actualEvidence = Get-CandidateFileEvidence $candidate
        foreach (${relPath} in $fileList) {
            $expected = $diag.files.PSObject.Properties[${relPath}].Value
            if ($null -eq $expected -or $actualEvidence[${relPath}].sha256 -cne [string]$expected.sha256 -or $actualEvidence[${relPath}].blob -cne [string]$expected.blob) {
                throw "Candidato cambió después de validar: ${relPath}"
            }
        }
        Assert-CandidateTree $candidate ([string]$diag.candidate_tree)

        $expectedHead = [string]$diag.base_commit
        $verifiedBase = (git rev-parse "$expectedHead^{commit}").Trim()
        if ($verifiedBase -cne $expectedHead) { throw "Base certificada inválida." }
        $agentTree = [string]$diag.candidate_tree
        git cat-file -e "$agentTree^{tree}"
        Assert-ProcessSuccess "árbol certificado inválido"
        git diff --check $expectedHead $agentTree
        Assert-ProcessSuccess "revisión del diff contra la base"
        $agentCommit = (git commit-tree $agentTree -p $expectedHead -m "[${AgentId}] $Message").Trim()
        Assert-ProcessSuccess "git commit-tree"
        $currentMain = (git rev-parse refs/heads/main).Trim()
        $newCommit = $agentCommit

        if ($currentMain -ne $expectedHead) {
            $mergedTree = (git merge-tree --write-tree $currentMain $agentCommit).Trim()
            Assert-ProcessSuccess "git merge-tree; posible conflicto"
            git diff --check $currentMain $mergedTree
            Assert-ProcessSuccess "diff del merge contra main"
            git diff --check $agentCommit $mergedTree
            Assert-ProcessSuccess "diff del merge contra el candidato"
            $mergedCommit = (git commit-tree $mergedTree -p $currentMain -p $agentCommit -m "merge: ${AgentId} into main").Trim()
            Assert-ProcessSuccess "git commit-tree merge"
            $finalCandidate = New-CandidateSnapshot $mergedTree $false
            Invoke-CandidateValidation $currentMain $finalCandidate $mergedTree $true
            if ((git rev-parse refs/heads/main).Trim() -ne $currentMain) {
                throw "main avanzó durante la validación; recomponer candidato sin sobrescribir trabajo."
            }
            $newCommit = $mergedCommit
        }

        git update-ref refs/heads/main $newCommit $currentMain
        Assert-ProcessSuccess "CAS update-ref"
        Write-JsonAtomic $diagPath @{ status = "CONSUMED"; agent = ${AgentId}; commit = $newCommit; candidate = $candidate; timestamp = (Get-Date).ToUniversalTime().ToString("o") }
        Update-AgentHeartbeat "IDLE_MERGED"
        Write-Output "[OK] Commit y Merge sobre sujeto validado: $newCommit -> refs/heads/main; índice compartido intacto"
    }

    "RescueStaleAgents" {
        $nowUnix = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        $hbFiles = Get-ChildItem -LiteralPath (Join-Path $TxRoot "heartbeats") -Filter "*.json" -ErrorAction SilentlyContinue
        foreach ($file in $hbFiles) {
            $data = Get-Content -LiteralPath $file.FullName | ConvertFrom-Json
            if ($data.agent_id -notmatch '^[A-Za-z0-9_-]+$' -or $data.agent_id -ne $file.BaseName) {
                throw "Heartbeat con identidad inválida: $($file.FullName)"
            }
            $ageSeconds = $nowUnix - [long]$data.timestamp
            if ($ageSeconds -gt 300 -and $data.status -ne "IDLE_MERGED") {
                $staleAgent = $data.agent_id
                $staleStage = Join-Path $TxRoot "staging\$staleAgent"
                $rescueDest = Join-Path $RescueDir "$staleAgent-$nowUnix-$([Guid]::NewGuid().ToString('N'))"
                if (Test-Path -LiteralPath $staleStage) {
                    Assert-NoReparse $staleStage
                    Copy-Item -LiteralPath $staleStage -Destination $rescueDest -Recurse -Force
                }
                Write-Output "[RESCATE] Respaldo de $staleAgent en $rescueDest. Locks y originales preservados."
            }
        }
    }

    "StartGroupPushDaemon" {
        Write-Output "[DAEMON] Autopista Group-Push + vigilancia GitHub Actions iniciada."
        $lastPushedSha = ""
        $ciPending = @{}
        $ciPendingPath = Join-Path $TxRoot "ci_pending.json"
        if (Test-Path -LiteralPath $ciPendingPath -PathType Leaf) {
            $savedPending = Get-Content -LiteralPath $ciPendingPath | ConvertFrom-Json
            foreach ($property in $savedPending.PSObject.Properties) {
                $ciPending[$property.Name] = [string]$property.Value
            }
        }
        while ($true) {
            $localHead = (git rev-parse refs/heads/main).Trim()
            if ($localHead -ne $lastPushedSha) {
                git fetch origin main
                Assert-ProcessSuccess "git fetch origin main"
                $remoteHead = (git rev-parse origin/main).Trim()
                if ($localHead -ne $remoteHead) {
                    $update = "refs/heads/main $localHead refs/heads/main $remoteHead"
                    $update | python -I -B scripts/pre_push_gate.py verify origin "https://github.com/Jhona-la/Trader-Gemini.git" --receipt logs/prepush/attestation.json
                    Assert-ProcessSuccess "attestation del árbol actual"
                    git push --atomic origin "${localHead}:refs/heads/main"
                    if ($LASTEXITCODE -ne 0) {
                        Write-JsonAtomic $CiAlertsFile @{ status = "PUSH_FAILED_URGENT"; commit_sha = $localHead; timestamp = (Get-Date).ToUniversalTime().ToString("o") }
                        throw "Push rechazado; daemon detenido sin reintentos ciegos."
                    }
                }
                $lastPushedSha = $localHead
                if ($localHead -ne $remoteHead) {
                    $ciPending[$localHead] = $remoteHead
                    Write-JsonAtomic $ciPendingPath $ciPending
                }
            }

            if ($ciPending.Count -gt 0) {
                $runJson = gh run list --branch main --event push --limit 100 --json databaseId,status,conclusion,headSha,name,createdAt | ConvertFrom-Json
                Assert-ProcessSuccess "gh run list"
                foreach ($sha in @($ciPending.Keys)) {
                    $expected = @(Get-PushWorkflowScope $ciPending[$sha] $sha)
                    if ($expected.Count -eq 0) {
                        [void]$ciPending.Remove($sha)
                        Write-JsonAtomic $ciPendingPath $ciPending
                        Write-JsonAtomic (Join-Path $TxRoot "ci_not_required.json") @{ status = "CI_NOT_REQUIRED"; commit_sha = $sha; timestamp = (Get-Date).ToUniversalTime().ToString("o") }
                        continue
                    }
                    $latestByName = @{}
                    foreach ($run in @($runJson | Where-Object { $_.headSha -eq $sha })) {
                        if (-not $latestByName.ContainsKey($run.name)) { $latestByName[$run.name] = $run }
                    }
                    $failed = @($latestByName.Values | Where-Object { $_.status -eq "completed" -and $_.conclusion -ne "success" })
                    if ($failed.Count -gt 0) {
                        $failedLogs = gh run view $failed[0].databaseId --log-failed 2>&1 | Out-String
                        Write-JsonAtomic $CiAlertsFile @{ status = "CI_FAILED_URGENT"; commit_sha = $sha; run_id = $failed[0].databaseId; timestamp = (Get-Date).ToUniversalTime().ToString("o"); error_log = $failedLogs }
                        throw "CI falló; alerta escrita y daemon detenido."
                    }
                    $missing = @($expected | Where-Object { -not $latestByName.ContainsKey($_) })
                    if ($missing.Count -gt 0) { continue }
                    $incomplete = @($latestByName.Values | Where-Object { $_.status -ne "completed" })
                    if ($incomplete.Count -gt 0) { continue }
                    $notGreen = @($latestByName.Values | Where-Object { $_.conclusion -ne "success" })
                    if ($notGreen.Count -eq 0) {
                        [void]$ciPending.Remove($sha)
                        Write-JsonAtomic $ciPendingPath $ciPending
                        Write-JsonAtomic (Join-Path $TxRoot "ci_green.json") @{ status = "CI_GREEN"; commit_sha = $sha; workflows = $expected; timestamp = (Get-Date).ToUniversalTime().ToString("o") }
                    }
                }
            }
            Start-Sleep -Seconds 2
        }
    }
}
} finally {
    if ($null -ne $agentMutex) {
        $agentMutex.ReleaseMutex()
        $agentMutex.Dispose()
    }
}
