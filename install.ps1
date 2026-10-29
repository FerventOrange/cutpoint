# Cutpoint installer for Windows (PowerShell 5.1 and 7).
#
#   irm https://raw.githubusercontent.com/FerventOrange/cutpoint/main/install.ps1 | iex
#   $env:CUTPOINT_LITE = 1; irm https://raw.githubusercontent.com/FerventOrange/cutpoint/main/install.ps1 | iex
#
# Installs uv and ffmpeg when they are missing, then installs Cutpoint as a uv tool
# with the right torch build for this machine. Re-running it upgrades or switches mode.
# Run `.\install.ps1 -Help` for the options. `irm | iex` cannot pass parameters, so every
# option also has a CUTPOINT_* environment variable. ASCII only (Windows PowerShell 5.1
# reads a file without a byte order mark as ANSI).
#
# Release stamping: the release process sets the line below to the version just published on
# PyPI (for example: $CUTPOINT_VERSION_DEFAULT = "0.4.0") together with install.sh, and
# attaches/serves the script from that tag. The env var CUTPOINT_VERSION overrides it at run time.
param(
    [switch]$Lite,
    [switch]$Yes,
    [switch]$DryRun,
    [switch]$Cpu,
    [switch]$Cuda,
    [string]$Source = "",
    [switch]$NoFfmpeg,
    [switch]$Ollama,
    [switch]$NoOllama,
    [switch]$Help
)

function Invoke-CutpointInstaller {
    param(
        [bool]$Lite, [bool]$Yes, [bool]$DryRun, [bool]$Cpu, [bool]$Cuda,
        [string]$Source, [bool]$NoFfmpeg, [bool]$Ollama, [bool]$NoOllama, [bool]$Help, [object[]]$Extra
    )

    $CUTPOINT_VERSION_DEFAULT = "0.4.0"

    # uv is installed from this pinned installer when missing (keep in step with install.sh and the Dockerfile pin).
    $UV_VERSION = "0.13.5"
    # Oldest uv that has `uv tool install --torch-backend`.
    $UV_MIN_MAJOR = 0
    $UV_MIN_MINOR = 13

    $GIT_URL = "https://github.com/FerventOrange/cutpoint.git"
    $PYTHON_VERSION = "3.12"
    $CUDA_BACKEND = "cu126"
    $OLLAMA_MODEL = "qwen3-vl:8b"
    $OLLAMA_URL = "http://127.0.0.1:11434"
    $OLLAMA_WINGET_ID = "Ollama.Ollama"
    $OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"
    $FFMPEG_ID = "Gyan.FFmpeg"

    # Native commands print their own errors; the script checks $LASTEXITCODE.
    $ErrorActionPreference = "Continue"

    # All mutable state lives here so the nested functions can change it.
    $S = @{
        Lite = $Lite; Yes = $Yes; DryRun = $DryRun; NoFfmpeg = $NoFfmpeg
        Torch = ""; FlagTorch = ""; Source = $Source
        Backend = "none"; BackendWhy = ""
        BinDir = ""; OrigPath = ""; BinOnPath = $false
        Req = ""; NoSources = $false
        FfmpegMissing = $false; InstalledVersion = "?"
        SetupDone = "no"; OllamaDone = "no"
        OllamaChoice = ""; OllamaStatus = "not checked"; OllamaUp = $false; OllamaHint = ""
        Version = ""
    }

    # ------------------------------------------------------------ output helpers

    function Say([string]$Text) { [Console]::Out.WriteLine($Text) }
    function Warn([string]$Text) { [Console]::Error.WriteLine("WARNING: $Text") }
    function Die([string]$Text) {
        [Console]::Error.WriteLine("ERROR: $Text")
        throw "CPEXIT:1"
    }

    function Test-On([string]$Value) {
        if ([string]::IsNullOrEmpty($Value)) { return $false }
        return -not ($Value -in @("0", "no", "NO", "false", "FALSE"))
    }

    function Have([string]$Name) {
        return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
    }

    function Get-Usage {
        return @'
Cutpoint installer

Usage: install.ps1 [options]
       irm https://raw.githubusercontent.com/FerventOrange/cutpoint/main/install.ps1 | iex

Installs uv and ffmpeg when missing, then Cutpoint as a uv tool (run again to upgrade;
do not use `uv tool upgrade`, it drops the torch build).

Modes:
  (default)         Full: every feature. With an NVIDIA GPU: CUDA torch and GPU libraries
                    ("cutpoint[all]", several GB). Without: CPU torch ("cutpoint[events,ui,keyring,ski]").
  -Lite             Lite: no torch, no events/GPU extras ("cutpoint[ui,keyring,ski]"). Analyze, clips,
                    assemble and edit work on the CPU. Lite has no Ollama: it does not offer, install
                    or start it (the AI assistant can use a third-party provider instead).

Options:
  -Yes              Answer yes to the optional steps (install Ollama, cutpoint setup, Ollama model).
  -Ollama           Install Ollama when it is missing, without asking. Full mode only: with -Lite
                    it prints a warning and is ignored.
  -NoOllama         Never offer to install Ollama (full mode; lite never offers it).
  -DryRun           Print every command that would run and change nothing.
  -Cpu              Force CPU torch even when an NVIDIA GPU is found.
  -Cuda             Force CUDA torch (cu126) even when no NVIDIA GPU is found.
  -Source SRC       pypi (default; the pinned release), git or git@REF (branch, tag or commit
                    of the GitHub repository), or a local path (checkout, wheel or sdist).
  -NoFfmpeg         Do not install ffmpeg.
  -Help             Show this help.

Environment (`irm | iex` cannot pass parameters; a parameter wins over its variable; any value
except empty or 0 counts as on):
  CUTPOINT_LITE=1          same as -Lite
  CUTPOINT_YES=1           same as -Yes
  CUTPOINT_DRY_RUN=1       same as -DryRun
  CUTPOINT_NO_FFMPEG=1     same as -NoFfmpeg
  CUTPOINT_OLLAMA=1|0      same as -Ollama / -NoOllama
  CUTPOINT_TORCH=cpu|cuda  same as -Cpu / -Cuda
  CUTPOINT_SOURCE=SRC      same as -Source
  CUTPOINT_VERSION=X.Y.Z   release to install for -Source pypi (default: the version in this script)
  CUTPOINT_NO_MODIFY_PATH=1  do not run `uv tool update-shell`
  UV_TOOL_DIR, UV_TOOL_BIN_DIR, XDG_BIN_HOME   where uv puts tools and their commands
                           (default: %APPDATA%\uv\tools and ~\.local\bin)

Optional prompts (default No; with no interactive console the answer is No unless -Yes is given):
  - Install Ollama (full mode only; lite skips this step), when it is not found. Ollama runs the
    local AI models for --vision rating and the AI assistant. It uses
    `winget install --id Ollama.Ollama -e`, or prints the download link (https://ollama.com/download)
    when winget is missing or fails. Afterwards it waits for the server; it starts `ollama serve`
    (hidden) only when the Ollama app has not. A declined or failed Ollama install never fails the
    Cutpoint install.
  - Pull qwen3-vl:8b (about 6 GB), full mode only, when Ollama is running.
  - "cutpoint setup" (model weights, about 8 GB), full mode only.
'@
    }

    # Wrong command line: usage and the message on stderr, exit 2.
    function Bad-Usage([string]$Text) {
        [Console]::Error.WriteLine((Get-Usage))
        [Console]::Error.WriteLine("ERROR: $Text")
        throw "CPEXIT:2"
    }

    # Print one argument so it can be pasted into PowerShell.
    function Quote-Arg([string]$Arg) {
        if ($Arg -match '^[A-Za-z0-9_./:=@%+,\\-]+$') { return $Arg }
        if ($Arg -match '["$`]') { return "'" + $Arg.Replace("'", "''") + "'" }
        return '"' + $Arg + '"'
    }

    function Show-Cmd([string[]]$Words) {
        $out = ""
        foreach ($w in $Words) { $out += " " + (Quote-Arg $w) }
        Say "+$out"
    }

    # Print the command; run it unless -DryRun. Returns $true when it succeeded (or was dry).
    function Invoke-Run([string[]]$Words) {
        Show-Cmd $Words
        if ($S.DryRun) { return $true }
        $exe = $Words[0]
        $rest = @()
        if ($Words.Count -gt 1) { $rest = $Words[1..($Words.Count - 1)] }
        & $exe @rest | Out-Host
        return ($LASTEXITCODE -eq 0)
    }

    # True when the console can answer prompts (also under `irm | iex`, where stdin is the console).
    function Test-Interactive {
        if (-not [Environment]::UserInteractive) { return $false }
        try { if ([Console]::IsInputRedirected) { return $false } } catch { return $false }
        return $true
    }

    # Ask "question": returns $true for yes. Default No.
    function Ask([string]$Question) {
        if ($S.Yes) {
            Say "$Question yes (-Yes)"
            return $true
        }
        if ($S.DryRun) {
            Say "$Question no (dry run does not prompt; default No)"
            return $false
        }
        if (-not (Test-Interactive)) {
            Say "$Question no (no terminal to ask; use -Yes to accept)"
            return $false
        }
        $reply = ""
        try { $reply = Read-Host "$Question [y/N]" } catch { $reply = "" }
        return ($reply -match '^(y|yes)$')
    }

    # ------------------------------------------------------------ arguments

    function Read-Arguments {
        if ($Extra -and $Extra.Count -gt 0) {
            Bad-Usage "unknown option: $($Extra[0])"
        }
        if ($Cpu -and $Cuda) { Bad-Usage "-Cpu and -Cuda contradict each other" }
        if ($Cpu) { $S.FlagTorch = "cpu"; $S.Torch = "cpu" }
        if ($Cuda) { $S.FlagTorch = "cuda"; $S.Torch = "cuda" }
        if ($Lite -and $S.FlagTorch) {
            Bad-Usage "-Lite has no torch, so it cannot be combined with -$((Get-Culture).TextInfo.ToTitleCase($S.FlagTorch))"
        }
        if (Test-On $env:CUTPOINT_LITE) { $S.Lite = $true }
        if (Test-On $env:CUTPOINT_YES) { $S.Yes = $true }
        if (Test-On $env:CUTPOINT_DRY_RUN) { $S.DryRun = $true }
        if (Test-On $env:CUTPOINT_NO_FFMPEG) { $S.NoFfmpeg = $true }
        if ($Ollama -and $NoOllama) { Bad-Usage "-Ollama and -NoOllama contradict each other" }
        if ($env:CUTPOINT_OLLAMA) {
            if (Test-On $env:CUTPOINT_OLLAMA) { $S.OllamaChoice = "1" } else { $S.OllamaChoice = "0" }
        }
        if ($Ollama) { $S.OllamaChoice = "1" }
        if ($NoOllama) { $S.OllamaChoice = "0" }
        if (-not $S.Source) { $S.Source = [string]$env:CUTPOINT_SOURCE }
        if (-not $S.Torch) {
            switch ([string]$env:CUTPOINT_TORCH) {
                "" { }
                "cpu" { $S.Torch = "cpu" }
                "cuda" { $S.Torch = "cuda" }
                default { Die "CUTPOINT_TORCH must be cpu or cuda, got: $($env:CUTPOINT_TORCH)" }
            }
        }
        if (-not $S.Source) { $S.Source = "pypi" }
        $S.Version = $env:CUTPOINT_VERSION
        if (-not $S.Version) { $S.Version = $CUTPOINT_VERSION_DEFAULT }
    }

    # ------------------------------------------------------------ platform

    function Test-Platform {
        if ($PSVersionTable.PSEdition -eq "Core" -and -not $IsWindows) {
            Die "This installer supports Windows. On Linux and macOS use install.sh."
        }
    }

    function Get-Arch {
        $a = $env:PROCESSOR_ARCHITEW6432
        if (-not $a) { $a = $env:PROCESSOR_ARCHITECTURE }
        return [string]$a
    }

    # Only real "GPU n:" lines count; nvidia-smi's NVML errors mention "GPU" but list none.
    function Test-Nvidia {
        $candidates = @("nvidia-smi")
        if ($env:SystemRoot) { $candidates += (Join-Path $env:SystemRoot "System32\nvidia-smi.exe") }
        if ($env:ProgramFiles) { $candidates += (Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI\nvidia-smi.exe") }
        foreach ($smi in $candidates) {
            $cmd = Get-Command $smi -ErrorAction SilentlyContinue
            if (-not $cmd) { continue }
            try {
                $lines = @(& $cmd.Source -L 2>$null)
                foreach ($line in $lines) {
                    if ("$line" -match '^GPU [0-9]') { return $true }
                }
            } catch { }
        }
        return $false
    }

    function Select-Backend {
        $S.Backend = "none"
        $S.BackendWhy = ""
        if ($S.Lite) {
            if ($S.Torch) {
                if ($S.FlagTorch) {
                    Warn "-$((Get-Culture).TextInfo.ToTitleCase($S.FlagTorch)) is ignored in lite mode (lite has no torch)."
                } else {
                    Warn "CUTPOINT_TORCH is ignored in lite mode (lite has no torch)."
                }
            }
            $S.BackendWhy = "lite mode"
            return
        }
        $arch = Get-Arch
        $x64 = ($arch -in @("AMD64", "x86_64"))
        switch ($S.Torch) {
            "cpu" { $S.Backend = "cpu"; $S.BackendWhy = "-Cpu" }
            "cuda" {
                if (-not $x64) { Die "CUDA torch is not available on $arch." }
                $S.Backend = $CUDA_BACKEND
                $S.BackendWhy = "-Cuda"
            }
            default {
                if (Test-Nvidia) {
                    if ($x64) {
                        $S.Backend = $CUDA_BACKEND
                        $S.BackendWhy = "NVIDIA GPU found"
                    } else {
                        $S.Backend = "cpu"
                        $S.BackendWhy = "NVIDIA GPU found, but $arch has no $CUDA_BACKEND wheels (use -Cuda to try)"
                    }
                } else {
                    $S.Backend = "cpu"
                    $S.BackendWhy = "no NVIDIA GPU found"
                }
            }
        }
    }

    # ------------------------------------------------------------ bin dir and PATH

    function Normalize-Dir([string]$Dir) {
        try { $full = [IO.Path]::GetFullPath([Environment]::ExpandEnvironmentVariables($Dir)) } catch { return $Dir }
        $root = [IO.Path]::GetPathRoot($full)
        if ($full -ne $root) { $full = $full.TrimEnd('\', '/') }
        return $full
    }

    function Test-InPath([string]$PathValue, [string]$Dir) {
        $want = Normalize-Dir $Dir
        foreach ($entry in ($PathValue -split ';')) {
            if (-not $entry.Trim()) { continue }
            if ((Normalize-Dir $entry.Trim('"')) -ieq $want) { return $true }
        }
        return $false
    }

    # Same order as uv: UV_TOOL_BIN_DIR, XDG_BIN_HOME, $XDG_DATA_HOME/../bin, ~/.local/bin.
    function Get-DefaultBinDir {
        if ($env:UV_TOOL_BIN_DIR) { $dir = $env:UV_TOOL_BIN_DIR }
        elseif ($env:XDG_BIN_HOME) { $dir = $env:XDG_BIN_HOME }
        elseif ($env:XDG_DATA_HOME) { $dir = Join-Path $env:XDG_DATA_HOME "..\bin" }
        else {
            $home_ = $env:USERPROFILE
            if (-not $home_) { $home_ = $HOME }
            if (-not $home_) { Die "USERPROFILE is not set." }
            $dir = Join-Path $home_ ".local\bin"
        }
        return (Normalize-Dir $dir)
    }

    # Put the bin dir first on PATH; BinOnPath records whether the user's own PATH had it.
    function Use-BinDir {
        $S.BinOnPath = Test-InPath $S.OrigPath $S.BinDir
        if (-not (Test-InPath $env:PATH $S.BinDir)) {
            $env:PATH = "$($S.BinDir);$($env:PATH)"
        }
    }

    function Initialize-Path {
        $S.OrigPath = [string]$env:PATH
        $S.BinDir = Get-DefaultBinDir
        Use-BinDir
    }

    # Once uv is there, take the directory from uv itself.
    function Sync-BinDir {
        if ($S.DryRun) { return }
        $uvbin = ""
        try { $uvbin = [string](& uv tool dir --bin 2>$null | Select-Object -First 1) } catch { return }
        if ($LASTEXITCODE -ne 0 -or -not $uvbin) { return }
        $uvbin = Normalize-Dir $uvbin.Trim()
        if ($uvbin -ine $S.BinDir) {
            $S.BinDir = $uvbin
            Use-BinDir
        }
    }

    # Pick up PATH changes made by installers that ran in this session (winget, uv).
    function Update-SessionPath {
        $parts = @()
        foreach ($scope in @("Machine", "User")) {
            $v = [Environment]::GetEnvironmentVariable("Path", $scope)
            if ($v) { $parts += ($v -split ';') }
        }
        foreach ($p in $parts) {
            if ($p.Trim() -and -not (Test-InPath $env:PATH $p)) {
                $env:PATH = "$($env:PATH);$p"
            }
        }
    }

    # ------------------------------------------------------------ ffmpeg

    function Install-Ffmpeg {
        if ((Have "ffmpeg") -and (Have "ffprobe")) {
            Say "ffmpeg: found ($((Get-Command ffmpeg).Source))"
            return
        }
        $S.FfmpegMissing = $true
        if ($S.NoFfmpeg) {
            Say "ffmpeg: not found, skipped (-NoFfmpeg). Install ffmpeg and ffprobe before using Cutpoint."
            return
        }
        if (-not (Have "winget")) {
            Say "ffmpeg: not found."
            Warn "winget is not available: install ffmpeg yourself (winget install --id $FFMPEG_ID -e, or https://www.gyan.dev/ffmpeg/builds/)."
            return
        }
        Say "ffmpeg: not found. Installing it with winget ($FFMPEG_ID)."
        $ok = Invoke-Run @("winget", "install", "--id", $FFMPEG_ID, "-e", "--accept-package-agreements", "--accept-source-agreements")
        if (-not $ok) {
            Warn "winget install failed; run: winget install --id $FFMPEG_ID -e"
        }
        if (-not $S.DryRun) {
            Update-SessionPath
            if ((Have "ffmpeg") -and (Have "ffprobe")) { $S.FfmpegMissing = $false }
        }
    }

    # ------------------------------------------------------------ uv

    function Get-UvVersionText {
        try {
            $t = [string](& uv --version 2>$null | Select-Object -First 1)
            if ($t -match '^uv\s+(\S+)') { return $Matches[1] }
        } catch { }
        return ""
    }

    function Test-UvVersionOk {
        $v = Get-UvVersionText
        if ($v -notmatch '^(\d+)\.(\d+)') { return $false }
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -gt $UV_MIN_MAJOR) { return $true }
        return ($major -eq $UV_MIN_MAJOR -and $minor -ge $UV_MIN_MINOR)
    }

    function Install-Uv {
        $url = "https://astral.sh/uv/$UV_VERSION/install.ps1"
        if ($S.DryRun) {
            Say "+ `$env:UV_INSTALL_DIR='$($S.BinDir)'; `$env:UV_NO_MODIFY_PATH='1'; irm $url | iex"
            return
        }
        try {
            [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        } catch { }
        $tmp = Join-Path ([IO.Path]::GetTempPath()) ("uv-install-" + [guid]::NewGuid().ToString("N") + ".ps1")
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $tmp
        } catch {
            Remove-Item -Force -ErrorAction SilentlyContinue $tmp
            Die "Could not download $url"
        }
        Say "+ powershell <uv installer $UV_VERSION>"
        $psExe = (Get-Process -Id $PID).Path
        $oldDir = $env:UV_INSTALL_DIR
        $oldMod = $env:UV_NO_MODIFY_PATH
        $env:UV_INSTALL_DIR = $S.BinDir
        $env:UV_NO_MODIFY_PATH = "1"
        try {
            & $psExe -NoProfile -ExecutionPolicy Bypass -File $tmp | Out-Host
            $rc = $LASTEXITCODE
        } finally {
            $env:UV_INSTALL_DIR = $oldDir
            $env:UV_NO_MODIFY_PATH = $oldMod
            Remove-Item -Force -ErrorAction SilentlyContinue $tmp
        }
        if ($rc -ne 0) {
            Die "The uv installer failed. Install uv yourself (https://docs.astral.sh/uv/) and run again."
        }
        if (-not (Have "uv")) {
            Die "uv was installed but is not on PATH ($($S.BinDir)). Open a new terminal and run again."
        }
    }

    function Confirm-Uv {
        if (Have "uv") {
            if (Test-UvVersionOk) {
                Say "uv: found (uv $(Get-UvVersionText))"
                return
            }
            Say "uv: $(Get-UvVersionText) is older than $UV_MIN_MAJOR.$UV_MIN_MINOR (needed for --torch-backend)."
            if ($S.DryRun) {
                Say "+ uv self update $UV_VERSION"
                return
            }
            if (Invoke-Run @("uv", "self", "update", $UV_VERSION)) {
                if (Test-UvVersionOk) { return }
            }
            Die "Could not upgrade uv here (it was probably installed with pip, WinGet or Scoop). Upgrade it that way to $UV_MIN_MAJOR.$UV_MIN_MINOR or newer, or remove it so this script can install its own, then run again."
        }
        Say "uv: not found. Installing uv $UV_VERSION."
        Install-Uv
    }

    # ------------------------------------------------------------ cutpoint

    # Sets Req (the requirement) and NoSources.
    function Resolve-Source {
        # ui and ski are extras before the packaging change and empty aliases after it, so spelling
        # them out is correct on both layouts. keyring is an extra only in releases after 0.3.0
        # (0.3.0 has none: uv warns and skips it), so stamp a release that has it. Keep in step with
        # install.sh. ("all" includes them.)
        if ($S.Backend -eq "none") { $extra = "[ui,keyring,ski]" }
        elseif ($S.Backend -eq "cpu") { $extra = "[events,ui,keyring,ski]" }
        else { $extra = "[all]" }
        $S.NoSources = $false
        $src = $S.Source
        if ($src -eq "pypi") {
            $S.Req = "cutpoint$extra==$($S.Version)"
        } elseif ($src -eq "git" -or $src.StartsWith("git@")) {
            $ref = $src.Substring(3).TrimStart("@")
            $suffix = ""
            if ($ref) { $suffix = "@$ref" }
            $S.Req = "cutpoint$extra @ git+$GIT_URL$suffix"
            $S.NoSources = $true
        } else {
            if (-not (Test-Path -LiteralPath $src)) {
                Die "-Source path does not exist: $src"
            }
            $full = (Resolve-Path -LiteralPath $src).ProviderPath
            $full = $full.TrimEnd('\', '/')
            if ($full -match '^[A-Za-z]:$') { $full += "\" }
            $enc = $full.Replace("%", "%25").Replace(" ", "%20").Replace("#", "%23").Replace("?", "%3F").Replace("\", "/")
            if ($enc.StartsWith("//")) { $url = "file:$enc" } else { $url = "file:///$enc" }
            $S.Req = "cutpoint$extra @ $url"
            # The checkout's own [tool.uv.sources] pins torch to the cu126 index and would beat
            # --torch-backend; a PyPI wheel has no such table, so ignore it to match a release.
            $S.NoSources = $true
        }
    }

    function Install-Cutpoint {
        $words = @("uv", "tool", "install", $S.Req)
        if ($S.Backend -ne "none") { $words += @("--torch-backend", $S.Backend) }
        $words += @("--python", $PYTHON_VERSION, "--reinstall")
        if ($S.NoSources) { $words += "--no-sources" }
        if (-not $S.Lite) {
            if ($S.Backend -eq $CUDA_BACKEND) {
                Say "Downloading Cutpoint with CUDA torch and GPU libraries (several GB); this can take a while."
            } else {
                Say "Downloading Cutpoint with CPU torch (about 600 MB to download, about 2 GB on disk); this can take a while."
            }
        }
        if (-not (Invoke-Run $words)) {
            Die "uv tool install failed. Fix the error above and run again."
        }
    }

    function Update-Shell {
        if (Test-On $env:CUTPOINT_NO_MODIFY_PATH) { return }
        if ($S.BinOnPath) { return }
        Show-Cmd @("uv", "tool", "update-shell")
        if ($S.DryRun) { return }
        # uv skips the registry when the bin dir is on PATH, and Initialize-Path put it there: ask uv
        # with the PATH the user had. It exits 2 with "already up-to-date" on a re-run, which is fine.
        $newPath = $env:PATH
        $env:PATH = $S.OrigPath
        try {
            $uvExe = (Get-Command uv -ErrorAction SilentlyContinue)
            if (-not $uvExe) { $uvExe = Get-Command (Join-Path $S.BinDir "uv.exe") -ErrorAction SilentlyContinue }
            $text = ""
            $rc = 1
            if ($uvExe) {
                $text = ((& $uvExe.Source tool update-shell 2>&1 | ForEach-Object { "$_" }) -join "`n")
                $rc = $LASTEXITCODE
            }
        } finally {
            $env:PATH = $newPath
        }
        if ($rc -eq 0) {
            Say $text
        } elseif ($text -match "already up-to-date") {
            Say "Shell configuration is already up to date."
        } else {
            [Console]::Error.WriteLine($text)
            Warn "uv tool update-shell failed; add $($S.BinDir) to PATH yourself."
        }
    }

    function Test-Install {
        if ($S.DryRun) {
            Say "+ cutpoint --version"
            Say "+ cutpoint capabilities --json"
            return
        }
        if (-not (Have "cutpoint")) {
            Die "cutpoint is not on PATH after the install ($($S.BinDir)). Open a new terminal and run: cutpoint --version"
        }
        Show-Cmd @("cutpoint", "--version")
        $ver = ((& cutpoint --version 2>&1 | ForEach-Object { "$_" }) -join "`n")
        if ($LASTEXITCODE -ne 0) { Die "cutpoint --version failed: $ver" }
        $S.InstalledVersion = $ver.Trim()
        Say $S.InstalledVersion
        Show-Cmd @("cutpoint", "capabilities", "--json")
        $caps = ((& cutpoint capabilities --json 2>$null | ForEach-Object { "$_" }) -join "`n")
        if ($LASTEXITCODE -eq 0 -and $caps -match '"ok":\s*true') {
            Say "capabilities: ok"
        } else {
            Warn "cutpoint capabilities --json did not report ok; run 'cutpoint capabilities' to see why."
        }
    }

    # ------------------------------------------------------------ ollama

    function Test-OllamaServer {
        try {
            $null = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 -Uri "$OLLAMA_URL/api/version"
            return $true
        } catch { return $false }
    }

    # Wait up to about 20 seconds for the server.
    function Wait-Ollama {
        for ($n = 0; $n -lt 20; $n++) {
            if (Test-OllamaServer) { return $true }
            Start-Sleep -Seconds 1
        }
        return (Test-OllamaServer)
    }

    # True when the ollama command is usable; also finds the per-user install that is not on PATH yet.
    function Find-Ollama {
        if (Have "ollama") { return $true }
        if ($env:LOCALAPPDATA) {
            $dir = Join-Path $env:LOCALAPPDATA "Programs\Ollama"
            if (Test-Path -LiteralPath (Join-Path $dir "ollama.exe")) {
                $env:PATH = "$($env:PATH);$dir"
                return $true
            }
        }
        return $false
    }

    # The Ollama app normally starts the server itself; start `ollama serve` (hidden) only when it did not.
    function Start-OllamaServe {
        if ($S.DryRun) {
            Say "+ ollama serve   (hidden; only when the server is not up after 20 s)"
            return $true
        }
        Say "Ollama is not answering at ${OLLAMA_URL}: starting 'ollama serve' in the background."
        try {
            Start-Process -FilePath (Get-Command ollama).Source -ArgumentList "serve" -WindowStyle Hidden
        } catch { return $false }
        return (Wait-Ollama)
    }

    # Install Ollama with winget; returns $true when it succeeded (or is only planned).
    function Install-OllamaPackage {
        if (-not (Have "winget")) {
            Warn "Ollama not installed: winget is not available; download Ollama from $OLLAMA_DOWNLOAD_URL"
            $S.OllamaStatus = "skipped (winget not found; download Ollama from $OLLAMA_DOWNLOAD_URL)"
            return $false
        }
        $ok = Invoke-Run @("winget", "install", "--id", $OLLAMA_WINGET_ID, "-e", "--silent", "--accept-package-agreements", "--accept-source-agreements")
        if (-not $ok) {
            Warn "The Ollama install failed; install it later from $OLLAMA_DOWNLOAD_URL (or: winget install --id $OLLAMA_WINGET_ID -e)"
            $S.OllamaStatus = "failed (install it from $OLLAMA_DOWNLOAD_URL)"
            return $false
        }
        return $true
    }

    # Offer to install Ollama; sets OllamaStatus and OllamaUp (true when the server is, or in a dry run
    # would be, up). Never fails the install.
    function Invoke-OllamaOffer {
        $S.OllamaUp = $false
        $S.OllamaHint = ""
        if (Find-Ollama) {
            $S.OllamaStatus = "already installed"
            if (Test-OllamaServer) {
                $S.OllamaUp = $true
            } elseif (-not $S.DryRun) {
                $S.OllamaHint = "Ollama is installed but not running; start it, then: ollama pull $OLLAMA_MODEL"
            }
            return
        }
        if (Test-OllamaServer) {
            $S.OllamaStatus = "server running at $OLLAMA_URL (no ollama command; not installed)"
            return
        }
        if ($S.OllamaChoice -eq "0") {
            $S.OllamaStatus = "skipped (-NoOllama or CUTPOINT_OLLAMA=0)"
            return
        }
        if ($S.OllamaChoice -ne "1" -and -not (Ask "Install Ollama (local AI models for vision rating and the AI assistant)?")) {
            $S.OllamaStatus = "declined"
            return
        }
        if ($S.OllamaChoice -eq "1") {
            Say "Installing Ollama (-Ollama or CUTPOINT_OLLAMA=1)."
        }
        if (-not (Install-OllamaPackage)) { return }
        $S.OllamaStatus = "installed"
        if ($S.DryRun) {
            Say "+ wait up to 20 s for $OLLAMA_URL/api/version"
            $null = Start-OllamaServe
            $S.OllamaUp = $true
            return
        }
        Update-SessionPath
        if (-not (Find-Ollama)) {
            Warn "ollama is not on PATH after the install; open a new terminal."
            $S.OllamaStatus = "installed (not on PATH yet; open a new terminal)"
            return
        }
        if ((Wait-Ollama) -or (Start-OllamaServe)) {
            $S.OllamaUp = $true
        } else {
            Warn "Ollama is installed but the server did not start; run: ollama serve"
            $S.OllamaStatus = "installed (server not running; start it with: ollama serve)"
        }
    }

    function Invoke-OptionalDownloads {
        $S.SetupDone = "no"
        $S.OllamaDone = "no"
        $S.OllamaStatus = "not checked"
        $S.OllamaUp = $false
        # Lite has no Ollama (the AI assistant can use a third-party provider); setup is full-only too.
        if ($S.Lite) {
            $S.OllamaStatus = "not part of lite"
            if ($S.OllamaChoice -eq "1") {
                Warn "Ollama is not part of lite, so -Ollama / CUTPOINT_OLLAMA=1 is ignored (the AI assistant can use a third-party provider)."
            }
            return
        }
        Invoke-OllamaOffer
        if (Ask "Download models now with 'cutpoint setup' (about 8 GB)?") {
            if (Invoke-Run @("cutpoint", "setup")) {
                $S.SetupDone = "yes"
            } else {
                Warn "cutpoint setup failed; run it again later."
                $S.SetupDone = "failed"
            }
        }
        if ($S.OllamaUp) {
            if (Ask "Pull $OLLAMA_MODEL with Ollama (about 6 GB)?") {
                if (Invoke-Run @("ollama", "pull", $OLLAMA_MODEL)) {
                    $S.OllamaDone = "yes"
                } else {
                    Warn "ollama pull failed; run it again later."
                    $S.OllamaDone = "failed"
                }
            }
        }
    }

    function Write-Summary {
        Say ""
        if ($S.DryRun) {
            Say "Dry run: nothing was changed."
            return
        }
        if ($S.Lite) { $mode = "lite (no torch, no events/GPU extras)" } else { $mode = "full" }
        $tools = "?"
        try { $t = [string](& uv tool dir 2>$null | Select-Object -First 1); if ($t) { $tools = $t } } catch { }
        Say "Cutpoint is installed."
        Say "  Mode:      $mode"
        Say "  Torch:     $($S.Backend) ($($S.BackendWhy))"
        Say "  Version:   $($S.InstalledVersion)"
        Say "  Source:    $($S.Source)"
        Say "  Installed: $tools\cutpoint (command in $($S.BinDir))"
        Say "  Ollama:    $($S.OllamaStatus)"
        Say ""
        Say "Next steps:"
        if (-not $S.BinOnPath) {
            Say "  - Open a new terminal so cutpoint is on your PATH (or: `$env:PATH = `"$($S.BinDir);`$env:PATH`")."
        }
        if ($S.FfmpegMissing) {
            Say "  - Install ffmpeg and ffprobe (not found): winget install --id $FFMPEG_ID -e"
        }
        if ($S.OllamaHint) {
            Say "  - $($S.OllamaHint)"
        }
        if ($S.OllamaUp -and $S.OllamaDone -ne "yes") {
            if ($S.OllamaDone -eq "failed") {
                Say "  - The model pull failed; run it again (about 6 GB): ollama pull $OLLAMA_MODEL"
            } else {
                Say "  - Pull the vision model (about 6 GB): ollama pull $OLLAMA_MODEL"
            }
        }
        if ($S.OllamaDone -eq "yes") {
            Say "  - Ollama model pulled: $OLLAMA_MODEL (used by --vision and AI Chat)."
        }
        if ((-not $S.Lite) -and $S.SetupDone -ne "yes") {
            Say "  - Download the models (about 8 GB): cutpoint setup"
        }
        Say "  - Start the browser workspace: cutpoint ui"
        Say "  - Edit a video: cutpoint edit PATH-TO-VIDEO"
        Say "  - Instructions for AI agents: cutpoint guide"
        if ($S.Lite) {
            Say "  - Lite has no torch: for every feature run this installer again without -Lite."
        }
        Say '  - Upgrade or switch mode: run this installer again (not "uv tool upgrade", it drops the torch build).'
    }

    # ------------------------------------------------------------ main

    try {
        if ($Help) {
            [Console]::Out.WriteLine((Get-Usage))
            return 0
        }
        Read-Arguments
        Test-Platform
        Initialize-Path
        Select-Backend
        Resolve-Source

        if ($S.Lite) {
            Say "Cutpoint installer: lite mode, source $($S.Source)"
        } else {
            Say "Cutpoint installer: full mode, torch $($S.Backend) ($($S.BackendWhy)), source $($S.Source)"
        }
        if ($S.DryRun) { Say "Dry run: printing commands, changing nothing." }

        Install-Ffmpeg
        Confirm-Uv
        Sync-BinDir
        Install-Cutpoint
        Update-Shell
        Test-Install
        Invoke-OptionalDownloads
        Write-Summary
        return 0
    } catch {
        $m = "$($_.Exception.Message)"
        if ($m -match '^CPEXIT:(\d+)$') { return [int]$Matches[1] }
        [Console]::Error.WriteLine("ERROR: $m")
        return 1
    }
}

# Everything is defined above; this is what runs, so a truncated download does nothing.
$cutpointExit = Invoke-CutpointInstaller -Lite ([bool]$Lite) -Yes ([bool]$Yes) -DryRun ([bool]$DryRun) `
    -Cpu ([bool]$Cpu) -Cuda ([bool]$Cuda) -Source $Source -NoFfmpeg ([bool]$NoFfmpeg) `
    -Ollama ([bool]$Ollama) -NoOllama ([bool]$NoOllama) -Help ([bool]$Help) -Extra $args
Remove-Item -Path Function:\Invoke-CutpointInstaller -ErrorAction SilentlyContinue
if ($PSCommandPath) {
    # Run as a file: report the exit code. (Under `irm | iex` there is no file and exit would close the terminal.)
    exit $cutpointExit
}
# Under `irm | iex` the script-level parameters and $cutpointExit live in the caller's scope: remove them.
$global:LASTEXITCODE = [int]$cutpointExit
Remove-Variable -Name Lite, Yes, DryRun, Cpu, Cuda, Source, NoFfmpeg, Ollama, NoOllama, Help, cutpointExit -ErrorAction SilentlyContinue
if ($global:LASTEXITCODE -ne 0) { throw "Cutpoint installer failed (exit $global:LASTEXITCODE)." }
