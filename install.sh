#!/bin/sh
# Cutpoint installer for Linux (and macOS, CPU only).
#
#   curl -LsSf https://raw.githubusercontent.com/FerventOrange/cutpoint/main/install.sh | sh
#   curl -LsSf https://raw.githubusercontent.com/FerventOrange/cutpoint/main/install.sh | sh -s -- --lite
#
# Installs uv and ffmpeg when they are missing, then installs Cutpoint as a uv tool
# with the right torch build for this machine. Re-running it upgrades or switches mode.
# Run `sh install.sh --help` for the options. POSIX sh; checked with shellcheck, sh -n, dash -n.
#
# Release stamping: the release process sets the line below to the version just published on
# PyPI (for example: sed -i 's/^CUTPOINT_VERSION_DEFAULT=.*/CUTPOINT_VERSION_DEFAULT="0.4.0"/'
# install.sh) and attaches/serves the script from that tag. The env var CUTPOINT_VERSION
# overrides it at run time.
CUTPOINT_VERSION_DEFAULT="0.4.0"

# uv is installed from this pinned installer when missing (keep in step with the Dockerfile pin).
UV_VERSION="0.13.5"
# Oldest uv that has `uv tool install --torch-backend`.
UV_MIN_MAJOR=0
UV_MIN_MINOR=13

GIT_URL="https://github.com/FerventOrange/cutpoint.git"
PYTHON_VERSION="3.12"
CUDA_BACKEND="cu126"
OLLAMA_MODEL="qwen3-vl:8b"
OLLAMA_URL="http://127.0.0.1:11434"

usage() {
  cat <<'EOF'
Cutpoint installer

Usage: install.sh [options]

Installs uv and ffmpeg when missing, then Cutpoint as a uv tool (run again to upgrade).

Modes:
  (default)         Full: every feature. With an NVIDIA GPU: CUDA torch and GPU libraries
                    ("cutpoint[all]", several GB). Without: CPU torch ("cutpoint[events,ui,keyring,ski]").
  --lite            Lite: no torch, no events/GPU extras ("cutpoint[ui,keyring,ski]"). Analyze, clips,
                    assemble and edit work on the CPU. Lite has no Ollama: it does not offer, install
                    or start it (the AI assistant can use a third-party provider instead).

Options:
  --yes, -y         Answer yes to the optional steps (install Ollama, cutpoint setup, Ollama model).
  --ollama          Install Ollama when it is missing, without asking. Full mode only: with --lite
                    it prints a warning and is ignored.
  --no-ollama       Never offer to install Ollama (full mode; lite never offers it).
  --dry-run         Print every command that would run and change nothing.
  --cpu             Force CPU torch even when an NVIDIA GPU is found.
  --cuda            Force CUDA torch (cu126) even when no NVIDIA GPU is found.
  --source SRC      pypi (default; the pinned release), git or git@REF (branch, tag or commit
                    of the GitHub repository), or a local path (checkout, wheel or sdist).
  --no-ffmpeg       Do not install ffmpeg.
  --help, -h        Show this help.

Environment (a flag wins over its variable; any value except empty or 0 counts as on):
  CUTPOINT_LITE=1          same as --lite
  CUTPOINT_YES=1           same as --yes
  CUTPOINT_DRY_RUN=1       same as --dry-run
  CUTPOINT_NO_FFMPEG=1     same as --no-ffmpeg
  CUTPOINT_OLLAMA=1|0      same as --ollama / --no-ollama
  CUTPOINT_TORCH=cpu|cuda  same as --cpu / --cuda
  CUTPOINT_SOURCE=SRC      same as --source
  CUTPOINT_VERSION=X.Y.Z   release to install for --source pypi (default: the version in this script)
  CUTPOINT_NO_MODIFY_PATH=1  do not run `uv tool update-shell`
  UV_TOOL_DIR, UV_TOOL_BIN_DIR, XDG_BIN_HOME   where uv puts tools and their commands
                           (default: ~/.local/share/uv/tools and ~/.local/bin)

Optional prompts (default No; they read from /dev/tty, so they also work under `curl | sh`;
with no terminal the answer is No unless --yes is given):
  - Install Ollama (full mode only; lite skips this step), when it is not found. Ollama runs the
    local AI models for --vision rating and the AI assistant. Linux: Ollama's official script
    (curl -fsSL https://ollama.com/install.sh | sh), which needs sudo (without a terminal only
    when `sudo -n` works or you are root). macOS: `brew install ollama`, or the download link
    when Homebrew is missing. Afterwards it waits for the server: macOS runs
    `brew services start ollama`; Linux relies on the ollama.service the official script sets up.
    Only when no service exists does it start `ollama serve` in the background (log:
    ~/.local/state/cutpoint/ollama-serve.log). A declined or failed
    Ollama install never fails the Cutpoint install.
  - Pull qwen3-vl:8b (about 6 GB), full mode only, when Ollama is running.
  - "cutpoint setup" (model weights, about 8 GB), full mode only.
EOF
}

# ---------------------------------------------------------------- output helpers

say() { printf '%s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

have() { command -v "$1" >/dev/null 2>&1; }

is_on() {
  case "${1:-}" in
    '' | 0 | no | NO | false | FALSE) return 1 ;;
    *) return 0 ;;
  esac
}

# Print one argument so it can be pasted into a shell.
quote_arg() {
  case "$1" in
    '' | *[!A-Za-z0-9_./:=@%+,-]*)
      case "$1" in
        *'"'* | *'$'* | *'`'* | *\\* | *'!'*)
          printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
          ;;
        *) printf '"%s"' "$1" ;;
      esac
      ;;
    *) printf '%s' "$1" ;;
  esac
}

show_cmd() {
  out=""
  for a in "$@"; do
    out="$out $(quote_arg "$a")"
  done
  say "+${out}"
}

# Print the command; run it unless --dry-run.
run() {
  show_cmd "$@"
  if [ "$DRY_RUN" = 1 ]; then
    return 0
  fi
  "$@"
}

# True when a terminal can be opened for prompts (also under `curl | sh`).
has_tty() { (: </dev/tty) 2>/dev/null; }

# ask "question": prints the question and returns 0 for yes. Default No.
ask() {
  if [ "$ASSUME_YES" = 1 ]; then
    say "$1 yes (--yes)"
    return 0
  fi
  if [ "$DRY_RUN" = 1 ]; then
    say "$1 no (dry run does not prompt; default No)"
    return 1
  fi
  if ! has_tty; then
    say "$1 no (no terminal to ask; use --yes to accept)"
    return 1
  fi
  printf '%s [y/N] ' "$1" >&2
  reply=""
  read -r reply </dev/tty || reply=""
  case "$reply" in
    y | Y | yes | YES | Yes) return 0 ;;
    *) return 1 ;;
  esac
}

# ---------------------------------------------------------------- arguments

# Wrong command line: usage and the message on stderr, exit 2.
bad_usage() {
  usage >&2
  printf 'ERROR: %s\n' "$1" >&2
  exit 2
}

parse_args() {
  LITE=0
  ASSUME_YES=0
  DRY_RUN=0
  NO_FFMPEG=0
  TORCH=""
  FLAG_TORCH=""
  OLLAMA_CHOICE=""
  FLAG_OLLAMA=""
  LITE_FLAG=0
  SOURCE=""
  is_on "${CUTPOINT_LITE:-}" && LITE=1
  is_on "${CUTPOINT_YES:-}" && ASSUME_YES=1
  is_on "${CUTPOINT_DRY_RUN:-}" && DRY_RUN=1
  is_on "${CUTPOINT_NO_FFMPEG:-}" && NO_FFMPEG=1
  SOURCE=${CUTPOINT_SOURCE:-}
  if [ -n "${CUTPOINT_OLLAMA:-}" ]; then
    if is_on "$CUTPOINT_OLLAMA"; then OLLAMA_CHOICE=1; else OLLAMA_CHOICE=0; fi
  fi
  while [ $# -gt 0 ]; do
    case "$1" in
      --lite)
        LITE=1
        LITE_FLAG=1
        ;;
      --yes | -y) ASSUME_YES=1 ;;
      --dry-run) DRY_RUN=1 ;;
      --no-ffmpeg) NO_FFMPEG=1 ;;
      --ollama | --no-ollama)
        case "$1" in
          --ollama) want=1 ;;
          *) want=0 ;;
        esac
        if [ -n "$FLAG_OLLAMA" ] && [ "$FLAG_OLLAMA" != "$want" ]; then
          bad_usage "--ollama and --no-ollama contradict each other"
        fi
        FLAG_OLLAMA=$want
        OLLAMA_CHOICE=$want
        ;;
      --cpu | --cuda)
        want=${1#--}
        if [ -n "$FLAG_TORCH" ] && [ "$FLAG_TORCH" != "$want" ]; then
          bad_usage "--cpu and --cuda contradict each other"
        fi
        FLAG_TORCH=$want
        TORCH=$want
        ;;
      --source)
        case "${2:-}" in
          '' | -*) bad_usage "missing value for --source (pypi, git[@ref] or a path)" ;;
        esac
        SOURCE=$2
        shift
        ;;
      --source=*) SOURCE=${1#--source=} ;;
      --help | -h)
        usage
        exit 0
        ;;
      *)
        bad_usage "unknown option: $1"
        ;;
    esac
    shift
  done
  if [ "$LITE_FLAG" = 1 ] && [ -n "$FLAG_TORCH" ]; then
    bad_usage "--lite has no torch, so it cannot be combined with --$FLAG_TORCH"
  fi
  if [ -z "$TORCH" ]; then
    case "${CUTPOINT_TORCH:-}" in
      '') ;;
      cpu | cuda) TORCH=${CUTPOINT_TORCH} ;;
      *) die "CUTPOINT_TORCH must be cpu or cuda, got: $CUTPOINT_TORCH" ;;
    esac
  fi
  [ -n "$SOURCE" ] || SOURCE=pypi
  CUTPOINT_VERSION=${CUTPOINT_VERSION:-$CUTPOINT_VERSION_DEFAULT}
}

# ---------------------------------------------------------------- platform

detect_os() {
  OS=$(uname -s 2>/dev/null || echo unknown)
  ARCH=$(uname -m 2>/dev/null || echo unknown)
  case "$OS" in
    Linux | Darwin) ;;
    *) die "This installer supports Linux and macOS. On Windows use install.ps1." ;;
  esac
}

nvidia_found() {
  [ "$OS" = Linux ] || return 1
  for smi in nvidia-smi /usr/lib/wsl/lib/nvidia-smi; do
    if have "$smi" || [ -x "$smi" ]; then
      if "$smi" -L 2>/dev/null | grep -q '^GPU [0-9]'; then
        return 0
      fi
    fi
  done
  [ -r /proc/driver/nvidia/version ]
}

choose_backend() {
  BACKEND=none
  BACKEND_WHY=""
  if [ "$LITE" = 1 ]; then
    if [ -n "$TORCH" ]; then
      if [ -n "$FLAG_TORCH" ]; then
        warn "--$FLAG_TORCH is ignored in lite mode (lite has no torch)."
      else
        warn "CUTPOINT_TORCH is ignored in lite mode (lite has no torch)."
      fi
    fi
    BACKEND_WHY="lite mode"
    return 0
  fi
  case "$TORCH" in
    cpu)
      BACKEND=cpu
      BACKEND_WHY="--cpu"
      ;;
    cuda)
      [ "$OS" = Linux ] || die "CUDA torch is not available on $OS."
      BACKEND=$CUDA_BACKEND
      BACKEND_WHY="--cuda"
      ;;
    *)
      if nvidia_found; then
        case "$ARCH" in
          x86_64 | amd64)
            BACKEND=$CUDA_BACKEND
            BACKEND_WHY="NVIDIA GPU found"
            ;;
          *)
            BACKEND=cpu
            BACKEND_WHY="NVIDIA GPU found, but $ARCH has no $CUDA_BACKEND wheels (use --cuda to try)"
            ;;
        esac
      else
        BACKEND=cpu
        BACKEND_WHY="no NVIDIA GPU found"
      fi
      ;;
  esac
}

# ---------------------------------------------------------------- bin dir and PATH

# Drop "name/.." pairs textually so string comparisons with PATH entries work (the directories
# need not exist yet, so no cd).
norm_dir() {
  nd=$1
  while :; do
    nd_next=$(printf '%s' "$nd" | sed 's|/[^/][^/]*/\.\./|/|; s|/[^/][^/]*/\.\.$||')
    [ "$nd_next" != "$nd" ] || break
    nd=$nd_next
  done
  printf '%s' "$nd"
}

# Same order as uv: UV_TOOL_BIN_DIR, XDG_BIN_HOME, $XDG_DATA_HOME/../bin, ~/.local/bin.
default_bin_dir() {
  if [ -n "${UV_TOOL_BIN_DIR:-}" ]; then
    BIN_DIR=$UV_TOOL_BIN_DIR
  elif [ -n "${XDG_BIN_HOME:-}" ]; then
    BIN_DIR=$XDG_BIN_HOME
  elif [ -n "${XDG_DATA_HOME:-}" ]; then
    BIN_DIR=$XDG_DATA_HOME/../bin
  else
    [ -n "${HOME:-}" ] || die "HOME is not set."
    BIN_DIR=$HOME/.local/bin
  fi
  BIN_DIR=$(norm_dir "$BIN_DIR")
}

# Put BIN_DIR first on PATH; BIN_ON_PATH records whether the user's own PATH had it.
use_bin_dir() {
  case ":$ORIG_PATH:" in
    *":$BIN_DIR:"*) BIN_ON_PATH=1 ;;
    *) BIN_ON_PATH=0 ;;
  esac
  case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
      PATH="$BIN_DIR:$PATH"
      export PATH
      ;;
  esac
}

setup_path() {
  ORIG_PATH=$PATH
  default_bin_dir
  use_bin_dir
}

# Once uv is there, take the directory from uv itself.
sync_bin_dir() {
  [ "$DRY_RUN" = 1 ] && return 0
  uvbin=$(uv tool dir --bin 2>/dev/null) || return 0
  [ -z "$uvbin" ] || uvbin=$(norm_dir "$uvbin")
  if [ -n "$uvbin" ] && [ "$uvbin" != "$BIN_DIR" ]; then
    BIN_DIR=$uvbin
    use_bin_dir
  fi
}

# ---------------------------------------------------------------- ffmpeg

# Run a command as root: directly, through sudo (sudo -n without a terminal), or just print it.
priv_run() {
  if [ "$(id -u)" = 0 ]; then
    run "$@"
  elif have sudo; then
    if has_tty; then
      run sudo "$@"
    else
      run sudo -n "$@"
    fi
  else
    show_cmd "$@"
    warn "sudo is not available: run the command above as root."
    return 1
  fi
}

install_ffmpeg() {
  if have ffmpeg && have ffprobe; then
    say "ffmpeg: found ($(command -v ffmpeg))"
    return 0
  fi
  if [ "$NO_FFMPEG" = 1 ]; then
    say "ffmpeg: not found, skipped (--no-ffmpeg). Install ffmpeg and ffprobe before using Cutpoint."
    FFMPEG_MISSING=1
    return 0
  fi
  FFMPEG_MISSING=1
  say "ffmpeg: not found. Installing it with your package manager (sudo may ask for your password)."
  if [ "$OS" = Darwin ]; then
    if have brew; then
      run brew install ffmpeg || warn "brew install ffmpeg failed; install ffmpeg yourself."
    else
      warn "Install ffmpeg (for example with Homebrew: brew install ffmpeg)."
    fi
  elif have apt-get; then
    if ! { priv_run apt-get update && priv_run apt-get install -y ffmpeg; }; then
      warn "Could not install ffmpeg; run: sudo apt-get install ffmpeg"
    fi
  elif have dnf; then
    priv_run dnf install -y ffmpeg ||
      warn "Could not install ffmpeg; Fedora needs RPM Fusion for it (https://rpmfusion.org)."
  elif have pacman; then
    priv_run pacman -S --needed --noconfirm ffmpeg ||
      warn "Could not install ffmpeg; run: sudo pacman -S ffmpeg"
  elif have zypper; then
    priv_run zypper --non-interactive install ffmpeg ||
      warn "Could not install ffmpeg; openSUSE Leap may need the Packman repository."
  else
    warn "No supported package manager (apt-get, dnf, pacman, zypper) found: install ffmpeg yourself."
  fi
  if [ "$DRY_RUN" != 1 ] && have ffmpeg && have ffprobe; then
    FFMPEG_MISSING=0
  fi
}

# ---------------------------------------------------------------- uv

uv_version_ok() {
  v=$(uv --version 2>/dev/null | awk '{print $2}')
  [ -n "$v" ] || return 1
  major=${v%%.*}
  rest=${v#*.}
  minor=${rest%%.*}
  major=${major%%[!0-9]*}
  minor=${minor%%[!0-9]*}
  [ -n "$major" ] && [ -n "$minor" ] || return 1
  if [ "$major" -gt "$UV_MIN_MAJOR" ]; then
    return 0
  fi
  [ "$major" -eq "$UV_MIN_MAJOR" ] && [ "$minor" -ge "$UV_MIN_MINOR" ]
}

install_uv() {
  url="https://astral.sh/uv/$UV_VERSION/install.sh"
  if [ "$DRY_RUN" = 1 ]; then
    say "+ curl -LsSf $url | UV_INSTALL_DIR=$BIN_DIR UV_NO_MODIFY_PATH=1 sh"
    return 0
  fi
  tmp=$(mktemp "${TMPDIR:-/tmp}/uv-install.XXXXXX") || die "Could not create a temporary file."
  if have curl; then
    curl -LsSf "$url" -o "$tmp" || {
      rm -f "$tmp"
      die "Could not download $url"
    }
  elif have wget; then
    wget -qO "$tmp" "$url" || {
      rm -f "$tmp"
      die "Could not download $url"
    }
  else
    rm -f "$tmp"
    die "Neither curl nor wget found: install one of them (or uv from https://docs.astral.sh/uv/) and run again."
  fi
  say "+ sh <uv installer $UV_VERSION>"
  if ! UV_INSTALL_DIR="$BIN_DIR" UV_NO_MODIFY_PATH=1 sh "$tmp"; then
    rm -f "$tmp"
    die "The uv installer failed. Install uv yourself (https://docs.astral.sh/uv/) and run again."
  fi
  rm -f "$tmp"
  have uv || die "uv was installed but is not on PATH ($BIN_DIR). Open a new terminal and run again."
}

ensure_uv() {
  if have uv; then
    if uv_version_ok; then
      say "uv: found ($(uv --version 2>/dev/null))"
      return 0
    fi
    say "uv: $(uv --version 2>/dev/null) is older than $UV_MIN_MAJOR.$UV_MIN_MINOR (needed for --torch-backend)."
    if [ "$DRY_RUN" = 1 ]; then
      say "+ uv self update $UV_VERSION"
      return 0
    fi
    if run uv self update "$UV_VERSION"; then
      uv_version_ok && return 0
    fi
    die "Could not upgrade uv here (it was probably installed with pip, a package manager or Homebrew). Upgrade it that way to $UV_MIN_MAJOR.$UV_MIN_MINOR or newer, or remove it so this script can install its own, then run again."
  fi
  say "uv: not found. Installing uv $UV_VERSION."
  install_uv
}

# ---------------------------------------------------------------- ollama

OLLAMA_INSTALL_URL="https://ollama.com/install.sh"
OLLAMA_DOWNLOAD_URL="https://ollama.com/download"

ollama_up() {
  if have curl; then
    curl -fsS -m 2 "$OLLAMA_URL/api/version" >/dev/null 2>&1
  elif have wget; then
    wget -q -T 2 -O /dev/null "$OLLAMA_URL/api/version" 2>/dev/null
  else
    return 1
  fi
}

# Wait up to about 20 seconds for the server.
wait_ollama() {
  n=0
  while [ "$n" -lt 20 ]; do
    ollama_up && return 0
    sleep 1
    n=$((n + 1))
  done
  ollama_up
}

# True on Linux when Ollama's official script installed the systemd service (it then owns the server).
ollama_service() {
  [ "$OS" != Darwin ] && have systemctl && systemctl cat ollama.service >/dev/null 2>&1
}

# Start `ollama serve` in the background (macOS, or Linux without the service) and wait for it.
start_ollama_serve() {
  olog="${XDG_STATE_HOME:-$HOME/.local/state}/cutpoint/ollama-serve.log"
  if [ "$DRY_RUN" = 1 ]; then
    if [ "$OS" = Darwin ]; then
      say "+ nohup ollama serve > $olog 2>&1 &   (only when 'brew services start ollama' fails)"
    else
      say "+ nohup ollama serve > $olog 2>&1 &   (only when the server is not up after 20 s)"
    fi
    return 0
  fi
  mkdir -p "${olog%/*}" || return 1
  say "Ollama is not answering at $OLLAMA_URL: starting 'ollama serve' in the background (log: $olog)."
  nohup ollama serve >"$olog" 2>&1 </dev/null &
  wait_ollama
}

# Sets OLLAMA_WHY and returns 1 when Ollama cannot be installed here.
ollama_can_install() {
  OLLAMA_WHY=""
  if [ "$OS" = Darwin ]; then
    if ! have brew; then
      OLLAMA_WHY="Homebrew not found; download Ollama from $OLLAMA_DOWNLOAD_URL"
      return 1
    fi
    return 0
  fi
  if ! have curl; then
    OLLAMA_WHY="curl not found; install Ollama yourself: curl -fsSL $OLLAMA_INSTALL_URL | sh"
    return 1
  fi
  [ "$DRY_RUN" = 1 ] && return 0
  if [ "$(id -u)" != 0 ] && ! has_tty; then
    if ! { have sudo && sudo -n true 2>/dev/null; }; then
      OLLAMA_WHY="needs sudo and there is no terminal; run: curl -fsSL $OLLAMA_INSTALL_URL | sh"
      return 1
    fi
  fi
  return 0
}

# Run the install; returns 1 on failure.
ollama_install() {
  if [ "$OS" = Darwin ]; then
    run brew install ollama
    return
  fi
  if [ "$DRY_RUN" = 1 ]; then
    say "+ curl -fsSL $OLLAMA_INSTALL_URL | sh"
    return 0
  fi
  otmp=$(mktemp "${TMPDIR:-/tmp}/ollama-install.XXXXXX") || return 1
  say "+ curl -fsSL $OLLAMA_INSTALL_URL | sh   (sudo may ask for your password)"
  if curl -fsSL "$OLLAMA_INSTALL_URL" -o "$otmp" && sh "$otmp"; then
    rm -f "$otmp"
    return 0
  fi
  rm -f "$otmp"
  return 1
}

# Offer to install Ollama; sets OLLAMA_STATUS and OLLAMA_UP (1 when the server is, or in a dry
# run would be, up). Never fails the install.
offer_ollama() {
  OLLAMA_UP=0
  OLLAMA_HINT=""
  if have ollama; then
    OLLAMA_STATUS="already installed"
    if ollama_up; then
      OLLAMA_UP=1
    elif [ "$DRY_RUN" != 1 ]; then
      OLLAMA_HINT="Ollama is installed but not running; start it, then: ollama pull $OLLAMA_MODEL"
    fi
    return 0
  fi
  if ollama_up; then
    OLLAMA_STATUS="server running at $OLLAMA_URL (no ollama command; not installed)"
    return 0
  fi
  if [ "$OLLAMA_CHOICE" = 0 ]; then
    OLLAMA_STATUS="skipped (--no-ollama or CUTPOINT_OLLAMA=0)"
    return 0
  fi
  if [ "$OLLAMA_CHOICE" != 1 ] && ! ask "Install Ollama (local AI models for vision rating and the AI assistant)?"; then
    OLLAMA_STATUS="declined"
    return 0
  fi
  if ! ollama_can_install; then
    warn "Ollama not installed: $OLLAMA_WHY"
    OLLAMA_STATUS="skipped ($OLLAMA_WHY)"
    return 0
  fi
  if [ "$OLLAMA_CHOICE" = 1 ]; then
    say "Installing Ollama (--ollama or CUTPOINT_OLLAMA=1)."
  fi
  if ! ollama_install; then
    warn "The Ollama install failed; install it later from $OLLAMA_DOWNLOAD_URL"
    OLLAMA_STATUS="failed (install it from $OLLAMA_DOWNLOAD_URL)"
    return 0
  fi
  OLLAMA_STATUS="installed"
  if [ "$DRY_RUN" = 1 ]; then
    if [ "$OS" = Darwin ]; then
      say "+ brew services start ollama"
    else
      say "+ if systemctl cat ollama.service succeeds: only wait for the service (no second server)"
    fi
    say "+ wait up to 20 s for $OLLAMA_URL/api/version"
    start_ollama_serve
    OLLAMA_UP=1
    return 0
  fi
  if ! have ollama; then
    warn "ollama is not on PATH after the install; open a new terminal."
    OLLAMA_STATUS="installed (not on PATH yet; open a new terminal)"
    return 0
  fi
  if [ "$OS" = Darwin ]; then
    if run brew services start ollama; then
      if wait_ollama; then
        OLLAMA_UP=1
      else
        warn "The ollama service is not answering; check: brew services info ollama"
        OLLAMA_STATUS="installed (service not answering: brew services info ollama)"
      fi
      return 0
    fi
  elif ollama_service; then
    if wait_ollama; then
      OLLAMA_UP=1
    else
      warn "The ollama service is not answering; check: systemctl status ollama"
      OLLAMA_STATUS="installed (service not answering: systemctl status ollama)"
    fi
    return 0
  elif wait_ollama; then
    OLLAMA_UP=1
    return 0
  fi
  if start_ollama_serve && ollama_up; then
    OLLAMA_UP=1
  else
    warn "Ollama is installed but the server did not start; run: ollama serve"
    OLLAMA_STATUS="installed (server not running; start it with: ollama serve)"
  fi
}

# ---------------------------------------------------------------- cutpoint

# Sets REQ (the requirement) and NO_SOURCES.
resolve_source() {
  # ui and ski are extras before the packaging change and empty aliases after it, so spelling
  # them out is correct on both layouts. keyring is an extra only in releases after 0.3.0
  # (0.3.0 has none: uv warns and skips it), so stamp a release that has it. Drop these extras
  # once the minimum supported release is past the packaging change. ("all" includes them.)
  if [ "$BACKEND" = none ]; then
    extra="[ui,keyring,ski]"
  elif [ "$BACKEND" = cpu ]; then
    extra="[events,ui,keyring,ski]"
  else
    extra="[all]"
  fi
  NO_SOURCES=0
  case "$SOURCE" in
    pypi)
      REQ="cutpoint${extra}==$CUTPOINT_VERSION"
      ;;
    git | git@*)
      ref=${SOURCE#git}
      ref=${ref#@}
      REQ="cutpoint${extra} @ git+$GIT_URL${ref:+@$ref}"
      NO_SOURCES=1
      ;;
    *)
      if [ ! -e "$SOURCE" ]; then
        die "--source path does not exist: $SOURCE"
      fi
      dir=$(dirname -- "$SOURCE")
      base=$(basename -- "$SOURCE")
      if abs=$(CDPATH='' cd -- "$dir" 2>/dev/null && pwd); then
        case "$abs" in
          /) abs_path="/$base" ;;
          *) abs_path="$abs/$base" ;;
        esac
      else
        abs_path=$SOURCE
      fi
      [ "$base" != . ] || abs_path=$abs
      enc=$(printf '%s' "$abs_path" | sed 's/%/%25/g; s/ /%20/g; s/#/%23/g; s/?/%3F/g')
      REQ="cutpoint${extra} @ file://$enc"
      # The checkout's own [tool.uv.sources] pins torch to the cu126 index and would beat
      # --torch-backend; a PyPI wheel has no such table, so ignore it to match a release.
      NO_SOURCES=1
      ;;
  esac
}

install_cutpoint() {
  set -- uv tool install "$REQ"
  if [ "$BACKEND" != none ]; then
    set -- "$@" --torch-backend "$BACKEND"
  fi
  set -- "$@" --python "$PYTHON_VERSION" --reinstall
  if [ "$NO_SOURCES" = 1 ]; then
    set -- "$@" --no-sources
  fi
  if [ "$LITE" = 0 ]; then
    if [ "$BACKEND" = "$CUDA_BACKEND" ]; then
      say "Downloading Cutpoint with CUDA torch and GPU libraries (several GB); this can take a while."
    else
      say "Downloading Cutpoint with CPU torch (about 600 MB to download, about 2 GB on disk); this can take a while."
    fi
  fi
  run "$@" || die "uv tool install failed. Fix the error above and run again."
}

update_shell() {
  if is_on "${CUTPOINT_NO_MODIFY_PATH:-}"; then
    return 0
  fi
  if [ "$BIN_ON_PATH" = 1 ]; then
    return 0
  fi
  # uv skips the rc files when the bin dir is on PATH, and setup_path put it there: ask uv with
  # the PATH the user had.
  show_cmd uv tool update-shell
  if [ "$DRY_RUN" = 1 ]; then
    return 0
  fi
  # uv picks the rc files from $SHELL; some bare environments (containers, env -i) lack it.
  # It exits 2 with "already up-to-date" on a re-run, which is fine.
  if shell_out=$(env PATH="$ORIG_PATH" SHELL="${SHELL:-/bin/bash}" "$(command -v uv)" tool update-shell 2>&1); then
    say "$shell_out"
    return 0
  fi
  case "$shell_out" in
    *"already up-to-date"*) say "Shell configuration is already up to date." ;;
    *)
      printf '%s\n' "$shell_out" >&2
      warn "uv tool update-shell failed; add $BIN_DIR to PATH yourself."
      ;;
  esac
}

verify() {
  if [ "$DRY_RUN" = 1 ]; then
    say "+ cutpoint --version"
    say "+ cutpoint capabilities --json"
    return 0
  fi
  have cutpoint || die "cutpoint is not on PATH after the install ($BIN_DIR). Open a new terminal and run: cutpoint --version"
  show_cmd cutpoint --version
  INSTALLED_VERSION=$(cutpoint --version 2>&1) || die "cutpoint --version failed: $INSTALLED_VERSION"
  say "$INSTALLED_VERSION"
  show_cmd cutpoint capabilities --json
  caps=$(cutpoint capabilities --json 2>/dev/null) && CAPS_OK=1 || CAPS_OK=0
  case "$caps" in
    *'"ok": true'*) ;;
    *) CAPS_OK=0 ;;
  esac
  if [ "$CAPS_OK" = 1 ]; then
    say "capabilities: ok"
  else
    warn "cutpoint capabilities --json did not report ok; run 'cutpoint capabilities' to see why."
  fi
}

optional_downloads() {
  SETUP_DONE="no"
  OLLAMA_DONE="no"
  OLLAMA_STATUS="not checked"
  OLLAMA_UP=0
  # Lite has no Ollama (the AI assistant can use a third-party provider); setup is full-only too.
  if [ "$LITE" = 1 ]; then
    OLLAMA_STATUS="not part of lite"
    if [ "$OLLAMA_CHOICE" = 1 ]; then
      warn "Ollama is not part of lite, so --ollama / CUTPOINT_OLLAMA=1 is ignored (the AI assistant can use a third-party provider)."
    fi
  else
    offer_ollama
  fi
  if [ "$LITE" = 0 ]; then
    if ask "Download models now with 'cutpoint setup' (about 8 GB)?"; then
      if run cutpoint setup; then
        SETUP_DONE=yes
      else
        warn "cutpoint setup failed; run it again later."
        SETUP_DONE=failed
      fi
    fi
  fi
  if [ "$OLLAMA_UP" = 1 ]; then
    if ask "Pull $OLLAMA_MODEL with Ollama (about 6 GB)?"; then
      if run ollama pull "$OLLAMA_MODEL"; then
        OLLAMA_DONE=yes
      else
        warn "ollama pull failed; run it again later."
        OLLAMA_DONE=failed
      fi
    fi
  fi
}

summary() {
  say ""
  if [ "$DRY_RUN" = 1 ]; then
    say "Dry run: nothing was changed."
    return 0
  fi
  if [ "$LITE" = 1 ]; then
    mode="lite (no torch, no events/GPU extras)"
  else
    mode="full"
  fi
  tools=$(uv tool dir 2>/dev/null || echo "?")
  say "Cutpoint is installed."
  say "  Mode:      $mode"
  say "  Torch:     $BACKEND ($BACKEND_WHY)"
  say "  Version:   $INSTALLED_VERSION"
  say "  Source:    $SOURCE"
  say "  Installed: $tools/cutpoint (command in $BIN_DIR)"
  say "  Ollama:    $OLLAMA_STATUS"
  say ""
  say "Next steps:"
  if [ "$BIN_ON_PATH" = 0 ]; then
    say "  - Open a new terminal so cutpoint is on your PATH (or: export PATH=\"$BIN_DIR:\$PATH\")."
  fi
  if [ "${FFMPEG_MISSING:-0}" = 1 ]; then
    say "  - Install ffmpeg and ffprobe (not found)."
  fi
  if [ -n "${OLLAMA_HINT:-}" ]; then
    say "  - $OLLAMA_HINT"
  fi
  if [ "$OLLAMA_UP" = 1 ] && [ "$OLLAMA_DONE" != yes ]; then
    if [ "$OLLAMA_DONE" = failed ]; then
      say "  - The model pull failed; run it again (about 6 GB): ollama pull $OLLAMA_MODEL"
    else
      say "  - Pull the vision model (about 6 GB): ollama pull $OLLAMA_MODEL"
    fi
  fi
  if [ "$OLLAMA_DONE" = yes ]; then
    say "  - Ollama model pulled: $OLLAMA_MODEL (used by --vision and AI Chat)."
  fi
  if [ "$LITE" = 0 ] && [ "$SETUP_DONE" != yes ]; then
    say "  - Download the models (about 8 GB): cutpoint setup"
  fi
  say "  - Start the browser workspace: cutpoint ui"
  say "  - Edit a video: cutpoint edit PATH-TO-VIDEO"
  say "  - Instructions for AI agents: cutpoint guide"
  if [ "$LITE" = 1 ]; then
    say "  - Lite has no torch: for every feature run this installer again without --lite."
  fi
  say "  - Upgrade or switch mode: run this installer again."
}

main() {
  FFMPEG_MISSING=0
  INSTALLED_VERSION="?"
  parse_args "$@"
  detect_os
  setup_path
  choose_backend
  resolve_source

  if [ "$LITE" = 1 ]; then
    say "Cutpoint installer: lite mode, source $SOURCE"
  else
    say "Cutpoint installer: full mode, torch $BACKEND ($BACKEND_WHY), source $SOURCE"
  fi
  [ "$DRY_RUN" = 0 ] || say "Dry run: printing commands, changing nothing."

  install_ffmpeg
  ensure_uv
  sync_bin_dir
  install_cutpoint
  update_shell
  verify
  optional_downloads
  summary
}

# Everything is defined above; this last line is what runs, so a truncated download does nothing.
main "$@"
