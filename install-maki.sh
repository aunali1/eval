#!/usr/bin/env bash
# install-maki.sh — register the maki coding agent (https://github.com/tontinton/maki,
# pinned release v0.5.5) in Harbor 0.22.0 and datacurve-pier 0.3.1, so that
# `harbor run -a maki` and `pier run --agent maki` parse and resolve.
#
# -----------------------------------------------------------------------------------
# CLI-registration verification performed against the unpacked wheels
# (ground truth: /tmp/harbor-pkg/h and /tmp/pier-pkg/pier):
#
#   Harbor (h/harbor/cli/jobs.py + h/harbor/agents/factory.py):
#     * AgentName is `class AgentName(str, Enum)` in h/harbor/models/agent/name.py.
#     * CLI runs() `-a/--agent` is typed `str | None` (jobs.py:541), metavar lists
#       AgentName.values(); the string is validated downstream:
#       factory.py `create_agent_from_config` checks `name in AgentName.values()` and
#       then does `AgentName(name)`, resolving via `_AGENT_MAP[AgentName.MAKI]`.
#     * => appending the `MAKI = "maki"` member + the _AGENT_MAP row makes
#       `-a maki` parse and resolve.
#
#   Pier (pier/pier/cli/jobs.py):
#     * AgentName is `class AgentName(str, Enum)` in pier/pier/models/agent/name.py.
#     * CLI runs() `--agent` is typed `AgentName | None` (jobs.py:306-314), so Typer
#       validates the raw argument directly against the (patched) enum; conversion to
#       enum happens at parse time, and factory.py `AgentName("maki")` must succeed.
#     * factory.py keeps `_AGENTS: list[type[BaseAgent]]` (not a string map) and
#       builds `_AGENT_MAP = {AgentName(agent.name()): agent for agent in _AGENTS}`.
#     * => patching the enum AND importing Maki + appending it to _AGENTS makes
#       `--agent maki` parse and resolve.
# -----------------------------------------------------------------------------------
#
# All patches are idempotent and refuse to patch twice.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAKI_AGENT_DIR="$SCRIPT_DIR/maki-agent"
HARBOR_MAKI_SRC="$MAKI_AGENT_DIR/harbor_maki.py"
PIER_MAKI_SRC="$MAKI_AGENT_DIR/pier_maki.py"
MAKI_RELEASE_TAG="v0.5.5"

log() { printf '[install-maki] %s\n' "$*"; }
fail() { printf '[install-maki] ERROR: %s\n' "$*" >&2; exit 1; }

[ -f "$HARBOR_MAKI_SRC" ] || fail "missing $HARBOR_MAKI_SRC"
[ -f "$PIER_MAKI_SRC" ] || fail "missing $PIER_MAKI_SRC"

# -----------------------------------------------------------------------------
# Python environment discovery
# -----------------------------------------------------------------------------
# Prints a (possibly multi-word) interpreter command that can `import $1`.
find_pkg_python() {
    local pkg="$1" py bin vbin shebang_py
    for py in python3 python; do
        if command -v "$py" >/dev/null 2>&1 && "$py" -c "import $pkg" >/dev/null 2>&1; then
            echo "$py"
            return 0
        fi
    done
    # Console script next to the interpreter that hosts the package.
    bin="$(command -v "$pkg" 2>/dev/null || true)"
    if [ -n "$bin" ]; then
        vbin="$(dirname "$(readlink -f "$bin")")"
        for py in "$vbin/python3" "$vbin/python"; do
            if [ -x "$py" ] && "$py" -c "import $pkg" >/dev/null 2>&1; then
                echo "$py"
                return 0
            fi
        done
        # Shebang interpreter of the console script.
        shebang_py="$(head -n1 "$vbin/$pkg" 2>/dev/null | sed -n 's/^#! *//p' | awk '{print $1}')"
        if [ -n "$shebang_py" ] && [ -x "$shebang_py" ] \
            && "$shebang_py" -c "import $pkg" >/dev/null 2>&1; then
            echo "$shebang_py"
            return 0
        fi
    fi
    # uv-based stacks (harbor is installed via uv): try project venv without syncing.
    if command -v uv >/dev/null 2>&1; then
        if uv run --no-sync python3 -c "import $pkg" >/dev/null 2>&1; then
            echo "uv run --no-sync python3"
            return 0
        fi
    fi
    return 1
}

pkg_site_packages() {
    local py="$1" pkg="$2"
    # shellcheck disable=SC2086
    $py - "$pkg" <<'PYEOF'
import os, sys
pkg = sys.argv[1]
mod = __import__(pkg)
print(os.path.dirname(os.path.dirname(mod.__file__)))
PYEOF
}

# -----------------------------------------------------------------------------
# 1. Harbor registration
# -----------------------------------------------------------------------------
log "locating Harbor python environment..."
HARBOR_PY="$(find_pkg_python harbor)" || fail "could not find a python with 'harbor' importable"
HARBOR_SP="$(pkg_site_packages "$HARBOR_PY" harbor)" || fail "could not resolve harbor site-packages"
[ -d "$HARBOR_SP/harbor/agents/installed" ] || fail "$HARBOR_SP does not look like harbor site-packages"
log "Harbor interpreter: $HARBOR_PY"
log "Harbor site-packages: $HARBOR_SP"

cp "$HARBOR_MAKI_SRC" "$HARBOR_SP/harbor/agents/installed/maki.py"
log "installed harbor/agents/installed/maki.py"

# shellcheck disable=SC2086
$HARBOR_PY - "$HARBOR_SP" <<'PYEOF'
import sys
from pathlib import Path

sp = Path(sys.argv[1])

# --- harbor/models/agent/name.py: append `MAKI = "maki"` before classmethods ---
name_py = sp / "harbor/models/agent/name.py"
src = name_py.read_text()
if "MAKI" in src:
    print("[install-maki] harbor AgentName already contains MAKI; skipping enum patch")
else:
    marker = "\n    @classmethod"
    idx = src.find(marker)
    if idx == -1:
        sys.exit("could not find '@classmethod' anchor in harbor/models/agent/name.py")
    src = src[:idx] + '\n    MAKI = "maki"\n' + src[idx:]
    name_py.write_text(src)
    print("[install-maki] patched harbor AgentName: MAKI = 'maki'")

# --- harbor/agents/factory.py: add _AGENT_MAP row ---
factory_py = sp / "harbor/agents/factory.py"
src = factory_py.read_text()
if "harbor.agents.installed.maki" in src:
    print("[install-maki] harbor _AGENT_MAP already contains maki; skipping factory patch")
else:
    anchor = 'AgentName.CLAUDE_CODE: "harbor.agents.installed.claude_code:ClaudeCode",'
    if anchor not in src:
        sys.exit("could not find CLAUDE_CODE anchor row in harbor/agents/factory.py")
    row = '        AgentName.MAKI: "harbor.agents.installed.maki:Maki",\n'
    src = src.replace(anchor, anchor + "\n" + row, 1)
    factory_py.write_text(src)
    print("[install-maki] patched harbor AgentFactory._AGENT_MAP")
PYEOF

# shellcheck disable=SC2086
$HARBOR_PY - <<'PYEOF'
from harbor.models.agent.name import AgentName
assert AgentName("maki").value == "maki", "harbor AgentName('maki') failed"
from harbor.agents.factory import AgentFactory
cls = AgentFactory.get_agent_class(AgentName.MAKI)
assert cls.name() == "maki", "harbor Maki.name() != 'maki'"
print("[install-maki] Harbor registration verified: -a maki resolves to harbor.agents.installed.maki:Maki")
PYEOF

# -----------------------------------------------------------------------------
# 2. Pier registration
# -----------------------------------------------------------------------------
log "locating pier python environment..."
PIER_PY="$(find_pkg_python pier)" || fail "could not find a python with 'pier' importable"
PIER_SP="$(pkg_site_packages "$PIER_PY" pier)" || fail "could not resolve pier site-packages"
[ -d "$PIER_SP/pier/agents/installed" ] || fail "$PIER_SP does not look like pier site-packages"
log "pier interpreter: $PIER_PY"
log "pier site-packages: $PIER_SP"

cp "$PIER_MAKI_SRC" "$PIER_SP/pier/agents/installed/maki.py"
log "installed pier/agents/installed/maki.py"

# shellcheck disable=SC2086
$PIER_PY - "$PIER_SP" <<'PYEOF'
import sys
from pathlib import Path

sp = Path(sys.argv[1])

# --- pier/models/agent/name.py: append `MAKI = "maki"` before classmethods ---
name_py = sp / "pier/models/agent/name.py"
src = name_py.read_text()
if "MAKI" in src:
    print("[install-maki] pier AgentName already contains MAKI; skipping enum patch")
else:
    marker = "\n    @classmethod"
    idx = src.find(marker)
    if idx == -1:
        sys.exit("could not find '@classmethod' anchor in pier/models/agent/name.py")
    src = src[:idx] + '\n    MAKI = "maki"\n' + src[idx:]
    name_py.write_text(src)
    print("[install-maki] patched pier AgentName: MAKI = 'maki'")

# --- pier/agents/factory.py: import Maki + append to _AGENTS list ---
factory_py = sp / "pier/agents/factory.py"
src = factory_py.read_text()
if "pier.agents.installed.maki" in src:
    print("[install-maki] pier factory already imports maki; skipping factory patch")
else:
    import_anchor = "from pier.agents.installed.mini_swe_agent import MiniSweAgent\n"
    if import_anchor not in src:
        sys.exit("could not find MiniSweAgent import anchor in pier/agents/factory.py")
    src = src.replace(
        import_anchor, import_anchor + "from pier.agents.installed.maki import Maki\n", 1
    )
    list_anchor = "        MiniSweAgent,\n"
    if list_anchor not in src:
        sys.exit("could not find MiniSweAgent _AGENTS list anchor in pier/agents/factory.py")
    src = src.replace(list_anchor, list_anchor + "        Maki,\n", 1)
    factory_py.write_text(src)
    print("[install-maki] patched pier AgentFactory._AGENTS (import Maki; append Maki)")
PYEOF

# shellcheck disable=SC2086
$PIER_PY - <<'PYEOF'
from pier.models.agent.name import AgentName
assert AgentName("maki").value == "maki", "pier AgentName('maki') failed"
from pier.agents.factory import AgentFactory
assert AgentName.MAKI in AgentFactory._AGENT_MAP, "Maki not in pier AgentFactory._AGENT_MAP"
cls = AgentFactory._AGENT_MAP[AgentName.MAKI]
assert cls.name() == "maki", "pier Maki.name() != 'maki'"
print("[install-maki] pier registration verified: --agent maki resolves to pier.agents.installed.maki:Maki")
PYEOF

# -----------------------------------------------------------------------------
# 3. Install the maki binary into the build runtime itself (smoke testing)
# -----------------------------------------------------------------------------
if command -v maki >/dev/null 2>&1; then
    log "maki already on PATH: $(command -v maki) ($(maki --version 2>/dev/null || echo unknown))"
else
    command -v curl >/dev/null 2>&1 || fail "curl is required to install the maki binary"
    log "installing maki $MAKI_RELEASE_TAG into /usr/local/bin ..."
    export MAKI_INSTALL_DIR=/usr/local/bin
    curl -fsSL https://maki.sh/install.sh | sh -s "$MAKI_RELEASE_TAG"
    chmod 0755 /usr/local/bin/maki
fi
log "maki binary: $(maki --version)"

log "done. maki registered in Harbor (-a maki) and pier (--agent maki); binary at /usr/local/bin/maki"
