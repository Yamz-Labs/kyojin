"""CPU tests for the container entrypoint generated in flake.nix (kyojinEntryPoint).

The script is authored inside flake.nix, which substitutes nothing but the flavor list, so the
test reads the block out of the flake and runs it under bash with a stub `hf` and a stub server.
That keeps the test on the shipped text and off the network: no hub, no credentials, no model.
The four behaviours covered are the ones the sync/offline fallback exists for:

  - a failed sync over a complete pack serves it, with a warning and exit status 0;
  - a failed sync over an empty or incomplete pack stops with its own message;
  - HF_HUB_OFFLINE never invokes the sync at all;
  - HF_HUB_OFFLINE without an installed pack stops with a different message.

Run: python -m pytest docker/test_flake_entrypoint.py
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FLAKE = REPO_ROOT / "flake.nix"
BLOCK_START = 'kyojinEntryPoint = pkgs.writeShellScript "kyojin-entrypoint" ' + "''"
BLOCK_END = "          '';"
REPO_ID = "yamz-labs/Qwen3.8-Flash-Next-EXL3-Yamz"


def flavors():
    """The flake derives the flavor list from tools/<name>/serve.py; same rule here."""
    return sorted(p.parent.name for p in (REPO_ROOT / "tools").glob("*/serve.py"))


def entrypoint_source():
    """The shell the flake ships, with its two nix interpolations substituted."""
    lines = FLAKE.read_text().splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if BLOCK_START in line) + 1
        end = next(i for i in range(start, len(lines)) if lines[i] == BLOCK_END)
    except StopIteration:
        raise AssertionError(f"no kyojinEntryPoint block found in {FLAKE}")
    src = textwrap.dedent("\n".join(lines[start:end])) + "\n"
    # Inside a nix '' string, ''${ is an escaped shell ${; the shipped script has the shell form.
    src = src.replace("''${", "${")
    assert "''" not in src, "unhandled nix escaping in the entrypoint: " + src
    src = src.replace('${lib.concatStringsSep " | " flavors}', " | ".join(flavors()))
    src = src.replace('${lib.concatStringsSep " " flavors}', " ".join(flavors()))
    assert "${lib." not in src, "unhandled nix interpolation in the entrypoint"
    for marker in ("set -euo pipefail", "hf download", "HF_HUB_OFFLINE", "config.json"):
        assert marker in src, f"extracted entrypoint is missing {marker!r}"
    return src


def stub(python: str) -> str:
    """An executable script; the interpreter is this one, since sandbox /bin has only sh."""
    return f"#!{sys.executable}\n{python}"


HF_STUB = """
import os, sys
argv = sys.argv[1:]
log = os.environ.get("HF_LOG")
if log:
    with open(log, "a") as f:
        f.write(" ".join(argv) + "\\n")
rc = int(os.environ.get("HF_RC", "1"))
if rc == 0 and os.environ.get("HF_CREATE") == "1":
    with open(os.path.join(argv[argv.index("--local-dir") + 1], "config.json"), "w") as f:
        f.write('{"model_type": "qwen4_exp"}\\n')
print(f"hf: (stub) exiting {rc}", file=sys.stderr)
sys.exit(rc)
"""

SERVE_STUB = """
import os, sys
log = os.environ.get("SERVE_LOG")
if log:
    with open(log, "a") as f:
        f.write(" ".join(sys.argv[1:]) + "\\n")
sys.exit(0)
"""


class Result:
    def __init__(self, completed, tmp_path):
        self.code = completed.returncode
        self.stdout = completed.stdout
        self.stderr = completed.stderr
        self.hf_log = tmp_path / "hf.log"
        self.serve_log = tmp_path / "serve.log"

    @property
    def hf_calls(self):
        return self.hf_log.read_text().splitlines() if self.hf_log.exists() else []

    @property
    def serve_calls(self):
        return self.serve_log.read_text().splitlines() if self.serve_log.exists() else []


def run(tmp_path, pack="none", marker=None, hf_rc=1, offline=None, revision=None, create=False):
    """Run the entrypoint once. pack: none (empty slot), complete, incomplete, empty-config."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "hf").write_text(stub(HF_STUB))
    (bin_dir / "hf").chmod(0o755)
    (bin_dir / "kyojin-serve").write_text(stub(SERVE_STUB))
    (bin_dir / "kyojin-serve").chmod(0o755)
    script = tmp_path / "kyojin-entrypoint"
    script.write_text(entrypoint_source())

    slot = tmp_path / "slot"
    slot.mkdir()
    if pack == "complete":
        (slot / "config.json").write_text('{"model_type": "qwen4_exp"}\n')
        (slot / "model.safetensors").write_text("w")
    elif pack == "incomplete":
        (slot / "model.safetensors").write_text("w")
    elif pack == "empty-config":
        (slot / "config.json").write_text("")
    if marker is not None:
        (slot / ".kyojin-repo").write_text(marker)

    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "KYOJIN_FLAVOR": "qwen",
        "KYOJIN_MODEL_REPO": REPO_ID,
        "KYOJIN_MODEL_DIR": str(slot),
        "KYOJIN_HOME": str(tmp_path / "home"),
        "KYOJIN_BIN": str(bin_dir / "kyojin-serve"),
        "HF_LOG": str(tmp_path / "hf.log"),
        "SERVE_LOG": str(tmp_path / "serve.log"),
        "HF_RC": str(hf_rc),
    }
    if offline is not None:
        env["HF_HUB_OFFLINE"] = offline
    if revision is not None:
        env["KYOJIN_MODEL_REVISION"] = revision
    if create:
        env["HF_CREATE"] = "1"
    completed = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, capture_output=True, text=True)
    result = Result(completed, tmp_path)
    result.slot = slot
    return result


# ---- a failed sync over a complete pack serves it

def test_failed_sync_serves_the_installed_pack(tmp_path):
    r = run(tmp_path, pack="complete", hf_rc=1)
    assert r.code == 0
    assert "WARNING: syncing" in r.stderr and "serving the pack already at" in r.stderr
    assert len(r.hf_calls) == 1 and r.hf_calls[0].startswith(f"download {REPO_ID} --local-dir ")
    assert len(r.serve_calls) == 1 and "--model" in r.serve_calls[0]


def test_failed_sync_serves_a_complete_pack_without_a_marker(tmp_path):
    r = run(tmp_path, pack="complete", hf_rc=1)
    assert r.code == 0 and len(r.serve_calls) == 1
    # the marker records a verified sync; a failed one must not write it
    assert not (r.slot / ".kyojin-repo").exists()


# ---- a failed sync over an empty or incomplete pack stops

def test_failed_sync_stops_on_an_empty_slot(tmp_path):
    r = run(tmp_path, pack="none", hf_rc=1)
    assert r.code != 0
    assert "failed and" in r.stderr and "holds no complete pack" in r.stderr
    assert "offline (HF_HUB_OFFLINE)" not in r.stderr
    assert r.serve_calls == []
    assert not (r.slot / ".kyojin-repo").exists()


def test_failed_sync_stops_on_an_incomplete_slot(tmp_path):
    r = run(tmp_path, pack="incomplete", hf_rc=1)
    assert r.code != 0
    assert "failed and" in r.stderr and "holds no complete pack" in r.stderr
    assert r.serve_calls == []


def test_an_empty_config_json_is_not_a_pack(tmp_path):
    r = run(tmp_path, pack="empty-config", hf_rc=1)
    assert r.code != 0
    assert "holds no complete pack" in r.stderr
    assert r.serve_calls == []


# ---- HF_HUB_OFFLINE skips the sync instead of attempting it

def test_offline_never_invokes_the_sync(tmp_path):
    r = run(tmp_path, pack="complete", hf_rc=0, offline="1")
    assert r.hf_calls == [] and not r.hf_log.exists()
    assert r.code == 0
    assert "HF_HUB_OFFLINE is set, not syncing" in r.stderr
    assert len(r.serve_calls) == 1


def test_offline_without_a_pack_stops_with_its_own_message(tmp_path):
    r = run(tmp_path, pack="none", offline="1")
    assert r.hf_calls == [] and not r.hf_log.exists()
    assert r.code != 0
    assert "offline (HF_HUB_OFFLINE)" in r.stderr
    assert "failed and" not in r.stderr  # not the sync-failed message
    assert r.serve_calls == []


def test_offline_zero_still_syncs(tmp_path):
    r = run(tmp_path, pack="complete", hf_rc=1, offline="0")
    assert len(r.hf_calls) == 1
    assert "HF_HUB_OFFLINE is set" not in r.stderr
    assert "WARNING: syncing" in r.stderr and r.code == 0


# ---- the slot and the marker still decide before any sync

def test_a_slot_holding_another_repo_stops_before_the_sync(tmp_path):
    r = run(tmp_path, pack="complete", marker="someone/other-pack", hf_rc=0)
    assert r.code == 2
    assert "holds someone/other-pack" in r.stderr
    assert r.hf_calls == [] and not r.hf_log.exists()
    assert r.serve_calls == []


# ---- a successful sync: revision forwarded, marker written, a no-op sync is not a pack

def test_a_successful_sync_forwards_the_revision_and_records_it(tmp_path):
    r = run(tmp_path, pack="none", hf_rc=0, revision="SC_4.00bpw_H5_V6", create=True)
    assert r.code == 0
    assert r.hf_calls == [f"download {REPO_ID} --revision SC_4.00bpw_H5_V6 --local-dir {r.slot}"]
    assert (r.slot / ".kyojin-repo").read_text() == f"{REPO_ID}@SC_4.00bpw_H5_V6"
    assert len(r.serve_calls) == 1


def test_a_sync_that_downloaded_nothing_is_not_a_pack(tmp_path):
    # a revision that matches nothing downloads no files and still exits 0
    r = run(tmp_path, pack="none", hf_rc=0)
    assert r.code == 2
    assert "left no config.json" in r.stderr
    assert r.serve_calls == []
    assert not (r.slot / ".kyojin-repo").exists()
