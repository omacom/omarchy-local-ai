#!/bin/bash

set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
export HOME="$scratch" ENGINE_ROOT="$ROOT" PYTHONDONTWRITEBYTECODE=1
export XDG_CONFIG_HOME="$HOME/.config"

python3 -I <<'PY'
import copy
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys
import tomllib
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.environ["ENGINE_ROOT"])
import agent_local as a
import runpy
bench = SimpleNamespace(**runpy.run_path(str(Path(os.environ["ENGINE_ROOT"]) / "tools/bench")))


def check(name, condition, detail=""):
  if not condition:
    print(f"not ok - {name}", file=sys.stderr)
    if detail:
      print(detail, file=sys.stderr)
    raise SystemExit(1)
  print(f"ok - {name}", flush=True)


def skip(name):
  print(f"ok - {name} # SKIP", flush=True)


def raises(name, fn, message=""):
  try:
    fn()
  except a.LocalError as error:
    check(name, message in str(error), str(error))
    return
  check(name, False, "expected LocalError")


runtime = {
  "image": "registry.example/test/server@sha256:" + "a" * 64,
  "tier": "third-party", "review": "reviews/server.toml", "port": 8080,
  "health": {"probe": ["curl", "http://localhost:8080/health"]},
  "command": ["server"], "reserved": ["--host", "API_KEY"],
  "inputs": {"model": {"arg": "--model"}, "draft": {"arg": "--draft"},
             "ctx": {"arg": "--ctx"}, "alias": {"arg": "--alias"}, "key": {"env": "API_KEY"}},
}
variant = {"ctx": 4096, "peak_ram_gib": 16, "peak_gpu_gib": 16}
artifact = {"repo": "test/model", "revision": "b" * 40, "license": "mit",
            "files": ["model.gguf", "tokenizer/*.json"], "model": "model.gguf"}
recipe = {"name": "Test model", "summary": "Fixture", "runtime": "server",
          "hardware": [{"gpu": "amd-strix-halo", "ram_class": 64, "measured": "measurements/model.toml"}],
          "artifacts": [artifact], "variant": [variant]}
gpu = {"vendor": "amd", "family": "amd-strix-halo", "memory_gib": 48}
profile = {"platform": "x86", "ram_class": 64, "ram_gib": 64, "gpus": [gpu]}
entry = {"gpu": gpu["family"], "ram_class": 64, "recipes": ["model"]}
check("valid recipe and runtime fixtures are accepted", not a.validate_runtime("server", runtime) and not a.validate_recipe("model", recipe, {"server": runtime}))

for rid in ("..", "a..b", "-a", "a-", ".a", "a."):
  check(f"recipe and runtime reject unsafe id {rid}", bool(a.validate_runtime(rid, runtime)) and bool(a.validate_recipe(rid, recipe, {"server": runtime})))
for name, mutate, fragment in (
  ("reserved input target", lambda r: r["inputs"].update(model={"arg": "--host"}), "targets reserved"),
  ("duplicate input target", lambda r: r["inputs"].update(alias={"arg": "--model"}), "same target"),
  ("host IPC", lambda r: r.update(run={"ipc": "host"}), "not an allowed"),
  ("privileged container", lambda r: r.update(run={"privileged": True}), "not an allowed"),
  ("command line key without opt-in", lambda r: r["inputs"].update(key={"arg": "--key"}), "key_on_cmdline = true"),
  ("a positional input other than the model", lambda r: r["inputs"].update(alias={"positional": True}), "only the model can be positional"),
  ("command line key without shell", lambda r: (r.update(key_on_cmdline=True), r["inputs"].update(key={"arg": "--key"})), "needs shell"),
):
  changed = copy.deepcopy(runtime)
  mutate(changed)
  check(f"runtime rejects {name}", any(fragment in e for e in a.validate_runtime("server", changed)))
# A key may use its own reserved environment variable, as in the valid fixture.
changed = copy.deepcopy(runtime)
changed.update(key_on_cmdline=True, shell="/bin/bash")
changed["inputs"]["key"] = {"arg": "--host"}
check("key cannot commandeer a reserved argument", any("reserved" in e for e in a.validate_runtime("server", changed)))

for flag in ("--host", "--host=0.0.0.0", "--model", "--model=other"):
  changed = copy.deepcopy(recipe)
  changed["server"] = {"args": [flag]}
  check(f"server args cannot override {flag}", any("server.args" in e for e in a.validate_recipe("model", changed, {"server": runtime})))
for field, value in (("files", ["../weights"]), ("files", ["/weights"]), ("dest", "../outside"), ("dest", "/outside")):
  changed = copy.deepcopy(recipe)
  changed["artifacts"][0][field] = value
  check(f"artifact rejects escaping {field} {value}", any("relative path" in e for e in a.validate_recipe("model", changed, {"server": runtime})))
for role in ("model", "draft"):
  changed = copy.deepcopy(recipe)
  changed["artifacts"][0][role] = "model.gguf"
  changed["artifacts"].append(copy.deepcopy(changed["artifacts"][0]))
  check(f"duplicate {role} role is rejected", any(f".{role}: already named" in e for e in a.validate_recipe("model", changed, {"server": runtime})))
check("identical best selectors are rejected", any("same selectors" in e for e in a.validate_best({"class": [entry, entry]}, {"model": recipe})))
check("best cannot recommend an unmeasured class", any("no measured" in e for e in a.validate_best({"class": [dict(entry, ram_class=128)]}, {"model": recipe})))

catalog = SimpleNamespace(recipes={"model": copy.deepcopy(recipe)}, runtimes={"server": runtime}, best={"class": [entry]})
check("best requires an exact GPU and RAM pair", a.match_class({"class": [entry]}, profile, dict(gpu, family="other")) is None and a.match_class({"class": [entry]}, dict(profile, ram_class=128), gpu) is None)
check("best requires a GPU selector", any(".gpu" in e for e in a.validate_best({"class": [{"ram_class": 64, "recipes": ["model"]}]}, {"model": recipe})))
check("best rejects platform selectors", any(".platform" in e for e in a.validate_best({"class": [dict(entry, platform="x86")]}, {"model": recipe})))
catalog.recipes["model"]["variant"] = [variant, dict(variant, ctx=8192), dict(variant, ctx=16384, peak_ram_gib=60)]
check("largest fitting context wins", a.resolve_best(catalog, profile, gpu)[1]["ctx"] == 8192)
for reason, bad in (("non-fitting variants", dict(recipe, variant=[dict(variant, peak_gpu_gib=100)])),
                    ("unmeasured hardware", dict(recipe, hardware=[dict(recipe["hardware"][0], ram_class=128)]))):
  catalog.recipes["first"] = bad
  catalog.best = {"class": [dict(entry, recipes=["first", "model"])]}
  result = a.resolve_best(catalog, profile, gpu)
  check(f"best falls through {reason}", result[0] == "model" and result[2][0][0] == "first")
catalog.best = {"class": []}
try:
  a.resolve_best(catalog, profile, gpu)
except a.LocalError as error:
  check("missing recommendation reports the profile", all(str(v) in str(error) for v in (gpu["family"], 64, "x86")))
else:
  check("missing recommendation raises", False)
rival = dict(gpu, family="amd-other")
tied = dict(profile, gpus=[gpu, rival])
raises("different GPU families tied for memory require selection", lambda: a.choose_gpu(tied), "--gpu")
check("explicit GPU resolves a memory tie", a.choose_gpu(tied, "amd-other") == rival)
with patch.object(a, "platform_name", return_value="aarch64-n1x"), patch.object(a, "amd_gpus", return_value=[]), patch.object(a, "meminfo", return_value={"MemTotal": 120 * 2**30}):
  n1x = a.machine_profile()
check("an N1x laptop has a unified NVIDIA GPU of its own family", n1x["ram_class"] == 128 and [(g["vendor"], g["family"], g["memory_gib"]) for g in n1x["gpus"]] == [("nvidia", "nvidia-n1x", 120)])
check("GPU peak exceeding capacity does not fit", not a.fits(dict(variant, peak_gpu_gib=49), profile, gpu)[0])
check("RAM fit preserves desktop reserve", not a.fits(dict(variant, peak_ram_gib=57), profile, gpu)[0])
with patch.object(a, "free_memory", return_value=(1, 1)), patch.object(a, "top_memory_users", return_value="browser (12 GiB), editor (4 GiB)"), patch.object(a.time, "sleep"):
  raises("headroom failure lists largest memory users", lambda: a.check_headroom(variant, gpu), "browser (12 GiB), editor (4 GiB)")

payload = b'{"ok":true}'
oid = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
tree = [{"type": "file", "path": "model.gguf", "size": 2, "lfs": {"oid": "c" * 64, "size": 3}},
        {"type": "file", "path": "tokenizer/x.json", "size": len(payload), "oid": oid}]
with patch.object(a, "hf_tree", return_value=tree):
  changed = copy.deepcopy(recipe)
  changed["artifacts"][0]["dest"] = "UD"
  files, roles = a.build_artifacts(changed)
  check("LFS expansion uses published SHA256 and LFS size", files[0].get("sha256") == "c" * 64 and files[0]["size"] == 3)
  check("small files retain git blob identity", files[1].get("git_oid") == oid)
  check("dest prefixes the full repository path", files[1]["dest"] == "UD/tokenizer/x.json" and roles["model"] == "UD/model.gguf")
  changed["artifacts"][0]["model"] = "."
  check("a '.' role names the directory its artifact lands in", a.build_artifacts(changed)[1]["model"] == "UD" and not a.validate_recipe("model", changed, {"server": runtime}))
  del changed["artifacts"][0]["dest"]
  check("a '.' role needs a dest", any("needs a dest" in e for e in a.validate_recipe("model", changed, {"server": runtime})))
  changed["artifacts"][0]["files"] = ["missing*"]
  raises("unmatched artifact pattern raises", lambda: a.build_artifacts(changed), "no file matching")
  changed = copy.deepcopy(recipe)
  changed["artifacts"].append(dict(artifact, repo="other/model"))
  raises("different artifacts cannot overwrite a destination", lambda: a.build_artifacts(changed), "Two artifacts land")
path = Path(os.environ["HOME"]) / "blob"
path.write_bytes(payload)
check("git blob hash verifies intact file", a.verify_file(path, files[1]))
path.write_bytes(payload[:-1] + b"!")
check("git blob hash rejects same-size one-byte corruption", not a.verify_file(path, files[1]))
record = {"recipe": "model", "method": 'say "hello"\nnext\\line', "ctx": 4096,
          "machine": {"platform": "x86", "flags": ["a", "b"]}, "memory": {"peak": 16.5},
          "artifacts": [{"path": "one", "sha256": "a" * 64}, {"path": "two", "git_oid": oid}],
          "short": {"ttft": 1.5, "ok": True}}
check("bench-shaped TOML round-trips tables, arrays and escaped strings", tomllib.loads(bench.toml_dumps(record)) == record)
check("bench names an x86 CPU from cpuinfo", bench.cpu_name("processor\t: 0\nmodel name\t: AMD RYZEN AI MAX+ PRO 395\n", "Model name: ignored") == "AMD RYZEN AI MAX+ PRO 395")
check("bench names an ARM CPU's core types from lscpu", bench.cpu_name("processor\t: 0\nBogoMIPS\t: 2000.00\n", "Vendor ID: ARM\nModel name: Cortex-X925\nModel name: Cortex-A725\nModel name: Cortex-X925\n") == "Cortex-X925 + Cortex-A725")
check("bench reads the NVIDIA driver version", bench.nvidia_driver("NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  615.78.08  Release Build  (root@host)\n") == "615.78.08"
      and bench.nvidia_driver("NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.95.05  Release Build\n") == "580.95.05" and bench.nvidia_driver("") == "")

# Models run side by side, each on its own port; an agent pointed at one must
# never land on another, so ports are distinct, stable, and freed on removal.
first, second = a.port_for("model-a"), a.port_for("model-b")
check("each model gets its own loopback port", first == a.BASE_PORT and second == a.BASE_PORT + 1)
check("a model keeps its port across lookups", a.port_for("model-a") == first)
ports = a.load_ports()
ports.pop("model-a")
a.write_atomic(a.ports_path(), json.dumps(ports))
check("a removed model's port is reused before a new one", a.port_for("model-c") == first and a.port_for("model-b") == second)

# Materialized file lists cannot escape the store and must carry real hashes.
good = {"dest": "w/model.gguf", "size": 3, "sha256": "a" * 64}
for name, record, roles in (
  ("an escaping destination", {**good, "dest": "../../.bashrc"}, {}),
  ("a malformed hash", {**good, "sha256": "../x"}, {}),
  ("a role naming no file", good, {"model": "other.gguf"}),
  ("a role naming a parent directory", good, {"model": ".."}),
):
  try:
    a.validate_files([record], roles)
  except a.LocalError:
    check(f"a manifest with {name} is refused", True)
  else:
    check(f"a manifest with {name} is refused", False)

a.validate_files([good], {"model": "w"})
check("a manifest role may name the directory holding its files", True)

# A new file set never touches the one a running or previous unit mounts.
def plant(record, body):
  path = a.blob_path(record)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  Path(path).write_bytes(body)
old = {"dest": "w/model.gguf", "size": 3, "sha256": hashlib.sha256(b"old").hexdigest()}
new = {"dest": "w/model.gguf", "size": 3, "sha256": hashlib.sha256(b"new").hexdigest()}
plant(old, b"old")
plant(new, b"new")
first_set = a.materialize("swap", [old], {"model": "w/model.gguf"})
second_set = a.materialize("swap", [new], {"model": "w/model.gguf"})
check("each file set gets its own directory", first_set != second_set)
check("setting up a new file set leaves the previous one intact", Path(a.fileset_dir("swap", first_set), "w/model.gguf").read_bytes() == b"old")

# Exercise typed unit generation, then the real Quadlet parser without starting services.
hostile = '50% $HOME "quoted space" \\tail'
changed = copy.deepcopy(recipe)
changed["server"] = {"args": ["--extra", hostile]}
rt = copy.deepcopy(runtime)
rt["inputs"]["alias"] = {"env": "MODEL_ALIAS"}
rt["health"]["probe"] = ["probe", hostile]
roles = {"model": hostile}
text = a.render_container("model", changed, rt, variant, roles, gpu, "a" * 16)
check("server args escape systemd substitutions, quotes and backslashes", '"50%% $$HOME \\"quoted space\\" \\\\tail"' in text)
check("probe escapes substitutions while retaining JSON argv", "HealthCmd=" + json.dumps(["probe", hostile]).replace("%", "%%").replace("$", "$$") in text)
# Map model through env to pass the same hostile value through Environment=.
env_rt = copy.deepcopy(rt)
env_rt["inputs"]["model"] = {"env": "MODEL_PATH"}
env_text = a.render_container("model", changed, env_rt, variant, roles, gpu, "a" * 16)
check("environment values are quoted and escaped", 'Environment="MODEL_PATH=/models/50%% $$HOME \\"quoted space\\" \\\\tail"' in env_text)
positional_rt = copy.deepcopy(runtime)
positional_rt["command"] = ["server", "serve"]
positional_rt["inputs"]["model"] = {"positional": True}
_, argv, _, _ = a.server_invocation(positional_rt, recipe, variant, {"model": "w"}, "model")
check("a positional model follows the command", not a.validate_runtime("server", positional_rt) and argv[:3] == ["server", "serve", "/models/w"] and "--model" not in argv)
for location in ("name", "args", "env", "command"):
  r, t = copy.deepcopy(changed), copy.deepcopy(env_rt)
  role_values = dict(roles)
  if location == "name":
    r["name"] = "bad\nname"
  elif location == "args":
    r["server"]["args"] = ["bad\narg"]
  elif location == "env":
    role_values["model"] = "bad\npath"
  else:
    t["command"] = ["bad\ncommand"]
  raises(f"renderer rejects newline in {location}", lambda: a.render_container("model", r, t, variant, role_values, gpu, "a" * 16), "control characters")
bad_probe = copy.deepcopy(rt)
bad_probe["health"]["probe"] = ["bad\nprobe"]
try:
  a.render_container("model", changed, bad_probe, variant, roles, gpu, "a" * 16)
except a.LocalError:
  check("renderer rejects probe newline", True)
else:
  check("renderer rejects probe newline", False)
generator = next((p for p in ("/usr/lib/systemd/user-generators/podman-user-generator", "/usr/lib/systemd/system-generators/podman-system-generator") if Path(p).exists()), None)
if generator:
  quadlets = Path(a.quadlet_dir())
  quadlets.mkdir(parents=True)
  (quadlets / "omarchy-agent-local.container").write_text(text)
  (quadlets / "omarchy-agent-local.network").write_text(a.render_network())
  env = dict(os.environ, QUADLET_UNIT_DIRS=str(quadlets))
  result = subprocess.run([generator, "--user", "--dryrun"], env=env, capture_output=True, text=True)
  check("Quadlet accepts generated units", result.returncode == 0, result.stdout + result.stderr)
  exec_line = next((line.split("=", 1)[1] for line in result.stdout.splitlines() if line.startswith("ExecStart=") and " run " in line), "")
  # systemd doubles literal percent and dollars; shlex handles its emitted quoting here.
  argv = [re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m[1], 16)), arg)
          for arg in shlex.split(exec_line.replace("%%", "%").replace("$$", "$"))]
  def option(flag, value):
    return f"{flag}={value}" in argv or any(argv[i:i+2] == [flag, value] for i in range(len(argv)))
  check("Quadlet keeps internal network and container restrictions", option("--network", "omarchy-agent-local") and "--read-only" in argv and option("--cap-drop", "all") and option("--security-opt", "no-new-privileges"), exec_line)
  check("Quadlet publishes only the loopback endpoint", option("--publish", f"127.0.0.1:{a.port_for('model')}:8080"), exec_line)
  check("hostile server argument survives Quadlet as one argument", hostile in argv, repr(argv))
else:
  skip("Quadlet parser unavailable: Podman generator is not installed")

# The default root must work without an Omarchy checkout or a catalog override.
check("installed catalog loads and validates from the engine directory", not a.load_catalog().errors() and a.load_catalog().root == os.environ["ENGINE_ROOT"])
root = Path(os.environ["HOME"])
repo = root / "catalog-fixture"
repo.mkdir()
def write_toml(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(bench.toml_dumps(value))
write_toml(repo / "catalog.toml", {"schema": 1, "version": "0.1.0"})
write_toml(repo / "best.toml", {"class": [entry]})
write_toml(repo / "runtimes/server.toml", runtime)
write_toml(repo / "recipes/model.toml", recipe)
write_toml(repo / "reviews/server.toml", {"image": runtime["image"], "publisher": "Fixture", "reviewer": "Tester", "date": "2026-01-01", "license_url": "https://example.test/license", "license_sha256": "a" * 64, "checked": ["key", "network"]})
write_toml(repo / "measurements/model.toml", {"ctx": 4096})
os.environ["OMARCHY_LOCAL_AI_CATALOG"] = str(repo)
check("catalog root can be overridden", a.load_catalog().root == str(repo))
write_toml(repo / "catalog.toml", {"schema": 999})
raises("invalid installed catalog is rejected", a.load_catalog, "failed validation")
write_toml(repo / "catalog.toml", {"schema": 1, "version": "0.1.0"})
# The package command follows /usr/bin-style symlinks to find its engine.
link = root / "bin/omarchy-local-ai"
link.parent.mkdir()
link.symlink_to(Path(os.environ["ENGINE_ROOT"]) / "omarchy-local-ai")
result = subprocess.run([str(link), "validate", os.environ["ENGINE_ROOT"]], cwd=root, capture_output=True, text=True)
check("entry script imports its engine through a symlink", result.returncode == 0, result.stdout + result.stderr)
with patch.object(a, "machine_profile", return_value=profile), patch.object(a, "running_models", return_value=set()), patch.object(a, "known_models", return_value=[]):
  check("quiet refresh succeeds", a.cmd_refresh(["--quiet"]) == 0)
check("refresh writes both panel caches", Path(a.state_dir(), "list.json").is_file() and Path(a.state_dir(), "status.json").is_file())

# Removing a recipe drops the blobs only it used, even one still linked from
# the seed copy it was set up from, and keeps those another recipe links.
store = Path(a.data_dir())
for name in ("kept", "seeded"):
  (store / "blobs/sha256").mkdir(parents=True, exist_ok=True)
  (store / "blobs/sha256" / name).write_text(name)
for rid, name in (("gone", "kept"), ("gone", "seeded"), ("other", "kept")):
  (store / "models" / rid).mkdir(parents=True, exist_ok=True)
  os.link(store / "blobs/sha256" / name, store / "models" / rid / name)
seed = Path(os.environ["HOME"]) / "seed"
seed.mkdir()
os.link(store / "blobs/sha256/seeded", seed / "seeded")
with patch.object(a, "write_unit_files"):
  a._remove_files("gone")
check("removal drops a blob only a seed copy still links", not (store / "blobs/sha256/seeded").exists() and (seed / "seeded").read_text() == "seeded")
check("removal keeps a blob another recipe links", (store / "blobs/sha256/kept").exists())

# An interrupted removal leaves the model listed, so remove can finish it.
Path(a.model_manifest_path("halfway")).parent.mkdir(parents=True, exist_ok=True)
Path(a.model_manifest_path("halfway")).write_text(json.dumps({"files": []}))
with patch.object(a, "unit_active", return_value=False), patch.object(a, "write_status"), patch.object(a, "_remove_files", side_effect=KeyboardInterrupt):
  try:
    a.cmd_remove(["halfway"])
  except KeyboardInterrupt:
    pass
check("an interrupted removal keeps the model's manifest", Path(a.model_manifest_path("halfway")).exists() and not Path(a.removing_path("halfway")).exists())

# Validation accepts exactly the run-option shapes the generator implements.
for option, value in (("devices", "render"), ("devices", {"render": True}),
                      ("group_add", ["keep-groups"]), ("memlock", ["unlimited"])):
  changed = dict(runtime, run={option: value})
  check(f"run option {option} rejects {value!r}", bool(a.validate_runtime("server", changed)))
check("supported run-option shapes validate", not a.validate_runtime("server", dict(runtime, run={"devices": ["render", "kfd"], "group_add": "keep-groups", "memlock": "unlimited"})))
for fileset in (None, "", "../outside", "a" * 16 + "\n"):
  raises("renderer requires a safe file set", lambda: a.render_container("model", recipe, runtime, variant, {"model": "model.gguf"}, gpu, fileset), "file set")
check("unit mounts the exact immutable file set", "models/model/" + "a" * 16 + ":/models:ro" in text)
check("unit injects key through Podman secrets", f"Secret={a.SECRET},type=env,target={a.KEY_ENV}" in text and f"Secret={a.SECRET},type=env,target=API_KEY" in text)

# A saved setup is installed even if it has never started. Replacing it only
# changes the current manifest, and removal considers surviving manifests.
manifest = {"schema": 1, "recipe_id": "current", "recipe": recipe, "runtime": runtime,
            "runtime_id": "server", "image": runtime["image"], "variant": variant,
            "files": [], "roles": {"model": "model.gguf"}, "fileset": "a" * 16,
            "gpu": gpu["family"], "profile": profile, "catalog": {"version": "0.1.0"},
            "observed": {"load_seconds": 12, "ram_used_gib": 16, "gpu_used_gib": 16}}
path = a.save_model(manifest, started=False)
check("a current manifest alone means installed", a.recipe_installed("current"))
updated = dict(manifest, variant=dict(variant, ctx=8192))
check("saving replaces the same current manifest", a.save_model(updated, started=False) == path and a.load_model("current")["variant"]["ctx"] == 8192)
with patch.object(a, "known_models", return_value=["current"]), patch.object(a, "unit_props", return_value={}), patch.object(a, "free_memory", return_value=(64, 48)):
  pending = a.write_status(a.load_catalog(), profile, {"current": "activating"}, pending={"current": dict(updated, observed=None)})
check("pending activation uses the current manifest's observations", pending["models"]["current"]["observed"] == manifest["observed"] and pending["models"]["current"]["unit"] == "activating")
a.save_model(dict(manifest, recipe_id="survivor"), started=False)
with patch.object(a, "known_models", return_value=["current", "survivor"]), patch.object(a, "write_unit_files"), patch.object(a.subprocess, "run") as rmi:
  a._remove_files("current")
  check("removal retains an image used by a surviving current manifest", not rmi.called)
with patch.object(a, "known_models", return_value=["current"]), patch.object(a, "write_unit_files"), patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as rmi:
  a._remove_files("current")
  check("removal drops an image no surviving manifest uses", rmi.call_args.args[0] == ["podman", "rmi", runtime["image"]])

# A recipe that takes images says so to the agents, and only text and image are allowed.
image_recipe = {**copy.deepcopy(recipe), "input": ["text", "image"]}
check("a recipe can take images", not [e for e in a.validate_recipe("model", image_recipe, {"server": runtime}, root) if ".input" in e])
check("a recipe has to take text", any(".input" in e for e in a.validate_recipe("model", {**copy.deepcopy(recipe), "input": ["image"]}, {"server": runtime}, root)))
with patch.object(a, "resolve_model", return_value={"recipe_id": "model", "recipe": image_recipe, "variant": {"ctx": 4096}}):
  overlay = json.loads(open(a.opencode_overlay("model")[0]).read())["provider"][a.OVERLAY_PROVIDER]["models"]["model"]
  check("the OpenCode overlay declares image input", overlay["attachment"] and overlay["modalities"]["input"] == ["text", "image"])

# Setups saved while memory is busy still owe authentication/model-ID checks.
cat = a.load_catalog()
rid = next(iter(cat.recipes))
rec = cat.recipes[rid]
rt = cat.runtimes[rec["runtime"]]
with patch.object(a, "load_catalog", return_value=cat), patch.object(a, "machine_profile", return_value=profile), patch.object(a, "compatibility", return_value=(variant, "")), patch.object(a, "preflight"), patch.object(a, "build_artifacts", return_value=([], {"model": "model.gguf"})), patch.object(a, "materialize", return_value="a" * 16), patch.object(a, "ensure_image"), patch.object(a, "ensure_key", return_value="test-key"), patch.object(a, "unit_active", return_value=False), patch.object(a, "check_headroom", side_effect=a.LocalError(a.NOT_ENOUGH_MEMORY)), patch.object(a, "write_unit_files"), patch.object(a, "refresh_list_cache"), patch.object(a, "write_status"), patch.object(a, "prune_filesets"):
  check("memory-busy first setup succeeds without activation", a.cmd_use([rid]) == 0 and a.load_model(rid)["unchecked"])
for response, succeeds in (([(200, b"")], False),
                           ([(401, b""), (200, b'{"data":[{"id":"other"}]}')], False),
                           ([(401, b""), (200, json.dumps({"data": [{"id": rid}]}).encode())], True)):
  with patch.object(a, "unit_active", return_value=False), patch.object(a, "write_status"), patch.object(a, "evict_idle_weights"), patch.object(a, "check_headroom", return_value=(100.0, 90.0)), patch.object(a, "free_memory", return_value=(52.0, 64.0)), patch.object(a, "ensure_key", return_value="test-key"), patch.object(a, "http_status", side_effect=response) as http, patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0)) as systemctl:
    if succeeds:
      a._start(SimpleNamespace(recipe=rid), cat, profile)
      check("first checked start clears the unchecked flag", not a.load_model(rid).get("unchecked") and http.call_count == 2)
      check("a start records the memory the model took", a.load_model(rid)["observed"]["ram_used_gib"] == 48.0)
    else:
      raises("first start refuses failed authentication or wrong model ID", lambda: a._start(SimpleNamespace(recipe=rid), cat, profile))
      check("failed first check stops the model and retains its check obligation", ("stop", a.unit_name(rid)) in [call.args for call in systemctl.call_args_list] and a.load_model(rid)["unchecked"])

# A first setup is listed while it fetches: bytes so far, total and speed, in
# the panel's status, until its manifest exists or remove dismisses it.
a.clear_installing("fresh")
a.begin_installing("fresh", {"name": "Fresh model", "vendor": "qwen"})
status = json.loads(Path(a.state_dir(), "status.json").read_text())
check("a setup shows as installing before it fetches anything", status["models"]["fresh"]["unit"] == "installing" and status["models"]["fresh"]["install"]["phase"] == "waiting" and status["models"]["fresh"]["name"] == "Fresh model")
progress = a.Progress(rid="fresh")
progress.samples = [(progress.started - 1, 0)]
progress.grow(4 * 2**20)
progress.published = 0
progress.add(3 * 2**20)
progress.published = 0
progress.pulled(2**20, 2 * 2**20)
row = json.loads(Path(a.state_dir(), "status.json").read_text())["models"]["fresh"]
check("progress counts weights and image layers toward one bar", row["install"]["phase"] == "downloading" and row["install"]["done_bytes"] == 4 * 2**20 and row["install"]["total_bytes"] == 6 * 2**20, json.dumps(row["install"]))
check("progress reports a download speed", row["install"]["rate_bytes_s"] > 0)
with patch.object(a, "known_models", return_value=[]), patch.object(a, "unit_props", return_value={}):
  listed = a.status_record(a.load_catalog(), profile)
check("a full status rewrite keeps the setup's row", listed["models"]["fresh"]["unit"] == "installing")
with patch.object(a, "known_models", return_value=["current"]), patch.object(a, "unit_props", return_value={}), patch.object(a, "free_memory", return_value=(64, 48)):
  a.write_status(a.load_catalog(), profile)
a.publish_installing("current", {"name": "Current", "phase": "downloading"})
check("setting up an installed model again leaves its row as it is", json.loads(Path(a.state_dir(), "status.json").read_text())["models"]["current"]["unit"] != "installing")

a.clear_installing("timed")
a.begin_installing("timed", {"name": "Timed"}, 48.0)
check("a setup's row carries the catalog's load time for the panel's bar", json.loads(Path(a.state_dir(), "status.json").read_text())["models"]["timed"]["expected_load_seconds"] == 48.0)
a.clear_installing("timed")
check("a first start runs against the catalog's load time", a.expected_load_seconds("never-started", {"measured_load_seconds": 48.0}) == 48.0)

# A failed first setup keeps its row with the reason, and remove dismisses it.
a.clear_installing(rid)
with patch.object(a, "load_catalog", return_value=cat), patch.object(a, "machine_profile", return_value=profile), patch.object(a, "compatibility", return_value=(variant, "")), patch.object(a, "load_model", return_value=None), patch.object(a, "preflight"), patch.object(a, "build_artifacts", return_value=([], {"model": "model.gguf"})), patch.object(a, "materialize", side_effect=a.LocalError("Need 90 GiB free for downloads")), patch.object(a, "write_status"):
  raises("a failed setup reports its reason", lambda: a.cmd_use([rid]), "Need 90 GiB")
failed = a.load_installing(rid)
check("a failed setup stays listed with its reason", failed is not None and failed["phase"] == "failed" and failed["error"].startswith("Need 90 GiB"), json.dumps(failed))
check("a setup records its process for cancelling", failed["pid"] == os.getpid())
a.clear_installing(rid)
Path(a.model_manifest_path("dismissed")).unlink(missing_ok=True)
a.begin_installing("dismissed", {"name": "Dismissed"})
a.fail_installing("dismissed", "Download failed")
with patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0)) as stop, patch.object(a, "known_models", return_value=[]), patch.object(a, "write_unit_files"), patch.object(a, "refresh_list_cache"), patch.object(a, "write_status"):
  check("remove dismisses a failed setup", a.cmd_remove(["dismissed"]) == 0 and a.load_installing("dismissed") is None)
  check("remove stops the setup's unit first", ("stop", a.install_unit("dismissed")) in [call.args for call in stop.call_args_list])

# A setup waiting for another's lock is already cancellable, and stops if its
# record was removed while it waited.
class Waiting:
  def __init__(self, during=None):
    self.during = during
  def __enter__(self):
    self.seen = (a.load_installing(rid) or {}).get("pid")
    if self.during:
      self.during()
    return self
  def __exit__(self, *exc):
    return False
a.clear_installing(rid)
waiting = Waiting()
with patch.object(a, "load_catalog", return_value=cat), patch.object(a, "machine_profile", return_value=profile), patch.object(a, "compatibility", return_value=(variant, "")), patch.object(a, "load_model", return_value=None), patch.object(a, "Lock", return_value=waiting), patch.object(a, "_use_setup", return_value=0), patch.object(a, "write_status"):
  a.cmd_use([rid])
check("a setup records its process before it waits for the lock", waiting.seen == os.getpid())
a.update_installing(rid, pid=os.getppid())
with patch.object(a, "load_catalog", return_value=cat), patch.object(a, "machine_profile", return_value=profile), patch.object(a, "compatibility", return_value=(variant, "")), patch.object(a, "load_model", return_value=None), patch.object(a, "Lock", return_value=Waiting()), patch.object(a, "_use_setup", return_value=0), patch.object(a, "write_status"):
  a.cmd_use([rid])
check("a second use of the same recipe leaves the live owner's process in place", a.load_installing(rid)["pid"] == os.getppid())
with patch.object(a, "load_catalog", return_value=cat), patch.object(a, "machine_profile", return_value=profile), patch.object(a, "compatibility", return_value=(variant, "")), patch.object(a, "load_model", return_value=None), patch.object(a, "Lock", return_value=Waiting(lambda: a.clear_installing(rid))), patch.object(a, "_use_setup", return_value=0) as setup, patch.object(a, "write_status"):
  raises("a setup removed while it waited stops", lambda: a.cmd_use([rid]), "cancelled")
  check("a cancelled setup fetches nothing", not setup.called and a.load_installing(rid) is None)

# Bytes an interrupted download left on disk count as done, but not as speed.
resumed = a.Progress()
resumed.grow(10 * 2**20)
resumed.resumed(4 * 2**20)
done_now, total_now, rate_now = resumed.snapshot()
check("a resumed download's saved bytes count toward the bar", done_now == 4 * 2**20 and total_now == 10 * 2**20)
check("a resumed download's saved bytes don't count as speed", rate_now == 0)

# A new attempt replaces a failed setup's record rather than inheriting it.
a.begin_installing("retried", {"name": "Retried"})
a.fail_installing("retried", "Download failed")
check("a failed setup can be set up again", a.installing_idle("retried"))
a.begin_installing("retried", {"name": "Retried"})
fresh_record = a.load_installing("retried")
check("a retry starts without the old failure", fresh_record["phase"] == "waiting" and fresh_record["error"] == "")
check("a setup under way isn't replaced", not a.installing_idle("retried"))
a.clear_installing("retried")

# Cancelling stops a model its setup already started and drops only its own partial downloads.
staging = Path(a.data_dir(), "staging")
staging.mkdir(parents=True, exist_ok=True)
mine, other = "a" * 64 + ".part", "b" * 64 + ".part"
(staging / mine).write_bytes(b"x" * 10)
(staging / other).write_bytes(b"y" * 10)
a.begin_installing("cancelled", {"name": "Cancelled"})
a.update_installing("cancelled", partials=[mine, "../escape.part"])
with patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0)) as stop, patch.object(a, "known_models", return_value=[]), patch.object(a, "write_unit_files"), patch.object(a, "refresh_list_cache"), patch.object(a, "write_status"):
  a.cmd_remove(["cancelled"])
check("cancelling a setup stops its model service", ("stop", a.unit_name("cancelled")) in [call.args for call in stop.call_args_list])
check("cancelling a setup drops only its own partial downloads", not (staging / mine).exists() and (staging / other).exists())

# --detach hands the setup to a transient user unit and returns.
a.clear_installing(rid)
with patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=3)), patch.object(a, "load_model", return_value=None), patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as run:
  check("a detached setup returns at once", a.detach_use(rid, SimpleNamespace(ctx=8192, gpu=None, seed=None, unqualified=False), cat.recipes[rid]) == 0)
  argv = run.call_args.args[0]
check("a detached setup runs use in its own user unit", argv[:3] == ["systemd-run", "--user", "--collect"] and f"--unit={a.install_unit(rid)}" in argv and argv[-4:-2] == ["use", rid] and argv[-2:] == ["--ctx", "8192"], " ".join(argv))
check("a detached setup is listed before its unit starts", a.load_installing(rid)["phase"] == "waiting")
a.clear_installing(rid)
with patch.dict(os.environ, {"HF_TOKEN": "hf_secret_value", "OMARCHY_LOCAL_AI_CATALOG": "relative/catalog"}), patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=3)), patch.object(a, "load_model", return_value=None), patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as run:
  a.detach_use(rid, SimpleNamespace(ctx=None, gpu=None, seed=None, unqualified=False), cat.recipes[rid])
  argv, env = run.call_args.args[0], run.call_args.kwargs["env"]
check("a detached setup never puts the token on a command line", not any("hf_secret_value" in arg for arg in argv) and "--setenv=HF_TOKEN" in argv and env["HF_TOKEN"] == "hf_secret_value")
check("a detached setup gets the catalog as an absolute path", env["OMARCHY_LOCAL_AI_CATALOG"] == os.path.abspath("relative/catalog"))
a.clear_installing(rid)

# Refresh shares one catalog/profile across both caches.
with patch.object(a, "load_catalog", return_value=cat) as load, patch.object(a, "machine_profile", return_value=profile) as machine, patch.object(a, "known_models", return_value=[]), patch.object(a, "running_models", return_value=set()):
  a.cmd_refresh(["--quiet"])
  check("refresh loads catalog and machine profile once", load.call_count == machine.call_count == 1)

PY
