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
# The machine running the tests may be on a circle; only the tests that say so are.
a.circle_status = lambda: None


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
for name, args, fragment in (("a reserved flag", ["--host"], "reserved"), ("an input's flag", ["--model=other"], "inputs"),
                             ("a shared flag", ["--threads", "2"], "already in server.args")):
  changed = copy.deepcopy(recipe)
  changed["server"] = {"args": ["--threads", "4"]}
  changed["variant"] = [dict(variant, args=args)]
  check(f"variant args reject {name}", any(".variant[0].args" in e and fragment in e for e in a.validate_recipe("model", changed, {"server": runtime})))
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
small = dict(variant, ctx=8192, args=["--parallel", "1"])
catalog.recipes["model"]["variant"] = [small, dict(variant, ctx=16384, peak_ram_gib=60, args=["--parallel", "3"])]
picked = a.resolve_best(catalog, profile, gpu)[1]
check("a smaller machine falls back to the variant sized for it", picked is small)
catalog.recipes["model"]["variant"] = [dict(small, ctx=16384, peak_ram_gib=40), dict(small, ctx=16384, peak_ram_gib=60, args=["--parallel", "3"])]
check("at the same context a smaller machine takes the smaller setup", a.resolve_best(catalog, profile, gpu)[1]["peak_ram_gib"] == 40)
check("at the same context a larger machine takes the larger setup", a.resolve_best(catalog, dict(profile, ram_gib=128), dict(gpu, memory_gib=96))[1]["args"] == ["--parallel", "3"])
_, argv, _, _ = a.server_invocation(runtime, dict(recipe, server={"args": ["--vision"]}), picked, {"model": "w"}, "model")
check("a variant's args follow the shared server args", argv[-3:] == ["--vision", "--parallel", "1"])
for reason, bad in (("non-fitting variants", dict(recipe, variant=[dict(variant, peak_gpu_gib=100)])),
                    ("unmeasured hardware", dict(recipe, hardware=[dict(recipe["hardware"][0], ram_class=128)]))):
  catalog.recipes["first"] = bad
  catalog.best = {"class": [dict(entry, recipes=["first", "model"])]}
  result = a.resolve_best(catalog, profile, gpu)
  check(f"best falls through {reason}", result[0] == "model" and result[2][0][0] == "first")
catalog.recipes["first"] = recipe
catalog.recipes["model"]["variant"] = [variant, dict(variant, ctx=8192)]
result = a.resolve_best(catalog, profile, gpu, ctx=8192)
check("best falls through a recipe without the requested context", result[0] == "model" and result[1]["ctx"] == 8192 and "ctx 8192" in result[2][0][1])
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
spark = dict(recipe, hardware=[{"gpu": "nvidia-gb10", "ram_class": 128, "measured": "measurements/model.toml"}])
check("an N1x runs what a GB10 with the same memory runs", a.qualified(spark, n1x, n1x["gpus"][0]) and a.match_class({"class": [dict(entry, gpu="nvidia-gb10", ram_class=128)]}, n1x, n1x["gpus"][0]) is not None)
check("best treats the N1x and GB10 as one class", any("same selectors" in e for e in a.validate_best({"class": [dict(entry, gpu="nvidia-gb10", ram_class=128, recipes=["spark"]), dict(entry, gpu="nvidia-n1x", ram_class=128, recipes=["spark"])]}, {"spark": spark})))
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

# Paired machines: a runtime spanning two runs on the host network only with a [pair] table,
# binds its API to loopback through flags it reserves, and leaves NCCL settings to the engine.
pair_rt = copy.deepcopy(runtime)
pair_rt.update(command=["server", "serve"], listen={"host": "--host", "port": "--port"}, reserved=["--host", "--port", "API_KEY"],
               run={"devices": ["cdi-nvidia", "infiniband"], "network": "fabric", "memlock": "unlimited"},
               health={"probe": ["curl", "http://127.0.0.1:@PORT@/health"]},
               pair={"nodes": 2, "master_port": 29620, "state": "/state",
                     "env": {"NCCL_COMM_ID": "@MASTER_ADDR@:@MASTER_PORT@", "TF_RANK": "@RANK@"}})
check("a paired runtime fixture is accepted", not a.validate_runtime("pair", pair_rt), "\n".join(a.validate_runtime("pair", pair_rt)))
for name, mutate, fragment in (
  ("host network without a pair", lambda r: r.pop("pair"), "only for runtimes with a [pair]"),
  ("a pair on the internal network", lambda r: r["run"].pop("network"), "needs run.network"),
  ("a pair without bind flags", lambda r: r.pop("listen"), ".listen"),
  ("bind flags in the command", lambda r: r.update(command=["server", "--port", "1"]), "sets --host, --port itself"),
  ("unreserved bind flags", lambda r: r.update(reserved=["API_KEY"]), "must reserve"),
  ("NCCL settings in the runtime", lambda r: r["pair"]["env"].update(NCCL_IB_HCA="mlx5_0"), "NCCL_"),
  ("three nodes", lambda r: r["pair"].update(nodes=3), "only 2"),
  ("no state path", lambda r: r["pair"].pop("state"), "pair.state"),
  ("a state path over the models", lambda r: r["pair"].update(state="/models"), "pair.state"),
  ("the link's NCCL interface", lambda r: r["pair"]["env"].update(NCCL_SOCKET_IFNAME="eth0"), "NCCL_SOCKET_IFNAME"),
):
  changed = copy.deepcopy(pair_rt)
  mutate(changed)
  check(f"paired runtime rejects {name}", any(fragment in e for e in a.validate_runtime("pair", changed)), "\n".join(a.validate_runtime("pair", changed)))

spark = {"vendor": "nvidia", "family": "nvidia-gb10", "memory_gib": 121.9, "unified": True, "sysfs": ""}
pair_gpu = dict(spark, family="nvidia-gb10-pair", nodes=2, memory_gib=121.7, peer="10.99.0.2")
pair_profile = {"platform": "aarch64-gb10", "ram_class": 128, "ram_gib": 121.9, "gpus": [spark, pair_gpu]}
check("a pair outranks its own node", a.choose_gpu(pair_profile)["family"] == "nvidia-gb10-pair")
single = dict(recipe, hardware=[{"gpu": "nvidia-gb10", "ram_class": 128, "measured": "m.toml"}], runtime="server")
paired = dict(recipe, hardware=[{"gpu": "nvidia-gb10-pair", "ram_class": 128, "measured": "m.toml"}], runtime="pair",
              variant=[{"ctx": 4096, "peak_ram_gib": 104, "peak_gpu_gib": 104}])
check("a pair machine runs single-node recipes on its own GPU", a.recipe_gpu(pair_profile, single)["family"] == "nvidia-gb10" and a.recipe_gpu(pair_profile, paired)["family"] == "nvidia-gb10-pair")
nv_rt = dict(runtime, run={"devices": ["cdi-nvidia"]})
pair_catalog = SimpleNamespace(recipes={"single": single, "paired": paired}, runtimes={"server": nv_rt, "pair": pair_rt},
                               best={"class": [{"gpu": "nvidia-gb10", "ram_class": 128, "recipes": ["single"]},
                                               {"gpu": "nvidia-gb10-pair", "ram_class": 128, "recipes": ["paired"]}]})
check("best on a pair is the pair's recipe", a.resolve_best(pair_catalog, pair_profile)[0] == "paired")
check("best without the pair is the node's own", a.resolve_best(pair_catalog, dict(pair_profile, gpus=[spark]))[0] == "single")
small = dict(pair_profile, gpus=[spark, dict(pair_gpu, memory_gib=100)])
check("best falls back to the node's class when the pair recipe doesn't fit", a.resolve_best(pair_catalog, small)[0] == "single")
check("a paired runtime never runs on one node, nor a single one on the pair", not a.runtime_supports(pair_rt, spark) and not a.runtime_supports(nv_rt, pair_gpu))
pair_text = a.render_container("paired", paired, pair_rt, paired["variant"][0], {"model": "model"}, pair_gpu, "b" * 16)
pair_port = a.port_for("paired")
check("the pair unit shares the host network and publishes nothing", "Network=host" in pair_text and "PublishPort" not in pair_text)
check("the pair unit binds its API to loopback on the model's port", f'"--host" "127.0.0.1" "--port" "{pair_port}"' in pair_text and f"127.0.0.1:{pair_port}/health" in pair_text)
check("the pair unit mounts rank 0's state folder writable and reads the link from its environment file", f"models/paired/{'b' * 16}.rank0:/state\n" in pair_text and "EnvironmentFile=%h/.local/state/omarchy/agent-local/pair/paired.env" in pair_text)
check("the pair unit starts and stops rank 1 with itself", "ExecStartPre=/usr/bin/env omarchy-agent-local link-worker start paired" in pair_text and "ExecStopPost=-/usr/bin/env omarchy-agent-local link-worker stop paired" in pair_text)
_, rank_argv, _, _ = a.head_argv(pair_rt, paired, paired["variant"][0], {"model": "model"}, "paired")
check("both ranks run the runtime's command, bound to loopback", rank_argv[-4:] == ["--host", "127.0.0.1", "--port", str(a.port_for("paired"))] and rank_argv[:2] == ["server", "serve"])
# Each rank gets its own link settings, and rank 1 the head's command inside the same restrictions.
setup = {"schema": 1, "recipe_id": "paired", "runtime": pair_rt, "recipe": paired, "variant": paired["variant"][0], "roles": {"model": "model"},
         "fileset": "e" * 16, "pair": {"peer": "bob@10.99.1.2", "host": "spark2", "home": "/home/bob", "image_id": "f" * 64}}
a.write_pair_setup(setup)
here = {"addr": "10.99.1.1", "dev": "enp1", "hcas": ["rocep1", "roceP2"], "gid": "5"}
there = {"addr": "10.99.1.2", "dev": "enp1", "hcas": ["rocep1", "roceP2"], "gid": "3"}
with patch.object(a, "pair_link", return_value=(here, there)), patch.object(a, "peer_headroom") as headroom, patch.object(a, "pair_ssh") as ssh, patch.object(a, "pair_limits") as limits:
  a.pair_worker_start("paired")
head_env = dict(line.split("=", 1) for line in Path(a.pair_env_path("paired")).read_text().splitlines())
worker = shlex.split(ssh.call_args.args[1])
worker_env = dict(worker[i + 1].split("=", 1) for i, word in enumerate(worker) if word == "-e")
check("rank 0's environment is rank 0's side of the link", head_env == {"NCCL_COMM_ID": "10.99.1.1:29620", "TF_RANK": "0", "NCCL_SOCKET_IFNAME": "enp1", "NCCL_IB_HCA": "rocep1,roceP2", "NCCL_IB_GID_INDEX": "5"}, repr(head_env))
check("rank 1's environment is rank 1's side, same rendezvous", worker_env["TF_RANK"] == "1" and worker_env["NCCL_COMM_ID"] == "10.99.1.1:29620" and worker_env["NCCL_IB_GID_INDEX"] == "3" and "API_KEY" not in worker_env, repr(worker_env))
check("rank 1 starts on the peer as its user, with the head's command and the same restrictions",
      ssh.call_args.args[0] == "bob@10.99.1.2" and headroom.called
      and limits.called and all(flag in worker for flag in ("--read-only", "no-new-privileges", "/dev/infiniband", "f" * 64))
      and worker[worker.index("--cap-drop") + 1] == "all"
      and f"/home/bob/.local/share/omarchy/agent-local/models/paired/{'e' * 16}:/models:ro" in worker
      and f"/home/bob/.local/share/omarchy/agent-local/models/paired/{'e' * 16}.rank1:/state" in worker
      and worker[worker.index("f" * 64) + 1:][:2] == ["server", "serve"] and worker[-4:] == ["--host", "127.0.0.1", "--port", str(a.port_for("paired"))], repr(worker))
os.unlink(a.pair_setup_path("paired"))
# Setup copies the image (only when the peer lacks it) and the file set, and makes both ranks' state folders.
ran = []
def fake_run(argv, **kwargs):
  ran.append(argv)
  return SimpleNamespace(returncode=0, stdout="f" * 64 + "\n", stderr="")
manifest = {"fileset": "e" * 16}
with patch.object(a.subprocess, "run", side_effect=fake_run), patch.object(a.subprocess, "Popen") as save, \
     patch.object(a, "pair_ssh", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as ssh, patch.object(a, "load_installing", return_value=None):
  a.pair_setup("paired", manifest, pair_rt, {"peer": "bob@10.99.1.2", "host": "spark2", "home": "/home/bob"})
rsync = next(argv for argv in ran if argv[0] == "rsync")
check("setup records the peer and the image by ID", manifest["pair"] == {"peer": "bob@10.99.1.2", "host": "spark2", "home": "/home/bob", "image_id": "f" * 64})
check("setup copies the file set to the peer, not an image it already has", not save.called and rsync[-1] == f"bob@10.99.1.2:/home/bob/.local/share/omarchy/agent-local/models/paired/{'e' * 16}/")
check("setup makes both ranks' state folders", Path(a.rank_dir("paired", "e" * 16, 0)).is_dir() and any(f"{'e' * 16}.rank1" in call.args[1] for call in ssh.call_args_list))
check("the pair unit keeps the container restrictions", all(line in pair_text for line in ("ReadOnly=true", "DropCapability=all", "NoNewPrivileges=true", "AddDevice=/dev/infiniband")))
if generator:
  quadlets = Path(a.quadlet_dir())
  (quadlets / "omarchy-agent-local.container").write_text(pair_text)
  result = subprocess.run([generator, "--user", "--dryrun"], env=dict(os.environ, QUADLET_UNIT_DIRS=str(quadlets)), capture_output=True, text=True)
  check("Quadlet accepts the pair unit", result.returncode == 0 and "--network host" in result.stdout.replace("=", " "), result.stdout[-2000:] + result.stderr)
scope = {}
exec(a.FABRIC_SIDE, scope)
check("the fabric probe runs on its own and finds no RoCE device on loopback", (scope["fabric_side"]("127.0.0.1") or {}).get("hca") is None)
base = Path(a.data_dir(), "models", "keep")
for name in ("c" * 16, "c" * 16 + ".rank0", "d" * 16, "d" * 16 + ".rank0"):
  (base / name).mkdir(parents=True)
a.prune_filesets("keep", "c" * 16)
check("pruning keeps a file set's rank inputs and drops the others'", sorted(p.name for p in base.iterdir()) == ["c" * 16, "c" * 16 + ".rank0"])
with patch.object(a, "platform_name", return_value="aarch64-gb10"), patch.object(a, "amd_gpus", side_effect=lambda: []):
  check("no pair without a record of one", "pair" not in a.machine_profile())
  Path(a.pair_path()).parent.mkdir(parents=True, exist_ok=True)
  Path(a.pair_path()).write_text(json.dumps({"peer": "10.99.0.2", "host": "spark2", "home": "/home/u", "ram_gib": 121.9}))
  cached = a.machine_profile()
  check("a recorded pair adds the pair GPU without probing", [g["family"] for g in cached["gpus"]] == ["nvidia-gb10", "nvidia-gb10-pair"] and cached["gpus"][1]["peer"] == "10.99.0.2", json.dumps(cached))
  os.unlink(a.pair_path())
  # The record keeps the SSH user link setup was given, and routes by the address alone.
  routed = []
  with patch.object(a, "pair_python", return_value={"gpu": "NVIDIA GB10", "ram_total": 128 * 2**30, "home": "/home/bob", "host": "spark2"}), \
       patch.object(a, "_fabric", side_effect=lambda address: routed.append(address) or {"hca": "rocep1s0f0"}):
    a.record_pair(pair_profile, "bob@10.99.1.2")
  check("a recorded peer keeps the SSH user it was set up with", a.cached_pair(pair_profile)["peer"] == "bob@10.99.1.2" and routed == ["10.99.1.2"], repr(routed))
  with patch.object(a, "local_ports", return_value={"enp1": ["10.99.1.1/24"], "lan0": ["192.168.1.5/24"]}), \
       patch.object(a, "pair_python", return_value={"enp1": ["10.99.1.2/24"]}):
    check("link setup reaches the peer at its own cable address, as the same user", a.cable_peer("bob@spark-760a.local") == "bob@10.99.1.2")
  with patch.object(a, "pair_python", return_value={"gpu": "", "ram_total": 128 * 2**30, "home": "/", "host": "x"}):
    raises("only a GB10 is recorded as a peer", lambda: a.record_pair(pair_profile, "10.99.1.9"), "isn't a DGX Spark")
  os.unlink(a.pair_path())

for name, script in (("host", a.HOST_SETUP.replace("@CDI_COMPAT@", a.CDI_COMPAT)), ("link", a.LINK_SETUP), ("firewall", a.FIREWALL_SETUP)):
  result = subprocess.run(["bash", "-n", "-c", script], capture_output=True, text=True)
  check(f"link setup's {name} script parses", result.returncode == 0, result.stderr)
# Cable addressing finishes whatever an earlier setup left, and leaves ports that aren't shared alone.
fresh = a.plan_link({"enp1": [], "enP2": []}, {"enp1": [], "enP2": []})
check("a fresh cable gets a /24 per port, .1 here and .2 there", fresh == (["enP2=10.99.0.1/24", "enp1=10.99.1.1/24"], ["enP2=10.99.0.2/24", "enp1=10.99.1.2/24"]), repr(fresh))
half = a.plan_link({"enp1": ["10.99.1.1/24"], "lan0": ["192.168.1.5/24"]}, {"enp1": []})
check("an interrupted setup gives the peer the other end of this side's /24 and ignores unshared ports", half == ([], ["enp1=10.99.1.2/24"]), repr(half))
check("a cable addressed on both ends is left alone", a.plan_link({"enp1": ["10.99.1.1/24"]}, {"enp1": ["10.99.1.2/24"]}) == ([], []))
check("a new /24 avoids networks either end already uses", a.plan_link({"enp1": [], "enp2": ["10.99.0.1/24"]}, {"enp1": [], "enp2": ["10.99.0.2/24"]})[0] == ["enp1=10.99.1.1/24"])
rules = a.link_rules({"enp1": ["10.99.1.1/24"], "lan0": ["192.168.1.5/24"]}, {"enp1": ["10.99.1.2/24"], "lan0": ["192.168.1.9/24"], "ib9": ["10.0.0.2/8"]}, {"enp1"})
check("the firewall lets in only the other end's address, on the rails alone", rules == (["enp1=10.99.1.2"], ["enp1=10.99.1.1"]), repr(rules))
with patch.object(a, "pair_ssh", side_effect=a.LocalError("unreachable")), patch.object(a, "known_models", return_value=[]):
  raises("removing a paired model stops while its peer can't be reached", lambda: a.pair_remove({"recipe_id": "paired", "pair": {"peer": "10.99.1.2", "host": "spark2", "home": "/home/u", "image_id": "c" * 64}}), "--local-only")
  a.pair_remove({"recipe_id": "paired", "pair": {"peer": "10.99.1.2", "host": "spark2", "home": "/home/u", "image_id": "c" * 64}}, local_only=True)
  check("--local-only removes a paired model whose peer is gone", True)
compile(a.CDI_COMPAT, "nvidia-cdi-podman-compat", "exec")
check("the CDI compatibility script compiles", True)

# A Spark on the cable before setup is offered by the address mDNS names; a linked one by its host.
with patch.object(a, "cable_neighbors", return_value=["fe80::1%enp1", "fe80::2%enp1"]), patch.object(a, "mdns_names", return_value={"fe80::2": "spark-760a"}):
  offered = a.link_probe(pair_profile)
check("a Spark on the cable waiting for setup is offered by its mDNS name", offered == {"state": "available", "address": "fe80::2%enp1", "host": "spark-760a"} and a.link_state() == offered, repr(offered))
check("a linked Spark is shown as linked", a.link_probe(dict(pair_profile, pair={"host": "spark2"})) == {"state": "linked", "host": "spark2"})
with patch.object(a, "cable_neighbors", return_value=[]):
  check("an empty cable offers nothing", a.link_probe(pair_profile) == {"state": "none"})
check("machines that can't pair offer nothing", a.link_probe(dict(pair_profile, platform="x86")) == {"state": "none"})
with patch.object(a, "cable_neighbors", return_value=["fe80::2%enp1"]), patch.object(a, "mdns_names", return_value={}), \
     patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as sent:
  a.link_notify(pair_profile)
  a.link_notify(pair_profile)
notices = [call.args[0] for call in sent.call_args_list if call.args and call.args[0][:1] == ["omarchy-notification-send"]]
check("a newly cabled Spark is announced once, with link setup as its action", len(notices) == 1 and notices[0][-1] == "omarchy-agent-local link setup", repr(notices))
os.unlink(os.path.join(a.state_dir(), "link-notified"))
# A refresh that found the Spark first doesn't stop the announcement.
with patch.object(a, "cable_neighbors", return_value=["fe80::3%enp1"]), patch.object(a, "mdns_names", return_value={}):
  a.link_probe(pair_profile)
  with patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as sent:
    a.link_notify(pair_profile)
check("a Spark a refresh found first is still announced", any(c.args and c.args[0][:1] == ["omarchy-notification-send"] for c in sent.call_args_list))
os.unlink(os.path.join(a.state_dir(), "link-notified"))
# A send that failed (no notification daemon yet, say) is tried again next time.
with patch.object(a, "cable_neighbors", return_value=["fe80::4%enp1"]), patch.object(a, "mdns_names", return_value={}):
  with patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="", stderr="")):
    a.link_notify(pair_profile)
  with patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as sent:
    a.link_notify(pair_profile)
check("a failed announcement is tried again", any(c.args and c.args[0][:1] == ["omarchy-notification-send"] for c in sent.call_args_list))
os.unlink(os.path.join(a.state_dir(), "link-notified"))
os.unlink(a.link_path())
# The parsers read what the tools print.
ping = """PING ff02::1%enp1 (ff02::1%enp1) 56 data bytes
64 bytes from fe80::adfd:1ade:3de8:d49b%enp1: icmp_seq=1 ttl=64 time=0.067 ms
64 bytes from fe80::aba5:f382:c378:65aa%enp1: icmp_seq=1 ttl=64 time=0.133 ms
64 bytes from fe80::5a6b:10b2:a79:3111%enp1: icmp_seq=1 ttl=64 time=0.472 ms
64 bytes from fe80::5a6b:10b2:a79:3111%enp1: icmp_seq=2 ttl=64 time=0.401 ms (DUP!)
"""
own = a.own_link_local(json.dumps([{"ifname": "enp1", "addr_info": [{"family": "inet6", "local": "fe80::adfd:1ade:3de8:d49b", "scope": "link"}]},
                                   {"ifname": "enp2", "addr_info": [{"family": "inet6", "local": "fe80::aba5:f382:c378:65aa", "scope": "link"},
                                                                    {"family": "inet6", "local": "fd00::1", "scope": "global"}]}]))
check("own link-local addresses are read from ip -j", own == {"fe80::adfd:1ade:3de8:d49b", "fe80::aba5:f382:c378:65aa"}, repr(own))
check("a ping's answers drop this machine's own addresses and repeats", a.ping_answers(ping, own) == ["fe80::5a6b:10b2:a79:3111"])
check("a failed ping or ip read gives nothing", a.ping_answers("", own) == [] and a.own_link_local("not json") == set())
avahi = "+;enp1;IPv6;spark-760a;SSH Remote Terminal;local\n=;enp1;IPv6;spark-760a;SSH Remote Terminal;local;spark-760a.local;fe80::9514:4be5:80e5:7b8b;22;\n=;enP7s7;IPv4;spark-760a;SSH Remote Terminal;local;spark-760a.local;192.168.1.162;22;\n"
check("mDNS names come from avahi-browse's resolved IPv6 lines", a.mdns_ssh_hosts(avahi) == {"fe80::9514:4be5:80e5:7b8b": "spark-760a"})
# Once a pair is recorded, status says linked whatever the last probe saw.
a.write_atomic(a.link_path(), json.dumps({"state": "available", "address": "fe80::2%enp1", "host": "x"}))
with patch.object(a, "known_models", return_value=[]), patch.object(a, "installing_models", return_value=[]), patch.object(a, "resolve_best", side_effect=a.LocalError("none")):
  linked = a.status_record(SimpleNamespace(version="test", best={}, recipes={}), dict(pair_profile, pair={"host": "spark2"}))["link"]
check("status shows a recorded pair as linked, not a stale offer", linked == {"state": "linked", "host": "spark2"}, repr(linked))
os.unlink(a.link_path())
check("the cable hook parses", subprocess.run(["bash", "-n", os.path.join(os.environ["ENGINE_ROOT"], "system/90-omarchy-local-ai-link")]).returncode == 0)

# A model that stopped unexpectedly says why: this start's own recorded reason, else the
# server's last error line, else systemd's result.
journal = "info: weights read\nerror: ncclCommInitRank: unhandled system error (run with NCCL_DEBUG=INFO for details) (2)\nsome trailing line"
started = int(__import__("time").time() - 60)
with patch.object(a, "unit_props", return_value={"InactiveExitTimestamp": "@%d" % started}), patch.object(a, "journal_tail", return_value=journal) as tail:
  a.clear_error("crashy")
  check("a crash shows the server's own last error", a.failure_reason("crashy", "exit-code").startswith("ncclCommInitRank: unhandled system error"))
  check("only this start's log is searched, not an earlier run's", tail.call_args.args[2] == started * 1000, repr(tail.call_args))
  a.record_error("crashy", "Locked memory limited: run link setup")
  check("a reason the start already recorded isn't overwritten", a.failure_reason("crashy", "exit-code") == "Locked memory limited: run link setup")
with patch.object(a, "unit_props", return_value={"InactiveExitTimestamp": "@%d" % int(__import__("time").time() + 60)}), patch.object(a, "journal_tail", return_value="nothing useful"):
  check("an older reason gives way, and with no server error systemd's result is shown", a.failure_reason("crashy", "exit-code") == "Stopped unexpectedly (exit-code)")
a.clear_error("crashy")
# logs prints the last error, the model's journal and, for a pair, rank 1's log from the peer.
a.record_error("crashy", "Locked memory limited: run link setup")
a.write_atomic(a.model_manifest_path("crashy"), json.dumps({"recipe_id": "crashy", "pair": {"peer": "10.99.1.2", "host": "spark2", "home": "/home/u"}}))
printed = []
with patch.object(a, "say", side_effect=printed.append), \
     patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="Nov 10 rank0 line", stderr="")), \
     patch.object(a, "pair_ssh", return_value=SimpleNamespace(returncode=0, stdout="rank1 line", stderr="")) as peer_logs:
  a.cmd_logs(["crashy", "-n", "50"])
check("logs shows the last error, this machine's log and rank 1's from the peer",
      printed[0] == "Last error: Locked memory limited: run link setup" and "Nov 10 rank0 line" in printed and "rank1 line" in printed
      and "podman logs --tail 50" in peer_logs.call_args.args[1], repr(printed))
os.unlink(a.model_manifest_path("crashy"))
a.clear_error("crashy")
# A failed first setup's reason lives in its installing record.
a.write_atomic(a.installing_path("crashy"), json.dumps({"recipe_id": "crashy", "phase": "failed", "error": "Download failed",
                                                       "pair": {"peer": "10.99.1.2", "host": "spark2", "home": "/home/u"}}))
printed = []
with patch.object(a, "say", side_effect=printed.append), \
     patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")), \
     patch.object(a, "pair_ssh", return_value=SimpleNamespace(returncode=0, stdout="rank1 failed", stderr="")):
  a.cmd_logs(["crashy"])
check("logs shows a failed setup's saved reason", printed[:1] == ["Last error: Download failed"], repr(printed))
check("a failed first paired start still shows rank 1's log", "rank1 failed" in printed, repr(printed))
os.unlink(a.installing_path("crashy"))

# The default root must work without an Omarchy checkout or a catalog override.
check("installed catalog loads and validates from the engine directory", not a.load_catalog().errors() and a.load_catalog().root == os.environ["ENGINE_ROOT"])
root = Path(os.environ["HOME"])
repo = root / "catalog-fixture"
repo.mkdir()
def write_toml(path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(bench.toml_dumps(value))
write_toml(repo / "catalog.toml", {"schema": 1})
write_toml(repo / "best.toml", {"class": [entry]})
write_toml(repo / "runtimes/server.toml", runtime)
write_toml(repo / "recipes/model.toml", recipe)
write_toml(repo / "reviews/server.toml", {"image": runtime["image"], "publisher": "Fixture", "reviewer": "Tester", "date": "2026-01-01", "license_url": "https://example.test/license", "license_sha256": "a" * 64, "checked": ["key", "network"]})
write_toml(repo / "measurements/model.toml", {"ctx": 4096})
os.environ["OMARCHY_LOCAL_AI_CATALOG"] = str(repo)
check("catalog root can be overridden", a.load_catalog().root == str(repo))
write_toml(repo / "catalog.toml", {"schema": 999})
raises("invalid installed catalog is rejected", a.load_catalog, "failed validation")
write_toml(repo / "catalog.toml", {"schema": 1})
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
            "gpu": gpu["family"], "profile": profile,
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

# pi's own extension lists every model set up, each at its own port, and
# follows them: none without pi's folder, gone with the last model.
def pi_registered(path):
  text = open(path).read()
  name, _, rest = text.split("pi.registerProvider(", 1)[1].partition(", ")
  return json.loads(name), json.loads(rest.rsplit(");", 1)[0])
pi_set_up = {"alpha": {"recipe_id": "alpha", "recipe": image_recipe, "variant": {"ctx": 4096}},
             "beta": {"recipe_id": "beta", "recipe": {**copy.deepcopy(recipe), "name": "Beta"}, "variant": {"ctx": 8192}}}
pi_ext = Path(a.pi_extension_path())
with patch.object(a, "known_models", return_value=["alpha", "beta"]), patch.object(a, "load_model", side_effect=pi_set_up.get):
  a.sync_pi_extension()
  check("no pi folder, no pi extension", not pi_ext.exists() and not pi_ext.parent.parent.exists())
  pi_ext.parent.parent.mkdir(parents=True)
  a.sync_pi_extension()
  name, provider = pi_registered(pi_ext)
  urls = {m["id"]: m["baseUrl"] for m in provider["models"]}
  check("pi's extension registers the omarchy-local provider", name == "omarchy-local" == a.OVERLAY_PROVIDER)
  check("pi's extension lists every model set up at its own port", urls == {"alpha": a.endpoint("alpha"), "beta": a.endpoint("beta")} and urls["alpha"] != urls["beta"], repr(urls))
  check("pi's extension reads the key from its file", provider["apiKey"] == "!cat -- " + shlex.quote(a.key_path()))
  check("pi's extension carries each model's context and input", [(m["contextWindow"], m["input"]) for m in provider["models"]] == [(4096, ["text", "image"]), (8192, ["text"])])
  with patch.object(a, "resolve_model", return_value=pi_set_up["beta"]):
    path, overlay_name, overlay_model = a.pi_overlay("beta")
  name, provider = pi_registered(path)
  check("a local launch overlay is the same provider, the launched model first and the rest after", (name, overlay_name, overlay_model) == ("omarchy-local", "omarchy-local", "beta") and [m["id"] for m in provider["models"]] == ["beta", "alpha"])
  mtime = pi_ext.stat().st_mtime_ns
  a.sync_pi_extension()
  check("an unchanged list leaves pi's extension alone", pi_ext.stat().st_mtime_ns == mtime)
with patch.object(a, "known_models", return_value=[]):
  a.sync_pi_extension()
  check("pi's extension goes with the last model", not pi_ext.exists())
import fcntl
def sync_while_locked():
  # Another writer can't take the status lock while pi's extension is written.
  with open(os.path.join(a.state_dir(), "status.lock"), "w") as probe:
    try:
      fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
      raise AssertionError("pi's extension was written outside the status lock")
    except BlockingIOError:
      pass
  raise OSError("read-only")
with patch.object(a, "status_record", return_value={"models": {}, "share_error": ""}), patch.object(a, "publish_shares"), patch.object(a, "sync_pi_extension", side_effect=sync_while_locked) as sync:
  check("a status write syncs pi's extension under its lock and survives it failing", "models" in a.write_status(a.load_catalog(), profile) and sync.called)
check("agents default to a 16k reply cap", a.max_output_tokens({"recipe": recipe, "variant": {"ctx": 131072}}) == 16384)
capped = {**copy.deepcopy(recipe), "server": {"args": ["--max-tokens", "32768", "--vision"]}}
check("agents get the server's reply cap", a.max_output_tokens({"recipe": capped, "variant": {"ctx": 131072}}) == 32768)
check("a variant's reply cap counts too", a.max_output_tokens({"recipe": recipe, "variant": {"ctx": 131072, "args": ["--max-tokens=65536"]}}) == 65536)
shipped = tomllib.load(open(os.path.join(os.environ["ENGINE_ROOT"], "recipes", "qwen3.8-27b-tensorfold.toml"), "rb"))
check("shipped recipes give agents a 64k reply cap", all(a.max_output_tokens({"recipe": shipped, "variant": v}) == 65536 for v in shipped["variant"]))
# Agents re-send every screenshot they have read, and tensorfold takes only 4 images per request by default.
recipe_dir = os.path.join(os.environ["ENGINE_ROOT"], "recipes")
seeing = [tomllib.load(open(os.path.join(recipe_dir, name), "rb"))["server"]["args"] for name in sorted(os.listdir(recipe_dir)) if "tensorfold" in name]
seeing = [args for args in seeing if "--vision" in args]
check("tensorfold recipes that take images allow 50 per request", seeing and all("--vision-max-images" in args and args[args.index("--vision-max-images") + 1] == "50" for args in seeing))
check("the reply cap never exceeds the context", a.max_output_tokens({"recipe": capped, "variant": {"ctx": 8192}}) == 8192)
with patch.object(a, "resolve_model", return_value={"recipe_id": "model", "recipe": capped, "variant": {"ctx": 131072}}):
  check("the pi overlay carries the server's reply cap", '"maxTokens": 32768' in open(a.pi_overlay("model")[0]).read())
  check("the OpenCode overlay carries the server's reply cap", json.loads(open(a.opencode_overlay("model")[0]).read())["provider"][a.OVERLAY_PROVIDER]["models"]["model"]["limit"]["output"] == 32768)

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
check("a pair's memory is both ranks'", a.ram_used_gib({"pair": {"peer": "10.99.1.2"}}, 100.0, 10.0, 120.0, 29.0) == 181.0 and a.ram_used_gib({"pair": {"peer": "10.99.1.2"}}, 100.0, 10.0, None, 29.0) == 180.0 and a.ram_used_gib({}, 100.0, 10.0, None, None) == 90.0)
check("a pair's memory is shown against both machines", a.nodes_ram_gib({"pair": {"peer": "10.99.1.2"}}, {"ram_gib": 121.9, "pair": {"peer": "10.99.1.2", "ram_gib": 121.7}}) == 243.6 and a.nodes_ram_gib({}, {"ram_gib": 121.9, "pair": {"peer": "10.99.1.2", "ram_gib": 121.7}}) == 121.9)
check("a pair's memory total is its own peer's, not a later link's", a.nodes_ram_gib({"pair": {"peer": "10.99.1.2"}, "profile": {"pair": {"peer": "10.99.1.2", "ram_gib": 121.7}}}, {"ram_gib": 121.9, "pair": {"peer": "10.99.1.3", "ram_gib": 60.0}}) == 243.6)

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
second = Waiting()
with patch.object(a, "load_catalog", return_value=cat), patch.object(a, "machine_profile", return_value=profile), patch.object(a, "compatibility", return_value=(variant, "")), patch.object(a, "load_model", return_value=None), patch.object(a, "Lock", return_value=second), patch.object(a, "_use_setup", return_value=0), patch.object(a, "write_status"):
  a.cmd_use([rid])
check("a second use of the same recipe leaves the live owner's process in place while it waits", second.seen == os.getppid())
check("a queued use takes the setup over once it holds the lock", a.load_installing(rid)["pid"] == os.getpid())
a.update_installing(rid, pid=os.getppid())
class Interrupted(Waiting):
  def __enter__(self):
    raise KeyboardInterrupt
with patch.object(a, "load_catalog", return_value=cat), patch.object(a, "machine_profile", return_value=profile), patch.object(a, "compatibility", return_value=(variant, "")), patch.object(a, "load_model", return_value=None), patch.object(a, "Lock", return_value=Interrupted()), patch.object(a, "_use_setup", return_value=0), patch.object(a, "write_status"):
  try:
    a.cmd_use([rid])
  except KeyboardInterrupt:
    pass
check("interrupting a use still waiting leaves the active setup's record", (a.load_installing(rid) or {}).get("pid") == os.getppid())
a.update_installing(rid, image=runtime["image"])
with patch.object(a, "load_catalog", return_value=cat), patch.object(a, "machine_profile", return_value=profile), patch.object(a, "Lock", return_value=Waiting()), patch.object(a, "load_model", return_value=None), patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0)), patch.object(a, "stop_setup_process"), patch.object(a, "write_status"), patch.object(a, "refresh_list_cache"), patch.object(a, "_remove_files", return_value=(0, [])) as removed:
  a.cmd_remove([rid])
check("cancelling a setup removes the image it pulled", removed.call_args.args == (rid, runtime["image"]))
with patch.object(a, "known_models", return_value=[]), patch.object(a, "write_unit_files"), patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as rmi:
  a._remove_files(rid, runtime["image"])
check("removal drops a cancelled setup's image no installed model uses", rmi.call_args.args[0] == ["podman", "rmi", runtime["image"]])
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

# ---- tailnet sharing

ts_status = {
  "BackendState": "Running",
  "Self": {"DNSName": "me.tail.ts.net.", "OS": "linux", "Online": True},
  "Peer": {
    "a": {"DNSName": "box.tail.ts.net.", "OS": "linux", "Online": True},
    "b": {"DNSName": "away.tail.ts.net.", "OS": "linux", "Online": False},
    "c": {"DNSName": "phone.tail.ts.net.", "OS": "android", "Online": True},
    "d": {"DNSName": "plain.tail.ts.net.", "OS": "linux", "Online": True},
    "e": {"DNSName": "flaky.tail.ts.net.", "OS": "linux", "Online": True},
  },
}
listing = {"schema": 1, "host": "box", "dns": "box.tail.ts.net", "ram_total_gib": 121.9, "models": [
  {"recipe": "big-model", "name": "Big\x1b[31m Model", "vendor": "qwen", "architecture": "moe", "ctx": 262144, "max_tokens": 10 ** 9, "input": ["text", "image", "audio"], "unit": "active",
   "observed": {"ram_used_gib": 80, "evil": "x"}, "since_ms": None, "decode_tok_s": 40},
  {"recipe": "../escape", "name": "bad"},
  {"recipe": "odd-state", "unit": "rebooting", "architecture": "<b>moe</b>"},
]}
clean = a.clean_listing(listing, "box.tail.ts.net")
check("a peer's listing keeps only recipe ids it can put in a path", [m["recipe"] for m in clean["models"]] == ["big-model", "odd-state"])
check("a peer's listing loses control characters and unknown inputs", clean["models"][0]["name"] == "Big[31m Model" and clean["models"][0]["input"] == ["text", "image"])
check("a peer's listing keeps only known states and measurements", clean["models"][1]["unit"] == "inactive" and "evil" not in clean["models"][0]["observed"])
check("a peer's listing keeps a known architecture and drops any other", clean["models"][0]["architecture"] == "moe" and clean["models"][1]["architecture"] == "")
check("a peer's reply cap is kept within its context, and a missing one is the default",
      clean["models"][0]["max_tokens"] == 262144 and clean["models"][1]["max_tokens"] == a.DEFAULT_MAX_OUTPUT_TOKENS)
raises("a listing of another schema is refused", lambda: a.clean_listing({"schema": 2, "models": []}, "box"), "Not an Omarchy")
check("a number too big for a float is no number", a._number(10 ** 400) is None)
class Huge:
  status = 200
  def __enter__(self): return self
  def __exit__(self, *args): return False
  def read(self, size=-1):
    return (b'{"schema": 1, "models": [' + b" " * (2 * a.HTTP_MAX_BYTES))[:size if size >= 0 else None]
with patch.object(a.urllib.request, "urlopen", return_value=Huge()):
  state, huge, _ = a.fetch_listing("big.tail.ts.net")
check("a peer's oversized listing is read only up to the cap and left unreachable", state == "unreachable" and huge is None)
with patch.object(a, "http_status", return_value=(200, b"[" * 100000 + b"]" * 100000)):
  state, deep, _ = a.fetch_listing("deep.tail.ts.net")
check("a peer's deeply nested listing leaves only that peer unreachable", state == "unreachable" and deep is None)
with patch.object(a.urllib.request, "urlopen", side_effect=__import__("http.client").client.BadStatusLine("garbage")):
  state, bad, _ = a.fetch_listing("bad.tail.ts.net")
check("a peer's malformed HTTP answer leaves only that peer unreachable", state == "unreachable" and bad is None)

answers = {
  "box.tail.ts.net": ("online", clean, ""),
  "plain.tail.ts.net": ("not_sharing", None, ""),
  "flaky.tail.ts.net": ("unreachable", None, "HTTP 502"),
}
previous = {"hosts": {
  "away": {"host": "away", "dns": "away.tail.ts.net", "last_seen_ms": 1_000_000, "ram_total_gib": 64},
  "flaky": {"host": "flaky", "dns": "flaky.tail.ts.net", "last_seen_ms": 900_000},
  "gone": {"host": "gone", "dns": "gone.tail.ts.net", "last_seen_ms": 2_000_000 - a.FORGET_UNPAIRED_MS},
}, "models": {
  "away/small-model": {"id": "away/small-model", "host": "away", "recipe": "small-model", "name": "Small", "unit": "active", "since_ms": 5},
  "flaky/mid-model": {"id": "flaky/mid-model", "host": "flaky", "recipe": "mid-model", "name": "Mid", "unit": "active"},
  "gone/old-model": {"id": "gone/old-model", "host": "gone", "recipe": "old-model", "unit": "active"},
}}
now = 2_000_000
with patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  net = a.survey_network(previous, now=now)
check("the survey lists a sharing peer's models by host/recipe", net["models"]["box/big-model"]["endpoint"] == "https://box.tail.ts.net/local-ai/big-model/v1" and net["models"]["box/big-model"]["offline"] == "")
check("a model its host has stopped is left off the list", "box/odd-state" not in net["models"])
check("the survey skips this machine, phones and peers sharing nothing", set(net["hosts"]) == {"box", "away", "flaky"})
check("sharing needs HTTPS certificates on the tailnet", net["can_share"] is False)
with patch.object(a, "tailscale_status", return_value={**ts_status, "CertDomains": ["me.tail.ts.net"]}), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  check("a tailnet with MagicDNS and HTTPS can share", a.survey_network(previous, now=now)["can_share"] is True)
# A circle is surveyed beside the tailnet: its hosts at Circle's port, a host
# on both reached over Tailscale.
circle = {"phase": "running", "self": {"name": "me", "addresses": ["10.37.0.3"]}, "peers": [
  {"name": "spark", "addresses": ["10.37.0.1"], "online": True},
  {"name": "box", "addresses": ["10.37.0.4"], "online": True},
]}
answers["spark.circle.internal"] = ("online", clean, "")
with patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "circle_status", return_value=circle), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  both = a.survey_network(previous, now=now)
check("a circle's sharing host is listed at Circle's port", both["models"]["spark/big-model"]["endpoint"] == f"http://spark.circle.internal:{a.CIRCLE_PORT}/local-ai/big-model/v1" and both["circle"])
check("a host on the tailnet and the circle is reached over Tailscale", both["hosts"]["box"]["dns"] == "box.tail.ts.net")
check("a circle can share without Tailscale's certificates", both["can_share"] is True)
with patch.object(a, "tailscale_status", return_value=None), patch.object(a, "circle_status", return_value=circle):
  check("a host is paired by its tailnet name on a tailnet, by its circle name off one",
        a.pairing_name(["spark.circle.internal"]) == "spark.circle.internal" and a.pairing_name(["box.tail.ts.net", "box.circle.internal"]) == "box.circle.internal")
check("a host that went offline keeps its models, marked offline", net["models"]["away/small-model"]["offline"] == "offline" and net["models"]["away/small-model"]["unit"] == "inactive" and net["hosts"]["away"]["state"] == "offline")
check("an online host that stopped answering is marked unreachable", net["models"]["flaky/mid-model"]["offline"] == "unreachable" and net["hosts"]["flaky"]["error"] == "HTTP 502")
check("offline hosts keep their last-seen time", net["hosts"]["away"]["last_seen_ms"] == 1_000_000 and net["hosts"]["box"]["last_seen_ms"] == now)
check("an unpaired host unseen for a week is dropped", "gone/old-model" not in net["models"])
with patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  later = a.survey_network(net, now=now + a.FORGET_UNPAIRED_MS)
check("an unpaired offline host is forgotten after a week", "away/small-model" not in later["models"] and "box/big-model" in later["models"])

# Pairing records survive; a paired host stays listed however long it's away.
a.ensure_dir(a.remotes_dir(), 0o700)
a.write_atomic(a.remote_key_path("away"), "k" * 32, mode=0o600)
a.write_atomic(os.path.join(a.remotes_dir(), "away.json"), json.dumps({"host": "away", "dns": "away.tail.ts.net", "ssh": "away"}))
with patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  later = a.survey_network(net, now=now + 10 * a.FORGET_UNPAIRED_MS)
check("a paired offline host keeps its models however long it's away", later["models"]["away/small-model"]["paired"] and later["models"]["away/small-model"]["offline"] == "offline")
with patch.object(a, "tailscale_status", return_value=None), patch.object(a, "fetch_listing") as fetch:
  alone = a.survey_network(net, now=now)
check("without tailscaled, paired hosts show offline and nothing is fetched", not fetch.called and alone["models"]["away/small-model"]["offline"] == "offline" and not alone["tailnet"])
with patch.object(a, "tailscale_status", return_value=None), patch.object(a, "fetch_listing") as fetch:
  blind = a.survey_network(previous, now=now)
check("without tailscaled, an unpaired host seen this week stays, offline", blind["models"].get("flaky/mid-model", {}).get("offline") == "offline" and "gone/old-model" not in blind["models"], repr(sorted(blind["models"])))
a.write_atomic(a.network_path(), json.dumps(later))
with patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  a.cmd_forget(["away"])
forgotten = a.load_network()
check("forgetting a host drops its key and its models at once", not os.path.exists(a.remote_key_path("away")) and "away/small-model" not in forgotten["models"] and "box/big-model" in forgotten["models"])

# Pairing is per model: a host's key connects only the models it names.
def pair_away(**record):
  a.write_atomic(a.remote_key_path("away"), "k" * 32, mode=0o600)
  a.write_atomic(os.path.join(a.remotes_dir(), "away.json"), json.dumps({"host": "away", "dns": "away.tail.ts.net", "ssh": "away", **record}))
def away_paired():
  with patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
    return {mid: m["paired"] for mid, m in a.survey_network(later, now=now)["models"].items() if m["host"] == "away"}
pair_away()
check("a host paired whole connects every model it shares", away_paired() == {"away/small-model": True})
pair_away(models=["other-model"])
check("a host paired for one model leaves its others unpaired", away_paired() == {"away/small-model": False})
raises("a model that isn't connected can't be used", lambda: a.resolve_remote("away/small-model"), "isn't paired")
a.write_atomic(a.network_path(), json.dumps(later))
with patch.object(a.subprocess, "run", side_effect=AssertionError("no SSH")), patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  a.cmd_pair(["away.tail.ts.net", "small-model"])
check("pairing another model of a paired host needs no SSH", a.load_remotes()["away"]["models"] == ["other-model", "small-model"] and away_paired() == {"away/small-model": True})
with patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  a.cmd_forget(["away/small-model"])
check("disconnecting a model keeps the host's others and its key", a.load_remotes()["away"]["models"] == ["other-model"] and os.path.exists(a.remote_key_path("away")) and a.load_network()["models"]["away/small-model"]["paired"] is False)
with patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  a.cmd_forget(["away/small-model"])
check("disconnecting a model that isn't connected leaves the host paired", a.load_remotes()["away"]["models"] == ["other-model"])
# A host paired whole: disconnecting one model needs no list of the others,
# so a model it shares later, or an empty survey, can't disconnect more.
pair_away()
a.write_atomic(a.network_path(), json.dumps({"hosts": {}, "models": {}}))
with patch.object(a, "tailscale_status", return_value=None), patch.object(a, "fetch_listing"):
  a.cmd_forget(["away/small-model"])
whole = a.load_remotes()["away"]
check("disconnecting a model of a host paired whole keeps the key and every other model, even with no survey",
      os.path.exists(a.remote_key_path("away")) and not a.connected(whole, "small-model") and a.connected(whole, "new-model"))
a.write_atomic(a.network_path(), json.dumps(later))
with patch.object(a.subprocess, "run", side_effect=AssertionError("no SSH")), patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  a.cmd_pair(["away", "small-model"])
check("pairing it again reconnects it", a.connected(a.load_remotes()["away"], "small-model"))
pair_away(models=["small-model"])
with patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  a.cmd_forget(["away/small-model"])
check("disconnecting the only model of a one-at-a-time pairing unpairs the host", "away" not in a.load_remotes() and not os.path.exists(a.remote_key_path("away")))

# The same host reached by another SSH name keeps what it connects.
pair_away(models=["other-model"])
export = json.dumps({"host": "away", "dns": "away.tail.ts.net", "key": "n" * 32})
with patch.object(a.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, export + "\n", "")), patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "fetch_listing", side_effect=lambda dns: answers[dns]):
  a.cmd_pair(["alice@away", "small-model"])
again = a.load_remotes()["away"]
check("pairing a host again under another name adds to its models", again["models"] == ["other-model", "small-model"] and again["ssh"] == "alice@away")
# sshd answering but refusing every key isn't sshd being off: an agent left
# unstarted offers no key at all.
with patch.object(a.subprocess, "run", return_value=subprocess.CompletedProcess([], 255, "", "alice@new: Permission denied (publickey).\n")):
  raises("pairing with a host that refuses this machine's keys points at the SSH agent", lambda: a.cmd_pair(["alice@new"]), "SSH agent")
  # The panel shows the first line alone, so it names the machine short.
  raises("a pairing error's first line names the machine without user or domain", lambda: a.cmd_pair(["alice@new.tail1234.ts.net"]), "new refused this machine's SSH keys\n")
check("an address stays whole as a machine's name", a.short_host("dhh@100.90.162.80") == "100.90.162.80")
# The agents panel pairs with no terminal (stdin here is the test script), so
# ssh must ask nothing: a prompt would wait forever. A new host's key is taken.
with patch.object(a.subprocess, "run", return_value=subprocess.CompletedProcess([], 255, "", "Host key verification failed.\n")) as run:
  raises("pairing with a host whose key changed says how to forget the old one, by host alone", lambda: a.cmd_pair(["alice@new.tail1234.ts.net"]), "ssh-keygen -R new.tail1234.ts.net")
check("pairing without a terminal asks ssh for no prompts and takes a new host's key", {"BatchMode=yes", "StrictHostKeyChecking=accept-new"} <= set(run.call_args[0][0]), repr(run.call_args[0][0]))
a.write_atomic(a.remote_key_path("away"), "k" * 32, mode=0o600)
a.write_atomic(os.path.join(a.remotes_dir(), "away.json"), json.dumps({"host": "away", "dns": "away.tail.ts.net", "ssh": "away"}))
raises("a network model is named host/recipe", lambda: a.split_remote_id("box/../x"), "host/recipe")
raises("an SSH target can't smuggle options", lambda: a.ssh_command({"ssh": "-oProxyCommand=x", "dns": ""}, "share", "--export"), "Not an SSH target")
check("remote commands quote their arguments", a.ssh_command({"ssh": "box", "dns": "box.tail.ts.net"}, "share", "--export")[-2:] == ["box", "omarchy-agent-local share --export"])

# A paired, running remote model becomes an overlay with its own provider.
a.write_atomic(a.remote_key_path("box"), "b" * 32, mode=0o600)
a.write_atomic(os.path.join(a.remotes_dir(), "box.json"), json.dumps({"host": "box", "dns": "box.tail.ts.net", "ssh": "box"}))
with patch.object(a, "fetch_listing", return_value=("online", clean, "")):
  path, provider, model = a.pi_overlay("box/big-model")
  source = open(path).read()
  check("a remote pi overlay points at the host's tailnet endpoint and key", provider == "omarchy-local-box" and model == "big-model" and "https://box.tail.ts.net/local-ai/big-model/v1" in source and a.remote_key_path("box") in source and "b" * 32 not in source)
  oc = json.loads(open(a.opencode_overlay("box/big-model")[0]).read())["provider"]["omarchy-local-box"]
  check("a remote OpenCode overlay keeps the model's context and inputs", oc["models"]["big-model"]["limit"]["context"] == 262144 and oc["models"]["big-model"]["attachment"])
  raises("a remote model that isn't running can't be launched", lambda: a.pi_overlay("box/odd-state"), "isn't running")
raises("an unpaired host's model can't be launched", lambda: a.pi_overlay("plain/x"), "isn't paired")

# Paired hosts' models join pi's extension, one provider per host with its
# key, whether or not the host answers right now; unpaired ones don't.
seen = {"box/big-model": a.remote_record("box", "box.tail.ts.net", clean["models"][0], True, 128, 1),
        "box/odd-state": {**a.remote_record("box", "box.tail.ts.net", clean["models"][1], True, 128, 1), "offline": "offline"},
        "plain/x": a.remote_record("plain", "plain.tail.ts.net", {**clean["models"][0], "recipe": "x"}, False, 64, 1)}
a.write_atomic(a.network_path(), json.dumps({"hosts": {}, "models": seen}))
Path(a.pi_extension_path()).parent.parent.mkdir(parents=True, exist_ok=True)
with patch.object(a, "known_models", return_value=[]):
  a.sync_pi_extension()
  registered = open(a.pi_extension_path()).read()
  check("pi's extension lists a paired host's models under its own provider and key, offline ones too",
        registered.count("pi.registerProvider(") == 1 and '"omarchy-local-box"' in registered and '"big-model"' in registered and '"odd-state"' in registered
        and "https://box.tail.ts.net/local-ai/big-model/v1" in registered and a.remote_key_path("box") in registered and "b" * 32 not in registered)
  check("pi's extension leaves out an unpaired host's models", "plain" not in registered)
  with patch.object(a, "fetch_listing", return_value=("online", clean, "")):
    source = open(a.pi_overlay("box/big-model")[0]).read()
  check("a remote pi overlay keeps the rest of its host's models, the launched one first",
        source.index('"big-model"') < source.index('"odd-state"') and "omarchy-local\"," not in source)
  # Qwen Code is launched with the endpoint and model, and reads the key itself.
  with patch.object(a, "fetch_listing", return_value=("online", clean, "")):
    path, provider, model = a.qwen_overlay("box/big-model")
  qwen = json.loads(open(path).read())
  check("a Qwen Code overlay carries the endpoint, model, limits and key file, never the key",
        (provider, model) == ("omarchy-local-box", "big-model") and qwen["baseUrl"] == "https://box.tail.ts.net/local-ai/big-model/v1"
        and qwen["model"] == "big-model" and qwen["maxTokens"] == 262144
        and qwen["keyFile"] == a.remote_key_path("box") and qwen["contextWindow"] == 262144 and "b" * 32 not in open(path).read())
  with patch.object(a, "survey_network", return_value={"hosts": {}, "models": {}}):
    a.write_network()
  check("a network write syncs pi's extension, dropping models it no longer sees", not os.path.exists(a.pi_extension_path()))
check("the overlay command offers Qwen Code", "qwen" in a.OVERLAYS)

# Sharing: routes follow shares.json, the listing never carries the key.
handlers = {}
def fake_tailscale(*args, timeout=10):
  if args[:2] == ("serve", "status"):
    return SimpleNamespace(returncode=0, stdout=json.dumps({"Web": {"me.tail.ts.net:443": {"Handlers": dict(handlers)}}}), stderr="")
  path, target = args[-2], args[-1]
  if target == "off":
    handlers.pop(path, None)
  else:
    handlers[path] = {"Text": target[5:]} if target.startswith("text:") else {"Proxy": target}
  return SimpleNamespace(returncode=0, stdout="", stderr="")
share_model = dict(manifest, recipe_id="current")
a.save_model(share_model, started=False)
with patch.object(a, "known_models", return_value=["current"]), patch.object(a, "unit_props", return_value={"ActiveState": "active"}), patch.object(a, "free_memory", return_value=(64, 48)), \
     patch.object(a, "tailscale", side_effect=fake_tailscale), patch.object(a, "tailscale_status", return_value=ts_status), \
     patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0, stderr="")) as proxy_systemctl:
  record = a.write_status(cat, profile)
  check("nothing touches Tailscale Serve until a model is shared", handlers == {} and not record["models"]["current"]["shared"])
  a.save_shares(["current"])
  record = a.write_status(cat, profile)
  served = json.loads(handlers["/local-ai/models.json"]["Text"])
  check("a shared model is served through the share proxy", handlers["/local-ai/current"] == {"Proxy": f"http://127.0.0.1:{a.SHARE_PROXY_PORT}/current"} and record["models"]["current"]["shared"])
  check("sharing starts the share proxy", ("start", a.SHARE_PROXY_UNIT) in [call.args for call in proxy_systemctl.call_args_list])
  check("the listing names the model and its state, never the key", served["models"][0]["recipe"] == "current" and served["models"][0]["unit"] == "active" and "key" not in served and "key" not in served["models"][0])
  handlers["/other"] = {"Proxy": "http://127.0.0.1:9"}
  a.save_shares([])
  a.write_status(cat, profile)
  check("unsharing removes only our routes", handlers == {"/other": {"Proxy": "http://127.0.0.1:9"}} and not os.path.exists(a.published_path()))
  check("unsharing the last model stops the share proxy", proxy_systemctl.call_args_list[-1].args == ("stop", a.SHARE_PROXY_UNIT))
def refuse(*args, timeout=10):
  if args[:2] == ("serve", "status"):
    return SimpleNamespace(returncode=0, stdout="{}", stderr="")
  return SimpleNamespace(returncode=1, stdout="", stderr="sending serve config: 401 Unauthorized: must be root, or be an operator")
with patch.object(a, "known_models", return_value=["current"]), patch.object(a, "unit_props", return_value={"ActiveState": "active"}), patch.object(a, "free_memory", return_value=(64, 48)), \
     patch.object(a, "tailscale", side_effect=refuse), patch.object(a, "tailscale_status", return_value=ts_status), \
     patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0, stderr="")):
  a.save_shares(["current"])
  record = a.write_status(cat, profile)
  check("a refused share is reported with the fix", record["share_error"] == a.OPERATOR_HINT)
a.save_shares([])
# On a circle alone, the share proxy answers itself on this machine's circle addresses.
with patch.object(a, "known_models", return_value=["current"]), patch.object(a, "unit_props", return_value={"ActiveState": "active"}), patch.object(a, "free_memory", return_value=(64, 48)), \
     patch.object(a, "tailscale", side_effect=AssertionError("no Tailscale")), patch.object(a, "tailscale_status", return_value=None), \
     patch.object(a, "circle_status", return_value={"phase": "running", "self": {"name": "me", "addresses": ["10.37.0.3", "fd7a::3"]}}), \
     patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0, stderr="")):
  a.save_shares(["current"])
  record = a.write_status(cat, profile)
  check("on a circle the share proxy listens on its addresses and publishes the listing",
        "--listen 10.37.0.3 --listen fd7a::3" in open(a.share_proxy_unit_path()).read() and json.loads(open(a.published_path()).read())["models"][0]["recipe"] == "current" and record["share_error"] == "")
  a.save_shares([])
  a.write_status(cat, profile)

# The share proxy: a request the model accepts counts its machine for the
# window, a refused one doesn't, and a streamed answer arrives as it's sent.
import http.client
import http.server
import threading

release = threading.Event()

class FakeModel(http.server.BaseHTTPRequestHandler):
  protocol_version = "HTTP/1.1"
  def log_message(self, *args):
    pass
  def do_POST(self):
    if self.headers.get("Transfer-Encoding") == "chunked":
      body = b""
      while size := int(self.rfile.readline().strip(), 16):
        body += self.rfile.read(size)
        self.rfile.readline()
      self.rfile.readline()
    else:
      body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
    if self.headers.get("Authorization") != "Bearer k":
      self.send_response(401)
      self.send_header("Content-Length", "0")
      self.end_headers()
      return
    self.send_response(200)
    self.send_header("Content-Type", "text/event-stream")
    self.send_header("Transfer-Encoding", "chunked")
    self.end_headers()
    for index, part in enumerate((b"data: " + body + b"\n\n", b"data: [DONE]\n\n")):
      # The second part waits until the first has reached the client, so a
      # proxy that held the answer back until it ended would never finish.
      if index:
        release.wait(10)
      self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
      self.wfile.flush()
    self.wfile.write(b"0\r\n\r\n")

def serve(handler):
  server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
  server.daemon_threads = True
  threading.Thread(target=server.serve_forever, daemon=True).start()
  return server

now = [1_000_000]
tracker = a.ClientTracker(a.clients_path(), window_ms=1000, clock=lambda: now[0])
model = serve(FakeModel)
proxy = serve(a.make_share_proxy(tracker, ports=lambda: {"shared": model.server_address[1], "private": model.server_address[1]}, shares=lambda: ["shared"]))

def ask(path, key="k", client="100.64.0.2"):
  connection = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=10)
  connection.request("POST", path, body=b"hello", headers={"Authorization": f"Bearer {key}", "X-Forwarded-For": client, "Content-Type": "application/json"})
  answer = connection.getresponse()
  body = answer.read()
  connection.close()
  return answer.status, body

def counts():
  return json.loads(open(a.clients_path()).read())["models"]

connection = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=10)
connection.request("POST", "/shared/v1/chat/completions", body=iter([b"hel", b"lo"]), encode_chunked=True, headers={"Authorization": "Bearer k", "X-Forwarded-For": "100.64.0.2"})
answer = connection.getresponse()
first = answer.read1(65536)
release.set()
rest = answer.read()
connection.close()
check("the proxy streams each part of the answer as the model sends it", answer.status == 200 and first == b"data: hello\n\n" and rest == b"data: [DONE]\n\n", f"{answer.status} {first!r} {rest!r}")
release.clear()
release.set()
status, body = ask("/shared/v1/chat/completions")
check("a request with a length passes its body through", status == 200 and body == b"data: hello\n\ndata: [DONE]\n\n", f"{status} {body!r}")
check("an accepted request counts its machine", counts() == {"shared": 1})
ask("/shared/v1/chat/completions")
check("the same machine counts once", counts() == {"shared": 1})
status, _ = ask("/shared/v1/chat/completions", key="wrong", client="100.64.0.3")
check("a request the model refuses doesn't count", status == 401 and counts() == {"shared": 1})
ask("/shared/v1/chat/completions", client="100.64.0.3")
check("a second machine counts as a second user", counts() == {"shared": 2})
status, _ = ask("/private/v1/chat/completions", client="100.64.0.4")
check("a model that isn't shared isn't reachable through the proxy", status == 404 and counts() == {"shared": 2})
now[0] += 600
ask("/shared/v1/chat/completions", client="100.64.0.3")
now[0] += 500
check("a machine idle past the window stops counting", tracker.expire() is not None and counts() == {"shared": 1})
now[0] += 600
check("once everyone is idle nothing counts and nothing is pending", tracker.expire() is None and counts() == {})
# Facing the circle, the proxy answers the routes Tailscale Serve would, and
# the caller is whoever connected, whatever header it sends.
facing = a.circle_server("127.0.0.1", a.make_share_proxy(tracker, ports=lambda: {"shared": model.server_address[1]}, shares=lambda: ["shared"], mesh=True), port=0)
threading.Thread(target=facing.serve_forever, daemon=True).start()
def ask_circle(method, path):
  connection = http.client.HTTPConnection("127.0.0.1", facing.server_address[1], timeout=10)
  connection.request(method, path, body=b"hello" if method == "POST" else None, headers={"Authorization": "Bearer k", "X-Forwarded-For": "100.64.0.9"})
  answer = connection.getresponse()
  body = answer.read()
  connection.close()
  return answer.status, body
a.write_atomic(a.published_path(), '{"schema":1,"models":[]}\n')
check("the circle gets the listing at /local-ai/models.json", ask_circle("GET", "/local-ai/models.json") == (200, b'{"schema":1,"models":[]}'))
check("the circle reaches a shared model under /local-ai only", ask_circle("POST", "/local-ai/shared/v1/chat/completions")[0] == 200 and ask_circle("POST", "/shared/v1/chat/completions")[0] == 404)
check("a circle caller counts by its own address, not a header it sends", tracker.seen["shared"].keys() == {"127.0.0.1"}, repr(tracker.seen))
os.unlink(a.published_path())
facing.shutdown()
proxy.shutdown()
model.shutdown()

PY
