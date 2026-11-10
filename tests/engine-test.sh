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
               pair={"nodes": 2, "master_port": 29620, "prepare": ["prep", "@RANK@", "@WORLD@"], "worker": ["lanes", "@RANK@", "@MASTER_ADDR@"],
                     "env": {"TF_GID": "@GID_INDEX@"}, "head_env": {"TF_MASTER": "@MASTER_ADDR@"}})
check("a paired runtime fixture is accepted", not a.validate_runtime("pair", pair_rt), "\n".join(a.validate_runtime("pair", pair_rt)))
for name, mutate, fragment in (
  ("host network without a pair", lambda r: r.pop("pair"), "only for runtimes with a [pair]"),
  ("a pair on the internal network", lambda r: r["run"].pop("network"), "needs run.network"),
  ("a pair without bind flags", lambda r: r.pop("listen"), ".listen"),
  ("bind flags in the command", lambda r: r.update(command=["server", "--port", "1"]), "sets --host, --port itself"),
  ("unreserved bind flags", lambda r: r.update(reserved=["API_KEY"]), "must reserve"),
  ("NCCL settings in the runtime", lambda r: r["pair"]["env"].update(NCCL_IB_HCA="mlx5_0"), "NCCL_"),
  ("three nodes", lambda r: r["pair"].update(nodes=3), "only 2"),
  ("a served folder outside /prepared", lambda r: r["pair"].update(model="/prepared/../etc"), "under /prepared"),
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
check("the pair unit mounts rank 0's inputs and reads the link from its environment file", f"models/paired/{a.rank_folder('b' * 16, 0, pair_rt)}:/prepared:ro" in pair_text and "EnvironmentFile=%h/.local/state/omarchy/agent-local/pair/paired.env" in pair_text)
check("the pair unit starts and stops rank 1 with itself", "ExecStartPre=/usr/bin/env omarchy-agent-local link-worker start paired" in pair_text and "ExecStopPost=-/usr/bin/env omarchy-agent-local link-worker stop paired" in pair_text)
check("a paired runtime may serve its prepared folder", '"--model" "/prepared/model"' in a.render_container("paired", paired, dict(pair_rt, pair=dict(pair_rt["pair"], model="/prepared/model")), paired["variant"][0], {"model": "model"}, pair_gpu, "b" * 16))
same_rt = copy.deepcopy(pair_rt)
for key in ("worker", "prepare"):
  same_rt["pair"].pop(key)
same_rt["pair"].update(state="/state", env={"NCCL_COMM_ID": "@MASTER_ADDR@:@MASTER_PORT@", "TF_RANK": "@RANK@"})
check("a paired runtime may run the head's command on both ranks, with its own NCCL settings", not a.validate_runtime("same", same_rt), "\n".join(a.validate_runtime("same", same_rt)))
same_text = a.render_container("paired", paired, same_rt, paired["variant"][0], {"model": "model"}, pair_gpu, "b" * 16)
check("a state path mounts the rank folder writable", f"models/paired/{'b' * 16}.rank0:/state\n" in same_text)
_, same_argv, _, _ = a.head_argv(same_rt, paired, paired["variant"][0], {"model": "model"}, "paired")
check("rank 1's default command is the head's, bound to loopback", same_argv[-4:] == ["--host", "127.0.0.1", "--port", str(a.port_for("paired"))] and same_argv[:2] == ["server", "serve"])
# Each rank gets its own link settings, and rank 1 the head's command inside the same restrictions.
setup = {"schema": 1, "recipe_id": "paired", "runtime": same_rt, "recipe": paired, "variant": paired["variant"][0], "roles": {"model": "model"},
         "fileset": "e" * 16, "pair": {"peer": "bob@10.99.1.2", "host": "spark2", "home": "/home/bob", "image_id": "f" * 64}}
a.write_pair_setup(setup)
here = {"addr": "10.99.1.1", "dev": "enp1", "hcas": ["rocep1", "roceP2"], "gid": "5"}
there = {"addr": "10.99.1.2", "dev": "enp1", "hcas": ["rocep1", "roceP2"], "gid": "3"}
with patch.object(a, "pair_link", return_value=(here, there)), patch.object(a, "peer_headroom") as headroom, patch.object(a, "pair_ssh") as ssh, patch.object(a, "pair_python"):
  a.pair_worker_start("paired")
head_env = dict(line.split("=", 1) for line in Path(a.pair_env_path("paired")).read_text().splitlines())
worker = shlex.split(ssh.call_args.args[1])
worker_env = dict(worker[i + 1].split("=", 1) for i, word in enumerate(worker) if word == "-e")
check("rank 0's environment is rank 0's side of the link", head_env == {"NCCL_COMM_ID": "10.99.1.1:29620", "TF_RANK": "0", "TF_MASTER": "10.99.1.1", "NCCL_SOCKET_IFNAME": "enp1", "NCCL_IB_HCA": "rocep1,roceP2", "NCCL_IB_GID_INDEX": "5"}, repr(head_env))
check("rank 1's environment is rank 1's side, same rendezvous", worker_env["TF_RANK"] == "1" and worker_env["NCCL_COMM_ID"] == "10.99.1.1:29620" and worker_env["NCCL_IB_GID_INDEX"] == "3" and "API_KEY" not in worker_env, repr(worker_env))
check("rank 1 starts on the peer as its user, with the head's command and the same restrictions",
      ssh.call_args.args[0] == "bob@10.99.1.2" and headroom.called
      and all(flag in worker for flag in ("--read-only", "no-new-privileges", "/dev/infiniband", "f" * 64))
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
  a.pair_setup("paired", manifest, same_rt, {"peer": "bob@10.99.1.2", "host": "spark2", "home": "/home/bob"})
rsync = next(argv for argv in ran if argv[0] == "rsync")
check("setup records the peer and the image by ID", manifest["pair"] == {"peer": "bob@10.99.1.2", "host": "spark2", "home": "/home/bob", "image_id": "f" * 64})
check("setup copies the file set to the peer, not an image it already has", not save.called and rsync[-1] == f"bob@10.99.1.2:/home/bob/.local/share/omarchy/agent-local/models/paired/{'e' * 16}/")
check("setup makes both ranks' state folders", Path(a.rank_dir("paired", "e" * 16, 0)).is_dir() and any(f"{'e' * 16}.rank1" in call.args[1] for call in ssh.call_args_list))
# Prepared files are a generation per image and prepare command, so an update never rewrites the running setup's.
updated = dict(pair_rt, image=pair_rt["image"].replace("a" * 64, "b" * 64))
check("a runtime update prepares into its own generation", a.rank_folder("e" * 16, 0, pair_rt) != a.rank_folder("e" * 16, 0, updated) and a.rank_folder("e" * 16, 0, same_rt) == "e" * 16 + ".rank0")
base = Path(a.data_dir(), "models", "gen")
old_gen, new_gen = a.rank_folder("e" * 16, 0, pair_rt), a.rank_folder("e" * 16, 0, updated)
for name in ("e" * 16, old_gen, new_gen):
  (base / name).mkdir(parents=True, exist_ok=True)
with patch.object(a, "load_model", return_value={"runtime": updated}):
  a.prune_filesets("gen", "e" * 16)
check("pruning keeps only the committed setup's generation", sorted(p.name for p in base.iterdir()) == sorted(["e" * 16, new_gen]))
# Prepare runs once per rank, without network, and a failure stops the setup.
ran = []
def fake_prepare(argv, **kwargs):
  ran.append(argv)
  return SimpleNamespace(returncode=0, stdout="f" * 64 + "\n", stderr="")
with patch.object(a.subprocess, "run", side_effect=fake_prepare), patch.object(a.subprocess, "Popen"), \
     patch.object(a, "pair_ssh", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")), patch.object(a, "load_installing", return_value=None):
  a.pair_setup("paired", {"fileset": "e" * 16}, pair_rt, {"peer": "10.99.1.2", "host": "spark2", "home": "/home/bob"})
prepares = [argv for argv in ran if any(word == "prep" or " prep " in word for word in argv)]
check("prepare runs for rank 0 here and rank 1 on the peer, without network",
      len(prepares) == 2 and prepares[0][-3:] == ["prep", "0", "2"] and "none" in prepares[0]
      and "prep 1 2" in " ".join(prepares[1]) and "--network none" in " ".join(prepares[1]), repr(prepares))
with patch.object(a.subprocess, "run", side_effect=lambda argv, **k: SimpleNamespace(returncode=1 if "prep" in argv else 0, stdout="f" * 64, stderr="")), \
     patch.object(a, "pair_ssh", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")), patch.object(a, "load_installing", return_value=None):
  raises("a failed prepare stops the setup", lambda: a.pair_setup("paired", {"fileset": "e" * 16}, pair_rt, {"peer": "10.99.1.2", "host": "spark2", "home": "/home/bob"}), "Preparing rank 0")
# Disk for the prepared files is checked on both Sparks before anything is written, and
# prepare containers carry names remove can stop.
big = dict(pair_rt, pair=dict(pair_rt["pair"], prepared_gib=100))
ran.clear()
with patch.object(a.subprocess, "run", side_effect=fake_prepare), patch.object(a, "load_installing", return_value=None), \
     patch.object(a.shutil, "disk_usage", return_value=SimpleNamespace(free=500 * 2**30)), \
     patch.object(a, "pair_ssh", return_value=SimpleNamespace(returncode=0, stdout="%d\n0\n" % (50 * 2**30), stderr="")) as disk:
  raises("a peer without room for the files and prepared inputs stops setup first", lambda: a.pair_setup("paired", {"fileset": "e" * 16, "files": [{"size": 10 * 2**30}]}, big, {"peer": "10.99.1.2", "host": "spark2", "home": "/home/bob"}), "spark2 has 50 GiB free")
measured = disk.call_args_list[0].args[1].split("du -sbc ", 1)[1].split(" 2>")[0]
check("the peer credits only the file set and the current generation as already there",
      shlex.split(measured) == [f"/home/bob/.local/share/omarchy/agent-local/models/paired/{'e' * 16}", a.rank_dir("paired", "e" * 16, 1, "/home/bob", big)], measured)
check("nothing is prepared or copied when the disk check fails", not [argv for argv in ran if "prep" in argv or argv[:1] == ["rsync"]], repr(ran))
check("prepare containers are named for their recipe and rank", "--name" in prepares[0] and a.prepare_name("paired", 0) in prepares[0] and a.prepare_name("paired", 1) in " ".join(prepares[1]))
with patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as local, patch.object(a, "pair_ssh") as remote, patch.object(a, "known_models", return_value=[]):
  a.pair_remove({"recipe_id": "paired", "pair": {"peer": "10.99.1.2", "host": "spark2", "home": "/home/bob", "image_id": "c" * 64}})
check("remove stops a prepare a cancelled setup left running, here and on the peer",
      any(call.args[0] == ["podman", "rm", "-f", a.prepare_name("paired", 0)] for call in local.call_args_list) and a.prepare_name("paired", 1) in remote.call_args.args[1])
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

PY
