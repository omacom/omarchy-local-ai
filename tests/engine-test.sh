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
import tempfile
import tomllib
from types import SimpleNamespace
from unittest.mock import mock_open, patch

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
  "command": ["server"], "reserved": ["--host", "API_KEY"], "gpus": ["amd"], "run": {"devices": ["render"]},
  "inputs": {"model": {"arg": "--model"}, "draft": {"arg": "--draft"},
             "ctx": {"arg": "--ctx"}, "alias": {"arg": "--alias"}, "key": {"env": "API_KEY"}},
}
variant = {"ctx": 4096, "peak_ram_gib": 16, "peak_gpu_gib": 16}
artifact = {"repo": "test/model", "revision": "b" * 40, "license": "mit",
            "files": ["model.gguf", "tokenizer/*.json"], "model": "model.gguf"}
recipe = {"name": "Test model", "runtime": "server",
          "hardware": [{"gpu": "amd-strix-halo", "measured": "measurements/model.toml"}],
          "artifacts": [artifact], "variant": [variant]}
gpu = {"vendor": "amd", "family": "amd-strix-halo", "memory_gib": 48, "render_node": "/dev/dri/renderD128"}
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
check("hardware rejects ram_class", any("per chip" in e for e in a.validate_recipe("model", dict(recipe, hardware=[{"gpu": "amd-strix-halo", "ram_class": 64, "measured": "measurements/model.toml"}]), {"server": runtime})))
check("hardware rejects duplicate chip entries", any("duplicate measurement" in e for e in a.validate_recipe("model", dict(recipe, hardware=[{"gpu": "amd-strix-halo", "measured": "measurements/model.toml"}, {"gpu": "amd-strix-halo", "measured": "measurements/model.toml"}]), {"server": runtime})))
check("hardware rejects non-string platform without crashing", any(".platform" in e for e in a.validate_recipe("model", dict(recipe, hardware=[{"gpu": "amd-strix-halo", "platform": [], "measured": "measurements/model.toml"}]), {"server": runtime})))
check("identical best selectors are rejected", any("same selectors" in e for e in a.validate_best({"class": [entry, entry]}, {"model": recipe})))
check("best accepts another RAM class for a measured GPU", not a.validate_best({"class": [dict(entry, ram_class=128)]}, {"model": recipe}))
check("best cannot recommend on an unmeasured GPU", any("no measured" in e for e in a.validate_best({"class": [dict(entry, gpu="nvidia-gb10")]}, {"model": recipe})))

catalog = SimpleNamespace(recipes={"model": copy.deepcopy(recipe)}, runtimes={"server": runtime}, best={"class": [entry]})
check("best requires the same GPU", a.match_class({"class": [entry]}, profile, dict(gpu, family="other")) is None)
check("a smaller machine doesn't take a bigger one's best", a.match_class({"class": [dict(entry, ram_class=128)]}, profile, gpu) is None)
check("a bigger machine takes the nearest smaller class's best", a.match_class({"class": [dict(entry, ram_class=32), entry]}, dict(profile, ram_class=192), gpu)["ram_class"] == 64)
check("a bigger machine keeps its own best", a.match_class({"class": [entry, dict(entry, ram_class=128, recipes=["other"])]}, dict(profile, ram_class=128), gpu)["recipes"] == ["other"])
check("a kind a bigger class doesn't list comes from a smaller one", a.match_class({"class": [dict(entry, music=["song"]), dict(entry, ram_class=128, recipes=["other"])]}, dict(profile, ram_class=128), gpu, "music")["music"] == ["song"])
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
                    ("a runtime that doesn't run on this GPU", dict(recipe, runtime="nvidia-only"))):
  catalog.runtimes["nvidia-only"] = dict(runtime, run={"devices": ["cdi-nvidia"]}, gpus=["nvidia"])
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
with patch.object(a, "platform_name", return_value="aarch64-n1x"), patch.object(a, "amd_gpus", return_value=[]), patch.object(a, "intel_gpus", return_value=[]), patch.object(a, "meminfo", return_value={"MemTotal": 120 * 2**30}):
  n1x = a.machine_profile()
check("an N1x laptop has a unified NVIDIA GPU of its own family", n1x["ram_class"] == 128 and [(g["vendor"], g["family"], g["memory_gib"]) for g in n1x["gpus"]] == [("nvidia", "nvidia-n1x", 120)])
spark = dict(recipe, hardware=[{"gpu": "nvidia-gb10", "measured": "measurements/model.toml"}])
check("an N1x runs what a GB10 with the same memory runs", a.runtime_supports(dict(runtime, gpus=["nvidia-gb10"]), n1x["gpus"][0]) and a.match_class({"class": [dict(entry, gpu="nvidia-gb10", ram_class=128)]}, n1x, n1x["gpus"][0]) is not None)
check("best treats the N1x and GB10 as one class", any("same selectors" in e for e in a.validate_best({"class": [dict(entry, gpu="nvidia-gb10", ram_class=128, recipes=["spark"]), dict(entry, gpu="nvidia-n1x", ram_class=128, recipes=["spark"])]}, {"spark": spark})))
# Families that share a class may each have a measurement: a machine shows its own family's,
# and one chip measurement applies across every RAM class.
shared = Path(os.environ["HOME"]) / "shared-measurements"
shared.mkdir()
for family, speed in (("gb10", 60.0), ("n1x", 50.0), ("halo", 73.3)):
  (shared / f"{family}.toml").write_text(f"[long]\ndecode_tok_s = {speed}\n")
both_measured = {"hardware": [{"gpu": "nvidia-gb10", "measured": "gb10.toml"}, {"gpu": "nvidia-n1x", "measured": "n1x.toml"}]}
speeds = [a.load_measurement(SimpleNamespace(root=str(shared)), recipe, {"ram_class": 128, "platform": "aarch64"}, {"family": family})["long"]["decode_tok_s"]
          for recipe, family in ((both_measured, "nvidia-n1x"), (both_measured, "nvidia-gb10"), ({"hardware": both_measured["hardware"][:1]}, "nvidia-n1x"))]
check("a machine shows its own family's measurement, else its class's", speeds == [50.0, 60.0, 60.0], str(speeds))
halo_measured = {"hardware": [{"gpu": "amd-strix-halo", "measured": "halo.toml"}]}
chip_speeds = [
  a.load_measurement(SimpleNamespace(root=str(shared)), halo_measured, {"ram_class": ram, "platform": "x86"}, {"family": "amd-strix-halo"})["long"]["decode_tok_s"]
  for ram in (64, 128, 192)
]
check("a chip's measurement applies across every RAM class", chip_speeds == [73.3, 73.3, 73.3], str(chip_speeds))
# An Intel laptop: integrated graphics on the xe driver counts as a GPU.
drm = Path(os.environ["HOME"]) / "intel-drm"
for driver in ("xe", "i915"):
  (drm / "drivers" / driver).mkdir(parents=True)
for address, render, device in (("0000:00:02.0", "renderD128", "0xb080"), ("0000:03:00.0", "renderD129", "0xb080")):
  (drm / "devices" / address).mkdir(parents=True)
  (drm / "devices" / address / "vendor").write_text("0x8086\n")
  (drm / "devices" / address / "device").write_text(device + "\n")
  (drm / "devices" / address / "driver").symlink_to(drm / "drivers" / "xe")
  (drm / render).mkdir()
  (drm / render / "device").symlink_to(drm / "devices" / address)
with patch.object(a, "TTM_PAGES_LIMIT", str(drm / "no-limit")):
  arc = a.intel_gpus(64, str(drm))
check("an Arc B390 is a unified GPU with three quarters of RAM, and a card beside it is not", [(g["family"], g["render_node"], g["memory_gib"]) for g in arc] == [("intel-arc-b390", "/dev/dri/renderD128", 48)])
(drm / "pages-limit").write_text(str(32 * 2**30 // os.sysconf("SC_PAGE_SIZE")))
with patch.object(a, "TTM_PAGES_LIMIT", str(drm / "pages-limit")):
  check("the kernel's page limit caps what an Intel GPU addresses", a.intel_gpus(64, str(drm))[0]["memory_gib"] == 32)
with patch.object(a, "meminfo", return_value={"MemAvailable": 20 * 2**30, "GPUReclaim": 28 * 2**30, "GPUActive": 2 * 2**30}):
  check("memory a GPU driver pooled counts as free, and what it holds as used", a.free_memory(arc[0]) == (48, 46))
(drm / "devices" / "0000:00:02.0" / "device").write_text("0x6420\n")
with patch.object(a, "TTM_PAGES_LIMIT", str(drm / "no-limit")):
  lunar = a.intel_gpus(64, str(drm))
check("integrated graphics on xe with no name of its own is named by its device id", [g["family"] for g in lunar] == ["intel-xe-6420"])
(drm / "devices" / "0000:00:02.0" / "driver").unlink()
(drm / "devices" / "0000:00:02.0" / "driver").symlink_to(drm / "drivers" / "i915")
check("integrated graphics on the older i915 driver is no GPU", a.intel_gpus(64, str(drm)) == [])

# Weak display-only AMD iGPUs (< 8 CUs, e.g. Radeon 610M / gfx1036) are excluded, while
# compute-capable APUs and discrete AMD cards remain.
kfd = Path(os.environ["HOME"]) / "amd-kfd"
amd_drm = Path(os.environ["HOME"]) / "amd-drm"
for idx, (ver, minor, simd, per_cu, vram_gb, gtt_gb) in enumerate((
    (100306, 128, 4, 2, 0.5, 22.63),    # Radeon 610M (2 CUs, unified) -> excluded
    (110003, 129, 24, 2, 0.5, 22.63),   # Radeon 780M (12 CUs, unified) -> kept
    (110501, 130, 80, 2, 0.5, 60.0),    # Strix Halo (40 CUs, unified) -> kept
    (110000, 131, 4, 2, 24.0, 32.0),    # Discrete AMD card (VRAM >= 4 GiB) -> kept
)):
  node = kfd / str(idx + 1)
  node.mkdir(parents=True)
  (node / "properties").write_text(f"gfx_target_version {ver}\ndrm_render_minor {minor}\nsimd_count {simd}\nsimd_per_cu {per_cu}\n")
  dev = amd_drm / f"renderD{minor}" / "device"
  dev.mkdir(parents=True)
  (dev / "mem_info_vram_total").write_text(f"{int(vram_gb * 2**30)}\n")
  (dev / "mem_info_gtt_total").write_text(f"{int(gtt_gb * 2**30)}\n")
found_amd = a.amd_gpus(str(kfd), str(amd_drm))
check("display-only AMD iGPUs (< 8 CUs) are excluded while APUs and discrete cards stay",
      [g["family"] for g in found_amd] == ["amd-gfx1103", "amd-strix-halo", "amd-gfx1100"], str([g["family"] for g in found_amd]))

# Discrete NVIDIA GPUs on x86: detected via nvidia-smi, classed by vram_class, and outrank iGPUs.
smi_rows = [
    "0, NVIDIA GeForce RTX 5070 Laptop GPU, 8151, 12.0, GPU-1111",
    "1, NVIDIA GeForce RTX 4090, 24576, 8.9, GPU-2222",
    "2, NVIDIA GeForce RTX 5090, 32607, 12.0, GPU-3333",
]
with patch.object(a, "_nvidia_smi", return_value=smi_rows):
  nv_cards = a.nvidia_gpus()
check("discrete NVIDIA GPUs parse name, VRAM, vram_class and unified=False",
      [(g["family"], g["vram_class"], g["unified"]) for g in nv_cards] == [
          ("nvidia-rtx5070-laptop", 8, False),
          ("nvidia-rtx4090", 24, False),
          ("nvidia-rtx5090", 32, False),
      ], str(nv_cards))
with patch.object(a, "platform_name", return_value="aarch64-gb10"), patch.object(a, "amd_gpus", return_value=[]), \
     patch.object(a, "intel_gpus", return_value=[]), patch.object(a, "nvidia_gpus", side_effect=AssertionError("unified platform should not probe discrete nvidia")), \
     patch.object(a, "meminfo", return_value={"MemTotal": 120 * 2**30}):
  check("unified NVIDIA platforms do not also probe discrete NVIDIA cards", [g["family"] for g in a.machine_profile()["gpus"]] == ["nvidia-gb10"])

igpu_big_gtt = {"vendor": "amd", "family": "amd-gfx1103", "render_node": "/dev/dri/renderD128", "unified": True, "memory_gib": 54.4}
dgpu_card = {"vendor": "amd", "family": "amd-gfx1100", "render_node": "/dev/dri/renderD129", "unified": False, "memory_gib": 24.0}
dual_amd = {"platform": "x86", "ram_gib": 64, "ram_class": 64, "gpus": [igpu_big_gtt, dgpu_card]}
small_img = {"kind": "image", "variant": [{"peak_ram_gib": 11, "peak_gpu_gib": 11}]}
big_vid = {"kind": "video", "variant": [{"peak_ram_gib": 38, "peak_gpu_gib": 38}]}
amd_rt = {"gpus": ["amd"]}
check("recipe_gpu prefers a discrete GPU when it fits, and falls back to a larger fitting integrated GPU",
      a.recipe_gpu(dual_amd, small_img, runtime=amd_rt) == dgpu_card
      and a.recipe_gpu(dual_amd, big_vid, runtime=amd_rt) == igpu_big_gtt)
dual_cat = SimpleNamespace(
  recipes={"big-vid": dict(big_vid, runtime="amd-rt", hardware=[{"gpu": "amd-gfx1103", "ram_class": 64, "measured": "m.toml"}])},
  runtimes={"amd-rt": amd_rt},
  best={"class": [{"gpu": "amd-gfx1103", "ram_class": 64, "recipes": ["big-vid"], "videos": ["big-vid"]}]},
)
check("resolve_best preserves the fitting GPU it selected",
      a.resolve_best(dual_cat, dual_amd, field="videos")[3] == igpu_big_gtt)
dgpu_5070 = nv_cards[0]
dual_profile = {"platform": "x86", "ram_gib": 30.63, "ram_class": 32, "gpus": [igpu_big_gtt, dgpu_5070]}
check("a discrete GPU outranks an integrated GPU even when GTT exceeds VRAM", a.choose_gpu(dual_profile) == dgpu_5070)
pair_gpu = {"vendor": "nvidia", "family": "nvidia-gb10-pair", "nodes": 2, "unified": True, "memory_gib": 120}
check("a paired unified GPU still outranks a single discrete GPU", a.choose_gpu({"platform": "aarch64-gb10", "ram_gib": 120, "ram_class": 128, "gpus": [dgpu_5070, pair_gpu]}) == pair_gpu)

with patch.object(a, "_nvidia_smi", return_value=["7168"]), patch.object(a, "meminfo", return_value={"MemAvailable": 20 * 2**30}):
  check("free memory on a discrete NVIDIA GPU reads free VRAM from nvidia-smi", a.free_memory(dgpu_5070) == (20.0, 7.0))

nv_cg = Path(os.environ["HOME"]) / "nv-cg"
(nv_cg / "unit").mkdir(parents=True)
(nv_cg / "unit" / "memory.stat").write_text(f"anon {2**30}\n")
(nv_cg / "unit" / "cgroup.procs").write_text("424242\n")
with patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="424242, 4096\n999999, 8192\n")):
  check("held_bytes includes discrete NVIDIA VRAM held by processes in the cgroup",
        a.held_bytes("/unit", str(nv_cg), str(nv_cg / "no-proc")) == 5 * 2**30)

cdi_rt = {"run": {"devices": ["cdi-nvidia"]}}
check("device_lines binds the selected discrete NVIDIA card by UUID and keeps all for unified NVIDIA",
      a.device_lines(cdi_rt, nv_cards[0]) == ["AddDevice=nvidia.com/gpu=GPU-1111"]
      and a.device_lines(cdi_rt, nv_cards[1]) == ["AddDevice=nvidia.com/gpu=GPU-2222"]
      and a.device_lines(cdi_rt, {"vendor": "nvidia", "family": "nvidia-gb10", "unified": True}) == ["AddDevice=nvidia.com/gpu=all"])

# Discrete vram_class matching, measurement lookup and validation.
vram_entry = {"gpu": "nvidia-rtx4090", "vram_class": 24, "recipes": ["dgpu-model"]}
vram_recipe = dict(recipe, runtime="cuda", hardware=[{"gpu": "nvidia-rtx4090", "measured": "gb10.toml"}])
check("vram_class matches a discrete GPU regardless of host ram_class, inheriting smaller classes only",
      a.match_class({"class": [vram_entry]}, {"ram_class": 64, "platform": "x86"}, nv_cards[1]) == vram_entry
      and a.match_class({"class": [dict(vram_entry, vram_class=16)]}, {"ram_class": 64, "platform": "x86"}, nv_cards[1]) == dict(vram_entry, vram_class=16)
      and a.match_class({"class": [dict(vram_entry, vram_class=32)]}, {"ram_class": 64, "platform": "x86"}, nv_cards[1]) is None)
check("load_measurement resolves per-chip entries for discrete GPUs",
      a.load_measurement(SimpleNamespace(root=str(shared)), vram_recipe, {"ram_class": 64, "platform": "x86"}, nv_cards[1]).get("long", {}).get("decode_tok_s") == 60.0)
check("best entries require exactly one of ram_class or vram_class, and hardware rejects vram_class",
      not a.validate_best({"class": [vram_entry]}, {"dgpu-model": vram_recipe})
      and any("ram_class" in e and "vram_class" in e for e in a.validate_best({"class": [dict(vram_entry, ram_class=64)]}, {"dgpu-model": vram_recipe}))
      and any("vram_class" in e for e in a.validate_recipe("m", dict(vram_recipe, hardware=[{"gpu": "nvidia-rtx4090", "vram_class": 24, "measured": "gb10.toml"}]), {"cuda": dict(runtime, gpus=["nvidia"], run={"devices": ["cdi-nvidia"]})})))
check("a runtime runs on the vendors and classes it lists, and no other",
      a.runtime_supports(dict(runtime, gpus=["intel"]), lunar[0]) and a.runtime_supports(dict(runtime, gpus=["intel-arc-b390"]), arc[0])
      and not a.runtime_supports(dict(runtime, gpus=["intel-arc-b390"]), lunar[0]) and not a.runtime_supports(dict(runtime, gpus=["amd"]), arc[0]))
check("a recipe nobody measured on this machine is available when its runtime runs here and it fits",
      a.compatibility(SimpleNamespace(recipes={"model": recipe}, runtimes={"server": dict(runtime, gpus=["intel"])}), "model", profile, arc[0])[0] is not None)
check("a runtime's gpus must be vendors or GPU classes its devices reach",
      any("gpus" in e for e in a.validate_runtime("server", dict(runtime, gpus=[])))
      and any("gpus" in e for e in a.validate_runtime("server", dict(runtime, gpus=["arm"])))
      and any("can't be reached" in e for e in a.validate_runtime("server", dict(runtime, gpus=["nvidia-gb10"], run={"devices": ["render"]})))
      and any("can't be reached" in e for e in a.validate_runtime("server", dict(runtime, gpus=["intel"], run={"devices": ["render", "kfd"]})))
      and any("can't be reached" in e for e in a.validate_runtime("server", dict(runtime, gpus=["amd"], run={"devices": []}))))
check("GPU peak exceeding capacity does not fit", not a.fits(dict(variant, peak_gpu_gib=49), profile, gpu)[0])
check("RAM fit preserves desktop reserve", not a.fits(dict(variant, peak_ram_gib=57), profile, gpu)[0])
with patch.object(a, "free_memory", return_value=(1, 1)), patch.object(a, "top_memory_users", return_value="browser (12 GiB), editor (4 GiB)"), patch.object(a.time, "sleep"):
  raises("headroom failure lists largest memory users", lambda: a.check_headroom(variant, gpu), "browser (12 GiB), editor (4 GiB)")

# A 64 GB Strix Halo whose desktop holds 13 GiB: fit kept its reserve, so start needs only the peak.
long_ctx = {"ctx": 262144, "peak_ram_gib": 42, "peak_gpu_gib": 40}
with patch.object(a, "free_memory", return_value=(48.6, 54)), patch.object(a, "choose_gpu", return_value=dict(gpu, memory_gib=54)), patch.object(a, "reclaim_page_cache"):
  check("start counts the desktop's reserve once", a.check_headroom(long_ctx, dict(gpu, memory_gib=54)) == (48.6, 54)
        and a.start_fit({"variant": long_ctx}, profile) == (True, ""))
# Its 48.6 GiB free is 20 not cached and 28.6 of cache: a 42 GiB start reclaims the 22 short and
# a margin, file pages only, and a 16 GiB one leaves the cache alone.
uid = os.getuid()
with patch.object(a, "meminfo", return_value={"MemFree": 16 * 2**30, "GPUReclaim": 4 * 2**30}), patch("builtins.open", mock_open()) as opened:
  a.reclaim_page_cache(16)
  untouched = not opened.called
  a.reclaim_page_cache(42)
  check("a start reclaims only the file cache it is short of, from the user's own cgroup",
        untouched and opened.call_args.args == (f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/memory.reclaim", "w")
        and opened().write.call_args.args == (f"{24 * 2**30} swappiness=0",))
with patch.object(a, "meminfo", return_value={"MemFree": 16 * 2**30}), patch("builtins.open", side_effect=OSError(22, "Invalid argument")):
  check("a kernel that refuses the reclaim leaves the start as it was", a.reclaim_page_cache(42) is None)

# A 64 GB Strix Halo on a stock kernel: the GPU maps half of RAM until use raises the cap.
halo = {"vendor": "amd", "family": "amd-strix-halo", "memory_gib": 31.6, "unified": True}
with patch.object(a, "platform_name", return_value="x86"), patch.object(a, "amd_gpus", return_value=[dict(halo)]), patch.object(a, "intel_gpus", return_value=[]), patch.object(a, "nvidia_gpus", return_value=[]), patch.object(a, "meminfo", return_value={"MemTotal": 62 * 2**30}):
  halo = a.machine_profile()["gpus"][0]
needs_more = dict(variant, peak_gpu_gib=38, peak_ram_gib=38)
check("a unified AMD GPU fits by the cap use can raise it to", halo["max_memory_gib"] == 54 and a.fits(needs_more, profile, halo)[0])
# A 64 GB Arc B390 on a stock kernel: the page limit holds it to half of RAM, Mesa to three quarters.
arc_b390 = {"vendor": "intel", "family": "intel-arc-b390", "memory_gib": 31.15, "unified": True}
with patch.object(a, "platform_name", return_value="x86"), patch.object(a, "amd_gpus", return_value=[]), patch.object(a, "intel_gpus", return_value=[dict(arc_b390)]), patch.object(a, "nvidia_gpus", return_value=[]), patch.object(a, "meminfo", return_value={"MemTotal": 62.3 * 2**30}):
  arc_b390 = a.machine_profile()["gpus"][0]
check("an Intel GPU fits by the three quarters Mesa lets it address once use raises the cap", abs(arc_b390["max_memory_gib"] - 46.7) < 0.1 and a.fits(needs_more, profile, arc_b390)[0]
      and not a.fits(dict(needs_more, peak_gpu_gib=48), profile, arc_b390)[0])
check("a GPU with VRAM of its own fits only by what it has", not a.fits(needs_more, profile, dict(halo, max_memory_gib=a.gpu_memory_ceiling_gib(dict(halo, unified=False), 62)))[0])
page = os.sysconf("SC_PAGE_SIZE")
with patch.object(a, "pending_pages_limit", return_value=54 * 2**30 // page), patch.object(a, "_read", return_value=str(31 * 2**30 // page)), patch.object(a, "choose_gpu", return_value=halo):
  raises("a model past the current cap waits for the reboot that raises it", lambda: a.check_headroom(needs_more, halo), "until a reboot")
  check("the panel is told to reboot, not that memory is short", a.start_fit({"variant": needs_more}, profile) == (False, a.REBOOT_FOR_GPU_MEMORY))
with patch.object(a, "pending_pages_limit", return_value=0), patch.object(a, "_read", return_value=str(31 * 2**30 // page)), patch.object(a, "choose_gpu", return_value=halo):
  raises("with no raise waiting, use is named instead of a reboot", lambda: a.check_headroom(needs_more, halo), "use raises its memory limit")
  check("and the panel isn't told a reboot would help", a.start_fit({"variant": needs_more}, profile) == (False, a.GPU_TOO_SMALL))

# How long a clip a video model makes: MiniMax H3's own line, 38 GiB at 56 frames and 0.155 a frame more.
h3 = {"fps": 24, "max_seconds": 15, "reference_frames": 56, "gib_per_frame": 0.155}
check("a 64 GB Strix Halo's 54 GiB makes up to 6 s, not 10", a.clip_seconds(h3, 38, 54) == 6)
check("a GPU with room for all of it makes the model's longest", a.clip_seconds(h3, 38, 280) == 15)
check("a GPU without room for a second makes none", a.clip_seconds(h3, 38, 30) == 0)
check("a recipe's download size is checked", all(any(".download_gib" in e for e in a.validate_recipe("v", {"kind": "image", "runtime": "x", "download_gib": size}, {}))
                                                  for size in (0, float("nan"), float("inf"))))
check("a recipe's clip table is checked", any(".video.gib_per_frame" in e for e in a.validate_recipe("v", {"kind": "video", "runtime": "x", "video": dict(h3, gib_per_frame=0)}, {}))
      and any(".video:" in e for e in a.validate_recipe("t", {"runtime": "x", "video": h3}, {})))
catalog_h3 = SimpleNamespace(recipes={"h3": {"kind": "video", "video": h3}})
record = {"models": {"h3": {"kind": "video", "unit": "active"}, "klein": {"kind": "image", "unit": "active"}}}
manifests = {"h3": {"variant": {"peak_gpu_gib": 38, "peak_ram_gib": 38}}, "klein": {"variant": {"peak_gpu_gib": 12, "peak_ram_gib": 12}}}
with patch.object(a, "load_model", side_effect=manifests.get), patch.object(a, "choose_gpu", return_value=dict(gpu, memory_gib=54)):
  a.add_clip_lengths(record, catalog_h3, profile)
check("another running model's peak comes off the clip's room", record["models"]["h3"]["max_seconds"] == 3 and "max_seconds" not in record["models"]["klein"])
record = {"models": {"h3": {"kind": "video", "unit": "active"}, "klein": {"kind": "image", "unit": "deactivating"}}}
with patch.object(a, "load_model", side_effect=manifests.get), patch.object(a, "choose_gpu", return_value=dict(gpu, memory_gib=54)):
  a.add_clip_lengths(record, catalog_h3, profile)
check("a model still stopping keeps its memory", record["models"]["h3"]["max_seconds"] == 3)
record = {"models": {"h3": {"kind": "video", "unit": "active"}, "klein": {"kind": "image", "unit": "active"}}}
cards = {"amd-a": dict(gpu, family="amd-a", memory_gib=54), "amd-b": dict(gpu, family="amd-b", memory_gib=24)}
split = {"h3": {"gpu": "amd-a", "variant": {"peak_gpu_gib": 38, "peak_ram_gib": 38}}, "klein": {"gpu": "amd-b", "variant": {"peak_gpu_gib": 12, "peak_ram_gib": 12}}}
with patch.object(a, "load_model", side_effect=split.get), patch.object(a, "choose_gpu", side_effect=lambda profile, family=None: cards[family]):
  a.add_clip_lengths(record, catalog_h3, profile)
check("a model on another GPU leaves the clip's room alone", record["models"]["h3"]["max_seconds"] == 6)
record = {"models": {"h3": {"kind": "video", "unit": "active"}, "big": {"kind": "text", "unit": "active"}}}
spark = {"nvidia-gb10": dict(gpu, family="nvidia-gb10", memory_gib=120), "nvidia-gb10-pair": dict(gpu, family="nvidia-gb10-pair", nodes=2, memory_gib=120)}
paired = {"h3": {"gpu": "nvidia-gb10", "variant": {"peak_gpu_gib": 46, "peak_ram_gib": 46}}, "big": {"gpu": "nvidia-gb10-pair", "variant": {"peak_gpu_gib": 70, "peak_ram_gib": 70}}}
with patch.object(a, "load_model", side_effect=paired.get), patch.object(a, "choose_gpu", side_effect=lambda profile, family=None: spark[family]):
  a.add_clip_lengths(record, catalog_h3, profile)
check("a paired model on linked Sparks shares the clip's GPU", record["models"]["h3"]["max_seconds"] == 3)
record = {"models": {"h3": {"kind": "video", "unit": "active"}, "klein": {"kind": "image", "unit": "installing"}}}
with patch.object(a, "load_model", side_effect={"h3": manifests["h3"]}.get), patch.object(a, "choose_gpu", return_value=dict(gpu, memory_gib=54)):
  a.add_clip_lengths(record, catalog_h3, profile, pending={"klein": manifests["klein"]})
check("a model being set up holds its memory before its setup is saved", record["models"]["h3"]["max_seconds"] == 3)
record = {"models": {"h3": {"kind": "video", "unit": "active"}, "qwen": {"kind": "text", "unit": "active"}}}
unified = {"h3": {"variant": {"peak_gpu_gib": 46, "peak_ram_gib": 46}}, "qwen": {"variant": {"peak_gpu_gib": 30, "peak_ram_gib": 40}}}
with patch.object(a, "load_model", side_effect=unified.get), patch.object(a, "choose_gpu", return_value=dict(gpu, family="nvidia-gb10", unified=True, memory_gib=120)):
  a.add_clip_lengths(record, catalog_h3, dict(profile, ram_gib=120))
check("a GPU sharing RAM leaves room for the desktop and the other models' RAM", record["models"]["h3"]["max_seconds"] == 9)

# The root script against a limine-mkinitcpio that fails a kernel and still exits 0, as it does.
fake = Path(os.environ["HOME"]) / "fake-limine"
fake.mkdir()
(fake / "limine-mkinitcpio").write_text("#!/bin/bash\necho 'mkinitcpio failed for kernel linux'\n")
(fake / "limine-mkinitcpio").chmod(0o755)
drop_in = fake / "limine-entry-tool.d" / "amd-gpu-memory.conf"
run_script = lambda: subprocess.run(["bash", "-c", a.GPU_MEMORY_SCRIPT, "test", str(drop_in), "14185506"], capture_output=True, env=dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}")).returncode
check("a failed rebuild fails and leaves no drop-in to skip the retry", run_script() == 1 and not drop_in.exists())
drop_in.write_text("# Omarchy's\n")
check("a failed rebuild puts the earlier drop-in back", run_script() == 1 and drop_in.read_text() == "# Omarchy's\n")
(fake / "limine-mkinitcpio").write_text("#!/bin/bash\necho rebuilt\n")
(fake / "not-a-dir").write_text("")
unwritable = subprocess.run(["bash", "-c", a.GPU_MEMORY_SCRIPT, "test", str(fake / "not-a-dir" / "amd-gpu-memory.conf"), "14185506"], capture_output=True, env=dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}"))
check("a drop-in that can't be written fails before the rebuild", unwritable.returncode == 1 and b"rebuilt" not in unwritable.stdout + unwritable.stderr)
check("a rebuild that succeeds keeps the raised cap", run_script() == 0 and drop_in.read_text() == 'KERNEL_CMDLINE[default]+=" ttm.pages_limit=14185506"\n')

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
check("unit stops with graphical-session without blocking compositor teardown", "PartOf=graphical-session.target\n" in text and "After=graphical-session.target\n" not in text)
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
               run={"devices": ["cdi-nvidia", "infiniband"], "network": "fabric", "memlock": "unlimited"}, gpus=["nvidia-gb10-pair"],
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
single = dict(recipe, hardware=[{"gpu": "nvidia-gb10", "measured": "m.toml"}], runtime="server")
paired = dict(recipe, hardware=[{"gpu": "nvidia-gb10-pair", "measured": "m.toml"}], runtime="pair",
              variant=[{"ctx": 4096, "peak_ram_gib": 104, "peak_gpu_gib": 104}])
check("a pair machine runs single-node recipes on its own GPU", a.recipe_gpu(pair_profile, single)["family"] == "nvidia-gb10" and a.recipe_gpu(pair_profile, paired)["family"] == "nvidia-gb10-pair")
nv_rt = dict(runtime, run={"devices": ["cdi-nvidia"]}, gpus=["nvidia-gb10"])
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
engine = a._quote(a.ENGINE)
check("the pair unit starts and stops rank 1 with itself, run by the engine's own path",
      f"ExecStartPre={engine} link-worker start paired" in pair_text and f"ExecStopPost=-{engine} link-worker stop paired" in pair_text
      and a.ENGINE.startswith("/") and os.path.basename(a.ENGINE) == "omarchy-local-ai", pair_text)
check("no unit hook goes through the PATH", "/usr/bin/env" not in pair_text and f"ExecStartPost=-{engine} status --write --started paired" in pair_text)
# Units an older engine wrote reach it through Omarchy's wrapper on the PATH, or by another engine's
# path, such as a checkout's; refresh and start point them at the engine running.
a.write_atomic(a.models_dir() + "/paired.json", "{}")
for old in ("/usr/bin/env omarchy-models", "/usr/bin/env omarchy-agent-local", '"/home/me/src/omarchy local ai/omarchy-local-ai"'):
  a.write_atomic(a.quadlet_path("paired"), pair_text.replace(f"{engine} ", f"{old} "))
  with patch.object(a, "systemctl") as reloaded:
    first = a.repair_unit_hooks()
    second = a.repair_unit_hooks()
  check(f"an older unit's hooks through {old} are pointed at this engine once, and the rest left as it was",
        first == ["paired"] and second == [] and a._read(a.quadlet_path("paired")) == pair_text.strip() and reloaded.call_count == 1, repr((first, second)))
# Older units also carried After=graphical-session.target; repair drops it once without altering anything else.
a.write_atomic(a.quadlet_path("paired"), pair_text.replace("PartOf=graphical-session.target\n", "PartOf=graphical-session.target\nAfter=graphical-session.target\n", 1))
with patch.object(a, "systemctl") as reloaded:
  first = a.repair_unit_hooks()
  second = a.repair_unit_hooks()
check("an older unit's After=graphical-session.target ordering is stripped once",
      first == ["paired"] and second == [] and a._read(a.quadlet_path("paired")) == pair_text.strip() and reloaded.call_count == 1, repr((first, second)))
# Under the lock: refresh leaves the units alone while another command holds it, as an update may be
# replacing one, and repairs them once it's free.
a.write_atomic(a.quadlet_path("paired"), pair_text.replace(f"{engine} ", "/usr/bin/env omarchy-models "))
import fcntl
holder = open(os.path.join(a.state_dir(), "lock"), "w")
fcntl.flock(holder, fcntl.LOCK_EX)
try:
  a.repair_unit_hooks(wait=False)
  busy = False
except a.LockBusy:
  busy = True
check("the repair doesn't wait for a held lock when told not to", busy)
untouched = "omarchy-models" in a._read(a.quadlet_path("paired")) and a.Lock.held == 0
fcntl.flock(holder, fcntl.LOCK_UN)
holder.close()
with patch.object(a, "systemctl"):
  freed = a.repair_unit_hooks(wait=False)
check("while the lock is held it writes nothing; once free it repairs", untouched and freed == ["paired"] and a._read(a.quadlet_path("paired")) == pair_text.strip())
os.unlink(a.quadlet_path("paired"))
os.unlink(a.models_dir() + "/paired.json")
check("a paired runtime may serve its prepared folder", '"--model" "/prepared/model"' in a.render_container("paired", paired, dict(pair_rt, pair=dict(pair_rt["pair"], model="/prepared/model")), paired["variant"][0], {"model": "model"}, pair_gpu, "b" * 16))
same_rt = copy.deepcopy(pair_rt)
for key in ("worker", "prepare"):
  same_rt["pair"].pop(key)
same_rt["pair"].update(state="/state", env={"NCCL_COMM_ID": "@MASTER_ADDR@:@MASTER_PORT@", "TF_RANK": "@RANK@"})
check("a paired runtime may run the head's command on both ranks, with its own NCCL settings", not a.validate_runtime("same", same_rt), "\n".join(a.validate_runtime("same", same_rt)))
same_text = a.render_container("paired", paired, same_rt, paired["variant"][0], {"model": "model"}, pair_gpu, "b" * 16)
check("a state path mounts the rank folder writable", f"models/paired/{'b' * 16}.rank0:/state\n" in same_text)
with patch.object(a, "data_dir", return_value=str(Path(os.environ["HOME"]) / ".local/state/omarchy/agent-local/data")):
  fallback_text = a.render_container("paired", paired, same_rt, paired["variant"][0], {"model": "model"}, pair_gpu, "b" * 16)
check("the pair unit mounts rank 0 from the fallback data root", f"Volume=%h/.local/state/omarchy/agent-local/data/models/paired/{'b' * 16}.rank0:/state" in fallback_text)
_, same_argv, _, _ = a.head_argv(same_rt, paired, paired["variant"][0], {"model": "model"}, "paired")
check("rank 1's default command is the head's, bound to loopback", same_argv[-4:] == ["--host", "127.0.0.1", "--port", str(a.port_for("paired"))] and same_argv[:2] == ["server", "serve"])
# Each rank gets its own link settings, and rank 1 the head's command inside the same restrictions.
setup = {"schema": 1, "recipe_id": "paired", "runtime": same_rt, "recipe": paired, "variant": paired["variant"][0], "roles": {"model": "model"},
         "fileset": "e" * 16, "pair": {"peer": "bob@10.99.1.2", "host": "spark2", "home": "/home/bob", "image_id": "f" * 64}}
a.write_pair_setup(setup)
here = {"addr": "10.99.1.1", "dev": "enp1", "hcas": ["rocep1", "roceP2"], "gid": "5"}
there = {"addr": "10.99.1.2", "dev": "enp1", "hcas": ["rocep1", "roceP2"], "gid": "3"}
with patch.object(a, "pair_link", return_value=(here, there)), patch.object(a, "peer_headroom") as headroom, patch.object(a, "pair_ssh") as ssh, patch.object(a, "pair_limits") as limits, patch.object(a, "pair_python"), \
     patch.object(a, "peer_data_dir", return_value="/home/bob/.local/state/omarchy/agent-local/data"):
  a.pair_worker_start("paired")
head_env = dict(line.split("=", 1) for line in Path(a.pair_env_path("paired")).read_text().splitlines())
worker = shlex.split(ssh.call_args.args[1])
worker_env = dict(worker[i + 1].split("=", 1) for i, word in enumerate(worker) if word == "-e")
check("rank 0's environment is rank 0's side of the link", head_env == {"NCCL_COMM_ID": "10.99.1.1:29620", "TF_RANK": "0", "TF_MASTER": "10.99.1.1", "NCCL_SOCKET_IFNAME": "enp1", "NCCL_IB_HCA": "rocep1,roceP2", "NCCL_IB_GID_INDEX": "5"}, repr(head_env))
check("rank 1's environment is rank 1's side, same rendezvous", worker_env["TF_RANK"] == "1" and worker_env["NCCL_COMM_ID"] == "10.99.1.1:29620" and worker_env["NCCL_IB_GID_INDEX"] == "3" and "API_KEY" not in worker_env, repr(worker_env))
check("rank 1 starts on the peer as its user, with the head's command and the same restrictions",
      ssh.call_args.args[0] == "bob@10.99.1.2" and headroom.called
      and limits.called and all(flag in worker for flag in ("--read-only", "no-new-privileges", "/dev/infiniband", "f" * 64))
      and worker[worker.index("--cap-drop") + 1] == "all"
      and f"/home/bob/.local/state/omarchy/agent-local/data/models/paired/{'e' * 16}:/models:ro" in worker
      and f"/home/bob/.local/state/omarchy/agent-local/data/models/paired/{'e' * 16}.rank1:/state" in worker
      and worker[worker.index("f" * 64) + 1:][:2] == ["server", "serve"] and worker[-4:] == ["--host", "127.0.0.1", "--port", str(a.port_for("paired"))], repr(worker))
os.unlink(a.pair_setup_path("paired"))
# Setup copies the image (only when the peer lacks it) and the file set, and makes both ranks' state folders.
ran = []
def fake_run(argv, **kwargs):
  ran.append(argv)
  return SimpleNamespace(returncode=0, stdout="f" * 64 + "\n", stderr="")
manifest = {"fileset": "e" * 16}
local_data = str(Path(os.environ["HOME"]) / ".local/state/omarchy/agent-local/data")
peer_data = "/home/bob/.local/state/omarchy/agent-local/data"
with patch.object(a.subprocess, "run", side_effect=fake_run), patch.object(a.subprocess, "Popen") as save, \
     patch.object(a, "pair_ssh", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as ssh, patch.object(a, "load_installing", return_value=None), \
     patch.object(a, "data_dir", return_value=local_data), patch.object(a, "peer_data_dir", return_value=peer_data):
  a.pair_setup("paired", manifest, same_rt, {"peer": "bob@10.99.1.2", "host": "spark2", "home": "/home/bob"})
rsync = next(argv for argv in ran if argv[0] == "rsync")
check("setup records the peer and the image by ID", manifest["pair"] == {"peer": "bob@10.99.1.2", "host": "spark2", "home": "/home/bob", "image_id": "f" * 64})
check("setup copies the file set to the peer's data root, not an image it already has", not save.called and rsync[-1] == f"bob@10.99.1.2:{peer_data}/models/paired/{'e' * 16}/")
check("setup makes both ranks' state folders under the data roots", Path(local_data, "models", "paired", f"{'e' * 16}.rank0").is_dir()
      and any(f"{peer_data}/models/paired/{'e' * 16}.rank1" in call.args[1] for call in ssh.call_args_list))
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
     patch.object(a, "pair_ssh", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")), patch.object(a, "load_installing", return_value=None), \
     patch.object(a, "peer_data_dir", return_value=peer_data):
  a.pair_setup("paired", {"fileset": "e" * 16}, pair_rt, {"peer": "10.99.1.2", "host": "spark2", "home": "/home/bob"})
prepares = [argv for argv in ran if any(word == "prep" or " prep " in word for word in argv)]
check("prepare runs for rank 0 here and rank 1 on the peer, without network",
      len(prepares) == 2 and prepares[0][-3:] == ["prep", "0", "2"] and "none" in prepares[0]
      and "prep 1 2" in " ".join(prepares[1]) and "--network none" in " ".join(prepares[1]), repr(prepares))
with patch.object(a.subprocess, "run", side_effect=lambda argv, **k: SimpleNamespace(returncode=1 if "prep" in argv else 0, stdout="f" * 64, stderr="")), \
     patch.object(a, "pair_ssh", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")), patch.object(a, "load_installing", return_value=None), \
     patch.object(a, "peer_data_dir", return_value=peer_data):
  raises("a failed prepare stops the setup", lambda: a.pair_setup("paired", {"fileset": "e" * 16}, pair_rt, {"peer": "10.99.1.2", "host": "spark2", "home": "/home/bob"}), "Preparing rank 0")
# Disk for the prepared files is checked on both Sparks before anything is written, and
# prepare containers carry names remove can stop.
big = dict(pair_rt, pair=dict(pair_rt["pair"], prepared_gib=100))
# A served model folder links into /models, which only resolves inside the container.
served = Path(a.rank_dir("paired", "e" * 16, 0, runtime=big), "model")
served.mkdir(parents=True, exist_ok=True)
(served / "config.json").unlink(missing_ok=True)
(served / "config.json").symlink_to("/models/model/config.json")
ran.clear()
with patch.object(a.subprocess, "run", side_effect=fake_prepare), patch.object(a, "load_installing", return_value=None), \
     patch.object(a.shutil, "disk_usage", return_value=SimpleNamespace(free=500 * 2**30)), \
     patch.object(a, "peer_data_dir", return_value=peer_data), \
     patch.object(a, "pair_ssh", return_value=SimpleNamespace(returncode=0, stdout="%d\n0\n" % (50 * 2**30), stderr="")) as disk:
  raises("a peer without room for the files and prepared inputs stops setup first", lambda: a.pair_setup("paired", {"fileset": "e" * 16, "files": [{"size": 10 * 2**30}]}, big, {"peer": "10.99.1.2", "host": "spark2", "home": "/home/bob"}), "spark2 has 50 GiB free")
measured = disk.call_args_list[0].args[1].split("du -sbc ", 1)[1].split(" 2>")[0]
check("the peer credits only the file set and the current generation as already there",
      shlex.split(measured) == [f"{peer_data}/models/paired/{'e' * 16}", a.rank_dir("paired", "e" * 16, 1, peer_data, big)], measured)
check("nothing is prepared or copied when the disk check fails", not [argv for argv in ran if "prep" in argv or argv[:1] == ["rsync"]], repr(ran))
check("prepare containers are named for their recipe and rank", "--name" in prepares[0] and a.prepare_name("paired", 0) in prepares[0] and a.prepare_name("paired", 1) in " ".join(prepares[1]))
with patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as local, patch.object(a, "pair_ssh") as remote, patch.object(a, "known_models", return_value=[]), \
     patch.object(a, "peer_data_dir", return_value=peer_data):
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
with patch.object(a, "platform_name", return_value="aarch64-gb10"), patch.object(a, "amd_gpus", side_effect=lambda: []), patch.object(a, "intel_gpus", return_value=[]):
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
check("a newly cabled Spark is announced once, with link setup as its action", len(notices) == 1 and notices[0][-1] == "omarchy-models link setup", repr(notices))
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
  check("an older reason gives way, and with no server error it stopped unexpectedly, without systemd's result code", a.failure_reason("crashy", "exit-code") == "Stopped unexpectedly")
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
# Reading a failed model's logs dismisses the failure: the recorded reason goes, and so does the
# unit's failed state, which reads as stopped unexpectedly for as long as it stays.
a.record_error("current", "Stopped unexpectedly")
with patch.object(a, "load_catalog"), patch.object(a, "machine_profile", return_value=profile), patch.object(a, "write_status") as wrote, \
     patch.object(a, "unit_active", return_value=False), patch.object(a, "systemctl") as ctl:
  a.cmd_dismiss(["current"])
check("dismissing a failure forgets its reason and resets the unit", not Path(a.error_path("current")).exists() and ctl.call_args.args == ("reset-failed", a.unit_name("current")) and wrote.called)

# An update is what setting a model up again from the catalog would change. A setup made from the
# catalog as it is has none, and neither has one whose image only moved to another name.
def update_for(recipe_now, runtime_now):
  now = SimpleNamespace(recipes={"current": recipe_now}, runtimes={recipe_now["runtime"]: runtime_now})
  with patch.object(a, "choose_gpu", return_value=gpu), patch.object(a, "compatibility", return_value=(manifest["variant"], "")):
    return a.pending_update(manifest, now, profile)
renamed = dict(runtime, image=runtime["image"].replace("/test/", "/moved/"))
check("a model set up from the catalog as it is has no update", update_for(recipe, runtime) is None and update_for(recipe, renamed) is None)
check("a new image digest is an update", update_for(recipe, dict(runtime, image=runtime["image"].replace("a" * 64, "b" * 64))) == {"image": True, "weights": False, "settings": False, "download_bytes": None})
repinned = dict(recipe, artifacts=[dict(recipe["artifacts"][0], revision="f" * 40)] + recipe["artifacts"][1:])
check("a new weights pin is an update", update_for(repinned, runtime) == {"image": False, "weights": True, "settings": False, "download_bytes": None})
retuned = dict(recipe, server={"args": recipe.get("server", {}).get("args", []) + ["--slots", "2"]})
check("new server arguments are an update", update_for(retuned, runtime) == {"image": False, "weights": False, "settings": True, "download_bytes": None})
check("a new warm-up is an update", update_for(recipe, dict(runtime, warmup={"prompts": 1, "pieces": 20000})) == {"image": False, "weights": False, "settings": True, "download_bytes": None})
check("a recipe the catalog dropped has no update", a.pending_update(manifest, SimpleNamespace(recipes={}, runtimes={}), profile) is None)
# A pair's workers start from the manifest's own pair table, so a change there is an update too.
check("a paired runtime's new worker settings are an update", update_for(recipe, dict(runtime, pair={"nodes": 2, "env": {"RESERVE": "9"}}))["settings"])
# The download is sized by refresh, which asks the registry and Hugging Face, and shown once it has:
# the new image's layers the current one lacks, and the weights not in the store.
newer = dict(runtime, image=runtime["image"].replace("a" * 64, "b" * 64))
layers = {newer["image"]: {"sha256:base": 100, "sha256:server": 40}, runtime["image"]: {"sha256:base": 100, "sha256:old": 30}}
with patch.object(a, "known_models", return_value=["current"]), patch.object(a, "load_model", return_value=manifest), patch.object(a, "choose_gpu", return_value=gpu), \
     patch.object(a, "compatibility", return_value=(manifest["variant"], "")), patch.object(a, "image_layers", side_effect=lambda image: layers[image]), \
     patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=1)):
  a.measure_updates(SimpleNamespace(recipes={"current": recipe}, runtimes={"server": newer}), profile)
check("an update's download is the image layers the current image lacks", update_for(recipe, newer)["download_bytes"] == 40, str(update_for(recipe, newer)))
# What an update fetches depends on what is stored now, so a refresh measures again rather than
# trusting the last answer; an unreachable registry keeps it.
def remeasure(**how):
  with patch.object(a, "known_models", return_value=["current"]), patch.object(a, "load_model", return_value=manifest), patch.object(a, "choose_gpu", return_value=gpu), \
       patch.object(a, "compatibility", return_value=(manifest["variant"], "")), patch.object(a, "update_download_bytes", **how):
    a.measure_updates(SimpleNamespace(recipes={"current": recipe}, runtimes={"server": newer}), profile)
  return update_for(recipe, newer)["download_bytes"]
check("a refresh sizes an update again", remeasure(return_value=140) == 140)
check("an unreachable registry keeps the last size", remeasure(side_effect=a.LocalError("offline")) == 140)
check("a size measured for another update isn't shown for this one", update_for(recipe, dict(runtime, image=runtime["image"].replace("a" * 64, "c" * 64)))["download_bytes"] is None)
index = {"manifests": [{"digest": "sha256:x86", "platform": {"architecture": "amd64"}}, {"digest": "sha256:arm", "platform": {"architecture": "arm64"}}]}
with patch.object(a, "registry_json", side_effect=lambda url, accept: index if url.endswith("a" * 64) else {"layers": [{"digest": "sha256:l", "size": 7}]}) as asked, \
     patch.object(a.os, "uname", return_value=SimpleNamespace(machine="aarch64")):
  check("a multi-arch image's layers are this machine's build's", a.image_layers(runtime["image"]) == {"sha256:l": 7} and asked.call_args.args[0].endswith("/manifests/sha256:arm"))
# Updating is use with the context and GPU the model was set up with. A model that was stopped
# stays stopped, the agents' default stays the model the user last started, and a model that is
# loading its update still says it is updating.
last = Path(a.state_dir()) / "last"
for was_running, stops in ((False, 1), (True, 0)):
  last.write_text("other\n")
  def set_up(argv):
    last.write_text("current\n")
    with patch.object(a, "unit_props", return_value={}), patch.object(a, "free_memory", return_value=(64, 48)), patch.object(a, "known_models", return_value=["current"]):
      set_up.loading = a.write_status(a.load_catalog(), profile, {"current": "activating"}, pending={"current": a.load_model("current")})["models"]["current"]
    return 0
  write_status = a.write_status
  with patch.object(a, "machine_profile", return_value=profile), patch.object(a, "write_status", side_effect=lambda *args, **kw: write_status(*args, **kw) if kw.get("pending") else None), \
       patch.object(a, "pending_update", return_value={"image": True, "weights": False, "settings": False}), \
       patch.object(a, "unit_active", side_effect=[was_running, True]), patch.object(a, "cmd_use", side_effect=set_up) as use, patch.object(a, "cmd_stop", return_value=0) as stop:
    a.cmd_update(["current"])
  check(f"update sets a {'running' if was_running else 'stopped'} model up again and leaves it that way",
        use.call_args.args[0] == ["current", "--gpu", gpu["family"], "--ctx", str(a.load_model("current")["variant"]["ctx"])] and stop.call_count == stops and not Path(a.updating_path("current")).exists(),
        f"{use.call_args} {stop.call_count}")
  check("an update leaves the agents' default model alone", last.read_text() == "other\n")
  check("a model loading its update says it is updating", set_up.loading["updating"] is True and set_up.loading["unit"] == "activating", str(set_up.loading))
last.unlink()
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
  with patch.object(a, "unit_active", return_value=False), patch.object(a, "write_status"), patch.object(a, "evict_idle_weights"), patch.object(a, "check_headroom", return_value=(100.0, 90.0)), patch.object(a, "free_memory", return_value=(52.0, 64.0)), patch.object(a, "ensure_key", return_value="test-key"), patch.object(a, "http_status", side_effect=response) as http, patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0, stdout="")) as systemctl:
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
# Set up before their recipes named a maker, both models' manifests carry none.
marked = SimpleNamespace(version="test", best={"class": []}, recipes={"flux": {"vendor": "bfl"}, "plain": {}})
with patch.object(a, "known_models", return_value=["flux", "plain"]), patch.object(a, "installing_models", return_value=[]), \
     patch.object(a, "load_model", return_value={}), patch.object(a, "model_record", side_effect=lambda rid, manifest, profile: {"vendor": ""}):
  vendors = {rid: row["vendor"] for rid, row in a.status_record(marked, profile)["models"].items()}
check("a model set up before its recipe named a maker shows the catalog's mark", vendors == {"flux": "bfl", "plain": ""}, json.dumps(vendors))
with patch.object(a, "known_models", return_value=["current"]), patch.object(a, "unit_props", return_value={}), patch.object(a, "free_memory", return_value=(64, 48)):
  a.write_status(a.load_catalog(), profile)
a.publish_installing("current", {"name": "Current", "phase": "downloading", "done_bytes": 5, "total_bytes": 10})
current = json.loads(Path(a.state_dir(), "status.json").read_text())["models"]["current"]
check("setting up an installed model again leaves its row as it is", current["unit"] != "installing")
check("an update's download rides on the model's own row", current["update_progress"] == {"phase": "downloading", "done_bytes": 5, "total_bytes": 10, "rate_bytes_s": 0}, json.dumps(current.get("update_progress")))

# What a running model holds now: its anonymous memory and its GTT, once per GPU client, not
# the file cache it reads weights through, across its unit and the container's own cgroup.
with tempfile.TemporaryDirectory() as fake:
  unit = Path(fake, "cg", "app.slice", "m.service")
  (unit / "libpod-payload-1").mkdir(parents=True)
  (unit / "memory.stat").write_text("anon 1073741824\nfile 8589934592\n")
  (unit / "cgroup.procs").write_text("10\n")
  (unit / "libpod-payload-1" / "cgroup.procs").write_text("11\n")
  # Process 10 forked from 11 after it opened the GPU, so both show client 7.
  for pid, fds in {"10": {"3": "drm-client-id:\t7\ndrm-memory-gtt:\t2097152 KiB\n"}, "11": {"3": "drm-client-id:\t7\ndrm-memory-gtt:\t2097152 KiB\n", "4": "drm-client-id:\t7\ndrm-memory-gtt:\t2097152 KiB\n", "5": "pos: 0\n"}}.items():
    Path(fake, "proc", pid, "fdinfo").mkdir(parents=True)
    for fd, text in fds.items():
      Path(fake, "proc", pid, "fdinfo", fd).write_text(text)
  held = a.held_bytes("/app.slice/m.service", str(Path(fake, "cg")), str(Path(fake, "proc")))
check("a model holds its anonymous memory and its GTT once per GPU client, across its processes, not its file cache", held == 3 * 2**30, str(held))

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
with patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0, stdout="")) as stop, patch.object(a, "known_models", return_value=[]), patch.object(a, "write_unit_files"), patch.object(a, "refresh_list_cache"), patch.object(a, "write_status"):
  check("remove dismisses a failed setup", a.cmd_remove(["dismissed"]) == 0 and a.load_installing("dismissed") is None)
  check("remove stops the setup's unit first", ("stop", a.install_unit("dismissed")) in [call.args for call in stop.call_args_list])

# Stopped mid-load, systemd skips ExecStop and conmon ignores SIGTERM: the container goes first.
for state, removed in (("activating", True), ("active", False)):
  calls = []
  with patch.object(a, "unit_props", return_value={"ActiveState": state}), \
       patch.object(a, "systemctl", side_effect=lambda *args, **_: calls.append(args)), \
       patch.object(a.subprocess, "run", side_effect=lambda argv, **_: calls.append(tuple(argv))):
    a.stop_unit("loading")
  unit, container = a.unit_name("loading"), a.unit_base("loading")
  expected = [("stop", "--no-block", unit), ("podman", "rm", "-v", "-f", "-i", "-t", "0", container), ("stop", unit), ("reset-failed", unit)] if removed else [("stop", unit)]
  check(f"stopping a model that is {state} {'removes its container' if removed else 'leaves it to ExecStop'}", calls == expected, repr(calls))
a.clear_error("loading")
with patch.dict(os.environ, {"SERVICE_RESULT": "exit-code"}), patch.object(a, "unit_props", return_value={"InactiveExitTimestamp": "@1"}), patch.object(a, "write_status") as written:
  a.cmd_status(["--write", "--stopped", "loading"])
check("a model stopped mid-load ends stopped, not failed", written.call_args.args[2] == {"loading": "inactive"} and not os.path.exists(a.error_path("loading")), repr(written.call_args))

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
with patch.object(a, "load_catalog", return_value=cat), patch.object(a, "machine_profile", return_value=profile), patch.object(a, "Lock", return_value=Waiting()), patch.object(a, "load_model", return_value=None), patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0, stdout="")), patch.object(a, "stop_setup_process"), patch.object(a, "write_status"), patch.object(a, "refresh_list_cache"), patch.object(a, "_remove_files", return_value=(0, [])) as removed:
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

# Pre-sizing the runtime image reserves its download and unpack share upfront, skipped
# layers don't spike download speed, and unpacking advances the bar without faking speed.
planned = a.Progress()
planned.samples = [(planned.started - 1, 0)]
planned.expect_image(10 * 2**20)
planned.grow(20 * 2**20)
planned.add(20 * 2**20)
done_w, total_w, rate_w = planned.snapshot()
extract_share = int(10 * 2**20 * a.IMAGE_EXTRACT_WEIGHT)
check("pre-sizing reserves image download and unpack on the bar before the pull starts", done_w == 20 * 2**20 and total_w == 30 * 2**20 + extract_share and done_w < total_w, f"{done_w}/{total_w}")
skipped_prog = a.Progress()
skipped_prog.samples = [(skipped_prog.started - 1, 0)]
skipped_prog.pulled(4 * 2**20, 10 * 2**20, skipped=4 * 2**20, extract_done=0, extract_total=extract_share)
done_s, total_s, rate_s = skipped_prog.snapshot()
check("skipped image layers advance the bar without counting as network speed", done_s == 4 * 2**20 and total_s == 10 * 2**20 + extract_share and rate_s == 0, f"{done_s}/{total_s}@{rate_s}")
skipped_prog.samples = [(a.time.time() - 1, 4 * 2**20)]
skipped_prog.pulled(4 * 2**20, 10 * 2**20, skipped=4 * 2**20, extract_done=extract_share // 2, extract_total=extract_share)
done_e, total_e, rate_e = skipped_prog.snapshot()
check("image unpacking advances the bar without reporting a fake network rate", done_e == 4 * 2**20 + extract_share // 2 and rate_e == 0, f"{done_e}/{total_e}@{rate_e}")

# Extraction fraction moves smoothly across layer commits and disk writes, capping below 1 while running.
f0 = a._extract_fraction(0, 10, 0, 1000)
f1 = a._extract_fraction(0, 10, 500, 1000)
f2 = a._extract_fraction(5, 10, 1250, 1000)
f3 = a._extract_fraction(10, 10, 10000, 1000)
check("extraction fraction rises smoothly with writes and layer commits", 0 == f0 < f1 < f2 < f3 <= 0.98, f"{f0}, {f1}, {f2}, {f3}")

# Temporary pull byte tracking stays monotonic even if Podman deletes finished blob files.
with tempfile.TemporaryDirectory() as td:
  seen = {}
  p1 = Path(td, "1")
  p1.write_bytes(b"a" * 100)
  b1 = a._track_dir_bytes(td, seen)
  p1.unlink()
  Path(td, "2").write_bytes(b"b" * 50)
  b2 = a._track_dir_bytes(td, seen)
check("pull temp byte tracking never drops when blobs are removed", b1 == 100 and b2 == 150, f"{b1} -> {b2}")

# ensure_image reads exact blob bytes from its private TMPDIR, tracks overlay commits, and cleans up.
with tempfile.TemporaryDirectory() as fake_overlay:
  a.clear_installing("puller")
  a.begin_installing("puller", {"name": "Puller"})
  pull_prog = a.Progress(rid="puller")
  pull_prog.expect_image(1024)
  seen_snapshots = []
  orig_pulled = pull_prog.pulled
  def record_pulled(got, total, skipped=0, extract_done=None, extract_total=None):
    seen_snapshots.append((got, total, skipped, extract_done, extract_total))
    orig_pulled(got, total, skipped=skipped, extract_done=extract_done, extract_total=extract_total)
  pull_prog.pulled = record_pulled
  fake_pull = (
    "import os, sys, time; "
    "tmp = os.environ['TMPDIR']; "
    "print('Copying blob aaaaaaaaaaaa [----] 0.0b / 512b', flush=True); "
    "open(os.path.join(tmp, '1'), 'wb').write(b'x' * 512); "
    "time.sleep(0.35); "
    "print('Copying blob bbbbbbbbbbbb skipped: already exists', flush=True); "
    "print('Copying blob aaaaaaaaaaaa done   | ', flush=True); "
    "os.mkdir(os.path.join(os.environ['FAKE_OVERLAY'], 'c' * 64)); "
    "time.sleep(0.35)"
  )
  real_popen = a.subprocess.Popen
  def fake_popen(argv, **kwargs):
    env = dict(kwargs.get("env") or {}, FAKE_OVERLAY=fake_overlay)
    kwargs["env"] = env
    return real_popen([sys.executable, "-c", fake_pull], **kwargs)
  with patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=1)), \
       patch.object(a, "_podman_overlay_dir", return_value=fake_overlay), \
       patch.object(a.subprocess, "Popen", side_effect=fake_popen):
    a.ensure_image("ghcr.io/example/img@sha256:" + "d" * 64, pull_prog, expected_layers={"aaaaaaaaaaaa": 512, "bbbbbbbbbbbb": 512}, rid="puller")
  check("ensure_image reaches 100% of download and unpack on completion", seen_snapshots and seen_snapshots[-1][:3] == (1024.0, 1024.0, 512.0) and seen_snapshots[-1][3] == seen_snapshots[-1][4] and seen_snapshots[-1][4] > 0, repr(seen_snapshots[-3:]))
  check("ensure_image saw intermediate TMPDIR byte progress before PTY done", any(512 <= s[0] < 1024 for s in seen_snapshots) or any(s[3] and s[3] < s[4] for s in seen_snapshots), repr(seen_snapshots))
  check("ensure_image clears its temp dir after pulling", not list(Path(a.data_dir(), "staging").glob("podman-pull-*")))
  a.clear_installing("puller")

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
pull_dir = staging / "podman-pull-cancelled"
pull_dir.mkdir(exist_ok=True)
(pull_dir / "1").write_bytes(b"z" * 12)
a.begin_installing("cancelled", {"name": "Cancelled"})
a.update_installing("cancelled", partials=[mine, "../escape.part"], pull_tmp=pull_dir.name)
with patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0, stdout="")) as stop, patch.object(a, "known_models", return_value=[]), patch.object(a, "write_unit_files"), patch.object(a, "refresh_list_cache"), patch.object(a, "write_status"):
  a.cmd_remove(["cancelled"])
check("cancelling a setup stops its model service", ("stop", a.unit_name("cancelled")) in [call.args for call in stop.call_args_list])
check("cancelling a setup drops only its own partial downloads and pull temp dir", not (staging / mine).exists() and (staging / other).exists() and not pull_dir.exists())

# Cancelling a background update on an installed model also cleans up its pull temp dir.
upd_pull = staging / "podman-pull-update"
upd_pull.mkdir(exist_ok=True)
(upd_pull / "1").write_bytes(b"u" * 16)
a.save_model({"schema": 1, "recipe_id": "updating-model", "files": [], "recipe": {}, "runtime": {}, "variant": {}, "image": ""}, started=False)
a.begin_installing("updating-model", {"name": "Updating"})
a.update_installing("updating-model", pull_tmp=upd_pull.name)
with patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0, stdout="")), patch.object(a, "unit_active", return_value=False), patch.object(a, "known_models", return_value=[]), patch.object(a, "write_unit_files"), patch.object(a, "refresh_list_cache"), patch.object(a, "write_status"), patch.object(a, "_remove_files", return_value=(0, [])):
  a.cmd_remove(["updating-model"])
check("removing a model with a cancelled background update drops its pull temp dir", not upd_pull.exists() and a.load_installing("updating-model") is None)

# --detach hands the setup to a transient user unit and returns.
a.clear_installing(rid)
with patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=3)), patch.object(a, "load_model", return_value=None), patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as run:
  check("a detached setup returns at once", a.detach_use(rid, SimpleNamespace(ctx=8192, gpu=None, seed=None), cat.recipes[rid]) == 0)
  argv = run.call_args.args[0]
check("a detached setup runs use in its own user unit", argv[:3] == ["systemd-run", "--user", "--collect"] and f"--unit={a.install_unit(rid)}" in argv and argv[-4:-2] == ["use", rid] and argv[-2:] == ["--ctx", "8192"], " ".join(argv))
check("a detached setup is listed before its unit starts", a.load_installing(rid)["phase"] == "waiting")
a.clear_installing(rid)
with patch.dict(os.environ, {"HF_TOKEN": "hf_secret_value", "OMARCHY_LOCAL_AI_CATALOG": "relative/catalog"}), patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=3)), patch.object(a, "load_model", return_value=None), patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as run:
  a.detach_use(rid, SimpleNamespace(ctx=None, gpu=None, seed=None), cat.recipes[rid])
  argv, env = run.call_args.args[0], run.call_args.kwargs["env"]
check("a detached setup never puts the token on a command line", not any("hf_secret_value" in arg for arg in argv) and "--setenv=HF_TOKEN" in argv and env["HF_TOKEN"] == "hf_secret_value")
check("a detached setup gets the catalog as an absolute path", env["OMARCHY_LOCAL_AI_CATALOG"] == os.path.abspath("relative/catalog"))
a.clear_installing(rid)
detach_cat = SimpleNamespace(
  recipes={"big-vid": dict(kind="video", runtime="amd-rt", hardware=[{"gpu": "amd-gfx1103", "ram_class": 64, "measured": "m.toml"}], variant=[{"ctx": 4096, "peak_ram_gib": 38, "peak_gpu_gib": 38}])},
  runtimes={"amd-rt": {"gpus": ["amd"]}},
  best={"class": [{"gpu": "amd-gfx1103", "ram_class": 64, "recipes": ["big-vid"], "videos": ["big-vid"]}]},
)
detach_prof = {"platform": "x86", "ram_gib": 64, "ram_class": 64, "gpus": [
  {"vendor": "amd", "family": "amd-gfx1103", "render_node": "/dev/dri/renderD128", "unified": True, "memory_gib": 54.4},
  {"vendor": "amd", "family": "amd-gfx1100", "render_node": "/dev/dri/renderD129", "unified": False, "memory_gib": 24.0},
]}
with patch.object(a, "load_catalog", return_value=detach_cat), patch.object(a, "machine_profile", return_value=detach_prof), \
     patch.object(a, "load_measurement", return_value={}), patch.object(a, "raise_gpu_memory"), \
     patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=3)), patch.object(a, "load_model", return_value=None), \
     patch.object(a.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as run:
  a.cmd_use(["best", "video", "--detach"])
  argv = run.call_args.args[0]
check("use best --detach preserves the resolved GPU and variant context",
      ["use", "big-vid", "--ctx", "4096", "--gpu", "amd-gfx1103"] == argv[-6:], " ".join(argv))
a.clear_installing("big-vid")

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
# The local models panel pairs with no terminal (stdin here is the test script), so
# ssh must ask nothing: a prompt would wait forever. A new host's key is taken.
with patch.object(a.subprocess, "run", return_value=subprocess.CompletedProcess([], 255, "", "Host key verification failed.\n")) as run:
  raises("pairing with a host whose key changed says how to forget the old one, by host alone", lambda: a.cmd_pair(["alice@new.tail1234.ts.net"]), "ssh-keygen -R new.tail1234.ts.net")
check("pairing without a terminal asks ssh for no prompts and takes a new host's key", {"BatchMode=yes", "StrictHostKeyChecking=accept-new"} <= set(run.call_args[0][0]), repr(run.call_args[0][0]))
a.write_atomic(a.remote_key_path("away"), "k" * 32, mode=0o600)
a.write_atomic(os.path.join(a.remotes_dir(), "away.json"), json.dumps({"host": "away", "dns": "away.tail.ts.net", "ssh": "away"}))
raises("a network model is named host/recipe", lambda: a.split_remote_id("box/../x"), "host/recipe")
raises("an SSH target can't smuggle options", lambda: a.ssh_command({"ssh": "-oProxyCommand=x", "dns": ""}, "share", "--export"), "Not an SSH target")
check("remote commands run the engine by its own command, quoted", a.ssh_command({"ssh": "box", "dns": "box.tail.ts.net"}, "share", "--export")[-2:] == ["box", "omarchy-local-ai share --export"])

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
  os.unlink(a.share_error_path())
  with patch.object(a, "tailscale_status", return_value=None):
    record = a.write_status(cat, profile)
  check("a refresh before tailscaled is up reports nothing to fix", record["share_error"] == "", repr(record["share_error"]))
a.save_shares([])

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
proxy = serve(a.make_share_proxy(tracker, ports=lambda: {"shared": model.server_address[1], "private": model.server_address[1]}, shares=lambda: ["shared"], key=lambda: "k"))

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
def locked(rid):
  with open(a.hold_path(rid), "a") as probe:
    try:
      a.fcntl.flock(probe, a.fcntl.LOCK_EX | a.fcntl.LOCK_NB)
      return False
    except BlockingIOError:
      return True
working = locked("shared")
release.set()
rest = answer.read()
connection.close()
check("the proxy streams each part of the answer as the model sends it", answer.status == 200 and first == b"data: hello\n\n" and rest == b"data: [DONE]\n\n", f"{answer.status} {first!r} {rest!r}")
for _ in range(100):
  if not locked("shared"):
    break
  a.time.sleep(0.02)
check("the proxy holds the model until the answer is through", working and not locked("shared"))
import socket
stalled = socket.create_connection(("127.0.0.1", proxy.server_address[1]))
stalled.sendall(b"POST /shared/v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 100\r\n\r\nhel")
a.time.sleep(0.2)
check("an upload without the key that never finishes holds nothing", not locked("shared"))
stalled.close()
# With the key, the hold is taken before the request reaches the model.
stalled = socket.create_connection(("127.0.0.1", proxy.server_address[1]))
stalled.sendall(b"POST /shared/v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer k\r\nContent-Length: 100\r\n\r\nhel")
for _ in range(100):
  if locked("shared"):
    break
  a.time.sleep(0.02)
check("a request with the key holds the model before it reaches it", locked("shared"))
stalled.close()
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
proxy.shutdown()
model.shutdown()


# ---- image models

# An image runtime: sd-server names no model, so it maps no alias, and takes a text encoder and a VAE.
sd_rt = {"image": "registry.example/test/sd@sha256:" + "d" * 64, "tier": "omarchy", "api": "sdcpp", "port": 1234,
         "health": {"probe": ["curl", "http://127.0.0.1:1234/sdcpp/v1/models/status"]}, "command": ["sd-server"],
         "reserved": ["--listen-port", "SD_API_KEY"], "run": {"devices": ["render"], "tmpfs": "/tmp"}, "gpus": ["amd"],
         "inputs": {"model": {"arg": "--diffusion-model"}, "text_encoder": {"arg": "--llm"}, "vae": {"arg": "--vae"}, "key": {"env": "SD_API_KEY"}}}
picture_variant = {"peak_ram_gib": 14, "peak_gpu_gib": 14}
picture = {"name": "Picture", "kind": "image", "runtime": "sd", "hardware": recipe["hardware"],
           "artifacts": [dict(artifact, files=["model.gguf"]),
                         {"repo": "test/encoder", "revision": "c" * 40, "license": "mit", "files": ["te.gguf"], "text_encoder": "te.gguf"},
                         {"repo": "test/vae", "revision": "e" * 40, "license": "mit", "files": ["vae/ae.safetensors"], "vae": "vae/ae.safetensors"}],
           "variant": [picture_variant], "server": {"args": ["--steps", "8"]}}
both = {"server": runtime, "sd": sd_rt}
check("an image runtime without an alias and an image recipe without a context validate", not a.validate_runtime("sd", sd_rt) and not a.validate_recipe("picture", picture, both),
      "\n".join(a.validate_runtime("sd", sd_rt) + a.validate_recipe("picture", picture, both)))
check("a text recipe still needs a context", any(".ctx" in e for e in a.validate_recipe("model", dict(recipe, variant=[picture_variant]), both)))
check("an OpenAI runtime still needs an alias", any("'alias'" in e for e in a.validate_runtime("server", dict(runtime, inputs={k: v for k, v in runtime["inputs"].items() if k != "alias"}))))
check("a recipe's kind is text, image, video or music", any(".kind" in e for e in a.validate_recipe("model", dict(recipe, kind="audio"), both)))
check("a runtime's api is a known one", any(".api" in e for e in a.validate_runtime("sd", dict(sd_rt, api="grpc"))))
check("a text encoder or VAE needs a runtime input for it", {"no text_encoder input", "no vae input"} <= {e.split(" but runtime ")[1].split("' has ")[1] for e in a.validate_recipe("picture", dict(picture, runtime="server"), both) if " but runtime " in e})
trees = {"test/model": [{"type": "file", "path": "model.gguf", "size": 1, "lfs": {"oid": "1" * 64, "size": 1}}],
         "test/encoder": [{"type": "file", "path": "te.gguf", "size": 1, "lfs": {"oid": "2" * 64, "size": 1}}],
         "test/vae": [{"type": "file", "path": "vae/ae.safetensors", "size": 1, "lfs": {"oid": "3" * 64, "size": 1}}]}
with patch.object(a, "hf_tree", side_effect=lambda repo, revision, cache: trees[repo]):
  picture_files, picture_roles = a.build_artifacts(picture)
a.validate_files(picture_files, picture_roles)
_, picture_argv, picture_env, _ = a.server_invocation(sd_rt, picture, picture_variant, picture_roles, "picture")
check("the diffusion model, text encoder and VAE reach the server, and no context",
      picture_argv == ["sd-server", "--diffusion-model", "/models/model.gguf", "--llm", "/models/te.gguf", "--vae", "/models/vae/ae.safetensors", "--steps", "8"] and picture_env == {}, repr(picture_argv))
picture_unit = a.render_container("picture", picture, sd_rt, picture_variant, picture_roles, dict(gpu, render_node="/dev/dri/renderD128"), "f" * 16)
check("a runtime asking for a /tmp tmpfs gets one, others don't", "Tmpfs=/tmp" in picture_unit and "Tmpfs=" not in text and not a.validate_runtime("sd", sd_rt))
check("an image model's variant is chosen without a context", a.pick_variant(picture, profile, gpu) == (picture_variant, ""))

# A video model is an image model's kin: no context, and its sound decoder and its LoRA reach
# the server, the LoRA as the folder sd-server finds it in.
video_rt = dict(sd_rt, inputs=dict(sd_rt["inputs"], audio_vae={"arg": "--audio-vae"}, lora={"arg": "--lora-model-dir"}))
clip = dict(picture, name="Clip", kind="video",
            artifacts=picture["artifacts"] + [{"repo": "test/av", "revision": "b" * 40, "license": "mit", "files": ["vae/audio.safetensors", "loras/turbo.safetensors"],
                                               "audio_vae": "vae/audio.safetensors", "lora": "loras/turbo.safetensors"}])
check("a video recipe validates without a context", not a.validate_recipe("clip", clip, {"sd": video_rt}), "\n".join(a.validate_recipe("clip", clip, {"sd": video_rt})))
clip_trees = dict(trees, **{"test/av": [{"type": "file", "path": "vae/audio.safetensors", "size": 1, "lfs": {"oid": "5" * 64, "size": 1}},
                                        {"type": "file", "path": "loras/turbo.safetensors", "size": 1, "lfs": {"oid": "6" * 64, "size": 1}}]})
with patch.object(a, "hf_tree", side_effect=lambda repo, revision, cache: clip_trees[repo]):
  clip_files, clip_roles = a.build_artifacts(clip)
_, clip_argv, _, _ = a.server_invocation(video_rt, clip, picture_variant, clip_roles, "clip")
check("a video model's sound decoder and LoRA folder reach the server",
      clip_argv[clip_argv.index("--audio-vae") + 1] == "/models/vae/audio.safetensors" and clip_argv[clip_argv.index("--lora-model-dir") + 1] == "/models/loras", repr(clip_argv))
raises("asking an image model for a context names none", lambda: a.pick_variant(picture, profile, gpu, ctx=4096), "have: none")

# Image models are recommended beside text ones, never instead of them.
images_best = {"class": [dict(entry, images=["picture"])]}
check("best.toml lists image models apart from text ones", not a.validate_best(images_best, {"model": recipe, "picture": picture}))
check("an image model can't compete for the agents' default", any(".recipes: 'picture' is of kind image, not text" in e for e in a.validate_best({"class": [dict(entry, recipes=["picture", "model"])]}, {"model": recipe, "picture": picture})))
check("nor can a text model be offered as an image one", any(".images: 'model' is of kind text, not image" in e for e in a.validate_best({"class": [dict(entry, images=["model"])]}, {"model": recipe, "picture": picture})))
img_catalog = SimpleNamespace(root=str(repo), recipes={"model": recipe, "picture": picture}, runtimes=both, best=images_best)
check("best stays the text recommendation, and image models resolve on their own", a.resolve_best(img_catalog, profile, gpu)[0] == "model" and a.resolve_best(img_catalog, profile, gpu, field="images")[0] == "picture")
rows = {row["id"]: row for row in a.recipe_listing(img_catalog, profile)}
check("list rows carry their kind and the best of each kind", (rows["picture"]["kind"], rows["model"]["kind"]) == ("image", "text") and rows["picture"]["best"] and rows["model"]["best"]
      and rows["picture"]["available"] and rows["picture"]["context"] is None and rows["picture"]["reason"] == "", repr(rows["picture"]))
# Only one build of each (kind, name) is offered on a machine: the build aimed at a bigger or
# more specific GPU hides smaller/generic fallbacks when it fits, and the fallback surfaces when
# it does not.
small_rt = dict(copy.deepcopy(runtime), gpus=["amd"])
big_rt = dict(copy.deepcopy(runtime), gpus=["amd-strix-halo"])
small_rec = dict(copy.deepcopy(recipe), runtime="small", variant=[{"ctx": 131072, "peak_ram_gib": 22, "peak_gpu_gib": 21}])
big_rec = dict(copy.deepcopy(recipe), runtime="big", variant=[{"ctx": 262144, "peak_ram_gib": 40, "peak_gpu_gib": 38}])
dedup_catalog = SimpleNamespace(
  root=str(repo),
  recipes={"model-small": small_rec, "model-big": big_rec},
  runtimes={"small": small_rt, "big": big_rt},
  best={"class": [dict(entry, recipes=["model-big"])]},
)
big_prof = {"platform": "x86", "ram_gib": 62.0, "ram_class": 64, "gpus": [dict(gpu, memory_gib=54.0, max_memory_gib=54.0)]}
small_prof = {"platform": "x86", "ram_gib": 30.5, "ram_class": 32, "gpus": [dict(gpu, memory_gib=22.9, max_memory_gib=22.9)]}
with patch.object(a, "recipe_installed", return_value=False):
  on_big = {row["id"]: row for row in a.recipe_listing(dedup_catalog, big_prof)}
  on_small = {row["id"]: row for row in a.recipe_listing(dedup_catalog, small_prof)}
check("a bigger machine offers only the stronger build of the same model",
      on_big["model-big"]["available"] and not on_big["model-small"]["available"]
      and "model-big" in on_big["model-small"]["reason"], repr(on_big))
check("a smaller machine falls back to the smaller build alone",
      on_small["model-small"]["available"] and not on_small["model-big"]["available"], repr(on_small))
with patch.object(a, "recipe_installed", side_effect=lambda rid: rid == "model-small"):
  on_big_installed = {row["id"]: row for row in a.recipe_listing(dedup_catalog, big_prof)}
check("an already-installed fallback stays listed alongside the stronger build",
      on_big_installed["model-big"]["available"] and on_big_installed["model-small"]["available"], repr(on_big_installed))
# An experimental recipe is kept for trying by name: listed once set up or when asked for, never recommended.
trial = dict(copy.deepcopy(recipe), experimental=True)
trial_catalog = SimpleNamespace(root=str(repo), recipes={"model": recipe, "trial": trial}, runtimes=both, best=images_best)
check("experimental takes only true or false", any(".experimental: true or false" in e for e in a.validate_recipe("trial", dict(trial, experimental="yes"), {"server": runtime})))
check("an experimental recipe is otherwise valid", not a.validate_recipe("trial", trial, {"server": runtime}))
check("best.toml can't recommend an experimental recipe", any("'trial' is experimental" in e for e in a.validate_best({"class": [dict(entry, recipes=["trial", "model"])]}, {"model": recipe, "trial": trial})))
with patch.object(a, "recipe_installed", return_value=False):
  hidden = {row["id"] for row in a.recipe_listing(trial_catalog, profile)}
  asked = {row["id"]: row for row in a.recipe_listing(trial_catalog, profile, experimental=True)}
check("the listing leaves an experimental recipe out", hidden == {"model"}, repr(hidden))
check("asked for, it lists it and says it's experimental", asked["trial"]["experimental"] and not asked["model"]["experimental"], repr(asked))
with patch.object(a, "recipe_installed", side_effect=lambda rid: rid == "trial"):
  installed = {row["id"] for row in a.recipe_listing(trial_catalog, profile)}
check("once set up, it's listed so it can be started, stopped and removed", installed == {"model", "trial"}, repr(installed))
printed, saved = [], []
with patch.object(a, "load_catalog", return_value=trial_catalog), patch.object(a, "machine_profile", return_value=profile), \
     patch.object(a, "running_models", return_value=set()), patch.object(a, "recipe_installed", return_value=False), \
     patch.object(a, "write_list_cache", side_effect=saved.append), patch("builtins.print", side_effect=printed.append):
  a.cmd_list(["--json", "--experimental"])
shown = {row["id"] for row in json.loads(printed[0])}
check("list --experimental shows it, and the panel's saved listing still leaves it out",
      shown == {"model", "trial"} and len(saved) == 1 and {row["id"] for row in saved[0]} == {"model"}, repr((shown, saved)))
# A retired recipe is no longer offered: left out like an experimental one, and kept so a setup keeps working.
old = dict(copy.deepcopy(recipe), retired=True)
old_catalog = SimpleNamespace(root=str(repo), recipes={"model": recipe, "trial": trial, "old": old}, runtimes=both, best=images_best)
check("retired takes only true or false", any(".retired: true or false" in e for e in a.validate_recipe("old", dict(old, retired="yes"), {"server": runtime})))
check("a retired recipe is otherwise valid", not a.validate_recipe("old", old, {"server": runtime}))
check("a recipe is experimental or retired, not both", any("experimental or retired, not both" in e for e in a.validate_recipe("old", dict(old, experimental=True), {"server": runtime})))
check("best.toml can't recommend a retired recipe", any("'old' is retired" in e for e in a.validate_best({"class": [dict(entry, recipes=["old", "model"])]}, {"model": recipe, "old": old})))
with patch.object(a, "recipe_installed", return_value=False):
  hidden = {row["id"] for row in a.recipe_listing(old_catalog, profile)}
  asked = {row["id"]: row for row in a.recipe_listing(old_catalog, profile, retired=True)}
check("the listing leaves a retired recipe out", hidden == {"model"}, repr(hidden))
check("asked for retired, it lists that alone and says it's retired", set(asked) == {"model", "old"} and asked["old"]["retired"] and not asked["model"]["retired"], repr(asked))
with patch.object(a, "recipe_installed", side_effect=lambda rid: rid == "old"):
  installed = {row["id"] for row in a.recipe_listing(old_catalog, profile)}
check("once set up, a retired recipe is listed so it can be started, stopped and removed", installed == {"model", "old"}, repr(installed))
printed, saved = [], []
with patch.object(a, "load_catalog", return_value=old_catalog), patch.object(a, "machine_profile", return_value=profile), \
     patch.object(a, "running_models", return_value=set()), patch.object(a, "recipe_installed", return_value=False), \
     patch.object(a, "write_list_cache", side_effect=saved.append), patch("builtins.print", side_effect=printed.append):
  a.cmd_list(["--json", "--retired"])
shown = {row["id"] for row in json.loads(printed[0])}
check("list --retired shows it, and the panel's saved listing still leaves it out",
      shown == {"model", "old"} and len(saved) == 1 and {row["id"] for row in saved[0]} == {"model"}, repr((shown, saved)))
# Video and music models are recommended beside them the same way, each kind its own use best.
check("best.toml lists video models of their own", not a.validate_best({"class": [dict(entry, videos=["clip"])]}, {"model": recipe, "clip": clip}))
check("best.toml lists a class with media models but no text one", not a.validate_best({"class": [dict({k: v for k, v in entry.items() if k != "recipes"}, videos=["clip"])]}, {"model": recipe, "clip": clip}))
check("best.toml rejects a class that recommends nothing", any("at least one model" in e for e in a.validate_best({"class": [dict(entry, recipes=[])]}, {"model": recipe})))
check("a kind's list takes only that kind", any(".music: 'picture' is of kind image, not music" in e for e in a.validate_best({"class": [dict(entry, music=["picture"])]}, {"model": recipe, "picture": picture})))
raises("a kind with nothing recommended says so", lambda: a.resolve_best(img_catalog, profile, gpu, field="videos"), "No video model is recommended")
said = []
with patch.object(a, "load_catalog", return_value=img_catalog), patch.object(a, "machine_profile", return_value=profile), patch.object(a, "say", side_effect=said.append):
  asked = a.cmd_use(["best"])
check("best alone names each kind's pick and sets nothing up", asked == 2 and "  text: model" in said and "  image: picture" in said and "  video: none for this machine yet" in said, repr(said))

# Status rows say what kind of model they are; an image model's has no context.
picture_manifest = {"schema": 1, "recipe_id": "picture", "recipe": picture, "runtime": sd_rt, "runtime_id": "sd", "image": sd_rt["image"],
                    "variant": picture_variant, "files": picture_files, "roles": picture_roles, "fileset": "f" * 16, "gpu": gpu["family"], "profile": profile}
# An image model's start measures nothing, since its server reads its weights in with the
# first image: it holds its measured peak. A text model holds what its start took.
check("an image model's memory is its peak, not what its start took",
      a.memory_used_gib(dict(picture_manifest, observed={"load_seconds": 4, "ram_used_gib": -1.2})) == picture_variant["peak_ram_gib"])
check("a text model's memory is what its start took",
      a.memory_used_gib({"recipe": recipe, "variant": {"peak_ram_gib": 40}, "observed": {"ram_used_gib": 38.2}}) == 38.2)
with patch.object(a, "unit_props", return_value={}), patch.object(a, "free_memory", return_value=(64, 48)):
  picture_row = a.model_record("picture", picture_manifest, profile)
  text_row = a.model_record("current", manifest, profile)
check("an image model's status row says so, without a context, and fits by its memory", (picture_row["kind"], picture_row["ctx"], picture_row["fits"], picture_row["endpoint"]) == ("image", None, True, a.endpoint("picture")), repr(picture_row))
check("a text model's status row says text", text_row["kind"] == "text" and text_row["ctx"] == variant["ctx"])
# An edit model says so in its status row; only an image model may be one.
editor_manifest = dict(picture_manifest, recipe=dict(picture, edit=True))
with patch.object(a, "unit_props", return_value={}), patch.object(a, "free_memory", return_value=(64, 48)):
  check("an edit model's status row says it edits", a.model_record("picture", editor_manifest, profile)["edit"] is True and not picture_row["edit"])
check("only an image model can be an edit model", any(".edit" in e for e in a.validate_recipe("model", dict(recipe, edit=True), {"server": runtime})))
a.begin_installing("picture-setup", picture)
check("an image model being set up is listed as one", json.loads(Path(a.state_dir(), "status.json").read_text())["models"]["picture-setup"]["kind"] == "image")
a.clear_installing("picture-setup")
check("an installing row from before kinds reads as text", a.install_row("old", {"phase": "waiting"})["kind"] == "text")

# An image model is never the agents' model: not the last one, not the default, not an overlay or pi provider.
Path(a.state_dir(), "last").unlink(missing_ok=True)
a.save_model(picture_manifest)
check("starting an image model doesn't make it the last one started", not Path(a.state_dir(), "last").exists())
a.write_atomic(os.path.join(a.state_dir(), "last"), "picture\n")
check("a last that names an image model isn't the agents' default", a.last_model_id() is None)
with patch.object(a, "known_models", return_value=["picture"]), patch.object(a, "unit_active", return_value=True):
  raises("an agent launch never falls back to a running image model", lambda: a.resolve_model(running_only=True), "No local model is running")
  raises("an image model named for an agent is refused", lambda: a.overlay_target("picture"), "image model")
  check("pi's providers leave image models out", a.local_targets() == [])
  check("start and stop still find an image model by name", a.resolve_model("picture", running_only=True)["recipe_id"] == "picture")
os.unlink(os.path.join(a.state_dir(), "last"))

# An image model is shared on the tailnet like a text one, for Fantasi on the other machines.
with patch.object(a, "shared_recipes", return_value=["picture"]):
  listed = a.shared_listing({"models": {"picture": dict(picture_row, edit=True)}, "ram_total_gib": 64}, img_catalog, profile, "me.tail.ts.net")["models"]
  check("a shared image model is listed with its kind and whether it edits", [(m["recipe"], m["kind"], m["edit"]) for m in listed] == [("picture", "image", True)], repr(listed))
  handlers.clear()
  with patch.object(a, "tailscale", side_effect=fake_tailscale), patch.object(a, "tailscale_status", return_value=ts_status), patch.object(a, "systemctl", return_value=SimpleNamespace(returncode=0, stderr="")):
    a.publish_shares({"models": {"picture": picture_row}, "ram_total_gib": 64}, img_catalog, profile, force=True)
  check("and gets a tailnet route", "/local-ai/picture" in handlers, repr(handlers))
# A paired host lists each shared model's speeds for the GPU that model runs on: single-node
# recipes match the single GPU rather than the pair GPU that outranks it in choose_gpu().
pair_meas_dir = Path(os.environ["HOME"], "pair-shared-catalog")
(pair_meas_dir / "measurements").mkdir(parents=True)
(pair_meas_dir / "measurements" / "single.toml").write_text("[long]\ndecode_tok_s = 52.2\nprefill_tok_s = 3033.0\n")
(pair_meas_dir / "measurements" / "pair.toml").write_text("[long]\ndecode_tok_s = 64.8\nprefill_tok_s = 1095.5\n")
(pair_meas_dir / "measurements" / "cpu.toml").write_text("[long]\ndecode_tok_s = 12.5\nprefill_tok_s = 80.0\n")
single_rt = dict(runtime, gpus=["nvidia-gb10"])
pair_rt = dict(runtime, gpus=["nvidia-gb10-pair"], pair=True)
cpu_rt = dict(runtime, gpus=["cpu"], run={"tmpfs": "/tmp"})
single_rec = dict(recipe, runtime="single", hardware=[{"gpu": "nvidia-gb10", "measured": "measurements/single.toml"}])
pair_rec = dict(recipe, runtime="pair", hardware=[{"gpu": "nvidia-gb10-pair", "measured": "measurements/pair.toml"}])
cpu_rec = dict(recipe, runtime="cpu", hardware=[{"gpu": "cpu", "measured": "measurements/cpu.toml"}])
pair_shared_catalog = SimpleNamespace(root=str(pair_meas_dir),
                                      recipes={"single": single_rec, "pair": pair_rec, "cpu-only": cpu_rec},
                                      runtimes={"single": single_rt, "pair": pair_rt, "cpu": cpu_rt}, best={"class": []})
pair_shared_profile = {"platform": "aarch64-gb10", "ram_class": 128, "ram_gib": 128, "gpus": [
  {"vendor": "nvidia", "family": "nvidia-gb10", "render_node": "/dev/dri/renderD128", "memory_gib": 110, "max_memory_gib": 110, "unified": True},
  {"vendor": "nvidia", "family": "nvidia-gb10-pair", "render_node": "/dev/dri/renderD128", "memory_gib": 220, "max_memory_gib": 220, "unified": True, "nodes": 2},
]}
shared_rows = {rid: dict(text_row, recipe=rid, name=rid) for rid in ("single", "pair", "cpu-only")}
with patch.object(a, "shared_recipes", return_value=["single", "pair", "cpu-only"]), patch.object(a, "load_model", return_value=None):
  pair_listed = {m["recipe"]: (m["decode_tok_s"], m["prefill_tok_s"]) for m in a.shared_listing({"models": shared_rows, "ram_total_gib": 128}, pair_shared_catalog, pair_shared_profile, "spark.tail.ts.net")["models"]}
check("a paired host publishes each shared model's own GPU measurement, including single-node and CPU recipes",
      pair_listed == {"single": (52.2, 3033.0), "pair": (64.8, 1095.5), "cpu-only": (12.5, 80.0)}, repr(pair_listed))
with patch.object(a, "shared_recipes", return_value=["single"]), patch.object(a, "load_model", return_value={"gpu": "nvidia-gb10", "recipe": single_rec, "runtime": single_rt, "variant": variant}):
  single_listed = a.shared_listing({"models": {"single": shared_rows["single"]}, "ram_total_gib": 128}, pair_shared_catalog, pair_shared_profile, "spark.tail.ts.net")["models"]
check("a shared model's manifest GPU selects its measurement on a paired host",
      [(m["decode_tok_s"], m["prefill_tok_s"]) for m in single_listed] == [(52.2, 3033.0)], repr(single_listed))
with patch.object(a, "shared_recipes", return_value=["cpu-only"]), patch.object(a, "load_model", return_value={"gpu": "cpu", "recipe": cpu_rec, "runtime": cpu_rt, "variant": variant}):
  cpu_listed = a.shared_listing({"models": {"cpu-only": shared_rows["cpu-only"]}, "ram_total_gib": 128}, pair_shared_catalog, {"platform": "x86", "ram_class": 128, "ram_gib": 128, "gpus": []}, "host.tail.ts.net")["models"]
check("a host with no supported GPU still publishes a shared CPU model's measurement",
      [(m["decode_tok_s"], m["prefill_tok_s"]) for m in cpu_listed] == [(12.5, 80.0)], repr(cpu_listed))
peer = a.clean_listing({"schema": a.NETWORK_SCHEMA, "models": [dict(listed[0]), {"recipe": "old", "name": "Old"}]}, "me.tail.ts.net")["models"]
check("a peer's listing keeps an image model's kind, and a row without one is text", [(m["kind"], m["edit"]) for m in peer] == [("image", True), ("text", False)], repr(peer))
image_remote = a.remote_record("spark", "spark.tail.ts.net", peer[0], True, 128, 0)
with patch.object(a, "resolve_remote", return_value=image_remote):
  raises("a paired host's image model is refused as an agent's", lambda: a.overlay_target("spark/picture"), "image model")
with patch.object(a, "load_remotes", return_value={"spark": {"host": "spark"}}), patch.object(a, "load_network", return_value={"hosts": {}, "models": {"spark/picture": image_remote}}):
  check("nor is it one of pi's providers", a.remote_targets() == [])

# sd-server lists no recipe id: it's ready when its model status, asked with the key, says so.
status_url = f"http://127.0.0.1:{a.port_for('picture')}/sdcpp/v1/models/status"
with patch.object(a, "http_status", side_effect=[(401, b""), (200, b'{"state": "ready", "model": "sd-cpp-local"}')]) as http:
  a.post_start_checks("picture", "test-key", sd_rt)
check("an sd-server is checked by its model status, without and then with the key",
      [call.args for call in http.call_args_list] == [(status_url,), (status_url, "test-key")], repr(http.call_args_list))
for name, answers, message in (("that answers without the key", [(200, b'{"state": "ready"}')], "doesn't enforce its key"),
                               ("whose model isn't ready", [(401, b""), (200, b'{"state": "loading"}')], "loading, not ready"),
                               ("that answers no status", [(401, b""), (200, b"not json")], "unknown state")):
  with patch.object(a, "http_status", side_effect=answers):
    raises(f"an sd-server {name} is refused", lambda: a.post_start_checks("picture", "test-key", sd_rt), message)
os.unlink(a.model_manifest_path("picture"))

# A music model runs on acestep.cpp's ace-server, which takes the folder holding its models and
# loads one only when asked for a song: it's ready when it has found every model a song takes.
ace_rt = {"image": "registry.example/test/ace@sha256:" + "e" * 64, "tier": "omarchy", "api": "acestep", "port": 8085,
          "health": {"probe": ["curl", "http://127.0.0.1:8085/health"]}, "command": ["ace-server"],
          "reserved": ["--port", "ACE_API_KEY"], "run": {"devices": ["render"], "tmpfs": "/tmp"}, "gpus": ["amd"],
          "inputs": {"model": {"arg": "--models"}, "key": {"env": "ACE_API_KEY"}}}
song = {"name": "Song", "kind": "music", "runtime": "ace", "hardware": recipe["hardware"],
        "artifacts": [{"repo": "test/ace", "revision": "a" * 40, "license": "mit", "files": ["lm.gguf", "dit.gguf"], "dest": "acestep", "model": "."}],
        "variant": [picture_variant]}
check("a music runtime and a music recipe validate, without an alias or a context", not a.validate_runtime("ace", ace_rt) and not a.validate_recipe("song", song, {"ace": ace_rt}),
      "\n".join(a.validate_runtime("ace", ace_rt) + a.validate_recipe("song", song, {"ace": ace_rt})))
ace_tree = [{"type": "file", "path": name, "size": 1, "lfs": {"oid": oid * 64, "size": 1}} for name, oid in (("lm.gguf", "7"), ("dit.gguf", "8"))]
with patch.object(a, "hf_tree", return_value=ace_tree):
  song_files, song_roles = a.build_artifacts(song)
_, song_argv, _, _ = a.server_invocation(ace_rt, song, picture_variant, song_roles, "song")
check("a music model's folder reaches the server", song_argv == ["ace-server", "--models", "/models/acestep"], repr(song_argv))
props_url = f"http://127.0.0.1:{a.port_for('song')}/props"
found = b'{"models": {"lm": ["lm.gguf"], "embedding": ["te.gguf"], "dit": ["dit.gguf"], "vae": ["vae.gguf"]}}'
with patch.object(a, "http_status", side_effect=[(401, b""), (200, found)]) as http:
  a.post_start_checks("song", "test-key", ace_rt)
check("an ace-server is checked by the models it found, without and then with the key",
      [call.args for call in http.call_args_list] == [(props_url,), (props_url, "test-key")], repr(http.call_args_list))
for name, answers, message in (("that answers without the key", [(200, found)], "doesn't enforce its key"),
                               ("missing a model a song takes", [(401, b""), (200, b'{"models": {"lm": ["lm.gguf"], "embedding": [], "dit": ["dit.gguf"]}}')], "found no embedding, vae model"),
                               ("that lists no models", [(401, b""), (200, b"not json")], "found no lm, embedding, dit, vae model")):
  with patch.object(a, "http_status", side_effect=answers):
    raises(f"an ace-server {name} is refused", lambda: a.post_start_checks("song", "test-key", ace_rt), message)

# Stable Audio 3 music runtime: api = "sa3" checks /props for dit, decoder, text_encoder, tokenizer,
# and vocals = false marks an instrumental-only music recipe.
sa3_cuda_rt = {"image": "registry.example/test/sa3-cuda@sha256:" + "d" * 64, "tier": "omarchy", "api": "sa3", "port": 8085,
               "health": {"probe": ["curl", "http://127.0.0.1:8085/health"]}, "command": ["sa3-server"],
               "reserved": ["--port", "SA3_API_KEY"], "run": {"devices": ["cdi-nvidia"], "tmpfs": "/tmp"}, "gpus": ["nvidia-gb10"],
               "inputs": {"model": {"arg": "--models"}, "key": {"env": "SA3_API_KEY"}}}
sa3_cuda_song = {"name": "Stable Audio 3 Medium", "vendor": "stability", "kind": "music", "vocals": False, "runtime": "sa3-cuda",
                 "hardware": [{"gpu": "nvidia-n1x", "measured": "measurements/model.toml"}],
                 "artifacts": [{"repo": "test/sa3", "revision": "a" * 40, "license": "other", "files": ["dit.gguf"], "dest": "sa3", "model": "."}],
                 "variant": [picture_variant]}
check("an sa3 music runtime and instrumental recipe validate",
      not a.validate_runtime("sa3-cuda", sa3_cuda_rt) and not a.validate_recipe("sa3-song-cuda", sa3_cuda_song, {"sa3-cuda": sa3_cuda_rt}),
      "\n".join(a.validate_runtime("sa3-cuda", sa3_cuda_rt) + a.validate_recipe("sa3-song-cuda", sa3_cuda_song, {"sa3-cuda": sa3_cuda_rt})))
check("vocals is only for music recipes", any(".vocals" in e for e in a.validate_recipe("model", dict(recipe, vocals=False), {"server": runtime})))
sa3_found = b'{"models": {"dit": ["dit.gguf"], "decoder": ["same.gguf"], "text_encoder": ["t5.gguf"], "tokenizer": ["vocab.gguf"]}}'
with patch.object(a, "http_status", side_effect=[(401, b""), (200, sa3_found)]):
  a.post_start_checks("sa3-song-cuda", "test-key", sa3_cuda_rt)
with patch.object(a, "http_status", side_effect=[(401, b""), (200, b'{"models": {"dit": ["dit.gguf"]}}')]):
  raises("an sa3-server missing parts is refused", lambda: a.post_start_checks("sa3-song-cuda", "test-key", sa3_cuda_rt), "found no decoder, text_encoder, tokenizer model")
sa3_inst_rec = a.begin_installing("sa3-song-cuda", sa3_cuda_song)
sa3_inst_row = a.install_row("sa3-song-cuda", sa3_inst_rec)
check("an installing instrumental music recipe keeps vocals = False",
      sa3_inst_rec.get("vocals") is False and sa3_inst_row.get("vocals") is False, repr((sa3_inst_rec, sa3_inst_row)))
a.clear_installing("sa3-song-cuda")

# A CPU-only music model (Stable Audio 3): runs with gpus = ["cpu"], no GPU devices, peak_gpu_gib = 0,
# vocals = false, stays available on machines with no supported GPU, and yields to a GPU build of
# the same model when one runs on the machine.
sa3_rt = {"image": "registry.example/test/sa3@sha256:" + "d" * 64, "tier": "omarchy", "api": "sa3", "port": 8085,
          "health": {"probe": ["curl", "http://127.0.0.1:8085/health"]}, "command": ["sa3-server"],
          "reserved": ["--port", "SA3_API_KEY"], "run": {"tmpfs": "/tmp"}, "gpus": ["cpu"],
          "inputs": {"model": {"arg": "--models"}, "key": {"env": "SA3_API_KEY"}}}
sa3_variant = {"peak_ram_gib": 4, "peak_gpu_gib": 0}
sa3_song = {"name": "Stable Audio 3 Medium", "vendor": "stability", "kind": "music", "vocals": False, "runtime": "sa3",
            "hardware": [{"gpu": "cpu", "measured": "measurements/model.toml"}],
            "artifacts": [{"repo": "test/sa3", "revision": "a" * 40, "license": "other", "files": ["dit.tflite"], "dest": "sa3", "model": "."}],
            "variant": [sa3_variant]}
check("a CPU music runtime and instrumental recipe validate",
      not a.validate_runtime("sa3", sa3_rt) and not a.validate_recipe("sa3-song", sa3_song, {"sa3": sa3_rt}),
      "\n".join(a.validate_runtime("sa3", sa3_rt) + a.validate_recipe("sa3-song", sa3_song, {"sa3": sa3_rt})))
check("a CPU runtime rejects GPU devices", any("can't use GPU devices" in e for e in a.validate_runtime("sa3", dict(sa3_rt, run={"devices": ["render"]}))))
cpu_only_profile = {"platform": "x86", "ram_class": 32, "ram_gib": 16, "gpus": []}
cpu_catalog = SimpleNamespace(root=str(repo), recipes={"model": recipe, "sa3-song": sa3_song, "sa3-song-cuda": sa3_cuda_song},
                              runtimes={"server": runtime, "sa3": sa3_rt, "sa3-cuda": sa3_cuda_rt}, best={"class": []})
cpu_rows = {row["id"]: row for row in a.recipe_listing(cpu_catalog, cpu_only_profile)}
check("a CPU recipe is available on a machine with no supported GPU, while GPU recipes stay unavailable",
      cpu_rows["sa3-song"]["available"] and cpu_rows["sa3-song"]["vocals"] is False and not cpu_rows["model"]["available"] and not cpu_rows["sa3-song-cuda"]["available"], repr(cpu_rows))
check("a CPU recipe doesn't fit when RAM is below peak plus desktop reserve",
      not {row["id"]: row for row in a.recipe_listing(cpu_catalog, dict(cpu_only_profile, ram_gib=8))}["sa3-song"]["available"])
gb10_profile = {"platform": "aarch64-gb10", "ram_class": 128, "ram_gib": 128,
                "gpus": [{"vendor": "nvidia", "family": "nvidia-gb10", "render_node": "/dev/dri/renderD128", "memory_gib": 110, "max_memory_gib": 110, "unified": True}]}
gb10_rows = {row["id"]: row for row in a.recipe_listing(cpu_catalog, gb10_profile)}
check("when a GPU build of the same model fits, the CPU fallback yields to it in listings",
      gb10_rows["sa3-song-cuda"]["available"] and not gb10_rows["sa3-song"]["available"], repr(gb10_rows))
sa3_manifest = {"schema": 1, "recipe_id": "sa3-song", "recipe": sa3_song, "runtime": sa3_rt, "runtime_id": "sa3", "image": sa3_rt["image"],
                "variant": sa3_variant, "files": [], "roles": {"model": "sa3"}, "fileset": "d" * 16, "gpu": "cpu", "profile": cpu_only_profile}
with patch.object(a, "unit_props", return_value={}), patch.object(a, "free_memory", return_value=(10, 16)):
  sa3_row = a.model_record("sa3-song", sa3_manifest, cpu_only_profile)
check("a CPU music model's status row carries vocals = False and fits by RAM",
      sa3_row["kind"] == "music" and sa3_row["vocals"] is False and sa3_row["fits"] is True, repr(sa3_row))
sa3_unit = a.render_container("sa3-song", sa3_song, sa3_rt, sa3_variant, {"model": "sa3"}, a.cpu_device(cpu_only_profile), "d" * 16)
check("a CPU container unit adds no GPU devices", "AddDevice=" not in sa3_unit and "Tmpfs=/tmp" in sa3_unit)

# Vulkan Stable Audio 3 runtime for AMD GPUs: render node, tmpfs, api = "sa3", multi-artifact dir.
sa3_vk_rt = {"image": "registry.example/test/sa3-vulkan@sha256:" + "e" * 64, "tier": "omarchy", "api": "sa3", "port": 8085,
             "health": {"probe": ["curl", "http://127.0.0.1:8085/health"]}, "command": ["sa3-server"],
             "reserved": ["--port", "SA3_API_KEY"], "run": {"devices": ["render"], "group_add": "keep-groups", "tmpfs": "/tmp"},
             "gpus": ["amd"], "inputs": {"model": {"arg": "--models"}, "key": {"env": "SA3_API_KEY"}}}
sa3_vk_variant = {"peak_ram_gib": 4, "peak_gpu_gib": 4}
sa3_vk_song = {"name": "Stable Audio 3 Medium", "vendor": "stability", "kind": "music", "vocals": False, "runtime": "sa3-vulkan",
               "hardware": [{"gpu": "amd-strix-halo", "measured": "measurements/model.toml"}],
               "artifacts": [
                 {"repo": "test/sa3-m", "revision": "a" * 40, "license": "other", "files": ["dit.gguf", "same.gguf"], "dest": "sa3", "model": "."},
                 {"repo": "test/t5gemma", "revision": "b" * 40, "license": "gemma", "files": ["t5.gguf", "vocab.gguf"], "dest": "sa3"},
               ],
               "variant": [sa3_vk_variant]}
check("an AMD Vulkan sa3 runtime and multi-artifact instrumental recipe validate",
      not a.validate_runtime("sa3-vulkan", sa3_vk_rt) and not a.validate_recipe("sa3-vk-song", sa3_vk_song, {"sa3-vulkan": sa3_vk_rt}),
      "\n".join(a.validate_runtime("sa3-vulkan", sa3_vk_rt) + a.validate_recipe("sa3-vk-song", sa3_vk_song, {"sa3-vulkan": sa3_vk_rt})))
amd_profile = {"platform": "x86", "ram_class": 192, "ram_gib": 188,
               "gpus": [{"vendor": "amd", "family": "amd-strix-halo", "render_node": "/dev/dri/renderD128",
                         "vram_gib": 0.5, "gtt_gib": 180, "unified": True, "memory_gib": 180.5, "max_memory_gib": 180.5, "sysfs": ""}]}
sa3_vk_unit = a.render_container("sa3-vk-song", sa3_vk_song, sa3_vk_rt, sa3_vk_variant, {"model": "sa3"}, amd_profile["gpus"][0], "e" * 16)
check("an AMD Vulkan sa3 container unit mounts the render node and tmpfs",
      "AddDevice=/dev/dri/renderD128" in sa3_vk_unit and "Tmpfs=/tmp" in sa3_vk_unit and "--models" in sa3_vk_unit)


# A runtime's warm-up: a malformed one is refused; one prompt of the model's own vocabulary goes
# out with the key, nothing is sent without a tokenizer, and a failure doesn't fail the start.
for bad in ({"prompts": 0, "pieces": 20000}, {"prompts": 1}, {"prompts": True, "pieces": 20000}):
  changed = dict(runtime, warmup=bad)
  check(f"runtime rejects warmup {bad}", any(".warmup" in e for e in a.validate_runtime("server", changed)))
check("an OpenAI runtime takes a warm-up", not a.validate_runtime("server", dict(runtime, warmup={"prompts": 1, "pieces": 20000})))
tok = Path(a.data_dir(), "models", "warm", "f" * 16, "model")
# A rank folder's tokenizer.json links into /models, which only resolves inside the container.
rank_tok = Path(a.data_dir(), "models", "warm", "f" * 16 + ".abc.rank0", "model")
rank_tok.mkdir(parents=True, exist_ok=True)
(rank_tok / "tokenizer.json").unlink(missing_ok=True)
(rank_tok / "tokenizer.json").symlink_to("/models/model/tokenizer.json")
tok.mkdir(parents=True, exist_ok=True)
(tok / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"\u0120river": 0, "stone": 1, "<|eos|>": 2}}}))
sent = []
class Reply:
  def __enter__(self): return self
  def __exit__(self, *exc): return False
  def read(self, *a): return b"{}"
def fake_urlopen(request, timeout=None):
  sent.append(request)
  return Reply()
published = []
def fake_urlopen_checking(request, timeout=None):
  published.append(a.warming("warm"))
  return fake_urlopen(request, timeout)
with patch.object(a.urllib.request, "urlopen", side_effect=fake_urlopen_checking):
  a.warm_up("warm", "f" * 16, "test-key", {"warmup": {"prompts": 1, "pieces": 300}}, publish=lambda: published.append("published"))
check("the model shows loading while it warms up, and running after", published[:2] == ["published", True] and not a.warming("warm"), repr(published))
body = json.loads(sent[0].data) if sent else {}
text = body.get("messages", [{}])[0].get("content", "")
check("the warm-up sends one prompt of the model's vocabulary with the key",
      len(sent) == 1 and sent[0].get_header("Authorization") == "Bearer test-key" and " river" in text and "<|eos|>" not in text and body.get("max_tokens") == 1)
sent.clear()
with patch.object(a.urllib.request, "urlopen", side_effect=fake_urlopen):
  a.warm_up("missing", "f" * 16, "test-key", {"warmup": {"prompts": 1, "pieces": 300}})
check("no tokenizer, no warm-up prompt", not sent)
with patch.object(a.urllib.request, "urlopen", side_effect=a.urllib.error.URLError("refused")):
  a.warm_up("warm", "f" * 16, "test-key", {"warmup": {"prompts": 2, "pieces": 300}})
check("a failed warm-up leaves no marker behind", not a.warming("warm"))
Path(a.warming_path("warm")).write_text("999999999\n")
check("a marker whose process is gone holds nothing back", not a.warming("warm"))
# An app holds a model while it runs; stop --unless-used leaves a held model, or one a tailnet
# machine used lately, and stops the rest. The hold ends when the holder's input closes.
holder = subprocess.Popen([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, sys.argv[1]); import agent_local as a; a.resolve_model = lambda rid: {'recipe_id': rid}; sys.exit(a.main(['hold', 'held']))", os.environ["ENGINE_ROOT"]],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
for _ in range(200):
  if a.hold_exclusively("held", []) is False:
    break
  __import__("time").sleep(0.05)
Path(a.clients_path()).write_text(json.dumps({"models": {"shared": 1}}))
stopped = []
def stop_unless_used(*rids):
  with patch.object(a, "resolve_model", side_effect=lambda rid: {"recipe_id": rid}), patch.object(a, "running_models", return_value={"held", "shared", "idle"}), \
       patch.object(a, "load_catalog", return_value={}), patch.object(a, "machine_profile", return_value={}), patch.object(a, "write_status"), \
       patch.object(a, "evict_idle_weights"), patch.object(a, "stop_unit", side_effect=stopped.append):
    return a.cmd_stop(["--unless-used", *rids])
stop_unless_used()
check("stop --unless-used stops only what no app holds and no tailnet machine uses", stopped == ["idle"], repr(stopped))
stopped.clear()
stop_unless_used("idle", "idle")
check("a model named twice is stopped once", stopped == ["idle"], repr(stopped))
holder.communicate(b"", timeout=10)
stopped.clear()
stop_unless_used("held", "shared")
check("a hold ends when the holder's input closes", holder.returncode == 0 and stopped == ["held"], repr((holder.returncode, stopped)))
os.unlink(a.clients_path())
PY
