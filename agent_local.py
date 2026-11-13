"""Local models for Omarchy agents: recipes from the packaged catalog, run as Podman Quadlet units.

The catalog (omacom/omarchy-local-ai) chooses what runs: recipes, runtime definitions and the
per-machine recommendation in best.toml. This module owns the boundary every choice runs inside:
an internal network with no egress, a read-only root, dropped capabilities, allowlisted devices
and limits, the key as a Podman secret, typed unit generation, and post-start checks that the
server answers only with the key. See docs/engine.md.
"""

import argparse
import concurrent.futures
import contextlib
import fcntl
import fnmatch
import hashlib
import ipaddress
import itertools
import json
import os
import pty
import re
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tomllib
import urllib.error
import urllib.request

SCHEMA = 1
BASE_PORT = 11435
UNIT_PREFIX = "omarchy-agent-local-model-"
NETWORK_UNIT = "omarchy-agent-local.network"
NETWORK_NAME = "omarchy-agent-local"
SECRET = "omarchy-agent-local-key"
KEY_ENV = "OMARCHY_LOCAL_KEY"
KEY_PLACEHOLDER = "@OMARCHY_LOCAL_KEY@"
RESERVE_GIB = 8
# How often a setup rewrites its progress for the panel, and the window its speed is averaged over.
PROGRESS_PUBLISH_SECONDS = 0.5
PROGRESS_WINDOW_SECONDS = 4
NOT_ENOUGH_MEMORY = "Not enough memory to start model"
RAM_CLASSES = (32, 64, 96, 128, 192, 256, 512)
# A discrete GPU is classed by its own VRAM (`vram_class`), not by the system's RAM: the RAM
# only has to hold the host side (`peak_ram_gib`, checked beside it). Unified-memory GPUs keep `ram_class`.
VRAM_CLASSES = (8, 12, 16, 24, 32, 48, 80, 96)
STANDARD_INPUTS = ("model", "draft", "ctx", "alias", "key", "temperature", "top_p", "top_k")
GFX_FAMILIES = {"gfx1151": "amd-strix-halo", "gfx1150": "amd-strix-point"}
# NVIDIA platforms whose GPU shares the system's memory. The N1x is the GB10's GPU in a
# laptop: it keeps its own family name so a measurement records which machine it came from,
# but it runs what a GB10 with the same memory runs.
NVIDIA_UNIFIED_FAMILIES = {"aarch64-gb10": "nvidia-gb10", "aarch64-n1x": "nvidia-n1x"}
# Families that share recipes, qualification and recommendations: only memory sets them apart.
GPU_CLASSES = {"nvidia-n1x": "nvidia-gb10"}


def gpu_class(family):
  return GPU_CLASSES.get(family, family)

ID_RE = re.compile(r"^[a-z0-9]+([.-][a-z0-9]+)*$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
IMAGE_RE = re.compile(r"^[a-z0-9.-]+(:[0-9]+)?(/[a-z0-9._-]+)+@sha256:[0-9a-f]{64}$")
ENV_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
ARG_RE = re.compile(r"^-{1,2}[A-Za-z0-9][A-Za-z0-9-]*$")
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

# Run options a runtime may ask for, each mapped to exactly one Quadlet line. Read-only root,
# capabilities and privileges are not options: every container gets the same. The network is
# the internal one, except for runtimes whose ranks span two machines (network = "fabric"):
# they share the host's network namespace to reach each other over the ConnectX-7 link, and
# their API is bound to loopback by the engine.
RUN_OPTIONS = {
  "devices": ("render", "kfd", "cdi-nvidia", "infiniband"),
  "group_add": ("keep-groups",),
  "memlock": ("unlimited",),
  "network": ("fabric",),
}
MAX_SHM_GIB = 16
# A machine joined to another of its kind by a direct ConnectX-7 cable runs recipes across both
# as one GPU family of its own, with each machine's memory as the per-node figure.
PAIR_FAMILIES = {"nvidia-gb10": "nvidia-gb10-pair"}
# The link's NCCL settings, which the engine finds on each machine at every start.
PAIR_NCCL = ("NCCL_SOCKET_IFNAME", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX")


class LocalError(Exception):
  """A failure with its full explanation, and optionally a one-line reason
  short enough for the agents panel to show where a model's progress bar was."""

  def __init__(self, message, short=None):
    super().__init__(message)
    self.short = short or str(message).splitlines()[0]


# ---------------------------------------------------------------- paths


def home():
  return os.path.expanduser("~")


def state_dir():
  return os.path.join(home(), ".local", "state", "omarchy", "agent-local")


def data_dir():
  return os.path.join(home(), ".local", "share", "omarchy", "agent-local")


def config_dir():
  return os.path.join(home(), ".config", "omarchy", "agent-local")


def quadlet_dir():
  return os.path.join(home(), ".config", "containers", "systemd")


def key_path():
  return os.path.join(config_dir(), "key")


def ensure_dir(path, mode=0o755):
  os.makedirs(path, mode=mode, exist_ok=True)
  return path


def write_atomic(path, text, mode=0o644):
  ensure_dir(os.path.dirname(path))
  fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-")
  try:
    with os.fdopen(fd, "w") as handle:
      handle.write(text)
    os.chmod(tmp, mode)
    os.replace(tmp, path)
  except BaseException:
    if os.path.exists(tmp):
      os.unlink(tmp)
    raise


def gib(value):
  return value / 2**30


def say(message):
  print(message, flush=True)


# ---------------------------------------------------------------- machine profile


def _read(path, default=""):
  try:
    with open(path) as handle:
      return handle.read().strip()
  except OSError:
    return default


def _kv_file(path):
  values = {}
  for line in _read(path).splitlines():
    parts = line.split()
    if len(parts) == 2:
      values[parts[0]] = parts[1]
  return values


def meminfo():
  values = {}
  for line in _read("/proc/meminfo").splitlines():
    name, _, rest = line.partition(":")
    number = rest.strip().split()
    if number:
      values[name] = int(number[0]) * 1024
  return values


def platform_name():
  try:
    return subprocess.run(["omarchy-hw-platform"], capture_output=True, text=True, timeout=10).stdout.strip() or "x86"
  except (OSError, subprocess.SubprocessError):
    return "x86"


def gfx_name(version):
  version = int(version)
  return f"gfx{version // 10000}{(version // 100) % 100:x}{version % 100:x}"


def amd_gpus(kfd_root="/sys/class/kfd/kfd/topology/nodes", drm_root="/sys/class/drm"):
  gpus = []
  try:
    nodes = sorted(os.listdir(kfd_root))
  except OSError:
    return gpus
  for node in nodes:
    props = _kv_file(os.path.join(kfd_root, node, "properties"))
    version = int(props.get("gfx_target_version", "0") or 0)
    if not version:
      continue
    minor = props.get("drm_render_minor", "")
    render = f"renderD{minor}"
    device = os.path.join(drm_root, render, "device")
    vram = int(_read(os.path.join(device, "mem_info_vram_total"), "0") or 0)
    gtt = int(_read(os.path.join(device, "mem_info_gtt_total"), "0") or 0)
    gfx = gfx_name(version)
    gpus.append({
      "vendor": "amd",
      "family": GFX_FAMILIES.get(gfx, f"amd-{gfx}"),
      "gfx": gfx,
      "render_node": f"/dev/dri/{render}",
      "vram_gib": round(gib(vram), 2),
      "gtt_gib": round(gib(gtt), 2),
      "unified": vram < 4 * 2**30,
      "memory_gib": round(gib(vram + gtt), 2),
      "sysfs": device,
    })
  return gpus


def nvidia_family(name):
  """`nvidia-rtx3090` from "NVIDIA GeForce RTX 3090"; a name without a model number keeps its words."""
  words = re.sub(r"^NVIDIA\s+(GeForce\s+)?", "", name.strip(), flags=re.I)
  return "nvidia-" + (re.sub(r"[^a-z0-9]+", "", words.lower()) or "gpu")


def nvidia_gpus(smi_output=None):
  """Discrete NVIDIA GPUs, one per `nvidia-smi` row of "name, memory.total MiB". Their
  memory is their own VRAM, apart from the system's, unlike the GB10's."""
  if smi_output is None:
    try:
      smi_output = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
      return []
  gpus = []
  for index, line in enumerate(smi_output.splitlines()):
    name, _, total = line.partition(",")
    try:
      mib = float(total)
    except ValueError:
      continue
    gpus.append({
      "vendor": "nvidia",
      "family": nvidia_family(name),
      "name": name.strip(),
      "nvidia_index": index,
      "render_node": "",
      "vram_gib": round(mib / 1024, 2),
      "gtt_gib": 0,
      "unified": False,
      "vram_class": vram_class(mib / 1024),
      "memory_gib": round(mib / 1024, 2),
      "sysfs": "",
    })
  return gpus


def nvidia_free_gib(index):
  try:
    out = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    return float(out.strip()) / 1024
  except (OSError, subprocess.SubprocessError, ValueError):
    return None


def ram_class(ram_gib):
  for size in RAM_CLASSES:
    if ram_gib <= size:
      return size
  return RAM_CLASSES[-1]


def vram_class(vram_gib):
  for size in VRAM_CLASSES:
    if vram_gib <= size:
      return size
  return VRAM_CLASSES[-1]


def memory_class(profile, gpu):
  """The class this GPU is matched on: its VRAM if it has its own, else the machine's RAM."""
  return gpu.get("vram_class") or profile["ram_class"]


def entry_class(entry):
  """The class a [[hardware]] or best.toml [[class]] entry names."""
  return entry.get("vram_class") or entry.get("ram_class")


def class_text(profile, gpu):
  return f"{gpu['vram_class']} GB VRAM" if gpu.get("vram_class") else f"{profile['ram_class']} GB"


def entry_class_text(entry):
  return f"{entry['vram_class']} GB VRAM" if entry.get("vram_class") else f"{entry.get('ram_class')} GB"


def class_errors(entry, where):
  """Exactly one of ram_class and vram_class, each from its list."""
  has_ram, has_vram = "ram_class" in entry, "vram_class" in entry
  if has_ram == has_vram:
    return [f"{where}: needs exactly one of ram_class (one of {RAM_CLASSES}) and vram_class (one of {VRAM_CLASSES})"]
  if has_ram and entry["ram_class"] not in RAM_CLASSES:
    return [f"{where}.ram_class: must be one of {RAM_CLASSES}"]
  if has_vram and entry["vram_class"] not in VRAM_CLASSES:
    return [f"{where}.vram_class: must be one of {VRAM_CLASSES}"]
  return []


def machine_profile():
  """This machine's GPUs and memory. The machine `link setup` joined to this one (pair.json)
  adds a pair GPU."""
  mem = meminfo()
  ram = gib(mem.get("MemTotal", 0))
  platform = platform_name()
  gpus = amd_gpus()
  if platform in NVIDIA_UNIFIED_FAMILIES:
    gpus.append({"vendor": "nvidia", "family": NVIDIA_UNIFIED_FAMILIES[platform], "render_node": "", "vram_gib": 0, "gtt_gib": 0, "unified": True, "memory_gib": round(ram, 2), "sysfs": ""})
  else:
    gpus.extend(nvidia_gpus())
  profile = {"platform": platform, "ram_gib": round(ram, 2), "ram_class": ram_class(ram), "gpus": gpus}
  pair = cached_pair(profile)
  if pair:
    local = next(gpu for gpu in gpus if gpu["family"] in PAIR_FAMILIES)
    # Memory figures are per node: the smaller machine bounds what each rank may hold.
    gpus.append(dict(local, family=PAIR_FAMILIES[local["family"]], nodes=2, memory_gib=round(min(ram, pair["ram_gib"]), 2), peer=pair["peer"]))
    profile["pair"] = pair
  return profile


def choose_gpu(profile, family=None):
  gpus = profile["gpus"]
  if family:
    matches = [gpu for gpu in gpus if gpu["family"] == family]
    if not matches:
      raise LocalError(f"No {family} GPU on this machine (found: {', '.join(g['family'] for g in gpus) or 'none'})")
    return max(matches, key=lambda gpu: gpu["memory_gib"])
  if not gpus:
    raise LocalError("No supported GPU found on this machine")
  # A pair outranks its own single node, whose memory per node it shares.
  rank = lambda gpu: (gpu.get("nodes", 1), gpu["memory_gib"])
  best = max(gpus, key=rank)
  rivals = {gpu["family"] for gpu in gpus if rank(gpu) == rank(best)}
  if len(rivals) > 1:
    raise LocalError(f"Several GPU families tie for the most memory ({', '.join(sorted(rivals))}); choose one with --gpu")
  return best


def recipe_gpu(profile, recipe, family=None):
  """The GPU a recipe runs on here: the one named, else the best of this machine's GPUs the
  recipe was measured on (a pair machine still runs single-node recipes on its own GPU)."""
  if family:
    return choose_gpu(profile, family)
  measured = {hw.get("gpu") for hw in recipe.get("hardware", [])}
  matches = [gpu for gpu in profile["gpus"] if gpu["family"] in measured]
  if matches:
    return max(matches, key=lambda gpu: (gpu.get("nodes", 1), gpu["memory_gib"]))
  return choose_gpu(profile)


# ---------------------------------------------------------------- paired machines
#
# Two DGX Sparks cabled through their ConnectX-7 ports run one model as two ranks: this machine
# serves rank 0 and the API, the peer runs rank 1. The peer is found on the cable, reached by
# key-based SSH (no password prompts: BatchMode), and needs rootless Podman with the NVIDIA CDI
# GPU and its RDMA devices. Nothing of the engine is installed there; this machine drives it.

def pair_key():
  return os.path.join(home(), ".ssh", "id_ed25519")


def pair_ssh_argv(tty=False):
  """SSH to the peer with the key `link setup` installed there, and only that key: an agent
  offering several (1Password's does) would spend the server's MaxAuthTries before it."""
  argv = ["ssh", "-t" if tty else "-T", "-o", "ConnectTimeout=4", "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=15"]
  if not tty:
    argv += ["-o", "BatchMode=yes"]
  if os.path.exists(pair_key()):
    argv += ["-o", "IdentitiesOnly=yes", "-i", pair_key()]
  return argv

# This machine's (or, sent over SSH, the peer's) side of the cable to an address: the netdev and
# source address the route takes, its RoCE device, the RoCE v2 GID index holding that IPv4
# address (an all-zero or v1 entry makes NCCL fail a minute in), and every RoCE device with an
# IPv4 RoCE v2 GID at the same index, the link's own first: a Spark's QSFP port is two PCIe
# links, each its own netdev and rail. Standard library only, so it runs unchanged on the peer.
FABRIC_SIDE = r'''
import ipaddress, json, os, subprocess
def fabric_side(peer):
  route = subprocess.run(["ip", "-o", "-4", "route", "get", peer], capture_output=True, text=True).stdout.split()
  if "dev" not in route or "src" not in route:
    return None
  dev, src = route[route.index("dev") + 1], route[route.index("src") + 1]
  base = "/sys/class/infiniband"
  def hca_of(netdev):
    try:
      return sorted(os.listdir(f"/sys/class/net/{netdev}/device/infiniband"))[0]
    except (OSError, IndexError):
      return None
  hca = hca_of(dev)
  if not hca:
    return {"addr": src, "dev": dev, "hca": None, "gid": None, "hcas": []}
  def v2_ipv4(device, index):
    try:
      gid = open(f"{base}/{device}/ports/1/gids/{index}").read().strip()
      kind = open(f"{base}/{device}/ports/1/gid_attrs/types/{index}").read().strip()
    except OSError:
      return None
    return gid if gid.startswith("0000:0000:0000:0000:0000:ffff:") and "v2" in kind else None
  want = "0000:0000:0000:0000:0000:ffff:" + "{:02x}{:02x}:{:02x}{:02x}".format(*ipaddress.IPv4Address(src).packed)
  gid = next((index for index in sorted(os.listdir(f"{base}/{hca}/ports/1/gids"), key=int) if v2_ipv4(hca, index) == want), None)
  hcas = [hca]
  if gid is not None:
    for other in sorted(os.listdir(base)):
      if other != hca and v2_ipv4(other, gid):
        hcas.append(other)
  def netdevs(device):
    try:
      return sorted(os.listdir(f"{base}/{device}/device/net"))
    except OSError:
      return []
  return {"addr": src, "dev": dev, "hca": hca, "gid": gid, "hcas": hcas, "devs": [d for h in hcas for d in netdevs(h)]}
def ports():
  """Each RDMA-capable netdev with a carrier, and its IPv4 addresses as address/prefix."""
  links = json.loads(subprocess.run(["ip", "-j", "-4", "addr"], capture_output=True, text=True).stdout or "[]")
  found = {}
  for link in links:
    dev = link.get("ifname", "")
    try:
      carrier = open(f"/sys/class/net/{dev}/carrier").read().strip() == "1"
    except OSError:
      carrier = False
    if carrier and os.path.isdir(f"/sys/class/net/{dev}/device/infiniband"):
      found[dev] = [f"{a['local']}/{a['prefixlen']}" for a in link.get("addr_info", []) if a.get("family") == "inet"]
  return found
def node():
  name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
  mem = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in open("/proc/meminfo") if line.split()[1:2]}
  return {"gpu": name, "ram_total": mem.get("MemTotal", 0), "ram_available": mem.get("MemAvailable", 0), "home": os.path.expanduser("~"), "host": os.uname().nodename}
'''


def _fabric(peer):
  scope = {}
  exec(FABRIC_SIDE, scope)
  return scope["fabric_side"](peer)


def pair_ssh(peer, command, input=None, timeout=60, check=True):
  """Runs a shell command on the peer. Every value interpolated into it is shlex-quoted by the caller."""
  try:
    result = subprocess.run(pair_ssh_argv() + [peer, command], input=input, capture_output=True, text=True, timeout=timeout)
  except subprocess.TimeoutExpired:
    raise LocalError(f"The peer {peer} didn't answer within {timeout}s", short="Peer machine not answering")
  if check and result.returncode != 0:
    raise LocalError(f"On {peer}: {command.split()[0]} failed: {result.stderr.strip()[-800:]}", short="Failed on the peer machine")
  return result


def pair_python(peer, expression, timeout=20):
  """Evaluates FABRIC_SIDE's functions on the peer and returns the JSON result."""
  program = FABRIC_SIDE + f"\nprint(json.dumps({expression}))\n"
  result = pair_ssh(peer, "python3 -I -", input=program, timeout=timeout)
  return json.loads(result.stdout)


PEER_RE = re.compile(r"^([A-Za-z0-9._-]+@)?[A-Za-z0-9._:%-]+$")


def peer_address(peer):
  """The address part of an SSH target, user@address or address."""
  return peer.rsplit("@", 1)[-1]


def pair_path():
  return os.path.join(state_dir(), "pair.json")


def cached_pair(profile):
  if profile["platform"] not in NVIDIA_UNIFIED_FAMILIES or NVIDIA_UNIFIED_FAMILIES[profile["platform"]] not in PAIR_FAMILIES:
    return None
  try:
    with open(pair_path()) as handle:
      pair = json.load(handle)
  except (OSError, ValueError):
    return None
  return pair if isinstance(pair, dict) and pair.get("peer") else None


def cable_peer(peer):
  """The peer as SSH reaches it over the cable from now on: its address on a port both machines
  share, with the user it was given."""
  theirs, _ = link_rules(local_ports(), pair_python(peer, "ports()"), set(local_ports()))
  if not theirs:
    raise LocalError("The cable has no address on both ends: check that both ends of the link are up")
  user = peer.rsplit("@", 1)[0] + "@" if "@" in peer else ""
  return user + theirs[0].split("=", 1)[1]


def record_pair(profile, peer):
  """Records the machine link setup joined to this one, as SSH reaches it over the cable, once it
  shows a GB10 of this machine's RAM class with a RoCE device on the route to it."""
  node = pair_python(peer, "node()", timeout=8)
  if "GB10" not in node.get("gpu", "") or ram_class(gib(node["ram_total"])) != profile["ram_class"]:
    raise LocalError(f"{peer} isn't a DGX Spark of this one's memory ({node.get('gpu') or 'no GPU'}, {gib(node['ram_total']):.0f} GiB)")
  side = _fabric(peer_address(peer))
  if not side or not side.get("hca"):
    raise LocalError(f"The route to {peer_address(peer)} doesn't go over a ConnectX-7 port")
  pair = {"peer": peer, "host": node["host"], "home": node["home"], "ram_gib": round(gib(node["ram_total"]), 2)}
  write_atomic(pair_path(), json.dumps(pair, indent=2) + "\n")
  return pair


def pair_link(pair):
  """Both sides of the cable as the ranks use them now: rank 0 here, rank 1 on the peer.
  Rails are used only when both sides have as many."""
  here = _fabric(peer_address(pair["peer"]))
  if not here or not here.get("hca") or here.get("gid") is None:
    raise LocalError(f"No RoCE v2 address on this machine's link to {pair['peer']}: is it the ConnectX-7 cable, with an IPv4 address on each end?", short="Pair link has no RoCE address")
  there = pair_python(pair["peer"], f"fabric_side({here['addr']!r})")
  if not there or not there.get("hca") or there.get("gid") is None:
    raise LocalError(f"The peer {pair['peer']} has no RoCE v2 address on its link back to {here['addr']}", short="Pair link has no RoCE address")
  if len(here["hcas"]) != len(there["hcas"]):
    here["hcas"], there["hcas"] = here["hcas"][:1], there["hcas"][:1]
  return here, there


# ---------------------------------------------------------------- validation


def _safe_string(value, where, errors):
  if not isinstance(value, str) or not value:
    errors.append(f"{where}: must be a non-empty string")
    return False
  if CONTROL_RE.search(value):
    errors.append(f"{where}: contains control characters")
    return False
  return True


def _safe_relpath(value, where, errors, glob=False):
  if not _safe_string(value, where, errors):
    return False
  parts = value.split("/")
  if value.startswith("/") or any(part in ("", ".", "..") for part in parts):
    errors.append(f"{where}: must be a relative path without '.', '..' or empty parts")
    return False
  if not glob and any(ch in value for ch in "*?["):
    errors.append(f"{where}: must not contain glob characters")
    return False
  return True


def validate_runtime(rid, runtime, root=None):
  errors = []
  where = f"runtimes/{rid}"
  if not ID_RE.match(rid):
    errors.append(f"{where}: id must match {ID_RE.pattern}")
  image = runtime.get("image", "")
  if not isinstance(image, str) or not IMAGE_RE.match(image):
    errors.append(f"{where}: image must be <registry>/<path>@sha256:<64 hex>")
  tier = runtime.get("tier")
  if tier not in ("omarchy", "third-party"):
    errors.append(f"{where}: tier must be 'omarchy' or 'third-party'")
  if tier == "third-party":
    review = runtime.get("review", "")
    if _safe_relpath(review, f"{where}.review", errors) and root:
      path = os.path.join(root, review)
      if not os.path.isfile(path):
        errors.append(f"{where}.review: {review} is missing")
      else:
        try:
          with open(path, "rb") as handle:
            data = tomllib.load(handle)
          if data.get("image") != image:
            errors.append(f"{where}.review: reviews a different image digest")
          for field in ("publisher", "reviewer", "date", "license_url", "license_sha256", "checked"):
            if not data.get(field):
              errors.append(f"{where}.review: missing {field}")
        except tomllib.TOMLDecodeError as error:
          errors.append(f"{where}.review: {error}")
  port = runtime.get("port")
  if not isinstance(port, int) or not 1 <= port <= 65535:
    errors.append(f"{where}.port: must be 1-65535")
  health = runtime.get("health", {})
  probe = health.get("probe") if isinstance(health, dict) else None
  if not isinstance(probe, list) or not probe:
    errors.append(f"{where}.health.probe: must be a non-empty list")
  else:
    for index, part in enumerate(probe):
      _safe_string(part, f"{where}.health.probe[{index}]", errors)
  command = runtime.get("command", [])
  if not isinstance(command, list):
    errors.append(f"{where}.command: must be a list")
  else:
    for index, part in enumerate(command):
      _safe_string(part, f"{where}.command[{index}]", errors)
  run = runtime.get("run", {})
  if not isinstance(run, dict):
    errors.append(f"{where}.run: must be a table")
    run = {}
  for option, value in run.items():
    if option == "shm_size_gib":
      if not isinstance(value, (int, float)) or not 0 < value <= MAX_SHM_GIB:
        errors.append(f"{where}.run.shm_size_gib: must be between 0 and {MAX_SHM_GIB}")
      continue
    if option not in RUN_OPTIONS:
      errors.append(f"{where}.run.{option}: not an allowed run option ({', '.join(sorted(list(RUN_OPTIONS) + ['shm_size_gib']))})")
      continue
    if option == "devices":
      if not isinstance(value, list):
        errors.append(f"{where}.run.devices: must be a list")
        continue
      values = value
    else:
      if not isinstance(value, str):
        errors.append(f"{where}.run.{option}: must be a scalar string")
        continue
      values = [value]
    for item in values:
      if item not in RUN_OPTIONS[option]:
        errors.append(f"{where}.run.{option}: '{item}' is not allowed ({', '.join(RUN_OPTIONS[option])})")
  reserved = runtime.get("reserved", [])
  if not isinstance(reserved, list) or not all(isinstance(item, str) for item in reserved):
    errors.append(f"{where}.reserved: must be a list of strings")
    reserved = []
  inputs = runtime.get("inputs", {})
  if not isinstance(inputs, dict):
    errors.append(f"{where}.inputs: must be a table")
    inputs = {}
  targets = {}
  for name, mapping in inputs.items():
    if name not in STANDARD_INPUTS:
      errors.append(f"{where}.inputs.{name}: not a standard input ({', '.join(STANDARD_INPUTS)})")
      continue
    if not isinstance(mapping, dict) or len(mapping) != 1 or set(mapping) - {"arg", "env", "positional"}:
      errors.append(f"{where}.inputs.{name}: must be exactly one of {{ arg = ... }}, {{ env = ... }} or {{ positional = true }}")
      continue
    kind, target = next(iter(mapping.items()))
    # A server that takes its model as a bare argument (`serve <dir>`) gets it right after the command.
    if kind == "positional":
      if name != "model" or target is not True:
        errors.append(f"{where}.inputs.{name}: only the model can be positional, as {{ positional = true }}")
      continue
    if kind == "env" and not ENV_RE.match(str(target)):
      errors.append(f"{where}.inputs.{name}: bad environment variable name")
    if kind == "arg" and not ARG_RE.match(str(target)):
      errors.append(f"{where}.inputs.{name}: bad argument name")
    if (kind, target) in targets:
      errors.append(f"{where}.inputs.{name}: same target as {targets[(kind, target)]}")
    targets[(kind, target)] = name
    # The key is the one input that belongs in a reserved switch, and only in one
    # that names a key: never a bind address, port or download source.
    if target in reserved and not (name == "key" and "key" in target.lower()):
      errors.append(f"{where}.inputs.{name}: targets reserved {target}")
    if name == "key" and "key" not in str(target).lower():
      errors.append(f"{where}.inputs.key: {target} doesn't name a key")
    if kind == "env" and target == KEY_ENV:
      errors.append(f"{where}.inputs.{name}: {KEY_ENV} belongs to Omarchy")
  for required in ("model", "alias", "key"):
    if required not in inputs:
      errors.append(f"{where}.inputs: must map '{required}'")
  key = inputs.get("key", {})
  if isinstance(key, dict) and "arg" in key:
    if runtime.get("key_on_cmdline") is not True:
      errors.append(f"{where}.inputs.key: an argument key needs key_on_cmdline = true, shown as a warning")
    if not runtime.get("shell"):
      errors.append(f"{where}: key_on_cmdline needs shell = '/bin/bash' to substitute the key inside the container")
  if runtime.get("shell") and runtime.get("shell") != "/bin/bash":
    errors.append(f"{where}.shell: only /bin/bash is supported")
  errors += validate_pair_runtime(where, runtime, run)
  return errors


def validate_pair_runtime(where, runtime, run):
  """A runtime spanning two machines: its own network, the ranks' state folder and environment,
  and the flags the engine binds its API to loopback with. Both ranks run the runtime's command,
  which their servers compare."""
  errors = []
  pair = runtime.get("pair")
  fabric = run.get("network") == "fabric"
  if pair is None:
    if fabric:
      errors.append(f"{where}.run.network: 'fabric' is only for runtimes with a [pair] table")
    return errors
  if not fabric:
    errors.append(f"{where}.pair: a paired runtime needs run.network = 'fabric'")
  if not isinstance(pair, dict):
    return errors + [f"{where}.pair: must be a table"]
  if pair.get("nodes") != 2:
    errors.append(f"{where}.pair.nodes: only 2 is supported")
  port = pair.get("master_port")
  if not isinstance(port, int) or not 1024 <= port <= 65535:
    errors.append(f"{where}.pair.master_port: must be 1024-65535")
  state = pair.get("state")
  if not (isinstance(state, str) and re.match(r"^/[a-z][a-z0-9_-]*$", state) and state not in ("/models", "/tmp")):
    errors.append(f"{where}.pair.state: must be one top-level path such as /state, not /models or /tmp")
  env = pair.get("env", {})
  if not isinstance(env, dict):
    errors.append(f"{where}.pair.env: must be a table")
    env = {}
  for key, value in env.items():
    if not ENV_RE.match(key) or key == KEY_ENV or key in PAIR_NCCL:
      errors.append(f"{where}.pair.env.{key}: not an allowed variable name ({', '.join(PAIR_NCCL)} are the engine's)")
    _safe_string(value, f"{where}.pair.env.{key}", errors)
  listen = runtime.get("listen")
  if not (isinstance(listen, dict) and set(listen) == {"host", "port"} and all(isinstance(v, str) and ARG_RE.match(v) for v in listen.values())):
    errors.append(f"{where}.listen: a host-network runtime names its bind flags, as {{ host = '--host', port = '--port' }}")
  elif any(flag in runtime.get("command", []) for flag in listen.values()):
    errors.append(f"{where}.command: the engine sets {', '.join(listen.values())} itself")
  elif not all(flag in runtime.get("reserved", []) for flag in listen.values()):
    errors.append(f"{where}.reserved: must reserve {', '.join(listen.values())}")
  return errors


ARCHITECTURES = ("dense", "moe")
INPUTS = ("text", "image")


def validate_recipe(rid, recipe, runtimes, root=None):
  errors = []
  where = f"recipes/{rid}"
  if not ID_RE.match(rid):
    errors.append(f"{where}: id must match {ID_RE.pattern}")
  for field in ("name", "summary"):
    _safe_string(recipe.get(field), f"{where}.{field}", errors)
  # The model maker, naming the logo the agents panel shows (assets/<vendor>.svg).
  if "vendor" in recipe and not (isinstance(recipe["vendor"], str) and ID_RE.match(recipe["vendor"])):
    errors.append(f"{where}.vendor: must match {ID_RE.pattern}")
  # Dense or mixture of experts, shown on the recipe's card.
  if "architecture" in recipe and recipe["architecture"] not in ARCHITECTURES:
    errors.append(f"{where}.architecture: must be one of {', '.join(ARCHITECTURES)}")
  # What the model takes, which the agent overlays declare: an agent sending a
  # screenshot to a model it thinks is text-only fails every later request.
  if "input" in recipe and not (isinstance(recipe["input"], list) and "text" in recipe["input"] and all(item in INPUTS for item in recipe["input"]) and len(set(recipe["input"])) == len(recipe["input"])):
    errors.append(f"{where}.input: must list text and optionally image")
  runtime_id = recipe.get("runtime")
  runtime = runtimes.get(runtime_id)
  if runtime is None:
    errors.append(f"{where}.runtime: unknown runtime '{runtime_id}'")
    runtime = {}
  inputs = runtime.get("inputs", {})
  hardware = recipe.get("hardware", [])
  if not isinstance(hardware, list) or not hardware:
    errors.append(f"{where}.hardware: needs at least one [[hardware]] entry")
    hardware = []
  for index, entry in enumerate(hardware):
    if not isinstance(entry, dict) or not isinstance(entry.get("gpu"), str) or class_errors(entry, ""):
      errors.append(f"{where}.hardware[{index}]: needs gpu and exactly one of ram_class (one of {RAM_CLASSES}) and vram_class (one of {VRAM_CLASSES})")
      continue
    measured = entry.get("measured")
    if _safe_relpath(measured, f"{where}.hardware[{index}].measured", errors) and root and not os.path.isfile(os.path.join(root, measured)):
      errors.append(f"{where}.hardware[{index}].measured: {measured} is missing")
  artifacts = recipe.get("artifacts", [])
  if not isinstance(artifacts, list) or not artifacts:
    errors.append(f"{where}.artifacts: needs at least one [[artifacts]] entry")
    artifacts = []
  roles = {}
  for index, artifact in enumerate(artifacts):
    at = f"{where}.artifacts[{index}]"
    if not isinstance(artifact, dict):
      errors.append(f"{at}: must be a table")
      continue
    if not isinstance(artifact.get("repo"), str) or not re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", artifact.get("repo", "")):
      errors.append(f"{at}.repo: must be <owner>/<name>")
    if not SHA_RE.match(str(artifact.get("revision", ""))):
      errors.append(f"{at}.revision: must be a 40-hex commit")
    _safe_string(artifact.get("license"), f"{at}.license", errors)
    files = artifact.get("files", [])
    if not isinstance(files, list) or not files:
      errors.append(f"{at}.files: must be a non-empty list")
      files = []
    for findex, pattern in enumerate(files):
      _safe_relpath(pattern, f"{at}.files[{findex}]", errors, glob=True)
    if "dest" in artifact:
      _safe_relpath(artifact["dest"], f"{at}.dest", errors)
    for role in ("model", "draft"):
      if role in artifact:
        # "." names the directory the artifact lands in, for servers that load a checkpoint
        # directory rather than one file; a dest keeps it apart from the other artifacts.
        if artifact[role] == ".":
          named = "dest" in artifact
          if not named:
            errors.append(f"{at}.{role}: '.' names this artifact's directory, so it needs a dest")
        else:
          named = _safe_relpath(artifact[role], f"{at}.{role}", errors)
          if named and not any(fnmatch.fnmatchcase(artifact[role], pattern) for pattern in files):
            errors.append(f"{at}.{role}: not matched by this artifact's files")
        if named:
          if role in roles:
            errors.append(f"{at}.{role}: already named by artifacts[{roles[role]}]")
          roles[role] = index
  if "model" not in roles:
    errors.append(f"{where}.artifacts: one artifact must name the model file")
  if "draft" in roles and runtime and "draft" not in inputs:
    errors.append(f"{where}.artifacts: names a draft but runtime '{runtime_id}' has no draft input")
  variants = recipe.get("variant", [])
  if not isinstance(variants, list) or not variants:
    errors.append(f"{where}.variant: needs at least one [[variant]]")
    variants = []
  for index, variant in enumerate(variants):
    at = f"{where}.variant[{index}]"
    if not isinstance(variant, dict) or not isinstance(variant.get("ctx"), int) or variant["ctx"] <= 0:
      errors.append(f"{at}.ctx: must be a positive integer")
      continue
    for field in ("peak_ram_gib", "peak_gpu_gib"):
      if not isinstance(variant.get(field), (int, float)) or variant[field] <= 0:
        errors.append(f"{at}.{field}: must be a positive number")
    if variant.get("ctx") and runtime and "ctx" not in inputs:
      errors.append(f"{at}: runtime '{runtime_id}' has no ctx input")
  sampling = recipe.get("sampling", {})
  if not isinstance(sampling, dict):
    errors.append(f"{where}.sampling: must be a table")
    sampling = {}
  for name, value in sampling.items():
    if name not in ("temperature", "top_p", "top_k") or not isinstance(value, (int, float)):
      errors.append(f"{where}.sampling.{name}: unknown or not a number")
  args = recipe.get("server", {}).get("args", []) if isinstance(recipe.get("server", {}), dict) else None
  if not isinstance(args, list):
    errors.append(f"{where}.server.args: must be a list")
    args = []
  reserved = set(runtime.get("reserved", []))
  mapped = {mapping.get("arg") for mapping in inputs.values() if isinstance(mapping, dict)}

  def check_args(args, at):
    flags = set()
    for index, arg in enumerate(args):
      if not _safe_string(arg, f"{at}[{index}]", errors):
        continue
      flag = arg.split("=", 1)[0]
      if flag.startswith("-"):
        flags.add(flag)
      if flag in reserved:
        errors.append(f"{at}[{index}]: {flag} is reserved by runtime '{runtime_id}'")
      if flag in mapped:
        errors.append(f"{at}[{index}]: {flag} is set by Omarchy from the recipe's inputs")
      if KEY_PLACEHOLDER in arg:
        errors.append(f"{at}[{index}]: may not reference the key")
    return flags

  shared = check_args(args, f"{where}.server.args")
  # A variant's own args follow the shared ones, for settings that depend on the memory it is
  # sized for (streams, caches). They add flags; overriding a shared one would leave both.
  for index, variant in enumerate(variants):
    if not isinstance(variant, dict) or "args" not in variant:
      continue
    at = f"{where}.variant[{index}].args"
    if not isinstance(variant["args"], list):
      errors.append(f"{at}: must be a list")
      continue
    for flag in sorted(check_args(variant["args"], at) & shared):
      errors.append(f"{at}: {flag} is already in server.args")
  return errors


def validate_best(best, recipes):
  errors = []
  classes = best.get("class", [])
  if not isinstance(classes, list) or not classes:
    return ["best.toml: needs at least one [[class]]"]
  seen = {}
  for index, entry in enumerate(classes):
    where = f"best.toml class[{index}]"
    if not isinstance(entry, dict):
      errors.append(f"{where}: must be a table")
      continue
    errors.extend(class_errors(entry, where))
    if not _safe_string(entry.get("gpu"), f"{where}.gpu", errors):
      continue
    if "platform" in entry:
      errors.append(f"{where}.platform: not a class selector; use gpu and ram_class or vram_class")
    selectors = (gpu_class(entry.get("gpu")), entry_class(entry))
    if selectors in seen:
      errors.append(f"{where}: same selectors as class[{seen[selectors]}]")
    seen[selectors] = index
    names = entry.get("recipes", [])
    if not isinstance(names, list) or not names:
      errors.append(f"{where}.recipes: must be a non-empty list")
      continue
    for name in names:
      recipe = recipes.get(name)
      if recipe is None:
        errors.append(f"{where}: unknown recipe '{name}'")
        continue
      qualified = any(
        entry_class(hw) == entry_class(entry)
        and gpu_class(hw.get("gpu")) == gpu_class(entry.get("gpu"))
        for hw in recipe.get("hardware", []) if isinstance(hw, dict)
      )
      if not qualified:
        errors.append(f"{where}: recipe '{name}' has no measured [[hardware]] entry for this class")
  return errors


def _load_toml_dir(root, sub):
  items = {}
  directory = os.path.join(root, sub)
  if not os.path.isdir(directory):
    return items
  for name in sorted(os.listdir(directory)):
    if name.endswith(".toml"):
      with open(os.path.join(directory, name), "rb") as handle:
        items[name[:-5]] = tomllib.load(handle)
  return items


class Catalog:
  def __init__(self, root):
    self.root = root
    with open(os.path.join(root, "catalog.toml"), "rb") as handle:
      self.meta = tomllib.load(handle)
    self.runtimes = _load_toml_dir(root, "runtimes")
    self.recipes = _load_toml_dir(root, "recipes")
    with open(os.path.join(root, "best.toml"), "rb") as handle:
      self.best = tomllib.load(handle)

  def errors(self):
    errors = []
    if self.meta.get("schema") != SCHEMA:
      errors.append(f"catalog.toml: schema {self.meta.get('schema')} is not {SCHEMA}")
    for rid, runtime in self.runtimes.items():
      errors += validate_runtime(rid, runtime, self.root)
    for rid, recipe in self.recipes.items():
      errors += validate_recipe(rid, recipe, self.runtimes, self.root)
    errors += validate_best(self.best, self.recipes)
    return errors


def load_catalog_dir(root):
  try:
    return Catalog(root)
  except (OSError, tomllib.TOMLDecodeError) as error:
    raise LocalError(f"Catalog at {root} can't be read: {error}")


# ---------------------------------------------------------------- installed catalog


def load_catalog():
  root = os.environ.get("OMARCHY_LOCAL_AI_CATALOG", os.path.dirname(os.path.realpath(__file__)))
  catalog = load_catalog_dir(root)
  errors = catalog.errors()
  if errors:
    raise LocalError("Catalog failed validation:\n  " + "\n  ".join(errors))
  return catalog


# ---------------------------------------------------------------- compatibility and best


def runtime_supports(runtime, gpu):
  if bool(runtime.get("pair")) != (gpu.get("nodes", 1) == 2):
    return False
  devices = runtime.get("run", {}).get("devices", [])
  if gpu["vendor"] == "amd":
    return "cdi-nvidia" not in devices
  if gpu["vendor"] == "nvidia":
    return "kfd" not in devices and "render" not in devices
  return False


def qualified(recipe, profile, gpu):
  return any(
    gpu_class(hw.get("gpu")) == gpu_class(gpu["family"]) and entry_class(hw) == memory_class(profile, gpu) and hw.get("platform") in (None, profile["platform"])
    for hw in recipe.get("hardware", [])
  )


def fits(variant, profile, gpu):
  """Static fit of a variant on this machine: (ok, reason)."""
  if variant["peak_gpu_gib"] > gpu["memory_gib"]:
    return False, f"needs {variant['peak_gpu_gib']:g} GiB GPU-addressable memory, this GPU has {gpu['memory_gib']:.0f}"
  if variant["peak_ram_gib"] + RESERVE_GIB > profile["ram_gib"]:
    return False, f"needs {variant['peak_ram_gib']:g} GiB RAM plus {RESERVE_GIB} GiB for the desktop, this machine has {profile['ram_gib']:.0f}"
  return True, ""


def pick_variant(recipe, profile, gpu, ctx=None):
  # The largest context first, and at the same context the larger setup (more streams, a bigger
  # prompt cache) where it fits, so a smaller machine falls back to the variant sized for it.
  variants = sorted(recipe.get("variant", []), key=lambda v: (v["ctx"], v["peak_ram_gib"]), reverse=True)
  if ctx:
    variants = [v for v in variants if v["ctx"] == ctx]
    if not variants:
      raise LocalError(f"No measured variant with ctx {ctx} (have: {', '.join(str(v['ctx']) for v in recipe.get('variant', []))})")
  reasons = []
  for variant in variants:
    ok, reason = fits(variant, profile, gpu)
    if ok:
      return variant, ""
    reasons.append(reason)
  return None, reasons[-1] if reasons else "no variants"


def compatibility(catalog, rid, profile, gpu, ctx=None):
  recipe = catalog.recipes[rid]
  runtime = catalog.runtimes.get(recipe.get("runtime"), {})
  if not runtime_supports(runtime, gpu):
    return None, f"runtime {recipe.get('runtime')} can't run on {gpu['family']}"
  if not qualified(recipe, profile, gpu):
    classes = ", ".join(f"{hw.get('gpu')} {entry_class_text(hw)}" for hw in recipe.get("hardware", []))
    return None, f"not measured on {gpu['family']} {class_text(profile, gpu)} (measured: {classes})"
  return pick_variant(recipe, profile, gpu, ctx)


def match_class(best, profile, gpu):
  return next((entry for entry in best["class"]
               if gpu_class(entry["gpu"]) == gpu_class(gpu["family"]) and entry_class(entry) == memory_class(profile, gpu)), None)


def resolve_best(catalog, profile, gpu=None, ctx=None):
  """The first fitting recipe of the class of `gpu`, or, with none named, of this machine's
  GPUs from the strongest down: a pair's class first, then its own node's."""
  if gpu is not None:
    return _resolve_class(catalog, profile, gpu, ctx)
  gpus = sorted(profile["gpus"], key=lambda g: (g.get("nodes", 1), g["memory_gib"]), reverse=True)
  if not gpus:
    raise LocalError("No supported GPU found on this machine")
  failures = []
  for candidate in gpus:
    try:
      return _resolve_class(catalog, profile, candidate, ctx)
    except LocalError as error:
      failures.append((candidate, error))
  # The most telling failure: the first GPU that has a recommendation at all.
  classed = [error for candidate, error in failures if match_class(catalog.best, profile, candidate)]
  raise classed[0] if classed else failures[0][1]


def _resolve_class(catalog, profile, gpu, ctx=None):
  entry = match_class(catalog.best, profile, gpu)
  if entry is None:
    near = [rid for rid, recipe in catalog.recipes.items() if any(gpu_class(hw.get("gpu")) == gpu_class(gpu["family"]) for hw in recipe.get("hardware", []))]
    raise LocalError(
      f"No recommendation for this machine yet ({gpu['family']}, {class_text(profile, gpu)}, {profile['platform']}).\n"
      + (f"Recipes measured on the same GPU family: {', '.join(near)}" if near else "No recipes are measured on this GPU family yet.")
    )
  skipped = []
  for rid in entry["recipes"]:
    if rid not in catalog.recipes:
      continue
    try:
      variant, reason = compatibility(catalog, rid, profile, gpu, ctx)
    except LocalError as error:
      # A requested context this recipe has no variant for passes to the next recommendation.
      variant, reason = None, str(error)
    if variant:
      return rid, variant, skipped
    skipped.append((rid, reason))
  details = "; ".join(f"{rid}: {reason}" for rid, reason in skipped)
  raise LocalError(f"No recommended recipe fits this machine: {details}")


def recipe_listing(catalog, profile, running=()):
  rows = []
  try:
    gpu = choose_gpu(profile)
  except LocalError as error:
    gpu, gpu_error = None, str(error)
  best_id = None
  if gpu:
    try:
      best_id = resolve_best(catalog, profile)[0]
    except LocalError:
      best_id = None
  for rid in sorted(catalog.recipes):
    recipe = catalog.recipes[rid]
    runtime = catalog.runtimes.get(recipe.get("runtime"), {})
    if gpu:
      own = recipe_gpu(profile, recipe)
      variant, reason = compatibility(catalog, rid, profile, own)
    else:
      own, variant, reason = None, None, gpu_error
    review = load_review(catalog, runtime) if runtime.get("tier") == "third-party" else {}
    measured = load_measurement(catalog, recipe, profile, own) if variant else {}
    rows.append({
      "id": rid,
      "name": recipe.get("name", rid),
      "summary": recipe.get("summary", ""),
      "runtime": recipe.get("runtime", ""),
      "tier": runtime.get("tier", ""),
      "publisher": review.get("publisher", ""),
      "vendor": recipe.get("vendor", ""),
      "architecture": recipe.get("architecture", ""),
      "installed": recipe_installed(rid),
      "available": variant is not None,
      "reason": "" if variant else reason,
      "best": rid == best_id,
      "active": rid in running,
      # What the card shows: the measured footprint, context and speed at a
      # long prompt on this machine class.
      "memory_gib": variant["peak_ram_gib"] if variant else None,
      "context": variant["ctx"] if variant else None,
      "decode_tok_s": measured.get("long", {}).get("decode_tok_s"),
      "prefill_tok_s": measured.get("long", {}).get("prefill_tok_s"),
    })
  return rows


def recipe_installed(rid):
  """Installed means the recipe has a current manifest."""
  return os.path.isfile(model_manifest_path(rid))


def load_measurement(catalog, recipe, profile, gpu):
  """The bench numbers the catalog recorded for this recipe on this machine class."""
  for hw in recipe.get("hardware", []):
    if gpu_class(hw.get("gpu")) == gpu_class(gpu["family"]) and entry_class(hw) == memory_class(profile, gpu) and hw.get("platform") in (None, profile["platform"]):
      try:
        with open(os.path.join(catalog.root, hw["measured"]), "rb") as handle:
          return tomllib.load(handle)
      except (OSError, KeyError, tomllib.TOMLDecodeError):
        return {}
  return {}


def load_review(catalog, runtime):
  path = runtime.get("review")
  if not path:
    return {}
  try:
    with open(os.path.join(catalog.root, path), "rb") as handle:
      return tomllib.load(handle)
  except (OSError, tomllib.TOMLDecodeError):
    return {}


def write_list_cache(rows):
  write_atomic(os.path.join(state_dir(), "list.json"), json.dumps(rows, indent=2) + "\n")


# ---------------------------------------------------------------- Hugging Face artifacts


def hf_token():
  token = os.environ.get("HF_TOKEN", "")
  if not token:
    token = _read(os.path.join(home(), ".cache", "huggingface", "token"))
  return token


def http_json(url, timeout=60):
  headers = {"User-Agent": "omarchy-agent-local"}
  token = hf_token()
  if token and url.startswith("https://huggingface.co/"):
    headers["Authorization"] = f"Bearer {token}"
  request = urllib.request.Request(url, headers=headers)
  try:
    with urllib.request.urlopen(request, timeout=timeout) as response:
      return json.load(response)
  except (urllib.error.URLError, TimeoutError, ValueError) as error:
    raise LocalError(f"Can't read {url}: {error}")


def hf_tree(repo, revision, cache):
  key = f"{repo}@{revision}"
  if key not in cache:
    cache[key] = http_json(f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=1")
  return cache[key]


def build_artifacts(recipe, tree_cache=None):
  """Expand a recipe's artifacts against Hugging Face's file list at each pinned revision."""
  tree_cache = {} if tree_cache is None else tree_cache
  files = []
  destinations = {}
  roles = {}
  for artifact in recipe["artifacts"]:
    tree = [entry for entry in hf_tree(artifact["repo"], artifact["revision"], tree_cache) if entry.get("type") == "file"]
    for pattern in artifact["files"]:
      matched = [entry for entry in tree if fnmatch.fnmatchcase(entry["path"], pattern)]
      if not matched:
        raise LocalError(f"{artifact['repo']}@{artifact['revision'][:12]} has no file matching {pattern}")
      for entry in matched:
        path = entry["path"]
        if path.startswith("/") or ".." in path.split("/"):
          raise LocalError(f"{artifact['repo']} lists an unsafe path: {path}")
        dest = f"{artifact['dest']}/{path}" if artifact.get("dest") else path
        if dest in destinations:
          if destinations[dest]["repo"] == artifact["repo"] and destinations[dest]["path"] == path:
            continue
          raise LocalError(f"Two artifacts land at {dest}")
        lfs = entry.get("lfs") or {}
        record = {
          "repo": artifact["repo"],
          "revision": artifact["revision"],
          "path": path,
          "dest": dest,
          "size": int(lfs.get("size", entry.get("size", 0))),
          "license": artifact["license"],
        }
        if lfs.get("oid"):
          record["sha256"] = lfs["oid"]
        else:
          record["git_oid"] = entry["oid"]
        destinations[dest] = record
        files.append(record)
    for role in ("model", "draft"):
      if role in artifact:
        target = artifact[role]
        if target == ".":
          roles[role] = artifact["dest"]
        else:
          roles[role] = f"{artifact['dest']}/{target}" if artifact.get("dest") else target
  return files, roles


def blob_path(record):
  if "sha256" in record:
    return os.path.join(data_dir(), "blobs", "sha256", record["sha256"])
  return os.path.join(data_dir(), "blobs", "git", record["git_oid"])


def _hasher(record):
  if "sha256" in record:
    return hashlib.sha256(), record["sha256"]
  digest = hashlib.sha1()
  digest.update(f"blob {record['size']}\0".encode())
  return digest, record["git_oid"]


def verify_file(path, record):
  if os.path.getsize(path) != record["size"]:
    return False
  digest, expected = _hasher(record)
  with open(path, "rb") as handle:
    while chunk := handle.read(16 * 2**20):
      digest.update(chunk)
  return digest.hexdigest() == expected


class Progress:
  """Bytes a setup fetches: printed every few seconds and, with a recipe id,
  published to the agents panel through the setup's installing record."""

  def __init__(self, total=0, rid=None):
    self.total = total
    self.done = 0
    self.image = (0, 0)
    self.lock = threading.Lock()
    self.last = 0
    self.published = 0
    self.started = time.time()
    self.rid = rid
    # (time, bytes) samples over the last few seconds, for the speed the panel shows
    self.samples = [(self.started, 0)]

  def grow(self, amount):
    with self.lock:
      self.total += amount
    self.publish(force=True)

  def add(self, amount):
    with self.lock:
      self.done += amount
      now = time.time()
      if now - self.last >= 5 or self.done >= self.total:
        self.last = now
        rate = self.done / max(now - self.started, 0.001)
        say(f"  {gib(self.done):6.1f} / {gib(self.total):.1f} GiB  {rate / 2**20:6.0f} MiB/s")
    self.publish()

  def resumed(self, amount):
    """Bytes already on disk from an interrupted download: done, but not part
    of the speed, so every sample moves up with them."""
    with self.lock:
      self.done += amount
      self.samples = [(at, done + amount) for at, done in self.samples]
    self.publish()

  def pulled(self, got, total):
    """The runtime image's layers so far, which count toward the same bar."""
    with self.lock:
      self.image = (got, total)
    self.publish()

  def snapshot(self):
    with self.lock:
      done = self.done + self.image[0]
      total = self.total + self.image[1]
      now = time.time()
      self.samples.append((now, done))
      while len(self.samples) > 2 and now - self.samples[1][0] > PROGRESS_WINDOW_SECONDS:
        self.samples.pop(0)
      first_time, first_done = self.samples[0]
      rate = (done - first_done) / (now - first_time) if now - first_time > 0.2 else 0
      return done, total, rate

  def publish(self, force=False):
    if not self.rid:
      return
    now = time.time()
    if not force and now - self.published < PROGRESS_PUBLISH_SECONDS:
      return
    self.published = now
    done, total, rate = self.snapshot()
    update_installing(self.rid, phase="downloading", done_bytes=int(done), total_bytes=int(total), rate_bytes_s=int(rate))


def download(record, progress):
  url = f"https://huggingface.co/{record['repo']}/resolve/{record['revision']}/{urllib.request.quote(record['path'])}"
  staging = ensure_dir(os.path.join(data_dir(), "staging"))
  part = os.path.join(staging, os.path.basename(blob_path(record)) + ".part")
  if os.path.exists(part) and os.path.getsize(part) <= record["size"]:
    progress.resumed(os.path.getsize(part))
  for attempt in range(5):
    offset = os.path.getsize(part) if os.path.exists(part) else 0
    if offset > record["size"]:
      os.unlink(part)
      offset = 0
    if offset == record["size"]:
      break
    headers = {"User-Agent": "omarchy-agent-local"}
    if offset:
      headers["Range"] = f"bytes={offset}-"
    token = hf_token()
    if token:
      headers["Authorization"] = f"Bearer {token}"
    try:
      with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        if offset and response.status != 206:
          offset = 0
          mode = "wb"
        else:
          mode = "ab" if offset else "wb"
        with open(part, mode) as handle:
          while chunk := response.read(8 * 2**20):
            handle.write(chunk)
            progress.add(len(chunk))
      break
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
      if attempt == 4:
        raise LocalError(f"Download of {record['repo']}/{record['path']} failed: {error}")
      time.sleep(3 * (attempt + 1))
  if not verify_file(part, record):
    os.unlink(part)
    raise LocalError(f"{record['repo']}/{record['path']} doesn't match its published hash; discarded")
  target = blob_path(record)
  ensure_dir(os.path.dirname(target))
  os.chmod(part, 0o644)
  os.replace(part, target)


def adopt_seed(record, seed_dir):
  if not seed_dir:
    return False
  candidate = os.path.join(seed_dir, record["dest"])
  if not os.path.isfile(candidate) or os.path.getsize(candidate) != record["size"]:
    return False
  say(f"  verifying {record['dest']} from {seed_dir}")
  if not verify_file(candidate, record):
    say(f"  {candidate} doesn't match the published hash; downloading instead")
    return False
  target = blob_path(record)
  ensure_dir(os.path.dirname(target))
  try:
    os.link(candidate, target)
  except OSError:
    shutil.copyfile(candidate, target)
  os.chmod(target, 0o644)
  return True


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FILESET_RE = re.compile(r"^[0-9a-f]{16}$")


def validate_files(files, roles):
  """Refuse a file list that could write outside its model directory or name
  a blob by anything but its hash. A replayed manifest is user-editable."""
  errors = []
  if not isinstance(files, list) or not files:
    raise LocalError("Manifest has no files")
  dests = set()
  for index, record in enumerate(files):
    where = f"files[{index}]"
    if not isinstance(record, dict):
      raise LocalError(f"Manifest {where} is not a table")
    _safe_relpath(record.get("dest"), f"{where}.dest", errors)
    if not isinstance(record.get("size"), int) or record["size"] < 0:
      errors.append(f"{where}.size: must be a non-negative integer")
    if "sha256" in record:
      if not SHA256_RE.match(str(record["sha256"])):
        errors.append(f"{where}.sha256: must be 64 hex characters")
    elif not SHA_RE.match(str(record.get("git_oid", ""))):
      errors.append(f"{where}: needs sha256 or a 40-hex git_oid")
    dests.add(record.get("dest"))
  if len(dests) != len(files):
    errors.append("files: two records share a destination")
  for role, path in (roles or {}).items():
    if role not in ("model", "draft"):
      errors.append(f"roles.{role}: not a role")
    elif not _safe_relpath(path, f"roles.{role}", errors):
      continue
    elif path not in dests and not any(isinstance(dest, str) and dest.startswith(path + "/") for dest in dests):
      errors.append(f"roles.{role}: must name one of the files or a directory holding them")
  if errors:
    raise LocalError("Manifest is invalid:\n  " + "\n  ".join(errors))


def fileset_id(files):
  digest = hashlib.sha256()
  for record in sorted(files, key=lambda r: r["dest"]):
    digest.update(f"{record['dest']}\0{record.get('sha256') or record.get('git_oid')}\n".encode())
  return digest.hexdigest()[:16]


def fileset_dir(rid, fileset):
  """An immutable directory per exact file set."""
  return os.path.join(data_dir(), "models", rid, fileset)


def prune_filesets(rid, keep):
  """After a setup commits, drop this recipe's other file sets."""
  base = os.path.join(data_dir(), "models", rid)
  if not os.path.isdir(base) or not FILESET_RE.match(keep or ""):
    return
  for name in os.listdir(base):
    # A paired recipe's rank folders (<fileset>.rank<N>) go with their file set.
    if name != keep and not name.startswith(keep + "."):
      shutil.rmtree(os.path.join(base, name), ignore_errors=True)
  # The peer's copies too, once the setup that replaced them is the current one.
  setup = (load_model(rid) or {}).get("pair")
  if setup:
    peer_base = os.path.dirname(peer_fileset_dir(setup, rid, keep))
    q = shlex.quote
    try:
      pair_ssh(setup["peer"], f"cd {q(peer_base)} && for d in *; do case $d in {q(keep)}|{q(keep)}.*) ;; *) rm -rf -- \"$d\" ;; esac; done", check=False)
    except LocalError:
      pass


def materialize(rid, files, roles, seed_dir=None, progress=None):
  """Fetch a run's files and link them into their own file-set directory.
  Returns the file set id the unit mounts."""
  validate_files(files, roles)
  missing = [record for record in files if not os.path.isfile(blob_path(record))]
  missing = [record for record in missing if not adopt_seed(record, seed_dir)]
  if missing and load_installing(rid) is not None:
    update_installing(rid, partials=[os.path.basename(blob_path(record)) + ".part" for record in missing])
  need = sum(record["size"] for record in missing)
  if need:
    ensure_dir(data_dir())
    free = shutil.disk_usage(data_dir()).free
    if need + 2 * 2**30 > free:
      raise LocalError(f"Need {gib(need):.1f} GiB free for downloads, {gib(free):.1f} GiB available in {data_dir()}")
    say(f"Downloading {len(missing)} file(s), {gib(need):.1f} GiB, from Hugging Face")
    progress = progress or Progress()
    progress.grow(need)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
      for future in [pool.submit(download, record, progress) for record in missing]:
        future.result()
  fileset = fileset_id(files)
  target = fileset_dir(rid, fileset)
  if os.path.isdir(target):
    return fileset
  staging = target + ".partial"
  shutil.rmtree(staging, ignore_errors=True)
  ensure_dir(staging)
  # Paths are built from the resolved staging directory, so a model store
  # moved to another disk through a symlink still passes the containment check.
  root = os.path.realpath(staging)
  for record in files:
    dest = os.path.normpath(os.path.join(root, record["dest"]))
    if not dest.startswith(root + os.sep):
      raise LocalError(f"Refusing a file outside the model directory: {record['dest']}")
    ensure_dir(os.path.dirname(dest))
    if not (os.path.realpath(os.path.dirname(dest)) + os.sep).startswith(root + os.sep):
      raise LocalError(f"Refusing a file outside the model directory: {record['dest']}")
    os.link(blob_path(record), dest)
  os.replace(staging, target)
  return fileset


# ---------------------------------------------------------------- unit generation


def _systemd_escape(value):
  if CONTROL_RE.search(value):
    raise LocalError(f"Refusing a value with control characters: {value!r}")
  return value.replace("%", "%%").replace("$", "$$")


def _quote(value):
  escaped = _systemd_escape(value).replace("\\", "\\\\").replace('"', '\\"')
  return f'"{escaped}"'


def server_invocation(runtime, recipe, variant, roles, rid):
  """Returns (entrypoint, argv, env) with the key as KEY_PLACEHOLDER when it's an argument."""
  inputs = runtime["inputs"]
  values = {
    "model": f"/models/{roles['model']}",
    "draft": f"/models/{roles['draft']}" if roles.get("draft") else None,
    "ctx": str(variant["ctx"]),
    "alias": rid,
    "key": KEY_PLACEHOLDER,
  }
  for name, value in recipe.get("sampling", {}).items():
    values[name] = f"{value:g}" if isinstance(value, float) else str(value)
  argv = list(runtime.get("command", []))
  env = {}
  skipped = []
  for name in STANDARD_INPUTS:
    value = values.get(name)
    if value is None:
      continue
    mapping = inputs.get(name)
    if mapping is None:
      if name in ("temperature", "top_p", "top_k"):
        skipped.append(name)
        continue
      if name in ("draft", "ctx"):
        raise LocalError(f"Runtime has no mapping for {name}")
      continue
    if "arg" in mapping:
      argv += [mapping["arg"], value]
    elif "positional" in mapping:
      argv.insert(len(runtime.get("command", [])), value)
    elif name != "key":
      env[mapping["env"]] = value
  argv += recipe.get("server", {}).get("args", [])
  argv += variant.get("args", [])
  entrypoint = None
  if KEY_PLACEHOLDER in argv:
    script = f'a=(); for x in "$@"; do [[ $x == {KEY_PLACEHOLDER} ]] && x=${KEY_ENV}; a+=("$x"); done; exec "${{a[@]}}"'
    argv = ["-c", script, "omarchy-local"] + argv
    entrypoint = runtime["shell"]
  return entrypoint, argv, env, skipped


def device_lines(runtime, gpu):
  lines = []
  for device in runtime.get("run", {}).get("devices", []):
    if device == "render":
      lines.append(f"AddDevice={gpu['render_node']}")
    elif device == "kfd":
      lines.append("AddDevice=/dev/kfd")
    elif device == "cdi-nvidia":
      # A discrete card gets only the one whose memory was checked; a unified GPU is the only one.
      lines.append(f"AddDevice=nvidia.com/gpu={gpu['nvidia_index'] if gpu.get('nvidia_index') is not None else 'all'}")
    elif device == "infiniband":
      lines.append("AddDevice=/dev/infiniband")
  return lines


# Quadlet otherwise orders every unit after the system's network-online.target
# through podman-user-wait-network-online.service. These units have no outside
# network to wait for, and where that target never arrives every start would
# sit out the 90s timeout first.
QUADLET_SECTION = ["[Quadlet]", "DefaultDependencies=false"]


def render_network():
  return "\n".join(QUADLET_SECTION + ["", "[Network]", f"NetworkName={NETWORK_NAME}", "Internal=true"]) + "\n"


def render_container(rid, recipe, runtime, variant, roles, gpu, fileset):
  for value in (rid, recipe.get("name", "")):
    if CONTROL_RE.search(value):
      raise LocalError("Refusing a recipe name with control characters")
  if not ID_RE.match(rid) or not IMAGE_RE.match(runtime["image"]) or not isinstance(fileset, str) or not FILESET_RE.fullmatch(fileset):
    raise LocalError("Refusing an invalid recipe id, image reference or file set")
  entrypoint, argv, env, _ = server_invocation(runtime, recipe, variant, roles, rid)
  run = runtime.get("run", {})
  key_mapping = runtime["inputs"]["key"]
  pair = runtime.get("pair")
  port = port_for(rid)
  lines = [
    "# Generated by omarchy-agent-local use; changes are overwritten.",
    "[Unit]",
    f"Description=Omarchy local model {rid}",
    "PartOf=graphical-session.target",
    "After=graphical-session.target",
    "StartLimitIntervalSec=10m",
    "StartLimitBurst=3",
    "",
    *QUADLET_SECTION,
    "",
    "[Container]",
    f"ContainerName={unit_base(rid)}",
    f"Image={runtime['image']}",
    f"Label=org.omarchy.agent-local.recipe={rid}",
  ]
  if pair:
    # Rank 0 shares the host's network to reach rank 1 over the cable; its API listens on
    # loopback only, on the model's own port.
    lines += [
      "Network=host",
      f"Volume=%h/.local/share/omarchy/agent-local/models/{rid}/{fileset}:/models:ro",
      f"Volume=%h/.local/share/omarchy/agent-local/models/{rid}/{fileset}.rank0:{pair['state']}",
      f"EnvironmentFile=%h/.local/state/omarchy/agent-local/pair/{rid}.env",
      "Ulimit=stack=67108864:67108864",
      "Tmpfs=/tmp",
    ]
    entrypoint, argv, env, _ = head_argv(runtime, recipe, variant, roles, rid)
  else:
    lines += [
      f"Network={NETWORK_UNIT}",
      f"PublishPort=127.0.0.1:{port}:{runtime['port']}",
      f"Volume=%h/.local/share/omarchy/agent-local/models/{rid}/{fileset}:/models:ro",
    ]
  lines.append(f"Secret={SECRET},type=env,target={KEY_ENV}")
  if "env" in key_mapping:
    lines.append(f"Secret={SECRET},type=env,target={key_mapping['env']}")
  lines += device_lines(runtime, gpu)
  if run.get("group_add") == "keep-groups":
    lines.append("GroupAdd=keep-groups")
  if run.get("memlock") == "unlimited":
    lines.append("Ulimit=memlock=-1:-1")
  if run.get("shm_size_gib"):
    lines.append(f"ShmSize={int(run['shm_size_gib'] * 1024)}m")
  for name, value in sorted(env.items()):
    lines.append(f"Environment={_quote(f'{name}={value}')}")
  lines += ["ReadOnly=true", "DropCapability=all", "NoNewPrivileges=true"]
  if entrypoint:
    lines.append(f"Entrypoint={_systemd_escape(entrypoint)}")
  if argv:
    lines.append("Exec=" + " ".join(_quote(arg) for arg in argv))
  probe = [part.replace("@PORT@", str(port)) for part in runtime["health"]["probe"]]
  for part in probe:
    _systemd_escape(part)
  probe_json = _systemd_escape(json.dumps(probe))
  lines += [
    # While loading, probe every 2s so the unit goes active as soon as the model
    # is ready; once it passes, the regular check takes over every 30s.
    "HealthStartupCmd=" + probe_json,
    "HealthStartupInterval=2s",
    "HealthStartupRetries=300",
    "HealthStartupSuccess=1",
    "HealthCmd=" + probe_json,
    "HealthInterval=30s",
    "HealthTimeout=5s",
    "HealthRetries=3",
    "HealthStartPeriod=10m",
    "HealthOnFailure=kill",
    "Notify=healthy",
    "",
    "[Service]",
    f"TimeoutStartSec={20 if pair else 12}m",
    "Restart=on-failure",
    "RestartSec=10s",
  ]
  if pair:
    # Rank 1 starts on the peer first and stops with this unit, however it ends; the start also
    # writes this unit's environment file, the link as it is now.
    lines += [
      f"ExecStartPre=/usr/bin/env omarchy-agent-local link-worker start {rid}",
      f"ExecStopPost=-/usr/bin/env omarchy-agent-local link-worker stop {rid}",
    ]
  lines += [
    # The agents panel watches status.json; every start, stop and crash rewrites it.
    # systemd resolves a bare name in its own search path, not PATH; env uses the
    # unit's PATH, which holds Omarchy's bin.
    # Each hook says how the transition ended: systemd still reports it as
    # activating or deactivating while its Post commands run.
    f"ExecStartPost=-/usr/bin/env omarchy-agent-local status --write --started {rid}",
    f"ExecStopPost=-/usr/bin/env omarchy-agent-local status --write --stopped {rid}",
  ]
  return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- systemd and podman


def systemctl(*args, check=False, timeout=None):
  return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, check=check, timeout=timeout)


# ---------------------------------------------------------------- one unit per model
#
# Every recipe runs as its own Quadlet unit on its own loopback port, so several
# models can run side by side when memory allows. Names never collide with the
# shared network unit (omarchy-agent-local.network becomes
# omarchy-agent-local-network.service), whatever a recipe is called.


def unit_base(rid):
  return f"{UNIT_PREFIX}{rid}"


def unit_name(rid):
  return f"{unit_base(rid)}.service"


def quadlet_path(rid):
  return os.path.join(quadlet_dir(), f"{unit_base(rid)}.container")


def ports_path():
  return os.path.join(state_dir(), "ports.json")


def load_ports():
  try:
    with open(ports_path()) as handle:
      ports = json.load(handle)
    return {rid: int(port) for rid, port in ports.items() if ID_RE.match(rid)}
  except (OSError, ValueError, AttributeError):
    return {}


def port_for(rid):
  """A stable loopback port per recipe, so a model's endpoint never moves."""
  ports = load_ports()
  if rid not in ports:
    taken = set(ports.values())
    port = BASE_PORT
    while port in taken:
      port += 1
    ports[rid] = port
    write_atomic(ports_path(), json.dumps(ports, indent=2, sort_keys=True) + "\n")
  return ports[rid]


def endpoint(rid):
  return f"http://127.0.0.1:{port_for(rid)}/v1"


def unit_props(rid, *names):
  result = systemctl("show", unit_name(rid), "--timestamp=unix", *(f"--property={name}" for name in names))
  props = {}
  for line in result.stdout.splitlines():
    key, _, value = line.partition("=")
    props[key] = value
  return props


def unit_active(rid):
  return unit_props(rid, "ActiveState").get("ActiveState") in ("active", "activating", "reloading")


SERVER_ERROR_RE = re.compile(r"^(error|tensorfold|fatal):\s*(.+)", re.IGNORECASE)


def failure_reason(rid, result):
  """Why a model stopped unexpectedly, for the panel's line: a reason this start already recorded
  (the start hook's, say), else the server's own last error line from this start, not an
  earlier one's, else systemd's result."""
  started = _unix_ms(unit_props(rid, "InactiveExitTimestamp").get("InactiveExitTimestamp"))
  if started:
    path = error_path(rid)
    if os.path.exists(path) and os.path.getmtime(path) * 1000 >= started:
      return _read(path)
    for line in reversed(journal_tail(rid, 60, started).splitlines()):
      match = SERVER_ERROR_RE.match(line.strip())
      if match:
        return match.group(2).strip()[:160]
  return f"Stopped unexpectedly ({result})"


def journal_tail(rid, lines=15, since_ms=None):
  since = ["--since", f"@{since_ms // 1000}"] if since_ms else []
  result = subprocess.run(["journalctl", "--user", "-u", unit_name(rid), *since, "-n", str(lines), "-o", "cat", "--no-pager"], capture_output=True, text=True)
  return result.stdout.strip()


def ensure_key():
  path = key_path()
  if not os.path.isfile(path):
    ensure_dir(config_dir(), 0o700)
    import secrets
    write_atomic(path, secrets.token_urlsafe(32), mode=0o600)
  os.chmod(path, 0o600)
  with open(path) as handle:
    key = handle.read().strip()
  # The secret is the bare key: a trailing newline from the file would reach the
  # server through --api-key or its key variable and never match a Bearer header.
  result = subprocess.run(["podman", "secret", "create", "--replace", SECRET, "-"], input=key, capture_output=True, text=True)
  if result.returncode != 0:
    raise LocalError(f"Can't store the API key as a Podman secret: {result.stderr.strip()}")
  return key


# Podman's per-layer bars, as it draws them on a terminal: "Copying blob
# <id> [===>---] 1.2GiB / 3.4GiB | 80 MiB/s", then "Copying blob <id> done".
PULL_BAR_RE = re.compile(r"Copying blob (\w+) .*?([\d.]+)\s*([KMG]?i?B|b) / ([\d.]+)\s*([KMG]?i?B|b)")
PULL_DONE_RE = re.compile(r"Copying blob (\w+) (?:done|skipped)")
PULL_UNITS = {"b": 1, "B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "KB": 1e3, "MB": 1e6, "GB": 1e9}


def ensure_image(image, progress=None):
  if subprocess.run(["podman", "image", "exists", image]).returncode == 0:
    return
  say(f"Pulling the runtime image {image.split('@')[0]}")
  # Podman only draws byte progress on a terminal, one redrawn bar per layer,
  # which scrolls into a wall of lines in a narrow window. It draws to a
  # pseudo-terminal here instead, and its bars become the one progress line
  # downloads use. Run from a systemd unit's environment, Podman would also
  # cap the pull at five minutes, too short for a large image.
  env = {key: value for key, value in os.environ.items() if key not in ("NOTIFY_SOCKET", "INVOCATION_ID", "JOURNAL_STREAM", "SYSTEMD_EXEC_PID")}
  master, slave = pty.openpty()
  fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 400, 0, 0))
  process = subprocess.Popen(["podman", "pull", image], stdin=subprocess.DEVNULL, stdout=slave, stderr=slave, env=env)
  os.close(slave)
  layers = {}
  started = time.time()
  shown = started
  pending = ""
  lines = []
  while True:
    try:
      chunk = os.read(master, 65536)
    except OSError:
      break
    if not chunk:
      break
    pending += re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "\n", chunk.decode("utf-8", "replace"))
    *complete, pending = re.split(r"[\r\n]", pending)
    for line in (line.strip() for line in complete):
      if not line:
        continue
      bar = PULL_BAR_RE.search(line)
      done = PULL_DONE_RE.search(line)
      if bar:
        layers[bar.group(1)] = (float(bar.group(2)) * PULL_UNITS.get(bar.group(3), 1), float(bar.group(4)) * PULL_UNITS.get(bar.group(5), 1))
      elif done and done.group(1) in layers:
        total = layers[done.group(1)][1]
        layers[done.group(1)] = (total, total)
      elif "Copying blob" not in line:
        lines.append(line)
    now = time.time()
    if layers and progress:
      progress.pulled(sum(value[0] for value in layers.values()), sum(value[1] for value in layers.values()))
    if layers and now - shown >= 5:
      shown = now
      got = sum(value[0] for value in layers.values())
      total = sum(value[1] for value in layers.values())
      say(f"  {gib(got):6.1f} / {gib(total):.1f} GiB  {got / max(now - started, 0.001) / 2**20:6.0f} MiB/s")
  os.close(master)
  if process.wait() != 0:
    raise LocalError(f"Can't pull {image}" + ("\n" + "\n".join(lines[-5:]) if lines else ""))
  total = sum(value[1] for value in layers.values())
  say(f"  {gib(total):6.1f} / {gib(total):.1f} GiB  in {time.time() - started:.0f}s")


def preflight(runtime, gpu):
  user = os.environ.get("USER") or home().rsplit("/", 1)[-1]
  for path in ("/etc/subuid", "/etc/subgid"):
    if not any(line.split(":", 1)[0] == user for line in _read(path).splitlines()):
      raise LocalError(f"{path} has no range for {user}; rootless Podman needs one (sudo usermod --add-subuids 100000-165535 --add-subgids 100000-165535 {user})")
  for line in device_lines(runtime, gpu):
    device = line.split("=", 1)[1]
    if device == "/dev/infiniband":
      continue
    if device.startswith("/dev/") and not os.access(device, os.R_OK | os.W_OK):
      raise LocalError(f"{device} isn't accessible to {user}; add the user to its group (ls -l {device}) and log in again")
  if runtime.get("pair"):
    if not os.path.isdir("/dev/infiniband"):
      raise LocalError("/dev/infiniband is missing: the RDMA devices of the ConnectX-7 ports are needed (rdma-core)")
    if not shutil.which("rsync"):
      raise LocalError("rsync is needed to copy the model files to the paired machine: omarchy agent local link setup")


# Whether this user's services may lock unlimited memory: the manager's default for its units and
# its own limit, which caps theirs.
MEMLOCK_CHECK = r'''[[ "$(systemctl --user show -p DefaultLimitMEMLOCK --value)" == infinity ]] && grep -q '^Max locked memory *unlimited' "/proc/$(pgrep -u "$(id -u)" -x systemd | head -1)/limits"'''


def pair_limits(peer):
  """RDMA registers the ranks' buffers as locked memory. A rootless container can't raise its
  limit past its user's, and systemd's user manager takes that from PAM at login, so a limit
  `link setup` just lifted applies after the next login. Checked at admission: a setup is
  prepared and kept even while it can't start yet."""
  fix = "run: omarchy agent local link setup"
  if subprocess.run(["bash", "-c", MEMLOCK_CHECK], capture_output=True).returncode != 0:
    raise LocalError(f"Locked memory is limited for services here, which RDMA between the two machines needs unlimited; {fix}", short="Locked memory limited: run link setup")
  if pair_ssh(peer, MEMLOCK_CHECK, check=False).returncode != 0:
    raise LocalError(f"Locked memory is limited for services on the peer {peer}, which RDMA between the two machines needs unlimited; {fix}", short="Locked memory limited on the peer: run link setup")


# No answer we read is near this; a peer's listing that is gets cut off, fails to parse and
# leaves that peer unreachable instead of filling this machine's memory.
HTTP_MAX_BYTES = 1 << 20


def http_status(url, key=None, timeout=10):
  headers = {"User-Agent": "omarchy-agent-local"}
  if key:
    headers["Authorization"] = f"Bearer {key}"
  try:
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
      return response.status, response.read(HTTP_MAX_BYTES)
  except urllib.error.HTTPError as error:
    return error.code, b""
  except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
    return 0, str(error).encode()


def post_start_checks(rid, key):
  base = f"{endpoint(rid)}/models"
  status, _ = http_status(base)
  if status not in (401, 403):
    raise LocalError(f"The server answered {status or 'nothing'} without the key; refusing a model that doesn't enforce its key")
  status, body = http_status(base, key)
  if status != 200:
    raise LocalError(f"The server answered {status or 'nothing'} with the key")
  try:
    ids = [entry.get("id") for entry in json.loads(body).get("data", [])]
  except ValueError:
    ids = []
  if rid not in ids:
    raise LocalError(f"Port {port_for(rid)} is serving {', '.join(map(str, ids)) or 'an unknown model'}, not {rid}")


def free_memory(gpu):
  mem = meminfo()
  available = gib(mem.get("MemAvailable", 0))
  gpu_free = gpu["memory_gib"]
  if gpu.get("sysfs"):
    used = int(_read(os.path.join(gpu["sysfs"], "mem_info_gtt_used"), "0") or 0) + int(_read(os.path.join(gpu["sysfs"], "mem_info_vram_used"), "0") or 0)
    gpu_free = gpu["memory_gib"] - gib(used)
  elif gpu.get("nvidia_index") is not None:
    # A reading that fails counts as no free memory, so admission waits and then refuses
    # instead of approving a start on a card that may be full.
    free = nvidia_free_gib(gpu["nvidia_index"])
    gpu_free = free if free is not None else 0.0
  return available, gpu_free


def top_memory_users(count=3):
  result = subprocess.run(["ps", "-eo", "rss=,comm=", "--sort=-rss"], capture_output=True, text=True)
  users = []
  for line in result.stdout.splitlines()[:count]:
    rss, _, name = line.strip().partition(" ")
    if rss.isdigit():
      users.append(f"{name.strip()} ({int(rss) / 2**20:.1f} GiB)")
  return ", ".join(users)


def evict_idle_weights(keep=()):
  """Drop the page cache of weights no running model needs.

  A model that just stopped leaves its weights in the page cache. The kernel
  would reclaim them on demand, but only page by page, so a runtime pinning
  tens of GiB in large contiguous blocks (halogen's KV pool) stalls on memory
  that is free in total but fragmented. POSIX_FADV_DONTNEED on files we own
  frees those pages at once, unprivileged, and the freed runs coalesce: on a
  128 GB Strix Halo, reading a 29 GiB model left 165 free 2 MiB blocks and
  evicting it brought back 5,301. Weights of `keep` recipes and of models
  still running are left alone; their runtimes may be reading them."""
  keep = set(keep) | running_models()
  wanted = set()
  for rid in keep:
    manifest = load_model(rid)
    if manifest:
      wanted.update(blob_path(record) for record in manifest["files"])
  for directory, _, names in os.walk(os.path.join(data_dir(), "blobs")):
    for name in names:
      path = os.path.join(directory, name)
      if path in wanted:
        continue
      try:
        fd = os.open(path, os.O_RDONLY)
      except OSError:
        continue
      try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
      finally:
        os.close(fd)


def check_headroom(variant, gpu):
  if gpu.get("nodes") == 2:
    pair_limits(gpu["peer"])
    peer_headroom(gpu["peer"], variant)
  for _ in range(15):
    available, gpu_free = free_memory(gpu)
    if available >= variant["peak_ram_gib"] + RESERVE_GIB and gpu_free >= variant["peak_gpu_gib"]:
      return available, gpu_free
    time.sleep(1)
  running = [rid for rid in known_models() if unit_active(rid)]
  names = [(load_model(rid) or {}).get("recipe", {}).get("name", rid) for rid in running]
  need = variant["peak_ram_gib"] + RESERVE_GIB
  short = f"{NOT_ENOUGH_MEMORY} (needs {need:g} GiB)"
  raise LocalError(
    f"Not enough free memory right now: the model needs {variant['peak_ram_gib']:g} GiB RAM plus {RESERVE_GIB} GiB for the desktop "
    f"and {variant['peak_gpu_gib']:g} GiB GPU-addressable; {available:.1f} GiB RAM and {gpu_free:.1f} GiB GPU are free.\n"
    + (f"Local models running: {', '.join(running)} (stop one with: omarchy agent local stop <recipe>)\n" if running else "")
    + f"Largest memory users: {top_memory_users()}",
    short=short,
  )


# ---------------------------------------------------------------- ranks on two machines
#
# A paired recipe's file set is materialized here as any other, then copied to the peer over the
# cable. Each rank keeps its server's caches in its own writable folder, <fileset>.rank<N>, at the
# runtime's state path. Rank 1 runs on the peer as a plain Podman container the head unit's hooks
# start and stop, with the same boundary as the head's: read-only root and model mounts, no
# capabilities, no new privileges, the GPU and RDMA devices only.

def head_argv(runtime, recipe, variant, roles, rid):
  """The server command both ranks run, bound to loopback on the model's port."""
  entrypoint, argv, env, skipped = server_invocation(runtime, recipe, variant, roles, rid)
  return entrypoint, argv + [runtime["listen"]["host"], "127.0.0.1", runtime["listen"]["port"], str(port_for(rid))], env, skipped


def worker_name(rid):
  return f"{UNIT_PREFIX}{rid}-rank1"


def rank_dir(rid, fileset, rank, home_dir=None):
  return os.path.join(home_dir or home(), ".local", "share", "omarchy", "agent-local", "models", rid, f"{fileset}.rank{rank}")


def peer_fileset_dir(pair, rid, fileset):
  return os.path.join(pair["home"], ".local", "share", "omarchy", "agent-local", "models", rid, fileset)


def pair_env_path(rid):
  return os.path.join(state_dir(), "pair", f"{rid}.env")


def pair_setup_path(rid):
  """The setup the paired unit runs: written with the unit, before its first start, since
  the current manifest is only saved once a setup has started."""
  return os.path.join(state_dir(), "pair", f"{rid}.json")


def write_pair_setup(manifest):
  if manifest and manifest.get("pair"):
    write_atomic(pair_setup_path(manifest["recipe_id"]), json.dumps(manifest, indent=2) + "\n")


def pair_manifest(rid):
  try:
    with open(pair_setup_path(rid)) as handle:
      return json.load(handle)
  except (OSError, ValueError):
    return resolve_model(rid)


def peer_headroom(peer, variant):
  node = pair_python(peer, "node()", timeout=10)
  available = gib(node["ram_available"])
  need = variant["peak_ram_gib"] + RESERVE_GIB
  if available < need:
    raise LocalError(
      f"Not enough free memory on the peer {node['host']}: rank 1 needs {variant['peak_ram_gib']:g} GiB plus {RESERVE_GIB} GiB, {available:.1f} GiB are free there.",
      short=f"{NOT_ENOUGH_MEMORY} on {node['host']} (needs {need:g} GiB)",
    )
  return available


def peer_free_gib(manifest):
  """Free memory on a paired setup's other machine; None when the setup isn't paired or the
  peer doesn't answer."""
  setup = manifest.get("pair")
  if not setup:
    return None
  try:
    return gib(pair_python(setup["peer"], "node()", timeout=10)["ram_available"])
  except (LocalError, ValueError, KeyError):
    return None


def ram_used_gib(manifest, before, after, peer_before, peer_after):
  """What a start took, on every machine the model runs on: a pair's figure is both ranks'.
  A peer that didn't answer counts as holding what this rank holds."""
  used = before - after
  if manifest.get("pair"):
    used += peer_before - peer_after if None not in (peer_before, peer_after) else used
  return round(used, 1)


def nodes_ram_gib(manifest, profile):
  """Total memory of the machines a model runs on, which its memory use is shown against."""
  total = profile.get("ram_gib") or gib(meminfo().get("MemTotal", 0))
  if manifest.get("pair"):
    # The peer the model was set up with, which is the one it starts on: a link made to another
    # machine since doesn't change it.
    linked, saved = profile.get("pair") or {}, (manifest.get("profile") or {}).get("pair") or {}
    peer = linked if linked.get("peer") == manifest["pair"].get("peer") else saved
    total += peer.get("ram_gib") or total
  return round(total, 1)


def pair_setup(rid, manifest, runtime, pair):
  """Everything rank 1 needs on the peer before the first start: the image, the file set and its
  state folder (rank 0's here too)."""
  fileset = manifest["fileset"]
  models = fileset_dir(rid, fileset)
  inspect = subprocess.run(["podman", "image", "inspect", "--format", "{{.Id}}", runtime["image"]], capture_output=True, text=True)
  ident = inspect.stdout.strip()
  if inspect.returncode != 0 or not re.match(r"^[0-9a-f]{64}$", ident):
    raise LocalError(f"Can't read the ID of {runtime['image']}: {inspect.stderr.strip()}")
  manifest["pair"] = {"peer": pair["peer"], "host": pair["host"], "home": pair["home"], "image_id": ident}
  # A first setup cancelled from here on still knows where its copies went.
  if load_installing(rid) is not None:
    update_installing(rid, pair=manifest["pair"])
  ensure_dir(rank_dir(rid, fileset, 0))
  peer = pair["peer"]
  q = shlex.quote
  if pair_ssh(peer, f"podman image exists {q(ident)}", check=False).returncode != 0:
    say(f"Copying the runtime image to {pair['host']} over the link")
    save = subprocess.Popen(["podman", "save", runtime["image"]], stdout=subprocess.PIPE)
    load = subprocess.run(pair_ssh_argv() + [peer, "podman load -q"], stdin=save.stdout, capture_output=True, text=True)
    save.stdout.close()
    if save.wait() != 0 or load.returncode != 0:
      raise LocalError(f"Copying the image to {pair['host']} failed: {load.stderr.strip()[-400:]}", short="Image copy to the peer failed")
    pair_ssh(peer, f"podman image exists {q(ident)}")
  target = peer_fileset_dir(pair, rid, fileset)
  say(f"Copying the model files to {pair['host']} over the link (only what it lacks)")
  pair_ssh(peer, f"mkdir -p {q(target)}")
  rsync = ["rsync", "-a", "--partial", "--delete", "--info=progress2", "--no-inc-recursive",
           "-e", shlex.join(pair_ssh_argv() + ["-o", "Compression=no", "-c", "aes128-gcm@openssh.com"]), f"{models}/", f"{peer}:{target}/"]
  if subprocess.run(rsync, stdin=subprocess.DEVNULL).returncode != 0:
    raise LocalError(f"Copying the model files to {pair['host']} failed", short="Copy to the peer failed")
  pair_ssh(peer, f"mkdir -p {q(rank_dir(rid, fileset, 1, pair['home']))}")


def _placeholders(text, values):
  for name, value in values.items():
    text = text.replace(name, value)
  return text


def pair_worker_start(rid):
  """From the head unit's ExecStartPre: the link as it is now into the head's environment file,
  then rank 1 on the peer."""
  manifest = pair_manifest(rid)
  runtime, setup, variant = manifest["runtime"], manifest.get("pair"), manifest["variant"]
  if not runtime.get("pair") or not setup:
    raise LocalError(f"{rid} isn't a paired setup")
  pair = {"peer": setup["peer"], "host": setup["host"], "home": setup["home"]}
  here, there = pair_link(pair)
  # Also here: systemd's restarts after a failure don't pass admission.
  pair_limits(pair["peer"])
  peer_headroom(pair["peer"], variant)
  spec = runtime["pair"]
  def environment(rank, side):
    values = {"@RANK@": str(rank), "@WORLD@": str(spec["nodes"]), "@MASTER_ADDR@": here["addr"],
              "@MASTER_PORT@": str(spec["master_port"]), "@RDMA_DEVICES@": ",".join(side["hcas"])}
    env = {name: _placeholders(value, values) for name, value in spec.get("env", {}).items()}
    env.update(NCCL_SOCKET_IFNAME=side["dev"], NCCL_IB_HCA=",".join(side["hcas"]), NCCL_IB_GID_INDEX=str(side["gid"]))
    return env
  write_atomic(pair_env_path(rid), "".join(f"{name}={value}\n" for name, value in sorted(environment(0, here).items())))
  fileset = manifest["fileset"]
  _, argv, worker, _ = head_argv(runtime, manifest["recipe"], variant, manifest["roles"], rid)
  worker.update(environment(1, there))
  run = runtime.get("run", {})
  command = ["podman", "run", "-d", "--replace", "--name", worker_name(rid), "--label", f"org.omarchy.agent-local.recipe={rid}",
             "--network", "host", "--read-only", "--tmpfs", "/tmp", "--cap-drop", "all", "--security-opt", "no-new-privileges", "--device", "nvidia.com/gpu=all", "--device", "/dev/infiniband",
             "--ulimit", "memlock=-1:-1", "--ulimit", "stack=67108864:67108864",
             "-v", f"{peer_fileset_dir(pair, rid, fileset)}:/models:ro", "-v", f"{rank_dir(rid, fileset, 1, pair['home'])}:{spec['state']}"]
  if run.get("shm_size_gib"):
    command += ["--shm-size", f"{int(run['shm_size_gib'] * 1024)}m"]
  for name, value in sorted(worker.items()):
    command += ["-e", f"{name}={value}"]
  command += [setup["image_id"], *argv]
  pair_ssh(pair["peer"], shlex.join(command), timeout=120)
  say(f"Rank 1 started on {pair['host']} ({there['addr']}, RoCE {','.join(there['hcas'])}); rank 0 rendezvous {here['addr']}:{spec['master_port']}")


def pair_worker_stop(rid):
  manifest = pair_manifest(rid) if os.path.exists(pair_setup_path(rid)) else (load_model(rid) or {})
  setup = manifest.get("pair")
  if not setup:
    return
  pair_ssh(setup["peer"], f"podman stop -t 30 {shlex.quote(worker_name(rid))}", timeout=90, check=False)


def pair_remove(manifest, local_only=False):
  """Removes a paired recipe's rank 1 container, files and (when unused) image from the peer.
  Fails while the peer can't be reached, before anything here is deleted, so remove can run
  again; local_only skips the peer, for one that's gone for good."""
  setup = manifest.get("pair")
  if not setup:
    return
  rid = manifest["recipe_id"]
  q = shlex.quote
  base = os.path.join(setup["home"], ".local", "share", "omarchy", "agent-local", "models", rid)
  others = {(load_model(other) or {}).get("pair", {}).get("image_id") for other in known_models() if other != rid}
  command = f"podman rm -f {q(worker_name(rid))} >/dev/null 2>&1; rm -rf -- {q(base)}"
  if setup["image_id"] not in others:
    command += f"; podman rmi {q(setup['image_id'])} >/dev/null 2>&1"
  try:
    pair_ssh(setup["peer"], command + "; true", timeout=600)
  except LocalError as error:
    if not local_only:
      raise LocalError(f"Can't reach {setup['host']} to remove {rid}'s copies there ({error.short}). Run remove again once it's back, "
                       f"or with --local-only if it's gone for good", short=f"{setup['host']} unreachable")
    say(f"Leaving {rid}'s copies on {setup['host']}, which can't be reached: {error.short}")
  for path in (pair_env_path(rid), pair_setup_path(rid)):
    if os.path.exists(path):
      os.unlink(path)


def cmd_link_worker(argv):
  parser = argparse.ArgumentParser(prog="omarchy agent local link-worker")
  parser.add_argument("action", choices=("start", "stop"))
  parser.add_argument("recipe")
  args = parser.parse_args(argv)
  if not ID_RE.match(args.recipe):
    raise LocalError("Not a recipe id")
  if args.action == "start":
    try:
      pair_worker_start(args.recipe)
    except LocalError as error:
      record_error(args.recipe, error.short)
      raise
  else:
    pair_worker_stop(args.recipe)
  return 0


# What `link setup` makes true on each machine, as one root shell script per machine: every step
# checks first, so it can run again at any time. RDMA needs unlimited locked memory (DGX OS ships
# it in 99-nv-spark-limits.conf, Arch doesn't); the worker runs rootless Podman with the NVIDIA CDI
# GPU and rsync; its user lingers, so rank 1 outlives the SSH session that started it.
HOST_SETUP = r'''
set -euo pipefail
user=$1
# Logins (PAM) and the user's systemd manager, whose units the models run in; a running manager
# gets it now: re-executed to read the default, then its own limit raised to match.
if ! grep -qs '^\*[[:space:]]*hard[[:space:]]*memlock[[:space:]]*unlimited' /etc/security/limits.d/*.conf; then
  mkdir -p /etc/security/limits.d
  printf '# RDMA between paired DGX Sparks registers large buffers as locked memory.\n*   soft    memlock   unlimited\n*   hard    memlock   unlimited\n' > /etc/security/limits.d/99-rdma-memlock.conf
  echo "  unlimited locked memory for logins"
fi
if [[ ! -f /etc/systemd/user.conf.d/99-rdma-memlock.conf ]]; then
  mkdir -p /etc/systemd/user.conf.d
  printf '# RDMA between paired DGX Sparks registers large buffers as locked memory.\n[Manager]\nDefaultLimitMEMLOCK=infinity\n' > /etc/systemd/user.conf.d/99-rdma-memlock.conf
  echo "  unlimited locked memory for user services"
fi
manager=$(pgrep -u "$user" -x systemd | head -1 || true)
if [[ -n "$manager" ]]; then
  # The default its services get, then its own limit, which caps theirs (a re-exec resets it).
  if [[ "$(systemctl --user -M "$user@" show -p DefaultLimitMEMLOCK --value)" != infinity ]]; then
    systemctl --user -M "$user@" daemon-reexec
    echo "  applied to $user's running services"
  fi
  if ! grep -q '^Max locked memory *unlimited' "/proc/$manager/limits"; then
    prlimit --pid "$manager" --memlock=unlimited:unlimited
  fi
fi
missing=()
for tool in podman rsync python3; do command -v "$tool" >/dev/null || missing+=("$tool"); done
if (( ${#missing[@]} )); then
  echo "  installing ${missing[*]}"
  if command -v pacman >/dev/null; then pacman -S --needed --noconfirm "${missing[@]}" >/dev/null
  elif command -v apt-get >/dev/null; then DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}" >/dev/null
  else echo "  no pacman or apt-get: install ${missing[*]} yourself" >&2; exit 1; fi
fi
# Podman before 5 can't read the CDI 0.7 spec nvidia-ctk 1.18+ writes (Ubuntu 24.04 ships 4.9): keep a 0.6 copy in
# /etc/cdi, rewritten after every nvidia-cdi-refresh.
major=$(podman version --format '{{.Client.Version}}' 2>/dev/null | cut -d. -f1)
if [[ -n "$major" ]] && (( major < 5 )) && grep -qs additionalGids /var/run/cdi/nvidia.yaml; then
  cat > /usr/local/sbin/nvidia-cdi-podman-compat <<'PYEOF'
@CDI_COMPAT@
PYEOF
  chmod 755 /usr/local/sbin/nvidia-cdi-podman-compat
  mkdir -p /etc/systemd/system/nvidia-cdi-refresh.service.d
  printf '[Service]\nExecStartPost=/usr/local/sbin/nvidia-cdi-podman-compat\n' > /etc/systemd/system/nvidia-cdi-refresh.service.d/podman-compat.conf
  systemctl daemon-reload
  /usr/local/sbin/nvidia-cdi-podman-compat
  echo "  NVIDIA CDI spec for Podman $major"
fi
if [[ "$(loginctl show-user "$user" -p Linger --value 2>/dev/null)" != yes ]]; then
  loginctl enable-linger "$user"
  echo "  lingering user $user"
fi
'''
CDI_COMPAT = r'''#!/usr/bin/python3 -I
# Written by omarchy-local-ai link setup. Podman before 5 predates CDI spec 0.7.0 and rejects the additionalGids
# field nvidia-ctk 1.18+ writes. This writes /etc/cdi/nvidia.yaml as the same spec without it, at 0.6.0, after every
# nvidia-cdi-refresh (a drop-in runs it), so rootless Podman keeps the GPU across driver updates.
import os, re
src = open("/var/run/cdi/nvidia.yaml").read()
out, skip, indent = [], False, 0
for line in src.splitlines():
    if skip:
        if line.lstrip().startswith("- ") and len(line) - len(line.lstrip()) >= indent:
            continue
        skip = False
    match = re.match(r"^(\s*)additionalGids:\s*$", line)
    if match:
        skip, indent = True, len(match.group(1))
        continue
    out.append(line)
text = re.sub(r"^cdiVersion: 0\.7\.0$", "cdiVersion: 0.6.0", "\n".join(out) + "\n", count=1, flags=re.M)
os.makedirs("/etc/cdi", exist_ok=True)
tmp = "/etc/cdi/.nvidia.yaml.tmp"
with open(tmp, "w") as handle:
    handle.write(text)
os.chmod(tmp, 0o644)
os.replace(tmp, "/etc/cdi/nvidia.yaml")'''

# Addresses for cable ports, as dev=address/prefix (plan_link chooses them), through NetworkManager.
LINK_SETUP = r'''
set -euo pipefail
command -v nmcli >/dev/null || { echo "  NetworkManager (nmcli) is needed to address the cable" >&2; exit 1; }
for spec in "$@"; do
  dev=${spec%%=*}; address=${spec#*=}
  name="omarchy-pair-$dev"
  for old in $(nmcli -t -f NAME,DEVICE connection show --active | awk -F: -v d="$dev" -v n="$name" '$2 == d && $1 != n {print $1}'); do
    nmcli connection modify "$old" connection.autoconnect no
  done
  if nmcli -t -f NAME connection show | grep -qx "$name"; then
    nmcli connection modify "$name" ipv4.method manual ipv4.addresses "$address"
  else
    nmcli connection add type ethernet ifname "$dev" con-name "$name" ipv4.method manual ipv4.addresses "$address" \
      ipv6.method link-local 802-3-ethernet.mtu 9000 connection.autoconnect yes >/dev/null
  fi
  nmcli connection up "$name" >/dev/null
  echo "  $dev: $address, MTU 9000"
done
'''


# The ranks' rendezvous, NCCL's sockets and the engine's SSH cross the cable. Where ufw guards a
# machine (Omarchy turns it on), what the peer's own address sends to a cable port is let in, ahead
# of the SSH rate limit, as dev=address pairs (link_rules); nothing else on that port. A broader
# rule an earlier setup inserted for the port goes.
FIREWALL_SETUP = r'''
set -euo pipefail
command -v ufw >/dev/null || exit 0
ufw status | grep -q '^Status: active' || exit 0
for spec in "$@"; do
  dev=${spec%%=*}; from=${spec#*=}
  if ufw status | grep -qE "^Anywhere on $dev +ALLOW +Anywhere +# omarchy-local-ai"; then
    ufw delete allow in on "$dev" >/dev/null || true
  fi
  if ! ufw status | grep -qE "^Anywhere on $dev +ALLOW +${from//./\\.}( |$)"; then
    ufw insert 1 allow in on "$dev" from "$from" comment "omarchy-local-ai pair link" >/dev/null
    echo "  firewall: allowing $from on $dev"
  fi
done
'''


def local_ports():
  scope = {}
  exec(FABRIC_SIDE, scope)
  return scope["ports"]()


def plan_link(here, there):
  """Addresses for the cable ports both machines share by name (dev=address/prefix, for each
  end). A port addressed on one end only gets the other host of that /24 on the other end, so
  an interrupted setup finishes; a port addressed on neither gets a fresh 10.99.N.0/24, .1 here
  and .2 there. Ports on only one machine aren't the cable and are left alone."""
  taken = {ipaddress.ip_interface(c).network for side in (here, there) for cidrs in side.values() for c in cidrs}
  mine, theirs, subnet = [], [], 0
  def other_end(cidr):
    iface = ipaddress.ip_interface(cidr)
    host = next(h for h in itertools.islice(iface.network.hosts(), 2) if h != iface.ip)
    return f"{host}/{iface.network.prefixlen}"
  for dev in sorted(set(here) & set(there)):
    a = [ipaddress.ip_interface(c) for c in here[dev]]
    b = [ipaddress.ip_interface(c) for c in there[dev]]
    if any(x.network == y.network for x in a for y in b):
      continue
    if a and not b:
      theirs.append(f"{dev}={other_end(here[dev][0])}")
    elif b and not a:
      mine.append(f"{dev}={other_end(there[dev][0])}")
    elif not a and not b:
      while ipaddress.ip_network(f"10.99.{subnet}.0/24") in taken:
        subnet += 1
      taken.add(ipaddress.ip_network(f"10.99.{subnet}.0/24"))
      mine.append(f"{dev}=10.99.{subnet}.1/24")
      theirs.append(f"{dev}=10.99.{subnet}.2/24")
  return mine, theirs


def link_rules(here, there, rails):
  """Firewall rules for the cable, as dev=address for each end: each rail port the ranks use (on
  both ends) lets in the other end's address on the same network, and only that. Other RDMA
  ports, on a shared network say, are left to the firewall as they were."""
  mine, theirs = [], []
  for dev in sorted(set(here) & set(there) & set(rails)):
    for x in here[dev]:
      for y in there[dev]:
        x_iface, y_iface = ipaddress.ip_interface(x), ipaddress.ip_interface(y)
        if x_iface.network == y_iface.network:
          mine.append(f"{dev}={y_iface.ip}")
          theirs.append(f"{dev}={x_iface.ip}")
  return mine, theirs


def root_script(script, *args, peer=None):
  """Runs a setup script as root here or on the peer; sudo asks for a password on this terminal when it needs one."""
  argv = ["sudo", "bash", "-c", script, "omarchy-local-pair", *args]
  if peer:
    argv = pair_ssh_argv(tty=True) + [peer, shlex.join(argv)]
  if subprocess.run(argv).returncode != 0:
    raise LocalError(f"Setting up {'the peer ' + peer if peer else 'this machine'} failed (see above)")


def cmd_link_setup(argv):
  parser = argparse.ArgumentParser(prog="omarchy agent local link setup",
                                   description="Prepare this Spark and the one cabled to it to run models across both. Safe to run again.")
  parser.add_argument("peer", nargs="?", help="the other Spark as SSH reaches it now, e.g. user@spark-1234.local (default: the linked one)")
  args = parser.parse_args(argv)
  profile = machine_profile()
  if NVIDIA_UNIFIED_FAMILIES.get(profile["platform"]) not in PAIR_FAMILIES:
    raise LocalError("Pairs are supported between DGX Sparks (GB10) only")
  available = link_probe(profile) if not args.peer and not cached_pair(profile) else {}
  peer = args.peer or (cached_pair(profile) or {}).get("peer") or available.get("address")
  if not peer or not PEER_RE.match(peer):
    raise LocalError("Name the other Spark as SSH reaches it, e.g.: omarchy agent local link setup spark-1234.local")
  if not local_ports():
    raise LocalError("No ConnectX-7 port has a link: connect the two Sparks' QSFP ports with a direct cable")

  say("SSH key to the peer")
  key = pair_key()
  if not os.path.exists(key):
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", f"{os.environ.get('USER', 'user')}@{os.uname().nodename}", "-f", key], check=True)
  if subprocess.run(pair_ssh_argv() + [peer, "true"], capture_output=True).returncode != 0:
    say(f"  copying {key}.pub to {peer} (asks for its password once)")
    if subprocess.run(["ssh-copy-id", "-i", key + ".pub", "-o", "StrictHostKeyChecking=accept-new", "-o", "PubkeyAuthentication=no", peer]).returncode != 0:
      raise LocalError(f"Can't log in to {peer} with a key: check the name and password")
  node = pair_python(peer, "node()")
  if "GB10" not in node.get("gpu", ""):
    raise LocalError(f"{peer} doesn't report a GB10 GPU ({node.get('gpu') or 'nvidia-smi found none'})")

  say("This machine")
  root_script(HOST_SETUP.replace("@CDI_COMPAT@", CDI_COMPAT), os.environ.get("USER") or os.path.basename(home()))
  say(f"The peer, {node['host']}")
  root_script(HOST_SETUP.replace("@CDI_COMPAT@", CDI_COMPAT), pair_ssh(peer, "id -un").stdout.strip(), peer=peer)

  say("Addresses on the cable")
  here, there = local_ports(), pair_python(peer, "ports()")
  if not set(here) & set(there):
    raise LocalError("The two Sparks share no linked ConnectX-7 port name: are they cabled port to port?")
  mine, theirs = plan_link(here, there)
  if mine:
    root_script(LINK_SETUP, *mine)
  if theirs:
    root_script(LINK_SETUP, *theirs, peer=peer)
  if not mine and not theirs:
    say(f"  {', '.join(sorted(set(here) & set(there)))} already addressed")

  pair = record_pair(profile, cable_peer(peer))

  say("Firewalls")
  near, far = pair_link(pair)
  mine, theirs = link_rules(local_ports(), pair_python(pair["peer"], "ports()"), set(near["devs"]) & set(far["devs"]))
  root_script(FIREWALL_SETUP, *mine)
  root_script(FIREWALL_SETUP, *theirs, peer=pair["peer"])
  say(f"Linked to {pair['host']} at {pair['peer']}")
  pair_limits(pair["peer"])
  return 0


def cmd_link(argv):
  if argv[:1] == ["setup"]:
    return cmd_link_setup(argv[1:])
  if argv[:1] == ["notify"]:
    link_notify(machine_profile())
    return 0
  parser = argparse.ArgumentParser(prog="omarchy agent local link")
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args(argv)
  profile = machine_profile()
  pair = profile.get("pair")
  state = link_probe(profile)
  if args.json:
    print(json.dumps(pair or state, indent=2))
    return 0
  if state["state"] == "available":
    say(f"Another Spark is on the cable: {state['host'] or state['address']}. Set both up with: omarchy agent local link setup")
    return 0
  if not pair:
    say("No Spark is linked to this one yet: cable them, then run omarchy agent local link setup <the other Spark>")
    return 0
  here, there = pair_link(pair)
  say(f"Linked to {pair['host']} at {pair['peer']} ({pair['ram_gib']:.0f} GiB)")
  say(f"  here  {here['addr']} on {here['dev']}, RoCE {','.join(here['hcas'])} (GID {here['gid']})")
  say(f"  there {there['addr']} on {there['dev']}, RoCE {','.join(there['hcas'])} (GID {there['gid']})")
  return 0


# ---------------------------------------------------------------- noticing a newly cabled Spark
#
# Before link setup there is no IPv4 on the cable and no key, but IPv6 link-local always works:
# every host on a cable port answers the all-nodes address. Answers from addresses that aren't
# this machine's are the other end. The agents panel offers setup when status.json's `link`
# says one is available.

def link_path():
  return os.path.join(state_dir(), "link.json")


def _output(argv):
  try:
    return subprocess.run(argv, capture_output=True, text=True, timeout=5).stdout
  except (OSError, subprocess.SubprocessError):
    return ""


def own_link_local(ip_json):
  """This machine's own link-local addresses, from `ip -6 -j addr`."""
  try:
    return {a["local"] for link in json.loads(ip_json or "[]") for a in link.get("addr_info", []) if a.get("scope") == "link"}
  except (ValueError, TypeError, KeyError, AttributeError):
    return set()


def ping_answers(ping_output, own):
  """The link-local addresses answering an all-nodes ping, other than this machine's own."""
  return [address for address in dict.fromkeys(re.findall(r"from (fe80:[0-9a-f:]+)%", ping_output)) if address not in own]


def mdns_ssh_hosts(avahi_output):
  """Host names by link-local address, from `avahi-browse -rpt _ssh._tcp`."""
  names = {}
  for line in avahi_output.splitlines():
    fields = line.split(";")
    if len(fields) > 7 and fields[0] == "=" and fields[2] == "IPv6" and fields[7].startswith("fe80:"):
      names[fields[7]] = fields[6].removesuffix(".local")
  return names


def cable_neighbors():
  """Link-local addresses answering on this machine's cable ports, other than its own, as
  address%port, in the order they answered."""
  own = own_link_local(_output(["ip", "-6", "-j", "addr"]))
  if not own:
    return []
  return [f"{address}%{dev}" for dev in local_ports() for address in ping_answers(_output(["ping", "-6", "-n", "-w", "1", f"ff02::1%{dev}"]), own)]


def mdns_names():
  return mdns_ssh_hosts(_output(["avahi-browse", "-rpt", "_ssh._tcp"]))


def link_probe(profile):
  """What the cable shows now: a linked peer, another Spark waiting to be set up, or none.
  Recorded in link.json for status.json."""
  if NVIDIA_UNIFIED_FAMILIES.get(profile["platform"]) not in PAIR_FAMILIES:
    state = {"state": "none"}
  elif profile.get("pair"):
    state = {"state": "linked", "host": profile["pair"]["host"]}
  else:
    neighbors = cable_neighbors()
    if neighbors:
      names = mdns_names()
      # One machine answers from each of its cable addresses; the one mDNS names reads best.
      address = next((n for n in neighbors if n.split("%")[0] in names), neighbors[0])
      state = {"state": "available", "address": address, "host": names.get(address.split("%")[0])}
    else:
      state = {"state": "none"}
  write_atomic(link_path(), json.dumps(state, indent=2) + "\n")
  return state


def link_notify(profile):
  """Probes the cable and, the first time another Spark shows up waiting for setup, says so on
  the desktop with an action that opens link setup in a terminal. From the NetworkManager
  dispatcher hook (system/90-omarchy-local-ai-link) when a cable port comes up."""
  ensure_dir(state_dir())
  # Both cable ports come up together, each running this: one at a time, against its own record
  # of what it announced, whichever probe found the Spark first.
  with open(os.path.join(state_dir(), "link-notified"), "a+") as record:
    fcntl.flock(record, fcntl.LOCK_EX)
    record.seek(0)
    announced = record.read().strip()
    state = link_probe(profile)
    if state["state"] != "available" or announced == state["address"]:
      return state
    name = state["host"] or "another DGX Spark"
    sent = subprocess.run(["omarchy-notification-send", "-g", "󰌘", "--app-name", "Local AI", f"{name} is connected to this Spark",
                           "Set both up to run models across them",
                           "--exec", "omarchy-launch-floating-terminal-with-presentation", "omarchy-agent-local link setup"],
                          capture_output=True)
    # Only what reached the desktop counts as announced; a failed send is tried again next time.
    if sent.returncode != 0:
      return state
    record.seek(0)
    record.truncate()
    record.write(state["address"] + "\n")
  return state


def link_state():
  try:
    with open(link_path()) as handle:
      return json.load(handle)
  except (OSError, ValueError):
    return {"state": "none"}


# ---------------------------------------------------------------- current manifests


def models_dir():
  return os.path.join(state_dir(), "models")


def model_manifest_path(rid):
  return os.path.join(models_dir(), f"{rid}.json")


def load_model(rid):
  """The manifest a recipe currently runs from, or None if it isn't set up."""
  try:
    with open(model_manifest_path(rid)) as handle:
      return json.load(handle)
  except (OSError, ValueError):
    return None


def known_models():
  """Every recipe set up on this machine, in no particular order."""
  if not os.path.isdir(models_dir()):
    return []
  return sorted(name[:-5] for name in os.listdir(models_dir()) if name.endswith(".json") and ID_RE.match(name[:-5]))


def last_model_id():
  rid = _read(os.path.join(state_dir(), "last"))
  return rid if rid and load_model(rid) else None


def default_model(running_only=False):
  """The model a launch means when none is named: the last one
  started, if it qualifies, else any running one."""
  last = last_model_id()
  if last and (not running_only or unit_active(last)):
    return load_model(last)
  for rid in known_models():
    if not running_only or unit_active(rid):
      return load_model(rid)
  return None


def resolve_model(rid=None, running_only=False):
  if rid:
    if not ID_RE.match(rid):
      raise LocalError("Not a recipe id")
    manifest = load_model(rid)
    if not manifest:
      raise LocalError(f"{rid} isn't set up. Run: omarchy agent local use {rid}")
    if running_only and not unit_active(rid):
      raise LocalError(f"{rid} isn't running. Start it with: omarchy agent local start {rid}")
    return manifest
  manifest = default_model(running_only)
  if not manifest:
    raise LocalError("No local model is running. Run: omarchy agent local use best" if running_only else "No local model set up. Run: omarchy agent local use best")
  return manifest


def save_model(manifest, started=True):
  rid = manifest["recipe_id"]
  path = model_manifest_path(rid)
  text = json.dumps(manifest, indent=2) + "\n"
  write_atomic(path, text)
  if started:
    write_atomic(os.path.join(state_dir(), "last"), rid + "\n")
  return path


def stop_seconds_path(rid):
  return os.path.join(state_dir(), "stops", rid)


def expected_stop_seconds(rid):
  try:
    return float(_read(stop_seconds_path(rid)))
  except ValueError:
    return None


def load_seconds_path(rid):
  return os.path.join(state_dir(), "loads", rid)


def expected_load_seconds(rid, manifest):
  """How long this model's last start took, which the panel's loading bar
  runs against. Every start reads weights cold (stops evict them), so the
  last start predicts the next far better than the setup run did."""
  try:
    return float(_read(load_seconds_path(rid)))
  except ValueError:
    # A first start has no history yet: the catalog's measurement stands in.
    return (manifest.get("observed") or {}).get("load_seconds") or manifest.get("measured_load_seconds")


def error_path(rid):
  return os.path.join(state_dir(), "errors", rid)


def record_error(rid, reason):
  """Keep why a model couldn't start, for the panel to show on its line."""
  write_atomic(error_path(rid), " ".join(str(reason).split()) + "\n")


def clear_error(rid):
  if os.path.exists(error_path(rid)):
    os.unlink(error_path(rid))


class Lock:
  def __enter__(self):
    ensure_dir(state_dir())
    self.handle = open(os.path.join(state_dir(), "lock"), "w")
    try:
      fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      say("Another omarchy agent local command is running; waiting for it")
      fcntl.flock(self.handle, fcntl.LOCK_EX)
    return self

  def __exit__(self, *exc):
    fcntl.flock(self.handle, fcntl.LOCK_UN)
    self.handle.close()


def write_unit_files(rid, container_text):
  ensure_dir(quadlet_dir())
  write_atomic(os.path.join(quadlet_dir(), NETWORK_UNIT), render_network())
  if container_text is None:
    if os.path.exists(quadlet_path(rid)):
      os.unlink(quadlet_path(rid))
  else:
    write_atomic(quadlet_path(rid), container_text)
  systemctl("daemon-reload", check=True)


def start_unit(rid):
  say("Starting the model (waiting until it's loaded and healthy)")
  process = subprocess.Popen(["systemctl", "--user", "start", unit_name(rid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  started = time.time()
  while process.poll() is None:
    time.sleep(1)
    elapsed = int(time.time() - started)
    if elapsed and elapsed % 15 == 0:
      say(f"  still loading ({elapsed}s)")
  if process.returncode != 0:
    raise LocalError(f"The model didn't start: {process.stderr.read().strip()}\n{journal_tail(rid)}")
  return time.time() - started


def activate(manifest, recipe, runtime, gpu, key, catalog, profile):
  """Start a recipe from a resolved manifest. Other models keep running; a
  failure restores this recipe's previous unit, and restarts it if it ran."""
  rid = manifest["recipe_id"]
  previous_text = open(quadlet_path(rid)).read() if os.path.isfile(quadlet_path(rid)) else None
  was_running = unit_active(rid)
  container_text = render_container(rid, recipe, runtime, manifest["variant"], manifest["roles"], gpu, manifest["fileset"])
  if was_running:
    say(f"Restarting {rid} with the new setup (its in-flight requests are interrupted)")
    write_status(catalog, profile, {rid: "deactivating"})
    systemctl("stop", unit_name(rid))
  clear_error(rid)
  try:
    evict_idle_weights(keep=[rid])
    before_ram, before_gpu = check_headroom(manifest["variant"], gpu)
    before_peer = peer_free_gib(manifest)
    write_unit_files(rid, container_text)
    write_pair_setup(manifest)
    systemctl("reset-failed", unit_name(rid))
    write_status(catalog, profile, {rid: "activating"}, pending={rid: manifest})
    load_seconds = start_unit(rid)
    post_start_checks(rid, key)
  except (LocalError, subprocess.CalledProcessError) as error:
    systemctl("stop", unit_name(rid))
    write_unit_files(rid, previous_text)
    if os.path.exists(pair_setup_path(rid)):
      os.unlink(pair_setup_path(rid))
    write_pair_setup(load_model(rid))
    if previous_text is not None and was_running:
      say(f"Restoring the previous setup of {rid}")
      systemctl("start", unit_name(rid))
    short = getattr(error, "short", None) or str(error).splitlines()[0]
    record_error(rid, short)
    write_status(catalog, profile)
    raise LocalError(str(error), short=short)
  after_ram, after_gpu = free_memory(gpu)
  manifest["observed"] = {
    "load_seconds": round(load_seconds, 1),
    "ram_used_gib": ram_used_gib(manifest, before_ram, after_ram, before_peer, peer_free_gib(manifest)),
    "gpu_used_gib": round(before_gpu - after_gpu, 1),
  }


# ---------------------------------------------------------------- commands


def _lookup_recipe(catalog, target):
  if target not in catalog.recipes:
    known = ", ".join(sorted(catalog.recipes)) or "none"
    raise LocalError(f"No recipe '{target}' (known: {known})")
  return catalog.recipes[target]


def cmd_use(argv):
  parser = argparse.ArgumentParser(prog="omarchy agent local use")
  parser.add_argument("target", help="best, or a recipe id")
  parser.add_argument("--ctx", type=int)
  parser.add_argument("--gpu")
  parser.add_argument("--seed", help="reuse matching files from this directory before downloading")
  parser.add_argument("--unqualified", action="store_true", help="run a recipe not measured on this machine class")
  parser.add_argument("--detach", action="store_true", help="set it up in the background; the agents panel shows its progress")
  args = parser.parse_args(argv)
  catalog = load_catalog()
  profile = machine_profile()
  gpu = choose_gpu(profile, args.gpu) if args.gpu else None
  if args.target == "best":
    rid, variant, skipped = resolve_best(catalog, profile, gpu, args.ctx)
    gpu = recipe_gpu(profile, catalog.recipes[rid], args.gpu)
    for skipped_id, reason in skipped:
      say(f"Skipping {skipped_id}: {reason}")
    say(f"Best for this machine ({gpu['family']}, {class_text(profile, gpu)}): {rid}")
  else:
    rid = args.target
    gpu = recipe_gpu(profile, _lookup_recipe(catalog, rid), args.gpu)
    variant, reason = compatibility(catalog, rid, profile, gpu, args.ctx)
    if variant is None:
      if not args.unqualified:
        raise LocalError(f"{rid} isn't qualified for this machine: {reason}\nRun with --unqualified to try anyway.")
      variant, fit_reason = pick_variant(catalog.recipes[rid], profile, gpu, args.ctx)
      if variant is None:
        raise LocalError(f"{rid} doesn't fit: {fit_reason}")
      say(f"Warning: {reason}")
  measured_load = load_measurement(catalog, catalog.recipes[rid], profile, gpu).get("load_seconds")
  if args.detach:
    return detach_use(rid, args, catalog.recipes[rid], measured_load)
  # A first setup shows in the panel from now on: waiting for the lock, then
  # downloading, then loading once activate() writes its pending row.
  fresh = load_model(rid) is None
  if fresh:
    if installing_idle(rid):
      begin_installing(rid, catalog.recipes[rid], measured_load)
    # The process carrying the setup is the one remove signals, from now on,
    # so a setup waiting for another's lock can be cancelled too. A second use
    # of the same recipe leaves a live owner in place; a detached unit takes
    # over the record its caller wrote.
    owner = (load_installing(rid) or {}).get("pid")
    if not (isinstance(owner, int) and owner != os.getpid() and os.path.exists(f"/proc/{owner}")):
      update_installing(rid, pid=os.getpid())
  try:
    with Lock():
      if fresh and load_model(rid) is None and load_installing(rid) is None:
        # remove cleared its record while it waited: nothing to fetch.
        raise LocalError(f"The setup of {rid} was cancelled")
      if fresh:
        # Whoever holds the lock does the work, so it is the one remove signals:
        # a queued retry takes over from the attempt that failed before it.
        update_installing(rid, pid=os.getpid())
      return _use_setup(rid, variant, args, catalog, profile, gpu, fresh, measured_load)
  except KeyboardInterrupt:
    # A caller still waiting on another's setup leaves that setup's record alone.
    if fresh and load_model(rid) is None and owns_installing(rid):
      clear_installing(rid)
      write_status(catalog, profile)
    raise
  except Exception as error:
    if fresh and load_model(rid) is None and owns_installing(rid):
      fail_installing(rid, getattr(error, "short", None) or (str(error).splitlines() or ["Setup failed"])[0])
    raise


def detach_use(rid, args, recipe, measured_load=None):
  """Run the setup as a transient user unit, so it outlives the panel or
  terminal that asked for it; its output goes to the journal."""
  unit = install_unit(rid)
  if systemctl("is-active", "--quiet", unit).returncode == 0:
    say(f"{rid} is already being set up; follow it with: journalctl --user -u {unit} -f")
    return 0
  systemctl("reset-failed", unit)
  if load_model(rid) is None and installing_idle(rid):
    begin_installing(rid, recipe, measured_load)
  entry = os.path.join(os.path.dirname(os.path.realpath(__file__)), "omarchy-local-ai")
  command = ["systemd-run", "--user", "--collect", "--quiet", f"--unit={unit}", f"--description=Set up local model {rid}"]
  # The unit takes these from systemd-run's own environment by name, so the
  # token's value never appears in a process's arguments. The catalog is made
  # absolute: the unit doesn't run from this directory.
  env = dict(os.environ)
  if env.get("OMARCHY_LOCAL_AI_CATALOG"):
    env["OMARCHY_LOCAL_AI_CATALOG"] = os.path.abspath(env["OMARCHY_LOCAL_AI_CATALOG"])
  for name in ("OMARCHY_LOCAL_AI_CATALOG", "HF_TOKEN"):
    if env.get(name):
      command.append(f"--setenv={name}")
  command += [entry, "use", rid]
  if args.ctx:
    command += ["--ctx", str(args.ctx)]
  if args.gpu:
    command += ["--gpu", args.gpu]
  if args.seed:
    command += ["--seed", os.path.abspath(args.seed)]
  if args.unqualified:
    command.append("--unqualified")
  result = subprocess.run(command, capture_output=True, text=True, env=env)
  if result.returncode != 0:
    reason = (result.stderr.strip().splitlines() or ["systemd-run failed"])[-1]
    if load_model(rid) is None:
      fail_installing(rid, reason)
    raise LocalError(f"Can't start the setup of {rid}: {reason}")
  say(f"Setting up {rid} in the background; the agents panel shows its progress, or follow it with: journalctl --user -u {unit} -f")
  return 0


def _use_setup(rid, variant, args, catalog, profile, gpu, fresh, measured_load=None):
  recipe = catalog.recipes[rid]
  runtime = catalog.runtimes[recipe["runtime"]]
  if runtime.get("tier") == "third-party":
    review = load_review(catalog, runtime)
    say(f"Runtime {recipe['runtime']} is third-party: {review.get('publisher', 'unknown publisher')}, license {review.get('license_url', 'unknown')}")
  if runtime.get("key_on_cmdline"):
    say(f"Warning: runtime {recipe['runtime']} takes its key as a command-line argument, visible in the process list to other local users")
  preflight(runtime, gpu)
  files, roles = build_artifacts(recipe)
  say(f"Recipe {rid}: {len(files)} file(s), {gib(sum(f['size'] for f in files)):.1f} GiB, context {variant['ctx']}")
  for license_name in sorted({f"{f['repo']} ({f['license']})" for f in files}):
    say(f"  weights: {license_name}")
  progress = Progress(rid=rid if fresh else None)
  fileset = materialize(rid, files, roles, args.seed, progress)
  if fresh:
    # Until the manifest exists, removal knows the image from the record.
    update_installing(rid, image=runtime["image"])
  ensure_image(runtime["image"], progress)
  key = ensure_key()
  if runtime.get("pair"):
    if not profile.get("pair"):
      raise LocalError(f"{rid} runs across two machines, and no paired machine was found on a ConnectX-7 link")
    say(f"Linked to {profile['pair']['host']} at {profile['pair']['peer']}")
  manifest = {
    "schema": SCHEMA,
    "recipe_id": rid,
    "runtime_id": recipe["runtime"],
    "profile": profile,
    "gpu": gpu["family"],
    "variant": variant,
    "recipe": recipe,
    "runtime": runtime,
    "image": runtime["image"],
    "files": files,
    "roles": roles,
    "fileset": fileset,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "measured_load_seconds": measured_load,
  }
  if runtime.get("pair"):
    pair_setup(rid, manifest, runtime, profile["pair"])
  _, _, _, skipped = server_invocation(runtime, recipe, variant, roles, rid)
  if skipped:
    say(f"Note: runtime {recipe['runtime']} takes no {', '.join(skipped)} default; clients choose their own")
  # Downloaded is installed: when other models hold the memory it needs,
  # the setup is kept, stopped, and starts later from the panel's play.
  if not unit_active(rid) and load_model(rid) is None:
    try:
      check_headroom(variant, gpu)
    except LocalError as error:
      write_unit_files(rid, render_container(rid, recipe, runtime, variant, roles, gpu, fileset))
      write_pair_setup(manifest)
      clear_error(rid)
      # Its first start runs the checks activate() would have.
      manifest["unchecked"] = True
      path = save_model(manifest, started=False)
      clear_installing(rid)
      prune_filesets(rid, fileset)
      refresh_list_cache(catalog, profile)
      write_status(catalog, profile)
      say("")
      say(f"Installed {rid}, but it can't start now: {error}")
      first = "Stop another model, then press" if error.short.startswith(NOT_ENOUGH_MEMORY) else "Then press"
      say(f"{first} play in the agents panel or run: omarchy agent local start {rid}")
      say(f"Manifest: {path}")
      return 0
  activate(manifest, recipe, runtime, gpu, key, catalog, profile)
  path = save_model(manifest)
  clear_installing(rid)
  prune_filesets(rid, fileset)
  refresh_list_cache(catalog, profile)
  write_status(catalog, profile)
  say("")
  say(f"Running {rid} at {endpoint(rid)} (model id {rid}, key in {key_path()})")
  say(f"Loaded in {manifest['observed']['load_seconds']:.0f}s, using about {manifest['observed']['ram_used_gib']:.0f} GiB")
  say(f"Manifest: {path}")
  say(f"Launch an agent against it with: omarchy agent --local-model {rid}")
  return 0


def running_models():
  return {rid for rid in known_models() if unit_active(rid)}


def refresh_list_cache(catalog, profile):
  rows = recipe_listing(catalog, profile, running_models())
  write_list_cache(rows)
  return rows


def cmd_list(argv):
  parser = argparse.ArgumentParser(prog="omarchy agent local list")
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args(argv)
  profile = machine_profile()
  catalog = load_catalog()
  rows = recipe_listing(catalog, profile, running_models())
  write_list_cache(rows)
  if args.json:
    print(json.dumps(rows, indent=2))
    return 0
  first = profile["gpus"][0] if profile["gpus"] else None
  gpu = first["family"] if first else "no GPU"
  memory = class_text(profile, first) if first else f"{profile['ram_class']} GB"
  paired = f" · linked to {profile['pair']['host']}" if profile.get("pair") else ""
  say(f"This machine: {gpu}, {memory}, {profile['platform']}{paired}")
  for row in rows:
    marks = [mark for mark, on in (("best", row["best"]), ("running", row["active"]), ("installed", row["installed"])) if on]
    tier = f" · third-party ({row['publisher']})" if row["tier"] == "third-party" else ""
    say(f"{'●' if row['available'] else '○'} {row['id']}{' [' + ', '.join(marks) + ']' if marks else ''}")
    say(f"    {row['name']} on {row['runtime']}{tier} — {row['summary']}")
    if not row["available"]:
      say(f"    unavailable: {row['reason']}")
  return 0


def _unix_ms(value):
  value = (value or "").lstrip("@")
  return int(value) * 1000 if value.isdigit() else None


def start_fit(manifest, profile):
  """Whether a stopped model fits in the memory free right now: (fits, reason).
  Rewritten with every start and stop, the moments free memory changes."""
  variant = manifest["variant"]
  try:
    gpu = choose_gpu(profile, manifest.get("gpu"))
  except LocalError as error:
    return False, error.short
  available, gpu_free = free_memory(gpu)
  if available >= variant["peak_ram_gib"] + RESERVE_GIB and gpu_free >= variant["peak_gpu_gib"]:
    return True, ""
  return False, "Not enough memory"


def model_record(rid, manifest, profile):
  props = unit_props(rid, "ActiveState", "SubState", "Result", "InactiveExitTimestamp", "ActiveExitTimestamp")
  unit = props.get("ActiveState", "inactive") or "inactive"
  # A removal in progress outranks the unit's own state, which passes
  # through stopping and stopped on the way.
  removing = _read(removing_path(rid))
  if removing:
    unit = "removing"
  recipe = manifest["recipe"]
  vendor = recipe.get("vendor", "")
  observed = manifest.get("observed") or {}
  fits, fit_reason = (True, "") if unit in ("active", "activating", "deactivating", "removing") else start_fit(manifest, profile)
  error = _read(error_path(rid))
  # A memory refusal isn't a failure to show: play is greyed out with its
  # reason before a start is even tried.
  if error.startswith(NOT_ENOUGH_MEMORY):
    error = ""
  if not error and unit == "failed":
    error = f"Stopped unexpectedly ({props.get('Result') or 'failed'})"
  return {
    # Why the last start failed, shown where the progress bar was.
    "error": error,
    # Whether play can start it now; the panel greys play out when it can't.
    "fits": fits,
    "fit_reason": fit_reason,
    # Whether the other machines on the tailnet can use it.
    "shared": rid in shared_recipes(),
    "recipe": rid,
    "name": recipe.get("name", rid),
    "vendor": vendor,
    "runtime": manifest["runtime_id"],
    "image": manifest["image"],
    "ctx": manifest["variant"]["ctx"],
    "unit": unit,
    "sub_state": props.get("SubState", ""),
    "result": props.get("Result", ""),
    "port": port_for(rid),
    "endpoint": endpoint(rid),
    "observed": observed or None,
    # What the panel's memory bar is a share of: a paired model's machines together.
    "ram_total_gib": nodes_ram_gib(manifest, profile),
    # When the current start or stop began, and how long the last load took:
    # the panel's progress bar measures the one against the other.
    "since_ms": int(os.path.getmtime(removing_path(rid)) * 1000) if removing else _unix_ms(props.get("InactiveExitTimestamp") if unit == "activating" else props.get("ActiveExitTimestamp") if unit == "deactivating" else ""),
    "expected_load_seconds": expected_load_seconds(rid, manifest),
    "expected_stop_seconds": expected_stop_seconds(rid),
    "expected_remove_seconds": float(removing) if removing else None,
  }


def status_record(catalog, profile):
  models = {rid: model_record(rid, load_model(rid), profile) for rid in known_models()}
  for rid in installing_models():
    record = load_installing(rid)
    if record is not None and rid not in models:
      models[rid] = install_row(rid, record)
  record = {
    "models": models,
    # Why sharing on the tailnet isn't working, for the panel to show.
    "share_error": _read(share_error_path()),
    "last": last_model_id(),
    "ram_total_gib": round(gib(meminfo().get("MemTotal", 0)), 1),
    "best": None,
    # A Spark on the cable: none, available (not set up yet: address, host) or linked (host).
    "link": {"state": "linked", "host": profile["pair"]["host"]} if profile.get("pair") else link_state(),
  }
  try:
    record["best"] = resolve_best(catalog, profile)[0]
  except LocalError:
    pass
  return record


def write_status(catalog, profile, transitions=None, pending=None, publish=True):
  """Rewrite status.json, which the agents panel watches. `transitions` reports
  a start or stop before systemd reaches it, so a click shows at once;
  `pending` describes a model being set up before its manifest is saved.

  Collecting and replacing happen under one lock, so a writer that started
  earlier can't land its older snapshot over a newer one."""
  ensure_dir(state_dir())
  with open(os.path.join(state_dir(), "status.lock"), "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    record = status_record(catalog, profile)
    now = int(time.time() * 1000)
    for rid, manifest in (pending or {}).items():
      previous = load_model(rid) or {}
      record["models"][rid] = model_record(rid, {**manifest, "observed": previous.get("observed")}, profile)
    for rid, unit in (transitions or {}).items():
      if rid in record["models"] and record["models"][rid]["unit"] != "removing":
        record["models"][rid]["unit"] = unit
        record["models"][rid]["since_ms"] = now if unit in ("activating", "deactivating") else None
        if unit == "active":
          record["models"][rid]["error"] = ""
    write_atomic(os.path.join(state_dir(), "status.json"), json.dumps(record, indent=2) + "\n")
    # The tailnet sees the same transitions: a shared model's listing is
    # republished with every status write that changes it.
    if publish:
      publish_shares(record, catalog, profile)
      error = _read(share_error_path())
      if error != record["share_error"]:
        record["share_error"] = error
        write_atomic(os.path.join(state_dir(), "status.json"), json.dumps(record, indent=2) + "\n")
    # Every setup, removal, start and refresh passes through here, so pi's
    # extension follows them, under the same lock so an older writer can't put
    # back a model a newer one removed. A failure to write it never fails theirs.
    try:
      sync_pi_extension()
    except OSError:
      pass
    return record


def cmd_status(argv):
  parser = argparse.ArgumentParser(prog="omarchy agent local status")
  parser.add_argument("--json", action="store_true")
  parser.add_argument("--write", action="store_true", help=argparse.SUPPRESS)
  parser.add_argument("--started", metavar="RECIPE", help=argparse.SUPPRESS)
  parser.add_argument("--stopped", metavar="RECIPE", help=argparse.SUPPRESS)
  args = parser.parse_args(argv)
  catalog = load_catalog()
  profile = machine_profile()
  if args.write:
    # From the units' hooks, which run before systemd finishes the transition
    # they report, so the outcome is passed in rather than read back.
    transitions = {}
    if args.started and ID_RE.match(args.started):
      clear_error(args.started)
      transitions[args.started] = "active"
    if args.stopped and ID_RE.match(args.stopped):
      result = os.environ.get("SERVICE_RESULT", "success")
      if result == "success":
        transitions[args.stopped] = "inactive"
      else:
        record_error(args.stopped, failure_reason(args.stopped, result))
        transitions[args.stopped] = "failed"
    write_status(catalog, profile, transitions)
    return 0
  record = status_record(catalog, profile)
  if args.json:
    print(json.dumps(record, indent=2))
    return 0
  if not record["models"]:
    say("No local model has been set up. Run: omarchy agent local use best")
    return 0
  for rid, model in record["models"].items():
    state = {"active": "running", "activating": "loading", "deactivating": "stopping"}.get(model["unit"], model["unit"])
    say(f"{rid} ({model['name']}) on {model['runtime']}: {state}")
    say(f"  endpoint {model['endpoint']} · model id {rid} · context {model['ctx']}")
    if model["unit"] == "failed":
      say(f"  last failure: {model['result']}")
      tail = journal_tail(rid, 5)
      if tail:
        say("  " + tail.replace("\n", "\n  "))
  if record["best"] and record["best"] not in record["models"]:
    say(f"The recommendation for this machine is {record['best']}: omarchy agent local use best")
  return 0


def cmd_stop(argv):
  parser = argparse.ArgumentParser(prog="omarchy agent local stop")
  parser.add_argument("recipe", nargs="?", help="the model to stop (default: every running one)")
  args = parser.parse_args(argv)
  catalog = load_catalog()
  profile = machine_profile()
  targets = [resolve_model(args.recipe)["recipe_id"]] if args.recipe else sorted(running_models())
  if not targets:
    say("No local model is running")
    return 0
  # A model takes seconds to let go of its memory; the panel says so meanwhile.
  write_status(catalog, profile, {rid: "deactivating" for rid in targets})
  for rid in targets:
    started = time.time()
    systemctl("stop", unit_name(rid))
    # How long this model takes to let go; the panel's stopping bar measures
    # the next stop against it.
    write_atomic(stop_seconds_path(rid), f"{time.time() - started:.1f}\n")
  # Give the memory back whole, not as cached weights the next model has to
  # fight the kernel for.
  evict_idle_weights()
  write_status(catalog, profile)
  say(f"Stopped {', '.join(targets)}")
  return 0


def cmd_start(argv):
  parser = argparse.ArgumentParser(prog="omarchy agent local start")
  parser.add_argument("recipe", nargs="?", help="the model to start (default: the last one started)")
  args = parser.parse_args(argv)
  catalog = load_catalog()
  profile = machine_profile()
  # Admission is serialized with use and remove: two starts checking the same
  # free memory at once would both pass and together overcommit it.
  with Lock():
    return _start(args, catalog, profile)


def _start(args, catalog, profile):
  manifest = resolve_model(args.recipe)
  rid = manifest["recipe_id"]
  if unit_active(rid):
    say(f"{rid} is already running at {endpoint(rid)}")
    return 0
  gpu = choose_gpu(profile, manifest["gpu"])
  # Shown as loading from the click on; the memory checks below take seconds.
  clear_error(rid)
  started = time.time()
  write_status(catalog, profile, {rid: "activating"})
  evict_idle_weights(keep=[rid])
  try:
    before_ram, before_gpu = check_headroom(manifest["variant"], gpu)
  except LocalError as error:
    record_error(rid, error.short)
    write_status(catalog, profile)
    raise
  before_peer = peer_free_gib(manifest)
  systemctl("reset-failed", unit_name(rid))
  result = systemctl("start", unit_name(rid))
  write_atomic(os.path.join(state_dir(), "last"), rid + "\n")
  if result.returncode != 0:
    record_error(rid, "Didn't start: omarchy agent local status shows why")
  else:
    # Timed from the same moment the panel's bar starts from, the click.
    write_atomic(load_seconds_path(rid), f"{time.time() - started:.1f}\n")
  write_status(catalog, profile)
  if result.returncode != 0:
    raise LocalError(f"{rid} didn't start: {result.stderr.strip()}\n{journal_tail(rid)}")
  if manifest.get("unchecked"):
    try:
      post_start_checks(rid, ensure_key())
    except LocalError as error:
      systemctl("stop", unit_name(rid))
      record_error(rid, error.short)
      write_status(catalog, profile)
      raise
    manifest.pop("unchecked")
  # The unit only reports started once the model is loaded and healthy, so
  # what memory went since the check is what it holds. A setup saved while
  # memory was busy gets its first measurement here; later starts refresh it.
  after_ram, after_gpu = free_memory(gpu)
  manifest["observed"] = {
    "load_seconds": round(time.time() - started, 1),
    "ram_used_gib": ram_used_gib(manifest, before_ram, after_ram, before_peer, peer_free_gib(manifest)),
    "gpu_used_gib": round(before_gpu - after_gpu, 1),
  }
  write_atomic(model_manifest_path(rid), json.dumps(manifest, indent=2) + "\n")
  write_status(catalog, profile)
  say(f"Started {rid} at {endpoint(rid)}")
  return 0


def cmd_remove(argv):
  parser = argparse.ArgumentParser(prog="omarchy agent local remove")
  parser.add_argument("recipe")
  parser.add_argument("--local-only", action="store_true", help="for a model across two machines: remove it here even if the other can't be reached")
  args = parser.parse_args(argv)
  catalog = load_catalog()
  profile = machine_profile()
  rid = args.recipe
  if not ID_RE.match(rid):
    raise LocalError("Not a recipe id")
  setting_up = load_installing(rid)
  if setting_up is not None:
    # Cancel first: the setup holds the lock this removal waits for.
    systemctl("stop", install_unit(rid))
    stop_setup_process(setting_up.get("pid"))
  with Lock():
    manifest = load_model(rid)
    if manifest is None and setting_up is not None:
      # Cancelled while its model loaded: activation starts the unit before
      # the manifest exists, so stop it before its files go.
      systemctl("stop", unit_name(rid))
      # Read again: the setup may have reached its image since the first look.
      setting_up = load_installing(rid) or setting_up
      if setting_up.get("pair"):
        pair_remove({"recipe_id": rid, "pair": setting_up["pair"]}, args.local_only)
      clear_installing(rid)
      freed, _ = _remove_files(rid, setting_up.get("image"))
      # Only this setup's half-downloaded files: others resume from theirs.
      for name in setting_up.get("partials") or []:
        if isinstance(name, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", name.removesuffix(".part")):
          path = os.path.join(data_dir(), "staging", name)
          if os.path.exists(path):
            freed += os.path.getsize(path)
            os.unlink(path)
      write_status(catalog, profile)
      refresh_list_cache(catalog, profile)
      say(f"Cancelled the setup of {rid}" + (f", freeing {gib(freed):.1f} GiB" if freed else ""))
      return 0
    running = unit_active(rid)
    size_gib = gib(sum(record.get("size", 0) for record in (manifest or {}).get("files", [])))
    # Shown as removing until it's gone: the marker outlasts the stop below,
    # whose hook would otherwise report the model as merely stopped.
    expected = (expected_stop_seconds(rid) or 10 if running else 0) + size_gib * removal_rate()
    if manifest:
      write_atomic(removing_path(rid), f"{expected:.1f}\n")
      write_status(catalog, profile)
    started = time.time()
    try:
      if running:
        say(f"Stopping {rid}")
        systemctl("stop", unit_name(rid))
      deleting = time.time()
      if manifest:
        pair_remove(manifest, args.local_only)
      result = _remove_files(rid)
      if size_gib > 1:
        write_atomic(os.path.join(state_dir(), "removal_rate"), f"{(time.time() - deleting) / size_gib:.4f}\n")
      # Only once everything else is gone: an interrupted removal leaves a
      # model that status still lists and remove can finish.
      if os.path.exists(model_manifest_path(rid)):
        os.unlink(model_manifest_path(rid))
      # A model set up again later starts out private.
      if rid in shared_recipes():
        save_shares(set(shared_recipes()) - {rid})
    finally:
      if os.path.exists(removing_path(rid)):
        os.unlink(removing_path(rid))
        # Also when interrupted: the panel would otherwise stay on Removing.
        write_status(catalog, profile)
  refresh_list_cache(catalog, profile)
  write_status(catalog, profile)
  freed, images = result
  say(f"Removed {rid} in {time.time() - started:.0f}s: {gib(freed):.1f} GiB of weights no other recipe uses" + (f", {len(images)} image(s)" if images else ""))
  return 0


def removing_path(rid):
  return os.path.join(state_dir(), "removing", rid)


def stop_setup_process(pid):
  """A setup run from a terminal, which has no unit to stop: ends it if the
  pid still belongs to this engine's use command."""
  if not isinstance(pid, int) or pid <= 1 or pid == os.getpid():
    return
  cmdline = _read(f"/proc/{pid}/cmdline").split("\0")
  if "use" not in cmdline or not any(part.endswith(("omarchy-local-ai", "agent_local.py")) for part in cmdline):
    return
  try:
    os.kill(pid, signal.SIGTERM)
  except OSError:
    return
  for _ in range(100):
    if not os.path.exists(f"/proc/{pid}"):
      return
    time.sleep(0.1)


# ---------------------------------------------------------------- setups in progress
#
# A setup has no manifest until its model first starts, so the panel learns of it
# from its installing record: what is being fetched, how far along and how fast,
# or why it failed. The record goes once the manifest exists; a failed one stays
# until `remove` dismisses it.

INSTALL_PHASES = ("waiting", "downloading", "failed")


def installing_path(rid):
  return os.path.join(state_dir(), "installing", f"{rid}.json")


def install_unit(rid):
  """The transient unit a detached setup runs in."""
  return f"omarchy-agent-local-use-{rid}.service"


def load_installing(rid):
  try:
    with open(installing_path(rid)) as handle:
      record = json.load(handle)
  except (OSError, ValueError):
    return None
  return record if isinstance(record, dict) and record.get("phase") in INSTALL_PHASES else None


def installing_models():
  folder = os.path.join(state_dir(), "installing")
  try:
    names = sorted(os.listdir(folder))
  except OSError:
    return []
  return [name[:-5] for name in names if name.endswith(".json") and ID_RE.match(name[:-5])]


def begin_installing(rid, recipe, load_seconds=None):
  """`load_seconds` is how long the catalog measured its start: the panel gives the
  download and the load that follows their shares of one bar from it."""
  record = {"name": recipe.get("name", rid), "vendor": recipe.get("vendor", ""), "phase": "waiting",
            "done_bytes": 0, "total_bytes": 0, "rate_bytes_s": 0, "since_ms": int(time.time() * 1000), "error": "",
            "expected_load_seconds": load_seconds}
  write_atomic(installing_path(rid), json.dumps(record) + "\n")
  publish_installing(rid, record)
  return record


def update_installing(rid, **fields):
  record = load_installing(rid)
  if record is None:
    return
  record.update(fields)
  write_atomic(installing_path(rid), json.dumps(record) + "\n")
  publish_installing(rid, record)


def owns_installing(rid):
  return (load_installing(rid) or {}).get("pid") == os.getpid()


def installing_idle(rid):
  """No setup of this recipe is under way: none listed, or only one that failed,
  which a new attempt replaces rather than inheriting its error and start."""
  record = load_installing(rid)
  return record is None or record.get("phase") == "failed"


def fail_installing(rid, reason):
  update_installing(rid, phase="failed", rate_bytes_s=0, error=reason)


def clear_installing(rid):
  if os.path.exists(installing_path(rid)):
    os.unlink(installing_path(rid))


def install_row(rid, record):
  """A setup in progress, shaped like an installed model's row so the panel
  places it the same way; `install` carries what its progress bar shows."""
  return {
    "error": record.get("error", ""),
    "fits": True,
    "fit_reason": "",
    "recipe": rid,
    "name": record.get("name", rid),
    "vendor": record.get("vendor", ""),
    "runtime": "",
    "image": "",
    "ctx": None,
    "unit": "installing",
    "sub_state": "",
    "result": "",
    "port": None,
    "endpoint": "",
    "observed": None,
    "since_ms": record.get("since_ms"),
    "expected_load_seconds": record.get("expected_load_seconds"),
    "expected_stop_seconds": None,
    "expected_remove_seconds": None,
    "install": {key: record.get(key, 0) for key in ("phase", "done_bytes", "total_bytes", "rate_bytes_s")},
  }


def publish_installing(rid, record):
  """Rewrite only this setup's row: progress lands twice a second, and the full
  status a write_status collects asks systemd about every model."""
  ensure_dir(state_dir())
  path = os.path.join(state_dir(), "status.json")
  with open(os.path.join(state_dir(), "status.lock"), "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
      with open(path) as handle:
        status = json.load(handle)
    except (OSError, ValueError):
      status = None
    if not isinstance(status, dict) or not isinstance(status.get("models"), dict):
      status = {"models": {}, "last": None, "ram_total_gib": round(gib(meminfo().get("MemTotal", 0)), 1), "best": None}
    # An installed model's own row wins: setting up a recipe again leaves it listed as it is.
    if status["models"].get(rid, {}).get("unit", "installing") != "installing":
      return
    status["models"][rid] = install_row(rid, record)
    write_atomic(path, json.dumps(status, indent=2) + "\n")


def removal_rate():
  """Seconds per GiB the last removal took, deleting weights and image."""
  try:
    return float(_read(os.path.join(state_dir(), "removal_rate")))
  except ValueError:
    return 0.1


def _remove_files(rid, setup_image=None):
  write_unit_files(rid, None)
  for path in (error_path(rid), stop_seconds_path(rid), load_seconds_path(rid)):
    if os.path.exists(path):
      os.unlink(path)
  if _read(os.path.join(state_dir(), "last")) == rid:
    os.unlink(os.path.join(state_dir(), "last"))
  ports = load_ports()
  if ports.pop(rid, None) is not None:
    write_atomic(ports_path(), json.dumps(ports, indent=2, sort_keys=True) + "\n")
  manifest = load_model(rid)
  removed_images = {manifest["image"]} if manifest else set()
  if isinstance(setup_image, str) and IMAGE_RE.match(setup_image):
    removed_images.add(setup_image)
  shutil.rmtree(os.path.join(data_dir(), "models", rid), ignore_errors=True)
  freed = 0
  # A blob stays only while another recipe's files link it. A link from
  # outside the store, like the --seed copy it was set up from, isn't a
  # reason to keep it, though dropping it frees no space.
  used = set()
  for directory, _, names in os.walk(os.path.join(data_dir(), "models")):
    for name in names:
      info = os.stat(os.path.join(directory, name))
      used.add((info.st_dev, info.st_ino))
  for directory, _, names in os.walk(os.path.join(data_dir(), "blobs")):
    for name in names:
      path = os.path.join(directory, name)
      info = os.stat(path)
      if (info.st_dev, info.st_ino) in used:
        continue
      if info.st_nlink == 1:
        freed += info.st_size
      os.unlink(path)
  kept_images = {load_model(other)["image"] for other in known_models() if other != rid}
  images = []
  for image in sorted(removed_images - kept_images):
    if subprocess.run(["podman", "rmi", image], capture_output=True).returncode == 0:
      images.append(image)
  return freed, images


def cmd_profile(argv):
  argparse.ArgumentParser(prog="omarchy agent local profile").parse_args(argv)
  profile = machine_profile()
  print(json.dumps(profile, indent=2))
  return 0


def cmd_refresh(argv):
  parser = argparse.ArgumentParser(prog="omarchy agent local refresh")
  parser.add_argument("--quiet", action="store_true")
  args = parser.parse_args(argv)
  catalog = load_catalog()
  profile = machine_profile()
  link_probe(profile)
  refresh_list_cache(catalog, profile)
  write_status(catalog, profile)
  if not args.quiet:
    say("Refreshed what fits this machine")
  return 0


def cmd_validate(argv):
  parser = argparse.ArgumentParser(prog="omarchy agent local validate")
  parser.add_argument("catalog_dir")
  parser.add_argument("--online", action="store_true", help="also check every Hugging Face file and recipe license")
  args = parser.parse_args(argv)
  catalog = load_catalog_dir(args.catalog_dir)
  errors = catalog.errors()
  if args.online and not errors:
    cache = {}
    for rid, recipe in catalog.recipes.items():
      try:
        build_artifacts(recipe, cache)
      except LocalError as error:
        errors.append(f"recipes/{rid}: {error}")
      for artifact in recipe["artifacts"]:
        info = http_json(f"https://huggingface.co/api/models/{artifact['repo']}/revision/{artifact['revision']}")
        declared = (info.get("cardData") or {}).get("license")
        if declared and not artifact["license"].startswith(str(declared)):
          errors.append(f"recipes/{rid}: {artifact['repo']} declares license '{declared}', recipe says '{artifact['license']}'")
  for error in errors:
    print(error)
  if errors:
    return 1
  say(f"Catalog valid: {len(catalog.recipes)} recipe(s), {len(catalog.runtimes)} runtime(s)")
  return 0


# ---------------------------------------------------------------- tailnet sharing
#
# A machine shares a model by putting its loopback endpoint behind
# `tailscale serve` at /local-ai/<recipe>, reachable only on the tailnet with
# Tailscale's own HTTPS certificate. The container, its loopback port and its
# key check stay exactly as they are; the model still refuses a request
# without the key. Beside the models, /local-ai/models.json lists what this
# machine shares and whether each is running. It never contains the key, which
# a client only gets by pairing over SSH.
#
# Every other machine finds those listings by asking its own tailscaled which
# peers are online and fetching each one's models.json. Hosts it has seen stay
# in network.json while they're away, marked offline, so a model that drops
# off the tailnet keeps its line in the panel instead of vanishing.

SHARE_ROOT = "/local-ai"
LISTING_ROUTE = SHARE_ROOT + "/models.json"
NETWORK_SCHEMA = 1
# How long an unpaired host that stopped answering stays listed as offline.
# A paired host stays until it's forgotten.
FORGET_UNPAIRED_MS = 7 * 24 * 3600 * 1000
LISTING_TIMEOUT = 4
HOST_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")
DNS_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)+$")
SSH_TARGET_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.@-]{0,252}$")
REMOTE_UNITS = ("installing", "active", "activating", "inactive", "failed", "deactivating", "removing")
OPERATOR_HINT = "Let this user change Tailscale Serve: sudo tailscale set --operator=$USER"


def shares_path():
  return os.path.join(config_dir(), "shares.json")


def shared_recipes():
  try:
    with open(shares_path()) as handle:
      recipes = json.load(handle).get("recipes", [])
  except (OSError, ValueError, AttributeError):
    return []
  return sorted({rid for rid in recipes if isinstance(rid, str) and ID_RE.match(rid)})


@contextlib.contextmanager
def share_lock():
  """Held while shares.json changes and while it's published, so overlapping commands can't
  lose each other's changes, and an older publisher can't bring back an unshared route."""
  ensure_dir(state_dir())
  with open(os.path.join(state_dir(), "shares.lock"), "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    yield


def save_shares(recipes):
  write_atomic(shares_path(), json.dumps({"recipes": sorted(set(recipes))}, indent=2) + "\n")


def share_error_path():
  return os.path.join(state_dir(), "share_error")


def published_path():
  return os.path.join(state_dir(), "shared.json")


def tailscale(*args, timeout=10):
  if not shutil.which("tailscale"):
    raise LocalError("Tailscale isn't installed")
  try:
    return subprocess.run(["tailscale", *args], capture_output=True, text=True, timeout=timeout)
  except subprocess.TimeoutExpired:
    raise LocalError(f"tailscale {args[0]} didn't answer")


def tailscale_status():
  """tailscaled's own view of the tailnet, or None without one."""
  try:
    result = tailscale("status", "--json", timeout=5)
  except LocalError:
    return None
  if result.returncode != 0:
    return None
  try:
    status = json.loads(result.stdout)
  except ValueError:
    return None
  return status if isinstance(status, dict) and status.get("BackendState") == "Running" else None


def node_dns(node):
  dns = str((node or {}).get("DNSName") or "").rstrip(".").lower()
  return dns if DNS_RE.match(dns) else ""


def self_dns(status=None):
  dns = node_dns((status or tailscale_status() or {}).get("Self"))
  if not dns:
    raise LocalError("This machine isn't on a tailnet with MagicDNS. Run: tailscale up")
  return dns


def serve_handlers(dns):
  """Our routes in the current serve config: path -> handler."""
  result = tailscale("serve", "status", "--json")
  if result.returncode != 0:
    raise LocalError(result.stderr.strip() or "Can't read the Tailscale Serve config")
  try:
    config = json.loads(result.stdout or "{}") or {}
  except ValueError:
    config = {}
  handlers = ((config.get("Web") or {}).get(f"{dns}:443") or {}).get("Handlers") or {}
  return {path: handler for path, handler in handlers.items() if path == SHARE_ROOT or path.startswith(SHARE_ROOT + "/")}


def serve_set(path, target):
  result = tailscale("serve", "--bg", "--yes", "--set-path", path, target)
  if result.returncode != 0:
    message = result.stderr.strip() or result.stdout.strip()
    if "operator" in message or "Unauthorized" in message or "Access denied" in message:
      raise LocalError(OPERATOR_HINT)
    raise LocalError(f"tailscale serve {path}: {message}")


def serve_off(path):
  result = tailscale("serve", "--yes", "--set-path", path, "off")
  if result.returncode != 0 and "not found" not in (result.stderr + result.stdout).lower():
    raise LocalError(f"tailscale serve {path} off: {(result.stderr or result.stdout).strip()}")


def shared_listing(record, catalog, profile, dns):
  """What other machines on the tailnet see: the shared models and their state."""
  try:
    gpu = choose_gpu(profile)
  except LocalError:
    gpu = None
  models = []
  for rid in shared_recipes():
    model = record["models"].get(rid)
    if not model:
      continue
    recipe = catalog.recipes.get(rid) or (load_model(rid) or {}).get("recipe") or {}
    measured = load_measurement(catalog, recipe, profile, gpu).get("long", {}) if gpu and recipe else {}
    manifest = load_model(rid) or {}
    models.append({
      "recipe": rid,
      "name": model["name"],
      "vendor": model["vendor"],
      # Dense or mixture of experts, which a peer's panel shows beside the
      # speeds instead of this machine's memory.
      "architecture": recipe.get("architecture", ""),
      "ctx": model["ctx"],
      "max_tokens": max_output_tokens(manifest) if manifest else DEFAULT_MAX_OUTPUT_TOKENS,
      "input": model_inputs(manifest) if manifest else ["text"],
      "unit": model["unit"],
      "error": model["error"],
      "fits": model["fits"],
      "since_ms": model["since_ms"],
      "expected_load_seconds": model["expected_load_seconds"],
      "expected_stop_seconds": model["expected_stop_seconds"],
      "observed": model["observed"],
      "decode_tok_s": measured.get("decode_tok_s"),
      "prefill_tok_s": measured.get("prefill_tok_s"),
    })
  return {
    "schema": NETWORK_SCHEMA,
    "host": dns.split(".")[0],
    "dns": dns,
    "ram_total_gib": record["ram_total_gib"],
    "models": models,
  }


def publish_shares(record, catalog, profile, force=False):
  """Bring the serve routes in line with shares.json and republish the
  listing when it changed. Errors land in share_error for the panel."""
  with share_lock():
    shares = shared_recipes()
    published = _read(published_path())
    if not shares and not published and not force:
      return
    try:
      dns = self_dns()
      have = serve_handlers(dns)
      want = {f"{SHARE_ROOT}/{rid}": f"http://127.0.0.1:{SHARE_PROXY_PORT}/{rid}" for rid in shares if rid in record["models"]}
      if want:
        ensure_share_proxy()
      for path, target in want.items():
        if (have.get(path) or {}).get("Proxy") != target:
          serve_set(path, target)
      for path in have:
        if path != LISTING_ROUTE and path not in want:
          serve_off(path)
      if want:
        text = json.dumps(shared_listing(record, catalog, profile, dns), sort_keys=True, separators=(",", ":"))
        if (have.get(LISTING_ROUTE) or {}).get("Text") != text:
          serve_set(LISTING_ROUTE, "text:" + text)
        write_atomic(published_path(), text + "\n")
      else:
        if LISTING_ROUTE in have:
          serve_off(LISTING_ROUTE)
        if os.path.exists(published_path()):
          os.unlink(published_path())
        stop_share_proxy()
      if os.path.exists(share_error_path()):
        os.unlink(share_error_path())
    except LocalError as error:
      write_atomic(share_error_path(), str(error) + "\n")
      if force:
        raise


def cmd_share(argv):
  parser = argparse.ArgumentParser(prog="omarchy agent local share", description="Share a model with the other machines on your tailnet")
  parser.add_argument("recipe", nargs="?")
  parser.add_argument("--off", action="store_true", help="stop sharing it")
  parser.add_argument("--export", action="store_true", help=argparse.SUPPRESS)
  args = parser.parse_args(argv)
  if args.export:
    # Read by `pair` on the other end of an SSH connection: the only way the
    # key leaves this machine.
    status = tailscale_status()
    dns = self_dns(status)
    print(json.dumps({"schema": NETWORK_SCHEMA, "host": dns.split(".")[0], "dns": dns, "key": ensure_key(), "shared": shared_recipes()}))
    return 0
  catalog = load_catalog()
  profile = machine_profile()
  if not args.recipe:
    shares = shared_recipes()
    if not shares:
      say("Nothing is shared. Share a model with: omarchy agent local share <recipe>")
    for rid in shares:
      say(f"{rid}: https://{self_dns()}{SHARE_ROOT}/{rid}/v1")
    return 0
  rid = args.recipe
  if not ID_RE.match(rid):
    raise LocalError("Not a recipe id")
  if not args.off:
    if not load_model(rid):
      raise LocalError(f"{rid} isn't set up. Run: omarchy agent local use {rid}")
    self_dns()
  with share_lock():
    shares = set(shared_recipes())
    if args.off:
      shares.discard(rid)
    else:
      shares.add(rid)
    save_shares(shares)
  try:
    record = write_status(catalog, profile, publish=False)
    publish_shares(record, catalog, profile, force=True)
  finally:
    write_status(catalog, profile, publish=False)
  if args.off:
    say(f"{rid} is no longer shared")
  else:
    say(f"Sharing {rid} on your tailnet at https://{self_dns()}{SHARE_ROOT}/{rid}/v1")
  return 0


# ---------------------------------------------------------------- share proxy
#
# `tailscale serve` hands a shared model's requests to one loopback proxy at
# /<recipe> instead of to the model's own port, so this machine can tell who
# is using it: serve puts the caller's tailnet address in X-Forwarded-For. The
# proxy forwards every request untouched, key check included, and streams the
# answer back as it comes. A machine whose request the model accepted counts
# as using it for ACTIVE_CLIENT_MS after its last one; clients.json holds the
# count per shared model, rewritten only when a count changes.

SHARE_PROXY_PORT = 11430
SHARE_PROXY_UNIT = "omarchy-agent-local-share.service"
ACTIVE_CLIENT_MS = 5 * 60 * 1000
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade"}


def clients_path():
  return os.path.join(state_dir(), "clients.json")


def share_proxy_unit_path():
  return os.path.join(home(), ".config", "systemd", "user", SHARE_PROXY_UNIT)


def ensure_share_proxy():
  """The proxy runs while anything is shared: written, reloaded and started on first need."""
  text = "\n".join([
    "[Unit]",
    "Description=Omarchy local AI: count who uses the shared models",
    "",
    "[Service]",
    "ExecStart=/usr/bin/env omarchy-agent-local share-proxy",
    "Restart=on-failure",
    "",
  ])
  path = share_proxy_unit_path()
  if _read(path) != text.strip():
    write_atomic(path, text)
    systemctl("daemon-reload")
  result = systemctl("start", SHARE_PROXY_UNIT)
  if result.returncode != 0:
    raise LocalError(f"Can't start the share proxy: {result.stderr.strip()}")


def stop_share_proxy():
  if os.path.exists(share_proxy_unit_path()):
    systemctl("stop", SHARE_PROXY_UNIT)


class ClientTracker:
  """The machines using each shared model, by the time of their last accepted request."""

  def __init__(self, path, window_ms=ACTIVE_CLIENT_MS, clock=lambda: int(time.time() * 1000)):
    self.path = path
    self.window_ms = window_ms
    self.clock = clock
    self.seen = {}
    self.written = None
    self.lock = threading.Condition()

  def _prune(self, now):
    for rid in list(self.seen):
      self.seen[rid] = {client: at for client, at in self.seen[rid].items() if now - at < self.window_ms}
      if not self.seen[rid]:
        del self.seen[rid]

  def _write(self):
    counts = {rid: len(clients) for rid, clients in sorted(self.seen.items())}
    if counts != self.written:
      write_atomic(self.path, json.dumps({"models": counts}, sort_keys=True) + "\n")
      self.written = counts

  def saw(self, rid, client):
    with self.lock:
      now = self.clock()
      self.seen.setdefault(rid, {})[client] = now
      self._prune(now)
      self._write()
      self.lock.notify()

  def expire(self):
    """Drop machines past the window, and say how long until the next one lapses."""
    with self.lock:
      now = self.clock()
      self._prune(now)
      self._write()
      times = [at for clients in self.seen.values() for at in clients.values()]
      return (min(times) + self.window_ms - now) / 1000 if times else None

  def run_expiry(self):
    # Held across the deadline check and the wait, so a machine saw() adds in
    # between can't slip past its notify. The condition's lock is reentrant.
    with self.lock:
      while True:
        wait = self.expire()
        self.lock.wait(timeout=None if wait is None else max(wait, 0.05))


def proxy_client(headers, address):
  forwarded = str(headers.get("X-Forwarded-For") or "").split(",")[0].strip()
  try:
    return str(ipaddress.ip_address(forwarded))
  except ValueError:
    return address


def make_share_proxy(tracker, ports=None, shares=None):
  import http.client
  import http.server
  ports = ports or load_ports
  shares = shares or shared_recipes

  class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
      pass

    def _body(self):
      """The request body as the model gets it: passed through in bounded
      blocks as it arrives, never held whole, so a caller without the key
      can't fill the proxy's memory before the model refuses it."""
      if "chunked" in str(self.headers.get("Transfer-Encoding") or "").lower():
        def chunks():
          while True:
            size = int(self.rfile.readline(1024).split(b";")[0].strip() or b"0", 16)
            if size == 0:
              while self.rfile.readline(1024).strip():
                pass
              return
            while size > 0:
              block = self.rfile.read(min(size, 65536))
              if not block:
                return
              size -= len(block)
              yield block
            self.rfile.readline(1024)
        return chunks(), True
      length = int(self.headers.get("Content-Length") or 0)
      if length <= 0:
        return None, False
      rfile = self.rfile

      class Remaining:
        def read(self, size=65536):
          nonlocal length
          block = rfile.read(min(size, length)) if length > 0 else b""
          length -= len(block)
          return block

      return Remaining(), False

    def _refuse(self, status):
      # The body is left unread, so the connection can't be reused for another request.
      self.close_connection = True
      self.send_response(status)
      self.send_header("Content-Length", "0")
      self.send_header("Connection", "close")
      self.end_headers()

    def _forward(self):
      rid, _, rest = self.path.lstrip("/").partition("/")
      port = ports().get(rid)
      if not ID_RE.match(rid) or rid not in shares() or not port:
        return self._refuse(404)
      body, chunked = self._body()
      headers = {name: value for name, value in self.headers.items() if name.lower() not in HOP_HEADERS and name.lower() != "host"}
      upstream = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
      try:
        upstream.request(self.command, "/" + rest, body=body, headers=headers, encode_chunked=chunked)
        answer = upstream.getresponse()
      except OSError:
        upstream.close()
        return self._refuse(502)
      # Only a request the model accepted counts: a wrong key or a bad path doesn't.
      if answer.status < 400:
        tracker.saw(rid, proxy_client(self.headers, self.client_address[0]))
      try:
        self.send_response(answer.status, answer.reason)
        for name, value in answer.getheaders():
          if name.lower() not in HOP_HEADERS and name.lower() != "content-length":
            self.send_header(name, value)
        length = answer.getheader("Content-Length")
        if self.command == "HEAD":
          self.send_header("Content-Length", length or "0")
          self.end_headers()
        elif length is not None:
          self.send_header("Content-Length", length)
          self.end_headers()
          while chunk := answer.read1(65536):
            self.wfile.write(chunk)
        else:
          # A streamed reply (server-sent events) goes out chunk by chunk as it arrives.
          self.send_header("Transfer-Encoding", "chunked")
          self.end_headers()
          while chunk := answer.read1(65536):
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.flush()
          self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
      finally:
        upstream.close()

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = _forward

  return Handler


def cmd_share_proxy(argv):
  import http.server
  argparse.ArgumentParser(prog="omarchy agent local share-proxy").parse_args(argv)
  tracker = ClientTracker(clients_path())
  tracker.expire()
  server = http.server.ThreadingHTTPServer(("127.0.0.1", SHARE_PROXY_PORT), make_share_proxy(tracker))
  server.daemon_threads = True
  threading.Thread(target=tracker.run_expiry, daemon=True).start()

  def stop(*_):
    raise SystemExit(0)

  signal.signal(signal.SIGTERM, stop)
  try:
    server.serve_forever()
  finally:
    # Nobody counts as using a model once nothing is listening.
    if os.path.exists(clients_path()):
      os.unlink(clients_path())
  return 0


# ---------------------------------------------------------------- tailnet clients


def remotes_dir():
  return os.path.join(config_dir(), "remotes")


def remote_key_path(host):
  return os.path.join(remotes_dir(), f"{host}.key")


def load_remotes():
  """Paired hosts: short host name -> {host, dns, ssh, paired_ms}."""
  remotes = {}
  if not os.path.isdir(remotes_dir()):
    return remotes
  for name in sorted(os.listdir(remotes_dir())):
    if not name.endswith(".json") or not HOST_RE.match(name[:-5]):
      continue
    try:
      with open(os.path.join(remotes_dir(), name)) as handle:
        remote = json.load(handle)
    except (OSError, ValueError):
      continue
    if isinstance(remote, dict) and remote.get("host") == name[:-5] and DNS_RE.match(str(remote.get("dns", ""))) and os.path.isfile(remote_key_path(name[:-5])):
      remotes[name[:-5]] = remote
  return remotes


def connected(remote, rid):
  """Whether this machine uses a paired host's model. A pairing made for one
  model at a time lists the models it connects; one made whole (`pair <host>`)
  connects all it shares but those it lists as disconnected."""
  if not remote:
    return False
  if isinstance(remote.get("models"), list):
    return rid in remote["models"]
  return rid not in (remote.get("disconnected") or [])


@contextlib.contextmanager
def remotes_lock():
  """Held across reading, changing and saving a pairing, so two pair or forget
  commands at once can't undo each other."""
  ensure_dir(remotes_dir(), 0o700)
  with open(os.path.join(remotes_dir(), ".lock"), "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    yield


def save_remote(remote):
  write_atomic(os.path.join(remotes_dir(), f"{remote['host']}.json"), json.dumps(remote, indent=2) + "\n", mode=0o600)


def unpair(host):
  for path in (os.path.join(remotes_dir(), f"{host}.json"), remote_key_path(host)):
    if os.path.exists(path):
      os.unlink(path)


def set_connected(remote, rid, on):
  """Connect or disconnect one model of a pairing, in place. Returns False
  when that leaves a one-at-a-time pairing with no model, which unpairs it."""
  if isinstance(remote.get("models"), list):
    remote["models"] = [m for m in remote["models"] if m != rid] + ([rid] if on else [])
    return bool(remote["models"])
  remote["disconnected"] = [m for m in remote.get("disconnected") or [] if m != rid] + ([] if on else [rid])
  return True


def network_path():
  return os.path.join(state_dir(), "network.json")


def load_network():
  try:
    with open(network_path()) as handle:
      network = json.load(handle)
  except (OSError, ValueError):
    return {"hosts": {}, "models": {}}
  if not isinstance(network, dict) or not isinstance(network.get("hosts"), dict) or not isinstance(network.get("models"), dict):
    return {"hosts": {}, "models": {}}
  return network


def _clean(value, limit=120):
  return CONTROL_RE.sub("", str(value or ""))[:limit]


def _number(value):
  try:
    number = float(value)
  except (TypeError, ValueError, OverflowError):
    return None
  return number if number == number and abs(number) < 1e15 else None


def clean_listing(listing, dns):
  """A peer's models.json as this machine will trust it: known shapes only,
  IDs it can put in a path, and text without control characters."""
  if not isinstance(listing, dict) or listing.get("schema") != NETWORK_SCHEMA or not isinstance(listing.get("models"), list):
    raise LocalError("Not an Omarchy model listing")
  models = []
  for model in listing["models"][:64]:
    if not isinstance(model, dict) or not ID_RE.match(str(model.get("recipe", ""))):
      continue
    observed = model.get("observed") if isinstance(model.get("observed"), dict) else {}
    inputs = [value for value in model.get("input") or ["text"] if value in INPUTS] if isinstance(model.get("input"), list) else ["text"]
    ctx = _number(model.get("ctx"))
    tokens = _number(model.get("max_tokens"))
    models.append({
      "recipe": model["recipe"],
      "name": _clean(model.get("name") or model["recipe"]),
      "vendor": model.get("vendor") if ID_RE.match(str(model.get("vendor", ""))) else "",
      "architecture": model.get("architecture") if model.get("architecture") in ARCHITECTURES else "",
      "ctx": int(ctx) if ctx and ctx > 0 else 0,
      "max_tokens": int(min(tokens, ctx or tokens)) if tokens and tokens > 0 else DEFAULT_MAX_OUTPUT_TOKENS,
      "input": inputs or ["text"],
      "unit": model.get("unit") if model.get("unit") in REMOTE_UNITS else "inactive",
      "error": _clean(model.get("error"), 300),
      "fits": model.get("fits") is not False,
      "since_ms": _number(model.get("since_ms")),
      "expected_load_seconds": _number(model.get("expected_load_seconds")),
      "expected_stop_seconds": _number(model.get("expected_stop_seconds")),
      "observed": {key: _number(observed.get(key)) for key in ("load_seconds", "ram_used_gib", "gpu_used_gib")} if observed else None,
      "decode_tok_s": _number(model.get("decode_tok_s")),
      "prefill_tok_s": _number(model.get("prefill_tok_s")),
    })
  return {"ram_total_gib": _number(listing.get("ram_total_gib")), "models": models}


def fetch_listing(dns):
  """(state, listing, error): online with its models, not_sharing, or unreachable.
  A peer's answer is untrusted: whatever it makes fetching, parsing or cleaning raise (a
  malformed response, nesting deep enough to recurse out) leaves that one peer
  unreachable, never the whole survey."""
  try:
    status, body = http_status(f"https://{dns}{LISTING_ROUTE}", timeout=LISTING_TIMEOUT)
    # Tailscale stops listening on 443 once a host's last serve route goes, so
    # an online host refusing the connection simply shares nothing.
    if status == 404 or (status == 0 and b"Connection refused" in body):
      return "not_sharing", None, ""
    if status != 200:
      return "unreachable", None, f"HTTP {status}" if status else body.decode(errors="replace")[:200]
    return "online", clean_listing(json.loads(body), dns), ""
  except Exception as error:
    return "unreachable", None, str(error)[:200]


def remote_record(host, dns, model, paired, ram_total_gib, last_seen_ms):
  rid = model["recipe"]
  return {
    **model,
    "id": f"{host}/{rid}",
    "host": host,
    "dns": dns,
    "paired": paired,
    "endpoint": f"https://{dns}{SHARE_ROOT}/{rid}/v1",
    "ram_total_gib": ram_total_gib,
    "last_seen_ms": last_seen_ms,
    # Set while the host can't be reached: why, for the line that says so.
    "offline": "",
  }


def survey_network(previous, now=None):
  """Every model shared on the tailnet, plus the ones seen before whose host
  is away now. `previous` is the last network.json."""
  now = now or int(time.time() * 1000)
  status = tailscale_status()
  remotes = load_remotes()
  hosts, models = {}, {}
  me = node_dns((status or {}).get("Self"))
  peers = {}
  for node in ((status or {}).get("Peer") or {}).values():
    dns = node_dns(node)
    if dns and dns != me and node.get("OS") == "linux":
      peers[dns.split(".")[0]] = (dns, node.get("Online") is True)
  for host, remote in remotes.items():
    peers.setdefault(host, (remote["dns"], False))
  for host, before in (previous.get("hosts") or {}).items():
    if isinstance(before, dict) and before.get("dns"):
      peers.setdefault(host, (before["dns"], False))
  online = {host: dns for host, (dns, up) in peers.items() if up}
  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    answers = dict(zip(online, pool.map(fetch_listing, online.values())))
  for host, (dns, up) in sorted(peers.items()):
    paired = host in remotes
    before = previous.get("hosts", {}).get(host) or {}
    state, listing, error = answers.get(host, ("offline", None, ""))
    if state == "online":
      hosts[host] = {"host": host, "dns": dns, "paired": paired, "state": "online", "error": "", "last_seen_ms": now, "ram_total_gib": listing["ram_total_gib"]}
      # Only what its host runs, or is loading, is on offer: a model it stops
      # leaves the list until it starts again.
      for model in listing["models"]:
        if model["unit"] not in ("active", "activating"):
          continue
        record = remote_record(host, dns, model, connected(remotes.get(host), model["recipe"]), listing["ram_total_gib"], now)
        models[record["id"]] = record
      continue
    if state == "not_sharing" and not paired:
      continue
    last_seen = before.get("last_seen_ms")
    if not paired and not (isinstance(last_seen, (int, float)) and now - last_seen < FORGET_UNPAIRED_MS):
      continue
    hosts[host] = {"host": host, "dns": dns, "paired": paired, "state": "not_sharing" if state == "not_sharing" else "offline" if state == "offline" else "unreachable",
                   "error": error, "last_seen_ms": last_seen, "ram_total_gib": before.get("ram_total_gib")}
    if state == "not_sharing":
      continue
    # Away: its models keep their lines, marked offline, as last seen.
    reason = "offline" if state == "offline" else "unreachable"
    for mid, model in previous.get("models", {}).items():
      if isinstance(model, dict) and model.get("host") == host and isinstance(model.get("recipe"), str) and ID_RE.match(model["recipe"]):
        models[mid] = {**model, "paired": connected(remotes.get(host), model["recipe"]), "offline": reason, "unit": "inactive", "since_ms": None}
  return {
    "schema": NETWORK_SCHEMA, "updated_ms": now, "tailnet": status is not None,
    # Whether `share` can work here: tailscale serve needs MagicDNS and HTTPS
    # certificates. The panel only offers sharing when it can.
    "can_share": bool(me) and bool((status or {}).get("CertDomains")),
    "self": me.split(".")[0] if me else "", "hosts": hosts, "models": models,
  }


def write_network(forget=None):
  """Survey the tailnet and rewrite network.json, which the panel watches."""
  ensure_dir(state_dir())
  with open(os.path.join(state_dir(), "network.lock"), "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    previous = load_network()
    if forget:
      previous["hosts"].pop(forget, None)
      previous["models"] = {mid: m for mid, m in previous["models"].items() if not (isinstance(m, dict) and m.get("host") == forget)}
    network = survey_network(previous)
    write_atomic(network_path(), json.dumps(network, indent=2) + "\n")
  # Pairing, forgetting and every survey pass through here, so pi's extension
  # follows paired hosts' models as status writes make it follow local ones.
  try:
    sync_pi_extension()
  except OSError:
    pass
  return network


def split_remote_id(value):
  host, _, rid = str(value or "").partition("/")
  if not HOST_RE.match(host) or not ID_RE.match(rid):
    raise LocalError("A network model is named host/recipe")
  return host, rid


def resolve_remote(mid, running_only=True):
  """A paired host's model, as its host lists it right now."""
  host, rid = split_remote_id(mid)
  remote = load_remotes().get(host)
  if not connected(remote, rid):
    raise LocalError(f"{host}/{rid} isn't paired. Run: omarchy agent local pair {host} {rid}")
  state, listing, error = fetch_listing(remote["dns"])
  if state != "online":
    raise LocalError(f"{host} doesn't share models right now" if state == "not_sharing" else f"{host} isn't reachable{': ' + error if error else ''}")
  for model in listing["models"]:
    if model["recipe"] == rid:
      if running_only and model["unit"] != "active":
        raise LocalError(f"{rid} isn't running on {host}")
      return remote_record(host, remote["dns"], model, True, listing["ram_total_gib"], int(time.time() * 1000))
  raise LocalError(f"{host} doesn't share {rid}")


def short_host(target):
  """An SSH target as people call the machine: no user, no tailnet domain.
  An address stays whole."""
  host = target.rsplit("@", 1)[-1]
  try:
    ipaddress.ip_address(host)
    return host
  except ValueError:
    return host.split(".")[0]


def ssh_command(remote, *argv, interactive=False):
  target = remote.get("ssh") or remote["dns"]
  if not SSH_TARGET_RE.match(target):
    raise LocalError("Not an SSH target")
  # A host met for the first time is trusted on sight, as the tailnet vouches
  # for who it is; one whose key has changed since is still refused.
  options = ["-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new"] + ([] if interactive else ["-o", "BatchMode=yes"])
  return ["ssh", *options, "--", target, shlex.join(["omarchy-agent-local", *argv])]


def cmd_pair(argv):
  parser = argparse.ArgumentParser(prog="omarchy agent local pair", description="Use the models another machine on your tailnet shares")
  parser.add_argument("host", help="its tailnet name, or an SSH host that reaches it")
  parser.add_argument("recipe", nargs="?", help="connect just this model; without one, every model it shares")
  args = parser.parse_args(argv)
  target = args.host
  if not SSH_TARGET_RE.match(target):
    raise LocalError("Not a host name")
  if args.recipe is not None and not ID_RE.match(args.recipe):
    raise LocalError("Not a recipe name")
  # A host already paired has given its key: connecting another of its models
  # needs no SSH.
  if args.recipe:
    with remotes_lock():
      remote = next((r for r in load_remotes().values() if target in (r["host"], r["dns"], r.get("ssh"))), None)
      if remote:
        set_connected(remote, args.recipe, True)
        save_remote(remote)
    if remote:
      write_network()
      say(f"Paired with {remote['host']}/{args.recipe}")
      return 0
  say(f"Pairing with {target} over SSH")
  # ssh exits 255 both when nothing answers and when sshd turns down every key
  # offered, so its stderr tells the two apart. On a terminal a key's
  # passphrase can be asked for there; without one (the agents panel pairs this
  # way) nothing can answer a prompt, so ssh asks none.
  result = subprocess.run(ssh_command({"ssh": target, "dns": ""}, "share", "--export", interactive=sys.stdin.isatty()), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
  sys.stderr.write(result.stderr)
  # The agents panel shows an error's first line on the model's line, so that
  # line names the machine short and fits there; the next says what to do.
  name = short_host(target)
  if result.returncode == 255 and "Permission denied" in result.stderr:
    raise LocalError(f"{name} refused this machine's SSH keys\nStart the SSH agent that holds your key (like 1Password's), or authorize a key on {target} in Setup > Security")
  if result.returncode == 255 and "Host key verification failed" in result.stderr:
    # known_hosts keys a host by its name or address, never the login.
    raise LocalError(f"{name}'s SSH host key has changed\nIf it was reinstalled, forget its old key with: ssh-keygen -R {target.rsplit('@', 1)[-1]}")
  if result.returncode == 255:
    raise LocalError(f"Can't reach {name} over SSH\nTurn on SSHD on {target} in Setup > Security")
  if result.returncode != 0:
    raise LocalError(f"Can't run omarchy-agent-local on {name}")
  try:
    export = json.loads(result.stdout.strip().splitlines()[-1])
    host, dns, key = export["host"], export["dns"], export["key"]
  except (ValueError, KeyError, IndexError, TypeError):
    raise LocalError(f"{name} didn't send its sharing details")
  if not HOST_RE.match(str(host)) or not DNS_RE.match(str(dns)) or not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", key):
    raise LocalError(f"{name} sent an invalid host name or key")
  with remotes_lock():
    write_atomic(remote_key_path(host), key, mode=0o600)
    # The same host reached by another name keeps the models it connects.
    remote = load_remotes().get(host) or {"models": []}
    remote.update(host=host, dns=dns, ssh=target, paired_ms=int(time.time() * 1000))
    if args.recipe:
      set_connected(remote, args.recipe, True)
    else:
      remote.pop("models", None)
      remote.pop("disconnected", None)
    save_remote(remote)
  network = write_network()
  shared = [m for m in network["models"].values() if m["host"] == host and m["paired"]]
  if shared:
    say(f"Paired with {host}: " + ", ".join(m["recipe"] + ("" if m["offline"] else f" ({m['unit']})") for m in shared))
  else:
    say(f"Paired with {host}. It shares no models yet; on {host}, run: omarchy agent local share <recipe>")
  return 0


def cmd_forget(argv):
  parser = argparse.ArgumentParser(prog="omarchy agent local forget", description="Unpair a host, or disconnect one of its models with host/recipe")
  parser.add_argument("host", help="host, or host/recipe for just that model")
  args = parser.parse_args(argv)
  host = args.host
  if "/" in host:
    host, rid = split_remote_id(host)
    with remotes_lock():
      remote = load_remotes().get(host)
      if not connected(remote, rid):
        say(f"{host}/{rid} isn't paired")
        return 0
      # The last model disconnected unpairs the host, dropping its key.
      if set_connected(remote, rid, False):
        save_remote(remote)
      else:
        unpair(host)
    write_network()
    say(f"Disconnected {host}/{rid}")
    return 0
  if not HOST_RE.match(host):
    raise LocalError("Not a host name")
  with remotes_lock():
    unpair(host)
  write_network(forget=host)
  say(f"Forgot {host}")
  return 0


def cmd_network(argv):
  parser = argparse.ArgumentParser(prog="omarchy agent local network", description="List the models shared on your tailnet")
  parser.add_argument("--json", action="store_true")
  parser.add_argument("--write", action="store_true", help=argparse.SUPPRESS)
  args = parser.parse_args(argv)
  network = write_network()
  if args.write:
    return 0
  if args.json:
    print(json.dumps(network, indent=2))
    return 0
  if not network["tailnet"]:
    say("This machine isn't on a tailnet. Run: tailscale up")
    return 0
  if not network["models"]:
    say("No machine on your tailnet shares a model. On one that runs a model: omarchy agent local share <recipe>")
  for mid, model in sorted(network["models"].items()):
    state = model["offline"] or {"active": "running", "activating": "loading", "deactivating": "stopping"}.get(model["unit"], model["unit"])
    pairing = "" if model["paired"] else f" · pair with: omarchy agent local pair {model['host']} {model['recipe']}"
    say(f"{mid} ({model['name']}): {state} · context {model['ctx']}{pairing}")
  return 0


# The provider local models are listed under, in pi's own extension and in the
# launch overlays, so the two are one provider rather than two to pick from. A
# paired host's models get their own, suffixed with the host.
OVERLAY_PROVIDER = "omarchy-local"


def overlay_target(rid=None):
  """The running model an overlay points at: a local recipe, or host/recipe
  on a paired machine of the tailnet."""
  if rid and "/" in rid:
    return remote_target(resolve_remote(rid))
  return local_target(resolve_model(rid, running_only=True))


def remote_target(model):
  """A paired host's model, from its listing or network.json, as an overlay target."""
  return {
    "provider": f"{OVERLAY_PROVIDER}-{model['host']}",
    "provider_name": f"Omarchy model on {model['host']}",
    "base_url": model["endpoint"],
    "key_path": remote_key_path(model["host"]),
    "model": model["recipe"],
    "name": model["name"],
    "ctx": model["ctx"],
    # A host that doesn't publish its cap yet gets the old default.
    "max_tokens": model.get("max_tokens") or DEFAULT_MAX_OUTPUT_TOKENS,
    "input": model["input"],
    "file": f"{model['host']}--{model['recipe']}",
  }


def local_target(manifest):
  rid = manifest["recipe_id"]
  return {
    "provider": OVERLAY_PROVIDER,
    "provider_name": "Omarchy local model",
    "base_url": endpoint(rid),
    "key_path": key_path(),
    "model": rid,
    "name": manifest["recipe"].get("name", rid),
    "ctx": manifest["variant"]["ctx"],
    "max_tokens": max_output_tokens(manifest),
    "input": model_inputs(manifest),
    "file": rid,
  }


def pi_model(target):
  return {
    "type": "chat",
    "id": target["model"],
    "name": target["name"],
    "baseUrl": target["base_url"],
    "reasoning": True,
    "input": target["input"],
    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    "contextWindow": target["ctx"],
    "maxTokens": target["max_tokens"],
  }


def pi_provider_source(header, targets):
  """A pi extension registering these models, each at its own endpoint, under
  their providers: this machine's, and one per paired host with its key."""
  providers = {}
  for target in targets:
    providers.setdefault(target["provider"], {
      "name": target["provider_name"],
      "apiKey": "!cat -- " + shlex.quote(target["key_path"]),
      "api": "openai-completions",
      "models": [],
    })["models"].append(pi_model(target))
  registrations = "".join(f"  pi.registerProvider({json.dumps(name)}, {json.dumps(provider, indent=2)});\n" for name, provider in providers.items())
  return f"// {header}\nexport default function (pi) {{\n{registrations}}}\n"


def local_targets():
  """Every model set up here as an overlay target. A manifest that can't be
  read as one is left out rather than costing the others their line."""
  targets = []
  for rid in known_models():
    try:
      targets.append(local_target(load_model(rid)))
    except (KeyError, TypeError, AttributeError):
      pass
  return targets


def remote_targets():
  """Every paired host's connected model this machine has seen, as overlay
  targets, whether or not its host answers right now: like a stopped local
  model, it keeps its place in the picker."""
  remotes = load_remotes()
  targets = []
  for model in load_network()["models"].values():
    try:
      if model.get("paired") and model["host"] in remotes:
        targets.append(remote_target(model))
    except (KeyError, TypeError, AttributeError):
      pass
  return sorted(targets, key=lambda target: (target["provider"], target["model"]))


def pi_extension_path():
  return os.path.join(home(), ".pi", "agent", "extensions", "omarchy-local.ts")


def sync_pi_extension():
  """Keep pi's own extension listing every model set up here and every paired
  host's, so plain `pi`, its /model picker and a default of
  omarchy-local/<recipe> or omarchy-local-<host>/<recipe> reach each one at
  its current endpoint with the current key, with nothing written by hand.
  Only where pi has made its folder; it goes when the last model does."""
  if not os.path.isdir(os.path.join(home(), ".pi", "agent")):
    return
  # Status and network writes both sync, so collecting and writing hold a
  # lock of their own: an older list can't land over a newer one.
  ensure_dir(state_dir())
  with open(os.path.join(state_dir(), "pi.lock"), "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    path = pi_extension_path()
    targets = local_targets() + remote_targets()
    if not targets:
      if os.path.exists(path):
        os.unlink(path)
      return
    source = pi_provider_source("Generated by omarchy-agent-local for every model set up here or paired on the tailnet; rewritten as they change.", targets)
    if _read(path) != source.strip():
      write_atomic(path, source)


def pi_overlay(rid=None):
  """A pi extension for a launch on a running model. It lists the rest of its
  provider's models too, as pi's own extension does, so loading it over that
  one leaves the /model picker whole; the launch picks this one with --model."""
  target = overlay_target(rid)
  siblings = local_targets() if target["provider"] == OVERLAY_PROVIDER else remote_targets()
  targets = [target] + [other for other in siblings if other["provider"] == target["provider"] and other["model"] != target["model"]]
  source = pi_provider_source("Generated by omarchy-agent-local overlay for `omarchy agent --local`; rewritten on every launch.", targets)
  # One file per model: launches of two models close together must not
  # swap each other's overlay before their agents read it.
  path = os.path.join(config_dir(), "overlays", f"pi-{target['file']}.ts")
  write_atomic(path, source)
  return path, target["provider"], target["model"]


DEFAULT_MAX_OUTPUT_TOKENS = 16384


def max_output_tokens(manifest):
  """The reply cap the server enforces, so agents never ask for less (or more) than it allows."""
  args = manifest["recipe"].get("server", {}).get("args", []) + manifest["variant"].get("args", [])
  value = None
  for index, arg in enumerate(args):
    if arg == "--max-tokens" and index + 1 < len(args):
      value = args[index + 1]
    elif arg.startswith("--max-tokens="):
      value = arg.split("=", 1)[1]
  try:
    tokens = int(value)
  except (TypeError, ValueError):
    return DEFAULT_MAX_OUTPUT_TOKENS
  return min(tokens, manifest["variant"]["ctx"]) if tokens > 0 else DEFAULT_MAX_OUTPUT_TOKENS


def model_inputs(manifest):
  return list(manifest["recipe"].get("input", ["text"]))


def opencode_overlay(rid=None):
  target = overlay_target(rid)
  overlay = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      target["provider"]: {
        "npm": "@ai-sdk/openai-compatible",
        "name": target["provider_name"],
        "options": {"baseURL": target["base_url"], "apiKey": "{file:" + target["key_path"] + "}"},
        "models": {target["model"]: {
          "name": target["name"],
          "limit": {"context": target["ctx"], "output": target["max_tokens"]},
          "attachment": "image" in target["input"],
          "modalities": {"input": target["input"], "output": ["text"]},
        }},
      }
    },
  }
  path = os.path.join(config_dir(), "overlays", f"opencode-{target['file']}.json")
  write_atomic(path, json.dumps(overlay, indent=2) + "\n")
  return path, target["provider"], target["model"]


def qwen_overlay(rid=None):
  """What Qwen Code is launched with: it takes an OpenAI-compatible endpoint
  from flags and its key from the environment, so this is the endpoint, the
  model and where the key is, for the launcher to read. No key in it."""
  target = overlay_target(rid)
  overlay = {
    "baseUrl": target["base_url"],
    "model": target["model"],
    "name": target["name"],
    "keyFile": target["key_path"],
    "contextWindow": target["ctx"],
    "maxTokens": target["max_tokens"],
  }
  path = os.path.join(config_dir(), "overlays", f"qwen-{target['file']}.json")
  write_atomic(path, json.dumps(overlay, indent=2) + "\n")
  return path, target["provider"], target["model"]


OVERLAYS = {"pi": lambda rid: pi_overlay(rid), "opencode": lambda rid: opencode_overlay(rid), "qwen": lambda rid: qwen_overlay(rid)}


def cmd_overlay(argv):
  parser = argparse.ArgumentParser(prog="omarchy agent local overlay")
  parser.add_argument("agent", choices=tuple(OVERLAYS))
  parser.add_argument("recipe", nargs="?", help="the running model, or host/recipe on a paired machine (default: the last one started here)")
  args = parser.parse_args(argv)
  for line in OVERLAYS[args.agent](args.recipe):
    print(line)
  return 0


def cmd_logs(argv):
  parser = argparse.ArgumentParser(prog="omarchy agent local logs")
  parser.add_argument("recipe", nargs="?", help="the model (default: the last one started)")
  parser.add_argument("-n", "--lines", type=int, default=200)
  args = parser.parse_args(argv)
  rid = args.recipe or last_model_id()
  if not rid or not ID_RE.match(rid):
    raise LocalError("Name the model: omarchy agent local logs <recipe>")
  # A first setup or start that failed keeps its reason, and its peer, in its installing record.
  installing = load_installing(rid) or {}
  error = _read(error_path(rid)) or installing.get("error", "")
  if error:
    say(f"Last error: {error}")
    say("")
  # Its setup's log while it's being set up or failed to be, then the model's own.
  for unit, what in ((install_unit(rid), "setup"), (unit_name(rid), "model")):
    out = subprocess.run(["journalctl", "--user", "-q", "-u", unit, "-n", str(args.lines), "--no-pager"], capture_output=True, text=True).stdout.strip()
    if out:
      say(f"== {rid}: {what} log on this machine ({unit})")
      say(out)
      say("")
  try:
    setup = pair_manifest(rid).get("pair")
  except LocalError:
    setup = installing.get("pair")
  if setup:
    say(f"== {rid}: rank 1 on {setup['host']}")
    result = pair_ssh(setup["peer"], f"podman logs --tail {int(args.lines)} {shlex.quote(worker_name(rid))} 2>&1", check=False, timeout=30)
    say((result.stdout or result.stderr).strip() or "(no log there)")
  return 0


COMMANDS = {
  "use": cmd_use,
  "list": cmd_list,
  "status": cmd_status,
  "start": cmd_start,
  "stop": cmd_stop,
  "remove": cmd_remove,
  "share": cmd_share,
  "network": cmd_network,
  "pair": cmd_pair,
  "forget": cmd_forget,
  "logs": cmd_logs,
  # Plumbing for the panel, omarchy agent, omarchy update and catalog
  # maintainers; they run but aren't listed.
  "profile": cmd_profile,
  "refresh": cmd_refresh,
  "validate": cmd_validate,
  "overlay": cmd_overlay,
  "link": cmd_link,
  "link-worker": cmd_link_worker,
  "share-proxy": cmd_share_proxy,
}

SUMMARIES = {
  "use": "Set up and run a model, best for this machine or a named recipe",
  "list": "List recipes and which fit this machine",
  "status": "Show the models that are set up, and whether a better one is recommended",
  "start": "Start a model that's set up, the last one started by default",
  "stop": "Stop a model, or every running one",
  "remove": "Remove a model's weights, manifests and unused image",
  "share": "Share a model with the other machines on your tailnet, or stop with --off",
  "network": "List the models shared on your tailnet, including hosts that are offline",
  "pair": "Pair with a machine over SSH to use the models it shares",
  "forget": "Unpair a machine and drop its models",
  "logs": "Show a model's logs, and its other half's on a paired Spark",
  "link": "Show the Spark cabled to this one; 'link setup <peer>' prepares both",
}


def usage():
  lines = ["Usage: omarchy agent local <command> [args]", ""]
  lines += [f"  {name:<10} {summary}" for name, summary in SUMMARIES.items()]
  lines += ["", "Run omarchy-agent-local <command> --help for its arguments."]
  return "\n".join(lines)


def main(argv):
  if not argv or argv[0] in ("-h", "--help", "help"):
    print(usage())
    return 0 if argv else 1
  command, argv = argv[0], argv[1:]
  if command not in COMMANDS:
    print(f"Unknown command: {command}\n\n{usage()}", file=sys.stderr)
    return 1
  try:
    return COMMANDS[command](argv)
  except LocalError as error:
    # On a terminal the error ends output from ssh, podman and the like, so it
    # gets a line of space and a red label to stand apart from it.
    if sys.stderr.isatty():
      print(f"\n\033[1;31mError:\033[0m {error}", file=sys.stderr)
    else:
      print(f"Error: {error}", file=sys.stderr)
    return 1
  except KeyboardInterrupt:
    return 130
