# Local model engine contract

`omarchy-local-ai` installs the engine and catalog together; Omarchy routes
`omarchy agent local <command>` to it. The engine reads the catalog beside
`agent_local.py`, following the entry script's symlink. Developers can select a
checkout catalog with `OMARCHY_LOCAL_AI_CATALOG`.

## Trust boundary

Catalog publication grants authority to execute the pinned runtime image and its
commands inside the engine's container boundary. Image review is a maintainer
responsibility; validation cannot establish that an image is benign. Third-party
runtimes reference a review of their exact image digest and disclose the publisher
and license when installed.

Every model runs in a rootless Podman container with an internal network without
external egress, a read-only root and model mount, all capabilities dropped,
`NoNewPrivileges`, and only allowlisted devices and run options. Its API is
published on loopback. The host downloads weights and pulls images before launch.
The server receives no agent workspace or host home mount.

Runtimes whose ranks span two machines (a `[pair]` table, `run.network = "fabric"`)
are the one exception to the internal network: both ranks share their host's network
namespace to reach each other over the direct ConnectX-7 cable (a TCP rendezvous on
the runtime's `master_port`, NCCL and RoCE on the cable's RDMA devices), with the
API bound to loopback by the engine through the runtime's reserved `listen` flags.
Every other restriction holds on both machines: read-only root and model mounts, all
capabilities dropped, `NoNewPrivileges`, and only the GPU and `/dev/infiniband`.

The host key is mode 0600 and passed as a Podman secret, never embedded in a unit.
Runtimes taking a key argument must explicitly opt in and use the engine's Bash
substitution inside the container; installation warns that local process listings
can expose it. Post-start checks require `/v1/models` to reject an unauthenticated
request, accept the key and advertise the recipe ID. A failed activation stops
the new unit and restores that recipe's previous unit and running state.

Quadlet generation uses typed fields and escapes systemd substitutions, quotes
and backslashes. Recipe arguments cannot replace runtime-reserved options or
mapped standard inputs. Catalog strings and paths are validated before use.

## Catalog formats

The real records are the examples and source of values:

| Path | Rules |
| --- | --- |
| [catalog.toml](../catalog.toml) | The catalog schema, 1. |
| [recipes/](../recipes/) | Name, summary, runtime, pinned artifacts, measured hardware, context variants, sampling and optional server arguments. Vendor names the panel logo; architecture is `dense` or `moe`; `input` is `["text"]` (the default) or `["text", "image"]`, which the agent overlays declare so agents can send screenshots. |
| [runtimes/](../runtimes/) | Digest-pinned image, tier, port, command, standard input mappings, reserved options and nonempty `health.probe` argv. A paired runtime adds `listen` (its bind flags), `@PORT@` in its probe and a `[pair]` table: `nodes = 2`, `master_port`, the `state` path where each rank's own folder is mounted writable for the server's caches, and `env` for both ranks (any names but the link's `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA` and `NCCL_IB_GID_INDEX`, which the engine sets), whose values may hold `@RANK@`, `@WORLD@`, `@MASTER_ADDR@`, `@MASTER_PORT@` and `@RDMA_DEVICES@`. Both ranks run the runtime's command with the same arguments. |
| [reviews/](../reviews/) | Third-party image digest, publisher, reviewer, date, license URL/hash and checked properties. |
| [measurements/](../measurements/) | Hardware evidence, artifact identities, context, memory, load time and short, long and code-edit prompt timings. |
| [best.toml](../best.toml) | Unique classes requiring `gpu` and one of `ram_class` or `vram_class`, with an ordered list of measured recipes. |

IDs contain lowercase alphanumeric components separated by single dots or hyphens.
Artifact repositories are `owner/name`, revisions are 40-hex commits, and files,
destinations and role paths must be relative without empty, `.` or `..` components.
File globs expand against the pinned Hugging Face tree. LFS files verify by SHA256;
small files verify by Git blob identity. Destinations cannot collide and model/draft
roles must name selected files, or `"."` for the whole directory an artifact with a
`dest` lands in, for servers that load a checkpoint directory. Variants require positive context and memory peaks.
A variant may add `args` after the recipe's server arguments, for settings sized to its memory such as streams
and caches; they follow the same rules and may not repeat a flag the server arguments already set.

Runtime inputs map exactly one argument or environment variable per supported
standard input; the model may instead be `{ positional = true }`, placed right after
the command for servers that take it bare (`serve <dir>`). Run options support a `devices` list (`render`, `kfd`,
`cdi-nvidia`), scalar `group_add = "keep-groups"`, scalar `memlock = "unlimited"`,
numeric `shm_size_gib` greater than zero and at most 16, `infiniband` among the
devices and scalar `network = "fabric"` for paired runtimes. Other run options are
rejected. `validate <catalog-dir>` checks the records and referenced evidence;
`--online` also checks upstream artifacts and licenses.

## Lifecycle

`use best` matches the GPU class and its memory class, tries the class's recipes
in order, checks runtime compatibility and measured hardware qualification, and
chooses the largest fitting context, and at the same context the variant with the larger memory peak that fits.
A GPU class is its family, except that the N1x (`nvidia-n1x`) shares the GB10's (`nvidia-gb10`): the same
chip, so what either can run depends only on memory. Measurements keep the family they ran on. `use <recipe>` supports `--ctx`, `--gpu`,
`--seed` and explicit `--unqualified`; static memory limits still apply. Admission
also checks current free RAM and GPU memory and reserves 8 GiB for the desktop.
A discrete NVIDIA GPU (an RTX 3090, say) is its own family, `nvidia-rtx3090`, named from `nvidia-smi`, and
its memory class is its VRAM rounded up to 8, 12, 16, 24, 32, 48, 80 or 96 GB: a `[[hardware]]` or `[[class]]`
entry for it says `vram_class = 24`, not `ram_class`. System RAM then only has to hold the recipe's
`peak_ram_gib`, which admission checks beside the VRAM. Unified-memory GPUs (GB10, N1x, Strix) keep `ram_class`.
Use, start and removal are serialized. Several models can run at once on distinct,
stable loopback ports, beginning at 11435.

Downloads share verified blobs. Each setup materializes an immutable
`models/<recipe>/<fileset>` directory, always mounted read-only at `/models`.
A new file set leaves the previous unit's files intact until activation succeeds.
Only then is the current manifest replaced and obsolete file sets pruned.

A first setup is listed before it has a manifest: its installing record
(`installing/<recipe>.json`) shows in `status.json` as a model whose unit is
`installing`, with what it has fetched of the weights and runtime image, and
how fast, rewritten twice a second. Once activation begins, the pending row
of an activating model replaces it, and the record goes when the manifest is
saved. A failed setup keeps its record, with the reason as `error`, until
`remove` dismisses it. `use --detach` resolves and checks the recipe, then
runs the setup in the transient user unit `omarchy-agent-local-use-<recipe>`,
so it outlives the panel or terminal that asked for it. `remove` of a setup in
progress cancels it, stopping that unit or the terminal's `use` process, and
drops what it fetched.

If a new setup cannot start because memory is busy, its manifest and unit are
saved with `unchecked = true`. Its first plain `start [recipe]` performs the
post-start authentication and model-ID checks; failure stops it and retains the
flag for a retry. Autostart cannot be enabled before that first checked start.
`start` defaults to the last started model; `stop [recipe]` defaults to all running
models.

`remove <recipe>` stops it, removes its unit and file sets, drops blobs unused by
other recipes and removes its image only if no surviving current manifest uses
it. The manifest remains until removal completes so interrupted removal can be
retried. A removal marker keeps the panel in `removing` through stop hooks and is
cleared even when interrupted. Load/stop observations and seconds per GiB removed
predict subsequent progress bars. Image pulls report aggregate progress on one
line per update.

## Paired machines

A DGX Spark cabled to another through their ConnectX-7 ports is a pair once
`link setup <peer>` has joined them: it records the other machine, a GB10 of the
same RAM class, by its address on the cable and the SSH user it was given, in
`pair.json`, which every other command reads. A pair adds the GPU family `nvidia-gb10-pair`
(`nodes = 2`, the smaller machine's memory per node), whose `best.toml` class is tried
before the machine's own; single-node recipes keep running on the machine's own GPU.

The peer needs key-based SSH from this machine, rootless Podman that runs
`nvidia.com/gpu=all`, unlimited locked memory for its user's systemd manager, and
rsync. Nothing of the engine is installed there. `link setup <peer>` makes all of
that true, safe to run again: it creates and copies an SSH key (asking for the peer's
password once), gives both machines unlimited locked memory (for logins in limits.d, for the
user's systemd manager in user.conf.d, applied to a running manager at once), installs Podman and rsync where missing, keeps a CDI
0.6 copy of NVIDIA's spec for Podman before 5 (Ubuntu's DGX OS) through a drop-in on
nvidia-cdi-refresh, lets the peer's user linger, and gives cable ports without an
address one /24 per PCIe link (10.99.N.1 here, .2 there, MTU 9000) through
NetworkManager (finishing a half-addressed cable on a re-run), and where ufw is active
lets in, on each rail port the ranks use, only the other end's address (the ranks'
rendezvous and NCCL sockets), ahead of Omarchy's SSH rate limit. `use` of a paired recipe:

1. materializes the file set here, as for any recipe, with rank 0's state folder
   `models/<recipe>/<fileset>.rank0` beside it;
2. copies the image to the peer (`podman save | podman load`, by image ID) unless it
   has it, and the file set with rsync over the cable;
3. makes rank 1's state folder there, and drops older file sets.

The head unit runs rank 0. Its `ExecStartPre` (`link-worker start`) reads the link as
it is now (each side's netdev, address, RoCE devices and the RoCE v2 GID index of its
IPv4 address), checks the peer's free memory, writes the unit's environment file and
starts rank 1 on the peer as `omarchy-agent-local-model-<recipe>-rank1`; its
`ExecStopPost` stops it. Start, stop and the panel treat the pair as one model.
Admission checks free memory on both machines. `remove` also removes the peer's
container, files and, when no other paired recipe uses it, its image; while the peer
can't be reached it fails before deleting anything, so it can run again, unless
given `--local-only` for a peer that's gone for good.

## State and panel API

Paths are relative to the user's home. JSON caches are atomically replaced.

| Path | Purpose |
| --- | --- |
| `.local/state/omarchy/agent-local/models/<recipe>.json` | One current resolved manifest: schema, recipe/runtime IDs and definitions, profile, GPU, variant, image, files, roles, required fileset, creation time, optional observations and unchecked flag. Its existence means installed. |
| `.local/state/omarchy/agent-local/status.json` | Current model state and recommendation; shape below. |
| `.local/state/omarchy/agent-local/list.json` | Array of recipe cards; shape below. |
| `.local/state/omarchy/agent-local/ports.json`, `last` | Stable port assignments and last started recipe. |
| `.local/state/omarchy/agent-local/loads/`, `stops/`, `removal_rate` | Measured progress timings. |
| `.local/state/omarchy/agent-local/errors/`, `removing/` | Per-model error and removal prediction. |
| `.local/state/omarchy/agent-local/installing/<recipe>.json` | A first setup in progress or failed: name, vendor, phase, bytes, speed, start, error and the `use` process id. |
| `.local/state/omarchy/agent-local/pair.json`, `pair/<recipe>.env` | The detected peer (address, host name, home, RAM); a paired unit's link environment, written at each start. |
| `.local/state/omarchy/agent-local/lock`, `status.lock` | Lifecycle admission and status snapshot locks. |
| `.local/share/omarchy/agent-local/blobs/`, `models/` | Shared blob store and immutable file sets. |
| `.config/containers/systemd/omarchy-agent-local-model-<recipe>.container` | Generated unit; shared `omarchy-agent-local.network` defines the internal network. |
| `.config/omarchy/agent-local/key`, `overlays/` | Private API key and per-recipe client overlays. |
| `.pi/agent/extensions/omarchy-local.ts` | Pi's standing providers for every model set up or paired, where pi is installed. |

`status.json` is `{models: {<recipe>: <model>}, last, ram_total_gib, best,
link}`. `last` and `best` may be null. `link` is what the
ConnectX-7 cable shows: `{state: "none"}`, `{state: "linked", host}`, or, for another
Spark answering on the cable's IPv6 link-local all-nodes address before `link setup`,
`{state: "available", address, host}` (`host` from mDNS, may be null). `refresh` and
`link` probe it; `link setup` with no peer named sets up the available one. The
package installs [system/90-omarchy-local-ai-link](../system/90-omarchy-local-ai-link)
to `/etc/NetworkManager/dispatcher.d/`: when a ConnectX-7 port comes up it runs
`link notify` for each logged-in user, which announces a newly available Spark once,
with a notification whose action opens `link setup` in a terminal. Each model contains:

- Identity: `recipe`, `name`, `vendor`, `runtime`, `image`, `ctx`.
- Connection: `port`, `endpoint` (including `/v1`).
- State: `unit`, `sub_state`, `result`, `error`, `fits`, `fit_reason`.
- Progress: `since_ms`, `expected_load_seconds`, `expected_stop_seconds`,
  `expected_remove_seconds`, and nullable `observed` (`load_seconds`,
  `ram_used_gib`, `gpu_used_gib`) and `ram_total_gib`, the memory of the
  machines the model runs on. A paired model's `ram_used_gib` and
  `ram_total_gib` cover both machines. Unavailable timings are null.
- Setup, on `installing` models only: `install` with `phase` (`waiting` for
  another setup to finish, `downloading`, or `failed`), `done_bytes` and
  `total_bytes` over the weights still to fetch and the runtime image, and
  `rate_bytes_s` averaged over the last few seconds. Their connection fields
  are empty.

Panel unit states are `active`, `activating`, `inactive`, `failed`, `deactivating`,
`removing` and `installing`. The panel uses `fits` to disable play and `error` to explain a
failure: the reason the start recorded, else the server's own last `error:` line,
else systemd's result. A failure links to `logs <recipe>`, which prints that
error, the setup and model journals and, for a paired model, rank 1's container
log from the peer. Transition timestamps are Unix milliseconds. Pending activation uses
the current manifest's observations until the new setup has been measured.

`list.json` rows contain `id`, `name`, `summary`, `runtime`, `tier`, `publisher`,
`vendor`, `architecture`, `installed`, `available`, `reason`, `best`, `active`,
`memory_gib`, `context`, `decode_tok_s` and `prefill_tok_s`. Unavailable variants
have null memory/context; missing measurements have null speeds.

The panel calls plain `start <recipe>`, `stop <recipe>`, `remove <recipe>`,
`use --detach <recipe>`, `refresh --quiet` on opening the add view, and
`status --write` when refreshing admission state, and opens `logs <recipe>` in a
terminal from a failure's error line. Units call `status --write`
with `--started <recipe>` or `--stopped <recipe>` from their hooks.
`list --json` and `status --json` print the corresponding shapes; `profile`
prints the machine profile. `refresh` rebuilds both caches using the installed
catalog without changing running models.

## Client integration

`overlay pi|opencode|qwen [recipe]` requires a running model and prints three lines:
overlay path, provider ID (`omarchy-local`) and recipe/model ID. With no recipe it
prefers the last started running model, then another running model. Omarchy's
`omarchy agent --local` and `--local-model <recipe>` consume this output.

Pi receives a generated TypeScript extension registering the provider; OpenCode
receives a JSON provider configuration; Qwen Code, which takes an
OpenAI-compatible endpoint from flags and its key from the environment,
receives a JSON file with `baseUrl`, `model`, `name`, `keyFile`,
`contextWindow` and `maxTokens` for the launcher to read. Both use the model's stable loopback
endpoint, context and recipe ID, with a file reference or command to read the
key. Overlays are per recipe so simultaneous launches cannot overwrite each
other's configuration. The generated files contain no literal key.

Pi also gets a standing extension, `~/.pi/agent/extensions/omarchy-local.ts`,
which pi loads on every start. It registers the same `omarchy-local` provider
with every model set up here, each at its own endpoint, and an
`omarchy-local-<host>` provider with its stored key for each paired host's
connected models in `network.json`, offline ones included. So plain `pi`, its
model picker and a default of `omarchy-local/<recipe>` or
`omarchy-local-<host>/<recipe>` work with no hand-written `models.json`. Every
status and network write rewrites it when the list has changed, under
`pi.lock`, and removes it with the last model. It is only written where
`~/.pi/agent` already exists. A pi overlay lists the rest of its provider's
models after the launched one, so loading it over the standing extension
leaves the picker whole.

## Tailnet sharing

`share <recipe>` lets the other machines of the user's tailnet use a model, and `share <recipe> --off` stops. The container, its loopback port and its key check are unchanged: `tailscale serve` proxies `https://<host>.<tailnet>.ts.net/local-ai/<recipe>`, reachable only on the tailnet, to the share proxy at `127.0.0.1:11430/<recipe>`, which forwards each request untouched to the model's loopback port and streams the answer back, and the model still refuses a request without the key. Beside the models, a `text:` route at `/local-ai/models.json` lists the shared models with their state, context, reply cap, inputs, memory and measured speeds. It never contains the key. Every status write republishes the listing when it changed, so remote machines see starts and stops as they happen. Changing Serve needs the user to be Tailscale's operator (`sudo tailscale set --operator=$USER`); a refusal is recorded as `share_error` in `status.json` with that fix. Removing a model unshares it.

The share proxy (`share-proxy`, run as the `omarchy-agent-local-share.service` user unit while anything is shared, and stopped once nothing is) counts who uses each shared model. A request the model accepted counts the machine that sent it, by the tailnet address `tailscale serve` puts in `X-Forwarded-For`, for five minutes after its last one; a refused request counts nothing. `.local/state/omarchy/agent-local/clients.json` is `{models: {<recipe>: <machines>}}`, rewritten only when a count changes, as a machine starts or lapses, and removed when the proxy stops. The agents panel watches it to say how many machines use a shared model.

`pair <host>` runs `omarchy-agent-local share --export` on the host over SSH, which prints its tailnet name and key, and stores them in `.config/omarchy/agent-local/remotes/<host>.{json,key}` (mode 0600). SSH is the only way a key leaves its machine. `forget <host>` deletes both and drops the host's models.

`network` asks the local tailscaled for its online Linux peers, fetches each one's `models.json` in parallel (4s timeout) and writes `network.json`: `{schema, updated_ms, tailnet, self, hosts, models}`. Models are keyed `host/recipe` and carry the local model fields plus `host`, `dns`, `paired`, `endpoint`, `ram_total_gib`, `last_seen_ms` and `offline`. A host that drops off keeps its models, with `offline` set to `offline` (Tailscale says so) or `unreachable` (online but not answering) and `last_seen_ms` from its last answer. Paired hosts are kept until forgotten; unpaired ones for a week. An online host serving no listing shares nothing and its models go. Listings are untrusted input: only known states, recipe IDs and inputs survive, and text loses control characters.

`overlay pi|opencode <host>/<recipe>` writes the same overlays for a paired host's running model, with provider `omarchy-local-<host>`, the tailnet endpoint and the stored key file, so `omarchy agent --local-model <host>/<recipe>` launches against it.

The agents panel watches `network.json` and runs `network --write` when it opens and every 20 seconds while it stays open. Each network model gets a line beside the local ones: pair while unpaired, start/stop and launch once paired, and an offline line with the last-seen time while its host is away.

Maintainers run [tools/bench](../tools/bench) from a checkout against a running
model. It imports the checkout engine and uses [bench/context.txt](../bench/context.txt)
for the long prompt. Measurements remain catalog evidence; the benchmark tool
and prompt corpus do not need to be installed.
