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
| [catalog.toml](../catalog.toml) | Schema 1 and the release version. |
| [recipes/](../recipes/) | Name, summary, runtime, pinned artifacts, measured hardware, context variants, sampling and optional server arguments. Vendor names the panel logo; architecture is `dense` or `moe`; `input` is `["text"]` (the default) or `["text", "image"]`, which the agent overlays declare so agents can send screenshots. |
| [runtimes/](../runtimes/) | Digest-pinned image, tier, port, command, standard input mappings, reserved options and nonempty `health.probe` argv. |
| [reviews/](../reviews/) | Third-party image digest, publisher, reviewer, date, license URL/hash and checked properties. |
| [measurements/](../measurements/) | Hardware evidence, artifact identities, context, memory, load time and short, long and code-edit prompt timings. |
| [best.toml](../best.toml) | Unique classes requiring both `gpu` and `ram_class`, with an ordered list of measured recipes. |

IDs contain lowercase alphanumeric components separated by single dots or hyphens.
Artifact repositories are `owner/name`, revisions are 40-hex commits, and files,
destinations and role paths must be relative without empty, `.` or `..` components.
File globs expand against the pinned Hugging Face tree. LFS files verify by SHA256;
small files verify by Git blob identity. Destinations cannot collide and model/draft
roles must name selected files, or `"."` for the whole directory an artifact with a
`dest` lands in, for servers that load a checkpoint directory. Variants require positive context and memory peaks.

Runtime inputs map exactly one argument or environment variable per supported
standard input; the model may instead be `{ positional = true }`, placed right after
the command for servers that take it bare (`serve <dir>`). Run options support a `devices` list (`render`, `kfd`,
`cdi-nvidia`), scalar `group_add = "keep-groups"`, scalar `memlock = "unlimited"`,
and numeric `shm_size_gib` greater than zero and at most 16. Other run options are
rejected. `validate <catalog-dir>` checks the records and referenced evidence;
`--online` also checks upstream artifacts and licenses.

## Lifecycle

`use best` matches the exact GPU family and RAM class, tries the class's recipes
in order, checks runtime compatibility and measured hardware qualification, and
chooses the largest fitting context. `use <recipe>` supports `--ctx`, `--gpu`,
`--seed` and explicit `--unqualified`; static memory limits still apply. Admission
also checks current free RAM and GPU memory and reserves 8 GiB for the desktop.
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

## State and panel API

Paths are relative to the user's home. JSON caches are atomically replaced.

| Path | Purpose |
| --- | --- |
| `.local/state/omarchy/agent-local/models/<recipe>.json` | One current resolved manifest: schema, recipe/runtime IDs and definitions, catalog version, profile, GPU, variant, image, files, roles, required fileset, creation time, optional observations and unchecked flag. Its existence means installed. |
| `.local/state/omarchy/agent-local/status.json` | Current model state and recommendation; shape below. |
| `.local/state/omarchy/agent-local/list.json` | Array of recipe cards; shape below. |
| `.local/state/omarchy/agent-local/ports.json`, `last` | Stable port assignments and last started recipe. |
| `.local/state/omarchy/agent-local/loads/`, `stops/`, `removal_rate` | Measured progress timings. |
| `.local/state/omarchy/agent-local/errors/`, `removing/` | Per-model error and removal prediction. |
| `.local/state/omarchy/agent-local/installing/<recipe>.json` | A first setup in progress or failed: name, vendor, phase, bytes, speed, start, error and the `use` process id. |
| `.local/state/omarchy/agent-local/lock`, `status.lock` | Lifecycle admission and status snapshot locks. |
| `.local/share/omarchy/agent-local/blobs/`, `models/` | Shared blob store and immutable file sets. |
| `.config/containers/systemd/omarchy-agent-local-model-<recipe>.container` | Generated unit; shared `omarchy-agent-local.network` defines the internal network. |
| `.config/omarchy/agent-local/key`, `overlays/` | Private API key and per-recipe client overlays. |

`status.json` is `{models: {<recipe>: <model>}, last, ram_total_gib,
catalog_version, best}`. `last` and `best` may be null. Each model contains:

- Identity: `recipe`, `name`, `vendor`, `runtime`, `image`, `ctx`, `catalog_version`.
- Connection: `port`, `endpoint` (including `/v1`).
- State: `unit`, `sub_state`, `result`, `error`, `fits`, `fit_reason`.
- Progress: `since_ms`, `expected_load_seconds`, `expected_stop_seconds`,
  `expected_remove_seconds`, and nullable `observed` (`load_seconds`,
  `ram_used_gib`, `gpu_used_gib`). Unavailable timings are null.
- Setup, on `installing` models only: `install` with `phase` (`waiting` for
  another setup to finish, `downloading`, or `failed`), `done_bytes` and
  `total_bytes` over the weights still to fetch and the runtime image, and
  `rate_bytes_s` averaged over the last few seconds. Their connection fields
  are empty.

Panel unit states are `active`, `activating`, `inactive`, `failed`, `deactivating`,
`removing` and `installing`. The panel uses `fits` to disable play and `error` to explain a
failure. Transition timestamps are Unix milliseconds. Pending activation uses
the current manifest's observations until the new setup has been measured.

`list.json` rows contain `id`, `name`, `summary`, `runtime`, `tier`, `publisher`,
`vendor`, `architecture`, `installed`, `available`, `reason`, `best`, `active`,
`memory_gib`, `context`, `decode_tok_s` and `prefill_tok_s`. Unavailable variants
have null memory/context; missing measurements have null speeds.

The panel calls plain `start <recipe>`, `stop <recipe>`, `remove <recipe>`,
`use --detach <recipe>`, `refresh --quiet` on opening the add view, and
`status --write` when refreshing admission state. Units call `status --write`
with `--started <recipe>` or `--stopped <recipe>` from their hooks.
`list --json` and `status --json` print the corresponding shapes; `profile`
prints the machine profile. `refresh` rebuilds both caches using the installed
catalog without changing running models.

## Client integration

`overlay pi|opencode [recipe]` requires a running model and prints three lines:
overlay path, provider ID (`llamacpp`) and recipe/model ID. With no recipe it
prefers the last started running model, then another running model. Omarchy's
`omarchy agent --local` and `--local-model <recipe>` consume this output.

Pi receives a generated TypeScript extension registering the provider; OpenCode
receives a JSON provider configuration. Both use the model's stable loopback
endpoint, context and recipe ID, with a file reference or command to read the
key. Overlays are per recipe so simultaneous launches cannot overwrite each
other's configuration. The generated files contain no literal key.

Maintainers run [tools/bench](../tools/bench) from a checkout against a running
model. It imports the checkout engine and uses [bench/context.txt](../bench/context.txt)
for the long prompt. Measurements remain catalog evidence; the benchmark tool
and prompt corpus do not need to be installed.
