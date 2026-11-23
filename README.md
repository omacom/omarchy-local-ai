# omarchy-local-ai

The local-model engine, recipe catalog and internals reference for Omarchy. The Arch package `omarchy-local-ai` installs this tree to `/usr/share/omarchy-local-ai` and symlinks `/usr/bin/omarchy-local-ai` to its executable entry script. Omarchy's thin `omarchy-models` wrapper exposes it as `omarchy models <command>`.

The engine resolves recipes, downloads pinned weights, and manages rootless Podman model servers with an internal network, read-only root, dropped capabilities, allowlisted devices and authenticated loopback endpoints. The catalog ships with the engine; it is never fetched at runtime. See [engine contract](docs/engine.md) for the security boundary, lifecycle and state formats.

## Run from a checkout

Python 3.11 or newer is required. On Omarchy, run:

```bash
./omarchy-local-ai list
./omarchy-local-ai use best text
./omarchy-local-ai validate .
./tests/engine-test.sh
```

Running models also requires Podman, crun and the Omarchy hardware helpers. Tests use a temporary home and do not start containers. If installed, the real Podman Quadlet generator checks generated units in dry-run mode.

The engine reads the catalog beside `agent_local.py`, following the installed entry script's symlink. Set `OMARCHY_LOCAL_AI_CATALOG` to another catalog directory for development. State and model data live in `~/.local/state/omarchy/agent-local` and `~/.local/share/omarchy/agent-local`.

## Catalog layout

- `catalog.toml` — the catalog schema.
- `best.toml` — ordered recommendations per hardware class: text recipes for the agents, and image models beside them.
- `recipes/<id>.toml` — kind (text or image), runtime, pinned Hugging Face artifacts, measured context variants, memory peaks and sampling defaults.
- `runtimes/<id>.toml` — image pinned by digest, the GPUs it works on (`gpus`), input mappings, health probe, post-start check (`api`) and allowlisted run options.
- `reviews/` — license and behavior reviews for third-party image digests.
- `measurements/` — speed and memory on a machine class, which `best.toml` recommends by.
- `images/<runtime>/Containerfile` — how Omacom builds a runtime's image, for runtimes whose image is built here. Checkout-only, like `tools/`.
- `images/<id>/` — the Containerfile and keys that build a runtime image Omacom doesn't publish, so its digest can be rebuilt and checked.
- `tools/bench` and `bench/context.txt` — checkout-only benchmark script and long prompt.

## Add a recipe

1. Add a recipe and any runtime/review it requires, or add a measured hardware class to an existing recipe. Every machine whose GPU the runtime lists in `gpus` and that has the memory for a variant is offered the recipe; `best.toml` decides which machines it is recommended to, and needs a measurement for each class it recommends it for.
2. Run `./omarchy-local-ai validate .`, then `./omarchy-local-ai use <id>` on the target hardware.
3. Run `./tools/bench <id> --output measurements/<id>-<gpu>-<ram>.toml` and reference the measurement in the recipe.
4. Submit the recipe, review and measurement together. CI validates the catalog online and runs engine tests. Merging catalog changes grants software publication authority within the engine's security boundary.

## Build a runtime image

A runtime whose image is built here has its Containerfile in `images/<image>/`: `llamacpp`, `sdcpp`, `sdcpp-cuda`, `acestep`, `acestep-cuda`, `tensorfold`, `tensorfold-dsv41`, `tensorfold-glm53`, `tensorfold-nemotron-bolt`, `llamacpp-hy4` and `inkling2-exl3`. The base image and upstream commits are pinned in the Containerfile, but packages come from the distribution at build time, so a rebuild gives a new digest even with nothing changed. To update one, for example to a newer llama.cpp:

1. Change the pinned `COMMIT` and its `BUILD` tag, or the base digest, in `images/llamacpp/Containerfile`, in a pull request. CI builds every image a pull request changes, to check it.
2. Merging it publishes the image: `.github/workflows/images.yml` pushes `ghcr.io/omacom/omarchy-local-ai/llamacpp:<version>-<run>` and puts the digest to pin in the run's summary. Run the workflow by hand, naming the image, to rebuild without a change, for the distribution's newer packages.
3. Pin `ghcr.io/omacom/omarchy-local-ai/llamacpp@<digest>` as `image` in `runtimes/llamacpp.toml`.
4. Review the new digest in `reviews/llamacpp-<first 12 hex of the digest>.toml`, rerunning the checks the last review lists (license, read-only root, no network, key, stop on SIGTERM), and point the runtime's `review` at it. Keep the old review: it records what was reviewed then.
5. Set the image up with `./omarchy-local-ai use <recipe>` and rebench every recipe on that runtime with `./tools/bench`, since a measurement records the image it was measured on.
6. Run `./omarchy-local-ai validate .` and `./tests/engine-test.sh`, and submit the runtime, review and measurements together.

The `llamacpp`, `sdcpp` and `acestep` builds also fail if the image lacks Mesa's AMD or Intel driver, and the `sdcpp` and `acestep` ones if the server uses AVX-512, which Intel laptops and older AMD machines don't have.

To build one locally instead: `podman build --format docker -t llamacpp images/llamacpp`. The GB10 and N1x images build on arm64 runners and the others on x86, as in the workflow's table; a new folder needs a line there.

## Releases and packaging

Validate and test, then tag the release `vX.Y.Z`; nothing in the repo is bumped. The `omarchy-pkgs` recipe builds the Arch package from that tag, installing the engine, catalog and docs together. Packaging is maintained in `omarchy-pkgs`; this repository is its source tree.

`bench/` and `tools/` do not need to ship in the Arch package. Measurement records still ship: recipe validation and the panel use them.

After package updates, `omarchy-models refresh --quiet` rewrites the panel's list and status caches. It does not download a catalog or change running models.
