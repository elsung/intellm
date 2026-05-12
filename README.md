# intellm

Interactive + scriptable launcher for local LLM inference via `llama.cpp`, with support for multiple per-backend builds side-by-side, named configs, KV-cache precision selection, and persistent prompt caching.

Built originally for an Intel Arc Pro B70 (Battlemage) rig. The launcher itself is backend-agnostic — it discovers builds from a registry file, so adding CUDA / ROCm / Metal builds is a one-line edit.

## What it does

- `intellm` (no args) → interactive walkthrough: build → mode (chat/server/bench) → model → context window → KV precision → prompt cache → launch
- `intellm --config <name>` → run a saved named config from `configs/`
- `intellm --build sycl --mode chat --model X.gguf --ctx 32768 --kv q8_0` → fully scripted
- `intellm --list models|builds|configs` (or `--list-json …`) → discovery for other tools

It auto-detects each model's trained context length from the GGUF header and offers it as the top option in the picker. It bakes in build-specific environment setup (e.g. sourcing Intel oneAPI before SYCL runs) so you can't accidentally run the SYCL build without its env activated.

## Layout

```
INTELLM_HOME/
├── intellm                       # the launcher
├── configs/
│   ├── default.conf              # auto-loaded when no --config given
│   ├── example.conf              # commented reference
│   └── <your-named-configs>.conf
├── llama.cpp/
│   ├── builds.conf               # backend registry (shortname, display, bin dir, env, extra args)
│   ├── env-sycl.sh               # sourced before running the SYCL build
│   ├── gguf-ctx.py               # reads max ctx from a GGUF header (no deps)
│   ├── src/                      # llama.cpp source clone (gitignored — bring your own)
│   └── build-*/                  # build directories, one per backend (gitignored)
└── *.gguf                        # models live at the root (or symlink in)
```

## Setup (from scratch)

Tested on Ubuntu 24.04, kernel 6.17, Intel Arc B70 (Battlemage). Adapt for other backends.

### 1. System packages

```bash
sudo apt install -y build-essential cmake git pkg-config libcurl4-openssl-dev \
                    libvulkan-dev vulkan-tools mesa-vulkan-drivers glslc \
                    spirv-headers glslang-tools libomp-dev ccache ninja-build
```

For Intel SYCL (oneAPI): see `llama.cpp/env-sycl.sh` for the env setup; install `intel-basekit` from Intel's apt repo and the `kobuk-team/intel-graphics` PPA for the Battlemage compute-runtime.

### 2. Clone this repo + llama.cpp

```bash
git clone https://github.com/<you>/intellm.git ~/intellm
cd ~/intellm
git clone --depth 1 https://github.com/ggml-org/llama.cpp llama.cpp/src
```

### 3. Build the backends you want

```bash
# Vulkan (works on Intel/AMD/NVIDIA Vulkan-capable GPUs)
cmake -S llama.cpp/src -B llama.cpp/build-vulkan-intel \
  -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=ON
cmake --build llama.cpp/build-vulkan-intel --config Release -j$(nproc)

# SYCL (Intel GPU, oneAPI 2026.0+)
source /opt/intel/oneapi/setvars.sh
cmake -S llama.cpp/src -B llama.cpp/build-sycl-intel -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
  -DGGML_SYCL_DEVICE_ARCH=bmg_g31 \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build llama.cpp/build-sycl-intel --config Release -j$(nproc)
```

`builds.conf` already registers both. Add a new backend by appending a line.

### 4. Put `intellm` on your PATH (optional)

```bash
mkdir -p ~/.local/bin
ln -s $(pwd)/intellm ~/.local/bin/intellm
# Case-insensitive aliases (Linux is case-sensitive, so these are explicit):
for name in intelLM inteLLM IntelLM INTELLM; do
  ln -s $(pwd)/intellm ~/.local/bin/$name
done
```

Ensure `~/.local/bin` is on your `$PATH` (Ubuntu default puts it there if the directory exists at login).

### 5. Drop in a model and go

```bash
# Either drop *.gguf files into INTELLM_HOME directly, or use the HF auto-download:
intellm --build vulkan --mode chat --model hf:bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M
```

## Configs

A config is a key=value shell file in `configs/`. The full surface:

```bash
build=sycl                # shortname from builds.conf
mode=chat                 # chat | server | bench
model=Qwen3.6-27B-Q6_K.gguf
ctx=32768
kv=q8_0                   # f16 | q8_0 | q4_0
host=0.0.0.0              # server only
port=8080                 # server only
slots=4                   # server only — parallel slots
prompt_cache=coding-agent # chat only — filename under KVCACHE_DIR
```

Flags always override config values. `configs/default.conf` is auto-loaded when `--config` is absent; `intellm --interactive` bypasses it.

## Environment

| Var | Default | Purpose |
|---|---|---|
| `INTELLM_HOME` | script location (autodetected) | Project root |
| `MODELS_DIR` | `$INTELLM_HOME` | Where to discover `*.gguf` files |
| `KVCACHE_DIR` | `/mnt/optane/kv-cache` | Persistent prompt-cache files |

## Optane / fast-storage notes

The launcher works fine on any storage, but if you have an Optane SSD or other low-latency block device, the design assumes:

- Models live on the fast volume (mmap'd, near-zero cold load).
- Swap is on the same volume (kernel can spill anonymous KV pages into ~10 µs latency).
- `KVCACHE_DIR` points there too (prompt-cache snapshots load instantly).

A reasonable setup:
```bash
sudo mkdir -p /mnt/optane/LLMs /mnt/optane/kv-cache
sudo chown -R $USER:$USER /mnt/optane/{LLMs,kv-cache}
ln -s /mnt/optane/LLMs ~/LLMs   # or use ~/LLMs as INTELLM_HOME directly
```

Recommended `sysctl` for an Optane swap rig:
```
vm.swappiness = 100
vm.vfs_cache_pressure = 50
```

## Backend notes

### Intel Arc B70 (Battlemage)

- Build SYCL with `-DGGML_SYCL_DEVICE_ARCH=bmg_g31` (not `bmg_g21`, which is the smaller B580 die).
- `env-sycl.sh` wraps the source of `setvars.sh` in `set +eu` because Intel's setup script isn't strict-mode-clean.
- `GGML_SYCL_DISABLE_OPT=1` was needed in mid-2025 against output corruption (issue #21893) — verified fixed in llama.cpp ≥ bbeb89d (2026-05). Re-enable if you see garbled output after a future pull.
- On MoE/hybrid (Mamba) models, Vulkan currently outperforms SYCL on prompt processing; on dense transformers SYCL is the better choice. Run `intellm --mode bench` on both and compare for your specific model.

## Research & decisions

This project is investigation-driven. Notes live in [`docs/`](./docs/):

- [`docs/benchmarks.md`](./docs/benchmarks.md) — measured perf numbers (always with hardware + commit hash + command line)
- [`docs/decisions.md`](./docs/decisions.md) — append-only log of design choices and why
- [`docs/research/`](./docs/research/) — deep-dive investigations: agent findings, paper notes, prototyping plans

Currently in flight: KV cache offloading on Intel GPU (extending usable context window beyond VRAM). See [`docs/research/2026-05-12-synthesis-kv-offload-plan.md`](./docs/research/2026-05-12-synthesis-kv-offload-plan.md) for the current plan.

## License

MIT. Bring your own llama.cpp (MIT) and models (their own licenses).
