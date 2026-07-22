# Core vs Domain — the private-fork boundary

NanoClaw's pitch is "one process and a handful of files" — and for the
**platform core** that stays true. But a lived-in private fork accumulates a
second kind of code: *personal domain* — integrations, policies, and data
shapes that belong to one operator's life, not to the platform. Epic #844
drew the line and built the seams. This document is the map, so future-you
at 2am doesn't dump domain into `src/` because the docs said "just change
the code."

## The two kinds of code

**Platform core** is what every fork needs to route a message into a
container and back: the orchestrator loop, channels, group queue,
container runner, DB substrate and migration machinery, task scheduler,
router, and the *registries* below. Core is generic: it knows there ARE
spawn gates; it doesn't know about flights.

**Personal domain** is everything whose reason to exist is one operator's
setup: the Hubitat smart-home listener, the flight-assist spawn gate and
location sink, the audible-backup sidecar, persona files, tile content.
Domain code plugs into core through a registry; core never imports it
except via `src/host-plugins/index.ts` registration.

## The seams (all landed in #844)

| Registry | Core surface | Domain plugs in via |
|---|---|---|
| IPC commands | `src/ipc-registry.ts` — `registerIpcHandler()` | own handler module registered at startup |
| Lifecycle | `src/host-lifecycle.ts` — `registerStartupHook()` / `registerShutdownHook()` | e.g. `src/host-plugins/hubitat/` (#848) |
| Spawn gates | `src/spawn-gates.ts` — `registerSpawnGate()` | e.g. `src/host-plugins/flight-assist-spawn-gate.ts` (#846) |
| Location sinks | `src/location-sinks.ts` — `registerLocationSink()` | e.g. `src/host-plugins/flight-assist-location-sink.ts` (#849) |
| Sidecars | `src/sidecar-runner.ts` + gitignored `config/sidecars.json` (#850) | a JSON entry, no TS change |
| Host plugins | `src/host-plugins/index.ts` — `registerHostPlugins()` | one registration line per plugin |

Reference extraction: the Hubitat plugin (`src/host-plugins/hubitat/`)
registers lifecycle hooks that **dynamically import** the listener, so an
unconfigured fork loads none of it. Core keeps only the
`smart_home_events` schema migration (see the comment at the CREATE TABLE
in `src/db.ts`).

## Where does this go? (checklist)

Work through in order; first match wins.

1. **Agent behavior, prompts, workflows, recurring agent tasks** → a
   **tile skill** (nanoclaw-core / trusted / untrusted / admin), delivered
   via the tessl registry.
2. **Per-chat capability on top of the trust-tier baseline** → an
   **overlay tile** (`containerConfig.additionalTiles`).
3. **Host-side code that only some forks want** — a device listener, a
   personal API integration, a policy gate, an outbound data sink → a
   **host plugin** under `src/host-plugins/`, wired through a registry
   above. Config-gate the registration (unset env = register nothing) and
   lazy-load heavy modules via dynamic import.
4. **A privileged host process with its own image/mounts** → a **sidecar**
   entry in `config/sidecars.json` (name-keyed, payload never supplies
   image or mounts).
5. **Everything every fork needs** — routing, isolation, scheduling,
   persistence machinery → **core** (`src/`). If you're about to name a
   personal service, a device, or a skill inside a core file, stop and
   re-run this checklist.

Single-skill *state* (tables, JSON shapes, migrations) has its own policy
— see [STATE-OWNERSHIP.md](STATE-OWNERSHIP.md).

## Private-fork rules

- **Host plugins are private-by-default.** They exist in
  `jbaruch/nanoclaw`, not in the public fork; `scripts/sync-to-public.sh`
  excludes `src/host-plugins/` wholesale in its rsync exclude list. (The
  script's older in-file regex scrubs have drifted from the post-#844
  layout — reconciling or retiring them is #869.) A plugin only goes
  public deliberately, as a generic example with the personal specifics
  removed.
- **Public sync must not grow personal domain.** When reviewing a
  `sync/*` PR on the public repo, `src/host-plugins/` content beyond the
  scrub-listed examples is a red flag, not a feature.
- **Core PRs stay domain-free.** A core change that imports from
  `src/host-plugins/` (other than `registerHostPlugins()` in startup
  wiring) is a boundary violation — move the code behind a registry
  instead.

## The "handful of files" caveat

README's "one process and a handful of files" describes the platform
core a fresh fork starts from. A lived-in private fork is core **plus**
its domain plugins — still one process, but "just change the code" means:
change *domain* code in a plugin or tile, change *core* only when the
platform itself needs a new seam.
