# Host config

## `sidecars.json` — named-sidecar registry

Privileged docker sidecars the host runs on a plugin's behalf via the
`run_sidecar` IPC command (#750). The registry is **data, not source**
(#850): adding or changing a sidecar is an edit to this file on the
host — no TypeScript change, no redeploy of the orchestrator image
(the file is read on every `run_sidecar` call).

- **Location:** `config/sidecars.json` under the orchestrator's working
  directory — `/app/config/` inside the container, where docker-compose
  bind-mounts this directory read-only (override with the
  `SIDECARS_CONFIG_PATH` env var). The real file is **gitignored** — it
  carries machine-specific host paths. Copy `sidecars.example.json`
  next to it to get started.
- **Missing file** = empty registry: every `run_sidecar` call returns an
  "Unknown sidecar" error naming this path. **Invalid file** (bad JSON,
  wrong shape, unexpandable `${VAR}`) = every call returns the specific
  validation error. It never silently degrades to an empty registry.

### Security model

This file is TRUSTED HOST CONFIG — the same trust boundary the entries
had when they were TypeScript. The image, bind mounts, and flag
allowlist come from here, **never** from the IPC payload: a compromised
container can only name a registered sidecar and append flags from that
entry's `allowedFlags`. Review edits to this file like code.

### Entry shape

```json
{
  "<sidecar-name>": {
    "image": "image:tag",
    "mounts": ["hostPath:containerPath", "..."],
    "baseArgs": ["--json"],
    "allowedFlags": ["--dry-run"],
    "timeoutMs": 600000,
    "maxBuffer": 10485760
  }
}
```

- `image` (required) — the docker image to `docker run --rm`. Pin
  registry-pulled images to a versioned tag or digest per
  `coding-policy: dependency-management`. A **locally built** image
  (e.g. `audible-backup:latest`, built on the NAS from its own
  Dockerfile and never pulled from a registry) may keep `latest`: the
  tag only ever changes when the operator explicitly rebuilds it, so
  the renewal mechanism IS the manual local rebuild — there is no
  scanner-trackable pin and no remote that could move underneath it.
  Note which case applies when adding an entry.
- `mounts` (default `[]`) — `host:container` bind specs. `${VAR}`
  placeholders expand from the host environment at load time, plus the
  built-in `${HOST_PROJECT_PARENT}` (the directory containing the
  project root). An unset variable is a config error, never an empty
  string.
- `baseArgs` (default `[]`) — args always passed to the image.
- `allowedFlags` (default `[]`) — the only flags an IPC caller may
  append.
- `timeoutMs` (default `600000`) / `maxBuffer` (default `10485760`) —
  execution limits.
