"""
stdlib-unittest contract test for `scripts/deploy.sh`'s
agent-container-selection predicate (the filter pipeline in step 5,
which delegates to `scripts/exclude-infra-containers.sh`).

Run with:
  python3 -m unittest scripts.test_deploy_agent_grep

The deploy script's force-kill loop is container-name-driven
(`docker ps --format '{{.Names}}' | ...`). A regression in the
predicate either
(a) silently leaves the orchestrator container (`nanoclaw`, no
trailing dash) in the agent pool — risking an exit-137 kill on the
orchestrator itself — or
(b) excludes real agent containers, leaving stale-tile sessions
running after deploy.

We test the predicate by extracting the literal pipeline from
deploy.sh and running it against a fixed set of container names,
asserting the post-filter output exactly. No Docker dependency; the
test feeds names through `bash -c` with the same pipeline the script
uses (which invokes the filter script from the repo root).
"""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path

_DEPLOY_SH = Path(__file__).resolve().parent / "deploy.sh"

# Anchor the search to step 5's specific predicate so we don't pick
# up unrelated `docker ps | grep` lines elsewhere in the script
# (e.g. spawn-collision detection).
_AGENT_PREDICATE_RE = re.compile(
    r"AGENTS=\$\(docker ps --format '\{\{\.Names\}\}'\s*\|\s*(.+?)\|\|\s*true\)",
    re.DOTALL,
)


def _extract_predicate() -> str:
    """Return the grep pipeline (without the leading `docker ps`)."""
    text = _DEPLOY_SH.read_text()
    match = _AGENT_PREDICATE_RE.search(text)
    if not match:
        raise RuntimeError(
            "Could not locate agent-selection predicate in deploy.sh — "
            "the regex anchor may be stale; check step 5's line in "
            "deploy.sh and update _AGENT_PREDICATE_RE."
        )
    return match.group(1).strip().rstrip("|").strip()


def _run_predicate(names: list[str]) -> list[str]:
    """Pipe `names` through the predicate; return the surviving set."""
    predicate = _extract_predicate()
    proc = subprocess.run(
        ["bash", "-c", f"{predicate} || true"],
        input="\n".join(names) + "\n",
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"predicate pipeline exited {proc.returncode}: stderr={proc.stderr!r}"
        )
    out = proc.stdout.strip()
    return out.split("\n") if out else []


class AgentPredicateTests(unittest.TestCase):
    def test_excludes_orchestrator(self) -> None:
        # `nanoclaw` is the orchestrator itself — its restart is
        # handled in deploy.sh step 7, not via the agent-close path.
        self.assertEqual(_run_predicate(["nanoclaw"]), [])

    def test_includes_real_agent_containers(self) -> None:
        # Agent containers are named
        # `nanoclaw-<safeName><sessionSuffix>-<Date.now()>` per
        # `src/container-runner.ts`. Test a representative set.
        agents = [
            "nanoclaw-old-wtf-1747934567890",
            "nanoclaw-trusted-12345-1747934567890",
            "nanoclaw-default-1747934567890",
        ]
        self.assertEqual(_run_predicate(agents), agents)

    def test_mixed_set_returns_agents_only(self) -> None:
        # Realistic `docker ps` output after deploy: orchestrator and
        # one in-flight agent.
        self.assertEqual(
            _run_predicate(
                [
                    "nanoclaw",
                    "nanoclaw-default-1747934567890",
                ]
            ),
            ["nanoclaw-default-1747934567890"],
        )

    def test_empty_input(self) -> None:
        # No containers running at all — predicate exits cleanly with
        # empty output (the `|| true` handles grep's exit 1).
        self.assertEqual(_run_predicate([]), [])

    def test_unrelated_container_names_ignored(self) -> None:
        # Non-`nanoclaw-` containers (e.g. host-side services on the
        # same Docker daemon) must not match.
        self.assertEqual(
            _run_predicate(
                [
                    "homeassistant",
                    "portainer",
                    "nanoclaw-default-1747934567890",
                ]
            ),
            ["nanoclaw-default-1747934567890"],
        )


if __name__ == "__main__":
    unittest.main()
