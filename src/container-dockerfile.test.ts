import { describe, it, expect } from 'vitest';
import fs from 'fs';
import path from 'path';

describe('container/Dockerfile', () => {
  const dockerfilePath = path.resolve(process.cwd(), 'container', 'Dockerfile');
  const contents = fs.readFileSync(dockerfilePath, 'utf8');

  it('installs the `gh` CLI', () => {
    // The cost-monitor dashboard skills (precheck-gating-monitor,
    // session-cap-monitor, daily-spend-rollup) shell out to `gh` for
    // dashboard-issue edits. Without `gh` on PATH inside the container,
    // GITHUB_TOKEN forwarding is dead weight — the skills would have no
    // way to reach the GitHub API at all.
    expect(contents).toMatch(/apt-get install -y[^\n]*\bgh\b/);
  });
});
