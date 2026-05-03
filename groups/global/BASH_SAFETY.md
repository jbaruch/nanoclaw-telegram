# Bash & git workflow safety

These rules apply to every shell, git, and `gh` invocation you make. They are intentionally restated here because the SDK's built-in `Bash` tool description has been trimmed and these safety patterns are no longer carried in-tool.

## Commit messages

- Always pass commit messages via HEREDOC for consistent multi-line formatting — not `-m "single line"`.
- End every commit message with a Co-Authored-By trailer: `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>` (use the current model alias if it differs).
- Focus the message on WHY, not WHAT. Use conventional-commits prefixes: `feat`, `fix`, `chore`, `refactor`, `docs`, `test`.
- Example HEREDOC commit:

  ```bash
  git commit -m "$(cat <<'EOF'
  fix(drive_planner): cascade-delete child events when parent removed

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

## Git safety — NEVER without explicit user permission

- `--no-verify` (skips pre-commit hooks)
- `--no-gpg-sign` (bypasses signing)
- `git push --force` / `git push --force-with-lease` to any remote
- `git reset --hard` (loses uncommitted work)
- `git checkout .` / `git restore .` (discards working tree)
- `git clean -f` (deletes untracked files)
- `git branch -D` (force-deletes local branch)
- `git commit --amend` after a hook failure — the failed commit didn't happen, so amending modifies the PREVIOUS commit and may destroy work. Instead: fix the issue, re-stage, and create a NEW commit.

## Staging

- Prefer `git add <specific-files>` over `git add -A` or `git add .`. The latter can accidentally stage `.env`, `credentials.json`, build artifacts, or other sensitive files.
- Before staging, glance at `git status` to confirm what's about to be added.

## Pre-commit context

- Before drafting a commit message, check `git status`, `git diff` (staged + unstaged), and `git log -5 --oneline` in parallel (multiple Bash calls in a single assistant message) to understand what's changing and follow the repo's commit-message style.

## PR creation

- Always pass `--repo <owner>/<repo>` explicitly to every `gh pr`, `gh issue`, `gh api` command. Never rely on `gh`'s default — it picks the upstream fork in some configurations and you can leak private content there. (This is enforced separately by the repo-chain rules but worth restating.)
- Pass PR body via HEREDOC, with a structured template:

  ```bash
  gh pr create --repo <owner>/<repo> --title "<short title>" --body "$(cat <<'EOF'
  ## Summary
  - <bullet 1>
  - <bullet 2>

  ## Test plan
  - [ ] <test 1>
  - [ ] <test 2>

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

- Keep titles ≤70 chars. Detail goes in the body.
- Never auto-merge. Open the PR and stop — the user reviews.

## Sleep / polling

- Don't write `while true; do sleep 5; ...; done` polling loops. Use `Bash` with `run_in_background: true` for fire-and-forget; use `mcp__nanoclaw__schedule_task` for recurring or future-time work.
- For "wait until X is true," use `until <check>; do sleep 2; done` ONCE in a Bash command — not chained or repeated.
- Don't `sleep` between commands that can run immediately. The runtime allows you to chain commands with `&&` or send multiple Bash calls in one assistant message (which run in parallel).

## Tool selection — prefer dedicated tools when one fits

- File search → `Glob` (NOT `find` or `ls`)
- Content search → `Grep` (NOT shell `grep` or `rg`)
- Reading files → `Read` (NOT `cat` / `head` / `tail`)
- Editing files → `Edit` (NOT `sed` / `awk`)
- Writing files → `Write` (NOT `echo >` / `cat <<EOF`)
- Output to user → text reply (NOT `echo` / `printf`)

`Bash` is for genuine shell-only operations: `gh`, `git`, `curl`, build/test scripts, package managers, system commands. Use it freely there.

## Path safety

- Quote any path containing spaces: `cd "path with spaces/file.txt"`.
- Prefer absolute paths over `cd` to keep the working directory stable across the session.
- When running `find`, search from `.` (or a specific path), not `/` — scanning the full filesystem can exhaust system resources on large trees.
