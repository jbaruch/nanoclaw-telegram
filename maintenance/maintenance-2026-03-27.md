# Maintenance Backlog — 2026-03-27

Generated from a conversation audit on 2026-03-27. Intended as a handoff to Desktop Claude on the host machine. Each item has enough detail to act on without follow-up questions.

---

## CRITICAL (blocking or broken)

### 1. SOUL.md reference broken

**File:** `/workspace/project/groups/main/CLAUDE.md` (also reflected in `/workspace/group/CLAUDE.md`), line 5

**Problem:** The instruction reads:
```
read `/workspace/project/groups/global/SOUL.md`
```
That file does not exist. The actual file is at `/workspace/project/groups/global/CLAUDE.md`. This means the persona rules never load at startup — the agent runs without its defined personality, communication style, and identity.

**Fix (pick one):**
- Option A: Rename the file — `mv /workspace/project/groups/global/CLAUDE.md /workspace/project/groups/global/SOUL.md`
- Option B: Fix the reference in CLAUDE.md — change the path on line 5 from `SOUL.md` to `CLAUDE.md`

Option A is cleaner because SOUL.md is a more intentional name and avoids confusion with the group-level CLAUDE.md files.

---

### 2. Telegram reply threading broken

**Problem:** Stale `agent-runner-src` cache in session directories contains the old `ipc-mcp-stdio.js` without the reply-to field. The cache was built before reply threading code was added, so it never picks up the new version.

**Fix — run on host:**
```bash
rm -rf groups/*/data/sessions/*/agent-runner-src
```

This forces a rebuild of agent-runner-src on next run. No data loss — only cached build artifacts are removed.

---

### 3. Dockerfile missing blog-writer-persona symlink

**File:** `/workspace/extra/projects/nanoclaw/container/Dockerfile`

**Problem:** The Dockerfile has a comment referencing the blog-writer-persona path but the actual symlink command is missing. Every container rebuild loses the symlink and the blog-writer skill breaks silently.

**Fix — add this line to the Dockerfile in the appropriate RUN block:**
```dockerfile
ln -s /workspace/extra/blogs/persona ~/.claude/blog-writer-persona
```

Place it alongside other symlink or setup commands. The comment is already there — this is just the missing execution.

---

## HIGH (functional issues)

### 4. OpenAI API key was wrong project/quota

**Problem:** The previous key was hitting the wrong project or quota. Replaced with new key `sk-proj-VnzdTpX...OtkUoWLcA` — already updated in `data/env/env` inside the container. But the running service hasn't picked it up yet.

**Fix — two steps:**

Step 1: Restart the service on host:
```bash
launchctl kickstart -k gui/$(id -u)/com.nanoclaw
```

Step 2: Add `OPENAI_API_KEY` to `.env` on host so future syncs preserve it:
```bash
# Add to host .env:
OPENAI_API_KEY=sk-proj-VnzdTpX...OtkUoWLcA
```
Without this, the next `cp .env data/env/env` will overwrite the fixed value.

---

### 5. GitHub token missing from data/env/env

**Problem:** `GITHUB_TOKEN` is not present in the container environment. This blocks native git push operations from inside the container.

**Fix:**
1. Add `GITHUB_TOKEN=<your-token>` to `.env` on host
2. Sync: `cp .env data/env/env` (or however env is synced)
3. Restart the service: `launchctl kickstart -k gui/$(id -u)/com.nanoclaw`

---

### 6. Task Scripts section duplicated across CLAUDE.md files

**Problem:** The "Task Scripts" section (how-it-works, wakeAgent, test-your-script-first, etc.) exists verbatim in two places:
- `/workspace/project/groups/main/CLAUDE.md` lines 329-364
- `/workspace/project/groups/global/CLAUDE.md` lines 82-117

These will drift. Already a maintenance liability.

**Fix:**
1. Remove the Task Scripts section from `/workspace/project/groups/main/CLAUDE.md` (lines 329-364)
2. Add a reference line pointing to the global version, e.g.:
   ```
   See `/workspace/project/groups/global/CLAUDE.md` for Task Scripts documentation.
   ```

---

## MEDIUM (architecture / maintenance)

### 7. Group management bloat in CLAUDE.md

**File:** `/workspace/project/groups/main/CLAUDE.md`

**Problem:** Lines 137-364 (~228 lines) contain group management docs: authentication, container mounts, SQLite queries, group registration, sender allowlist, scheduling for other groups. This operational reference material makes the main file hard to scan and harder to keep up to date.

**Fix:**
1. Extract lines 137-364 into a new file: `/workspace/project/groups/global/OPERATIONS.md`
2. Replace those lines in CLAUDE.md with a single reference:
   ```
   ## Operations Reference
   See `/workspace/project/groups/global/OPERATIONS.md` for group management, authentication, container config, and scheduling.
   ```

---

### 8. Heartbeat mixes features with health checks

**File:** `/workspace/group/HEARTBEAT.md` (or equivalent heartbeat config)

**Problem:** Heartbeat checks #10 (calendar) and #11 (email) are not health checks — they're scheduled features that produce output. Embedding them in the 15-minute heartbeat makes it heavier than it needs to be, and conflates two different concerns.

**Fix:**
1. Remove checks #10 and #11 from HEARTBEAT.md
2. Create separate scheduled tasks for calendar and email polling at their appropriate intervals
3. Heartbeat should only verify liveness and system health

---

### 9. No decision tree for Composio vs. agents

**Problem:** CLAUDE.md has no rule for when to use Composio tools directly versus spawning an agent. This creates ambiguity and leads to over-use of agents for simple API calls that Composio handles natively.

**Fix:** Add a decision rule to CLAUDE.md or OPERATIONS.md:
```
Use Composio tools directly for: single API calls, read operations, simple data fetches.
Spawn an agent for: multi-step workflows, anything requiring judgment, tasks with >2 tool calls.
```
Tune the rule based on actual observed patterns.

---

### 10. State files should be consolidated

**Problem:** `/workspace/group/heartbeat-state.json` and `/workspace/group/calendar-state.json` are separate files with no versioning. As more stateful features are added, this fragments further.

**Fix:**
1. Create `/workspace/group/nanoclaw-state.json` with a versioned schema:
   ```json
   {
     "version": 1,
     "heartbeat": { ... },
     "calendar": { ... }
   }
   ```
2. Migrate existing state from the two files into the new structure
3. Update all references to point to `nanoclaw-state.json`
4. Delete the old files once migration is confirmed working

---

### 11. Async ACK rule keeps getting violated

**Problem:** The rule "ACK first with send_message, then background agent, zero text output" has been rewritten 3+ times in CLAUDE.md and still misfires in practice. The rule is present but not prominent enough to override default response behavior.

**Fix options (try in order):**
- Move the async rule to the very top of CLAUDE.md, before any other content, as a standalone block with NO other text before it
- Add the rule to SOUL.md so it loads as part of identity, not just instructions
- Add an explicit violation example showing what NOT to do alongside the correct behavior

---

## LOW (polish)

### 12. 29 skills with no index

**Problem:** Skills exist on the filesystem but there's no way for the user or the agent to discover them without reading the directory. The agent can miss applicable skills.

**Fix:** Create `/workspace/project/groups/global/SKILLS.md` as a skill index with one line per skill: name, trigger phrase, and one-sentence description. Wire it into CLAUDE.md with: "Before starting any complex task, check SKILLS.md for an applicable skill."

---

### 13. Log rotation not implemented

**Problem:** Only a stopgap exists. `/workspace/project/logs/` will grow unbounded.

**Fix:** Set up proper log rotation on the host:
```bash
# Add to /etc/newsyslog.conf or equivalent, or create a launchd plist:
/path/to/project/logs/*.log  644  7  1024  *  J
```
Or use `logrotate` with a config in `/etc/logrotate.d/nanoclaw`.

---

### 14. External heartbeat spec not implemented

**File:** `/workspace/group/external-heartbeat-spec.md`

**Problem:** Spec exists but implementation is missing. The external heartbeat (presumably a watchdog that can alert if the container goes silent) is unbuilt.

**Fix:** Read the spec, implement it. No details here because the spec is the source of truth.

---

### 15. Key people in SOUL.md lack context

**File:** `/workspace/project/groups/global/SOUL.md` (or `/workspace/project/groups/global/CLAUDE.md` until issue #1 is resolved)

**Problem:** Leonid, Viktor, Simon, Patrick, and Chris are listed by first name only, with no roles, relationship descriptions, or behavioral notes. The agent can't use this information effectively — it doesn't know who to loop in, who has authority on what, or what tone to use with each person.

**Fix:** For each person, add at minimum:
- Full name (if applicable)
- Role or relationship to Baruch
- Any behavioral note (e.g., "prefers bullet points", "responds to DMs only", "technical audience")

---

---

### 16. Document files not downloaded from Telegram

**File:** `src/channels/telegram.ts`, line 527-529

**Problem:** The `message:document` handler only stores `[Document: filename]` as a placeholder — it never downloads the file content. This means the agent cannot read PDFs or other documents sent via Telegram.

**Current code:**
```ts
this.bot.on('message:document', (ctx) => {
  const name = ctx.message.document?.file_name || 'file';
  storeNonText(ctx, `[Document: ${name}]`);
```

**Fix:** Add download logic similar to how images and voice messages are handled — use `downloadTelegramFile(bot, ctx.message.document.file_id)` to get the buffer, save it to a path like `/workspace/group/documents/`, and store the path in the message content. For PDFs specifically, the agent can then use the Read tool to extract text.

Scope: medium effort. Consider file size limits (Telegram allows up to 20MB for bots).

---

*End of backlog. 16 items total: 3 critical, 3 high, 5 medium, 4 low, 1 addendum.*
