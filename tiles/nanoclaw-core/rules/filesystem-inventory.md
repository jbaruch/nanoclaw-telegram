# Filesystem Inventory — Read from Disk

When reporting on anything that exists on disk — installed skills, rules, tiles, files, config state — **always read from disk first**. Never synthesize from memory, conversation history, or prior context.

## The rule

Any claim about what is present on the filesystem MUST be backed by a live read:

- Tile contents → `ls /home/node/.claude/.tessl/tiles/…`
- File existence → `ls` or `Glob`
- File contents → `Read` or `cat`
- Config state → read the actual file

**Memory and conversation history are not substitutes for `ls`.**

## Why this matters

LLMs synthesize plausible-sounding answers from prior context. Stale tile structures, moved files, and renamed skills will produce confident but wrong reports. The filesystem is ground truth. The model is not.

## Applies to

- Tile inventory reports (skills per tile, rules per tile)
- "What's installed?" questions
- Verifying staging vs. installed state
- Any claim of the form "X is in tile Y" or "file Z exists"

When in doubt: run `ls` first, then answer.
