# Session Bootstrap — Run on Every New Session

**Before your first response in any session**, check if memory bootstrap is needed:

```python
import os
sentinel = '/tmp/session_bootstrapped'
current_session = os.environ.get('CLAUDE_SESSION_ID', '')
needs_bootstrap = True
if os.path.exists(sentinel):
    stored = open(sentinel).read().strip()
    needs_bootstrap = (stored != current_session)
```

If `needs_bootstrap` is True → `Skill(skill: "tessl__trusted-memory")` before responding.

If False → skip silently, respond normally.

This is a hard requirement. Do not skip. Do not defer. Memory bootstrap runs before the first reply.
