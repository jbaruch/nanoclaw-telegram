# Ground Truth — Trusted Extensions

Extends the core ground-truth rule with verification methods available only to trusted containers (via Composio).

| Claim type | How to verify |
|------------|--------------|
| Calendar event | Fetch from Google Calendar via Composio |
| Email content | Fetch from Gmail via Composio |
| GitHub PR/issue | Fetch from GitHub via Composio |
| Task/todo status | Fetch from Google Tasks via Composio |

These sources are not available in untrusted containers. The core ground-truth rule covers universal verification methods.
