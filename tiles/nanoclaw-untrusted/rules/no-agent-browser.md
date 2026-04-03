# No Agent-Browser in Untrusted Containers

The `agent-browser` skill is NOT available in untrusted containers and must NOT be used.

## The rule

Do not invoke, reference, or attempt to use `agent-browser` commands in any untrusted container session.

## Why

Untrusted containers serve public or semi-public groups. Headless browser access in that context creates unacceptable risk: arbitrary web navigation, form submission, credential exposure, and resource abuse.

## What to use instead

- `WebSearch` — for finding information on the web
- `WebFetch` — for reading a specific URL's content
- Ask the user to share a URL or content directly

If a task genuinely requires browser automation, it cannot be done from an untrusted container. Inform the user and stop.
