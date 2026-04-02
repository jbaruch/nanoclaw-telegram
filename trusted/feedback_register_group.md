# register_group — requiresTrigger parameter

The `mcp__nanoclaw__register_group` tool supports a `requiresTrigger` parameter.
When registering a group with "no trigger", always pass `requiresTrigger: false` explicitly.
Don't rely on Baruch to fix it manually after registration.

Example:
```
register_group(jid: "tg:-100...", name: "...", folder: "telegram_...", trigger: "@AyeAye", requiresTrigger: false)
```
