# Slack Message Formatting Rules

**When responding in a Slack channel** (workspace folder starts with `slack_` or JID contains Slack identifiers), always use mrkdwn syntax instead of standard Markdown. This is an always-on rule — no need to invoke a skill.

## Syntax

- Bold: `*text*` — never `**text**`
- Italic: `_text_`
- Links: `<url|text>` — never `[text](url)`
- Mentions: `<@USERID>`, `<#CHANNELID>`, `<!here>`, `<!channel>`
- Bullets: `•` — numbered lists are not supported natively (use `• 1. item`)
- Block quote: `> text`
- Emoji: `:white_check_mark:` shortcodes

## What NOT to use

- No `##` headings — use `*Bold text*` instead
- No `---` horizontal rules
- No `| tables |` — use code blocks or plain text
- No `[text](url)` links

## Example

```
*Summary*

• First point
• Second point: <https://example.com|link text>

> Note: something to highlight

:white_check_mark: Done
```
