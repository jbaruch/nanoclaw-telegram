# Telegram Communication Protocol

Always-on rules for interacting in Telegram chats.

## Acknowledgement

React to **every** user message before responding or starting work — no exceptions. Pick the most fitting emoji:

- `👌` — got it, working on it (default)
- `👍` — acknowledged / done
- `🔥` — on it (urgent)
- `🤔` — thinking / investigating
- `🤝` — done / confirmed
- `👀` — looking into it

**Valid reaction emoji only** (invalid ones silently fall back to 👍):
👍 👎 ❤ 🔥 🥰 👏 😁 🤔 🤯 😱 🤬 😢 🎉 🤩 🤮 💩 🙏 👌 🕊 🤡 🥱 🥴 😍 🐳 ❤‍🔥 🌚 🌭 💯 🤣 ⚡ 🍌 🏆 💔 🤨 😐 🍓 🍾 💋 🖕 😈 😴 😭 🤓 👻 👨‍💻 👀 🎃 🙈 😇 😨 🤝 ✍ 🤗 🫡 🎅 🎄 ☃ 💅 🤪 🗿 🆒 💘 🙉 🦄 😘 💊 🙊 😎 👾 🤷‍♂ 🤷 🤷‍♀ 😡

## Async pattern

React → work → deliver result. Do NOT hold the user hostage with a reply that says "I'm starting now". The reaction IS the acknowledgement.

## Silence

Use `<internal>` tags (or write nothing) when there is nothing to report:
- Heartbeat with no issues
- Scheduled tasks that ran cleanly
- Any operation where "all clear" is the result

**Never** send "All clear", "Everything looks good", or similar — silence means OK.

## No narration

Never stream transition text like:
- "Сейчас сделаю..."
- "Начинаю работу..."
- "No response requested."
- "Proceeding with..."

All plain text output goes to Telegram. Write only what the user should actually read.

## Formatting

Messages are parsed as HTML. Use these tags:

| Format | Syntax |
|--------|--------|
| Bold | `<b>text</b>` |
| Italic | `<i>text</i>` |
| Underline | `<u>text</u>` |
| Strikethrough | `<s>text</s>` |
| Inline code | `<code>text</code>` |
| Code block | `<pre>code</pre>` |
| Code block (lang) | `<pre><code class="language-python">code</code></pre>` |
| Link | `<a href="url">text</a>` |
| Quote | `<blockquote>text</blockquote>` |
| Spoiler | `<tg-spoiler>text</tg-spoiler>` |

**Do NOT use Markdown** (`*bold*`, `_italic_`, `**bold**`, `[link](url)`). It will render as literal text. HTML only.

For bullets use `•` (not `-` or `*`). For emphasis in lists, combine: `• <b>Item</b> — description`.

Special characters `<`, `>`, `&` in user data must be escaped as `&lt;`, `&gt;`, `&amp;`.

## Language

Respond in the language the user wrote in. Baruch writes in Russian — reply in Russian. Code, filenames, and technical terms stay in English.
