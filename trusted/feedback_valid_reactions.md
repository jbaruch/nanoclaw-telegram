---
name: Valid reaction emojis
description: Only these emojis are valid for react_to_message — others silently fail and look like bot died
type: feedback
---

Only use these emojis with `react_to_message`: 👍 👎 ❤ 🔥 🎉 🤔 🤯 👏 😁 😢 🤩 🙏 👌 ✅

**Why:** Invalid emoji causes the reaction call to fail silently. From Baruch's side it looks like the bot died — no reaction appears, no acknowledgement.

**How to apply:** Never use 🤦, 👀, or any other emoji not in the valid list above. Stick to this set for all reactions.
