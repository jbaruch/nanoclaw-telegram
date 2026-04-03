---
name: no-unverified-claims
description: "Adversarial fact-check skill that prevents hallucinated or outdated information from reaching users. Verifies URLs are reachable, cross-references API endpoints and parameters, checks product names and version numbers, and validates claims against live sources rather than training data. Use when about to state a fact about an external service, API, product, or URL — or when asked 'is this accurate', 'verify this', 'check if this exists', 'is this up to date', or 'double-check' anything sourced from training data."
---

# No Unverified Claims — Adversarial Fact Check

Invoke this skill when you are about to state a fact about an external service, API, product, or URL — especially anything from training data.

## Process

All of this happens inside <internal> tags — never surfaces to the user unless the conclusion is "I cannot verify this."

1. State the claim you are about to make. Be specific: what exactly are you claiming?

2. Spawn an adversarial Agent with this prompt (substitute [YOUR_CLAIM] with the actual claim):

---
You are the Verification Checker — an adversarial agent whose ONLY job is to prevent hallucinated or outdated facts from reaching the user.

The main agent is about to state this: [YOUR_CLAIM]

Challenge it relentlessly. Training data is never a source — demand a live lookup for every claim. Ask:
- Could this be outdated?
- What specific search query would confirm or deny this right now?

Suggest concrete verification steps such as:
- `web_search("'[service] [feature] 2026'")`
- `browse("[URL]")` and confirm the expected element or endpoint exists
- Check the service's official changelog or docs page for the current version

Only conclude "verified" if the main agent has performed a lookup and found a current, live source. Only conclude "unverifiable" if at least 2 approaches have been tried and none worked. If the claim is wrong or outdated, say so clearly and suggest what to do instead.
---

3. Perform each verification step the adversarial agent suggests using available tools (e.g., `web_search`, `browse`).

4. Iterate until one of these exit conditions is met:
   - **Verified** — a live source confirms the claim → state it with that source cited
   - **Wrong/outdated** — verification shows the claim is incorrect → retract it, find and state the correct current answer
   - **Unverifiable** — at least 2 approaches failed → tell the user "I'm not certain this is current; here's what I'd suggest verifying"

5. Never state the original claim with confidence if it came only from training data and could not be verified.

---

## Example: End-to-End Fact Check

**Claim to make:** "The OpenAI Chat Completions endpoint is `https://api.openai.com/v1/chat/completions`."

**Step 1 — State the claim:**
> Claiming that the OpenAI Chat Completions REST endpoint is at `https://api.openai.com/v1/chat/completions`.

**Step 2 — Adversarial agent challenges:**
> This could be outdated — OpenAI has versioned its API before. Search for the current endpoint and browse the official docs to confirm.

**Step 3 — Verification steps performed:**
```
web_search("OpenAI Chat Completions API endpoint 2025 site:platform.openai.com")
→ Result: Official docs confirm POST https://api.openai.com/v1/chat/completions as of 2025.

browse("https://platform.openai.com/docs/api-reference/chat/create")
→ Page loads; endpoint URL matches the claim.
```

**Step 4 — Conclusion:** Verified via live source (OpenAI platform docs, accessed 2025). State the claim with the source URL cited.

---

**Unverifiable path:** If both a `web_search` and a `browse` attempt fail to return an authoritative live source, conclude unverifiable and tell the user: "I'm not certain this is current — I'd recommend checking the official docs or contacting support directly."
