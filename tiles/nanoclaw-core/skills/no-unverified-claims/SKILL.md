---
name: no-unverified-claims
description: "Adversarial fact check to prevent hallucinated or outdated facts from reaching the user. Invoke when about to state a fact about an external service, API, product, or URL — especially anything from training data."
---

# No Unverified Claims — Adversarial Fact Check

Invoke this skill when you are about to state a fact about an external service, API, product, or URL — especially anything from training data.

## Process

All of this happens inside <internal> tags — never surfaces to the user unless the conclusion is "I cannot verify this."

1. State the claim you are about to make. Be specific: what exactly are you claiming?

2. Spawn an adversarial Agent with this prompt (substitute [YOUR_CLAIM] with the actual claim):

---
You are the Verification Checker. You are an adversarial agent whose ONLY job is to prevent hallucinated or outdated facts from reaching the user.

The main agent is about to state this: [YOUR_CLAIM]

Challenge it relentlessly:
- Could this be outdated? (APIs change, features get removed, URLs move)
- Is this based on training data or actual verification?
- What specific search query would confirm or deny this right now?
- What Composio tool or browser action could verify it?

For every claim, demand a source. "I know this from training" is not a source.

Suggest concrete verification steps:
- "Search for '[service] [feature] 2026'"
- "Browse to [URL] and check if [element] exists"
- "Check Composio for [API endpoint]"

Only conclude "verified" if the main agent has actually performed a lookup and found a current, live source. Only conclude "unverifiable" if you've tried at least 2 approaches and none worked.

If the claim turns out to be wrong or outdated: say so clearly and suggest what to do instead (search for the current method).
---

3. Perform each verification step the adversarial agent suggests.

4. Iterate until:
   - The claim is verified with a live source → state it with that source
   - The claim is proven wrong/outdated → retract it, find the correct current answer
   - Both agree it cannot be verified → tell the user "I'm not certain this is current, here's what I'd suggest verifying"

5. Never state the original claim with confidence if it came only from training data and couldn't be verified.
