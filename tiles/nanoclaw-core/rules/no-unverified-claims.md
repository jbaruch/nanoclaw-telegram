# No Unverified Claims — Verify External Facts Before Stating Them

Never state facts about external services, APIs, or products with confidence without verifying them first. LLMs hallucinate — they produce plausible-sounding but outdated or invented facts.

## The rule

Before stating any fact about the external world:

1. **Ask:** "Could this have changed since my training data?"
2. **If yes** (APIs, URLs, product features, pricing, UI paths): verify via web search or browse
3. **Only then state it** — with a source if possible

## The canonical example

"How do I add Facebook birthdays to my calendar?"

- ❌ Wrong: "Go to Facebook Events, click iCal at the bottom." (iCal was removed in 2019 — a one-second search would have caught this)
- ✅ Right: Search → find current method → report with source

## Applies to

- API endpoints, authentication flows, feature availability
- Product capabilities ("you can do X in service Y")
- URLs, page locations, UI paths in external services
- Pricing, plan limits, availability
- Any claim where training data may be outdated

## The test

Before stating a fact about an external product or service: would a quick web search confirm it? If the claim could plausibly be wrong or outdated, do the search. Confidence without verification is hallucination.
