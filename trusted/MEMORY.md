# Memory Index

## Feedback
- [feedback_google_workspace_invoice.md](feedback_google_workspace_invoice.md) — Google Workspace monthly invoice for sadogursky.com: skip in heartbeat, Baruch collects at year end.
- [feedback_tessl_tesla.md](feedback_tessl_tesla.md) — Voice transcriber always says "Tesla" when Baruch means "Tessl" (tile system). Understand from context.
- [feedback_heartbeat_silence.md](feedback_heartbeat_silence.md) — Heartbeat must be completely silent when all checks pass. No "all clear" messages.
- [feedback_streaming_text.md](feedback_streaming_text.md) — ALL plain text output streams to Telegram. Never write transition narration or "No response requested." — use `<internal>` or write nothing.
- [feedback_valid_reactions.md](feedback_valid_reactions.md) — Only valid emojis for react_to_message: 👍 👎 ❤ 🔥 🎉 🤔 🤯 👏 😁 😢 🤩 🙏 👌 ✅ — others silently fail and look like bot died.
- [feedback_no_online_cfps.md](feedback_no_online_cfps.md) — Never suggest online/virtual conferences in CFP section of morning brief. In-person only.
- [feedback_pin_morning_brief.md](feedback_pin_morning_brief.md) — Always pin the morning brief message (pin: true). It gets lost in chat noise.
- [feedback_email_software_updates.md](feedback_email_software_updates.md) — Software update emails for tools Baruch uses (e.g. Synergy 3) are actionable, not noise. Flag in heartbeat.
- [feedback_jpmorgan_legal_notices.md](feedback_jpmorgan_legal_notices.md) — J.P. Morgan "new online notice" emails are mandatory legal/regulatory docs (ADV2B etc.) sent to ~15 brokerage accounts. Do NOT report. Only flag actual banking actions: construction draws, payment failures, fraud.
- [feedback_amir_email.md](feedback_amir_email.md) — amir@sadogursky.com is a family mailing group (Baruch + Alice). Amazon/promo noise → ignore. School tuition, school notices, time-sensitive family items → flag.
- [feedback_package_delivery.md](feedback_package_delivery.md) — Do NOT flag standard package delivery notifications. Alice is home and receives them. Exception: signature required, temperature-sensitive, or delivery failure.
- [feedback_myhealth_vanderbilt.md](feedback_myhealth_vanderbilt.md) — My Health at Vanderbilt (MYHEALTH-NOREPLY@vumc.org): ignore Alice's messages, flag Baruch's and Amir's.
- [feedback_heroku_billing.md](feedback_heroku_billing.md) — Small Heroku billing invoices (≤$50/mo) are routine. Do NOT flag. Only flag if >$50 or service being cancelled.
- [feedback_amazon_subscribe_save.md](feedback_amazon_subscribe_save.md) — Amazon Subscribe & Save price change notifications are not actionable. Do NOT flag.
- [feedback_amazon_associates_tos.md](feedback_amazon_associates_tos.md) — Amazon Associates ToS/Operating Agreement updates are auto-accepted. Do NOT flag.
- [feedback_billing_general.md](feedback_billing_general.md) — All billing/invoice/subscription charge emails are silent. Everything is on autopay. Only exception: explicit manual payment required (payment failed, card declined, past due).
- [feedback_java_champions_ml.md](feedback_java_champions_ml.md) — Java Champions mailing list (javachampions@googlegroups.com) — mass community threads, not actionable. Silently ignore.

## Project
- [project_nanoclaw_status.md](project_nanoclaw_status.md) — NanoClaw feature status: what works, what's pending, research docs locations.
- [project_house_build.md](project_house_build.md) — New home construction at 1835 Burke Hollow Rd, Nolensville TN. Bear Ridge (Mike Minard) is GC, contract unsigned, $5.23M estimate, Tesla Solar pending, structural engineer (Floyd) waiting on roof load numbers.

## People
- [key-people.md](key-people.md) — Key people with Telegram usernames for recognition in group chats: Leonid (@ligolnik), Viktor (@gamussa), Simon Maple, Patrick Debois.

## Trust
- [trusted_senders.md](trusted_senders.md) — Baruch's username is @JBaruch. Stable identifier for trust decisions in any group.

- [feedback_tracks_tv_logins.md](feedback_tracks_tv_logins.md) — New device logins to Peacock/Netflix/Disney+ at night = tracks.tv (show tracker). Not a security alert, ignore in heartbeat.

## Credentials
- [credentials_scope.md](credentials_scope.md) — What credentials I actually have (only Composio). Never claim access to Reclaim, TripIt, or Google OAuth directly.

## Location
- To determine where Baruch currently IS — always check Google Calendar (events show real-time location). TripIt/travel-schedule.json shows planned trips, not current physical location. Calendar is ground truth.
- [feedback_nightly_errors.md](/workspace/trusted/feedback_nightly_errors.md) — Nightly errors requiring host action must be reported immediately, not silently logged.

## Blog Notes
- [blog-notes.md](https://github.com/jbaruch/nanoclaw/blob/main/blog-notes.md) — Blog raw notes live in the **root of the nanoclaw GitHub repo, main branch** (`blog-notes.md`). Update via Composio GitHub (`GITHUB_CREATE_OR_UPDATE_FILE_CONTENTS`, owner: jbaruch, repo: nanoclaw, branch: main). Baruch granted permission to push directly without approval ("ты туда можешь фигачить даже без моего апрувала").
