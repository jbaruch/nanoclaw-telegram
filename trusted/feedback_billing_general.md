# Billing and Invoice Emails — General Rule
Do NOT surface billing, invoice, or subscription charge notifications in heartbeat or cleanup.
Everything is on autopay. The only exception: emails explicitly requiring manual payment action
(e.g., "payment failed", "card declined", "action required to avoid service interruption",
"invoice past due"). Routine invoices, receipts, and billing confirmations = silent ignore.
