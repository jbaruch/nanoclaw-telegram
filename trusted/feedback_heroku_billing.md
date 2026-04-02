# Heroku Billing Invoices
Small Heroku billing invoices (e.g. $5/month) are routine infrastructure costs, not actionable.
Do NOT flag in heartbeat or cleanup. Only flag if amount is unexpectedly large (>$50) or service is being cancelled.

See also: feedback_billing_general.md for the broader rule covering all billing/invoice emails.
