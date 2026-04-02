---
name: check-orders
description: Fetches order-related emails from Gmail, updates orders-db.json, and flags anomalies (cancellations, refunds, purchases >$200, overdue deliveries). Silent on normal order flow. Use when the user asks about order status, order tracking, order emails, shipment status, purchase alerts, or needs to sync Gmail order data with the orders database.
---

You are AyeAye, Baruch's assistant. Check for order updates from Gmail and update the order database.

## Step 1: Load current orders database

Read `/workspace/group/orders-db.json`. Note the `last_checked` timestamp — you'll use it to avoid reprocessing old emails.

## Step 2: Fetch order-related emails from Gmail

Discover Gmail tool per `composio-preamble` rule, then run all queries below (`max_results: 20`, `include_spam_trash: false`):

| # | Query |
|---|-------|
| 1 | `from:auto-confirm@amazon.com` |
| 2 | `from:shipment-tracking@amazon.com` |
| 3 | `"Your order" (shipped OR delivered OR cancelled OR refund)` |
| 4 | `from:noreply@shopify.com OR from:no-reply@shopify.com` |
| 5 | `subject:(order confirmation OR order shipped OR order delivered OR order cancelled OR refund)` |

Deduplicate results by message ID across all queries.

Example invocation:
```
COMPOSIO_SEARCH_TOOLS(query="GMAIL_FETCH_EMAILS")
GMAIL_FETCH_EMAILS(query="from:auto-confirm@amazon.com", max_results=20, include_spam_trash=false)
```

## Step 3: Parse each email

For each email, extract the following fields:

**Source** — map sender domain:

| Domain | Source value |
|--------|-------------|
| amazon.com | `"amazon"` |
| shopify.com | `"shopify"` |
| shop.app | `"shop"` |
| anything else | `"other"` |

**Status** — map subject/snippet keywords:

| Keywords | Status value |
|----------|-------------|
| "on the way", "shipped" | `"shipped"` |
| "has been delivered", "delivered" | `"delivered"` |
| "cancelled", "canceled" | `"cancelled"` |
| "refunded", "refund" | `"refunded"` |
| "order confirmation", "ordered" | `"ordered"` |
| no match | `"unknown"` |

**Remaining fields:**
- **amount**: extract dollar amount from subject or snippet (`$XX.XX`, `Total: $XX`); if multiple found, use the largest; default `0`
- **currency**: `"USD"` (default)
- **description**: subject line, stripped of boilerplate (e.g. remove "Your Amazon.com order", keep item names)
- **order_date**: email received date (`YYYY-MM-DD`)
- **expected_delivery**: parsed date if mentioned (e.g. "arrives by Dec 5"); `null` if not found
- **email_message_id**: Gmail message ID

## Step 4: Merge into orders-db.json

For each parsed email:
1. Same `email_message_id` already exists → **skip** (already processed).
2. Same source + description + order_date but different status → **update** status and `last_updated`.
3. Otherwise → **add** as a new order entry with:
   - `id`: `{source}-{order_date}-{hash}` where `hash` is the first 8 hex chars of SHA-1 of the description:
     ```python
     import hashlib
     hash = hashlib.sha1(description.encode()).hexdigest()[:8]
     id = f"{source}-{order_date}-{hash}"
     ```
   - `flagged`: `false`
   - `flag_reason`: `null`

## Step 5: Flag anomalies

Review all orders. Set `flagged: true` and `flag_reason` if ANY condition applies:

| Condition | flag_reason |
|-----------|-------------|
| `status == "cancelled"` | `"Order cancelled"` |
| `status == "refunded"` | `"Refund/return"` |
| `amount > 200` | `"Large purchase: $${amount}"` |
| `status` is `"shipped"` or `"ordered"` AND `expected_delivery` is not null AND `expected_delivery` is more than 2 days before today | `"Overdue delivery"` |

Clear flags (`flagged: false`, `flag_reason: null`) on any orders where none of the above apply — handles status changes (e.g. a previously "overdue" order now marked "delivered").

## Step 6: Update orders-db.json

Write the updated database back to `/workspace/group/orders-db.json`:
```json
{
  "orders": [...],
  "last_updated": "<current ISO timestamp>",
  "last_checked": "<current ISO timestamp>"
}
```

## Step 7: Report flagged items

Collect all orders where `flagged: true`. If none → stay completely silent (no message).

If there are flagged orders, send a message via `mcp__nanoclaw__send_message`:

Format (Telegram HTML):
```
<b>📦 Order alerts:</b>

• <b>[description]</b> — [flag_reason] (<i>[source], [order_date]</i>)
```

One bullet per flagged order. Keep it concise.
