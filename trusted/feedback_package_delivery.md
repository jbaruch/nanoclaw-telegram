# Package Delivery Rule

Do NOT flag standard package delivery notifications (out for delivery, shipped, delivered) in heartbeat.

Exception — flag only if:
- Signature required
- Temperature-sensitive / needs to be received immediately
- Delivered to wrong address or lost

Rationale: Alice is typically home and receives packages. Standard deliveries resolve themselves.
