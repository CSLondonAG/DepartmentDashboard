# Chat concurrency utilisation update

Chat utilisation now uses a two-slot capacity model per agent.

- One active chat uses 50% of an agent's chat capacity.
- Two simultaneous chats use 100%.
- A third overlapping chat is capped for the KPI instead of increasing
  utilisation beyond the configured capacity.
- Work outside chat-eligible presence intervals is excluded as before.
- Shared `Available_All` time is fully chat-eligible. Email receives only the
  capacity left after active chats consume their modeled share. This prevents
  shared capacity from being counted in full for both channels.

The capacity is configured in `department_dashboard.py`:

```python
CHAT_CONCURRENCY_CAPACITY = 2
```

The Chat Utilization tile now reports capacity-adjusted **load-hours**, while
the per-agent table includes both chat slot-hours and normalized chat load.

Email utilization uses the capacity remaining after chat work:

- Email-only presence provides one full email-capacity hour per elapsed hour.
- During shared presence, zero active chats leave 100% email capacity.
- One or more active chats leave no email capacity because the agent cannot be
  assigned an email while handling chat.

Email work beyond that residual capacity is reported as excess demand instead
of being allowed to inflate the utilization percentage.

## Verification

Run the focused unit tests with:

```shell
python -m unittest discover -s tests -v
```
