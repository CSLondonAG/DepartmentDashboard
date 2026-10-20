# Chat concurrency utilisation update

Chat utilisation now uses a two-slot capacity model per agent.

- One active chat uses 50% of an agent's chat capacity.
- Two simultaneous chats use 100%.
- A third overlapping chat is capped for the KPI instead of increasing
  utilisation beyond the configured capacity.
- Work outside chat-eligible presence intervals is excluded as before.
- Shared `Available_All` time remains fully eligible in each standalone channel
  denominator. The chat and email percentages are therefore independent and
  must not be added together. This keeps each KPI bounded at 100%.

The capacity is configured in `department_dashboard.py`:

```python
CHAT_CONCURRENCY_CAPACITY = 2
```

The Chat Utilization tile now reports capacity-adjusted **load-hours**, while
the per-agent table includes both chat slot-hours and normalized chat load.

## Verification

Run the focused unit tests with:

```shell
python -m unittest discover -s tests -v
```
