"""Planned realtime scheduler.

Future role:
- Maintain frame and Stage 1 action buffers.
- Build Stage 2 requests with [t, t+10, t+20].
- Send only sampled requests to the VLM worker.
"""


class RealtimeScheduler:
    def __init__(self) -> None:
        raise NotImplementedError("RealtimeScheduler is planned for the realtime phase.")
