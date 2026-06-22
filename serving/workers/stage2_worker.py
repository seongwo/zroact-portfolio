"""Planned Stage 2 worker.

Future role:
- Load the VLM once when the serving process starts.
- Receive sampled 3-frame requests from the scheduler.
- Return stable JSON risk_state predictions with minimal model-loading delay.
"""


class Stage2Worker:
    def __init__(self) -> None:
        raise NotImplementedError("Stage2Worker is planned for the realtime phase.")
