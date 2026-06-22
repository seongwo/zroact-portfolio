"""Planned Stage 1 worker.

Future role:
- Load YOWOv3 once when the serving process starts.
- Receive frames from a queue or stream.
- Return bbox/action detections without launching a subprocess per request.
"""


class Stage1Worker:
    def __init__(self) -> None:
        raise NotImplementedError("Stage1Worker is planned for the realtime phase.")
