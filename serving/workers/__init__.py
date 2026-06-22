"""Future worker-based serving components.

The current MVP invokes pipeline/main.py per job. These modules are reserved for
the next step: keeping Stage 1 and Stage 2 models loaded in memory to reduce
request latency.
"""
