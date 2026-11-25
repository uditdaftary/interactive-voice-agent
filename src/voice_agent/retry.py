"""Retry timing shared by the optional backends that load a model or start a worker, so they all wait the same time."""

# After a backend fails to load or loses its worker, the next attempt waits this many seconds.
RETRY_COOLDOWN_S = 30.0
