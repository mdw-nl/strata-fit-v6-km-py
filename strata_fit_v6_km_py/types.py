from enum import Enum

class EventType(str, Enum):
    EXACT = "exact"
    CENSORED = "censored"
    INTERVAL = "interval"

class NoiseType(str, Enum):
    NONE = "NONE"
    GAUSSIAN = "GAUSSIAN"
    POISSON = "POISSON"
