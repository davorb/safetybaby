"""Synthesizes the two alert sounds. Original work, no third-party audio, so the repo can be public.

    python3 tvos/scripts/gen_sounds.py
"""

import math
import struct
import wave
from pathlib import Path

RATE = 44100
OUT = Path(__file__).parent.parent / "BabymonTV" / "Resources" / "Sounds"


def write(name: str, samples: list[float]) -> None:
    peak = max(abs(s) for s in samples) or 1.0
    with wave.open(str(OUT / name), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(struct.pack("<h", int(32767 * 0.95 * s / peak)) for s in samples))


def tone(freq: float, t: float) -> float:
    """A square-ish wave from the first odd harmonics: harsh and hard to sleep through."""
    return sum(math.sin(2 * math.pi * freq * k * t) / k for k in (1, 3, 5, 7))


def emergency() -> list[float]:
    """Three beep pairs alternating 1000 and 1500 Hz with hard edges, about 3 s."""
    out: list[float] = []
    for _ in range(6):
        for freq in (1000.0, 1500.0):
            n = int(RATE * 0.24)
            for i in range(n):
                edge = min(1.0, i / 200, (n - i) / 200)
                out.append(tone(freq, i / RATE) * edge)
            out.extend([0.0] * int(RATE * 0.04))
    return out


def chime() -> list[float]:
    """Two rising bell notes with exponential decay, about 1.3 s."""
    out: list[float] = []
    for freq in (880.0, 1320.0):
        n = int(RATE * 0.65)
        for i in range(n):
            t = i / RATE
            bell = math.sin(2 * math.pi * freq * t) + 0.4 * math.sin(2 * math.pi * freq * 2.76 * t)
            out.append(bell * math.exp(-4.0 * t) * min(1.0, i / 100))
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    write("emergency-alarm.wav", emergency())
    write("high-chime.wav", chime())
