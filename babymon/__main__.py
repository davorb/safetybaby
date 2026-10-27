"""CLI. `uv run babymon ...`

    babymon run [--nap | --no-nap] [--arm | --disarm]
        Start the service. Flags become one SetControls fact at startup; without
        flags the persisted controls apply.

    babymon probe camera [--seconds 30]
        Run only the camera actor; print frames/s, PCM s/s, stills/s, motion levels,
        and the segment files that appeared. The check for unit 5.
    babymon probe audio FILE.wav
        Print P(cry) and loudness per second. The check for unit 7.
    babymon probe vision FRAME.jpg [--reference REF.jpg]
        Print the typed Observation. The check for unit 8.
    babymon probe alert --priority {-2..2}
        Send a test notification with the latest still. The check for unit 9.
    babymon replay FACTS.jsonl
        Feed recorded facts through brain + RulesDecider and print effects. Lets a
        real night's facts become a regression fixture.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError
