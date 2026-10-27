"""Stand-in for ffmpeg in process tests. STUB_MODE picks the behaviour, STUB_LOG gets
one JSON line per invocation (pid, TZ, argv)."""

import json
import os
import sys
import time

FRAME = 640 * 360
PCM = 32000
JPEG = b"\xff\xd8stub-image\xff\xd9"


def main() -> None:
    args = sys.argv[1:]
    mode = os.environ["STUB_MODE"]
    entry = {"pid": os.getpid(), "tz": os.environ.get("TZ"), "argv": args}
    with open(os.environ["STUB_LOG"], "a") as log:
        log.write(json.dumps(entry) + "\n")

    if mode == "fail":
        print("rtsp://user:hunter2@camera/stream2: Connection refused", file=sys.stderr, flush=True)
        sys.exit(1)

    if mode.startswith("camera"):
        frames, pcm, stills = (int(a[5:]) for a in args if a.startswith("pipe:"))
        os.write(frames, bytes(FRAME))
        os.write(frames, b"\xff" * FRAME)
        if mode == "camera_ok":
            os.write(pcm, bytes(PCM))
            os.write(stills, JPEG[:5])
            os.write(stills, JPEG[5:])
    elif mode == "recorder_ok":
        for tick in range(1, 10_000):
            print(f"out_time_us={tick * 100_000}", flush=True)
            time.sleep(0.05)
    elif mode == "recorder_stall":
        print("out_time_us=100000", flush=True)
    time.sleep(60)


main()
