# Implementation units, in order

Each unit ends in a check a reviewer can rerun. Units 1 to 4 need no camera, network or keys.

1. **Scaffold and config.** Add `uv init`, `pyproject.toml` (Python 3.12, deps pinned to the measured versions), `config.py`, the `babymon` entry point, and an `import-linter` contract that keeps `domain`, `sleep`, `rules` and `brain` free of I/O libraries. **Check:** `uv run babymon run --help` works. `pytest tests/test_config.py` shows that a full `.env` parses, a partial Pushover group fails naming the missing variable, and absent vision and Pushover groups give `Capabilities(vision=False, push=False)`. `uv run lint-imports` passes and fails when `brain.py` imports `sqlite3`.

2. **Domain and sleep machine.** Write `domain.py` and `sleep.py`. **Check:** in `pytest tests/test_sleep.py`, table-driven signal scripts cover asleep to stirring after 3 s of MOVING, stirring to asleep after 120 s of QUIET, any state to crying after 3 s, crying to awake after 20 s, mixed MOVING and ACTIVE accumulating toward AWAKE, and a duplicated Tick not double-stepping.

3. **Brain, rules, gate.** Write `brain.py` and `rules.py`. **Check:** in `pytest tests/test_brain.py`, scripted fact sequences cover each hard rule. Stomach on two checks gives EMERGENCY. Face covered on one confident and one low-confidence check gives no alert. Head left over 120 s gives HIGH. Crying over 60 s gives HIGH. Awake during nap gives NORMAL, and nothing fires when nap is off. Falling asleep records a `SleepChanged` and sends no Notify. A condition held for 10 minutes re-alerts exactly at cooldown. Disarmed means no notifications. The tests also cover CONFIRM checks every 15 s while a streak is pending, a settled-after-motion request, and an in-flight timeout.

4. **Store.** Write `store.py`. **Check:** `pytest tests/test_store.py` runs in a tmp dir with fake segment and snapshot files named across 30 hours. `sweep` deletes exactly the files older than 24 h, keeps the segment covering the horizon, and a second `sweep` deletes nothing. `restore` returns the last *delivered* alert per kind. `ControlsChanged` round-trips through the controls row. `clip_at` picks the right segment and offset. `sleep_timeline` produces contiguous spans.

5. **Camera session.** Write `camera.py` (argv, supervisor, readers, `MotionMeter`, `split_jpegs`). **Check:** `uv run babymon probe camera --seconds 60` against the live C210 prints about 4 frames/s, 1 PCM window/s and 1 still/s, and shows one MP4 segment growing that `ffprobe` reads with h264 and aac streams. Blocking the camera IP with `iptables` for 20 s prints `CameraStatus(up=False)`, then a reconnect within the backoff. `pgrep ffmpeg` shows one process throughout. A unit test covers `split_jpegs` on a buffer cut mid-image.

6. **Motion calibration.** Tune the `MotionMeter` thresholds and ROI. **Check:** `probe camera` on the empty crib at night (IR) stays below `motion_moving` for 5 minutes, and a hand wave in the ROI exceeds `motion_active`.

7. **Audio actor.** Write `audio.py` with YAMNet ONNX. **Check:** `uv run babymon probe audio cry.wav` gives P(cry) above 0.6 on most seconds. A recording of the room's white-noise machine alone stays below 0.2. A cry mixed over white noise at realistic level stays above 0.6.

8. **Vision backend.** Write `vision.py`. **Check:** `uv run babymon probe vision` over 10 staged stills (doll on back, on side, on stomach, blanket over face, head left, empty crib, IR and daylight), each with the reference image, prints Observations. The expected labels are kept in `tests/vision_cases.toml`, and the run reports agreement per field. Without `ANTHROPIC_API_KEY` the command exits with "vision disabled".

9. **Notifier.** Write `notify.py`. **Check:** `uv run babymon probe alert --priority 1` arrives on both phones with the still attached. `--priority 2` repeats every 30 s until acknowledged. Without Pushover keys, `LogOnly` records `delivered=False`.

10. **Wiring.** Write `app.py` with fakes first. **Check:** an integration test runs `app.run` with a fake camera that publishes scripted facts and stills, a scripted `VisionBackend` returning stomach twice, and a recording `Notifier`. Exactly one EMERGENCY arrives, with a snapshot file on disk and an `AlertSent` row. Killing the task and restarting against the same data dir does not re-send within cooldown. A second test, with the delivery failing before the crash, re-sends.

11. **Dashboard.** Write `web.py` and one HTML page. **Check:** with the service running on the live camera, a browser on the LAN shows the frame updating, the current sleep state, a 24 h timeline, and the event list. A snapshot opens, and a clip link plays at the event's offset. The nap toggle appears in `/api/state` and survives a restart. "Set reference" replaces `reference.jpg`.

12. **Container and soak.** Add a Dockerfile (python 3.12 slim, ffmpeg, the YAMNet model) and compose with `restart: unless-stopped`. **Check:** a 26 h run on the NAS keeps the `segments/` size flat after hour 24 and keeps `snapshots/` and the event count bounded. A `docker kill` mid-run comes back with nap mode and cooldowns intact. CPU stays under one core on the NAS, measured with `docker stats`.
