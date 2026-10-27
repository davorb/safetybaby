# babymon: candidate design (opus)

## Problem

A self-hosted service watches a 4-month-old through a Tapo C210 and wakes a parent only when something needs attention. It must notice danger (face covered, on stomach, head turned left too long), sleep events (crying, awake during a nap), and its own blindness (camera offline). It must stay quiet otherwise. Inputs arrive at four very different rates: video at 4 fps, audio at 1 Hz, a cloud vision verdict about once a minute, and operator commands whenever. Several of them must be fused before a rule can fire. "Two consecutive checks" and "more than 2 minutes" are persistence rules, so time is part of the logic. The host may be a GPU-less NAS that crashes or restarts. Cooldowns, nap mode and the ring buffer must come back correct without a human. The decision step may later move to Jev, a text/JSON service, so it cannot depend on in-process history. The measured facts constrain the media path. One ffmpeg can emit raw gray frames and 16 kHz PCM. MP4 segments need audio transcoded to AAC. YAMNet ONNX runs on CPU, and its classes 19 and 20 are the cry classes. The service must also run without the Anthropic or Pushover keys, with vision and push shown as disabled.

## Usage (caller's view)

Operator:

```
$ cp .env.example .env    # BABYMON_CAMERA__USER, __PASSWORD; optional ANTHROPIC_API_KEY, BABYMON_PUSHOVER__*
$ uv run babymon run --nap          # dashboard on http://nas:8080
$ uv run babymon probe camera       # 30 s health check of the one ffmpeg session
$ uv run babymon probe vision crib.jpg --reference good.jpg
$ uv run babymon replay last-night.jsonl   # recorded facts -> effects, offline
```

Call site 1. A test of the decision logic needs no camera, network, or clock:

```python
t0, sec = datetime(2026, 10, 27, 2, 0, tzinfo=UTC), timedelta(seconds=1)
state = brain.initial_state(
    Restored(Controls(armed=True), None, {}), Capabilities(vision=True, push=True), t0
)
for i, obs in enumerate([stomach, stomach]):
    state, _ = brain.observe(
        state,
        VisionDone(t0 + 15 * i * sec, VisionReason.PERIODIC, obs, SnapshotRef("x.jpg")),
        tuning,
    )
state, _ = brain.observe(state, Tick(t0 + 31 * sec), tuning)
state, notes = brain.gate(
    state, RulesDecider(tuning).evaluate(brain.situation(state, tuning)), tuning
)
assert [(n.alert.kind, n.alert.priority) for n in notes] == [
    (AlertKind.ON_STOMACH, Priority.EMERGENCY)
]
```

Call site 2. The brain task in `app.py`, the only place the async decider meets the pure core:

```python
fact = await facts.get()
state, effects = brain.observe(state, fact, tuning)
if isinstance(fact, Tick):
    s = brain.situation(state, tuning)
    state, notifies = brain.gate(state, await decider.decide(s), tuning)
    published.set(s)
for e in (*effects, *notifies):
    dispatch(e, events, alerts, vision_requests)
```

Call site 3. Swapping a backend is a constructor change in `app.run`, and nothing else imports the SDK:

```python
vision = (
    ClaudeVision(settings.vision.api_key.get_secret_value(), settings.vision.model)
    if settings.vision
    else None
)
decider: Decider = RulesDecider(
    settings.tuning
)  # later: JevDecider(..., fallback=RulesDecider(...))
```

## Shape

**Data first.** `domain.py` holds the whole vocabulary. Facts flow in (`MotionSample`, `AudioSample`, `VisionDone`, `VisionFailed`, `CameraStatus`, `SetControls`, `Tick`). Effects flow out (`RequestVision`, `Notify`, `Record`). Events are persisted (`SleepChanged`, `VisionObserved`, `AlertSent`, `ControlsChanged`, `CameraChanged`, a pydantic discriminated union). `Situation` is the JSON contract read by both the decider and the dashboard. `Observation` uses StrEnums with explicit `UNKNOWN` members, so "the model could not tell" is a value and not a missing field (per type-system-discipline).

**Functional core, three pure functions.** `brain.observe(state, fact) -> (state, effects)` folds every fact. `brain.situation(state) -> Situation` projects. `brain.gate(state, verdicts) -> (state, notifies)` applies policy. Time comes only from `fact.at`. A 1 Hz `Tick` is the one fact that advances time-based rules, so a silent camera still produces `CAMERA_OFFLINE` instead of freezing the logic. The async `Decider` call sits between `situation` and `gate` in the shell. The core stays pure even after the decider becomes a network call (per boundary-discipline).

**Three tables instead of branch chains** (per model-the-domain).
- `sleep.TRANSITIONS[state][signal] -> Edge(to, dwell_s)`. Evidence is first reduced to one `Signal` by fixed precedence (CRYING > ACTIVE > EYES_OPEN > MOVING > QUIET). Dwell is tracked per target state, so mixed MOVING and ACTIVE seconds both count toward AWAKE. The unknown start state is STIRRING, which no rule alerts on. A restart therefore cannot fake an "awake during nap".
- `rules.RULES`, one `Rule(kind, holds, reason)` row per alert condition. The decider answers only "which conditions hold". It sees precomputed streaks and durations (`face_covered_streak`, `head_left_for_s`, `sleep_for_s`, `camera_down_for_s`). A decider is therefore a stateless function of one JSON value, which is exactly the shape Jev accepts.
- `brain.ALERT_POLICY[kind] -> (priority, title)`, with cooldown in `Tuning`. Priority and cooldown are policy, not decision, so swapping the decider cannot change how loud an alert is. An import-time check fails if the table misses an `AlertKind`.

**Streak semantics live in one place.** `VisionTrack` owns them. A confident in-crib reading increments or resets each streak. A low-confidence reading or a failed call leaves the streaks unchanged, because it is evidence of neither danger nor safety. `in_crib=False` resets everything, because a baby out of the crib is with a parent.

**Vision scheduling is in the core.** It is a pure decision over `VisionTrack`. Periodic checks run every 60 s. While a dangerous streak is pending they run every 15 s (`CONFIRM`), which brings "two consecutive checks" down from about 2 minutes to about 30 s. A check also runs when motion *ends* (`SETTLED_AFTER_MOTION`, at least 10 s apart), so the model sees the new resting position instead of motion blur. An in-flight request older than 45 s counts as lost.

**Ownership and channels** (per separate-before-serializing-shared-state). Each actor owns its own state. Camera owns ffmpeg, the pipes and `MotionMeter`. Audio owns the classifier and `NoiseFloor`. Vision owns its one in-flight call. Brain is the sole owner of `MonitorState`. Alerter owns delivery. Store owns the disk. Actors talk through exactly two channel kinds. The first is an ordered `asyncio.Queue`. One queue of facts runs into the brain, with many writers appending immutable facts. Two more run out of it, events to the writer and alerts to the alerter. The second is `Latest[T]`, a single-writer newest-wins cell, used for stills, PCM windows, vision requests and the published `Situation`. The web app never mutates state. A toggle becomes a `SetControls` patch fact, the same path the CLI flag uses. Patches, not full replacements, mean two phones toggling different switches cannot clobber each other.

**One ffmpeg, four outputs.** A single RTSP session on stream2 writes MP4 segments (video copied, audio AAC, per the measured recipe) and three pipes passed with `pass_fds`. The pipes carry 640x360 gray at 4 fps for motion, 16 kHz s16le for audio, and 1280x720 MJPEG at 1 fps as `Still`s. ffmpeg does all decoding and JPEG encoding. Python never decodes video, and every consumer (vision, dashboard, snapshots, Pushover attachment) shares one `Still` type. Pipe readers never await consumers, so a slow model call cannot back-pressure ffmpeg and stall recording. Segments use fragmented MP4 with UTC strftime names, so a segment still being written, or one cut short by a crash, plays in the dashboard.

**Restart convergence** (per make-operations-idempotent). Each item is derived from disk, not remembered.
- ffmpeg: each supervisor attempt starts from nothing (new pipes, process, meter). A stall watchdog applies, plus capped backoff.
- Ring buffer: file names are timestamps, so the directory listing is the index. `Store.sweep(now)` deletes what is older than 24 h. It runs at startup and every 5 minutes, and running it twice is a no-op.
- Cooldowns: `Store.restore` rebuilds the ledger from `AlertSent` rows with `delivered=True` only. If the process crashed between deciding and delivering, it re-alerts instead of staying silent.
- Controls: `ControlsChanged` events upsert a never-pruned controls row in the same transaction. The log and the row cannot drift.
- Sleep: the last `SleepChanged` resumes if it is under 10 minutes old. Otherwise STIRRING (see above).
- Reference snapshot: one file, written by tmp+rename.

**Failure policy.** Actors handle their expected failures. An unexpected exception ends the `TaskGroup` and the process, and Docker restarts it. Startup is the recovery path, so it is the path the tests exercise.

**Interface depth.** The public surface is small: `Camera.run`, `run_audio`, `run_vision`, `run_alerter`, the three brain functions, `Decider.decide`, `VisionBackend.observe`, `Notifier.deliver`, `AudioClassifier.cry_probability`, and `Store`. Behind `Camera.run` sit ffmpeg flags, fd plumbing, JPEG framing, reconnect and recording. Behind `Store` sit the SQLite schema, the file layout, retention, clip lookup and recovery. Behind `ClaudeVision` sit the SDK, the wire schema and the prompt. No SDK, HTTP, SQLite or ffmpeg type appears in a signature outside its own module (per boundary-discipline). An `import-linter` forbidden contract keeps `domain`, `sleep`, `rules` and `brain` free of `anthropic`, `httpx`, `fastapi`, `sqlite3`, `subprocess`, `numpy` and `asyncio`. The build, not a comment, enforces purity (per encode-lessons-in-structure).

**Deliberately not done.** No breathing detection. No local GPU model. No schedule for nap mode. No auth. No per-frame alerting. No segment index table.

## Synthesis decision

Four candidates (Opus, Fable, Sonnet, Haiku models) produced designs against the same brief. All four converged on the same core: a pure decision step that takes timestamped signals and returns effects, one inbox queue with one consumer that owns the monitor state, tables for sleep transitions and alert policy, a decider that reports which conditions hold while a policy table owns priority, and cooldowns rebuilt from delivered alerts. That convergence is the design.

The Opus candidate is the base. It was the only one consistent with every measured fact (AAC segments, YAMNet ONNX, startup without keys), it keeps the sleep machine in the core so a replacement decider only answers conditions, it projects a precomputed `Situation` as the JSON seam, and it starts the sleep machine in STIRRING so a restart cannot fake an awake-during-nap alert. A read-only cross-judge on a different model scored it 17 of 18 against 14, 13 and 9 for the others and recommended the same base.

Grafts applied to the base:

- Recorder as its own ffmpeg process (Fable, Sonnet, Haiku). Measured on 2026-10-27: the C210 serves three concurrent RTSP sessions on stream2 with no dropped frames, so the one-process argument fell away and failure isolation wins. `camera.py` keeps three pipes (gray frames, PCM, MJPEG stills); `recorder.py` owns the segment process; `store.py` still owns the segment directory and retention.
- A `DeliveryResult` fact from the alerter (Sonnet). On a failed delivery the brain removes the ledger entry, so the next Tick re-fires instead of waiting out the cooldown. Restore from disk already behaves this way; this makes the in-process path match.
- An instance lock (Sonnet). A `flock` on the data directory makes a second process exit instead of recording and alerting twice.
- Exhaustiveness and round-trip tests (Fable). Every `AlertKind` has exactly one rule and one policy row; `Situation` round-trips through JSON.
- Fact recording (Fable). `babymon run --record-facts FILE` appends every fact as JSONL so a real night becomes a `replay` fixture.
- `Restored` moves from `brain.py` to `domain.py` (cross-judge), so the store does not import the core.
- `AWAKE_DURING_NAP` holds only in AWAKE, not CRYING (cross-judge), so it no longer overlaps the CRYING alert.
- Priority as a domain `StrEnum` (NORMAL, HIGH, EMERGENCY) instead of Pushover integers (Fable). `notify.py` maps to -2..2 and adds retry and expire for EMERGENCY.

Rejected:

- A `Delivery` union carrying `retry_s` and `expire_s` in the domain (Sonnet, cross-judge). Those are Pushover transport parameters. Keeping them in `notify.py` is the boundary the design wants; the Pushover config requires both, which is the same guarantee one layer out.
- Deriving streaks from a pruned observation window (Fable, cross-judge). The counters in `VisionTrack` have one writer and one rule set (confident in-crib readings move them, low confidence leaves them, out of crib resets them). A window would re-encode the same rules in a derivation.
- `Persistence` objects per rule row (Haiku). Thresholds already live in `Tuning` and the rows read precomputed durations; another layer adds no deleted branch.
- `UnionDecider` with a rules floor (Sonnet). Right shape for the day Jev arrives, not needed while rules are the only decider. The `Decider` protocol already fits it.
- An `UNKNOWN` sleep state (Sonnet). The timeline derives gaps from `CameraChanged` events and the STIRRING start; a fifth state would add rows to every table for a display concern.

## Tradeoffs accepted

- We accept one ffmpeg process for recording and analysis in exchange for a single camera connection and a single reconnect point. A stall in any output restarts all of them, so recording gaps equal analysis gaps. Splitting into two processes is a change local to `camera.py` if the Tapo tolerates two sessions.
- We accept a 1 Hz decision cadence in exchange for a clock-free core. Alerts can lag their condition by up to 1 s.
- We accept that the brain holds the decider call on each Tick. With rules this costs microseconds. A slow remote decider would stall fact processing, and that is the moment to move it behind a `Latest` like vision (see open questions).
- We accept losing in-memory streaks and `head_left_since` on restart in exchange for a smaller persisted surface. Worst case, one restart delays an emergency by one confirm interval (about 15 s).
- We accept up to 4 Claude calls a minute while danger is pending in exchange for confirming an emergency in about 30 s instead of about 2 minutes.
- We accept sequential alert delivery. A Pushover outage retrying one alert for 60 s delays the next. Alerts are rare, and order matters more than parallelism.
- We accept that `Situation` duplicates fields that exist in `MonitorState` in another form. It is derived on every Tick and never stored, so it cannot drift.

## Alternatives considered

- **Per-actor state machines with merge at read time.** Audio would own crying, vision would own position streaks, and an alert policy would poll each. This loses because the rules span actors: awake-during-nap needs sleep state and controls, and crying needs audio durations together with sleep. Every rule would then re-implement time alignment across sources. The interface looks smaller per module, but callers must learn each module's timing internals.
- **Event-sourced brain, replay the log on restart.** Persisting every fact gives perfect recovery and free replay. It loses on volume: about 350k motion samples a day get written to SQLite to recover a few values that `Store.restore` derives from about five rows. The `replay` CLI keeps the debugging benefit without the storage cost.
- **Separate ffmpeg per concern** (recorder, analysis, and an on-demand 1080p still grab per vision call). Each process would be simpler, but it opens two or three RTSP sessions and adds a 2 to 3 s keyframe wait per vision call. It also moves "which frame did we judge" into a second process. Its interface is the same, so it remains the fallback inside `camera.py`.
- **Decider returns Alerts (priority, cooldown included).** This gives Jev more control. But a replacement decider could then silently downgrade an emergency, and Jev would need cooldown memory, which a stateless JSON contract cannot hold.

## Implementation reconciliation

- `VisionTrack.pending_danger` is a method `pending_danger(at, tuning)`, not a property. "Started but not yet alerting" needs `unsafe_streak` and `head_left_alert_s` from `Tuning`, and the clock for the head-left duration. It covers both position streaks and a head-left run under the threshold, so CONFIRM cadence applies to all three.
- `MonitorState.cry_probability: float` became `audio: AudioSample | None`. The 2 s decay needs the sample's timestamp; keeping the fact is one field instead of two.
- `MonitorState.at` moves only on Tick (and at startup). A Tick with `at <= state.at` is a no-op, which is what makes a duplicated Tick harmless (motion_peak would otherwise reset to QUIET and drop a pending transition). The ticker must emit strictly increasing instants and the first Tick must be later than the `initial_state` instant.
- `DeliveryResult(delivered=False)` clears the ledger entry only when `capabilities.push` is true. With push disabled, `LogOnly` fails every delivery, and clearing would re-fire every second and write an `AlertSent` row per second. With push off the cooldown stands, so the log shows one undelivered row per cooldown.
- `head_left_for_s` is `state.at - head_left_since`, wall time since the first LEFT reading, as long as every confident check since said LEFT. It is not the time between the first and the latest confirming check, so with 60 s periodic checks the alert fires at about 121 s rather than 180 s.
- The first Tick after startup schedules a PERIODIC vision check (no attempt yet, so the period has elapsed). `VisionTrack` fields got defaults so `VisionTrack()` is the empty track.
- `Settings` uses `extra="ignore"` so unrelated variables in `.env` do not fail startup. `pushover.user_keys` and `camera.motion_roi` parse comma-separated strings (`u1,u2`, `x,y,w,h`) instead of JSON. `python-dotenv` is a direct dependency: the `ANTHROPIC_API_KEY` alias is a settings source placed below env and dotenv, so an explicit `BABYMON_VISION__API_KEY` wins.
- `config.load` reports a missing required group (the whole camera block) as each variable the group needs, not as `BABYMON_CAMERA`.
- `notify.run_alerter` publishes a `DeliveryResult` fact after each attempt (docstring only; body is still the sketch).

## Open questions and risks

- Is "head to the left" the baby's left or the image's left? The sketch assumes the baby's left and the prompt will say so. The reference image should be taken with the head in the allowed position so the model can compare.
- Should `in_crib=False` suppress position alerts (the current default, on the reasoning that the baby is with a parent) or raise its own alert?
- Should an emergency condition that clears and returns inside the 10-minute cooldown alert again immediately? The brief says cooldown per type. For FACE_COVERED, a bypass when the condition cleared in between may be safer.
- When Jev arrives, should the hard rules stay as a local floor (`JevDecider(fallback=RulesDecider)` with verdicts unioned), so a Jev outage or a wrong answer can never suppress an emergency? This design assumes yes.
- Does the C210 accept a second RTSP session? It is unmeasured. The answer decides whether the one-process tradeoff is necessary or merely tidy.
- Vision cost and latency: about 1,440 Opus calls a day with two images each. Is that acceptable, or should the periodic interval stretch while ASLEEP and quiet?
- Fragmented-MP4 segment flags combined with the multi-output command are inferred, not measured. Unit 5 measures them.
- The motion ROI needs a one-time setup. Should the dashboard offer a rectangle picker, or is an env var enough for v1?

## Next implementation step

Write `domain.py`, `sleep.py`, `rules.py` and `brain.py` with table-driven pytest scripts for each hard rule, because every later unit only feeds facts into, or executes effects out of, that core.
