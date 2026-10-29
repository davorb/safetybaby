# babymon

A self-hosted baby monitor for a TP-Link Tapo C210. It watches the camera's RTSP stream, scores motion and crying on the host, asks Claude about the baby's position about once a minute while there is motion, records 24 hours of video, and sends a push notification only when a rule holds. It runs without the Claude and Pushover keys too. Vision and push are then shown as disabled and everything else works.

## What you need

- A Tapo C210 with a camera account (Tapo app, **Advanced settings**, **Camera account**) and a fixed IP on your LAN.
- `ffmpeg` on the host, or Docker.
- Python 3.12 with [uv](https://docs.astral.sh/uv/), or Docker.
- Optional: an Anthropic API key for the position checks.
- Optional: a Pushover application token and the user key of each phone that gets alerts.

## Run it with uv

1. Copy the example settings and fill in the camera account:

   ```
   cp .env.example .env
   ```

   `BABYMON_CAMERA__USER` and `BABYMON_CAMERA__PASSWORD` are required. Set `BABYMON_CAMERA__HOST` if the camera is not at `192.168.68.75`. Add `ANTHROPIC_API_KEY` to enable vision and `BABYMON_PUSHOVER__APP_TOKEN` plus `BABYMON_PUSHOVER__USER_KEYS` to enable push. Every other variable has a default that `.env.example` lists.

2. Download the cry classifier (16 MB) once:

   ```
   sh scripts/fetch-yamnet.sh
   ```

   It lands in `models/`. Set `BABYMON_YAMNET_PATH=./models/yamnet.onnx` and `BABYMON_DATA_DIR=./data` in `.env` to keep the model and the recordings inside the checkout.

3. Start the service:

   ```
   uv run babymon run
   ```

   The dashboard is on http://localhost:8080. The log's first lines say whether vision and push are enabled. `uv run babymon run --nap` starts with nap mode on and `--disarm` starts silenced. Without flags the switches keep their last persisted values.

To flip a switch while it runs, use the dashboard checkboxes or `curl -X POST localhost:8080/api/controls -H 'content-type: application/json' -d '{"nap_mode": true}'`. Both switches survive a restart.

## Other commands

| Command | What it does |
| --- | --- |
| `uv run babymon probe camera --seconds 30` | Runs the camera session and a recorder for 30 s. Prints frames per second, audio windows per second, stills per second, the motion level range and the segment files that appeared. |
| `uv run babymon probe audio clip.wav` | Prints loudness above the noise floor and P(cry) for each second of the clip. |
| `uv run babymon probe vision frame.jpg --reference good.jpg` | Prints the typed observation Claude returns for one frame. Exits with "vision disabled" without an API key. |
| `uv run babymon probe alert --priority high` | Sends one test notification with a fresh camera frame attached. Priorities are `normal`, `high` and `emergency`. |
| `uv run babymon reference` | Stores the current frame as the reference image of the correct position. The dashboard's **Set reference** button does the same. |
| `uv run babymon run --record-facts night.jsonl` | Runs the service and appends every fact the decision core receives to the file. |
| `uv run babymon replay night.jsonl` | Feeds a recorded file through the decision core offline and prints the effects it produces, one per line. |

## Run it with Docker

The image holds ffmpeg, the Python dependencies and the YAMNet model. `compose.yaml` reads `.env`, keeps the recordings in `./data`, publishes port 8080 and restarts the container unless you stop it.

```
docker compose up -d --build
docker compose logs -f
```

`BABYMON_DATA_DIR` and `BABYMON_YAMNET_PATH` in `.env` are ignored inside the container. The compose file sets them to `/data` and `/models/yamnet.onnx`. A one-off command runs the same image: `docker compose run --rm babymon probe camera`.

## Alerts

Each rule fires at most once per `BABYMON_TUNING__ALERT_COOLDOWN_S` (default 600 s) per kind, and only while armed. A condition that still holds after the cooldown fires again. Crying is the exception: a new crying episode alerts at once even inside the cooldown, and the cooldown only limits repeats within one episode. A delivery that fails clears the cooldown so the next second tries again. The thresholds below are the defaults in `.env.example`.

| Alert | Priority | Fires when |
| --- | --- | --- |
| Face covered | emergency | The nose or mouth is covered on 2 consecutive confident checks. |
| Baby on stomach | emergency | The baby is face down on 2 consecutive confident checks. |
| Head turned left | high | Every confident check for more than 120 s has the head turned to the baby's left. |
| Crying | high | The baby has cried for 3 s (`BABYMON_TUNING__CRYING_ALERT_S`). |
| Awake | normal | Nap mode is on and the sleep state is awake. |
| Camera offline | high | No video for more than 60 s, including right after a start. |

Pushover shows `emergency` with its own retry and expire (every 30 s for 10 min until acknowledged), `high` as a high-priority message and `normal` as a normal one. Each notification carries the latest frame. Without Pushover keys the alert is logged and recorded with `delivered=false`.

A check counts as confident when Claude's confidence is at least `BABYMON_TUNING__MIN_CONFIDENCE` (0.5). A low-confidence reading changes nothing, and a reading of the baby out of the crib resets every streak. While a dangerous reading waits for its second check, checks run every 15 s instead of every 60 s.

## The dashboard

The page at port 8080 shows the latest frame (refreshed every 2 seconds), the current sleep state and how long it has held, the **Armed** and **Nap mode** switches, a **Set reference** button, a 24-hour bar of sleep states and the event list for the same window. Each vision check and alert in the list links to its snapshot, and each event links to the recorded clip at the event's offset. Banners at the top say when the camera is down or when vision or push is disabled.

The sleep states are asleep, stirring, awake and crying. A fresh start begins in stirring, which no rule alerts on, so a restart cannot fake an "awake during nap" alert.

## Statistics

`/stats` shows sleep statistics for the last 1 to 90 days, and `GET /api/stats?days=N&tz=<IANA zone>` returns the same numbers as JSON. Pass the browser's time zone as `tz`, because the container runs in UTC. A sleep day runs from 07:00 to 07:00 local time and the night is 19:00 to 07:00. Event rows and one summary row per minute of motion and sound are kept for 90 days, so the history reaches back to the first start after the upgrade. Every block reports how much time it covers. Time when the camera was down or the process was not running counts as unknown, never as the last sleep state.

## End-to-end tests

`e2e/` holds browser tests for the dashboard, written with [Libretto](https://github.com/davorb/libretto). Each spec in `e2e/specs/` describes a page's behavior in markdown. Libretto turns it into the Playwright test under `e2e/.libretto/generated/`, which is committed. `e2e/specs/app.md` describes the pages and the fixed data the tests rely on.

The tests run against `scripts/e2e_server.py`, which serves the real dashboard over 21 days of synthetic data plus a few fixed recent events. It needs no camera and no keys. Libretto is linked from a checkout beside this repository (`../libretto`), so build it there first with `pnpm install && pnpm build`.

```
cd e2e && pnpm install && pnpm exec playwright install chromium
pnpm app &          # the e2e server on http://127.0.0.1:8090
pnpm test           # plain Playwright, no AI
```

Running the tests spends no AI tokens. After you change a spec, or change the dashboard on purpose and update `app.md`, run `pnpm generate`. It uses your `claude` login or `ANTHROPIC_API_KEY` and rewrites only the tests whose spec or `app.md` changed. When a test fails, `pnpm triage` says whether the dashboard or the test is wrong, and `pnpm fix` offers to update the spec and regenerate it.

## Vision cost

Each check sends up to two JPEG images, the reference and the current frame, downscaled to `BABYMON_VISION__API_FRAME_WIDTH` (640 px) wide. The periodic check runs every `BABYMON_TUNING__VISION_PERIOD_S` (60 s) only when motion was seen since the last check. On a still scene it runs every `BABYMON_TUNING__VISION_PERIOD_QUIET_S` (900 s). That is at most about 1,440 calls a day and about 96 on a quiet night. A check also runs when motion settles. `BABYMON_VISION__MODEL` picks the model (default `claude-haiku-5-5`). On Haiku 5.5 a quiet night costs on the order of a dollar or two a month. `claude-opus-5-5` costs roughly fifty times more per call, and the original default of one full-size check per minute on Opus came to about a thousand dollars a month, so change the model with care. A longer period delays "face covered" and "on stomach", which need two consecutive checks, by the same amount. While a dangerous reading is pending, checks run every `BABYMON_TUNING__VISION_CONFIRM_S` (15 s) regardless.

## What you must supply

- The camera account: `BABYMON_CAMERA__USER` and `BABYMON_CAMERA__PASSWORD`.
- `ANTHROPIC_API_KEY` for vision. Without it the position rules never fire.
- `BABYMON_PUSHOVER__APP_TOKEN` and `BABYMON_PUSHOVER__USER_KEYS` (comma-separated, one per phone) for push. Without them nothing reaches a phone.
- A reference image of the correct position, taken with `babymon reference` or the dashboard button, so the model can compare.

## What is on disk

`BABYMON_DATA_DIR` holds `babymon.sqlite3` (events, per-minute signal rollups and the switches), `segments/` (5-minute MP4 files named by their UTC start), `snapshots/` (the frame behind each vision check and alert) and `reference.jpg`. Segments and snapshots older than 24 hours, and events and minute rollups older than 90 days, are deleted at start and every 5 minutes. The switches and the reference image are kept. A second `babymon run` on the same directory exits at once, so one data directory serves one process.
