# Measured facts (orchestrator, 2026-10-27)

All verified by running the command on this machine against the live camera.

## ffmpeg recipes that work

Frames + audio from one process (stream2, TCP). 4 fps 640x360 8-bit gray raw frames, and 16 kHz s16le mono PCM:

```
ffmpeg -v error -rtsp_transport tcp -i "$RTSP_URL" \
  -map 0:v -vf "fps=4,scale=640:360,format=gray" -f rawvideo -pix_fmt gray <video sink> \
  -map 0:a -ac 1 -ar 16000 -f s16le <audio sink>
```
Measured: 24 frames in 6 s of video, audio bytes = 32000 per second. Frame size = 640*360 = 230400 bytes. For a long-running process send video to `pipe:1` and audio to a named FIFO (or run two ffmpeg processes).

Recorder. MP4 segments cannot carry the camera's `pcm_alaw` audio (ffmpeg refuses to write the header). Transcode audio to AAC, copy video:

```
ffmpeg -v error -rtsp_transport tcp -i "$RTSP_URL" -map 0 -c:v copy -c:a aac -b:a 32k \
  -f segment -segment_time 300 -reset_timestamps 1 -strftime 1 "clips/%Y%m%d-%H%M%S.mp4"
```
Measured with `-segment_time 3`: three playable files, durations 4.36 s, 2.26 s, 1.78 s (boundaries snap to keyframes, so segment length is approximate).

## Audio classifier

YAMNet as ONNX runs on CPU with onnxruntime 1.31.1.
- Download: https://huggingface.co/audiomagic/yamnet-onnx/resolve/main/yamnet.onnx (16 MB) and `yamnet_class_map.csv` from the same repo.
- Input: `waveform`, float32, 1-D, 16 kHz mono, range -1..1. Output `output_0` has shape (frames, 521) with one row per 0.48 s patch (hop 0.48 s, window 0.96 s); 1 s of audio gives 2 rows.
- Class indices: 19 "Crying, sobbing", 20 "Baby cry, infant cry", 14 "Baby laughter".
- Copies already downloaded at `/tmp/claude-1000/-home-davor-orca-workspaces-safety-baby-butterflyray/f0ccd243-4a62-41ae-a570-a858e65419bc/scratchpad/models/`.

## Environment

- Python 3.12.14 is installed through uv (`uv venv -p 3.12`). The system python is 3.14; pin 3.12 for wheel availability.
- These install on 3.12: onnxruntime 1.31.1, anthropic 2.3.0, opencv-python-headless 5.0.0, numpy, pydantic, fastapi, uvicorn, httpx.
- No `ANTHROPIC_API_KEY` or Pushover variables are set in this shell. The user must supply them. Build so that the service starts and runs ingest, motion, audio, recorder, and dashboard without them, and reports clearly that vision and alerts are disabled until the keys exist.
- Docker is installed on the desktop. The NAS runs Docker, Linux, no GPU.
