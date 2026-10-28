# babymon for Apple TV

A SwiftUI tvOS app for the babymon service: ambient monitor, alert takeover with sound, controls dock, history with clips, setup. It follows the "babymon for Apple TV" design and polls the same JSON routes as the web page.

Open `BabymonTV.xcodeproj` in Xcode and run on an Apple TV or a tvOS simulator. On first launch enter the server address (for example `http://192.168.1.24:8080`) or use the port 8080 scan.

## Alert sounds

`BabymonTV/Resources/Sounds/*.wav` are synthesized by `scripts/gen_sounds.py`, so no third-party audio is in the repo. Run `python3 tvos/scripts/gen_sounds.py` to regenerate them.

## Trying it without a camera

```
python3 tvos/scripts/mock_server.py some-frame.jpg
curl 'localhost:8080/mock?scenario=emergency'   # calm, pending, high, emergency, offline, starting
xcrun simctl launch booted se.davor.babymontv -serverURL http://127.0.0.1:8080
```

## Server change

`GET /api/events` accepts `since_id=N` and returns only events with a higher id. The TV uses it to fetch just what is new.
