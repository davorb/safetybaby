# babymon dashboard

A baby monitor's web dashboard. In these tests it runs from `scripts/e2e_server.py`: the real pages and API over synthetic data, with no camera and no AI behind it. The server rebuilds its data on every start.

## Live page (`/`)

The page title is "babymon". It polls `/api/state` every 2 s.

- Banners at the top. With the test server, vision and push are both disabled, so two warning banners show: "Vision is disabled: no Anthropic key. Position alerts are off." and "Push is disabled: alerts are logged, not delivered." The camera is up, so there is no camera banner.
- The live frame, an image with alt text "latest frame", refreshed every 2 s, and a "Full screen" button.
- A sleep-state pill (a button whose text is the state, e.g. "asleep") followed by how long, e.g. "for 25 min". The test server always reports asleep for 25 min.
- Two switches, checkboxes labelled "Armed" and "Nap mode". Armed starts on, Nap mode starts off. Changing one POSTs to `/api/controls`; the page holds the new value for 3 s and then re-reads the state. Below them a hint line starts with "Nap mode is off:" or "Nap mode is on:". The switches are shared server state: a test that changes one must set it back.
- A "Set reference" button. It stores the current frame and shows "Reference saved." next to it for 4 s.
- A "Statistics" link to `/stats`.
- "Last 24 hours": a coloured timeline bar with a legend of the states asleep, stirring, awake, crying, unknown.
- "Events": newest first, last 24 hours. Each row shows a time, a kind (Sleep, Vision, Alert, Controls, Camera) and details. Alert rows carry a priority badge ("high", "emergency", "normal") and, when undelivered, a "not delivered" badge. Vision rows list the observed facts, e.g. "stomach", "head left", "eyes closed", "in crib". The test server always adds, within the last hour: an undelivered high-priority alert titled "Crying" with the message "E2E crying for 4 s", a vision check showing the baby on the stomach in the parents' bed with the face covered ("stomach", "head left", "in parents' bed", "FACE COVERED"; it is the only vision row with both "FACE COVERED" and "in parents' bed"), and a sleep change "stirring to asleep". The list refreshes only every 30 s.

## Statistics page (`/stats`)

Heading "Statistics". A "Time range" button group with "24 h", "7 d" (pressed by default), "30 d" and "90 d"; the pressed one has `aria-pressed="true"`. A "Live" link back to `/`. Sections with headings such as "Data coverage", "At a glance", "Sleep per day", "Crying", "Alerts" and "Camera". The test server holds 21 days of data, so "30 d" and "90 d" cover more time than there is data.
