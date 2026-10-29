---
tags: [smoke]
---
# Live status

## Scenario: the live page shows the monitor's state
Steps:
- Open the live page

Expect:
- The page title is "babymon"
- The vision-disabled and push-disabled banners are shown
- The sleep state reads "asleep" and the duration reads "for 25 min"
- The live frame image is shown
