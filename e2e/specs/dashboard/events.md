# Events

## Scenario: recent events are listed newest first
Steps:
- Open the live page

Expect:
- The events list contains the alert titled "Crying" with the message "E2E crying for 4 s", a "high" badge and a "not delivered" badge
- The events list contains a vision row showing "FACE COVERED" and "in parents' bed"
- The "Crying" alert row comes before that vision row
