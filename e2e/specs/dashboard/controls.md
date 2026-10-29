# Controls

## Scenario: nap mode survives a reload and can be switched back off
Steps:
- Open the live page
- Turn on the "Nap mode" switch; the hint changes to "Nap mode is on"
- Reload the page; the "Nap mode" switch is still on
- Turn off the "Nap mode" switch; the hint changes to "Nap mode is off"
- Reload the page

Expect:
- The "Nap mode" switch is off
- The "Armed" switch is on

## Scenario: set reference saves the current frame
Steps:
- Open the live page
- Click "Set reference"

Expect:
- The note "Reference saved." appears
