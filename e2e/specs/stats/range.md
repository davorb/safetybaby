# Statistics

## Scenario: the statistics page opens from the live page
Steps:
- Open the live page
- Click the "Statistics" link

Expect:
- The URL is `/stats` and the heading "Statistics" is shown
- The "7 d" range button is pressed

## Scenario: the range selector switches the time range
Steps:
- Open the statistics page
- Click "30 d"

Expect:
- The "30 d" range button is pressed and "7 d" is not
- The "Data coverage" and "Sleep per day" sections are shown
