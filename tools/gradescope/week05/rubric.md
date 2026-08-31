# Week 5 — hand-marked question (25 points)

**Do earthquakes cluster — or is that just what randomness looks like?**

The autograder scores the two code parts. This is the part it cannot: mark it in Gradescope as a manual rubric item worth 25, on top of the autograder's score.

## What the student was asked

### ✏️ Your turn 8

You now have three numbers for the same question and, under each of them, the earthquakes they were
built from and the gaps between those earthquakes. A few sentences on each of these, using **your
own printed output**:

1. Which of the three four-year numbers would you quote, and what does the wider box buy and what
   does it cost? Name at least one earthquake it adds.
2. The Poisson formula assumes events arrive independently at a steady rate. Quote your shortest
   gap and your recurrence interval — and then be careful, because class showed that a world
   without clustering is already bunched, so one short gap on its own proves nothing. Is there
   anything in your two lists of gaps that chance alone would struggle to produce?
3. Say in which direction you distrust the number you quoted, and why. `p1` and `p3` differ only
   in where the record starts, so your own list of events has something to say about this.

## Criteria

| Points | A full-credit answer does this |
|---:|---|
| 6 | Names which of p1, p2 and p3 they would quote and gives its four-year percentage together with the recurrence interval and event count behind it, then states both sides of the wider box: the extra events it buys, and the cost. Names at least one specific earthquake from the printed table that only the two-degree box contains, such as 1901 Parkfield or 1983 Coalinga. Any of the three numbers can be the right one to quote if the trade is argued. |
| 6 | Quotes their own shortest gap in days against their recurrence interval in years, and concedes in the same breath that one short gap proves nothing, on the ground class established that a world without clustering is already bunched. An answer that treats a single short gap as proof of clustering does not earn this item. |
| 6 | Points at something in the two printed gap lists that chance would genuinely struggle to produce: the two-degree box's 0-day gap, two magnitude-6 events inside an hour against a 10.5-year average wait, or that gap set beside the 21,004-day silence, since a fixed total forces the crowded intervals and the empty ones to come together. |
| 7 | Names a direction of distrust and the reason for it, from their own list of events: aftershocks inflating the rate above the rate at which independent earthquakes begin, the whole estimate resting on 8 events in 126 years, or the events clumping into 1903-1926 so that moving the start date to 1930 gives 12% rather than 22% on a choice nothing in the data makes. |

## The model answer

Not a template to match word for word — a student who reaches the same conclusions from their own numbers, in their own words, has answered it.

Of my three numbers I would quote the one-degree box over the whole record:
22% in four years, from 8 earthquakes with a recurrence
interval of 15.8 years.
Widening to two degrees raises it to 32% and shortens the recurrence to
10.5 years, but it buys those extra 4 events by
reaching two degrees away: the 1901 Parkfield and 1983 Coalinga earthquakes are both in the
wider box, and shaking falls off with distance, so an earthquake that far from campus is not the
same hazard as one underneath it. More data is not automatically better data when the extra data
is answering a different question.

My shortest gap in the one-degree box is 52 days against a recurrence
interval of 15.8 years, and on its own that proves nothing — class's random
world produced a day holding 8 earthquakes without any clustering at all, and one
short gap in seven is exactly the sort of thing chance throws up. The two-degree box is a different
matter. Its shortest gap is 0 days, because two magnitude-6 earthquakes
happened within an hour of each other on the same date, and a process delivering one event every
10.5 years on average does not put two of them in one hour by luck. Then the
record swings the other way and the one-degree box waits 21,004 days,
57.5 years, for the next one. Class showed that the two go
together: piling events into a few short intervals has to leave the others emptier, because the
total is fixed. That is the same trade that gave the global catalogue 550 more
silent days than chance allows at one end and a day holding 128 at the other.

Which direction do I distrust it in? Both, which is the honest answer. The count includes
aftershocks, so the rate is higher than the rate at which *independent* earthquakes begin, and that
pushes the number up; the whole estimate rests on 8 events in 126 years, so
one event more or fewer moves it by several points. And the events are not spread evenly through
the record: 5 of my 8 fall between 1903 and
1926, and the same one-degree box started in 1930 instead holds
3 events in 96 years and gives 12% rather
than 22% — nearly a factor of 2, out of a choice
about the start date that nothing in the data makes for me. Neither window is obviously the right
one, because the early events are also the ones whose magnitudes were assigned decades later from
historical seismograms and felt reports, so they carry the most uncertainty. It is a single number
with nothing attached to say how firm it is, and putting an interval around a number like this one
is the next thing this course has to learn to do.

## The week's takeaways, for context

1. Randomness is already clumpy, so "it looks clustered" proves nothing on its own.
2. Real seismicity still exceeds chance by a wide margin, and that excess is aftershocks — physics, not noise.
3. To test whether a pattern is real, simulate the world where it is absent and compare.
