# Week 4 — hand-marked question (25 points)

**Where do earthquakes and volcanoes happen — and why there?**

The autograder scores the two code parts. This is the part it cannot: mark it in Gradescope as a manual rubric item worth 25, on top of the autograder's score.

## What the student was asked

### ✏️ Your turn 8

Two of the pictures in this notebook are counts on a log axis: the eruptions by VEI, and your own
magnitudes from part 6. One of them falls off at its low end and the other does not.

The numbers you need are these. From class, the three smallest VEI classes hold 1,019 at
VEI 0, 1,441 at VEI 1 and 4,030 at VEI 2. From part 6, the two smallest magnitude
bins are `counts[0]` and `counts[1]`, which the self-check printed for you; if part 6 did not run,
they came out 8,973 and 2,688 on the copy of the catalogue stored with
the course, so use those and say that is what you did.

In three or four sentences, using those numbers, say which chart has the broken low end and
explain what is different about how the two catalogues were made. Your answer should say what
would have to be true for the *other* chart's low end to break as well.

## Criteria

| Points | A full-credit answer does this |
|---:|---|
| 6 | Identifies the eruption chart as the broken one and cites its two lowest bars, 1,019 at VEI 0 and 1,441 at VEI 1, against the 4,030 at VEI 2, stating the expectation those bars violate: smaller eruptions must be commoner than larger ones. |
| 6 | Cites the first two entries of their own printed counts array, 8,973 earthquakes in the 5.5-6.0 bin against 2,688 in the next one up, and says the magnitude counts keep rising all the way to the edge of the plot instead of falling off. |
| 6 | Explains the difference as one of how the two records were made rather than of what the Earth does: a magnitude 5.5 earthquake is recorded by instruments on the far side of the planet, while an eruption is recorded only if somebody was nearby or it left a deposit, so small old eruptions in empty places are simply absent. |
| 7 | Answers the last clause with a condition on the recording network rather than on the earthquakes: the magnitude chart would break the same way if whole regions had no instruments, or if the query's magnitude floor were pushed low enough that only well-instrumented ground reported. |

## The model answer

Not a template to match word for word — a student who reaches the same conclusions from their own numbers, in their own words, has answered it.

The eruption chart is the broken one. Its two smallest classes hold 1,019 eruptions at VEI
0 and 1,441 at VEI 1, both below the 4,030 at VEI 2, even though smaller eruptions
must be commoner than larger ones. The magnitude chart does the opposite: 8,973
earthquakes in the 5.5-6.0 bin against 2,688 in the next one up, so the counts keep
rising all the way down to the edge of the plot. The difference is how the two records are made.
An earthquake of magnitude 5.5 is recorded by instruments on the far side of the planet, and the
network has been dense enough for that throughout the window this query asks for, so the
catalogue holds essentially every one wherever it happened. Eruptions are recorded by whoever was
nearby and by whatever the eruption left behind, so a small eruption in an empty place a thousand
years ago is simply absent — and the deficit gets worse the further back the window reaches: the
number of VEI 1 eruptions per VEI 2 rises from 0.36 over the whole record to
0.87 since 1950. For the magnitude chart to break the same way, the catalogue would
have to be missing earthquakes in whole regions, which is what catalogue completeness warns about
and what would start to happen if the query's magnitude floor were pushed low enough: a small
earthquake is only recorded where somebody has already put an instrument close by.

## The week's takeaways, for context

1. Earthquakes and volcanoes sit on plate boundaries; the deep earthquakes sit behind the trenches, on subducting slabs.
2. A map is a scatter plot of longitude against latitude — no map library required.
3. Magnitude and VEI are both logarithmic, so plot them on log axes or the figure lies.
