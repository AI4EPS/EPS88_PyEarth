# Week 8 — hand-marked question (25 points)

**Was our earthquake forecast wrong — or were we just unlucky?**

The autograder scores the two code parts. This is the part it cannot: mark it in Gradescope as a manual rubric item worth 25, on top of the autograder's score.

## What the student was asked

### ✏️ Your turn 9

Two or three sentences, quoting your own numbers from your turn 8.

In class you committed to a verdict — *wrong* or *unlucky* — using the fitting range class chose.
Say what your own range does to it: where five falls against your interval on the count, and
whether a reader of your notebook would come away thinking the forecast is broken or thinking the
last thirty-six years were busy. Then say what would have to be true for a single thirty-six-year
count to settle the question either way.

## Criteria

| Points | A full-credit answer does this |
|---:|---|
| 6 | Names which fitting range they took, 3.5 to 5.0 or 4.5 to 6.0, and quotes the three numbers their own range printed: the forecast, the top end of the 95% interval on the count, and the fraction of simulated worlds reaching five. Either fork is valid; the numbers must be theirs. |
| 6 | Says where five falls under their range, outside the interval or on and inside it, and states the verdict a reader of their notebook would come away with. Both verdicts are correct answers when they follow from the student's own printed fraction. |
| 6 | Sets their result against class's range (4.0 to 5.5, five on the edge, 3.1% of simulated worlds) and makes the point that the verdict turns on a choice nobody can defend as the only right one, so the answer moves more with the fitting range than with the observation. |
| 7 | Answers the last clause with a stated condition rather than a wish: five would have to fall outside the interval under every defensible fitting range, or the window would have to be long enough that the Poisson scatter, roughly the square root of the expected count, became small beside the gap being argued about. |

## The model answer

Not a template to match word for word — a student who reaches the same conclusions from their own numbers, in their own words, has answered it.

Fitting between 3.5 and 5.0 gives a forecast of 1.37 M7+ earthquakes, a
95% interval on the count topping out at 4, and five or more in only
1.2% of simulated worlds — so on my choice five is *outside* the interval and
the forecast does look broken, where in class I called it unlucky. Class's range, 4.0 to 5.5, gave
3.1%, with five sitting on the edge rather than beyond it, and the third
choice, 4.5 to 6.0, has only 57 events above magnitude 5.5 left to
fit, so its interval is wider still. The verdict therefore turns on a choice nobody can defend as
the only right one, and the answer moves more when I change my fitting range than the observation
moves it. For a single thirty-six-year count to settle the question, it would have to fall outside
the interval under *every* defensible fitting range — five does not — or the window would have to
be long enough that the Poisson scatter, which is roughly the square root of the expected count,
became small beside the gap being argued about.

## The week's takeaways, for context

1. A single number is not an answer. Report an interval.
2. Resampling your own data tells you how much your estimate would have wobbled.
3. An interval tells you what would have wobbled; check you are comparing like with like before you call a model broken.
