# Week 10 — hand-marked question (25 points)

**Earthquake or explosion — how does the world verify a nuclear test ban?**

The autograder scores the two code parts. This is the part it cannot: mark it in Gradescope as a manual rubric item worth 25, on top of the autograder's score.

## What the student was asked

### ✏️ Your turn 8

Two or three sentences, quoting your own printed numbers.

Your turn 6 gave you an F1 for logistic regression with depth and an F1 for the same model without
it. Quote both, say what the gap between them tells you about what the model had actually learned,
and say whether you would report the first of the two as a result. Then, in one more sentence:
name one clue this week used that would still be there if the event you were trying to catch were
a secret nuclear test rather than a quarry — or say that none would, and why.

## Criteria

| Points | A full-credit answer does this |
|---:|---|
| 6 | Quotes both of their own turn 6 F1 scores, logistic regression with depth (0.7701) and the same model with the depth column removed (0.0000), and reads the second against the always-earthquake baseline rather than against zero in the abstract. |
| 6 | Names what the gap means: essentially all the apparent skill was coming from one column, and that column is not a measurement of the ground but a value an analyst entered after the event had already been labelled a blast. Backs it with a concrete sign, the 360 blasts sharing the single depth -0.82 km or the constant 31.61 km depthError on 96% of them. |
| 6 | Answers whether they would report the 0.7701 as a result, and answers no on the leakage ground. Reporting it with a caveat attached does not earn this item, because the question is whether the number is a result at all. |
| 7 | Answers the secret-test sentence by reasoning from why each clue works rather than by guessing: the repeated location works because a quarry is worked from the same pit for decades, the hour and weekday clues because blasting is scheduled legal work, the depth because somebody typed it in once the label was chosen. Naming a clue that would survive earns this only if the case for its survival is actually made. |

## The model answer

Not a template to match word for word — a student who reaches the same conclusions from their own numbers, in their own words, has answered it.

With depth, my logistic regression scored F1 0.7701; with the depth column taken out and
nothing else changed, it scored 0.0000, which is the same F1 as the rule that answers
"earthquake" every time, and naive Bayes fell from 0.7612 to 0.0369, which is
barely different from it. So essentially all of the apparent skill was coming from one column, and
that column is not a measurement: 360 blasts share the single depth
-0.82 km to the nearest 10 metres across two different quarries, Boron and
Mojave, and `depthError` carries the same constant 31.61 km on
96% of the blasts, so somebody assigned that depth after they had
already decided the event was a blast. I would not report the first score as a result, because
the model was reading the answer off the label-maker's own notes rather than off the ground.

None of the four clues would survive. The repeated address works because a quarry is blasted from
the same pit for decades, the working-hours and weekday clues work because blasting is a legal job
with a shift pattern, and the depth clue works because an analyst wrote it down once the label was
already chosen. A test nobody has announced is one event, at a site with no history in the
catalogue, at an hour picked to attract no attention, and with nobody to type a depth in for it —
so all four go at once, and the honest answer is that this week's classifier would have nothing
left to read.

## The week's takeaways, for context

1. Accuracy lies when one class is rare: precision and recall are what you report.
2. If a column already knows the answer, you have leakage rather than a result.
3. Beat the simplest rule you can write by hand before believing any model.
