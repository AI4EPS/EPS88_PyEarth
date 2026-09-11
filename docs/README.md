# EPS 88 — PyEarth

*A Python Introduction to Earth Science*

Mondays 12:00–2:00 pm, McCone 265 · 2 units · **Instructor [Weiqiang Zhu](mailto:zhuwq@berkeley.edu)**

Office hours Tuesdays 12:00–1:00 pm, 285 McCone.

One question a week, answered with data and code you write yourself. Where do earthquakes and volcanoes happen, and why there? Which worlds besides Earth could hold liquid water? Is carbon dioxide rising faster than it used to? Can a machine learn to hear an earthquake?

The datasets are the ones research uses, not simplified teaching examples: the USGS earthquake catalogue, NASA's Exoplanet Archive, the Smithsonian's record of volcanic eruptions, and the seismic recordings used to train published deep-learning models. Python begins from zero in the first week, which ends with a figure you have plotted yourself, and reaches a neural network in the last. In between you learn to fit a model to data, to state how far it can be trusted, and to distinguish a real result from a coincidence. Thirteen notebooks, then a project of your own.

## Prerequisites

None. No programming experience is assumed — we build Python from zero. EPS 88 is a Data 8 connector; students who have taken Data 8 will recognize some of the statistics and move faster, but nothing here requires it.

## The weeks

Each link opens that notebook in your own DataHub account. **A week with no link has not been
released yet** — each one goes up before the class that uses it. Week 1.5 is practice: not graded,
nothing to submit, and its worked solutions are published alongside it.

| Week | Earth-science question | Python | Field |
|---:|---|---|---|
| 1 | [What was your birthquake?][w1] | Notebooks, lists | Seismology |
| 1.5 | [Practice — no class on Labor Day][p0] · [solutions][p0s] | Lists, loops, if, functions | Practice — not graded |
| 2 | Which of these worlds could have liquid water — and why does your test reject Earth? | Loops, Functions | Planetary science |
| 3 | Earth's elevation has two peaks. So does Mars's. Same reason? | Arrays, Tables | Planetary science / oceanography |
| 4 | Where do earthquakes and volcanoes happen — and why there? | Plotting, Maps | Seismology / volcanology |
| 5 | Do earthquakes cluster — or is that just what randomness looks like? | Probability, Monte Carlo | Seismology |
| 6 | How old is the universe? | Linear regression | Astronomy |
| 7 | How often does a Tambora happen? | Feature engineering | Volcanology |
| 8 | Was our earthquake forecast wrong — or were we just unlucky? | Confidence intervals | Seismology / oceanography |
| 9 | Is CO2 rising faster than it used to? | Model selection | Climate |
| 10 | Earthquake or explosion — how does the world verify a nuclear test ban? | Logistic regression | Seismology / policy |
| 11 | Where does a volcano get its magma? | SVM, Decision Trees | Volcanology / petrology |
| 12 | Can you find a fault that nobody mapped? | Clustering | Seismology / tectonics |
| 13 | Can a machine hear an earthquake? | Neural networks | Seismology |

## How the course works

One notebook a week. You work in it during class and continue in the same file at home. The
class questions apply a method you have just been shown; the homework asks the question the
class deliberately left open. Both halves are your own work, and both are graded. Submit the
notebook once, Sunday 23:59; solutions are published the following Wednesday.

## Grading

weekly notebook **75%** · project **15%** · participation **10%**

- **Weekly notebook** — one file per week, class work and homework together, submitted once.
  Due Sunday 23:59. The lowest weekly notebook is dropped.
- **Participation** — assessed from your contribution in class; nothing is submitted separately.
- **Project** — a track notebook or a question of your own. Lightning talks
  Monday 7 December.

**Late work** — 10% per day; nothing accepted once solutions post.

## What you will use

`python` · `numpy` · `pandas` · `matplotlib` · `scikit-learn` · `pytorch`

[w1]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F01_birthquake.ipynb&branch=main
[p0]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F01b_practice.ipynb&branch=main
[p0s]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F01b_practice_solution.ipynb&branch=main
