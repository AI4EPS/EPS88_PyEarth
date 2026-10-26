# EPS 88 — PyEarth

*A Python Introduction to Earth Science*

Mondays 12:00–2:00 pm, McCone 265 · 2 units · **Instructor [Weiqiang Zhu](mailto:zhuwq@berkeley.edu)**

Office hours Tuesdays 12:00–1:00 pm, 285 McCone.

One question a week, answered with data and code you write yourself. Where do earthquakes and volcanoes happen, and why there? Which worlds besides Earth could hold liquid water? Is carbon dioxide rising faster than it used to? Can a machine learn to hear an earthquake?

The datasets are the ones research uses, not simplified teaching examples: the USGS earthquake catalogue, NASA's Exoplanet Archive, the Smithsonian's record of volcanic eruptions, and the seismic recordings used to train published deep-learning models. Python begins from zero in the first week, which ends with a figure you have plotted yourself, and reaches a neural network in the last. In between you learn to fit a model to data, to state how far it can be trusted, and to distinguish a real result from a coincidence. Thirteen notebooks, then a project of your own.

## Prerequisites

None. No programming or statistics is assumed — we build Python from zero, and every statistical idea is taught here the first time it is used. EPS 88 is a Data 8 connector, but nothing here requires Data 8.

## The weeks

Each link opens that notebook in your own DataHub account. **A week with no link has not been
released yet** — each one goes up before the class that uses it. Week 1.5 is practice: not graded,
nothing to submit, and its worked solutions are published alongside it.

| Week | Earth-science question | Python | Field |
|---:|---|---|---|
| 1 | [What was your birthquake?][w1] · [solutions][w1s] | Notebooks, lists | Seismology |
| 1.5 | [Practice — no class on Labor Day][p0] · [solutions][p0s] | Lists, loops, if, functions | Practice — not graded |
| 2 | [Which of these worlds could have liquid water — and why does your test reject Earth?][w2] · [solutions][w2s] | Loops, Functions | Planetary science |
| 3 | [Earth's elevation has two peaks. So does Mars's. Same reason?][w3] · [solutions][w3s] | Arrays, Tables | Planetary science / oceanography |
| 4 | [Where do earthquakes and volcanoes happen — and why there?][w4] · [solutions][w4s] | Plotting, Maps | Seismology / volcanology |
| 5 | [A big earthquake has just struck. What will the next week bring?][w5] · [solutions][w5s] | Probability, simulation | Seismology |
| 6 | [How old is the universe?][w6] · [solutions][w6s] | Linear regression | Astronomy |
| 7 | [When will CO2 pass 450 ppm?][w7] | Model selection | Climate |
| 8 | [How long was the Earth frozen?][w8] | Confidence intervals | Geochronology |
| 9 | Earthquake or nuclear explosion? | Logistic regression | Seismology / policy |
| 10 | Quarry blast or earthquake? | Trees and SVM, Cross-validation | Seismology |
| 11 | Can a computer draw California's climate map? | Clustering | Climate / seismology |
| 12 | How hard will the ground shake? | Neural networks | Engineering seismology |
| 13 | Can a machine hear an earthquake? | Convolutional networks | Seismology |

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
[w2]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F02_liquid_water.ipynb&branch=main
[w3]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F03_two_peaks.ipynb&branch=main
[w4]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F04_where_and_why.ipynb&branch=main
[w5]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F05_what_comes_next.ipynb&branch=main
[w6]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F06_age_of_the_universe.ipynb&branch=main
[w7]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F07_co2_450_ppm.ipynb&branch=main
[w8]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F08_how_long_frozen.ipynb&branch=main
[w1s]: https://ai4eps.github.io/EPS88_PyEarth/notebooks/01_birthquake_solution/
[p0s]: https://ai4eps.github.io/EPS88_PyEarth/notebooks/01b_practice_solution/
[w2s]: https://ai4eps.github.io/EPS88_PyEarth/notebooks/02_liquid_water_solution/
[w3s]: https://ai4eps.github.io/EPS88_PyEarth/notebooks/03_two_peaks_solution/
[w4s]: https://ai4eps.github.io/EPS88_PyEarth/notebooks/04_where_and_why_solution/
[w5s]: https://ai4eps.github.io/EPS88_PyEarth/notebooks/05_what_comes_next_solution/
[w6s]: https://ai4eps.github.io/EPS88_PyEarth/notebooks/06_age_of_the_universe_solution/
