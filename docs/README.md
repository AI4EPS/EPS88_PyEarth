# EPS 88 — PyEarth

*A Python Introduction to Earth Science*

Mondays 12:00–2:00 pm, McCone 265 · 2 units · **Instructor [Weiqiang Zhu](mailto:zhuwq@berkeley.edu)**

One question a week, answered with data and code you write yourself. Where do earthquakes and volcanoes happen, and why there? Which worlds besides Earth could hold liquid water? Is carbon dioxide rising faster than it used to? Can a machine learn to hear an earthquake?

The datasets are the ones research uses, not simplified teaching examples: the USGS earthquake catalogue, NASA's Exoplanet Archive, the Smithsonian's record of volcanic eruptions, and the seismic recordings used to train published deep-learning models. Python begins from zero in the first week, which ends with a figure you have plotted yourself, and reaches a neural network in the last. In between you learn to fit a model to data, to state how far it can be trusted, and to distinguish a real result from a coincidence. Thirteen notebooks, then a project of your own.

## Prerequisites

None. No programming experience is assumed — we build Python from zero. EPS 88 is a Data 8 connector; students who have taken Data 8 will recognize some of the statistics and move faster, but nothing here requires it.

## The weeks

Each link opens that week's notebook in your own DataHub account.

| Week | Earth-science question | Python | Field |
|---:|---|---|---|
| 1 | [What was your birthquake?][w1] | Notebooks, lists | Seismology |
| 2 | [Which of these worlds could have liquid water — and why does your test reject Earth?][w2] | Loops, Functions | Planetary science |
| 3 | [Earth's elevation has two peaks. So does Mars's. Same reason?][w3] | Arrays, Tables | Planetary science / oceanography |
| 4 | [Where do earthquakes and volcanoes happen — and why there?][w4] | Plotting, Maps | Seismology / volcanology |
| 5 | [Do earthquakes cluster — or is that just what randomness looks like?][w5] | Probability, Monte Carlo | Seismology |
| 6 | [How old is the universe?][w6] | Linear regression | Astronomy |
| 7 | [How often does a Tambora happen?][w7] | Feature engineering | Volcanology |
| 8 | [Was our earthquake forecast wrong — or were we just unlucky?][w8] | Confidence intervals | Seismology / oceanography |
| 9 | [Is CO2 rising faster than it used to?][w9] | Model selection | Climate |
| 10 | [Earthquake or explosion — how does the world verify a nuclear test ban?][w10] | Logistic regression | Seismology / policy |
| 11 | [Where does a volcano get its magma?][w11] | SVM, Decision Trees | Volcanology / petrology |
| 12 | [Can you find a fault that nobody mapped?][w12] | Clustering | Seismology / tectonics |
| 13 | [Can a machine hear an earthquake?][w13] | Neural networks | Seismology |

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
  Monday 7 December; notebook due Wednesday 16 December, 23:59.

**Late work** — 10% per day; nothing accepted once solutions post.

## What you will use

`python` · `numpy` · `pandas` · `matplotlib` · `scikit-learn` · `pytorch`

[w1]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F01_birthquake.ipynb&branch=main
[w2]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F02_liquid_water.ipynb&branch=main
[w3]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F03_two_peaks.ipynb&branch=main
[w4]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F04_where_and_why.ipynb&branch=main
[w5]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F05_clustered_or_random.ipynb&branch=main
[w6]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F06_age_of_the_universe.ipynb&branch=main
[w7]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F07_how_often_tambora.ipynb&branch=main
[w8]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F08_wrong_or_unlucky.ipynb&branch=main
[w9]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F09_rising_faster.ipynb&branch=main
[w10]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F10_earthquake_or_explosion.ipynb&branch=main
[w11]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F11_where_magma.ipynb&branch=main
[w12]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F12_hidden_fault.ipynb&branch=main
[w13]: https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F13_machine_hears.ipynb&branch=main
