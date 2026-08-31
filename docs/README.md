# EPS 88 — PyEarth

*A Python Introduction to Earth Science* · Mondays 12:00–2:00 pm, McCone 265 · 2 units
**Instructor** Weiqiang Zhu <zhuwq@berkeley.edu>

One question a week, answered with data and code you write yourself. Where do earthquakes and volcanoes happen, and why there? Which worlds besides Earth could hold liquid water? Is carbon dioxide rising faster than it used to? Can a machine learn to hear an earthquake?

Nothing here is a teaching example. You work with the USGS earthquake catalogue, NASA's Exoplanet Archive, the Smithsonian's record of volcanic eruptions, and the seismic recordings used to train published deep-learning models. Python starts from zero in the first week and the first hour ends with a plot on your screen; by the last week you are training a neural network. In between you learn to fit a model to data, to say how far it can be trusted, and to tell a real result from a coincidence. Thirteen notebooks, then a project of your own.

## Prerequisites

No programming experience is assumed. EPS 88 is a Data 8 connector; students who have taken Data 8 will recognise some statistics and move faster, but nothing here requires it.

## The weeks

Each link opens that week's notebook in your own DataHub account.

| | Earth-science question | Python | Field |
|---|---|---|---|
| 1 | [What was your birthquake?](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F01_birthquake.ipynb&branch=main) | Notebooks, lists | seismology |
| 2 | [Which of these worlds could have liquid water — and why does your test reject Earth?](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F02_liquid_water.ipynb&branch=main) | Loops, Functions | planetary science |
| 3 | [Earth's elevation has two peaks. So does Mars's. Same reason?](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F03_two_peaks.ipynb&branch=main) | Arrays, Tables | planetary science / oceanography |
| 4 | [Where do earthquakes and volcanoes happen — and why there?](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F04_where_and_why.ipynb&branch=main) | Plotting, Maps | seismology / volcanology |
| 5 | [Do earthquakes cluster — or is that just what randomness looks like?](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F05_clustered_or_random.ipynb&branch=main) | Probability, Monte Carlo | seismology |
| 6 | [How old is the universe?](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F06_age_of_the_universe.ipynb&branch=main) | Linear regression | astronomy |
| 7 | [How often does a Tambora happen?](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F07_how_often_tambora.ipynb&branch=main) | Feature engineering | volcanology |
| 8 | [Was our earthquake forecast wrong — or were we just unlucky?](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F08_wrong_or_unlucky.ipynb&branch=main) | Confidence intervals | seismology / oceanography |
| 9 | [Is CO2 rising faster than it used to?](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F09_rising_faster.ipynb&branch=main) | Model selection | climate |
| 10 | [Earthquake or explosion — how does the world verify a nuclear test ban?](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F10_earthquake_or_explosion.ipynb&branch=main) | Logistic regression | seismology / policy |
| 11 | [Where does a volcano get its magma?](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F11_where_magma.ipynb&branch=main) | SVM, Decision Trees | volcanology / petrology |
| 12 | [Can you find a fault that nobody mapped?](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F12_hidden_fault.ipynb&branch=main) | Clustering | seismology / tectonics |
| 13 | [Can a machine hear an earthquake?](https://datahub.berkeley.edu/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FAI4EPS%2FEPS88_PyEarth&urlpath=lab%2Ftree%2FEPS88_PyEarth%2Fdocs%2Fnotebooks%2F13_machine_hears.ipynb&branch=main) | Neural networks | seismology |

## How the course works

One notebook a week. You work in it during class and continue in the same file at home — the
class questions are ones you have just been shown how to do, the homework asks something the
class deliberately did not answer. **All of it is your work and all of it is graded.** Submit the
whole notebook once, Sunday 23:59; solutions go up the following Wednesday.

## Marks

participation **10%** · weekly notebook **75%** · project **15%**

- **Weekly notebook** — one file per week, class work and homework together, submitted once.
  Due Sunday 23:59. The lowest weekly notebook is dropped.
- **Participation** — observed in the room; there is nothing separate to upload.
- **Project** — a track notebook or your own question. Lightning talks 2026-12-07;
  notebook due 2026-12-16 23:59.

**Late work** — 10% per day; nothing accepted once solutions post.

## What you will use

`python` · `numpy` · `pandas` · `matplotlib` · `scikit-learn` · `pytorch`
