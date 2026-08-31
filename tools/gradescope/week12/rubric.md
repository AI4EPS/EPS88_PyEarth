# Week 12 — hand-marked question (25 points)

**Can you find a fault that nobody mapped?**

The autograder scores the two code parts. This is the part it cannot: mark it in Gradescope as a manual rubric item worth 25, on top of the autograder's score.

## What the student was asked

### ✏️ Your turn 8

Name two clusters from the class run: one you would be willing to draw on a fault map, and one you
think the algorithm invented. For each, quote two numbers from your own output as your reason —
size along the axes, median depth, first event, size, distance, whichever you actually used. Then
name one measurement that is **not** in this notebook and that would settle which of you is right.

Four or five sentences.

## Criteria

| Points | A full-credit answer does this |
|---:|---|
| 6 | Names one cluster they would be willing to draw on a fault map and gives two numbers from their own output as the reason, drawn from the size, the median depth, the first event time, the extent along the PCA axes, or the share of the spread on axis 1. Any cluster is a valid choice provided two of its own numbers carry the argument. |
| 6 | Names one cluster they think the algorithm invented and gives two numbers for that one too, typically a small cluster sitting at or near the min_samples floor of 12, or Coso's 583 events at a median depth of 2.36 km, which is a real place but a shallow geothermal swarm rather than a fault worth drawing. |
| 6 | Ties the invented verdict to the settings rather than to the ground, using the turn 6 eps sweep or the coso_now counts to show that the cluster's existence or its extent changes when eps changes, so it is a property of the parameters and not of the desert. |
| 7 | Names one measurement genuinely outside this notebook that would settle the disagreement and says how it would decide. Focal mechanisms are the model answer, but relocated hypocentres, InSAR or GPS displacement, mapped surface rupture, or a geological field survey all earn it when the student says what agreement or disagreement in that measurement would prove. |

## The model answer

Not a template to match word for word — a student who reaches the same conclusions from their own numbers, in their own words, has answered it.

I would draw the long northwest–southeast limb of cluster 0, and only that. The cluster's
4,670 events run 58 km end to end inside a band about
10 km thick, with 95.9% of the spread on one horizontal
axis, and it holds the magnitude 7.1 and the first event in the catalogue, so it is
the thing that actually broke. I say "the limb" rather than "the cluster" because class showed
that round the magnitude 6.4 the same test gives 53.7% on an
axis pointing straight down: DBSCAN merged a second, crossing fault into the same group, and one
polygon on a map cannot be both. I think the algorithm invented cluster 7: it holds
12 events, exactly `min_samples`, and it is its own cluster at no other setting I
tried — at `eps=0.075` all 12 of them become noise,
and at `eps=0.3` they are absorbed into a group of
93. Its existence is a property of the settings rather
than of the desert. Cluster
6 is a harder case — it is a real place, since 583 events at a median depth of
2.36 km under Coso are not an accident, but "real" there means a shallow geothermal
swarm, not a fault I would draw. What would settle it is data this notebook does not have: the
**focal mechanisms** of the events in each cluster, which give the orientation of the plane that
slipped in each earthquake. If a cluster's earthquakes all slipped on planes with the same
orientation, and that orientation matches the plane PCA fitted through their locations, it is one
structure; if they point every which way, the cluster is a crowd of unrelated events that happened
to be close together.

## The week's takeaways, for context

1. Clustering finds structure with the labels hidden.
2. DBSCAN may say "this one belongs to nothing"; k-means must assign every point.
3. The parameters decide the answer, so report them alongside it.
