# Speech — draft v1 (~20 minutes)

Text meant to be **read aloud**, not projected: the slides carry the condensed content,
this is what you actually say. Colloquial register, first person plural ("we", "we
need"), as in your original brainstorm. `[SLIDE: ...]` marks where to change slide —
these are reference points, not rigid constraints.

**This is a draft to start working on together** — tell me where the tone doesn't sound
like you, where you want to cut, where you want to add a personal detail (especially in
sections 1 and 3, which come from work of yours I had no way to read anywhere) and we'll
fix it together sentence by sentence.

Estimated time, **actually recalculated** on the real word count of each slide (no
longer an eyeballed guess): **~1950 words total**, at 105 "effective" words/minute
(natural reading pace + pauses for slide changes and taking in the figures) → **about
18-19 minutes**, comfortably under the 20-minute target, with some margin for
last-minute questions or a more relaxed spoken pace. The old estimates (2950, then 3300
words) had fallen behind the edits made along the way and had never been recalculated
against the real text — this one is. Per-slide timestamps below are updated
accordingly.

---

## [SLIDE 1: Title] — 0:00

Good morning everyone. Today I'm talking about an idea I've been working on over the
past month: using machine learning to automatically recognize the crystal symmetry of a
structure, in situations where classical methods can't manage it. Let me say upfront
that this isn't finished work — it's more of an idea that's taking shape.

## [SLIDE 2-4: Where the problem comes from] — ~0:42

The problem we started from comes from Nested Sampling, which — as you've heard me and
Nico mention more than once — is a sampling method that explores the configuration space
of a system from high energies down to low ones, and gives us direct access to the
partition function over a wide range of temperatures. You can see it right here: the
contours shrink ring by ring, from the high-temperature region on the outside down to
low temperature at the center.

One of the key outputs is the trajectory of a huge number of structures — those are the
dots scattered across the contours in the diagram, each one a structure, each one tied to
the iteration it came from. The idea is: can we use this trajectory to build the phase
diagram — that is, to identify which crystal phases, and therefore which symmetries, the
system shows at a given temperature?

## [SLIDE 5: Example — the Cu–Zn (brass) phase diagram]

A concrete example, the one that actually started this idea, is the brass system. Here's the actual
diagram: copper content on the horizontal axis, temperature on the vertical one, and each
region is a different phase — alpha, beta, gamma — each with its own crystal
symmetry. Being able to label those regions starting from the structures found along the
trajectories was a huge amount of manual effort — effort this idea hopes, once finished,
to replace.

## [SLIDE 6-8: Why a classical symmetry finder is not enough] — ~2:01

So let's look at the main problem: the structures we get out of Nested Sampling are
certainly not clean or ideal. Here's what that actually looks like: an ideal lattice on
one side, the same lattice on the other with the atoms nudged out of place and a vacancy
missing. They come from Monte Carlo moves at finite temperature, and show various levels
of disorder: atoms are displaced from their exact symmetry positions, and there can be
atomic vacancies.

This means that classical libraries for identifying crystal symmetry — like spglib —
simply don't work on input this noisy. You can see why right here: spglib checks every
atom against a small tolerance circle around its ideal site, and a single atom outside
that circle is enough for the result to collapse to P1, no symmetry found.

One might think of "cleaning up" each structure with a local energy optimization before
analyzing it, but there are two problems here. The first is cost: we can't afford to
optimize every structure in the trajectory — we're talking about hundreds of thousands
of structures — so at best we can only do it for a subset.
The second, more fundamental one, is that even when we do, the optimization stops at the
nearest local minimum on the potential energy surface and there's no guarantee that minimum is one spglib can
recognize.

So we need something else.

## [SLIDE 9: First attempt — SOAP + UMAP] — ~3:37

The only approach that gave us concrete results in the brass case was the following: we
took the most representative structures for a given set of temperatures and
compositions, ran them through an annealed molecular dynamics simulation, and, using
UMAP to compare the SOAP descriptors of the structures seen during that simulation, built
a 2D map. I know this might sound complex but, put simply, we compared all the Nested Sampling
structures by building a two-dimensional map.

This worked reasonably well: that's the actual map on screen — every point a structure,
and clear clusters already forming. When we projected the known reference structures of
brass's alpha, beta and gamma phases, taken from the literature, into that same map,
they landed exactly inside those clusters.

## [SLIDE 10: First attempt — clusters back in T–composition space]

This let us split the phase diagram into different regions and identify, even if not
with analytical precision, the alpha, beta and gamma phases — that's what you see here:
the same temperature-composition plane, now colored by cluster, and the regions line up
fine enough with where we'd expect alpha, beta and gamma to sit.

## [SLIDE 11-15: What if we had no reference structures?] — ~4:36

This, though, left me with a question: what if we didn't have reference structures to
compare against? We know brass well, but for a new system we might not have any
alpha-beta-gamma phase already characterized in the literature to project against.

What we need, then, is something that already carries with it general knowledge of
crystal symmetries,

that's transferable — independent of the specific chemical system,

and that's robust to thermal noise and atomic vacancies.

The idea is to train a neural network model to recognize the space group, using
structures whose symmetry is known and certain.

## [SLIDE 16: A global descriptor for each structure] — ~5:33

Let's start from the simplest case: systems with a single atomic species. For every
structure we need a global descriptor — a vector of numbers that represents it. My
starting point was exactly the descriptors that had already worked for brass: SOAP,
computed atom by atom to quantify the local environment, then averaged to get a
descriptor for the whole structure.

This automatically gives us the invariances we need: to permutation of the atoms
— the order in which I list them doesn't matter — to rotation and translation of the
structure, and to the choice of the periodic cell used to represent it.

## [SLIDE 17: An explorable low-dimensional space]

One thing I didn't want to lose from the initial cluster-based approach is the ability
to *visually* explore this low-dimensional space: when a structure sits in a transition
region, or its phase isn't well defined, it's useful to be able to see "where it is" and
which symmetry it tends towards — not just get a hard label. For a structure like this one,
in the middle between a purely cubic region and a purely hexagonal one, the hard label would be,
in my opinion, somewhat restrictive.

## [SLIDE 18-19: What we need — encoder + classifier] — ~6:55

So the two things we need are: an **encoder**, which reduces the highly
multi-dimensional SOAP vector describing the structure to a compact representation,

and a **classifier**, which assigns it a symmetry label.

## [SLIDE 20: Why not an Autoencoder / VAE?]

Talking about an encoder, the first thing that usually comes to mind is an autoencoder,
or its variational variant. But in our case the reconstruction capability — so the whole
decoder — isn't needed at all. What we actually care about
is pulling different symmetries apart, not reconstructing the input.

## [SLIDE 21: How the SupCon loss works] — ~7:37

Looking through the literature, I came across what seemed to me the strategy closest to
what I needed: Supervised Contrastive Learning, from Khosla's 2020 paper.

The idea is different from a normal classifier: instead of learning directly to predict
a label, the model learns to build a space where objects with the same group or label
sit close to each other, and different ones sit far apart. I've included the loss
formula.

For every structure i in the training batch, we take the set of points sharing its
label — the positives, P — against all the others in the batch, A. For each point in P,
this ratio essentially quantifies how much the closeness between i and that point
dominates over the sum of similarities with every other element of the batch.

In practice, the loss pushes points with the same label closer together and pushes the
others apart. So in our case, the positives are structures with the same symmetry as the
anchor, while other symmetries represent the negatives.

## [SLIDE 22: Architecture · Phase 1 — learning the representation] — ~10:47

Based on the paper's results and its implementation, I designed the model as four
blocks that can be assembled together, used during a training process split into two
separate phases. I say it now: all four blocks are plain MLPs with ReLU as the
activation function, so there's definitely room to experiment in that direction.

In the first phase we train the **encoder** block and the **projection tail** together,
using exactly the SupCon loss we just saw. The encoder is the main body of parameters, and its output is what all
the other blocks then use. The projection tail further reduces this output, and it's on
its space — not directly on the encoder's — that the loss is computed: it's like an
extension of the encoder, used only during training and then discarded. That's because,
as mentioned in the paper, the lower dimensionality that helps the space separation also
lowers the amount of information that can be used.

## [SLIDE 23: Architecture · Phase 2 — frozen body, task tails]

In the second phase, with the body now frozen, we drop the projection tail and train two
other tails separately, as you see splitting off here: a **classification tail**, with a
normal softmax straight to a class, and a **visualization tail**, which again uses the
SupCon loss — same idea, but this time projecting down to two or three dimensions, so we
can directly plot the symmetry clusters.

## [SLIDE 24-26: Building the dataset] — ~13:07

Let's move on to practice. For the tests I've run, and that are still running, the dataset is
generated synthetically with Python's pyxtal library. This lets us generate thousands of
structures for each of the seven crystal systems,

to which we artificially add noise — small atomic displacements, vacancies — to get the
model used to exactly the kind of noise we find in real trajectories.

At the end of preprocessing, for every structure we have its SOAP vector and its labels
— crystal family and space group — for free, known by construction even after the
noise.

## [SLIDE 27-28: First obstacle — too many space groups at once]

The first attempt was to try to guess the space group directly, in a single pass: just
the SOAP vector into an encoder and a classifier that has to pick among all 230 space
groups at once. It didn't work well: there's too much variety, dozens of possible space
groups. As you can also see from the 2D map produced by the same model, the systems are
mostly clustered, while within each system the space groups are completely mixed
together.

## [SLIDE 29-30: The fix — family first, then a space-group expert]

So we moved to a two-step process: first a crystal-system classifier, with only seven
classes;

then, for each family, a dedicated "expert" model that only handles the space groups of
that system — so, for example: step one, the family model, predicting, say, tetragonal;
step two, that prediction selects the matching expert, which takes the same SOAP vector
and predicts the actual space group. Both steps start from the same SOAP descriptor, and
their only dependency is which expert gets selected, based on the prediction from step
1.

Both steps use the same four blocks we saw before — the only thing that changes is how
many hidden layers we give each MLP, and how wide they are.

Going forward, as I walk you through the results, I'm deliberately leaving the specific
architecture sizes off the slides. We're still tuning depth and width for both the
classifier and the experts, so any number I'd show you today could well be different
next week. What I want you to take from this is the shape of the approach, not a
snapshot of hyperparameters.

## [SLIDE 31: The family classifier — clear separation] — ~14:50

Let's move to some visual training results. The dataset is the one I just described:
the synthetic structures generated with pyxtal across the seven systems, with the
artificial noise added. On this dataset we trained both the family classifier and, for
each family, the space-group expert.

On the left, a two-dimensional map of our structures, the output of the visualization
block of a sample run: every point is a separate structure, colored by its family —
seven clean clusters. As you can see, the separation by family in this space is clear.
On the right, the result of the actual classifier, the confusion matrix: it shows strong
diagonality, with a mean accuracy of about 95%.

At this point, structures identified as belonging to a given family are passed to the
corresponding expert, to classify their space group.

## [SLIDE 32: One space-group expert per family] — ~16:01

On the left you see the grid of plots produced by the visualization tail for each
family: monoclinic, trigonal and tetragonal each form tight, well-separated clusters.
Cubic, though, is a mess of overlapping ones.

On the right, the confusion matrices tell the same story: excellent diagonality for
every family except cubic, which has some problems. For simplicity I've only put
the two side by side here — cubic and, for comparison, monoclinic, which is almost
perfectly diagonal.

From the many tests run so far, the problem doesn't seem to depend on the model, but
more likely on the input — the SOAP descriptor itself, for that specific family. I'm
still investigating this, so I don't have a definite answer yet.

## [SLIDE 33: Next step — MACE embedding instead of SOAP] — ~16:47

As a next step, the idea is to replace the SOAP input — limited to the local environment
within a certain cutoff radius — with the embedding from an already pre-trained
**MACE** model, which, as we know, natively captures angular interactions too, thanks to
its message-passing layers. Everything else in the diagram stays exactly the same — same
encoder, same tails — only the input box changes, from SOAP today to the MACE embedding
next. It's a hypothesis we want to test, especially to see how much it can improve the
most problematic cases. So far, first tests show the mean accuracy on predicting cubic
space groups rising from 0.4 to 0.7.

## [SLIDE 34: Conclusions] — ~17:29

To sum up: we started from the thermally noisy structures coming out of Nested Sampling
to build a model based on the SupCon loss, able to recover the symmetry of the phase —
first as a crystal family, then as a space group. Right now performance is good, with
the exception of the cubic space groups. There, I hope to get a definitive fix by
replacing the SOAP embedding with MACE. That's it for the presentation — questions or
ideas for improvement are very welcome.
