# USGLoc seminar — speaker script

## Opening — 4 min

### 1. Title *(~40s)*
Good afternoon everyone. Today I want to talk about a question that sounds
almost bureaucratic but turns out to be deeply technical: when a robot tells
you where it is — *should you believe it?
our system's answer to "where are you" is not a point, it's a distribution. 

### 2. Would you fly on it? *(~2 min)*
Every visual localization system you can download today — and some of them
are accurate — has one thing in common: they are all, always,
one hundred percent confident. Show them a perfect, well-lit view of a room
they mapped yesterday: they output a pose. Show them a half-occluded view of
a room where someone has moved all the furniture: they output a pose. *The
same kind of answer, with the same confidence.* **[pause]** A human pilot
who reported both situations identically would be grounded.

So here is the seminar in one sentence: how do we make a 3D map admit what
it does *not* know — and how do we turn that honesty into a safer robot?

### 3. Outline *(~40s)*
The plan: fundamentals first — localization, splatting, and what
"calibrated uncertainty" even means. Then the problem, and a tour of what
exists — with, importantly, what each family of methods *cannot* do. Then
our system, USGLoc; how we evaluate honesty, not just accuracy; and open
problems, several of which are genuinely up for grabs.

---

## Section: Fundamentals — 12 min

### 4. Visual localization 101 *(~2.5 min)*
The task: one query image, one pre-built map, estimate the full 6-DoF camera
pose. 
The classical name is the kidnapped-robot problem: the robot wakes up
somewhere it has been before, and has to figure out where.

Where is this used? AR headsets re-anchoring content
between sessions; robots and drones recovering after tracking loss; cars
positioning at lane level when GPS dies in a tunnel.

The dominant recipe hasn't changed in a decade 
a *coarse* stage — place recognition or a learned regressor gets you roughly
there — and a *fine* stage that aligns the image against the map. Thirty
years of research have gone into question one: where am I. Today is about
question two: *how sure am I* — which the field has mostly ignored.


### 5. From point clouds to Gaussian splatting *(~2.5 min)*
Three generations.
 First, sparse point clouds from structure-from-motion —
match keypoints, solve PnP; still a strong baseline.
Second, NeRF, 2020: the scene as a neural network; photorealistic, but rendering one frame took
seconds — painful when your localizer wants hundreds of renders. 
Third,
Gaussian splatting, 2023: the scene as a few million explicit little
ellipsoids — position, shape, opacity, color — rendered by projecting and
alpha-blending, photorealistic *at real-time rates*.

**[point at the equation]** This is the entire renderer: moi Gaussian
contributes its color, weighted by its opacity and by how much everything in
front of it lets through.  Hai đặc tính quan trọng. It's differentiable
theo the camera pose — nên bạn có thể *dùng
gradient descent* để tìm về đúng tư thế. Và bất kỳ đại lượng nào gắn với
từng splat cũng có thể "đi nhờ" như một kênh màu phụ.

### 6. Uncertainty 101 *(~2.5 min)*
Two kinds of not-knowing, and it matters which one you have. *Aleatoric*:
noise in the world — sensor noise, motion blur, a textureless wall. You
cannot fix it with more data. *Epistemic*: gaps in your knowledge — a
viewpoint you never mapped, or, crucially for us, a scene that *changed*
after mapping. Reducible — if you know where it is.

And one test that keeps everyone honest: calibration. A weather forecaster
who says seventy percent rain should get wet on seventy percent of those
days. For poses: a ninety-five percent credible region should contain the
true pose ninety-five percent of the time. Not more — that's wasted
conservatism. Not less — that's the dangerous case. **[point at the red
curve]** Below the diagonal is a system promising more than it delivers —
and a downstream safety filter cannot detect that from the inside. Deep
networks, out of the box, live below the diagonal; that's been known since
2017.

→ *So how do we get a network to honestly say "I don't know"? That is
evidential deep learning.*

### 7. Evidential deep learning in three minutes *(~2.5 min)*
Think about what softmax cannot do. Probabilities must sum to one. So
total ignorance — "I've never seen anything like this" — and genuine
ambiguity — "I've seen it a thousand times, it's a coin flip" — produce
*the same output*. There is no way to say "no idea."

The evidential move: instead of probabilities, the network outputs
*evidence* — non-negative counts, essentially. From evidence you form an
opinion: belief per class, plus one extra mass, the **vacuity** — the
explicit "I don't know" mass. One formula to remember: vacuity is C over S —
number of classes over total evidence. No evidence: vacuity one. Mountains
of consistent evidence: vacuity zero.

**[point at the triangle]** Geometrically: softmax lives on the bottom edge
of this triangle — vacuity identically zero. Evidential learning frees the
model to move up. One extra dimension, and "never seen this" versus "seen
it, it's ambiguous" become *different outputs*.

### 8. Dempster–Shafer fusion in three minutes *(~2 min)*
Now, maps are built from many observations — so we need to *combine*
opinions. Dempster's rule: agreement multiplies belief, and each side's
vacuity lets the other side speak. **[walk the table]** Two sensors both
say "chair," each with some vacuity — fused: belief 0.94, vacuity down to
0.06. Agreement compounds confidence. Exactly right for fusing many views
of a stable object.

**[the cliffhanger block — slow down]** But here's a question to hold onto:
what if the two sources *confidently disagree*? Sensor A: definitely a
chair. Sensor B: definitely a box. Intuition says the fused opinion should
scream "something is off here." Dempster's rule says... confident
compromise. That's Zadeh's paradox, it's fifty years old, and it comes back
in the second half of this talk as a *real bug we hit*. Remember it —
because "confident disagreement across views" is precisely the statistical
signature of an object that moved.

---

## Section: The Problem — 4 min

### 9. Motivation I: the scenario *(~2 min)*
Now the concrete setting. A drone relocalizing in a lived-in room. The map
is a semantic Gaussian-splatting map, trained once. Between mapping and
flight, life happened **[point at the sketch]**: green is what stayed,
orange is what moved. And the pose estimate is not consumed by a human —
it goes into a Kalman filter, fused with the IMU, hundreds of times per
minute. The filter's question is never just "where" — it is "where, *and
with what covariance*." Note the irony: the regions that mislead
localization are exactly the ones that changed — and the map has seen
enough to *know* that. Today's maps just have no vocabulary to say it.

### 10. Motivation II: what goes wrong *(~2 min)*
Three failure modes without uncertainty. One: no covariance means the
filter assumes a fixed noise model — so it is *overconfident exactly when
the scene is ambiguous, occluded, or changed*, which is when accuracy
actually degrades. Two: the existing remedy for scene change is a hard
mask — an LLM decides which objects are "reliable" — discrete in or out,
and the selector itself has documented failure cases. Three: when a room
has two similar corners, the error becomes *multi-modal* — "I'm here or
there" — and neither a point nor a single Gaussian can say that.

**[right column]** So three requirements: calibrated, graded, multi-modal.
Everything in USGLoc is designed backwards from these three.

---

## Section: Related Work — 7 min total (incl. gap slide)

*(These are table slides — do NOT read them. Walk one row each, let the
audience skim the rest, land the punchline block.)*

### 11. Related work I: visual localization *(~1.5 min)*
Four families, and for each I want you to look at the last column — the
limitation. Pose regression: fast, but silently overconfident away from its
training views. Scene coordinate regression — the accuracy kings — but
notice ACE-G: it *has* per-coordinate uncertainty, and uses it only to
throw away bad correspondences internally. Feed-forward matchers like
MASt3R: confidences again — consumed internally again. Semantic methods:
objects and scene graphs, but no metric distribution. **[punchline]** The
pattern, across all four: uncertainty is an *ingredient*, never a
*product*. Nobody hands the robot a calibrated covariance.

### 12. Related work II: rendering-based localization *(~2 min)*
Same exercise for the rendering world. NeRF refiners: precise but slow.
Particle filters — Loc-NeRF, MCLoc — this one hurts: they maintain a
genuine pose belief, hundreds of hypotheses... and then report only the
single best one. The belief dies before the output. The Gaussian-splatting
wave — two thousand twenty-four to twenty-six, half a dozen systems,
sub-centimeter accurate — all point outputs, all assuming the scene never
changes. And the closest work to ours, UGS-Loc at CVPR this year: it
renders Fisher-information uncertainty maps and uses them to guide its
search — genuinely clever — but the uncertainty is post-hoc, purely
geometric, and internal. Output: one pose. No covariance, no calibration,
no semantics. **[semantic row]** And the semantic pair — SGLoc, Chu & Kim —
robust to change, but via that discrete LLM gating from earlier.

### 13. Related work III: uncertainty methods *(~1.5 min)*
And the uncertainty community itself. Bayesian pose regressors: real
distributions, uncalibrated, accuracy-limited. Probabilistic PnP and
conformal methods: rigorous — for *objects*, not scenes. Wu and Campbell
showed pose error is non-Gaussian and needs a mixture — in 2D retrieval;
we lift exactly that insight to 6-DoF. Evidential learning has reached
scene-coordinate regression. And the mapping community — the BKI line,
E²-BKI, continuous semantic splatting — builds genuinely trustworthy maps
that *never localize anything*. **[punchline — slow]** Rigorous uncertainty
exists on both sides of the pipeline. It just never travels from the map to
the pose the robot consumes. Our job is precisely that transport.

### 14. The open cell *(~2 min)*
Here is the same story as a picture. **[walk the quadrants]** Top-left:
uncertainty-aware mapping. Bottom-right: accurate point-estimate
localization. Wu & Campbell: distributions, 2D only. UGS-Loc: on the
localization side, uncertainty inside, point out. The top-right cell —
calibrated SE(3) localization against a splatting map — is, to the best of
a rather exhaustive literature search, empty. **[claim block]** That is the
claim, precisely scoped: first localizer to *output* a calibrated pose
distribution from a Gaussian-splatting map, with evidential semantics
propagated end-to-end. Every word in that sentence is load-bearing.

---

## Section: USGLoc — 12 min

### 15. Key idea *(~2 min)*
If you remember one slide, make it this one. The vacuity — that "I don't
know" mass from the tutorial — is born when an observation is encoded, and
then *never dropped*. **[walk the chain left to right]** Fused per object —
with a conflict correction I'll show you. At matching: it gates
correspondences. At refinement: blended into a per-pixel weight — that's
the rendering equation from earlier, carrying trust instead of color. At
the optimum: a covariance. At the output: a mixture. Six stages, one
channel, each consuming uncertainty in its own natural form. And note
what's absent: no dynamic-object heuristics, no LLM deciding what's
reliable. The map's own uncertainty does the gating.

### 16. Contributions *(~1 min)*
Formally, four contributions **[gesture down the list, don't read]**: the
calibrated-distribution localizer itself; vacuity-gated matching that turns
retrieval ambiguity into an explicit pose mixture; the vacuity-weighted
refinement with a Laplace covariance; and — maybe the sneaky-important one —
a calibration-centric evaluation protocol, because you cannot claim honesty
without measuring it.

### 17. At a glance *(~1.5 min)*
The architecture. **[trace fig. 1]** Green: query and map both encoded into
the same evidential space — same head, comparable vacuities. Blue: coarse —
gated matching, top-k hypotheses with weights. Orange: fine — each
hypothesis refined by rendering, each getting its own covariance. Gold: the
mixture, condensed to a single Gaussian for the filter.

### 18. Problem formulation *(~1 min)*
One formal slide. We estimate the posterior over SE(3), not the argmax —
subject to calibration, which is a *testable* property. Both sides carry
the same quintuple — geometry, appearance, semantics, and vacuity. The
point of the symmetry: uncertainty is first-class on both sides *before*
matching starts. That's what makes propagation possible at all.

### 19. Method 1a: shared evidential head *(~1 min)*
The formulas you've already met — Dirichlet evidence, belief, vacuity.
Two things to add. The same head encodes map observations at build time
and Fast-SAM crops at query time — so map vacuity and query vacuity are on
the same scale; the gate later depends on that. And the softmax contrast is
now an *ablation*, not philosophy: swap EDL for softmax, watch calibration
collapse.

### 20. Method 1b: primitives + the Zadeh payoff *(~2 min)*
Building object-level primitives: cluster the splats, average the features
weighted by confidence, fuse the opinions with Dempster's rule. And here —
**[the caveat block]** — the cliffhanger pays off. Remember the question:
what happens when opinions confidently disagree? We hit this with real
moved objects: half the views say chair, half say box... and Dempster's
rule fuses them into *low* vacuity. The moved object looks *more* certain
than its members. The conflict — the eta in the denominator — is
renormalized away. The fix is one line: track the conflict, add it back —
effective vacuity, min of one and u plus eta-bar. Equivalently, Yager's
rule. Fifty-year-old paradox, one-line fix, and without it the entire
downstream weighting points the wrong way. Numbers in the backup.

### 21. Method 2: gated matching *(~1.5 min)*
Descriptors: CLIP feature plus geometry through small MLPs, same space for
2D and 3D. The score: cosine similarity times the evidential gate —
e to the minus lambda times *both* vacuities. Confident-confident pairs
dominate; a moved object on either side whispers instead of shouting.
Scene-level retrieval inherits the gate through attention pooling, and
ambiguity comes out as several comparably-scored submaps — which we *keep*,
as a mixture, instead of pretending the best one is certain.

### 22. Method 3: refinement *(~1.5 min)*
The promised trick. **[equation]** Vacuity alpha-blended exactly like
color — same kernel, one extra channel, essentially free. Now every
rendered pixel knows how much the map trusts it, and the photometric error
is weighted by e to the minus gamma U. Moved furniture pulls on the pose
with almost zero force. Continuous — requirement R2 — no mask, no
threshold. One subtlety that bites: the weights must be *detached* each
iteration. If gradients flow through them, the optimizer discovers it can
hide misalignment in high-vacuity regions instead of fixing the pose.
Classic IRLS discipline; details in backup.

### 23. Method 4: Laplace + temperature *(~1.5 min)*
From optimum to covariance: Laplace approximation — the curvature of the
loss. Flat basin, big covariance; sharp basin, tight. And the residual
scale means degraded queries automatically get inflated covariances.
**[honesty block]** But raw Laplace *lies*: real photometric residuals are
heavy-tailed and correlated, so it's overconfident. One scalar temperature,
fitted on a held-out split with a median estimator, fixes the scale. On our
synthetic benchmark that's ECE from 0.50 to 0.16 — from useless to honest —
while the *ranking* quality was good all along. Informative before,
honest after. You need both.

### 24. Method 5: the mixture *(~1 min)*
Assemble: weights from retrieval times render fidelity, prune what doesn't
re-render, and you have a Gaussian mixture on SE(3). When the scene is
ambiguous the mixture is visibly multi-modal — kept, not averaged. For the
filter: moment-matched to one Gaussian. The mixture is the honest answer;
the condensation is the practical one; you get both.

### 25. Algorithm *(~30s)*
The whole thing on one slide for reference — steps one to three learned,
four to six training-free. I'll leave it up a beat. **[pause, drink water]**

---

## Section: Evaluation — 7 min

### 26. Datasets *(~1.5 min)*
Standard benchmarks plus a semi-dynamic protocol — objects rearranged
between mapping and query, plus deliberately degraded maps, where the test
is not just accuracy but whether the covariance *grows* when it should.
**[trap block]** One community service announcement: 7Scenes has two
incompatible ground-truth conventions, and the same method can differ by
sixteen accuracy points between them. Everything modern reports SfM
pseudo-ground-truth; mixing the columns silently is the most common
citation error in this literature. Check your baselines' fine print.

### 27. Metrics *(~2 min)*
Accuracy as usual. Then the headline: four calibration metrics, each
answering one question. Coverage: is the covariance honest — do the
chi-square regions catch the truth at the advertised rate? ECE: the same,
as one number. AUSE **[sketch]**: is the uncertainty informative — if I
drop the queries the model *says* are uncertain, does error fall like the
oracle? NLL: the whole distribution, multi-modality included.
**[block]** And the reason you need both families: a covariance can rank
errors perfectly while being ten times too small, and vice versa. Honest
*and* informative — that's two separate exams.

### 28. Where the field stands *(~1.5 min)*
The accuracy ladder on 7Scenes, all as published. Down to SplatHLoc at
half a centimeter. Now the right column **[run finger down]**: none, none,
none, none, internal-only, none. Our goal is *not* to win this column by a
tenth of a millimeter — it's to sit in that group while being the only row
with something other than "none" on the right. A fair fight on accuracy,
no contest on honesty.

### 29. Status & synthetic evidence *(~1.5 min)*
Where we are: the full pipeline is implemented and unit-tested; the
numbers here are from a controlled synthetic benchmark — real-data results
are the current work, so consider this the mechanism check, honestly
labeled. Three things it already shows. The map separates moved from
static content by vacuity alone — 0.67 versus 0.16, no thresholds. The
mixture earns its keep — force it to one hypothesis and accuracy drops
by a third. And the temperature story **[table]**: raw Laplace ranks
well but lies; tempered does both; a fixed covariance — what a filter
assumes today — can't rank at all.

---

## Section: Outlook — 3 min

### 30. Open problems *(~2 min)*
Where this opens up beyond one system. Calibration under drift — a
temperature fitted today may be wrong in a month; self-recalibrating maps
are open. Foundation models have the same disease — VLM confidences are
uncalibrated too; same medicine might apply, bigger patient. Uncertainty
as a control signal — vacuity literally tells the robot where looking
would help; that's active perception on top of evidential maps. Guarantees
— conformal methods promise coverage by construction; nobody has done that
for scene-scale pose posteriors. And closing the loop with the filter and
real flights. **[invitation]** If any of these touches your research —
find me afterwards. Several of these are thesis-shaped.

### 31. Takeaways *(~1 min)*
Four sentences. Match the output to the consumer — filters need
distributions, not points. Give the map an explicit "I don't know" channel
and *use it everywhere*. Make uncertainty survive to the output — and
temper it, because raw Laplace flatters itself. And respect the machinery —
Dempster hides conflict, gradients cheat through weights; the difference
between elegant math and a working system is exactly these two footnotes.
Thank you — questions welcome.

---

## Backup slides — when to pull them

- **Dempster numerically** — if anyone questions the Zadeh story or asks
  "why not always Yager?" (Answer: for *agreeing* sources Dempster's
  reinforcement is desirable; we keep it and track conflict separately —
  the effective-vacuity form keeps both behaviours.)
- **IRLS detachment** — if an optimization person asks about
  differentiating through the weights.
- **Training objectives** — if someone asks what is learned vs.
  training-free.

## Anticipated Q&A

1. **"Why not just use a particle filter — you'd get a distribution for
   free."** Particles give an *empirical* belief that is uncalibrated and
   expensive — hundreds of renders per query; and existing ones report only
   the MAP anyway. We get the mixture from k refined hypotheses plus
   analytic Laplace components, and then *verify* calibration. Nothing
   stops a particle back-end from adopting our vacuity weighting, though —
   good extension.
2. **"How is this different from UGS-Loc, concretely?"** Three axes:
   *where* uncertainty comes from (their Fisher information is geometric,
   post-hoc, from a fixed map; our vacuity is evidential and semantic,
   built at mapping time), *what it's used for* (they guide internal
   search; we also gate matching, weight pixels, and shape the output), and
   *what comes out* (their best particle; our calibrated GMM). We also cite
   them as the strongest accuracy baseline.
3. **"Isn't a scalar temperature too crude?"** It's the honest minimum: one
   parameter, fitted on held-out data, huge calibration gain. Per-DoF or
   learned covariance heads are listed as future work — but every added
   parameter is another thing that can silently overfit the calibration
   split.
4. **"Does vacuity capture geometric ambiguity — textureless walls?"**
   Partially and indirectly — through the residual scale in the Laplace
   step, since photometric flatness inflates the covariance. The semantic
   vacuity targets *content* trust. The two are complementary; fusing an
   explicit geometric term (à la Fisher information) with the evidential
   one is an open extension.
5. **"Why should the filter trust your covariance more than a tuned
   constant?"** That is exactly what the fixed-cov baseline tests: same
   poses, best possible constant covariance — its AUSE is ~10× worse,
   meaning it cannot distinguish good from bad queries. A tuned constant
   is calibrated *on average* and wrong on every individual query.
6. **"7Scenes is saturated — why evaluate there?"** For comparability on
   accuracy (everyone reports it) — the *calibration* columns are where the
   contribution lives, and those are new on any benchmark. The semi-dynamic
   protocol is the differentiating testbed.
7. **"Real-data results?"** In progress; the synthetic benchmark validates
   the mechanism (dynamic-object vacuity, mixture behaviour, temperature).
   The 7Scenes harness and map-bridge format are already in the released
   code.

## Cut list for a 25-minute slot

Drop, in order: frame 4 (loc 101 — fold one sentence into the hook),
frame 8 (DST primer — the Zadeh payoff in Method 1b carries alone),
frames 11–13 (related-work tables — keep only frame 14, the open cell),
frame 18 (problem formulation), frame 25 (algorithm), frame 30 (open
problems — merge the invitation into takeaways). Result: ~24 min.
