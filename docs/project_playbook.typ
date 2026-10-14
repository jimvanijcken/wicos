// EE4C12 — Wireless Communication & Sensing — Final Project
// Working playbook for the DopNet hand-gesture classification assignment.
// Open in VS Code with the Tinymist extension for live preview.
// This is a planning document, not the final report (see "Writing the report"
// for the report's own required structure / headings).

#set page(paper: "a4", margin: (x: 2.2cm, y: 2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.")

#let accent = rgb("#0B6E78")
#let signal = rgb("#B0521A")

#let note(title, body) = block(
  fill: accent.lighten(92%), stroke: (left: 2.5pt + accent),
  inset: (x: 9pt, y: 7pt), radius: (right: 3pt), width: 100%,
  [#text(fill: accent, weight: "medium", size: 8pt, tracking: 0.05em)[#upper(title)] \ #body],
)
#let flag(title, body) = block(
  fill: signal.lighten(92%), stroke: (left: 2.5pt + signal),
  inset: (x: 9pt, y: 7pt), radius: (right: 3pt), width: 100%,
  [#text(fill: signal, weight: "medium", size: 8pt, tracking: 0.05em)[#upper(title)] \ #body],
)
#let decision(n, title, body) = block(
  width: 100%, inset: (y: 6pt), stroke: (bottom: 0.5pt + gray),
  grid(columns: (24pt, 1fr),
    [#text(fill: accent, weight: "medium", size: 8.5pt)[D#n]],
    [#text(weight: "semibold", size: 10.5pt)[#title] #linebreak() #body]
  )
)
#let paper(title, cite, use) = block(
  width: 100%, inset: (y: 6pt), stroke: (bottom: 0.5pt + gray),
  [
    #text(weight: "semibold", size: 10pt)[#title] #text(size: 8pt, fill: gray.darken(10%))[#cite]
    #v(2pt)
    #use
  ]
)

// ---------------------------------------------------------------- title ---
#align(center)[
  #text(fill: accent, size: 8.5pt, tracking: 0.15em, weight: "medium")[EE4C12 · WIRELESS COMMUNICATION \& SENSING · FINAL PROJECT]
  #v(0.2em)
  #text(size: 20pt, weight: "bold")[Radar gestures, start to submission]
  #v(0.2em)
  #text(size: 10pt, fill: gray.darken(20%))[
    Classifying four hand gestures from micro-Doppler spectrograms — the data, the decisions that carry the marks, and how two people build one report together.
  ]
]
#v(0.4em)
#grid(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr), column-gutter: 1em,
  [*Samples* \ 960], [*People* \ 6 (A–F)], [*Classes* \ 4],
  [*Report cap* \ 10 pages], [*Due* \ Week 8],
)
#v(0.2em)
#line(length: 100%, stroke: 0.5pt + gray)
#v(0.6em)

= What is being asked

Build a pipeline that names a gesture — _wave, pinch, swipe,_ or _click_ —
*regardless of who performed it*. That clause sets the evaluation protocol
(see §4). Task 1 is pre-processing (clean, normalise, optionally extract
features); Task 2 is modelling, with written arguments for both the model
and the validation procedure.

*Admissible or fail*, independent of score: unclear English, no deep neural
network tested, illegible figures, or a report that skips the prescribed
structure. Once admissible, neither polished English nor clean code earns
*extra* points — so clear the bar on both, then spend the remaining time on
argumentation and validation design.

Deliverables: a 10-page PDF report + Python code, one submission per pair,
to Brightspace. Week 7 has a lecturer feedback slot — arrive with numbers.

= What is in the data

Six archives (one per person), four gesture folders each, 40 CSVs per
folder — 960 samples total, 240 per class, perfectly balanced. Label is in
the filename: `PersonAg4s1.csv` = person A, gesture 4 (`g1`=wave, `g2`=pinch,
`g3`=swipe, `g4`=click), sample 1.

#figure(
  table(
    columns: (auto, auto, 1fr), stroke: 0.5pt + gray.lighten(40%), inset: 6pt,
    table.header([*Property*], [*Measured*], [*Implication*]),
    [Doppler axis], [800 bins, fixed], [No alignment needed across files.],
    [Time axis], [72–411 frames (med. 156)], [Ragged — must be made uniform before batching.],
    [Values], [dB, max = 0.0, floor ≈ −41], [Already per-sample normalised; only the floor is a free choice.],
    [Energy band], [bins ≈ 330–470], [≈ 82% of every input is noise floor, carrying no class signal.],
    [Sensor], [Ancortek 24 GHz FMCW, 750 MHz BW], [One sentence in the report shows you read the dataset docs.],
  ),
  caption: [Measured on the local DopNet subset — averaged over 60 sampled spectrograms across all six people.]
)

#flag("Open question for Week 7")[
  The brief mentions a test set "already separated" from training; full
  public DopNet has 3,052 files vs. 960 here. Design validation (§4) to work
  entirely within the 960 you have — treat an official test set as a bonus, not a rewrite.
]

= Task 1 — four pre-processing decisions

Each should be written up as _options → choice → evidence_ — that structure
is effectively the rubric.

#decision(1, "Crop the Doppler axis to bins ≈ 330–470", [
  Outside this band the mean is within 1 dB of the noise floor (table above).
  Cuts input rows 800→140, shrinking the first layer 5–6× and the overfit surface with it.
  Verify per-sample retained energy ≥ 99%, not just on the average.
])
#decision(2, "Fix the ragged time axis — resize or pad, test both", [
  *Resize* to 128 frames normalises gesture duration away (model sees shape only).
  *Pad/crop* with the noise floor preserves duration as a cue. These are different
  physical assumptions, not a style choice — run both through identical folds and report the gap.
])
#decision(3, "Clip the dynamic range, fit nothing on test folds", [
  Clip below ≈ −40 dB (just under the noise floor), rescale to [0,1]. Sweep 2–3
  clip values and state the choice. Any fitted scaler (`StandardScaler`, PCA) is
  fit on training folds only — fitting on everything is the classic leak.
])
#decision(4, "Explicit features: take centroid + bandwidth (see §3 below)", [
  Optional, but a feature baseline makes Task 2's CNN comparison a real result
  instead of a lone number. Of the three papers in the brief, centroid/bandwidth
  is the cheapest to implement and interpret — see the next section.
])

= The three reference papers

The brief names three papers as inspiration for Task 1's optional feature
extraction. One family is enough — pick, don't implement all three.

#paper(
  "1 — Amin, Zeng & Shan (2019), envelope features",
  [RadarConf 2019],
  [Tracks the upper/lower *envelope* of the micro-Doppler spectrogram (the
  outline traced by the strongest Doppler return over time) as a compact
  signature, classified with *k-NN under Manhattan (L1) distance* — chosen
  specifically because L1 doesn't wash out small-but-informative envelope
  values the way L2/PCA-based distances can. Shown to beat PCA and sparse
  Fourier-dictionary reconstruction as baselines. #text(fill: accent)[*Use:*]
  cheapest of the three to extract (two 1-D curves per sample) and a fast
  sanity-check baseline before investing in centroid/Chebyshev features.]
)
#paper(
  "2 — Fioranelli, Ritchie & Griffiths (2016), centroid & bandwidth moments",
  [IET RSN 10(9):1702–1710, §3.2],
  [At each time slice of the spectrogram, computes the power-weighted *centroid*
  (mean Doppler bin) and the *bandwidth* (spread around it) — two time series
  per sample — then takes their statistical *moments* (mean, variance,
  skewness, kurtosis) as a fixed-length feature vector. Originally built for
  armed/unarmed personnel classification from multistatic radar, not
  gestures, but the feature definition is domain-agnostic.
  #text(fill: accent)[*Use:*] the recommended family (D4) — ~20 lines of
  NumPy, physically interpretable, good report material (a centroid-vs-time
  plot per gesture class is an easy, legible figure).]
)
#paper(
  "3 — Pallotta et al. (2021), Chebyshev moments",
  [MetroAeroSpace 2021, pp. 182–187],
  [Computes *Chebyshev moments* — discrete orthogonal moments, so no
  continuous-approximation error, with a symmetry property that keeps
  computation cheap — directly on the cadence-velocity diagram (CVD) of each
  signal, treating it as a 2-D image rather than extracting 1-D curves first.
  Notably, it is tested on *this same DopNet dataset*.
  #text(fill: accent)[*Use:*] the strongest match to "spectrogram as an
  image" framing and the closest prior work to this exact assignment — worth
  citing even if you don't implement it, and a good stretch goal if D4's
  baseline undersells the physics features.]
)

= Task 2 — a model ladder

#figure(
  table(
    columns: (auto, auto, 1fr), stroke: 0.5pt + gray.lighten(40%), inset: 6pt,
    table.header([*Model*], [*Input*], [*Question it answers*]),
    [SVM (RBF) / kNN], [Centroid-bandwidth or envelope features], [Floor set by physics features alone.],
    [MLP], [Flattened, downsampled image], [Does convolution earn its keep over a plain dense net?],
    [Small CNN #text(fill: accent, size: 8pt)[(required)]], [140×128 normalised image], [Does learned spatial structure beat hand-crafted features?],
  ),
  caption: [All three evaluated on identical folds — otherwise the comparison is meaningless.]
)

With only 960 samples, keep the CNN small: 3–4 conv blocks (16/32/64
channels), batch-norm, light dropout, global average pooling, softmax —
not a from-scratch ResNet. Augment with time shifts/stretches, noise,
SpecAugment-style masking.

#flag("Do not flip the Doppler axis")[
  The axis is *signed velocity* — flipping reverses motion direction (a
  swipe-left becomes swipe-right) and corrupts the label. Standard image
  augmentation reflexes don't apply here; say so in the report.
]

Report accuracy and macro-F1 as mean ± std across folds, plus a confusion
matrix (expect pinch/click to confuse each other — both short, low-velocity).

#note("Environment")[
  TensorFlow 2.18 is installed, no PyTorch — use Keras unless you both install
  Torch. Pin `numpy<2.3` in `requirements.txt`; the installed 2.5.3 already
  warns on `import sklearn`.
]

= Validation — decides the grade

"Regardless of the person" specifies the *test distribution*: new user, not
seen in training. A random 80/20 split fails this badly here, because the
40 samples per person/gesture are segments cut from one continuous 30 s
recording — near-duplicates that leak across a random split and let the
model recognise the *person* instead of the *gesture*.

*Protocol:* hold out one person entirely as a final test set → leave-one-
person-out over the remaining five for model selection (5 folds, mean ±
std) → one final, single run on the held-out person.

#note("Cheap marks")[
  Report *both* random-split and leave-one-person-out accuracy and explain
  the gap — it's a real finding about inter-person variability, and costs
  one extra run. Expect random-split to look better and mean less.
]

= Writing the report

Structure is prescribed; deviating from it is an admissibility failure.

#figure(
  table(
    columns: (auto, auto, 1fr), stroke: 0.5pt + gray.lighten(40%), inset: 6pt,
    table.header([*Section*], [*Budget*], [*Must contain*]),
    [Members / Summary], [½ pg / \<200 w], [Names, emails, numbers; summary written last.],
    [ML pipeline], [1.5 pg], [*Workflow figure* — required by the brief.],
    [Task 1], [3 pg], [Four decisions as options→choice→evidence; Doppler-profile and feature-separation plots.],
    [Task 2], [4 pg], [Model ladder, validation protocol + why, results ± folds, confusion matrix, random-vs-LOPO gap.],
    [Conclusions], [\<200 w], [What worked; what more data/time would change.],
  ),
)

Figures: vector PDF not screenshots, ≥9pt axis labels, dB colour bars,
named/unit'd axes. Check legibility in greyscale. Five careful figures beat nine cramped ones.

= Working as a pair

The brief requires joint decisions + splittable tasks, and *either partner
must be able to argue any decision* — the tooling below makes that review
step cheap enough to actually happen.

#flag("Before the first commit")[
  `C:\Users\hawma` itself is a git repo pointing at an unrelated course
  remote. Create a fresh repo *inside* the WICOS folder (or move the project
  out of the home tree) before running any git command here.
]

*Repo:* one private GitHub repo for code + report, both as collaborators.
Gitignore `*.zip`, `data/`, checkpoints, `.ipynb_checkpoints` — archives stay
in OneDrive/SURFdrive, not git. Pin `requirements.txt` (incl. `numpy<2.3`).
Work on short feature branches, merge via PRs the other partner reviews.

*Notebooks* diverge badly in merges (base64 output blobs). Put real logic in
plain `.py` modules (`data.py`, `features.py`, `models.py`, `evaluate.py`);
keep notebooks thin — import + plot only. Use `nbstripout` if you must share
notebooks directly.

#note("Report tooling: Typst, in the repo, via Tinymist")[
  Lives beside the code, diffs as text, regenerated figures land on next
  compile. If a partner is allergic to git, use Overleaf instead rather than
  fighting it — don't split a Word doc for a 10-page figure-heavy report.
]

*Split by pipeline stage*, not "code vs. writing" — that split leaves one
partner unable to defend the model.

#figure(
  table(
    columns: (auto, 1fr, 1fr), stroke: 0.5pt + gray.lighten(40%), inset: 6pt,
    table.header([*Owner*], [*Builds*], [*Writes*]),
    [A], [Loader, Doppler crop, time-axis test, normalisation, feature extraction, SVM/kNN], [Task 1],
    [B], [CNN, augmentation, training loop, MLP control, confusion matrices], [Task 2],
    [Both], [Fold harness — build together, it decides every number], [Pipeline figure, summary, conclusions],
  ),
)

Keep `DECISIONS.md`: one line per choice, reason + evidence, appended as you
go — it becomes the Week 8 argumentation sections almost verbatim.

= Timeline

#figure(
  table(
    columns: (auto, 1fr, 1fr), stroke: 0.5pt + gray.lighten(40%), inset: 6pt,
    table.header([*Week*], [*Build*], [*Milestone*]),
    [5], [Loader, EDA, crop band fixed from measurement], [D1–D4 settled],
    [6], [Baselines, first CNN, LOPO fold harness], [Fold harness frozen — don't change it after],
    [7 #text(fill: signal, size: 8pt)[(checkpoint)]], [Final model, ablations], [Lecturer feedback — bring numbers],
    [8], [Report writing, figure polish], [Submit PDF + code],
  ),
)

*First session:* move the repo out of the home git tree · create GitHub repo
with `.gitignore` and pinned `requirements.txt` · write `data.py` and verify
960/240-per-class · plot one spectrogram per gesture with a dB colour bar ·
reproduce the Doppler-profile measurement on all 960 files · write the LOPO
fold generator before any model exists · start `DECISIONS.md`.

#v(0.6em)
#line(length: 100%, stroke: 0.5pt + gray)
#align(center)[
  #text(size: 7.5pt, fill: gray.darken(10%))[
    EE4C12 Final Project · Radar \& AI for HCI via Hand Gestures · Lecturer: Francesco Fioranelli \
    Dataset: DopNet, UCL Radar Group — acknowledge UCL and DopNet in the report.
  ]
]
