# EE4C12 WICOS Project Plan — Radar Hand-Gesture Classification (DopNet)

Team: Jim van IJcken, Huib Maas
Deadline: Week 8 (report PDF + Python code on Brightspace; late = grade 0)
Report template: `report-template.tex`

Each step lists what it produces and which report section it fills.

## Phase 0 — Setup

- [x] **1. Python environment.** Use `C:\Users\jimva\miniconda3\python.exe` (has numpy, pandas). Still install: `pip install matplotlib scikit-learn scipy torch ipykernel pygments` (pygments is needed for `minted` in the report).
- [ ] **2. Project structure.** Create `figures/` for report figures. Use git or a shared repo so we don't overwrite each other's work.
  - *Progress:* `figures/` created (notebook saves PDFs there via `FIG_DIR`); data moved to `data/`. No git repo yet.
- [ ] **3. Read the DopNet GitHub page** (<https://github.com/UCLRadarGroup/DopNet>). Note the radar parameters and what the spectrogram axes mean; this is needed to argue a meaningful pipeline.
- [ ] **4. Split the work.** Suggestion: Jim on Task 1 + classical models, Huib on the DNN. Make decisions together — each of us must be able to defend any choice.

## Phase 1 — Load and explore the data → *Task 1: Exploratory Data Analysis*

- [x] **5. Loader.** First cells of `wicos.ipynb`: loads all CSVs (no header row) from `data/` into `spectrograms` (list of numpy arrays) and `metadata` (DataFrame: `person`, `label` 0–3, `gesture` name). Labels from file names (`g1`=wave, `g2`=pinch, `g3`=swipe, `g4`=click). 960 samples confirmed (6 persons × 4 gestures × 40).
- [x] **6. Shapes.** Every sample has 800 Doppler bins; time bins vary from 43 to 448. Stacked histogram of lengths per gesture → `figures/length_histogram.pdf` (click shortest, swipe longest).
- [x] **7. Data quality.** No NaN/inf found. dB range −121.3 to 0.0; every sample's maximum is exactly 0 dB, i.e. already normalised to its own peak. Magnitude distribution per gesture with 50/90/95/99th percentiles → `figures/magnitude_histogram.pdf` (basis for the dB floor in step 13 and `vmin`).
- [x] **8. Example figure.** 1×4 plot, one spectrogram per gesture, labelled axes, colourbar → `figures/spectrograms.pdf`. Still to check: font legible at page width; `vmin=-60` should be set from the magnitude percentiles. Optional: same gesture from different people.
- [ ] **9. Write the EDA subsection** while fresh — these observations justify the pre-processing in Phase 3.
  - *Progress:* the three figures are included in `report-template.tex`; interpretation written in the notebook for shape and time bins. Still to do: magnitude-distribution interpretation, and the report text.

## Phase 2 — Decide the data split (before any training) → *Pipeline + "Appropriate use of data" (10%)*

- [x] **10. Choose the split.** Implemented and explained in the notebook's *Data split* section:
  - Test set: own hold-out, 20% (192), stratified on person × gesture (8 per combination) → `idx_test`. Used once, at the end.
  - Validation: stratified 5-fold CV on the 768 training samples (`kfold`) for choosing models and hyperparameters.
  - Leave-one-subject-out (`loso`, 6 folds, groups = `persons`) as an extra check of generalisation to unseen people, with the chosen settings; not used for selection.
  - Fold indices are positions within `idx_train` → use `idx_train[tr]`. `X` is built later (after pre-processing) and indexed with the same split.
- [x] **11. Fix random seeds** everywhere. TAs will rerun the code and must reproduce the reported results.
  - *Done:* `SEED = 42` in the config cell (`np.random.seed`, split and k-fold `random_state`). Add `torch.manual_seed(SEED)` when the CNN is built.

## Phase 3 — Pre-processing → *Task 1: Pre-processing (10%)*

- [ ] **12. Fixed size.** Crop, pad or resize the time axis to a fixed width; justify the choice.
- [ ] **13. Dynamic-range limit.** Floor at e.g. max − 40 dB to suppress noise.
- [ ] **14. Normalise.** Per-sample min–max, or a global z-score with statistics from the training folds only (anything else leaks data).
- [ ] **15. Optional Doppler crop** to the bins around zero velocity where the energy is.
- [ ] **16. Show that it helps.** Before/after figure, and if possible a small ablation of validation accuracy with and without each step.

## Phase 4 — Feature extraction and selection → *Task 1: Feature Extraction and Selection (15%)*

- [ ] **17. Implement at least two feature groups** from the suggested papers:
  - Envelope features (upper/lower/central envelope) — Amin et al., RadarConf 2019.
  - Doppler centroid and bandwidth over time, with mean/std/skewness/kurtosis — Fioranelli et al., IET RSN 2016, Sec. 3.2.
  - Optional: Chebyshev moments (Pallotta et al., 2021), or simple statistics (energy, entropy).
- [ ] **18. Feature selection.** Correlation analysis, mutual information or RFE, and/or PCA; justify the choice.
- [ ] **19. Results.** Per-class feature distributions, PCA/t-SNE scatter, importance ranking. Fill the features table.

## Phase 5 — Models → *Task 2: Model Selection and Design (20%)*

- [ ] **20. Baseline.** Logistic regression or k-NN on features.
- [ ] **21. Classical models.** SVM and random forest (or gradient boosting) on the selected features.
- [ ] **22. Deep neural network — MANDATORY (report fails without it).** Small CNN on the pre-processed spectrograms, e.g. 3 × (conv → batch-norm → ReLU → max-pool) → dropout → dense. Keep it small: the dataset is small. Optionally an MLP on features for comparison.
- [ ] **23. Document the architecture.** Layers and parameter count in a table or figure.

## Phase 6 — Training setup and hyperparameter tuning → *Task 2: Validation (10%) + Hyperparameter Tuning (20%)*

- [ ] **24. Metrics.** Cross-entropy loss; accuracy, macro-F1, confusion matrix; mean ± std over folds.
- [ ] **25. CNN training.** Adam, learning rate, batch size, early stopping on validation loss (= convergence criterion). Save train/validation loss and accuracy curves.
- [ ] **26. Tuning.**
  - Classical models: `GridSearchCV` / `RandomizedSearchCV` with CV inside the training data only.
  - CNN: tune learning rate, dropout and number of filters on the validation folds.
- [ ] **27. Record everything.** Search spaces, selected values, plot of validation score vs. key hyperparameters. Fills the hyperparameter table.

## Phase 7 — Final evaluation → *Task 2: Results and Comparison*

- [ ] **28. Retrain** each tuned model on the full training set.
- [ ] **29. Evaluate each model once** on the held-out test set. Fill the comparison table.
- [ ] **30. Confusion matrix analysis** of the best model: which gestures are confused and why (e.g. pinch vs. click are physically similar small motions). Optionally per-person performance.
- [ ] **31. Choose and argue the final model.** Accuracy vs. complexity; hand-crafted features vs. end-to-end CNN.

## Phase 8 — Mid-course feedback (Week 7)

- [ ] **32. Bring a draft** pipeline figure and early results to the lecturer feedback session. Ask whether the 10-page limit excludes the front page and references.

## Phase 9 — Write and finish the report (Weeks 7–8)

- [ ] **33. Workflow figure** for the pipeline section (vector PDF, readable text).
- [ ] **34. Fill the sections:** Pipeline → Task 1 → Task 2. Explain and reason; only short code snippets with `minted`.
- [ ] **35. Summary and Conclusions last.** Each under 200 words — count them.
- [ ] **36. Remove every `\placeholder{...}`.** Fill in emails and student numbers.
- [ ] **37. Check the fail rules:** DNN included, all figures legible, structure matches, English proof-read. Body at most 10 pages.
- [ ] **38. Code runs from scratch** in a clean environment, with a README on how to reproduce each figure and table.
- [ ] **39. Submit before the Week 8 deadline:** report PDF + Python code on Brightspace.

---

**Where the marks are:** hyperparameter tuning (20%), model selection (20%) and feature selection (15%). Spend most of the time on Phases 4–6, and write down the justification for every choice.
