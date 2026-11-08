# ML Semester Project — Idea Shortlist & Dataset Guide

**Course:** Machine Learning (BS-V — CS, SE & AI), Fall 2026, Sukkur IBA University
**Source of truth:** `ML_Semester_Project_Guidelines_Fall2026.pdf` (10 marks, groups of 2–3)
**Prepared:** 8 November 2026

---

## 0. Deadline reality check (read this first)

| Deliverable | Deadline in PDF | Today | Time left |
|---|---|---|---|
| Group registration + idea + proposal (1 mark) | Week 7 — 25 Sep 2026 | 8 Nov 2026 | **already past** |
| Implementation + GUI (4 marks) | Last week of November | 8 Nov 2026 | **≈ 2.5 weeks** |
| Report PDF (1 mark) | Last week of November | | same |
| Presentation + live demo (4 marks) | Before finals | | ~3 weeks |

Two consequences that drive every recommendation below:

1. **The proposal is already submitted.** If it named a dataset (this repo strongly suggests it did — a full ShopMind pipeline, trained model and dashboard already exist), *switching datasets now means redoing the proposal, the code and the report*. That is a high-risk move for 8 of the 10 marks.
2. **The rubric rewards process, not accuracy.** "You are graded on whether your process is defensible, not on accuracy." So a weak-but-honest model with a rigorous leakage audit, a real baseline, and an error analysis beats a strong model with a sloppy setup.

**Therefore:** the cheapest path to high marks is to **finish the existing ShopMind-AI project to rubric standard** (Section 6 lists every gap, all of them are hours of work, not days). The dataset ideas in this document are for (a) confirming your choice was good, or (b) a genuine switch if a section-mate already claimed Online Retail II.

---

## 1. The rules, decoded

Every row below is a hard requirement in the PDF. The third column is *how you prove it* — that is what actually gets marked.

| # | Requirement | What it means in practice | Evidence to show |
|---|---|---|---|
| R1 | Dataset ≥ 1,000 rows **and** ≥ 6 meaningful features | Row count *after* cleaning must still be ≥ 1,000 | One line of code printing `df.shape` post-cleaning |
| R2 | State source URL **and licence** | UCI page URL + licence name + DOI | Footnote in Section 1 |
| R3 | Trivial baseline you attempt to beat | Majority class / mean / one simple rule — as a **row in the results table** | Results table row |
| R4 | ≥ 3 models from ≥ 2 families, ≥ 1 classical | e.g. logistic regression + decision tree + random forest | Results table rows |
| R5 | Split decided **before** modelling | Write the split rule in Section 4 before any result exists | Split definition + `random_state` |
| R6 | Preprocessing fitted **inside the training fold only** | `Pipeline(StandardScaler(), model)` + `fit` on train only. **No** scaling/imputing before `train_test_split` | Code + a sentence in Section 2 |
| R7 | Chronological split for time series | Never `shuffle=True` on temporal data | `TimeSeriesSplit` or date cutoff in code |
| R8 | Evaluation beyond accuracy | precision/recall/F1 + confusion matrix; RMSE/MAE + residuals | Figures 4–6 in report |
| R9 | **Leakage audit** | Table: every feature → *would this exist at prediction time?* → kept or dropped | Table in Section 2 or 8 |
| R10 | Report ≤ 10 pages incl. figures & refs, numbered captioned figures/tables, referred to in text, **≥ 3 weeks of lecture content** | Cite specific week numbers for specific techniques | Section 7 mapping table |
| R11 | Documented notebooks: loading → preprocessing → training → tuning → evaluation; seeds fixed | Tuning must actually appear | `notebooks/01_...ipynb` etc. |
| R12 | Working Streamlit/Flask GUI, audience-chosen input → live prediction | Must accept input a stranger types in | Hosted link or ≤ 3 min recording |
| R13 | AI Use Disclosure | Which tool, for what, roughly how much | Short paragraph, honestly written |

**Not acceptable (auto-penalty):** re-running a public Kaggle notebook; toy datasets (Iris, Titanic, MNIST, Wine, Boston, sklearn built-ins) or **Telco Churn from Labs 2–3**; no evaluation section or no train/test separation; work already submitted for another course; two teams in one section with the same problem + dataset.

**Marks are most often lost on:** scaling/imputing before the split · accuracy-only reporting on imbalanced data · tuning on the test set · plots with no interpretation · no trivial baseline · random splits on time-series data · textbook definitions instead of your own results.

One distinction worth memorising, because it is the most commonly *stated-but-not-done* requirement: **"preprocessing inside the training fold" ≠ "no preprocessing".** It means the scaler/imputer/encoder is `fit` on train only, then `transform`ed on test.

---

## 2. Shortlist at a glance

All datasets are UCI, **CC BY 4.0** (citation + DOI required, commercial use allowed). Sizes are the actual files on the UCI pages.

| # | Problem | Dataset (UCI id) | Rows | Feat. | Task | Class imbalance | Leakage trap strength | Build effort | Switch-now risk |
|---|---|---|---|---:|---|---|---|---|---|
| **A** | 30-day hospital readmission risk | Diabetes 130-US Hospitals (296) | 101,766 | 47 | Classification | **Severe** (11.2% positive) | ★★★★★ | Medium | Medium |
| **B** | Student dropout early warning | Predict Students' Dropout & Academic Success (697) | 4,424 | 36 | Classification (3-class) | Moderate | ★★★★★ | **Low** | Low |
| **C** | Machine failure prediction | AI4I 2020 Predictive Maintenance (601) | 10,000 | 6 (+IDs) | Classification | **Severe** (3.4%) | ★★★★★ | **Low** | Low |
| **D** | Steel plant energy demand forecast | Steel Industry Energy Consumption (851) | 35,040 | 9 | Regression (time series) | n/a | ★★★★ | Medium | Medium |
| **E** | Credit card default risk | Default of Credit Card Clients (350) | 30,000 | 23 | Classification | Moderate (22.1%) | ★★★ | **Low** | Low |
| **F** | Telemarketing conversion | Bank Marketing (222) | 45,211 | 16 | Classification | Severe (11.7%) | ★★★★ | Low | Low |
| **G** | Retail product demand forecast | Online Retail II (502) | 1,067,371 | 13 engineered | Regression (time series) | n/a | ★★★ | **Already built** | **Zero** |

"Leakage trap strength" = how much genuinely non-obvious leakage auditing the dataset forces you to do. High is *good* here: R9 is a scored requirement, and a dataset with an obvious trap gives you something real to audit, find and fix — which is exactly the story the presentation rewards.

---

## 3. Idea cards

### A. Hospital 30-day readmission risk
- **Problem.** Flag diabetic inpatients who are likely to be readmitted within 30 days of discharge, so a hospital can target follow-up calls and discharge planning. Readmission is expensive and, in the US, penalised under the CMS Hospital Readmissions Reduction Program; the same triage logic applies to under-resourced public hospitals.
- **Dataset.** Diabetes 130-US Hospitals for Years 1999–2008 · `https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008` · DOI `10.24432/C5230J` · CC BY 4.0 · `diabetic_data.csv` 18.3 MB · 101,766 encounters × 47 features · 130 US hospitals, 1999–2008.
- **Target.** `readmitted` ∈ {`NO`, `>30`, `<30`}. Two framings are defensible: binary (`<30` = 1) or 3-class. Pick one and justify it; binary is easier to defend clinically.
- **Baseline.** Majority class "not readmitted" ≈ 88.8% accuracy — and a *useless* PR-AUC. That gap is your headline figure.
- **Models.** Multinomial/binomial **logistic regression** (linear, + `class_weight='balanced'`), **decision tree**, **random forest**, **gradient boosting**, optional **k-NN** → 3+ families.
- **Validation.** Split **by patient**, not by row.
- **Metrics.** PR-AUC (primary, not ROC-AUC), recall @ precision ≥ 0.20, per-class F1, confusion matrix, calibration curve + Brier score.
- **Leakage audit — the good part.**
  1. **`patient_nbr`**: ≈ 71.5k unique patients across 101,766 encounters. A random row split puts the *same patient* in train and test → inflated scores. Fix: `GroupShuffleSplit`/`GroupKFold` on `patient_nbr`, or keep only the first encounter per patient (then re-check R1: 71.5k rows, still fine).
  2. **`discharge_disposition_id`** contains "expired" codes (11, 19, 20, 21). Dead patients cannot be readmitted → these rows must be dropped *before* splitting, or the model learns "died ⇒ no readmission".
  3. **`weight`** is ~97% missing → drop the column, do not impute.
  4. **`examide` / `citoglipton`** are constant → zero variance, drop.
  5. `encounter_id` = ID → drop. `patient_nbr` = ID → drop as a feature, keep as a *group key*.
  6. `diag_1/2/3` are ICD-9 codes with hundreds of levels → map to ~9 disease chapters; otherwise high cardinality leaks patient identity.
  7. Impute inside the fold: `SimpleImputer(strategy='most_frequent')` on categoricals, **inside** `Pipeline`.
- **Error analysis material.** Plenty: readmitted patients with `number_inpatient == 0` (no prior signal at all), patients whose `diag_1` is missing, and the borderline `>30` cases that a 3-class model confuses with `<30`.
- **GUI.** Form: age band, admission type, time in hospital, number of lab procedures, prior inpatient/emergency visits, insulin status, top-3 diagnoses → predicted risk tier + probability. Audience picks a plausible patient, model answers live.
- **Risk.** Very popular dataset — check with your instructor that no other team in your section has it. Sensitive-attribute discussion (race/gender) is a maturity point, not a problem, if handled honestly.

### B. Student dropout early warning  ← *recommended alternative if you must switch*
- **Problem.** At enrolment (or end of year 1), predict which students will **drop out**, so the university can target advising and scholarship support. Directly relevant to your own institution — the strongest "why this matters" available to you, because the audience is the population.
- **Dataset.** Predict Students' Dropout and Academic Success · `https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success` · DOI `10.24432/C5MC89` · CC BY 4.0 · `data.csv` 520.7 KB · 4,424 students × 36 features · Portuguese higher-education institution, 17 undergraduate degrees.
- **Target.** `Target` ∈ {`Graduate`, `Dropout`, `Enrolled`}. Commonly reported split: 2,209 / 1,421 / 794 (verify with the snippet in Section 8 — do not copy my numbers, the rubric checks that your numbers match your code).
- **Baseline.** Majority class "Graduate" ≈ 49.9% accuracy → macro-F1 collapses. Also a one-rule baseline: 2nd-semester approved units ≥ 5 ⇒ Graduate.
- **Models.** **Logistic regression** (multinomial, `class_weight='balanced'`), **decision tree**, **random forest**, **gradient boosting** → 3 families.
- **Metrics.** Macro-F1 and balanced accuracy (headline), per-class recall for `Dropout` (the class you care about), confusion matrix, optional ordinal MAE if you justify treating the classes as ordered.
- **Leakage audit — the genuinely subtle one, and why this dataset is better than it looks.**
  - The 36 features split into **12 enrolment-time** columns and **24 curricular-performance** columns (`Curricular units 1st sem ...`, `... 2nd sem ...`). The target is the student's status **at the end of the normal course duration**. Using 2nd-semester grades to "predict" dropout is partly retrospection — most public notebooks on this dataset do exactly that and report inflated scores.
  - **Do this properly:** build **two models** — (i) enrolment-only, defensible deployment; (ii) enrolment + 1st semester, defensible at the end of year 1 — and report the precision/recall difference. That comparison *is* your report's value-add and makes your Section 6 (error analysis) and Section 8 (limitations) write themselves.
  - `Course` and `Application mode` are high-cardinality integer-coded categoricals → one-hot with `handle_unknown='ignore'`, fitted in the fold.
  - There is no student ID, so no group leak — say so explicitly in the audit (a negative finding is still evidence of auditing).
- **Error analysis.** `Enrolled` is the hard class (it is a transient state, not an outcome) — show 5–10 students predicted `Graduate` who dropped out and inspect their `Scholarship holder`, `Tuition fees up to date`, `Debtor` pattern.
- **GUI.** Student profile form → predicted outcome + dropout probability. Extremely demo-friendly; you can prefill a realistic profile in front of the audience.
- **Effort.** Lowest of all ideas: 0.5 MB, trains in seconds, no missing values, no date parsing.

### C. Machine failure prediction (predictive maintenance)
- **Problem.** Predict imminent machine failure from live sensor readings (air/process temperature, rotational speed, torque, tool wear) so a plant can service before an unplanned stop.
- **Dataset.** AI4I 2020 Predictive Maintenance · `https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset` · DOI `10.24432/C5HS5C` · CC BY 4.0 · `ai4i2020.csv` 509.8 KB · 10,000 rows × 14 columns.
- **Target.** `Machine failure` (339 positives, 3.39%). Multi-label variants `TWF/HDF/PWF/OSF/RNF` exist but are **leakage, see below**.
- **Baseline.** Always "no failure" = 96.61% accuracy, PR-AUC = 0.034. Single rule: `Tool wear > 200` ⇒ failure.
- **Models.** **Logistic regression**, **decision tree**, **random forest**, **gradient boosting**; try `class_weight='balanced'` vs no weighting as an ablation.
- **Leakage audit — the most dramatic of the whole list.**
  - `TWF`, `HDF`, `PWF`, `OSF`, `RNF` are the **five failure modes whose OR is `Machine failure`**. Leaving any of them in gives near-perfect scores that mean nothing. Dropping them is the single most important line in your report.
  - `UDI` and `Product ID` are identifiers (and `Product ID` embeds the quality variant `L/M/H`) → drop.
  - The data is **synthetic** and generated by deterministic thresholds. Disclose this in Section 1 and Section 8. It is legitimate — but say it out loud. The honest framing: "we test whether ML *rediscovers* the generating rules, and where it fails."
  - Because the failures are rule-generated, a tree can memorise the rules → this makes for a superb error-analysis section and a strong "ML vs physics" discussion.
- **Metrics.** PR-AUC, recall at precision ≥ 0.5, confusion matrix, and a **cost matrix** (missed failure ≫ false alarm) with an expected-cost-per-prediction comparison across models. That cost framing is the difference between a student project and a defensible one.
- **GUI.** Sliders for temperature/speed/torque/tool wear → failure probability + recommended action.
- **Risk.** 339 positives means a 20% test set has only ~68 positives → wide confidence intervals. Report bootstrap CIs on PR-AUC; that is a plus, not a minus.

### D. Steel plant energy demand forecasting
- **Problem.** Forecast 15-minute-ahead electricity consumption (kWh) for a steel plant so it can shift load, avoid demand-charge spikes and plan procurement.
- **Dataset.** Steel Industry Energy Consumption · `https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption` · DOI `10.24432/C52G8C` · CC BY 4.0 · `Steel_industry_data.csv` 2.6 MB · 35,040 rows (one year of 15-min readings, 2018) × 9 features · DAEWOO Steel Co., Gwangyang, South Korea.
- **Target.** `Usage_kWh` (regression).
- **Baseline.** Persistence (previous 15-min value) — and it is *strong*, because load is highly autocorrelated. Beating persistence is a real result; failing to beat it is an honest one.
- **Models.** **Ridge/linear regression** on lag features (linear), **decision tree**, **random forest**, **gradient boosting** → 3 families.
- **Validation.** **Chronological only**: train on Jan–Sep, validate Oct–Nov, test Dec. Never shuffle (R7).
- **Leakage audit.**
  - `date` → expand into hour/day/month/weekend features, but **never** use the *target's own timestamp statistics*; all lags must be `shift(1)`+.
  - `Lagging_Current_Reactive_Power` / `Leading_Current_Reactive_Power` are measured **at the same instant** as the target → for a *true* 15-min-ahead forecast they are unavailable at prediction time. Either drop them or reframe the task as "nowcasting" and say which.
  - **`Load_Type`** is a categorical derived from consumption thresholds → it is a discretised version of the target. Dropping it is mandatory and is the trap most teams fall into.
  - `CO2(tCO2)` is also co-measured → same treatment as reactive power.
- **Metrics.** MAE, RMSE, MAPE, plus residual-vs-time and residual-vs-hour-of-day plots (R8). Show the residual plot revealing a load pattern the model missed — that is a scored figure, not decoration.
- **GUI.** Pick a date + hour, or "next 24 hours" → forecast curve, overlaid on the actual.
- **Risk.** Single plant, one year → no seasonality-to-year generalisation. State it in Section 8.

### E. Credit card default risk
- **Problem.** Given six months of repayment/billing/payment history, predict whether a client defaults next month — the core of a bank's credit decision.
- **Dataset.** Default of Credit Card Clients · `https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients` · DOI `10.24432/C55S3H` · CC BY 4.0 · `default of credit card clients.xls` 5.3 MB · 30,000 clients × 23 features (Taiwan, Apr–Sep 2005 history, Oct 2005 default).
- **Baseline.** Always "no default" = 77.9% accuracy, ROC-AUC 0.5. Perfect illustration of the accuracy trap.
- **Models.** **Logistic regression** (interpretable, coefficients reportable as odds ratios), **decision tree**, **random forest**, **gradient boosting**, optional **k-NN**.
- **Metrics.** ROC-AUC, PR-AUC, **KS statistic** (banking standard), calibration curve, cost-sensitive threshold.
- **Leakage audit.**
  - `ID` → drop.
  - The features are months Apr–Sep, target is October default. This is **legitimately** leak-free *if and only if* you state the prediction point explicitly: "we stand on 1 October, we know September's bill and repayment." Write that sentence; it is the whole audit.
  - `EDUCATION` has undocumented codes 0, 5, 6; `MARRIAGE` has 0 → group into "other" (with a counted justification, not a guess).
  - No group/time overlap between clients → say so.
- **GUI.** Sliders/dropdowns for credit limit, 6 months of repayment status, bill and payment amounts → default probability + risk band.
- **Risk.** Widely used. It is *not* on the banned list (only Telco Churn, Iris/Titanic/MNIST/Wine/Boston are), but confirm section-level uniqueness.

### F. Bank telemarketing conversion
- **Problem.** Predict which clients will subscribe to a term deposit, so a bank calls the right 20% instead of all 45k — a genuine budget problem.
- **Dataset.** Bank Marketing · `https://archive.ics.uci.edu/dataset/222/bank+marketing` · DOI `10.24432/C5K306` · CC BY 4.0 · `bank.zip` 565.5 KB · use `bank-additional-full.csv` (41,188 × 20, **ordered by date**, May 2008 – Nov 2010) for a chronological split.
- **Baseline.** Always "no" = 88.3%; one rule: `poutcome == success`.
- **Models.** **Logistic regression**, **decision tree**, **random forest**, **gradient boosting**.
- **Leakage audit — the famous one.**
  - **`duration`** = length of the *last call*. The UCI documentation itself says it "highly affects the output target … should be discarded if the intention is to have a realistic predictive model." A campaign planner chooses who to call *before* the call happens, so `duration` does not exist at prediction time. Dropping it is the single highest-value audit finding available, and it visibly drops AUC — which is *good*, because it shows you were honest.
  - `day`/`month` are fine but create a time split obligation (use date order, not `shuffle`).
  - `pdays == -1` means "never contacted" → recode as a flag + NaN, do not leave -1 as a number.
- **Metrics.** PR-AUC, **lift at the top 10%/20%** and a cumulative-gains curve — the metric a marketing team actually uses.
- **GUI.** Client profile → subscription probability + "call / don't call" with the expected-lift number.
- **Risk.** Also popular; the `duration` trap is so well known that some instructors will expect you to catch it — which is fine, catch it *and quantify it*.

### G. Retail demand forecasting — *the project already in this repo*
- **Dataset.** Online Retail II · `https://archive.ics.uci.edu/dataset/502/online+retail+ii` · DOI `10.24432/C5CG6D` · CC BY 4.0 · `online_retail_II.xlsx` 43.5 MB · 1,067,371 transactions, UK online gift retailer, Dec 2009 – Dec 2011.
- **Why it is defensible.** Real transactional data, genuine business problem, chronological split, features built only from prior months, a unit test that proves no future information enters the features, and a working GUI.
- **Why it is currently under-marked.** See Section 6 — no naive baseline row, no tuning, no written leakage audit, no residual/error analysis, no EDA takeaways, no notebooks.
- **Honest weakness to own.** The published R² is ≈ 0 and RMSE is huge. The rubric explicitly does not punish this — but you must *explain* it (ratio targets with small denominators) rather than hide it. Your README already does this well; move that language into Section 8 of the report.

---

## 4. Recommendation

**Primary: stay with G (ShopMind-AI) and fix the rubric gaps.** You already own 4 of the 10 marks' worth of work (implementation + GUI). The remaining gaps are documentation and two small modelling additions, all listed in Section 6. Switching now spends your last 2.5 weeks rebuilding what exists.

**If you must switch, pick B (student dropout)** — smallest data, fastest training, no date parsing, and the enrolment-vs-semester leakage comparison gives you the strongest possible Section 6/7/8 story. Second choice **C (AI4I)**, because the `TWF/HDF/PWF/OSF/RNF` leakage is spectacular and easy to explain on a slide.

**Do not pick** anything smaller than 1,000 rows after cleaning, any sklearn built-in, Iris/Titanic/MNIST/Wine/Boston, or Telco Churn — all explicitly banned.

---

## 5. Cross-cutting design (applies to whichever you choose)

**Model families — satisfy R4 without thinking:**

| Family | Model | Why it earns marks |
|---|---|---|
| Linear | Logistic / Ridge regression | The course's classical model; interpretable coefficients you can quote |
| Tree (single) | Decision tree (`max_depth` tuned) | Gives you a readable rule — great for the "simple rule" baseline contrast |
| Tree (ensemble) | Random forest | Bagging, feature importance |
| Tree (boosted) | Gradient boosting | Your likely best model |
| Instance-based | k-NN (scaled) | A 5th family for cheap; shows the scaling requirement concretely |
| Baseline | Majority / mean / one rule | R3 — required row |

**Beat-the-baseline table** you can paste into the report:

| Model | Family | Primary metric | Secondary | Baseline beaten? |
|---|---|---:|---:|---|
| Majority-class / mean | Baseline | … | … | — |
| One simple rule | Baseline | … | … | … |
| Logistic / Ridge | Linear | `model.pkl` | | |
| Decision tree | Tree | | | |
| Random forest | Tree ensemble | | | |
| Gradient boosting | Boosted trees | | | |
| k-NN (optional) | Instance-based | | | |

**Leakage audit table format** (this is a deliverable, not a paragraph):

| Feature | Available at prediction time? | Verdict | Reason |
|---|---|---|---|
| `monthly_units_sold` (target month) | No | Dropped (it is the target) | — |
| `rolling_mean_3` | Yes (months t−3…t−1) | Kept | uses `shift(1)` |
| `patient_nbr` | No (identifier) | Dropped as feature | kept as group key |
| `discharge_disposition_id` = expired | n/a | **Rows removed** | dead patients cannot be readmitted |

**The four figures that earn R8 and R10 marks** (do not submit a plot without a one-line takeaway under it):
1. Class balance / target distribution → justifies your metric choice.
2. Model comparison bar chart with the baseline drawn as a dashed line → shows the baseline was beaten (or not).
3. Confusion matrix, or residual-vs-predicted plot → shows *where* it fails.
4. Error analysis panel: 5–10 specific wrong predictions in a table, each with one sentence of "why".

---

## 6. Existing ShopMind-AI — rubric gap analysis

Honest status against every rule. "Effort" is realistic working time for one team member.

| Rule | Current status in this repo | Action needed | Effort |
|---|---|---|---|
| R1 rows/features | ✅ 1,067,371 transactions → 73,679 product-months, 13 features | None | — |
| R2 URL + licence | ⚠️ URL in README, **licence not stated** | Add "CC BY 4.0, DOI 10.24432/C5CG6D" to README + report §1 | 5 min |
| R3 trivial baseline | ❌ Baseline exists only as an app fallback, **not in the results table** | Add `seasonal-naive` (same month last year) **and** `last-3-month mean` rows to the training comparison. Both must be evaluated on the identical validation window | 1–2 h |
| R4 3 models / 2 families | ⚠️ LR + RF + GB = 2 families, all default hyperparameters | Add a **single decision tree** and optionally **k-NN** on scaled features | 1 h |
| R5 split before modelling | ✅ Chronological: pre-Sep-2011 train, Sep–Dec-2011 validation | Document it in §4 | — |
| R6 preprocessing in-fold | ⚠️ No scaler/imputer is used at all, so nothing leaks — but this is unproven in writing | State "no scaler or imputer is fitted on validation data"; if you add k-NN, wrap in `Pipeline` | 30 min |
| R7 chronological | ✅ `month_date <` cutoff in `src/forecasting.py` | None | — |
| R8 beyond accuracy | ⚠️ MAE/RMSE/R² only; **no residual plot, no error analysis** | Add residual-vs-predicted, residual histogram, and the 5–10 worst-prediction table | 2–3 h |
| R9 leakage audit | ❌ Not written (though the code is genuinely leak-free and `tests/test_pipeline.py` *proves* it) | Write `docs/leakage_audit.md` from the table format in §5. Cite the existing test as machine-checked evidence — that is stronger than a paragraph | 1–2 h |
| R10 report | ❌ Not started | Use the skeleton in §7. Must cite ≥ 3 lecture weeks | 2–3 days |
| R11 notebooks + tuning | ❌ Python modules only; **no hyperparameter tuning anywhere** | 5 notebooks: 01 load, 02 clean+features, 03 EDA, 04 train+tune (`TimeSeriesSplit` + `GridSearchCV`, tuning on validation only), 05 evaluate + errors. Set global seeds | 1 day |
| R12 GUI | ✅ Streamlit dashboard, CSV upload + budget input, live output | Record a ≤ 3 min demo; host on Streamlit Community Cloud if possible | 2 h |
| R13 AI disclosure | ❌ Missing | 4-line paragraph | 10 min |
| EDA with takeaways | ❌ Missing | 4 figures, each with a one-line takeaway (§5) | 2 h |

**Net: roughly 2–3 focused days** to go from "good code, thin report" to a defensible submission — versus ~2 weeks to rebuild on a new dataset.

**One caution on R3.** Adding a baseline may *lower* your headline claim: the README reports Gradient Boosting ratio-MAE 2.9534. A seasonal-naive baseline on the same rows might be close to that, and if it is, you must report it. That is exactly what "defensible process" means, and it is the single most likely place a team loses marks by omission rather than by error.

---

## 7. Report skeleton (10 pages, 8 required sections)

| § | Content | Pages | Figures/tables |
|---|---|---|---|
| 1 | Task, dataset, why it matters. Include URL + licence + DOI here. | 1.0 | Table 1: dataset summary |
| 2 | Preprocessing: cleaning, encoding, scaling — **with reasoning**, plus the `Pipeline` statement | 1.25 | **Table 2: leakage audit** |
| 3 | EDA: 4 figures, each with a one-line takeaway | 1.25 | Fig 1–4 |
| 4 | Models compared, training objective, validation strategy (split defined *before* results) | 1.0 | Table 3: model/family table |
| 5 | Results: full comparison table **including both baselines** | 1.0 | Table 4 + Fig 5 (bar chart vs baseline line) |
| 6 | Error analysis: 5–10 wrong predictions and **why** each failed | 1.25 | Table 5 + Fig 6 |
| 7 | Techniques from class applied — **≥ 3 week numbers**, each tied to a specific decision | 0.5 | Table 6: week → technique → where used |
| 8 | Challenges, limitations, how addressed (own the weak R²) | 0.75 | — |
| — | References (UCI DOI, scikit-learn, paper) + AI Use Disclosure | 0.5 | — |

Layout rules that save pages: two figures side by side at 45% width; tables at 9 pt; no code screenshots (explicitly banned); number and caption everything, and refer to it in the text ("Table 4 shows…").

Section 7 mapping, ready to fill:

| Lecture week | Technique | Where it is used in your project |
|---|---|---|
| Week __ | EDA / data quality | §3 |
| Week __ | Missing values & encoding | §2 |
| Week __ | Train/test split & leakage | §4, Table 2 |
| Week __ | Linear/logistic regression | §5 |
| Week __ | Trees & ensembles | §5 |
| Week __ | Cross-validation & tuning | §4/NB04 |
| Week __ | Regression metrics / classification metrics | §5 |
| Week __ | Class imbalance / cost-sensitivity | §5 (if applicable) |

---

## 8. Verify every number yourself (the rubric checks this)

The PDF: "Numbers in the report that do not match the code mean a zero." So never copy a statistic from a blog, a paper, or from this document — print it.

```python
import pandas as pd

# A — Diabetes readmission
df = pd.read_csv('diabetic_data.csv')
print(df.shape)                              # expect (101766, 50)
print(df['readmitted'].value_counts(normalize=True).round(4))
print('unique patients:', df['patient_nbr'].nunique())
print('expired dispositions:', df['discharge_disposition_id'].isin([11, 19, 20, 21]).sum())
print('weight missing %:', round(df['weight'].eq('?').mean() * 100, 1))

# B — Student dropout
d = pd.read_csv('data.csv', sep=';')
print(d.shape, d['Target'].value_counts())

# C — AI4I predictive maintenance
a = pd.read_csv('ai4i2020.csv')
print(a['Machine failure'].mean(), a['Machine failure'].sum())
print(a[['TWF','HDF','PWF','OSF','RNF']].sum())

# D — Steel energy
s = pd.read_csv('Steel_industry_data.csv')
print(s['Usage_kWh'].describe(), s['Load_Type'].value_counts(normalize=True))

# E — Credit default
c = pd.read_excel('default of credit card clients.xls', header=1)
print(c['default payment next month'].mean())

# F — Bank marketing
b = pd.read_csv('bank-additional-full.csv', sep=';')
print(b['y'].value_counts(normalize=True))
```

Then immediately paste those printed numbers into your report — that is your reproducibility claim.

**Loading a UCI dataset in Colab** (CPU-friendly, no Kaggle account needed): download the ZIP from the dataset page, upload to Colab, or use `pip install ucimlrepo` and `fetch_ucirepo(id=...)`. Keep `data/raw/` out of Git — this repo already ignores it.

---

## 9. 2.5-week plan (8 → 26 Nov 2026)

| Days | Work |
|---|---|
| 8–10 Nov | Decide: stay (G) or switch (B/C). Write the leakage audit table for whichever you choose. Add the two baseline rows (this is the highest marks-per-hour item in the whole plan). |
| 11–13 Nov | Add decision tree (+ k-NN in a `Pipeline`). Add tuning with `TimeSeriesSplit` (or `GroupKFold`/`StratifiedKFold`) — **tuning data must never be the test window**. Set seeds. |
| 14–16 Nov | EDA notebook with 4 figures + one-line takeaways. Residual plots. |
| 17–19 Nov | Error analysis: extract 5–10 worst predictions, write one sentence each on *why*. This section is worth more than another 0.01 of MAE. |
| 20–22 Nov | Write report §1–5. Number all figures/tables. Fill the lecture-week mapping (ask your instructor for exact week numbers if unsure). |
| 23–24 Nov | §6–8, references, AI disclosure. Page-count audit (must be ≤ 10). |
| 25 Nov | GUI recording ≤ 3 min; host the app if possible. Full read-through: every number in the PDF vs the code. |
| 26 Nov | Buffer day. Submit. |

Leave the last two days empty on purpose — "10% per day, zero after three days" is not a forgiving policy.

---

## 10. Pre-submission checklist

- [ ] Dataset: source URL + licence + DOI stated in the report
- [ ] ≥ 1,000 rows and ≥ 6 features **after** cleaning (row count printed)
- [ ] Trivial baseline **in the results table**, evaluated on the same test window
- [ ] ≥ 3 models, ≥ 2 families, ≥ 1 classical
- [ ] Split defined before modelling; preprocessing fitted **inside** the training fold only
- [ ] Chronological split used for any time-series data (no `shuffle=True`)
- [ ] Metrics beyond accuracy: P/R/F1 + confusion matrix, or MAE/RMSE + residuals
- [ ] **Written leakage audit**, with a verdict per feature
- [ ] 5–10 wrong predictions analysed with reasons
- [ ] ≥ 3 lecture weeks explicitly referenced
- [ ] Report ≤ 10 pages, figures/tables numbered, captioned, and referred to in text
- [ ] No code or output screenshots
- [ ] Notebooks: load → clean → train → **tune** → evaluate, seeds fixed
- [ ] Streamlit/Flask app accepts audience input and returns a live prediction; recording or hosted link ready
- [ ] AI Use Disclosure written
- [ ] Every number in the report matches a fresh run of the code
- [ ] Team lead nominated; all members can explain every decision (the presentation is the safeguard — a section you cannot explain scores zero)
