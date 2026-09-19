
# Travel Marketplace Supply Health Performance

**Aakanksha Baid** — Analytics Strategy Leader · [LinkedIn](https://linkedin.com/in/aakankshabaid/) · [GitHub](https://github.com/AakankshaBaid)

<img width="2038" height="553" alt="hero_banner" src="https://github.com/user-attachments/assets/eef911ca-6887-40f9-9530-4499f78ea300" />

---

## Executive Summary

This project delivers an early-warning system that flags which hotel partners are on track to earn **less profit than comparable properties nearby (compset)** — weeks before that shortfall would otherwise surface in a quarterly business review. Rather than handing a partner-success team a random list of thousands of accounts to review, it hands them a short, ranked list sized to what the team can actually act on this cycle.

**Bottom line:** working the top 20% of that ranked list catches roughly **4 in 10 true underperformers — nearly double what the same team would catch working accounts in random order — using the exact same headcount and hours already budgeted.**

This is a public, fully-synthetic demonstration of a methodology used professionally in travel-marketplace analytics. No proprietary or production data appears anywhere in this repo.

---

## North Star Metric

**Outreach Efficiency Lift** — how much further the same partner-success team gets when their time is directed by this ranking instead of unprioritized coverage. Every decision in this project is judged against whether it moves this number. Current result: **1.98x — nearly double the results, with zero added headcount.**

---

## Business Problem

- Hotel partners may quietly earn less than comparable properties nearby, often for reasons like: thin content, pricing out of step with the market, limited room availability
- Partner-success teams can fix these issues, but can't reach every partner every cycle due to time constraint
- Today, a struggling partner is typically only caught *after* the fact: a complaint, a bad review, or a disappointing number already on the books
- **The decision this supports:** which partners should a capacity-constrained team call first this week, to catch a problem while it's still fixable?

---

## Skills

- Python — pandas, NumPy, scikit-learn, XGBoost, SHAP
- Imbalanced Classification & Threshold Optimization
- Leakage-Safe ML Pipelines (ColumnTransformer, Stratified Cross-Validation)
- Model Explainability (Permutation Importance, SHAP)
- Model Validation & Robustness Testing (Calibration, Out-of-Time, Subgroup, Bias)
- Business Translation & Cumulative Gains Analysis
- Synthetic Data Engineering at Scale
- Executive Communication & Stakeholder Framing

---

## Methodology

<img width="2067" height="437" alt="approach_pipeline" src="https://github.com/user-attachments/assets/02f53011-1bf4-40c5-8f0a-344f706263e2" />

1. **Hypothesis** — whether a hotel's profit is at least more than its competitive set's average is predictable from signals a partner-success team already has: content quality, price/discount position, availability, guest experience
2. **Data** — 150,000 synthetic hotel-partner records from a seeded generative process, including a 12-month time dimension; realistic imbalance and missingness
3. **Data Quality Audit** — missingness tied to partner tenure/content investment
4. **EDA** — univariate + correlation analysis to form hypotheses pre-modeling
5. **Preprocessing** — median/mode imputation, feature scaling, one-hot encoding, leakage-safe `Pipeline`
6. **Modeling** — Logistic Regression vs. balanced Random Forest vs. weighted XGBoost
7. **Evaluation** — 5-fold stratified CV, ROC/PR curves, cumulative gains
8. **Interpretability** — permutation importance + SHAP

Full code: [`notebooks/supply_performance_analysis.ipynb`](notebooks/supply_performance_analysis.ipynb)

---

## Results

**Model comparison (held-out test set, 30,000 partners):**

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression (balanced) | 0.746 | 0.561 |
| Random Forest (balanced) | 0.736 | 0.546 |
| XGBoost (weighted) | 0.743 | 0.556 |

5-fold CV (XGBoost): mean ROC-AUC 0.737, std 0.003 — stable across 150K rows.

**Model evaluation — ROC and Precision-Recall** *(Exhibit C)*

![Model Evaluation](assets/model_evaluation.png)

**Confusion matrix @ the top-20% operating point** *(Exhibit D)*

![Confusion Matrix](assets/confusion_matrix.png)

**What drives the risk score** *(Exhibit F)*

![Top Drivers](assets/top_drivers.png)

**Risk is concentrated, not spread evenly** *(Exhibit E)*

![Capture by Decile](assets/capture_by_decile.png)

**Cumulative gains:**

| Top % Flagged | Underperformers Captured | Precision | Recall | Lift vs. Random |
|---|---|---|---|---|
| 10% | 2,043 | 68.1% | 22.7% | 2.27x |
| **20%** | **3,567** | **59.4%** | **39.7%** | **1.98x** |
| 30% | 4,820 | 53.6% | 53.6% | 1.79x |
| 50% | 6,707 | 44.7% | 74.6% | 1.49x |

---

## Model Validation

A model that looks good on paper can still fail in the real world in specific, predictable ways. Before recommending this for company-wide use, it was tested against six of the most common failure modes — the same checks a rigorous analytics team runs before asking a business to trust a scoring tool with resourcing decisions.

**Model Validation Dashboard** *(Exhibit G)*

![Validation Dashboard](assets/validation_dashboard.png)

| Validation | Question | Result |
|---|---|---|
| **Calibration** | If the model says a partner has a 70% chance of underperforming, does that mean exactly 70%? | Not precisely — which is exactly why this tool is delivered as a **ranked list**, not a raw percentage. Rankings hold up even when the underlying number is imprecise. |
| **Out-of-Time Validation** | Does the model still work months from now, or only on the data it was built on? | Holds up. Tested on 3 months of data the model never saw during training — performance matched (slightly exceeded) results on familiar data. |
| **Subgroup Performance** | Does it work equally well across every market and region, or does it quietly fail one segment? | Consistent. Performance varies by under 2 points across every market segment and region tested — no group is being underserved. |
| **Hyperparameter Tuning** | Is this the best version of the model, or is there easy performance left on the table? | Already near the ceiling. An automated search across a dozen configurations found nothing worth changing — the model isn't under-built. |
| **Class Imbalance Handling** | Underperforming partners are the minority in this data — does the model just learn to ignore them? | Corrected for. Without adjustment, the model would miss 2 in 3 true underperformers at a standard cutoff; the correction applied more than doubles that catch rate. |
| **Selection Bias** | If this were trained only on the partners a team already pays attention to, would it still work on everyone else? | Passed this specific test — with a caveat: this check should be re-run the moment the model is trained on real historical data, since real-world scoping is rarely this clean. |

**Bottom line for decision-makers:** this wasn't evaluated once and shipped — it was pressure-tested against the specific ways scoring tools typically fail in production, and it held up on every check designed around the ranked-list approach.

---

## Business Impact

- **Nearly double the results from the same team.** Directing partner outreach by this ranking captures ~98% more true underperformers than working the portfolio in random order — with the identical number of calls.
- **Proven at real-world scale.** Built and tested against 150,000+ simulated hotel-partner records, not a small pilot sample.
- **Points to the solution, not just the problem.** The top three risk drivers — content completeness, price position, and promotion depth — are all things a partner-success rep can act on directly, unlike fixed attributes like star rating.
- **Built to be trusted, not just accurate.** Stress-tested against six separate ways a scoring model can quietly mislead a business (see Model Validation) before being recommended for rollout.
- **Decision:** default to the simpler, more transparent model for anything reps or partners see directly; reserve the more complex model for cases needing a deeper, per-partner explanation.

---

## Business Recommendations

1. **Roll this out as a ranked call list sized to the team's real capacity** — not a fixed cutoff. Let available hours each cycle decide how far down the list the team works.
2. **Give reps the reason a partner was flagged, not just the flag** — content, pricing, or promotions — so outreach starts with a fix.
3. **Favor the simpler, easier-to-explain model** for anything reps or partners will see directly; save the more complex model for cases needing deeper explanation.
4. **Treat the cutoff as a staffing decision owned by the business**, not a number buried inside the model.

---

## Scoping for a Phased Rollout

- Prove the model on the highest-value, highest-volume segment first
- Expand to the full portfolio once results build stakeholder trust
- Protects credibility during the window where the model is still earning belief

---

## Next Steps

- Causal validation via A/B test (flagged vs. holdout)
- Continuous target — regression on performance gap vs. binary flag
- Cost-sensitive thresholding — expected value vs. outreach cost
- Batch scoring pipeline + dashboard, drift monitoring, retrain cadence
- Re-run the selection-bias check the moment real historical data replaces synthetic data

---

## FAQ

- **Why synthetic data?** Full control over ground truth; avoids licensing and proprietary-data conflicts.
- **Why does Logistic Regression tie XGBoost?** Underlying signal is largely additive, not interaction-driven.
- **Who sets the outreach threshold?** The business, via the gains table — not the model.

---

## Notes

Fully synthetic. No proprietary business logic from any employer appear in this repo. Methodology reflects professional practice.

---

## Exhibits

| Exhibit | Description |
|---|---|
| A | Executive Overview — north-star metric and top-line KPIs (hero banner) |
| B | Methodology — the 8-step analytical approach |
| C | Model Evaluation — ROC and Precision-Recall curves |
| D | Confusion Matrix — outcomes at the recommended 20% cutoff |
| E | Capture by Decile — where risk concentrates in the ranked list |
| F | Key Drivers — the controllable levers behind the risk score |
| G | Model Validation Dashboard — all six robustness checks in one view |

---

## Repo Structure

```
.
├── README.md
├── requirements.txt
├── assets/
│   ├── hero_banner.png
│   ├── approach_pipeline.png
│   ├── model_evaluation.png
│   ├── confusion_matrix.png
│   ├── capture_by_decile.png
│   ├── top_drivers.png
│   ├── validation_dashboard.png
│   └── validation_results.json
├── data/
│   └── synthetic_hotel_supply_data.csv
├── src/
│   ├── generate_synthetic_data.py
│   └── build_assets.py
└── notebooks/
    └── supply_performance_analysis.ipynb
```

## Running It

```bash
pip install -r requirements.txt
python src/generate_synthetic_data.py      # writes data/synthetic_hotel_supply_data.csv (150K rows, 12-month span)
python src/build_assets.py                 # regenerates assets/*.png + validation checks (~3-4 min)
jupyter notebook notebooks/supply_performance_analysis.ipynb
```

Notebook is committed with outputs rendered — readable on GitHub without running anything.
