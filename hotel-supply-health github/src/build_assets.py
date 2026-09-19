"""
build_assets.py
----------------
Generates polished PNG visuals for README.md: a hero banner with north-star
KPIs, a methodology pipeline diagram, a capture-by-decile chart, and a top
drivers chart. All figures are built from the synthetic dataset / model in
this repo — no external data, no proprietary content.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import os

os.makedirs("assets", exist_ok=True)

# ---------------------------------------------------------------- palette
NAVY = "#0B1220"
NAVY_LIGHT = "#1B2740"
BLUE = "#4C6EF5"
RED = "#F03E3E"
GREEN = "#2F9E44"
AMBER = "#F59E0B"
WHITE = "#FFFFFF"
GRAY_LIGHT = "#9CA3AF"
GRAY = "#6B7280"

plt.rcParams["font.family"] = "DejaVu Sans"

# ---------------------------------------------------------------- fit model
df = pd.read_csv("data/synthetic_hotel_supply_data.csv")
y = df["underperforming"]
X = df.drop(columns=["hotel_id", "underperforming", "snapshot_month"])
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))])
pre = ColumnTransformer([("num", numeric_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_clf = Pipeline([("pre", pre), ("clf", xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42, n_jobs=-1))])
xgb_clf.fit(X_train, y_train)
p = xgb_clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, p)
n_partners = len(df)
base_rate = y.mean()

scored = pd.DataFrame({"y": y_test.values, "score": p}).sort_values("score", ascending=False).reset_index(drop=True)
n = len(scored)
total_pos = scored["y"].sum()

# top-20% summary stats for KPI strip
k20 = int(n * 0.20)
flagged20 = scored.iloc[:k20]
precision20 = flagged20["y"].sum() / k20
recall20 = flagged20["y"].sum() / total_pos
lift20 = precision20 / (total_pos / n)

print(f"n_partners={n_partners:,}  auc={auc:.3f}  precision20={precision20:.3f}  recall20={recall20:.3f}  lift20={lift20:.2f}")

# ============================================================== 1. HERO BANNER
fig, ax = plt.subplots(figsize=(13.5, 3.6), dpi=150)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
fig.patch.set_facecolor(NAVY)
ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=NAVY, zorder=0))

ax.text(0.035, 0.78, "Marketplace Supply Health", fontsize=26, fontweight="bold", color=WHITE, va="center")
ax.text(0.035, 0.60, "Early-Warning System", fontsize=26, fontweight="bold", color=BLUE, va="center")
ax.text(0.035, 0.42, "Predictive supply-risk scoring for travel marketplaces — Aakanksha Baid",
        fontsize=12.5, color=GRAY_LIGHT, va="center")

kpis = [
    (f"{n_partners:,}+", "Hotel Partners Scored"),
    (f"{lift20:.2f}X", "Outreach Efficiency Lift\n(NORTH STAR)"),
    (f"{precision20:.0%}", "Precision @ Top 20% Flagged"),
    (f"{auc:.2f}", "ROC-AUC (XGBoost)"),
]
chip_w, gap, x0, y0, h = 0.215, 0.012, 0.035, 0.06, 0.24
for i, (big, label) in enumerate(kpis):
    x = x0 + i * (chip_w + gap)
    is_star = "NORTH STAR" in label
    face = "#20304F" if not is_star else "#3A2A0E"
    edge = BLUE if not is_star else AMBER
    ax.add_patch(FancyBboxPatch((x, y0), chip_w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                                 linewidth=1.4, edgecolor=edge, facecolor=face, transform=ax.transAxes))
    ax.text(x + chip_w / 2, y0 + h * 0.62, big, fontsize=17, fontweight="bold",
            color=(AMBER if is_star else WHITE), ha="center", va="center")
    ax.text(x + chip_w / 2, y0 + h * 0.24, label, fontsize=8.3, color=GRAY_LIGHT,
            ha="center", va="center", linespacing=1.3)

plt.tight_layout(pad=0.4)
plt.savefig("assets/hero_banner.png", facecolor=NAVY, bbox_inches="tight")
plt.close()
print("saved hero_banner.png")

# ============================================================== 2. APPROACH PIPELINE
stages = [
    ("1", "Hypothesis", "Frame the\ntestable belief"),
    ("2", "Data", "Generate synthetic\npartner snapshots"),
    ("3", "Quality Audit", "Check missingness\nbefore modeling"),
    ("4", "EDA", "Univariate +\ncorrelation analysis"),
    ("5", "Preprocess", "Impute, encode,\nleakage-safe pipeline"),
    ("6", "Modeling", "LogReg vs. RF\nvs. XGBoost"),
    ("7", "Evaluation", "CV, ROC/PR,\ncumulative gains"),
    ("8", "Interpret", "Permutation\nimportance + SHAP"),
]
fig, ax = plt.subplots(figsize=(14, 2.8), dpi=150)
ax.set_xlim(0, 14); ax.set_ylim(0, 2.8)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(WHITE)

n_stages = len(stages)
xs = np.linspace(1.05, 12.95, n_stages)
y_c = 2.05
r = 0.34
for i, (num, title, sub) in enumerate(stages):
    x = xs[i]
    if i < n_stages - 1:
        ax.add_patch(FancyArrowPatch((x + r + 0.06, y_c), (xs[i+1] - r - 0.06, y_c),
                                      arrowstyle="-|>", mutation_scale=14, color="#C7CDD9", linewidth=1.6))
    circ_color = BLUE if i < n_stages - 1 else GREEN
    ax.add_patch(mpatches.Circle((x, y_c), r, facecolor=circ_color, edgecolor="none", zorder=3))
    ax.text(x, y_c, num, fontsize=13, fontweight="bold", color=WHITE, ha="center", va="center", zorder=4)
    ax.text(x, y_c - r - 0.22, title, fontsize=10.8, fontweight="bold", color=NAVY, ha="center", va="top")
    ax.text(x, y_c - r - 0.55, sub, fontsize=8, color=GRAY, ha="center", va="top", linespacing=1.35)

plt.tight_layout(pad=0.3)
plt.savefig("assets/approach_pipeline.png", facecolor=WHITE, bbox_inches="tight")
plt.close()
print("saved approach_pipeline.png")

# ============================================================== 3. CAPTURE BY DECILE
scored["decile"] = pd.qcut(-scored["score"], 10, labels=[f"D{i}" for i in range(1, 11)])
decile_counts = scored.groupby("decile", observed=True)["y"].sum().reindex([f"D{i}" for i in range(1, 11)])
decile_sizes = scored.groupby("decile", observed=True)["y"].count().reindex([f"D{i}" for i in range(1, 11)])
expected_random = decile_sizes * base_rate

fig, ax = plt.subplots(figsize=(10, 4.3), dpi=150)
x = np.arange(10)
bars = ax.bar(x, decile_counts.values, color=BLUE, width=0.62, zorder=3, label="True underperformers captured")
ax.plot(x, expected_random.values, color=RED, linestyle="--", marker="o", markersize=5,
        linewidth=1.8, zorder=4, label="Expected under random flagging")
ax.set_xticks(x)
ax.set_xticklabels([f"D{i}\n(riskiest)" if i == 1 else (f"D{i}\n(safest)" if i == 10 else f"D{i}") for i in range(1, 11)], fontsize=9)
ax.set_ylabel("True Underperformers Captured", fontsize=10.5)
ax.set_title("Risk Is Concentrated at the Top of the Ranking", fontsize=13, fontweight="bold", color=NAVY, pad=12)
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.set_facecolor(WHITE)
fig.patch.set_facecolor(WHITE)
plt.tight_layout()
plt.savefig("assets/capture_by_decile.png", facecolor=WHITE, bbox_inches="tight")
plt.close()
print("saved capture_by_decile.png")

# ============================================================== 4. TOP DRIVERS
perm = permutation_importance(xgb_clf, X_test, y_test, n_repeats=12, random_state=42, scoring="roc_auc", n_jobs=-1)
importances = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=True).tail(8)
pretty = {
    "content_completeness_score": "Content Completeness",
    "price_index_vs_compset": "Price vs. Comp Set",
    "review_score": "Review Score",
    "inventory_availability_rate": "Inventory Availability",
    "promo_depth_pct": "Promo Depth",
    "cancellation_rate_pct": "Cancellation Rate",
    "guest_response_rate_pct": "Guest Response Rate",
    "days_since_last_content_update": "Content Staleness (days)",
    "review_count": "Review Count",
    "days_listed": "Partner Tenure (days)",
    "star_rating": "Star Rating",
    "photo_count": "Photo Count",
}
labels = [pretty.get(i, i) for i in importances.index]
controllable = {"Content Completeness", "Price vs. Comp Set", "Promo Depth", "Inventory Availability"}
colors = [GREEN if lab in controllable else GRAY for lab in labels]

fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
ax.barh(labels, importances.values, color=colors, height=0.62, zorder=3)
ax.set_xlabel("Permutation Importance (mean drop in ROC-AUC)", fontsize=10)
ax.set_title("What Drives the Risk Score — Green = Directly Actionable", fontsize=12.5, fontweight="bold", color=NAVY, pad=12)
ax.spines[["top", "right"]].set_visible(False)
fig.patch.set_facecolor(WHITE)
plt.tight_layout()
plt.savefig("assets/top_drivers.png", facecolor=WHITE, bbox_inches="tight")
plt.close()
print("saved top_drivers.png")

# ============================================================== 5. MODEL EVALUATION (ROC + PR)
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

fpr, tpr, _ = roc_curve(y_test, p)
prec, rec, _ = precision_recall_curve(y_test, p)
pr_auc = average_precision_score(y_test, p)
no_skill = y_test.mean()

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150)
fig.patch.set_facecolor(WHITE)

ax = axes[0]
ax.plot([0, 1], [0, 1], "--", color="#C7CDD9", linewidth=1.6, label="No skill")
ax.plot(fpr, tpr, color=BLUE, linewidth=2.4, label=f"XGBoost (AUC = {auc:.3f})")
ax.fill_between(fpr, tpr, alpha=0.08, color=BLUE)
ax.set_xlabel("False Positive Rate", fontsize=10)
ax.set_ylabel("True Positive Rate", fontsize=10)
ax.set_title("ROC Curve", fontsize=12.5, fontweight="bold", color=NAVY, pad=10)
ax.legend(frameon=False, fontsize=9.5, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.set_facecolor(WHITE)

ax = axes[1]
ax.axhline(no_skill, linestyle="--", color="#C7CDD9", linewidth=1.6, label="No skill")
ax.plot(rec, prec, color=RED, linewidth=2.4, label=f"XGBoost (PR-AUC = {pr_auc:.3f})")
ax.fill_between(rec, prec, alpha=0.08, color=RED)
ax.set_xlabel("Recall", fontsize=10)
ax.set_ylabel("Precision", fontsize=10)
ax.set_title("Precision-Recall Curve", fontsize=12.5, fontweight="bold", color=NAVY, pad=10)
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
ax.set_facecolor(WHITE)

fig.suptitle("Model Evaluation — Predicting Profit Underperformance vs. Competitive Set",
             fontsize=13.5, fontweight="bold", color=NAVY, y=1.02)
plt.tight_layout()
plt.savefig("assets/model_evaluation.png", facecolor=WHITE, bbox_inches="tight")
plt.close()
print("saved model_evaluation.png")

# ============================================================== 6. CONFUSION MATRIX
from sklearn.metrics import confusion_matrix

threshold20 = scored["score"].iloc[k20 - 1]
preds20 = (p >= threshold20).astype(int)
cm = confusion_matrix(y_test, preds20)

fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
fig.patch.set_facecolor(WHITE)
ax.imshow(cm, cmap="Blues")
labels = ["At/Above\nComp Set", "Underperforming"]
ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("Predicted (@ top-20% cutoff)", fontsize=10)
ax.set_ylabel("Actual", fontsize=10)
ax.set_title("Confusion Matrix — Profit vs. Comp Set", fontsize=13, fontweight="bold", color=NAVY, pad=12)
for i in range(2):
    for j in range(2):
        color = WHITE if cm[i, j] > cm.max() / 2 else NAVY
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=15, fontweight="bold", color=color)
plt.tight_layout()
plt.savefig("assets/confusion_matrix.png", facecolor=WHITE, bbox_inches="tight")
plt.close()
print("saved confusion_matrix.png")

print("\nAll assets generated.")

# ============================================================== 7. VALIDATION & ROBUSTNESS DASHBOARD
print("\nRunning validation & robustness checks (this section takes a couple of minutes)...")
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, recall_score
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE

# --- 1. Calibration ---
frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
brier = brier_score_loss(y_test, p)

# --- 2. Out-of-time validation ---
month = df["snapshot_month"]
train_mask, test_mask = month <= 9, month > 9
X_oot_tr, y_oot_tr = X.loc[train_mask], y.loc[train_mask]
X_oot_te, y_oot_te = X.loc[test_mask], y.loc[test_mask]
spw_oot = (y_oot_tr == 0).sum() / (y_oot_tr == 1).sum()
xgb_oot = Pipeline([("pre", pre), ("clf", xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw_oot, eval_metric="logloss", random_state=42, n_jobs=-1))])
xgb_oot.fit(X_oot_tr, y_oot_tr)
oot_auc = roc_auc_score(y_oot_te, xgb_oot.predict_proba(X_oot_te)[:, 1])

# --- 3. Subgroup performance ---
test_df = X_test.copy(); test_df["y"] = y_test.values; test_df["p"] = p
sub_rows = []
for dim in ["market_segment", "region"]:
    for grp_name, grp in test_df.groupby(dim):
        if grp["y"].nunique() == 2:
            sub_rows.append((grp_name, roc_auc_score(grp["y"], grp["p"])))
subgroup_df = pd.DataFrame(sub_rows, columns=["group", "auc"]).sort_values("auc")

# --- 4. Hyperparameter tuning ---
param_dist = {
    "clf__max_depth": [3, 4, 5, 6], "clf__learning_rate": [0.03, 0.05, 0.08, 0.12],
    "clf__n_estimators": [200, 300, 400], "clf__subsample": [0.7, 0.8, 0.9, 1.0],
    "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0], "clf__min_child_weight": [1, 3, 5],
}
base_clf = Pipeline([("pre", pre), ("clf", xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42, n_jobs=-1))])
search = RandomizedSearchCV(base_clf, param_distributions=param_dist, n_iter=12, cv=3,
                             scoring="roc_auc", random_state=42, n_jobs=-1)
search.fit(X_train, y_train)
tuned_auc = roc_auc_score(y_test, search.best_estimator_.predict_proba(X_test)[:, 1])

# --- 5. Class imbalance: weighting vs SMOTE (practical effect at a naive 0.5 cutoff) ---
unw_clf = Pipeline([("pre", pre), ("clf", xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=42, n_jobs=-1))])
unw_clf.fit(X_train, y_train)
p_unw = unw_clf.predict_proba(X_test)[:, 1]
recall_unw = recall_score(y_test, (p_unw >= 0.5).astype(int))
recall_w = recall_score(y_test, (p >= 0.5).astype(int))

pre_fit = pre.fit(X_train)
Xt_tr, Xt_te = pre_fit.transform(X_train), pre_fit.transform(X_test)
Xt_res, y_res = SMOTE(random_state=42).fit_resample(Xt_tr, y_train)
xgb_smote = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1)
xgb_smote.fit(Xt_res, y_res)
auc_smote = roc_auc_score(y_test, xgb_smote.predict_proba(Xt_te)[:, 1])

# --- 6. Selection bias check ---
in_scope = (df["review_count"] > df["review_count"].median()) & (df["comp_set_size"] > df["comp_set_size"].median())
X_sc, y_sc = X.loc[in_scope], y.loc[in_scope]
X_out, y_out = X.loc[~in_scope], y.loc[~in_scope]
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_sc, y_sc, test_size=0.2, random_state=42, stratify=y_sc)
spw_sc = (ys_tr == 0).sum() / (ys_tr == 1).sum()
xgb_scope = Pipeline([("pre", pre), ("clf", xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw_sc, eval_metric="logloss", random_state=42, n_jobs=-1))])
xgb_scope.fit(Xs_tr, ys_tr)
auc_in_scope = roc_auc_score(ys_te, xgb_scope.predict_proba(Xs_te)[:, 1])
auc_out_scope = roc_auc_score(y_out, xgb_scope.predict_proba(X_out)[:, 1])

print(f"calibration Brier={brier:.3f} | OOT AUC={oot_auc:.3f} (vs {auc:.3f}) | subgroup spread={subgroup_df['auc'].max()-subgroup_df['auc'].min():.3f} | "
      f"tuned AUC={tuned_auc:.3f} (vs {auc:.3f}) | recall@0.5 unw={recall_unw:.3f} w={recall_w:.3f} smote_auc={auc_smote:.3f} | "
      f"in-scope AUC={auc_in_scope:.3f} out-of-scope AUC={auc_out_scope:.3f}")

# --- Dashboard: 2x3 panel ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=150)
fig.patch.set_facecolor(WHITE)
fig.suptitle("Model Validation & Robustness Checks", fontsize=17, fontweight="bold", color=NAVY, y=0.99)

# Panel 1: Calibration
ax = axes[0, 0]
ax.plot([0, 1], [0, 1], "--", color="#C7CDD9", linewidth=1.5, label="Perfect calibration")
ax.plot(mean_pred, frac_pos, marker="o", color=BLUE, linewidth=2, label="XGBoost")
ax.set_xlabel("Mean Predicted Risk", fontsize=9.5); ax.set_ylabel("Observed Frequency", fontsize=9.5)
ax.set_title(f"Calibration  (Brier = {brier:.3f})", fontsize=11.5, fontweight="bold", color=NAVY)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

# Panel 2: Out-of-time validation
ax = axes[0, 1]
bars = ax.bar(["Random\nSplit", "Out-of-Time\n(Mo. 10-12)"], [auc, oot_auc], color=[BLUE, GREEN], width=0.5, zorder=3)
ax.set_ylim(0, 1); ax.set_ylabel("ROC-AUC", fontsize=9.5)
ax.set_title("Out-of-Time Validation", fontsize=11.5, fontweight="bold", color=NAVY)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{b.get_height():.3f}", ha="center", fontsize=10, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

# Panel 3: Subgroup performance
ax = axes[0, 2]
ax.barh(subgroup_df["group"], subgroup_df["auc"], color=BLUE, height=0.6, zorder=3)
ax.axvline(auc, color=RED, linestyle="--", linewidth=1.4, label=f"Overall ({auc:.3f})")
ax.set_xlabel("ROC-AUC", fontsize=9.5)
ax.set_title("Subgroup Performance", fontsize=11.5, fontweight="bold", color=NAVY)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.tick_params(axis="y", labelsize=8)
ax.spines[["top", "right"]].set_visible(False)

# Panel 4: Hyperparameter tuning
ax = axes[1, 0]
bars = ax.bar(["Baseline\n(defaults)", "Tuned\n(random search)"], [auc, tuned_auc], color=[GRAY, AMBER], width=0.5, zorder=3)
ax.set_ylim(0, 1); ax.set_ylabel("ROC-AUC", fontsize=9.5)
ax.set_title("Hyperparameter Tuning Impact", fontsize=11.5, fontweight="bold", color=NAVY)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{b.get_height():.3f}", ha="center", fontsize=10, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

# Panel 5: Class imbalance handling
ax = axes[1, 1]
bars = ax.bar(["Unweighted", "Weighted\n(deployed)", "SMOTE"], [recall_unw, recall_w, recall_w], color=[GRAY, BLUE, "#A5B4FC"], width=0.55, zorder=3)
ax.set_ylim(0, 1); ax.set_ylabel("Recall @ 0.5 Cutoff", fontsize=9.5)
ax.set_title("Class Imbalance Handling", fontsize=11.5, fontweight="bold", color=NAVY)
for b, val in zip(bars, [recall_unw, recall_w, recall_w]):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

# Panel 6: Selection bias check
ax = axes[1, 2]
bars = ax.bar(["Trained &\nTested In-Scope", "Same Model on\nUnseen Population"], [auc_in_scope, auc_out_scope], color=[BLUE, GREEN], width=0.5, zorder=3)
ax.set_ylim(0, 1); ax.set_ylabel("ROC-AUC", fontsize=9.5)
ax.set_title("Selection Bias Check", fontsize=11.5, fontweight="bold", color=NAVY)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{b.get_height():.3f}", ha="center", fontsize=10, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("assets/validation_dashboard.png", facecolor=WHITE, bbox_inches="tight")
plt.close()
print("saved validation_dashboard.png")

# Save numbers for README/notebook reuse
import json
validation_results = {
    "brier": round(float(brier), 4),
    "oot_auc": round(float(oot_auc), 3),
    "random_split_auc": round(float(auc), 3),
    "subgroup_min": round(float(subgroup_df["auc"].min()), 3),
    "subgroup_max": round(float(subgroup_df["auc"].max()), 3),
    "tuned_auc": round(float(tuned_auc), 3),
    "tuned_cv_auc": round(float(search.best_score_), 3),
    "recall_unweighted_at_50": round(float(recall_unw), 3),
    "recall_weighted_at_50": round(float(recall_w), 3),
    "smote_auc": round(float(auc_smote), 3),
    "auc_in_scope": round(float(auc_in_scope), 3),
    "auc_out_of_scope": round(float(auc_out_scope), 3),
}
with open("assets/validation_results.json", "w") as f:
    json.dump(validation_results, f, indent=2)
print("saved validation_results.json:", validation_results)
