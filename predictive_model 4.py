"""Week 4: Predicting delivery time in a logistics network and using the model for optimisation.

Problem: predict total delivery time (hours) for each shipment from order, route and operating conditions.
Models: Linear Regression (baseline), Decision Tree, Random Forest, Gradient Boosting.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

for folder in ["data", "charts", "results"]:
    os.makedirs(folder, exist_ok=True)
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

# ================= 1. DATA SIMULATION =================
rng = np.random.default_rng(RANDOM_STATE)
n = 5000
carriers = ["CarrierA", "CarrierB", "CarrierC"]
modes = ["Standard", "Express", "Same Day"]
regions = ["North", "South", "East", "West", "Central"]

df = pd.DataFrame({
    "distance_km": rng.uniform(10, 1200, n).round(1),
    "weight_kg": rng.lognormal(2.2, 0.6, n).round(1),
    "num_stops": rng.integers(1, 6, n),
    "carrier": rng.choice(carriers, n, p=[0.5, 0.3, 0.2]),
    "shipping_mode": rng.choice(modes, n, p=[0.6, 0.3, 0.1]),
    "origin_region": rng.choice(regions, n),
    "month": rng.integers(1, 13, n),
    "warehouse_load_pct": rng.uniform(40, 100, n).round(1),
    "weather_severity": rng.integers(0, 4, n),   # 0 = clear ... 3 = severe
})
df["peak_season"] = df["month"].isin([11, 12]).astype(int)

# Hidden process used to generate the target. The model never sees this formula.
mode_base = {"Standard": 36, "Express": 18, "Same Day": 6}
carrier_eff = {"CarrierA": 0, "CarrierB": 3, "CarrierC": 7}      # hours of extra delay
noise = rng.normal(0, 4, n)
congestion = np.where(df["warehouse_load_pct"] > 85, (df["warehouse_load_pct"] - 85) * 0.9, 0)
df["delivery_hours"] = (
    df["shipping_mode"].map(mode_base)
    + df["distance_km"] * 0.035
    + df["num_stops"] * 2.5
    + df["weight_kg"] * 0.15
    + df["carrier"].map(carrier_eff)
    + congestion
    + df["weather_severity"] * 3.0
    + df["peak_season"] * 8.0
    + noise
).clip(lower=1).round(2)
df.to_csv("data/logistics_shipments.csv", index=False)
print("Dataset:", df.shape)

# ================= 2. DATA PREPARATION =================
target = "delivery_hours"
X = df.drop(columns=[target])
y = df[target]
cat_cols = ["carrier", "shipping_mode", "origin_region"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print("Train:", X_train.shape, "Test:", X_test.shape)

# ================= 3. MODEL COMPARISON (5-fold cross-validation on training data) =================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
}
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}
cv_rows = []
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
    cv_rows.append({
        "Model": name,
        "CV RMSE (h)": round(-scores["test_rmse"].mean(), 3),
        "CV RMSE std": round(scores["test_rmse"].std(), 3),
        "CV MAE (h)": round(-scores["test_mae"].mean(), 3),
        "CV R2": round(scores["test_r2"].mean(), 4),
    })
cv_table = pd.DataFrame(cv_rows).sort_values("CV RMSE (h)")
cv_table.to_csv("results/model_cv_comparison.csv", index=False)
print("\nCross-validation results:\n", cv_table.to_string(index=False))

# ================= 4. HYPERPARAMETER TUNING (Random Forest) =================
rf_pipe = Pipeline([("prep", preprocess),
                    ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1))])
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [8, 12, None],
    "model__min_samples_leaf": [1, 5],
}
grid = GridSearchCV(rf_pipe, param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
grid.fit(X_train, y_train)
print("\nBest RF params:", grid.best_params_, "| CV RMSE:", round(-grid.best_score_, 3))
pd.DataFrame(grid.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]] \
    .assign(mean_rmse=lambda d: -d["mean_test_score"]).sort_values("rank_test_score") \
    .to_csv("results/rf_grid_search.csv", index=False)

# Tune Gradient Boosting (the strongest model in cross-validation)
gb_pipe = Pipeline([("prep", preprocess),
                    ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))])
gb_grid = GridSearchCV(gb_pipe, {
    "model__n_estimators": [200, 300],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 4],
}, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
gb_grid.fit(X_train, y_train)
print("\nBest GB params:", gb_grid.best_params_, "| CV RMSE:", round(-gb_grid.best_score_, 3))
pd.DataFrame(gb_grid.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]] \
    .assign(mean_rmse=lambda d: -d["mean_test_score"]).sort_values("rank_test_score") \
    .to_csv("results/gb_grid_search.csv", index=False)

# Final candidates: default models plus the two tuned models, all scored by CV on training data
cv_scores = dict(zip(cv_table["Model"], cv_table["CV RMSE (h)"]))
cv_scores["Random Forest (tuned)"] = round(-grid.best_score_, 3)
cv_scores["Gradient Boosting (tuned)"] = round(-gb_grid.best_score_, 3)
cv_table = pd.concat([cv_table, pd.DataFrame([
    {"Model": "Random Forest (tuned)", "CV RMSE (h)": cv_scores["Random Forest (tuned)"]},
    {"Model": "Gradient Boosting (tuned)", "CV RMSE (h)": cv_scores["Gradient Boosting (tuned)"]},
])], ignore_index=True).sort_values("CV RMSE (h)")
cv_table.to_csv("results/model_cv_comparison_all.csv", index=False)
print("\nAll candidates by CV RMSE:\n", cv_table[["Model", "CV RMSE (h)"]].to_string(index=False))

# ================= 5. FINAL EVALUATION ON HELD-OUT TEST SET =================
final_models = {name: Pipeline([("prep", preprocess), ("model", m)]) for name, m in models.items()}
final_models["Random Forest (tuned)"] = grid.best_estimator_
final_models["Gradient Boosting (tuned)"] = gb_grid.best_estimator_

test_rows, preds = [], {}
for name, pipe in final_models.items():
    pipe.fit(X_train, y_train)
    p = pipe.predict(X_test)
    preds[name] = p
    test_rows.append({
        "Model": name,
        "Test RMSE (h)": round(np.sqrt(mean_squared_error(y_test, p)), 3),
        "Test MAE (h)": round(mean_absolute_error(y_test, p), 3),
        "Test R2": round(r2_score(y_test, p), 4),
    })
test_table = pd.DataFrame(test_rows).sort_values("Test RMSE (h)")
test_table.to_csv("results/model_test_results.csv", index=False)
print("\nTest-set results:\n", test_table.to_string(index=False))

# Model is chosen by cross-validation on the training data, NOT by the test set
best_name = cv_table.iloc[0]["Model"]  # lowest CV RMSE, incl. tuned models
print("\nSelected model (lowest CV RMSE):", best_name)
best_model = final_models[best_name]
best_pred = preds[best_name]

# Baseline: predicting the mean delivery time for every shipment
baseline_rmse = np.sqrt(mean_squared_error(y_test, np.full(len(y_test), y_train.mean())))
print(f"\nBaseline (predict mean) RMSE: {baseline_rmse:.3f} h")

# SLA check: share of shipments predicted to exceed 48 hours
sla = 72
sla_actual = (y_test > sla).mean() * 100
sla_pred = (best_pred > sla).mean() * 100
sla_acc = ((best_pred > sla) == (y_test > sla)).mean() * 100
print(f"SLA (>{sla} h) actual: {sla_actual:.1f}% | predicted: {sla_pred:.1f}% | classification accuracy: {sla_acc:.1f}%")

# ================= 6. CHARTS =================
fig, ax = plt.subplots(figsize=(8, 4.5))
melt = test_table.melt(id_vars="Model", value_vars=["Test RMSE (h)", "Test MAE (h)"], var_name="Metric", value_name="Hours")
sns.barplot(data=melt, x="Model", y="Hours", hue="Metric", ax=ax, palette="Blues")
ax.set_title("Test-set error by model (lower is better)"); ax.tick_params(axis="x", rotation=15)
plt.tight_layout(); plt.savefig("charts/01_model_comparison.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, best_pred, alpha=0.35, s=12, color="#2E75B6")
lims = [0, max(y_test.max(), best_pred.max()) + 5]
ax.plot(lims, lims, "r--", label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual delivery time (h)"); ax.set_ylabel("Predicted delivery time (h)")
ax.set_title(f"Predicted vs actual: {best_name}"); ax.legend()
plt.tight_layout(); plt.savefig("charts/02_predicted_vs_actual.png", dpi=150); plt.close()

residuals = y_test - best_pred
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(residuals, bins=40, kde=True, ax=ax, color="#1F4E79")
ax.axvline(0, color="red", linestyle="--")
ax.set_title("Residual distribution (actual minus predicted)"); ax.set_xlabel("Residual (h)")
plt.tight_layout(); plt.savefig("charts/03_residuals.png", dpi=150); plt.close()

# Feature importance from the selected model
chosen = best_model.named_steps["model"]
feature_names = (num_cols + list(best_model.named_steps["prep"]
                 .named_transformers_["cat"].get_feature_names_out(cat_cols)))
imp = pd.Series(chosen.feature_importances_, index=feature_names).sort_values(ascending=True)
imp.to_csv("results/feature_importance.csv", header=["importance"])
fig, ax = plt.subplots(figsize=(7, 5))
imp.tail(12).plot(kind="barh", ax=ax, color="#2E75B6")
ax.set_title(f"Top 12 features ({best_name} importance)"); ax.set_xlabel("Importance")
plt.tight_layout(); plt.savefig("charts/04_feature_importance.png", dpi=150); plt.close()

# ================= 7. OPTIMISATION USING THE MODEL (what-if scenarios) =================
# Each scenario changes operating conditions for the TEST shipments and re-predicts delivery time.
def scenario(label, changes, subset=None):
    Xs = X_test.copy()
    if subset is not None:
        Xs = Xs[subset(Xs)]
    base = best_model.predict(Xs)
    for col, fn in changes.items():
        Xs[col] = fn(Xs[col])
    new = best_model.predict(Xs)
    return {"Scenario": label, "Shipments": len(Xs),
            "Mean predicted time (h)": round(new.mean(), 2),
            "Change vs current (h)": round(new.mean() - base.mean(), 2),
            "Predicted SLA breaches (%)": round((new > sla).mean() * 100, 1)}

rows = [
    scenario("Current operations (no change)", {}),
    scenario("Cap warehouse load at 85% for all shipments",
             {"warehouse_load_pct": lambda s: s.clip(upper=85)}),
    scenario("Cap warehouse load at 85% in peak season only",
             {"warehouse_load_pct": lambda s: s.where(s <= 85, 85)},
             subset=lambda d: d["peak_season"] == 1),
    scenario("Move CarrierC volume to CarrierA",
             {"carrier": lambda s: s.replace("CarrierC", "CarrierA")}),
    scenario("Move CarrierB volume to CarrierA",
             {"carrier": lambda s: s.replace("CarrierB", "CarrierA")}),
    scenario("Reduce stops per route by 1 (min 1)",
             {"num_stops": lambda s: (s - 1).clip(lower=1)}),
    scenario("Reduce weather severity by 1 (min 0)",
             {"weather_severity": lambda s: (s - 1).clip(lower=0)}),
]
scen = pd.DataFrame(rows)
scen.to_csv("results/optimisation_scenarios.csv", index=False)
print("\nOptimisation scenarios:\n", scen.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5))
plot_df = scen.iloc[1:].sort_values("Change vs current (h)")
colors = ["#2E75B6" if v < 0 else "#C55A11" for v in plot_df["Change vs current (h)"]]
ax.barh(plot_df["Scenario"], plot_df["Change vs current (h)"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Change in mean predicted delivery time (hours)")
ax.set_title("What-if scenarios: effect on delivery time (negative is better)")
plt.tight_layout(); plt.savefig("charts/05_optimisation_scenarios.png", dpi=150); plt.close()

# Save summary numbers for the report
summary = {
    "n_total": int(n), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
    "target_mean": round(float(y.mean()), 2), "target_std": round(float(y.std()), 2),
    "best_model": best_name, "best_params": (gb_grid.best_params_ if "Gradient" in best_name else grid.best_params_),
    "baseline_rmse": round(float(baseline_rmse), 3),
    "sla_actual_pct": round(float(sla_actual), 1), "sla_pred_pct": round(float(sla_pred), 1),
    "sla_accuracy_pct": round(float(sla_acc), 1),
}
summary["sla_hours"] = sla
json.dump(summary, open("results/summary.json", "w"), indent=2, default=str)
print("\nSummary:", json.dumps(summary, indent=1, default=str))
print("\nDone.")
