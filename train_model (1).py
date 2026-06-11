"""
Train ML model for travel recommendation system.
Uses RandomForest + exports to JSON for use in the web app.
"""
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv("/home/claude/travel_dataset.csv")

# ─── Feature engineering ───────────────────────────────────────────────
le_type     = LabelEncoder()
le_season   = LabelEncoder()
le_budget   = LabelEncoder()
le_activity = LabelEncoder()

df["dest_type_enc"]   = le_type.fit_transform(df["destination_type"])
df["season_enc"]      = le_season.fit_transform(df["best_season"])
df["budget_enc"]      = le_budget.fit_transform(df["budget_tier"])
df["activity_enc"]    = le_activity.fit_transform(df["activity"])

# Cost ratio: can the student afford it?
df["cost_ratio"]      = df["avg_cost_per_day"] / df["daily_budget"]
df["total_activity_cost_ratio"] = df["activity_cost"] / df["daily_budget"]

features = [
    "dest_type_enc", "avg_cost_per_day", "destination_rating",
    "student_friendly", "season_enc", "budget_enc", "activity_enc",
    "activity_duration_hrs", "activity_cost", "daily_budget",
    "cost_ratio", "total_activity_cost_ratio"
]
X = df[features]
y = df["recommended"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ─── Train RandomForest ─────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cv  = cross_val_score(rf, X, y, cv=5).mean()

print(f"Test Accuracy:  {acc:.4f}")
print(f"Cross-Val Mean: {cv:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Recommended","Recommended"]))

feat_imp = dict(zip(features, rf.feature_importances_.tolist()))

# ─── Build lookup tables for JS ─────────────────────────────────────────
# For each destination × budget tier, compute ML score
results = []
for dest in df["destination"].unique():
    d = df[df["destination"]==dest].iloc[0]
    for budget_tier in ["low","medium","high"]:
        budget_num = {"low":500,"medium":1000,"high":2000}[budget_tier]
        row = {
            "dest_type_enc": int(le_type.transform([d["destination_type"]])[0]),
            "avg_cost_per_day": d["avg_cost_per_day"],
            "destination_rating": d["destination_rating"],
            "student_friendly": d["student_friendly"],
            "season_enc": int(le_season.transform([d["best_season"]])[0]),
            "budget_enc": int(le_budget.transform([budget_tier])[0]),
            "activity_enc": 0,
            "activity_duration_hrs": 3,
            "activity_cost": 200,
            "daily_budget": budget_num,
            "cost_ratio": d["avg_cost_per_day"] / budget_num,
            "total_activity_cost_ratio": 200 / budget_num,
        }
        prob = rf.predict_proba(pd.DataFrame([row]))[0][1]
        results.append({
            "destination": dest,
            "state": d["state"],
            "destination_type": d["destination_type"],
            "avg_cost_per_day": int(d["avg_cost_per_day"]),
            "rating": float(d["destination_rating"]),
            "student_friendly": int(d["student_friendly"]),
            "best_season": d["best_season"],
            "latitude": float(d["latitude"]),
            "longitude": float(d["longitude"]),
            "budget_tier": budget_tier,
            "ml_score": round(float(prob), 3),
            "recommended": bool(prob >= 0.5),
        })

# Activities per destination
activities = {}
for _, row in df.drop_duplicates(["destination","activity"]).iterrows():
    dest = row["destination"]
    if dest not in activities:
        activities[dest] = []
    activities[dest].append({
        "name": row["activity"],
        "duration_hrs": int(row["activity_duration_hrs"]),
        "cost": int(row["activity_cost"]),
    })

model_data = {
    "model_metadata": {
        "algorithm": "RandomForestClassifier",
        "n_estimators": 100,
        "test_accuracy": round(acc, 4),
        "cross_val_accuracy": round(cv, 4),
        "features": features,
        "feature_importances": {k: round(v,4) for k,v in sorted(feat_imp.items(), key=lambda x:-x[1])},
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    },
    "recommendations": results,
    "activities": activities,
    "label_encoders": {
        "destination_type": dict(zip(le_type.classes_.tolist(), le_type.transform(le_type.classes_).tolist())),
        "best_season": dict(zip(le_season.classes_.tolist(), le_season.transform(le_season.classes_).tolist())),
        "budget_tier": dict(zip(le_budget.classes_.tolist(), le_budget.transform(le_budget.classes_).tolist())),
    }
}

with open("/home/claude/model_data.json","w") as f:
    json.dump(model_data, f, indent=2)

joblib.dump(rf, "/home/claude/travel_model.pkl")
print(f"\nModel saved. JSON lookup: {len(results)} entries for {len(df['destination'].unique())} destinations.")
