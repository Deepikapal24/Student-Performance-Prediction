from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "StudentsPerformance.csv"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Fetch data automatically if it is not present locally.
if not DATA_PATH.exists():
    import urllib.request
    URL = "https://raw.githubusercontent.com/NadavKiani/Students-Performance-in-Exams/master/StudentsPerformance.csv"
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Dataset not found. Downloading from the public source...")
    urllib.request.urlretrieve(URL, DATA_PATH)

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower().replace("/", "_").replace(" ", "_") for c in df.columns]
score_cols = ["math_score", "reading_score", "writing_score"]
cat_cols = [c for c in df.columns if c not in score_cols]

# Basic cleaning
df = df.drop_duplicates().copy()
for c in cat_cols:
    df[c] = df[c].astype("string").str.strip().str.lower()
for c in score_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=score_cols + cat_cols)
for c in score_cols:
    df = df[df[c].between(0, 100)]
df["average_score"] = df[score_cols].mean(axis=1)
df["test_prep_completed"] = (df["test_preparation_course"] == "completed").astype(int)
df.to_csv(OUTPUT_DIR / "cleaned_student_performance.csv", index=False)

print("Dataset shape:", df.shape)
print("Missing values:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nSummary statistics:\n", df[score_cols].describe().round(2))

print("\nCategorical value counts:")
for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

# Visualizations (save to outputs/figures)
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8,5))
df[score_cols].melt(var_name="subject", value_name="score").pipe(
    lambda d: sns.histplot(data=d, x="score", hue="subject", element="step", stat="density", common_norm=False))
plt.title("Score distributions by subject"); plt.tight_layout(); plt.savefig(FIG_DIR/"01_score_distributions.png", dpi=160); plt.close()

plt.figure(figsize=(8,5))
sns.barplot(data=df, x="gender", y="math_score", errorbar=None)
plt.title("Average math score by gender"); plt.tight_layout(); plt.savefig(FIG_DIR/"02_math_by_gender.png", dpi=160); plt.close()

plt.figure(figsize=(8,5))
sns.barplot(data=df, x="test_preparation_course", y="math_score", errorbar=None)
plt.title("Average math score by test preparation"); plt.tight_layout(); plt.savefig(FIG_DIR/"03_math_by_test_prep.png", dpi=160); plt.close()

plt.figure(figsize=(10,5))
sns.barplot(data=df, x="parental_level_of_education", y="math_score", errorbar=None)
plt.xticks(rotation=25, ha="right"); plt.title("Average math score by parental education"); plt.tight_layout(); plt.savefig(FIG_DIR/"04_math_by_parent_education.png", dpi=160); plt.close()

plt.figure(figsize=(7,5))
sns.heatmap(df[score_cols].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
plt.title("Correlation between subject scores"); plt.tight_layout(); plt.savefig(FIG_DIR/"05_score_correlation.png", dpi=160); plt.close()

plt.figure(figsize=(7,5))
sns.boxplot(data=df, x="lunch", y="math_score")
plt.title("Math score by lunch type"); plt.tight_layout(); plt.savefig(FIG_DIR/"06_math_by_lunch.png", dpi=160); plt.close()

# Model: use background/preparation features only; reading/writing scores are excluded
# to avoid using other exam scores as predictors.
X = df[["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]]
y = df["math_score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
preprocessor = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore", drop="first"), X.columns.tolist())])
model = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
model.fit(X_train, y_train)
pred = model.predict(X_test)
metrics = {"R2": r2_score(y_test, pred), "MAE": mean_absolute_error(y_test, pred), "RMSE": np.sqrt(mean_squared_error(y_test, pred))}
print("\nEvaluation metrics:")
for k,v in metrics.items(): print(f"{k}: {v:.3f}")

plt.figure(figsize=(7,5))
sns.scatterplot(x=y_test, y=pred, alpha=.7)
plt.plot([0,100],[0,100], linestyle="--", color="red")
plt.xlim(0,100); plt.ylim(0,100); plt.xlabel("Actual math score"); plt.ylabel("Predicted math score")
plt.title("Actual vs predicted math scores"); plt.tight_layout(); plt.savefig(FIG_DIR/"07_actual_vs_predicted.png", dpi=160); plt.close()

feature_names = model.named_steps["preprocessor"].get_feature_names_out()
coefficients = model.named_steps["regressor"].coef_
coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
coef_df.to_csv(OUTPUT_DIR/"model_coefficients.csv", index=False)
print("\nLargest absolute coefficients (relative to reference categories):\n", coef_df.head(5)[["feature","coefficient"]].to_string(index=False))

# Correlation analysis: one-hot encode categorical predictors and correlate each
# category indicator with math_score. These are associations, not causal effects.
correlation_features = pd.get_dummies(X, drop_first=False, dtype=int)
correlation_df = correlation_features.copy()
correlation_df["math_score"] = y.to_numpy()
math_correlations = correlation_df.corr(numeric_only=True)["math_score"].drop("math_score")
math_correlations = math_correlations.reindex(math_correlations.abs().sort_values(ascending=False).index)
correlation_results = math_correlations.head(5).rename("correlation_with_math_score").reset_index()
correlation_results = correlation_results.rename(columns={"index": "factor"})
correlation_results.to_csv(OUTPUT_DIR / "top_factor_correlations.csv", index=False)
print("\nTop 5 factors by absolute correlation with math score:")
print(correlation_results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

plt.figure(figsize=(9,5))
plot_corr = math_correlations.head(5).sort_values()
sns.barplot(x=plot_corr.values, y=plot_corr.index)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Correlation with math score")
plt.ylabel("Encoded factor")
plt.title("Top 5 categorical factors associated with math score")
plt.tight_layout()
plt.savefig(FIG_DIR / "08_top_factor_correlations.png", dpi=160)
plt.close()

joblib.dump(model, OUTPUT_DIR/"linear_regression_model.joblib")
print(f"\nSaved model and outputs in: {OUTPUT_DIR}")
