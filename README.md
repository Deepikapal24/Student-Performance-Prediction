# Student Performance Prediction (Linear Regression)

A beginner-friendly mini ML project based on the **Students Performance in Exams** dataset. It predicts math score from demographic/background and preparation variables, performs exploratory data analysis, evaluates a Linear Regression model, and ranks model coefficients.

## Dataset

- Original dataset: [Kaggle — Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- A public CSV mirror is downloaded automatically by `download_data.py` / `train_model.py` if the local file is absent. Check the dataset's current terms before redistributing it.
- 1,000 records, with gender, race/ethnicity group, parental education, lunch type, test-preparation status, and math/reading/writing scores.

## Model results (executed notebook)\n\nOn the provided dataset with an 80/20 split (`random_state=42`), the executed notebook reports:\n\n- **R²:** 0.176\n- **MAE:** 11.27 points\n- **RMSE:** 14.16 points\n\nThese results are a baseline; the model uses demographic and preparation variables only and should not be treated as a definitive assessment of any student.\n\n## Features

- Data quality checks, cleaning, and engineered average score / test-prep flag
- Six EDA figures and an actual-vs-predicted plot
- One-hot encoding of categorical variables
- 80/20 train-test split (`random_state=42`)
- Linear Regression evaluated using R², MAE, and RMSE
- Coefficient ranking and saved model
- Streamlit interface for interactive predictions

## Run on Windows

1. Install Python 3.10+ from [python.org](https://www.python.org/downloads/). During installation, enable **Add Python to PATH**.
2. Extract this ZIP and open the folder in VS Code.
3. Open Terminal in VS Code and run:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r requirements.txt
   python train_model.py
   ```

4. Start the interactive app:

   ```bash
   streamlit run app.py
   ```

5. To open the notebook, run `jupyter notebook` and open `notebooks/student_performance_prediction.ipynb`.

## Outputs

Generated under `outputs/`: cleaned CSV, coefficient table, trained model, and charts under `outputs/figures/`.

## Important interpretation note

This dataset is observational. Coefficients describe associations relative to one-hot-encoding reference categories; they do not prove causation. The model uses background/preparation features only and excludes reading/writing scores to avoid using other exam outcomes as predictors. Treat predictions as educational demonstrations, not high-stakes decisions.

## GitHub upload

Create a new empty repository on GitHub, then run from this project folder:

```bash
git init
git add .
git commit -m "Add student performance prediction project"
git branch -M main
git remote add origin https://github.com/Deepikapal24/Student-Performance-Prediction.git
git push -u origin main
```

The GitHub remote URL above is configured for `Deepikapal24`. If you choose a different repository name, update the URL accordingly.


### Additional handbook analyses

The training script and notebook print `value_counts()` for every categorical column and rank the top five one-hot encoded category indicators by absolute correlation with math score. The correlation results are saved to `outputs/top_factor_correlations.csv`, with a visualization in `outputs/figures/08_top_factor_correlations.png`. Correlation indicates association, not causation.


### Visualization images

Standalone PNG charts are available in `images/visualizations/`. The executed notebook also includes the corresponding chart outputs.
