# Learning Notes (Colab)

## Thoughts
- Hyperparameter tuning adds clarity on which model and params work best for Iris.
- A clear tuned-results table makes comparisons quick.
- A model-selection table makes strategy, hyperparams, and rationale explicit.
- Train/test gap highlights potential overfitting early.

## Topics Covered
- GridSearchCV with StratifiedKFold
- Model pipelines with scaling
- Model evaluation metrics
- Model selection summary (strategy + rationale)
- Baseline train/test gap analysis
- Tuning impact comparison (baseline vs tuned CV)
- Feature importance (Random Forest)
- Baseline error analysis (confusion matrix + report)

## What I Learned
- What I learnt by doing this task ML Team
- Small, focused parameter grids keep tuning fast in Colab.
- Comparing CV vs test gives a clearer signal about generalization.
- A simple model outperforms a complex one if the data is clean and separable.
- Small datasets show higher variance. Stratified CV helps stabilize results.

## Issues / Open Questions
- Small dataset size (n=150) can cause higher variance across splits.
